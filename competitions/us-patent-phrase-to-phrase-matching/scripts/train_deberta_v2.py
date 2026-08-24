"""DeBERTa-v3-base fine-tune for US patent phrase-to-phrase matching (Pearson r).

Fixes over v1 (train_deberta.py):
  * robust Pearson (guards zero-variance -> returns 0.0 instead of nan)
  * never crashes when a fold has no "best" epoch (falls back to last epoch)
  * per-epoch diagnostics: train loss, grad norm, prediction std/mean
  * configurable AMP / optimiser implementation so the numerically stable
    combination found by scripts/diag_nan.py can be selected explicitly
  * mean-pooling head + BCE-with-logits loss (targets already in [0, 1])
  * GroupKFold by anchor (leak-free: an anchor never spans folds)
  * CPC section letter expanded to its (general-knowledge) title text
  * writes submission.csv + cv_score.json itself, so a killed parent process
    cannot lose the result

Offline only: uses the locally cached microsoft/deberta-v3-base snapshot.
"""
import argparse
import importlib.util
import json
import math
import os
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          DataCollatorWithPadding, get_cosine_schedule_with_warmup)

torch.set_num_threads(8)

SEED = 42
COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
DATA = f"{COMP}/data"

# CPC section titles - the 9 top-level section letters. General knowledge,
# not an external data file.
SECTION = {
    "A": "human necessities",
    "B": "performing operations transporting",
    "C": "chemistry metallurgy",
    "D": "textiles paper",
    "E": "fixed constructions",
    "F": "mechanical engineering lighting heating weapons blasting",
    "G": "physics",
    "H": "electricity",
    "Y": "general tagging of new technological developments",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_pearson(y, p):
    """Pearson r that returns 0.0 instead of nan for degenerate inputs."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if not np.isfinite(p).all():
        return float("nan")  # genuinely broken predictions -> surface it
    if y.std() < 1e-12 or p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def build_text(df):
    sec = df.context.str[0].map(SECTION).fillna("")
    return (df.anchor + " [SEP] " + df.target + " [SEP] "
            + sec + " " + df.context)


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, with_labels=True):
        self.enc = tokenizer(build_text(df).tolist(), truncation=True,
                             max_length=max_len)
        self.labels = df.score.values.astype("float32") if with_labels else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class MeanPoolRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.hidden_dropout_prob = 0.0
        cfg.attention_probs_dropout_prob = 0.0
        # ROOT CAUSE OF THE v1 NaN: the cached microsoft/deberta-v3-base
        # checkpoint is stored in float16, and transformers v5 loads a model in
        # the checkpoint's dtype by default. Training then ran with fp16 MASTER
        # weights: AdamW keeps exp_avg_sq (grad^2 ~ 1e-8) in the parameter
        # dtype, which underflows fp16 (min subnormal 6e-8) to exactly 0, so
        # the update denominator sqrt(v)+eps collapses and every weight becomes
        # inf/NaN. Forcing fp32 master weights is the fix; speed comes back via
        # bf16 autocast, which keeps the master copy in fp32.
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg,
                                                  dtype=torch.float32)
        assert next(self.backbone.parameters()).dtype == torch.float32, \
            "backbone must hold fp32 master weights"
        self.head = nn.Linear(cfg.hidden_size, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask, **kw):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled).squeeze(-1)


def make_optimizer(model, lr, head_lr, wd, impl):
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.backbone.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "lr": lr, "weight_decay": wd},
        {"params": [p for n, p in model.backbone.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "lr": lr, "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": head_lr,
         "weight_decay": 0.0},
    ]
    kw = {}
    if impl == "fused":
        kw["fused"] = True
    elif impl == "foreach":
        kw["foreach"] = True
    elif impl == "single":
        kw["foreach"] = False
        kw["fused"] = False
    return torch.optim.AdamW(groups, lr=lr, eps=1e-6, betas=(0.9, 0.999), **kw)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        lb = batch.pop("labels", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if amp is None:
            logits = model(**batch)
        else:
            with torch.autocast("cuda", dtype=amp):
                logits = model(**batch)
        preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        if lb is not None:
            labels.append(lb.numpy())
    return (np.concatenate(preds),
            np.concatenate(labels) if labels else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--n-splits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    ap.add_argument("--opt-impl", choices=["fused", "foreach", "single"],
                    default="fused")
    ap.add_argument("--subset", type=int, default=0, help="smoke: rows to keep")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: cap steps")
    ap.add_argument("--tag", default="deberta_v3_base")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    amp = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    smoke = args.subset > 0 or args.max_steps > 0

    seed = args.seed
    set_seed(seed)
    device = "cuda"
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    if args.subset:
        train = train.sample(args.subset, random_state=SEED).reset_index(drop=True)

    MODEL = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    collator = DataCollatorWithPadding(tokenizer)
    test_ds = PairDataset(test.assign(score=0.0), tokenizer, args.max_len,
                          with_labels=False)
    test_loader = DataLoader(test_ds, batch_size=128, collate_fn=collator,
                             num_workers=2, pin_memory=True)

    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(train, groups=train.anchor))
    oof = np.full(len(train), np.nan)
    test_preds, fold_scores = [], []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        if fold >= args.folds:
            break
        t0 = time.time()
        set_seed(seed + fold)
        tr_ds = PairDataset(train.iloc[tr_idx], tokenizer, args.max_len)
        va_ds = PairDataset(train.iloc[va_idx], tokenizer, args.max_len)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                               collate_fn=collator, num_workers=2,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=128, collate_fn=collator,
                               num_workers=2, pin_memory=True)

        # The GPU is shared with another benchmark job; allocation can fail
        # transiently. Retry rather than lose the whole run.
        model = None
        for attempt in range(12):
            try:
                model = MeanPoolRegressor(MODEL).to(device)
                break
            except (torch.AcceleratorError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"fold {fold}: OOM on model init (attempt {attempt}), "
                      f"retrying in 30s", flush=True)
                model = None
                torch.cuda.empty_cache()
                time.sleep(30)
        if model is None:
            raise RuntimeError("could not allocate model after 12 attempts")
        opt = make_optimizer(model, args.lr, args.head_lr, args.wd,
                             args.opt_impl)
        total_steps = len(tr_loader) * args.epochs
        sched = get_cosine_schedule_with_warmup(
            opt, int(0.1 * total_steps), total_steps)
        loss_fn = nn.BCEWithLogitsLoss()

        best_r, best_va, best_test = -2.0, None, None
        last_va, last_test = None, None
        step = 0
        n_skipped = 0
        stop = False
        for epoch in range(args.epochs):
            model.train()
            run_loss, run_gn, nb = 0.0, 0.0, 0
            for batch in tr_loader:
                labels = batch.pop("labels").to(device, non_blocking=True)
                batch = {k: v.to(device, non_blocking=True)
                         for k, v in batch.items()}
                if amp is None:
                    logits = model(**batch)
                    loss = loss_fn(logits.float(), labels)
                else:
                    with torch.autocast("cuda", dtype=amp):
                        logits = model(**batch)
                        loss = loss_fn(logits.float(), labels)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Guard: never apply an update from non-finite grads - a single
                # bad step poisons every weight and every later metric is nan.
                if not torch.isfinite(gn):
                    n_skipped += 1
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    continue
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                run_loss += loss.item()
                run_gn += float(gn)
                nb += 1
                step += 1
                if not math.isfinite(loss.item()):
                    print(f"!! non-finite train loss at step {step}", flush=True)
                    stop = True
                    break
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break
            va_pred, va_lab = predict(model, va_loader, device, amp)
            r = safe_pearson(va_lab, va_pred)
            last_va = va_pred
            print(f"fold {fold} epoch {epoch}: loss={run_loss/max(nb,1):.5f} "
                  f"gnorm={run_gn/max(nb,1):.3f} val_r={r:.4f} "
                  f"pred_std={va_pred.std():.4f} pred_mean={va_pred.mean():.4f} "
                  f"skipped={n_skipped} ({time.time()-t0:.0f}s)", flush=True)
            if math.isfinite(r) and r > best_r:
                best_r = r
                best_va = va_pred
                best_test, _ = predict(model, test_loader, device, amp)
            if stop:
                break

        if best_va is None:  # no usable epoch -> fall back to last state
            print(f"fold {fold}: no improving epoch, falling back to last",
                  flush=True)
            best_va = last_va
            best_test, _ = predict(model, test_loader, device, amp)
            best_r = safe_pearson(train.score.values[va_idx], best_va)

        oof[va_idx] = best_va
        test_preds.append(best_test)
        fold_scores.append(best_r)
        print(f"fold {fold} best r={best_r:.4f} "
              f"time={(time.time()-t0)/60:.1f}min", flush=True)
        # Checkpoint after every fold so a crash/kill still leaves usable preds.
        np.save(f"{COMP}/scripts/{args.tag}_oof_partial.npy", oof)
        np.save(f"{COMP}/scripts/{args.tag}_test_partial.npy",
                np.mean(test_preds, axis=0))
        json.dump({"folds_done": len(fold_scores),
                   "fold_scores": fold_scores},
                  open(f"{COMP}/scripts/{args.tag}_progress.json", "w"),
                  indent=2)
        del model, opt
        torch.cuda.empty_cache()
        if smoke:
            break

    if smoke:
        print("SMOKE OK")
        return

    mask = ~np.isnan(oof)
    r_oof = safe_pearson(train.score.values[mask], oof[mask])
    test_pred = np.mean(test_preds, axis=0)
    print(f"OOF Pearson r = {r_oof:.5f}  folds={[round(s,5) for s in fold_scores]}",
          flush=True)

    np.save(f"{COMP}/scripts/{args.tag}_oof.npy", oof)
    np.save(f"{COMP}/scripts/{args.tag}_test.npy", test_pred)
    json.dump({"model": args.tag, "oof_pearson": r_oof,
               "fold_scores": fold_scores, "n_folds": len(fold_scores),
               "args": vars(args)},
              open(f"{COMP}/scripts/{args.tag}_cv.json", "w"), indent=2)

    sub = pd.read_csv(f"{DATA}/sample_submission.csv")
    assert list(sub.id) == list(test.id), "id order mismatch"
    sub["score"] = np.clip(test_pred, 0, 1)
    sub.to_csv(f"{COMP}/submissions/{args.tag}.csv", index=False)
    if args.tag == "deberta_v3_base":
        sub.to_csv(f"{COMP}/submission.csv", index=False)
    print("submission written", flush=True)

    _spec = importlib.util.spec_from_file_location(
        "experiment_log",
        "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/"
        "experiment_log.py")
    experiment_log = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(experiment_log)
    experiment_log.log_experiment_v2(
        COMP, model=f"{args.tag}-meanpool-bce", metric="pearson_r",
        direction="maximize", score=r_oof,
        cv={"scheme": f"GroupKFold{args.n_splits}_by_anchor", "seed": seed,
            "fold_scores": [round(s, 5) for s in fold_scores]},
        features=["anchor [SEP] target [SEP] section-title context-code"],
        submission=f"submissions/{args.tag}.csv",
        notes=(f"{args.model} mean-pool head, BCEWithLogits, lr={args.lr}, "
               f"head_lr={args.head_lr}, batch={args.batch}, "
               f"epochs={args.epochs}, amp={args.amp}, opt={args.opt_impl}, "
               f"max_len={args.max_len}."))


if __name__ == "__main__":
    main()
