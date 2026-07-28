"""DeBERTa-v3-base 5-fold fine-tune for patent phrase similarity (Pearson r).

Offline: uses the locally cached microsoft/deberta-v3-base snapshot.
Input: pair encoding of "context_code anchor" vs "target", MSE regression head.
"""
import argparse
import importlib.util
import os
import random
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

SEED = 42
COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
DATA = f"{COMP}/data"
MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 64
BATCH = 64
EPOCHS = 3
LR = 2e-5
N_FOLDS = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, with_labels=True):
        self.enc = tokenizer(
            (df.context + " " + df.anchor).tolist(),
            df.target.tolist(),
            truncation=True, max_length=MAX_LEN)
        self.labels = df.score.values.astype("float32") if with_labels else None

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            lb = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch).logits.squeeze(-1)
            preds.append(out.float().cpu().numpy())
            if lb is not None:
                labels.append(lb.numpy())
    preds = np.concatenate(preds)
    return preds, (np.concatenate(labels) if labels else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda"
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    collator = DataCollatorWithPadding(tokenizer)
    test_ds = PairDataset(test.assign(score=0.0), tokenizer, with_labels=False)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collator,
                             num_workers=2, pin_memory=True)

    strat = (train.score.values * 4).astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, strat)):
        if fold >= args.folds:
            break
        t0 = time.time()
        set_seed(SEED + fold)
        tr_ds = PairDataset(train.iloc[tr_idx], tokenizer)
        va_ds = PairDataset(train.iloc[va_idx], tokenizer)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                               collate_fn=collator, num_workers=2,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=256, collate_fn=collator,
                               num_workers=2, pin_memory=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=1).to(device)
        model.config.problem_type = "regression"
        no_decay = ["bias", "LayerNorm.weight"]
        grouped = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        # NOTE: on this torch 2.14-dev/GB10 build, the foreach AND single-tensor
        # AdamW paths corrupt weights to NaN on the first step (verified by
        # isolation test with synthetic grads); fused=True is the only clean path.
        opt = torch.optim.AdamW(grouped, lr=LR, fused=True)
        total_steps = len(tr_loader) * EPOCHS
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        loss_fn = nn.MSELoss()

        best_r, best_va_pred, best_test_pred = -1.0, None, None
        step = 0
        for epoch in range(EPOCHS):
            model.train()
            for batch in tr_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**batch).logits.squeeze(-1)
                    loss = loss_fn(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if args.smoke and step >= 30:
                    print(f"smoke: 30 steps in {time.time()-t0:.1f}s "
                          f"(incl. model load)")
                    return
            va_pred, va_lab = evaluate(model, va_loader, device)
            r = pearsonr(va_lab, va_pred)[0]
            print(f"fold {fold} epoch {epoch}: val r={r:.4f}", flush=True)
            if r > best_r:
                best_r = r
                best_va_pred = va_pred
                best_test_pred, _ = evaluate(model, test_loader, device)

        oof[va_idx] = best_va_pred
        test_pred += best_test_pred / args.folds
        fold_scores.append(best_r)
        print(f"fold {fold} best r={best_r:.4f} time={(time.time()-t0)/60:.1f}min",
              flush=True)
        del model
        torch.cuda.empty_cache()

    used = np.concatenate([va for _, va in
                           list(skf.split(train, strat))[:args.folds]])
    r_oof = pearsonr(train.score.values[used], oof[used])[0]
    print(f"OOF Pearson r = {r_oof:.4f}  folds={fold_scores}")

    np.save(f"{COMP}/scripts/deberta_oof.npy", oof)
    np.save(f"{COMP}/scripts/deberta_test.npy", test_pred)

    sub = pd.read_csv(f"{DATA}/sample_submission.csv")
    assert list(sub.id) == list(test.id), "id order mismatch"
    sub["score"] = np.clip(test_pred, 0, 1)
    sub.to_csv(f"{COMP}/submissions/deberta_v3_base_5fold.csv", index=False)
    print("submission written")

    _spec = importlib.util.spec_from_file_location(
        "experiment_log",
        "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/"
        "experiment_log.py")
    experiment_log = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(experiment_log)
    experiment_log.log_experiment_v2(
        COMP, model="deberta-v3-base-5fold-mse", metric="pearson_r",
        direction="maximize", score=r_oof,
        cv={"scheme": "StratifiedKFold5_on_score", "seed": SEED,
            "fold_scores": [round(s, 5) for s in fold_scores]},
        features=["context+anchor [SEP] target, max_len=64"],
        submission="submissions/deberta_v3_base_5fold.csv",
        notes=f"lr={LR}, batch={BATCH}, epochs={EPOCHS}, bf16 autocast, "
              "best-epoch checkpointing per fold.")


if __name__ == "__main__":
    main()
