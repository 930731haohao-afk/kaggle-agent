"""Diagnostic: isolate the source of NaN / collapse in the DeBERTa fine-tune.

Runs several short configs on a small subset and reports, per step:
train loss, grad global norm, prediction std, and whether any weight is NaN.
"""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

torch.set_num_threads(8)
COMP = "/home/tjyen/ai_agents/kaggle/competitions/us-patent-phrase-to-phrase-matching"
DATA = f"{COMP}/data"
MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 64
N = 2048


class PairDataset(Dataset):
    def __init__(self, df, tok):
        self.enc = tok((df.context + " " + df.anchor).tolist(),
                       df.target.tolist(), truncation=True, max_length=MAX_LEN)
        self.labels = df.score.values.astype("float32")

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def run(tag, amp_dtype, opt_kind, lr, steps=60, batch=32):
    torch.manual_seed(0)
    np.random.seed(0)
    train = pd.read_csv(f"{DATA}/train.csv").sample(N, random_state=0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = PairDataset(train, tok)
    dl = DataLoader(ds, batch_size=batch, shuffle=True,
                    collate_fn=DataCollatorWithPadding(tok), drop_last=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1).cuda()
    model.config.problem_type = "regression"
    if opt_kind == "fused":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    elif opt_kind == "foreach":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
    elif opt_kind == "single":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False,
                                fused=False)
    loss_fn = nn.MSELoss()
    model.train()
    step = 0
    print(f"\n=== {tag} (amp={amp_dtype}, opt={opt_kind}, lr={lr}) ===",
          flush=True)
    done = False
    while not done:
        for batch_ in dl:
            labels = batch_.pop("labels").cuda()
            b = {k: v.cuda() for k, v in batch_.items()}
            if amp_dtype is None:
                logits = model(**b).logits.squeeze(-1)
                loss = loss_fn(logits.float(), labels)
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(**b).logits.squeeze(-1)
                    loss = loss_fn(logits.float(), labels)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step <= 3:
                nan_w = sum(int(torch.isnan(p).any()) for p in model.parameters())
                print(f"  step {step:3d} loss={loss.item():.5f} "
                      f"gnorm={gnorm.item():.4f} "
                      f"pred_std={logits.float().std().item():.5f} "
                      f"pred_mean={logits.float().mean().item():.4f} "
                      f"nan_params={nan_w}", flush=True)
                if nan_w:
                    print("  --> WEIGHTS WENT NaN", flush=True)
                    return
            if step >= steps:
                done = True
                break
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfgs = {
        "fp32_fused": (None, "fused", 2e-5),
        "bf16_fused": (torch.bfloat16, "fused", 2e-5),
        "bf16_foreach": (torch.bfloat16, "foreach", 2e-5),
        "fp32_foreach": (None, "foreach", 2e-5),
        "fp32_single": (None, "single", 2e-5),
    }
    for name, (dt, ok, lr) in cfgs.items():
        if which not in ("all", name):
            continue
        try:
            run(name, dt, ok, lr)
        except Exception as e:
            print(f"  !! {name} raised {type(e).__name__}: {e}", flush=True)
