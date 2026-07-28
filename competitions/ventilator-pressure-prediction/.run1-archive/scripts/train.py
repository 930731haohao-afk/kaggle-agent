"""Ventilator pressure prediction — bidirectional LSTM on per-breath sequences.

Metric: MAE on inspiratory rows only (u_out == 0). Validation replicates the mask.
Run: setsid nohup .venv/bin/python scripts/train.py > train.log 2>&1 < /dev/null &
"""
import argparse, json, os, time, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

COMP = "/home/tjyen/ai_agents/kaggle/competitions/ventilator-pressure-prediction"
SEED = 42
T = 80

p = argparse.ArgumentParser()
p.add_argument("--folds", type=int, default=5)
p.add_argument("--max-folds", type=int, default=5, help="how many folds actually trained")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--subset", type=int, default=0, help="use only N breaths (smoke test)")
p.add_argument("--tag", type=str, default="lstm")
p.add_argument("--time-budget", type=float, default=1e9, help="seconds; stop starting new folds after")
args = p.parse_args()

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.time()


def log(*a):
    print(f"[{time.time()-t_start:8.1f}s]", *a, flush=True)


# ---------------- data ----------------
def load(fn, has_y):
    cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"] + (["pressure"] if has_y else [])
    df = pd.read_csv(f"{COMP}/data/{fn}", usecols=["id"] + cols)
    df = df.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    return df


def build_features(df):
    """Return X (n_breaths, 80, F) float32, plus u_out mask and ids."""
    n = len(df) // T
    g = lambda c: df[c].to_numpy(np.float32).reshape(n, T)
    u_in, u_out, ts = g("u_in"), g("u_out"), g("time_step")
    R, C = g("R"), g("C")
    dt = np.diff(ts, axis=1, prepend=0.0).astype(np.float32)

    feats = []
    add = feats.append
    add(u_in); add(u_out); add(ts); add(dt)
    add(np.log1p(R)); add(np.log1p(C)); add(np.log1p(R * C))
    # R / C one-hot (constant per breath, broadcast)
    for v in (5.0, 20.0, 50.0):
        add((R == v).astype(np.float32))
    for v in (10.0, 20.0, 50.0):
        add((C == v).astype(np.float32))
    # cumulative / integral features
    area = np.cumsum(u_in * dt, axis=1, dtype=np.float32)
    add(area)
    add(np.cumsum(u_in, axis=1, dtype=np.float32))
    add(area / np.maximum(C, 1.0))
    # lags & leads of u_in
    for k in (1, 2, 3, 4):
        lag = np.concatenate([np.zeros((n, k), np.float32), u_in[:, :-k]], axis=1)
        lead = np.concatenate([u_in[:, k:], np.zeros((n, k), np.float32)], axis=1)
        add(lag); add(lead)
        add(u_in - lag); add(lead - u_in)
    # derivative
    add(np.diff(u_in, axis=1, prepend=0.0).astype(np.float32) / np.maximum(dt, 1e-3))
    # breath-level aggregates broadcast
    for arr in (u_in.mean(1), u_in.max(1), u_in.std(1), u_in[:, 0]):
        add(np.repeat(arr[:, None], T, axis=1).astype(np.float32))
    # u_out timing
    add(np.cumsum(u_out, axis=1, dtype=np.float32))
    add(np.repeat(u_out.argmax(1)[:, None].astype(np.float32), T, axis=1))
    # interactions
    add(u_in / np.maximum(R, 1.0))
    add(u_in * np.log1p(C))

    X = np.stack(feats, axis=2).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ids = df["id"].to_numpy(np.int64).reshape(n, T)
    y = df["pressure"].to_numpy(np.float32).reshape(n, T) if "pressure" in df else None
    return X, u_out, ids, y


log("loading data ...")
tr = load("train.csv", True)
te = load("test.csv", False)
if args.subset:
    keep = tr["breath_id"].unique()[: args.subset]
    tr = tr[tr["breath_id"].isin(keep)].reset_index(drop=True)
log(f"train rows {len(tr)}  test rows {len(te)}")

Xtr, uo_tr, id_tr, ytr = build_features(tr)
Xte, uo_te, id_te, _ = build_features(te)
PGRID = np.sort(tr["pressure"].unique()).astype(np.float32)
del tr, te
log(f"features: train {Xtr.shape}  test {Xte.shape}  pressure grid {len(PGRID)}")

mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd
F = Xtr.shape[2]

y_mu, y_sd = float(ytr.mean()), float(ytr.std())


class Net(nn.Module):
    def __init__(self, f, h, layers):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f, h), nn.LayerNorm(h), nn.SiLU())
        self.lstm = nn.LSTM(h, h, num_layers=layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * h, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.inp(x)
        x, _ = self.lstm(x)
        return self.head(x).squeeze(-1)


def masked_mae(pred, y, mask):
    return (np.abs(pred - y) * mask).sum() / mask.sum()


def snap(pred):
    idx = np.searchsorted(PGRID, pred)
    idx = np.clip(idx, 1, len(PGRID) - 1)
    lo, hi = PGRID[idx - 1], PGRID[idx]
    return np.where(np.abs(pred - lo) <= np.abs(hi - pred), lo, hi)


kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
oof = np.zeros_like(ytr)
oof_done = np.zeros(len(ytr), bool)
test_preds = []
fold_scores = []

Xte_t = torch.from_numpy(Xte)

for fold, (itr, iva) in enumerate(kf.split(Xtr)):
    if fold >= args.max_folds:
        break
    if fold > 0 and time.time() - t_start > args.time_budget:
        log(f"time budget hit; stopping before fold {fold}")
        break
    torch.manual_seed(SEED + fold)
    model = Net(F, args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(Xtr[itr]); yt = torch.from_numpy((ytr[itr] - y_mu) / y_sd)
    mt = torch.from_numpy((uo_tr[itr] == 0).astype(np.float32))
    xv = torch.from_numpy(Xtr[iva])
    nb = int(np.ceil(len(itr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * nb, pct_start=0.1)
    crit = nn.L1Loss(reduction="none")
    best = 1e9; best_state = None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(itr))
        tot = 0.0
        for b in range(nb):
            sl = perm[b * args.batch:(b + 1) * args.batch]
            xb, yb, mb = xt[sl].to(dev, non_blocking=True), yt[sl].to(dev), mt[sl].to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = (crit(out, yb) * mb).sum() / mb.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            model.eval()
            pv = []
            with torch.no_grad():
                for b in range(0, len(iva), 1024):
                    pv.append(model(xv[b:b + 1024].to(dev)).cpu().numpy())
            pv = np.concatenate(pv) * y_sd + y_mu
            sc = masked_mae(pv, ytr[iva], (uo_tr[iva] == 0).astype(np.float32))
            log(f"fold {fold} ep {ep+1}/{args.epochs} train {tot/nb:.4f} val_maskedMAE {sc:.5f}")
            if sc < best:
                best = sc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                oof[iva] = pv
    oof_done[iva] = True
    fold_scores.append(float(best))
    log(f"fold {fold} best masked MAE {best:.5f}")
    model.load_state_dict(best_state)
    model.eval()
    pt = []
    with torch.no_grad():
        for b in range(0, len(Xte), 1024):
            pt.append(model(Xte_t[b:b + 1024].to(dev)).cpu().numpy())
    test_preds.append(np.concatenate(pt) * y_sd + y_mu)
    del model, xt, yt, mt, xv
    torch.cuda.empty_cache()

# ---------------- evaluation ----------------
m = (uo_tr == 0).astype(np.float32) * oof_done[:, None]
cv_raw = float(masked_mae(oof, ytr, m))
cv_snap = float(masked_mae(snap(oof), ytr, m))
log(f"OOF ({oof_done.sum()} breaths) masked MAE raw {cv_raw:.5f}  snapped {cv_snap:.5f}")
use_snap = cv_snap < cv_raw

pred = np.mean(test_preds, axis=0)
if use_snap:
    pred = snap(pred)
sub = pd.DataFrame({"id": id_te.ravel(), "pressure": pred.ravel()}).sort_values("id")
ss = pd.read_csv(f"{COMP}/data/sample_submission.csv")
assert len(sub) == len(ss) and (sub["id"].to_numpy() == ss["id"].to_numpy()).all()
out = f"{COMP}/submission.csv" if not args.subset else f"{COMP}/submission_smoke.csv"
sub.to_csv(out, index=False)
log(f"wrote {out}  snap={use_snap}  folds_trained={len(test_preds)}")

json.dump({"cv_raw": cv_raw, "cv_snap": cv_snap, "fold_scores": fold_scores,
           "n_folds_trained": len(test_preds), "snap_applied": bool(use_snap),
           "elapsed_s": time.time() - t_start, "args": vars(args)},
          open(f"{COMP}/cv_result{'_smoke' if args.subset else ''}.json", "w"), indent=2)

if not args.subset:
    np.save(f"{COMP}/oof_{args.tag}.npy", oof)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "experiment_log",
        "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
    el = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(el)
    el.log_experiment_v2(
        COMP, model=f"BiLSTM-{args.layers}x{args.hidden}", metric="masked_MAE(u_out==0)",
        direction="minimize", score=min(cv_raw, cv_snap),
        cv={"strategy": f"{args.folds}-fold KFold on breath_id",
            "folds_trained": len(test_preds), "fold_scores": fold_scores,
            "params": {k: str(v) for k, v in vars(args).items()}},
        postprocess=["snap-to-pressure-grid"] if use_snap else [],
        submission="submission.csv",
        notes=f"{F} sequence features (lags/leads/area/R,C one-hot); "
              f"raw OOF {cv_raw:.5f} / snapped {cv_snap:.5f}; snap_applied={use_snap}")
log("done")
