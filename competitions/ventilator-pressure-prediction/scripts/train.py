"""Ventilator pressure prediction — run2: bidirectional LSTM, bf16 AMP, chunked checkpoint/resume.

Metric: MAE on inspiratory rows only (u_out == 0). Validation replicates the mask.
Changes vs run1: +12 features (RxC cross one-hot, cummax/cummean, reverse-cumsum),
bf16 autocast, GPU-resident data, 260-epoch OneCycle, full 5-fold ensemble.
Runs in ~9-minute foreground chunks: saves checkpoint and exits 3 when out of chunk
time; rerun same command to resume. Exit 0 = fully done.
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
p.add_argument("--max-folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=260)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--subset", type=int, default=0, help="use only N breaths (smoke test)")
p.add_argument("--tag", type=str, default="lstm_run2")
p.add_argument("--chunk-seconds", type=float, default=510.0, help="wall seconds per invocation before checkpoint+exit")
p.add_argument("--workdir", type=str, default=f"{COMP}/scripts/work")
args = p.parse_args()

os.makedirs(args.workdir, exist_ok=True)
CKPT = f"{args.workdir}/ckpt_{args.tag}.pt"
CACHE = f"{args.workdir}/cache_{args.tag}.npz"

torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.time()


def log(*a):
    print(f"[{time.time()-t_start:8.1f}s]", *a, flush=True)


def out_of_time():
    return time.time() - t_start > args.chunk_seconds


# ---------------- data ----------------
def load(fn, has_y):
    cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"] + (["pressure"] if has_y else [])
    df = pd.read_csv(f"{COMP}/data/{fn}", usecols=["id"] + cols)
    df = df.sort_values(["breath_id", "time_step"], kind="stable").reset_index(drop=True)
    return df


def build_features(df):
    """Return X (n_breaths, 80, F) float32, plus u_out mask and ids."""
    n = len(df) // T
    assert len(df) % T == 0
    g = lambda c: df[c].to_numpy(np.float32).reshape(n, T)
    u_in, u_out, ts = g("u_in"), g("u_out"), g("time_step")
    R, C = g("R"), g("C")
    dt = np.diff(ts, axis=1, prepend=0.0).astype(np.float32)

    feats = []
    add = feats.append
    add(u_in); add(u_out); add(ts); add(dt)
    add(np.log1p(R)); add(np.log1p(C)); add(np.log1p(R * C))
    for v in (5.0, 20.0, 50.0):
        add((R == v).astype(np.float32))
    for v in (10.0, 20.0, 50.0):
        add((C == v).astype(np.float32))
    # cross R x C one-hot (9 combos)
    for rv in (5.0, 20.0, 50.0):
        for cv in (10.0, 20.0, 50.0):
            add(((R == rv) & (C == cv)).astype(np.float32))
    # cumulative / integral features
    area = np.cumsum(u_in * dt, axis=1, dtype=np.float32)
    add(area)
    add(np.cumsum(u_in, axis=1, dtype=np.float32))
    add(area / np.maximum(C, 1.0))
    add(np.maximum.accumulate(u_in, axis=1))
    add(np.cumsum(u_in, axis=1, dtype=np.float32) / np.arange(1, T + 1, dtype=np.float32))
    add(np.cumsum(u_in[:, ::-1], axis=1, dtype=np.float32)[:, ::-1].copy())
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


if os.path.exists(CACHE):
    log("loading cached features ...")
    z = np.load(CACHE)
    Xtr, uo_tr, ytr = z["Xtr"], z["uo_tr"], z["ytr"]
    Xte, id_te, PGRID = z["Xte"], z["id_te"], z["PGRID"]
else:
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
    # normalize once, cache normalized
    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    np.savez(CACHE, Xtr=Xtr, uo_tr=uo_tr, ytr=ytr, Xte=Xte, id_te=id_te, PGRID=PGRID)
log(f"features: train {Xtr.shape}  test {Xte.shape}  pressure grid {len(PGRID)}")
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
splits = list(kf.split(Xtr))

# ---------------- checkpoint state ----------------
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    log(f"resuming: fold {ck['fold']} epoch {ck['epoch']}")
else:
    ck = {"fold": 0, "epoch": 0, "model": None, "opt": None, "sched": None,
          "best": 1e9, "best_state": None,
          "oof": np.zeros_like(ytr), "oof_done": np.zeros(len(ytr), bool),
          "fold_scores": []}

Xtr_t = torch.from_numpy(Xtr).to(dev)
Xte_t = torch.from_numpy(Xte).to(dev)
ytr_t = torch.from_numpy((ytr - y_mu) / y_sd).to(dev)
mask_t = torch.from_numpy((uo_tr == 0).astype(np.float32)).to(dev)


def predict(model, X_t, n):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, n, 2048):
            out.append(model(X_t[b:b + 2048]).float().cpu().numpy())
    return np.concatenate(out) * y_sd + y_mu


for fold in range(ck["fold"], min(args.max_folds, len(splits))):
    itr, iva = splits[fold]
    model = Net(F, args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    nb = int(np.ceil(len(itr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * nb, pct_start=0.1)
    if ck["model"] is not None and fold == ck["fold"]:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
    else:
        torch.manual_seed(SEED + fold)
        model = Net(F, args.hidden, args.layers).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * nb, pct_start=0.1)
        ck.update({"fold": fold, "epoch": 0, "best": 1e9, "best_state": None})
    itr_t = torch.from_numpy(itr).to(dev)
    iva_t = torch.from_numpy(iva).to(dev)

    for ep in range(ck["epoch"], args.epochs):
        model.train()
        torch.manual_seed(SEED * 1000 + fold * 997 + ep)   # deterministic, resume-safe
        perm = torch.randperm(len(itr), device=dev)
        tot = 0.0
        for b in range(nb):
            sl = itr_t[perm[b * args.batch:(b + 1) * args.batch]]
            xb, yb, mb = Xtr_t[sl], ytr_t[sl], mask_t[sl]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = (torch.abs(out - yb) * mb).sum() / mb.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            pv = predict(model, Xtr_t[iva_t], len(iva))
            sc = masked_mae(pv, ytr[iva], (uo_tr[iva] == 0).astype(np.float32))
            log(f"fold {fold} ep {ep+1}/{args.epochs} train {tot/nb:.4f} val_maskedMAE {sc:.5f}")
            if sc < ck["best"]:
                ck["best"] = float(sc)
                ck["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ck["oof"][iva] = pv
        if out_of_time() and ep < args.epochs - 1:
            ck.update({"fold": fold, "epoch": ep + 1,
                       "model": model.state_dict(), "opt": opt.state_dict(),
                       "sched": sched.state_dict()})
            torch.save(ck, CKPT)
            log(f"CHUNK_DONE fold {fold} epoch {ep+1}/{args.epochs} — resume to continue")
            sys.exit(3)

    # fold finished
    ck["oof_done"][iva] = True
    ck["fold_scores"].append(float(ck["best"]))
    log(f"fold {fold} best masked MAE {ck['best']:.5f}")
    model.load_state_dict(ck["best_state"])
    np.save(f"{args.workdir}/testpred_{args.tag}_f{fold}.npy", predict(model, Xte_t, len(Xte)))
    ck.update({"fold": fold + 1, "epoch": 0, "model": None, "opt": None, "sched": None,
               "best": 1e9, "best_state": None})
    torch.save(ck, CKPT)
    del model
    torch.cuda.empty_cache()
    if out_of_time() and fold + 1 < min(args.max_folds, len(splits)):
        log(f"CHUNK_DONE fold {fold} complete — resume to continue")
        sys.exit(3)

# ---------------- evaluation (all folds done) ----------------
oof, oof_done = ck["oof"], ck["oof_done"]
m = (uo_tr == 0).astype(np.float32) * oof_done[:, None]
cv_raw = float(masked_mae(oof, ytr, m))
cv_snap = float(masked_mae(snap(oof), ytr, m))
log(f"OOF ({int(oof_done.sum())} breaths) masked MAE raw {cv_raw:.5f}  snapped {cv_snap:.5f}")
use_snap = cv_snap < cv_raw

test_preds = [np.load(f"{args.workdir}/testpred_{args.tag}_f{f}.npy")
              for f in range(min(args.max_folds, len(splits)))]
pred = np.mean(test_preds, axis=0)
if use_snap:
    pred = snap(pred)
sub = pd.DataFrame({"id": id_te.ravel(), "pressure": pred.ravel()}).sort_values("id")
ss = pd.read_csv(f"{COMP}/data/sample_submission.csv")
assert len(sub) == len(ss) and (sub["id"].to_numpy() == ss["id"].to_numpy()).all()
out = f"{COMP}/submission.csv" if not args.subset else f"{COMP}/submission_smoke.csv"
sub.to_csv(out, index=False)
log(f"wrote {out}  snap={use_snap}  folds_trained={len(test_preds)}")

json.dump({"cv_raw": cv_raw, "cv_snap": cv_snap, "fold_scores": ck["fold_scores"],
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
        COMP, model=f"BiLSTM-{args.layers}x{args.hidden} (seq2seq, masked L1, bf16)",
        metric="masked_MAE(u_out==0)", direction="minimize",
        score=min(cv_raw, cv_snap),
        cv={"strategy": f"{args.folds}-fold KFold on breaths",
            "folds_trained": len(test_preds), "fold_scores": ck["fold_scores"],
            "cv_raw": cv_raw, "cv_snapped": cv_snap,
            "params": {k: str(v) for k, v in vars(args).items()}},
        postprocess=["snap-to-pressure-grid"] if use_snap else [],
        submission="submission.csv",
        notes=f"run2: {F} features (run1's 41 + RxC cross one-hot + cummax/cummean/rev-cumsum), "
              f"bf16 autocast, GPU-resident data, 260-epoch OneCycle, 5-fold mean ensemble; "
              f"raw OOF {cv_raw:.5f} / snapped {cv_snap:.5f}")
log("done")
