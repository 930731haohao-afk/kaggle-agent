"""s6e7 v3b — FIX the MLP (v3 produced NaN: numeric columns have heavy NaN that GBDTs
eat natively but an MLP cannot). Fix = per-fold median imputation + missing-indicator
features + standardization + grad clipping. Then re-blend the full 7-member pool
(5 GBDT + ExtraTrees + fixed MLP) and report best CV vs v2 (0.94998).

Folds = identical StratifiedKFold(5, shuffle=True, random_state=42) as the GBDT OOF.
Only writes a submission CSV if CV > v2; does NOT auto-submit to Kaggle.
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch, torch.nn as nn

COMP = Path(__file__).resolve().parent.parent; ROOT = COMP.parent.parent
_spec = importlib.util.spec_from_file_location("experiment_log", ROOT/".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(experiment_log)

SEED, NF, V2 = 42, 5, 0.9499789257329395
NUM = ["sleep_duration","heart_rate","bmi","calorie_expenditure","step_count","exercise_duration","water_intake"]
CAT = ["diet_type","stress_level","sleep_quality","physical_activity_level","smoking_alcohol","gender"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

tr = pd.read_csv(COMP/"data/train.csv"); te = pd.read_csv(COMP/"data/test.csv")
classes = sorted(tr["health_condition"].unique()); c2i = {c:i for i,c in enumerate(classes)}
y = tr["health_condition"].map(c2i).to_numpy()
for c in CAT:
    tr[c] = tr[c].fillna("missing").astype(str); te[c] = te[c].fillna("missing").astype(str)
folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(tr, y))
cw = (len(y)/(3*np.bincount(y))).astype(np.float32)
def balacc(yv, pred): return float(np.mean([(pred[yv==c]==c).mean() for c in range(3)]))
fixed = np.array([1,1.4,1.3])

# raw numeric (with NaN), missing-mask, one-hot cats
Xnum = tr[NUM].to_numpy(np.float32); Xnum_te = te[NUM].to_numpy(np.float32)
miss = np.isnan(Xnum).astype(np.float32); miss_te = np.isnan(Xnum_te).astype(np.float32)
dummies = pd.get_dummies(pd.concat([tr[CAT], te[CAT]], ignore_index=True), columns=CAT)
Xcat = dummies.iloc[:len(tr)].to_numpy(np.float32); Xcat_te = dummies.iloc[len(tr):].to_numpy(np.float32)
log(f"train={tr.shape} cw={cw.round(2)} num={len(NUM)} miss-ind={miss.shape[1]} cat_oh={Xcat.shape[1]} dev={DEV}")

class MLP(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128,3))
    def forward(s,x): return s.net(x)

def run_mlp():
    torch.manual_seed(SEED); np.random.seed(SEED)
    oof = np.zeros((len(tr),3), np.float32); test = np.zeros((len(te),3), np.float32)
    wce = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=DEV))
    for k,(tri,vai) in enumerate(folds):
        med = np.nanmedian(Xnum[tri], 0)                       # per-fold median (train only)
        def build(Xn, Mi):
            Xi = np.where(np.isnan(Xn), med, Xn)
            mu = np.where(np.isnan(Xnum[tri]), med, Xnum[tri]).mean(0); sd = np.where(np.isnan(Xnum[tri]), med, Xnum[tri]).std(0)+1e-6
            return np.concatenate([(Xi-mu)/sd, Mi], 1).astype(np.float32)
        Xtr_full = build(Xnum, miss); Xte_full = build(Xnum_te, miss_te)
        Xtr_full = np.concatenate([Xtr_full, Xcat], 1); Xte_full = np.concatenate([Xte_full, Xcat_te], 1)
        assert not np.isnan(Xtr_full).any(), "still NaN after impute"
        Xtri = torch.tensor(Xtr_full[tri], device=DEV); ytri = torch.tensor(y[tri], device=DEV)
        Xvai = torch.tensor(Xtr_full[vai], device=DEV); Xte_t = torch.tensor(Xte_full, device=DEV)
        m = MLP(Xtr_full.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
        n = len(tri); bs = 8192; best_ba, best_state, patience, ep = -1, None, 0, 0
        for ep in range(120):
            m.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = wce(m(Xtri[idx]), ytri[idx]); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            m.eval()
            with torch.no_grad(): vp = m(Xvai).softmax(1).cpu().numpy()
            ba = balacc(y[vai], (vp*fixed).argmax(1))
            if ba > best_ba + 1e-5: best_ba, best_state, patience = ba, {kk:v.cpu().clone() for kk,v in m.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 15: break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            oof[vai] = m(Xvai).softmax(1).cpu().numpy()
            test += m(Xte_t).softmax(1).cpu().numpy()/NF
        log(f"  MLP fold{k}: adj-balacc={best_ba:.4f} (stopped ep~{ep})")
    return oof, test

log("=== fixed MLP ===")
oof, test = run_mlp()
np.savez_compressed(COMP/"data/oof_mlp.npz", oof=oof, test=test)
log(f"mlp: solo adj-balacc={balacc(y,(oof*fixed).argmax(1)):.5f}  anynan={np.isnan(oof).any()}")

# ---------- re-blend full pool ----------
names = ["lgbm","cat","xgb","lgbm_tuned","lgbm_seed2024","et","mlp"]
O = np.stack([np.load(COMP/f"data/oof_{n}.npz")["oof"] for n in names]).astype(np.float64)
T = np.stack([np.load(COMP/f"data/oof_{n}.npz")["test"] for n in names]).astype(np.float64)
assert not np.isnan(O).any(), "NaN in pool"
A = np.linspace(0.7,3,16); rng = np.random.default_rng(SEED); M = len(names)
def tune(P):
    ba,adj = balacc(y,(P*fixed).argmax(1)), fixed
    for a in A:
        for b in A:
            s = balacc(y,(P*np.array([1,a,b])).argmax(1))
            if s>ba: ba,adj = s,np.array([1,a,b])
    return adj,ba
cand = np.vstack([rng.dirichlet(np.ones(M), size=30000), np.eye(M)])
sc = np.array([balacc(y,(np.tensordot(w,O,1)*fixed).argmax(1)) for w in cand])
top = cand[np.argsort(sc)[::-1][:60]]
best = (None,None,-1)
for w in top:
    adj,s = tune(np.tensordot(w,O,1))
    if s>best[2]: best = (w,adj,s)
w,adj,s = best
log(f"V3b BLEND balacc={s:.6f} (v2={V2:.6f}, Δ={s-V2:+.6f}) adj={adj.round(2)}")
log("  weights: " + ", ".join(f"{n}={wi:.3f}" for n,wi in zip(names,w) if wi>1e-3))
log(f"  MLP weight in winner: {w[names.index('mlp')]:.4f}   ET weight: {w[names.index('et')]:.4f}")

if s > V2 + 1e-5:
    pred = (np.tensordot(w,T,1)*adj).argmax(1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [classes[i] for i in pred]})
    out = COMP/"submissions"/"s6e7_blend_v3b.csv"; sub.to_csv(out, index=False)
    log(f"IMPROVED over v2 -> {out}  (pred dist={np.bincount(pred)/len(pred)})")
else:
    log(f"no meaningful improvement over v2 (Δ={s-V2:+.6f}) -> ceiling confirmed even with fixed MLP")
json.dump(dict(v3b_balacc=s, v2=V2, delta=s-V2, weights=dict(zip(names,w.tolist())), adjust=adj.tolist(),
               mlp_solo=balacc(y,(oof*fixed).argmax(1)),
               solo={n: balacc(y,(np.load(COMP/f'data/oof_{n}.npz')['oof']*fixed).argmax(1)) for n in names}),
          open(COMP/"scripts/v3b_results.json","w"), indent=2)
log("V3b DONE")
