"""s6e7 v3 — add STRUCTURALLY DIVERSE non-GBDT learners to the tapped-out GBDT pool.

Rationale (experience.md): at the balanced-accuracy ceiling the only untapped lever is
ensemble *diversity*. The v2 blend weights all sit on GBDT variants (cat/xgb/2x lgbm), so
a boosting-search can't add diversity. We add:
  * MLP  — GPU neural net, class-weighted cross-entropy (maximally different from trees)
  * ExtraTrees — bagging learner (different bias from boosting)
Then re-blend the full 7-member pool. If diversity doesn't help, the weight search zeros
the new members out -> a clean 'confirmed ceiling'. Only writes a submission if CV > v2.

Folds are the IDENTICAL StratifiedKFold(5, shuffle=True, random_state=42) used by the GBDT
OOF, so the stacked OOF rows align for a valid blend.
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import torch, torch.nn as nn

COMP = Path(__file__).resolve().parent.parent; ROOT = COMP.parent.parent
_spec = importlib.util.spec_from_file_location("experiment_log", ROOT/".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(experiment_log)

SEED, NF, NTH, V2 = 42, 5, 8, 0.9499789257329395
NUM = ["sleep_duration","heart_rate","bmi","calorie_expenditure","step_count","exercise_duration","water_intake"]
CAT = ["diet_type","stress_level","sleep_quality","physical_activity_level","smoking_alcohol","gender"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------- data ----------
tr = pd.read_csv(COMP/"data/train.csv"); te = pd.read_csv(COMP/"data/test.csv")
classes = sorted(tr["health_condition"].unique()); c2i = {c:i for i,c in enumerate(classes)}
y = tr["health_condition"].map(c2i).to_numpy()
for c in CAT:
    tr[c] = tr[c].fillna("missing").astype(str); te[c] = te[c].fillna("missing").astype(str)
folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(tr, y))
cw = (len(y)/(3*np.bincount(y))).astype(np.float32)     # class weights for macro-recall
masks = [y == c for c in range(3)]                       # for fast balanced accuracy

# feature matrix: standardized numerics + one-hot categoricals (global categories, per-fold scaling)
cat_dummies = pd.get_dummies(pd.concat([tr[CAT], te[CAT]], ignore_index=True), columns=CAT)
Xtr_cat = cat_dummies.iloc[:len(tr)].to_numpy(np.float32); Xte_cat = cat_dummies.iloc[len(tr):].to_numpy(np.float32)
Xtr_num = tr[NUM].to_numpy(np.float32); Xte_num = te[NUM].to_numpy(np.float32)
log(f"train={tr.shape} classes={classes} cw={cw.round(2)} feat_dim={Xtr_num.shape[1]+Xtr_cat.shape[1]} dev={DEV}")

def balacc(yv, pred):    # macro recall
    return float(np.mean([(pred[m]==c).mean() for c,m in enumerate([yv==0,yv==1,yv==2])]))

# ---------- MLP ----------
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
    Xte_c = torch.tensor(Xte_cat, device=DEV)
    for k,(tri,vai) in enumerate(folds):
        mu = Xtr_num[tri].mean(0); sd = Xtr_num[tri].std(0)+1e-6
        Xtr = np.concatenate([(Xtr_num-mu)/sd, Xtr_cat], 1).astype(np.float32)
        Xt  = np.concatenate([(Xte_num-mu)/sd, Xte_cat], 1).astype(np.float32)
        Xtri = torch.tensor(Xtr[tri], device=DEV); ytri = torch.tensor(y[tri], device=DEV)
        Xvai = torch.tensor(Xtr[vai], device=DEV)
        m = MLP(Xtr.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-5)
        n = len(tri); bs = 8192; best_ba, best_state, patience = -1, None, 0
        for ep in range(80):
            m.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                loss = wce(m(Xtri[idx]), ytri[idx]); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vp = m(Xvai).softmax(1).cpu().numpy()
            ba = balacc(y[vai], (vp*np.array([1,1.4,1.3])).argmax(1))
            if ba > best_ba: best_ba, best_state, patience = ba, {kk:v.cpu().clone() for kk,v in m.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 12: break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            oof[vai] = m(Xvai).softmax(1).cpu().numpy()
            test += m(Xte_c if False else torch.tensor(Xt, device=DEV)).softmax(1).cpu().numpy()/NF
        log(f"  MLP fold{k}: adj-balacc={best_ba:.4f} (stopped ep~{ep})")
    return oof, test

# ---------- ExtraTrees ----------
def run_et():
    oof = np.zeros((len(tr),3), np.float32); test = np.zeros((len(te),3), np.float32)
    Xtr = np.concatenate([Xtr_num, Xtr_cat], 1); Xte = np.concatenate([Xte_num, Xte_cat], 1)
    for k,(tri,vai) in enumerate(folds):
        m = ExtraTreesClassifier(n_estimators=400, max_features=0.6, min_samples_leaf=20,
                                 class_weight="balanced", n_jobs=NTH, random_state=SEED)
        m.fit(Xtr[tri], y[tri])
        oof[vai] = m.predict_proba(Xtr[vai]); test += m.predict_proba(Xte)/NF
        log(f"  ET fold{k}: adj-balacc={balacc(y[vai],(oof[vai]*np.array([1,1.4,1.3])).argmax(1)):.4f}")
    return oof, test

for name, fn in [("mlp", run_mlp), ("et", run_et)]:
    p = COMP/f"data/oof_{name}.npz"
    if p.exists():
        z = np.load(p); oof, test = z["oof"], z["test"]; log(f"=== {name} (cached) ===")
    else:
        log(f"=== {name} ==="); oof, test = fn(); np.savez_compressed(p, oof=oof, test=test)
    log(f"{name}: solo adj-balacc={balacc(y,(oof*np.array([1,1.4,1.3])).argmax(1)):.5f}")

# ---------- re-blend full 7-member pool ----------
names = ["lgbm","cat","xgb","lgbm_tuned","lgbm_seed2024","mlp","et"]
O = np.stack([np.load(COMP/f"data/oof_{n}.npz")["oof"] for n in names]).astype(np.float64)
T = np.stack([np.load(COMP/f"data/oof_{n}.npz")["test"] for n in names]).astype(np.float64)
M = len(names); rng = np.random.default_rng(SEED)
A = np.linspace(0.7, 3, 16)
fixed = np.array([1,1.4,1.3])

def tune_adjust(P):
    best_adj, best = fixed, balacc(y,(P*fixed).argmax(1))
    for a in A:
        for b in A:
            adj = np.array([1,a,b]); s = balacc(y,(P*adj).argmax(1))
            if s > best: best, best_adj = s, adj
    return best_adj, best

# stage 1: random Dirichlet weights, rank by fixed-adjust balacc
cand = rng.dirichlet(np.ones(M), size=25000)
cand = np.vstack([cand, np.eye(M)])                      # include solos
scores = np.empty(len(cand))
for i,w in enumerate(cand):
    bl = np.tensordot(w, O, 1); scores[i] = balacc(y,(bl*fixed).argmax(1))
top = cand[np.argsort(scores)[::-1][:40]]
# stage 2: full adjust-tune on top-40
best = (None, None, -1)
for w in top:
    bl = np.tensordot(w, O, 1); adj, s = tune_adjust(bl)
    if s > best[2]: best = (w, adj, s)
w, adj, s = best
log(f"V3 BLEND balacc={s:.5f} (v2={V2:.5f})  adj={adj.round(2)}")
log("  weights: " + ", ".join(f"{n}={wi:.3f}" for n,wi in zip(names,w) if wi > 1e-3))

if s > V2 + 1e-6:
    pred = (np.tensordot(w, T, 1)*adj).argmax(1)
    sub = pd.DataFrame({"id": te["id"], "health_condition": [classes[i] for i in pred]})
    out = COMP/"submissions"/"s6e7_blend_v3.csv"; sub.to_csv(out, index=False)
    experiment_log.log_experiment_v2(str(COMP), model=f"v3 blend +MLP+ET {dict(zip(names,w.round(3)))}",
        metric="balanced_accuracy", direction="maximize", score=round(s,6),
        cv=dict(scheme="StratifiedKFold", n_splits=NF, seed=SEED), submission=out.name,
        notes=f"diverse non-GBDT members added; adj={adj.tolist()}")
    log(f"IMPROVED over v2 -> {out}")
else:
    log(f"no improvement over v2 (Δ={s-V2:+.6f}); diversity did not help -> confirmed ceiling")
json.dump(dict(v3_balacc=s, v2=V2, weights=dict(zip(names,w.tolist())), adjust=adj.tolist(),
               solo={n: balacc(y,(np.load(COMP/f'data/oof_{n}.npz')['oof']*fixed).argmax(1)) for n in names}),
          open(COMP/"scripts/v3_results.json","w"), indent=2)
log("V3 DONE")
