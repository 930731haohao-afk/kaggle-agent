"""s6e7 v2 — Optuna-tune LGBM (fold-0 proxy, objective = balanced accuracy after prior-adjust),
then add tuned + seed-bag to the v1 pool and re-blend. experience.md recipe: tune strongest ->
add to pool (don't replace) -> seed bag.
"""
import importlib.util, json, time
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, optuna
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.model_selection import StratifiedKFold

COMP = Path(__file__).resolve().parent.parent; ROOT = COMP.parent.parent
_spec = importlib.util.spec_from_file_location("experiment_log", ROOT/".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(experiment_log)
SEED, NF, NTH = 42, 5, 8
NUM = ["sleep_duration","heart_rate","bmi","calorie_expenditure","step_count","exercise_duration","water_intake"]
CAT = ["diet_type","stress_level","sleep_quality","physical_activity_level","smoking_alcohol","gender"]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def adj_ba(y, P):
    best = BA(y, P.argmax(1)); grid=[0.7,0.85,1,1.2,1.5,1.7,2,2.5,3]
    for a in grid:
        for b in grid:
            s=BA(y,(P*np.array([1,a,b])).argmax(1)); best=max(best,s)
    return best

tr=pd.read_csv(COMP/"data/train.csv")
classes=sorted(tr["health_condition"].unique()); y=tr["health_condition"].map({c:i for i,c in enumerate(classes)}).to_numpy()
for c in CAT: tr[c]=tr[c].fillna("missing").astype("category")
X=tr[NUM+CAT]; cw={i:v for i,v in enumerate(len(y)/(3*np.bincount(y)))}
folds=list(StratifiedKFold(NF,shuffle=True,random_state=SEED).split(tr,y))
tri0,vai0=folds[0]

def objective(t):
    p=dict(objective="multiclass",num_class=3,class_weight=cw,n_jobs=NTH,seed=SEED,deterministic=True,
           force_row_wise=True,verbose=-1,n_estimators=1500,
           learning_rate=t.suggest_float("lr",0.015,0.06,log=True),
           num_leaves=t.suggest_int("num_leaves",31,255),
           min_child_samples=t.suggest_int("mcs",30,200),
           feature_fraction=t.suggest_float("ff",0.6,1.0),
           bagging_fraction=t.suggest_float("bf",0.6,1.0),bagging_freq=1,
           reg_lambda=t.suggest_float("l2",0.5,8.0,log=True),
           reg_alpha=t.suggest_float("l1",1e-3,5.0,log=True))
    m=lgb.LGBMClassifier(**p)
    m.fit(X.iloc[tri0],y[tri0],categorical_feature=CAT,eval_set=[(X.iloc[vai0],y[vai0])],
          callbacks=[lgb.early_stopping(60,verbose=False)])
    return adj_ba(y[vai0], m.predict_proba(X.iloc[vai0]))

log("Optuna fold-0 proxy (balanced accuracy)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective,n_trials=40,timeout=1500)
bp=study.best_params
log(f"best fold-0 balacc={study.best_value:.5f} params={bp}")

def full_oof(params,seed):
    oof=np.zeros((len(tr),3)); test_dummy=None
    te=pd.read_csv(COMP/"data/test.csv")
    for c in CAT: te[c]=pd.Categorical(te[c].fillna("missing"),categories=tr[c].cat.categories)
    Xt=te[NUM+CAT]; test=np.zeros((len(te),3))
    for k,(tri,vai) in enumerate(folds):
        m=lgb.LGBMClassifier(objective="multiclass",num_class=3,class_weight=cw,n_jobs=NTH,seed=seed,
            deterministic=True,force_row_wise=True,verbose=-1,n_estimators=1500,
            learning_rate=bp["lr"],num_leaves=bp["num_leaves"],min_child_samples=bp["mcs"],
            feature_fraction=bp["ff"],bagging_fraction=bp["bf"],bagging_freq=1,reg_lambda=bp["l2"],reg_alpha=bp["l1"])
        m.fit(X.iloc[tri],y[tri],categorical_feature=CAT,eval_set=[(X.iloc[vai],y[vai])],
              callbacks=[lgb.early_stopping(80,verbose=False)])
        oof[vai]=m.predict_proba(X.iloc[vai]); test+=m.predict_proba(Xt)/NF
    return oof,test,te

log("full 5-fold: tuned LGBM (seed 42) + seed-bag (seed 2024)...")
oof_t,test_t,te=full_oof(bp,42); log(f"lgbm_tuned OOF balacc={adj_ba(y,oof_t):.5f}")
oof_s,test_s,_=full_oof(bp,2024); log(f"lgbm_tuned_seed2024 OOF balacc={adj_ba(y,oof_s):.5f}")
np.savez_compressed(COMP/"data/oof_lgbm_tuned.npz",oof=oof_t,test=test_t)
np.savez_compressed(COMP/"data/oof_lgbm_seed2024.npz",oof=oof_s,test=test_s)

# re-blend the full pool
pool={n:(np.load(COMP/f"data/oof_{n}.npz")["oof"],np.load(COMP/f"data/oof_{n}.npz")["test"])
      for n in ["lgbm","cat","xgb","lgbm_tuned","lgbm_seed2024"]}
names=list(pool); O=np.stack([pool[n][0] for n in names]); T=np.stack([pool[n][1] for n in names])
def dec(P):
    best=(np.ones(3),BA(y,P.argmax(1)))
    for a in np.linspace(0.7,3,20):
        for b in np.linspace(0.7,3,20):
            s=BA(y,(P*np.array([1,a,b])).argmax(1))
            if s>best[1]: best=(np.array([1,a,b]),s)
    return best
best=(None,None,-1)
for w in product(np.linspace(0,1,5),repeat=len(names)):
    if abs(sum(w)-1)>1e-6: continue
    bl=np.tensordot(w,O,axes=1); adj,s=dec(bl)
    if s>best[2]: best=(np.array(w),adj,s)
w,adj,s=best
log(f"V2 BLEND balacc={s:.5f} weights={dict(zip(names,w.round(3)))} adj={adj.round(2)} (v1=0.94994)")
if s>0.94994:
    pred=(np.tensordot(w,T,axes=1)*adj).argmax(1)
    sub=pd.DataFrame({"id":te["id"],"health_condition":[classes[i] for i in pred]})
    out=COMP/"submissions"/"s6e7_blend_v2.csv"; sub.to_csv(out,index=False)
    experiment_log.log_experiment_v2(str(COMP),model=f"v2 blend +tuned+seedbag {dict(zip(names,w.round(3)))}",
        metric="balanced_accuracy",direction="maximize",score=round(s,6),
        cv=dict(scheme="StratifiedKFold",n_splits=NF,seed=SEED),submission=out.name,
        notes=f"Optuna fold-0 tuned LGBM + seed-bag added to pool; adj={adj.tolist()}")
    log(f"IMPROVED -> submission written: {out}")
else:
    log("no improvement over v1; not submitting")
json.dump(dict(v2_balacc=s,weights=dict(zip(names,w.tolist())),v1=0.94994,best_params=bp),
          open(COMP/"scripts/v2_results.json","w"),indent=2)
log("V2 DONE")
