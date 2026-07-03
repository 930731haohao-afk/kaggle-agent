"""Self-improvement Round 4: extend seed bagging to the dominant pool member.

Round 3 (exp #7) improved OOF 12.07347 -> 12.07143 with weights
{CAT 0.773, LGB_tuned_seed2 0.227}. CatBoost carries 77% of the blend, so a
seed-bagged CatBoost (same hyperparams, new random_seed) is the highest-EV cheap
addition (validated recipe: seed bagging is the cheapest residual gain; CAT fold
fit is ~0.5s). Also add a 3rd-seed tuned LGB since the 2nd seed earned 22.7%.

Single conceptual change: expand the pool with 2 more seed-bag members
(CAT_seed2, LGB_tuned_seed3), re-run OOF weight search. Same folds/seed/features.
Reuses Round 2/3 OOF/pred from _round23_pool.npz (deterministic, verified
reproducible in exp #4).
"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e9"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Strength", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
Xtr_full = build_features(train)
Xte_full = build_features(test)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.float64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)

ybin = pd.qcut(y, 10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, ybin))


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# --- load Round 2/3 pool (5 members) ---
z = np.load(f"{COMP}/scripts/_round23_pool.npz", allow_pickle=True)
names5 = [str(n) for n in z["names5"]]
oofs5 = z["oof5"]          # (n_train, 5)
preds5 = z["pred5"]        # (n_test, 5)
print(f"Loaded pool: {names5}  round3 score={float(z['s5']):.5f}")

# Optuna-tuned LGB params from Round 2 (exp #6 notes)
TUNED = dict(n_estimators=3000, learning_rate=0.019795655587585677, num_leaves=11,
             max_depth=3, min_child_samples=31, subsample=0.6018504151952752,
             subsample_freq=1, colsample_bytree=0.5006835428888052,
             reg_alpha=0.1120861378640153, reg_lambda=0.3020804107290036)


def fit_lgb(params, seed):
    import lightgbm as lgb
    p = dict(params)
    p.update(objective="regression", metric="rmse", random_state=seed, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def fit_cat(seed):
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=4000,
                              learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                              random_seed=seed, thread_count=-1, verbose=False,
                              allow_writing_files=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


t0 = time.time(); oof_cat2, pred_cat2 = fit_cat(SEED + 1000); t_cat2 = time.time() - t0
print(f"CAT_seed2 (seed={SEED+1000}) OOF RMSE = {rmse(y, oof_cat2):.5f}  [{t_cat2:.1f}s]")
t0 = time.time(); oof_lgbt3, pred_lgbt3 = fit_lgb(TUNED, SEED + 2000); t_lgbt3 = time.time() - t0
print(f"LGB_tuned_seed3 (seed={SEED+2000}) OOF RMSE = {rmse(y, oof_lgbt3):.5f}  [{t_lgbt3:.1f}s]")

names7 = names5 + ["CAT_seed2", "LGB_tuned_seed3"]
oofs7 = np.concatenate([oofs5, oof_cat2[:, None], oof_lgbt3[:, None]], axis=1)
preds7 = np.concatenate([preds5, pred_cat2[:, None], pred_lgbt3[:, None]], axis=1)


def weight_search(oofs, k):
    rng = np.random.RandomState(SEED)
    best_w = np.ones(k) / k
    best_s = rmse(y, oofs @ best_w)
    for _ in range(8000):
        w_try = rng.dirichlet(np.ones(k) * 2.0)
        s = rmse(y, oofs @ w_try)
        if s < best_s:
            best_s, best_w = s, w_try
    for _ in range(50):
        improved = False
        for i in range(k):
            for delta in (0.01, -0.01, 0.02, -0.02, 0.05, -0.05):
                w_try = np.array(best_w, dtype=float)
                w_try[i] = max(0.0, w_try[i] + delta)
                if w_try.sum() <= 0:
                    continue
                w_try = w_try / w_try.sum()
                s = rmse(y, oofs @ w_try)
                if s < best_s - 1e-9:
                    best_s, best_w = s, w_try
                    improved = True
        if not improved:
            break
    return best_w, best_s


w7, s7 = weight_search(oofs7, len(names7))
print(f"ROUND 4 (7-way) weights={dict(zip(names7, np.round(w7, 3)))} OOF RMSE={s7:.5f}")
print(f"vs Round 3 12.07143, vs previous best 12.07347")

# refine Round-3 5-way too with the finer coordinate steps for a fair comparison
w5r, s5r = weight_search(oofs5, len(names5))
print(f"(re-refined 5-way: {s5r:.5f})")

# pick whichever pool wins
if s7 <= s5r:
    final_names, final_w, final_s, final_oofs, final_preds = names7, w7, s7, oofs7, preds7
else:
    final_names, final_w, final_s, final_oofs, final_preds = names5, w5r, s5r, oofs5, preds5
final_pred = np.clip(final_preds @ final_w, 0, None)
final_oof = final_oofs @ final_w
print(f"FINAL: {len(final_names)}-way OOF RMSE = {final_s:.5f}")

# --- submission if improved over 12.073474 ---
sub_path = None
if final_s < 12.073474:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({ID: test[ID], TARGET: final_pred})
    sub_path = f"{SUB}/sub_blend_{final_s:.5f}_{stamp}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}  shape={sub.shape}")

base_models = [
    dict(model="CAT_seed2", rmse=round(rmse(y, oof_cat2), 5), time_s=round(t_cat2, 1),
         note=f"seed bag of orig CatBoost(depth6,l2=6) with random_seed={SEED+1000}"),
    dict(model="LGB_tuned_seed3", rmse=round(rmse(y, oof_lgbt3), 5), time_s=round(t_lgbt3, 1),
         note=f"3rd seed ({SEED+2000}) of Optuna-tuned LGB"),
]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model=f"{len(final_names)}-way blend: Round-3 pool + CAT_seed2 + LGB_tuned_seed3 (seed-bag expansion)",
    metric="rmse", direction="minimize", score=final_s,
    cv=dict(strategy=f"{N_SPLITS}fold_stratified_strength_decile", seed=SEED),
    features=FEATS, base_models=base_models,
    ensemble=dict(weights=dict(zip(final_names, [round(float(x), 3) for x in final_w])),
                  method="oof_weight_search(dirichlet8000+coord_descent_fine)"),
    submission=os.path.basename(sub_path) if sub_path else None,
    notes=(f"Round 4 self-improvement: seed-bag the dominant CatBoost member "
           f"(77% weight in Round 3) + 3rd tuned-LGB seed; 7-way pool weight search "
           f"{s7:.5f} vs re-refined 5-way {s5r:.5f}; kept the better. Result {final_s:.5f} "
           f"vs Round 3 12.07143 vs previous best 12.07347 "
           f"({'IMPROVED' if final_s < 12.071434 else 'no further improvement over Round 3'})."),
)
print(f"Logged experiment #{exp_id}")

np.savez(f"{COMP}/scripts/_round4_final.npz",
         oof=final_oof, pred=final_pred, score=final_s,
         names=np.array(final_names), w=np.array(final_w))
