"""Train LGB/XGB/CatBoost with KFold OOF on log1p(cost), blend, and write submission.

Metric: RMSLE (minimize). We train on log1p(cost) with a plain RMSE objective --
RMSLE(y, pred) = sqrt(mean((log1p(y) - log1p(pred))^2)), so this directly optimizes
the competition metric. Predictions are clipped to >=0 before the inverse expm1
transform (safety net; in practice predicted log-values stay well inside a positive
range given cost in [50.79, 149.75]).

CV: 5-fold KFold (shuffle) -- EDA found no temporal/group structure requiring
GroupKFold; store profiles repeat thousands of times so random folds are fine.

Usage: uv run python3 competitions/playground-series-s3e11/scripts/train.py {base|engineered}
  base       -- only the 15 raw columns (methodology check against generic baseline)
  engineered -- raw columns + amenity_count/ratios + fold-safe store_combo target encoding
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, base_feature_columns  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e11"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "cost", "id"
N_SPLITS, SEED = 5, 42
MODE = sys.argv[1] if len(sys.argv) > 1 else "engineered"
assert MODE in ("base", "engineered")

os.makedirs(SUB, exist_ok=True)

train_raw = pd.read_csv(f"{DATA}/train.csv")
test_raw = pd.read_csv(f"{DATA}/test.csv")

Xtr_full = build_features(train_raw)
Xte_full = build_features(test_raw)

if MODE == "base":
    FEATS = [c for c in base_feature_columns() if c in
             ["store_sales(in millions)", "unit_sales(in millions)", "total_children",
              "num_children_at_home", "avg_cars_at home(approx).1", "gross_weight",
              "recyclable_package", "low_fat", "units_per_case", "store_sqft",
              "coffee_bar", "video_store", "salad_bar", "prepared_food", "florist"]]
else:
    FEATS = base_feature_columns()  # includes amenity_count/ratios; store_te appended per-fold below

y_log = np.log1p(train_raw[TARGET].to_numpy(np.float64))
print(f"mode={MODE}  n_features(before store_te)={len(FEATS)}  train={Xtr_full.shape}  test={Xte_full.shape}")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(kf.split(Xtr_full))

# --- fold-safe target encoding of store_combo (engineered mode only) ---
te_train_col = np.zeros(len(Xtr_full))
te_test_col = np.zeros(len(Xte_full))
if MODE == "engineered":
    global_mean = y_log.mean()
    te_test_accum = np.zeros(len(Xte_full))
    for tr, va in folds:
        m = pd.Series(y_log[tr], index=Xtr_full.iloc[tr].index).groupby(Xtr_full.iloc[tr]["store_combo"]).mean()
        te_train_col[va] = Xtr_full.iloc[va]["store_combo"].map(m).fillna(global_mean).to_numpy()
        # test encoding: average the 5 fold-specific train->combo means mapped onto test,
        # matching how we average model predictions across folds
        te_test_accum += Xte_full["store_combo"].map(m).fillna(global_mean).to_numpy() / N_SPLITS
    te_test_col = te_test_accum
    FEATS = FEATS + ["store_te"]

X = Xtr_full[[c for c in FEATS if c != "store_te"]].to_numpy(np.float32)
Xtest = Xte_full[[c for c in FEATS if c != "store_te"]].to_numpy(np.float32)
if MODE == "engineered":
    X = np.hstack([X, te_train_col.reshape(-1, 1).astype(np.float32)])
    Xtest = np.hstack([Xtest, te_test_col.reshape(-1, 1).astype(np.float32)])

y = y_log
print(f"final n_features={len(FEATS)}: {FEATS}")


def rmsle_from_log(y_true_log, y_pred_log):
    pred_cost = np.clip(np.expm1(y_pred_log), 0, None)
    true_cost = np.expm1(y_true_log)
    return np.sqrt(mean_squared_error(np.log1p(true_cost), np.log1p(pred_cost)))


def run_lgb():
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=2000,
                  learning_rate=0.04, num_leaves=63, min_child_samples=60,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [LGB] fold{f} rmsle={rmsle_from_log(y[va], oof[va]):.5f} best_iter={m.best_iteration_}")
    return oof, pred


def run_xgb():
    import xgboost as xgb
    params = dict(objective="reg:squarederror", n_estimators=2000, learning_rate=0.04,
                  max_depth=7, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                  eval_metric="rmse", early_stopping_rounds=100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [XGB] fold{f} rmsle={rmsle_from_log(y[va], oof[va]):.5f} best_iter={m.best_iteration}")
    return oof, pred


def run_cat():
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", iterations=2500,
                               learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
                               random_seed=SEED, thread_count=-1, verbose=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict(X[va]); pred += m.predict(Xtest) / N_SPLITS
        print(f"  [CAT] fold{f} rmsle={rmsle_from_log(y[va], oof[va]):.5f} best_iter={m.get_best_iteration()}")
    return oof, pred


results = {}
for name, fn in [("LGB", run_lgb), ("XGB", run_xgb), ("CAT", run_cat)]:
    t0 = time.time()
    print(f"\n=== {name} ({MODE}) ===")
    oof, pred = fn()
    dt = time.time() - t0
    score = rmsle_from_log(y, oof)
    print(f"{name} OOF RMSLE = {score:.5f}  [{dt:.0f}s]")
    results[name] = dict(oof=oof, pred=pred, score=score, time=dt)

# --- weight search (coarse grid over simplex) ---
names = list(results)
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
best_w, best_s = None, 1e9
for w0 in np.arange(0, 1.01, 0.1):
    for w1 in np.arange(0, 1.01 - w0, 0.1):
        w2 = 1 - w0 - w1
        if w2 < -1e-9:
            continue
        blended_log = oofs @ np.array([w0, w1, w2])
        s = rmsle_from_log(y, blended_log)
        if s < best_s:
            best_s, best_w = s, (w0, w1, w2)
w = np.array(best_w)
print(f"\nBEST weighted blend {dict(zip(names, np.round(best_w, 2)))} OOF RMSLE = {best_s:.5f}")

final_pred_log = np.stack([results[n]["pred"] for n in names], axis=1) @ w
final_pred_cost = np.clip(np.expm1(final_pred_log), 0, None)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({ID: test_raw[ID], TARGET: final_pred_cost})
sub_path = f"{SUB}/sub_{MODE}_{best_s:.5f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub[TARGET].describe())

comp_dir = COMP
base_models = [dict(model=n, oof_rmsle=round(results[n]["score"], 5), time_s=round(results[n]["time"], 1))
               for n in names]
experiment_log.log_experiment_v2(
    comp_dir,
    model=f"{MODE} LGB+XGB+CAT blend (log1p target, KFold5)",
    metric="rmsle",
    direction="minimize",
    score=float(best_s),
    cv=dict(strategy="5fold_kfold_shuffle", seed=SEED),
    features=FEATS,
    base_models=base_models,
    ensemble=dict(weights=dict(zip(names, [round(float(x), 2) for x in w]))),
    submission=os.path.basename(sub_path),
    notes=(f"mode={MODE}; trained on log1p(cost) with RMSE objective, expm1+clip>=0 at predict "
           f"time; " + ("store_combo (store_sqft+5 amenity flags) K-fold target-encoded per fold, "
           "test encoding averaged across folds." if MODE == "engineered" else
           "raw 15 features only, no target encoding (methodology-only run).")),
)
print("Logged experiment.")
