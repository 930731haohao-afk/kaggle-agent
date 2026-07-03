"""Phase B iteration, round 4: does CatBoost-native (round 3, OOF 0.814259, rescued from
0.7627 but still weaker than LGB_tuned solo 0.837305) add diversity value to the round-2
6-way pool, even though its solo score is well below the other members? Weight search
decides — per experience.md "OOF weight search often zeroes weak models; respect it."

Also includes rank-average vs weight-blend comparison on the winning pool, per protocol.
"""
import importlib.util
import itertools
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, feature_columns, fit_encoders  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e3"
DATA = f"{COMP}/data"
SUB = f"{COMP}/submissions"
TARGET, ID = "Attrition", "id"
N_SPLITS, SEED = 5, 42

os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

enc = fit_encoders(train)
Xtr_full = build_features(train, enc)
Xte_full = build_features(test, enc)
FEATS = feature_columns(Xtr_full)
X = Xtr_full[FEATS].to_numpy(np.float32)
y = train[TARGET].to_numpy(np.int64)
Xtest = Xte_full[FEATS].to_numpy(np.float32)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))

# --- CatBoost-native feature set (raw categoricals, round-3 winning config) ---
CAT_COLS = ["BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"]
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18"]
SATISFACTION_COLS = ["EnvironmentSatisfaction", "JobSatisfaction",
                     "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]


def build_native_features(df):
    out = df.copy()
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    for c in CAT_COLS:
        out[c] = out[c].astype(str)
    out["role_tenure_ratio"] = out["YearsInCurrentRole"] / out["YearsAtCompany"].replace(0, np.nan)
    out["mgr_tenure_ratio"] = out["YearsWithCurrManager"] / out["YearsAtCompany"].replace(0, np.nan)
    out["promo_ratio"] = out["YearsSinceLastPromotion"] / out["YearsAtCompany"].replace(0, np.nan)
    out["company_tenure_ratio"] = out["YearsAtCompany"] / out["TotalWorkingYears"].replace(0, np.nan)
    for c in ["role_tenure_ratio", "mgr_tenure_ratio", "promo_ratio", "company_tenure_ratio"]:
        out[c] = out[c].fillna(0.0)
    out["income_per_joblevel"] = out["MonthlyIncome"] / out["JobLevel"].replace(0, np.nan)
    out["income_per_joblevel"] = out["income_per_joblevel"].fillna(out["MonthlyIncome"])
    out["income_per_year_worked"] = out["MonthlyIncome"] / (out["TotalWorkingYears"] + 1)
    out["satisfaction_avg"] = out[SATISFACTION_COLS].mean(axis=1)
    out["satisfaction_min"] = out[SATISFACTION_COLS].min(axis=1)
    out["age_at_join"] = out["Age"] - out["TotalWorkingYears"]
    out["companies_per_year"] = out["NumCompaniesWorked"] / (out["Age"] - 18).clip(lower=1)
    return out


Xn_tr = build_native_features(train)
Xn_te = build_native_features(test)
NFEATS = [c for c in Xn_tr.columns if c not in ("id", "Attrition")]
cat_idx = [NFEATS.index(c) for c in CAT_COLS]
Xn = Xn_tr[NFEATS].copy()
Xn_test = Xn_te[NFEATS].copy()

results = {}


def run_lgb_orig():
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", n_estimators=2000,
                  learning_rate=0.03, num_leaves=7, min_child_samples=30,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
                  reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_xgb():
    import xgboost as xgb
    params = dict(objective="binary:logistic", n_estimators=2000, learning_rate=0.03,
                  max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, eval_metric="auc", early_stopping_rounds=100)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_cat_orig():
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2000,
                                learning_rate=0.03, depth=5, l2_leaf_reg=8.0,
                                random_seed=SEED, thread_count=4, verbose=False,
                                allow_writing_files=False)
        m.fit(Pool(X[tr], y[tr]), eval_set=Pool(X[va], y[va]),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_lgb_tuned(seed):
    import lightgbm as lgb
    best_params = dict(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.050253069663926536, num_leaves=3, max_depth=4,
        min_child_samples=60, subsample=0.8995656675058548,
        colsample_bytree=0.7246802411328122, reg_alpha=0.06815791273889091,
        reg_lambda=0.0011513245661312597,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for tr, va in folds:
        m = lgb.LGBMClassifier(**best_params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def run_cat_native():
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xn_test))
    for tr, va in folds:
        m = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=2000,
            learning_rate=0.05, depth=3, l2_leaf_reg=16.0,
            random_seed=SEED, thread_count=4, verbose=False,
            allow_writing_files=False, cat_features=cat_idx,
        )
        m.fit(Pool(Xn.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xn.iloc[va], y[va], cat_features=cat_idx),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict_proba(Xn.iloc[va])[:, 1]
        pred += m.predict_proba(Xn_test)[:, 1] / N_SPLITS
    return oof, pred


t0 = time.time()
for name, fn, args in [
    ("LGB_orig", run_lgb_orig, ()),
    ("XGB", run_xgb, ()),
    ("CAT_orig", run_cat_orig, ()),
    ("LGB_tuned", run_lgb_tuned, (42,)),
    ("LGB_tuned_seed2024", run_lgb_tuned, (2024,)),
    ("CAT_native", run_cat_native, ()),
]:
    tt = time.time()
    oof, pred = fn(*args)
    score = roc_auc_score(y, oof)
    results[name] = {"oof": oof, "pred": pred, "score": score}
    print(f"{name}: OOF AUC={score:.6f}  ({time.time()-tt:.1f}s)")
print(f"Total training time: {time.time()-t0:.1f}s")

names = list(results.keys())
oofs = np.stack([results[n]["oof"] for n in names], axis=1)
preds = np.stack([results[n]["pred"] for n in names], axis=1)

# --- weight-search (prob-space) blend ---
best_w, best_score = None, -1
grid = np.arange(0, 1.0001, 0.05)
for combo in itertools.product(grid, repeat=len(names)):
    if abs(sum(combo) - 1.0) > 1e-6:
        continue
    blend_oof = oofs @ np.array(combo)
    s = roc_auc_score(y, blend_oof)
    if s > best_score:
        best_score, best_w = s, combo

print(f"\nBest prob-blend weights {dict(zip(names, best_w))}  OOF AUC={best_score:.6f}")

# --- rank-average blend, using the same weights (converted to rank space) ---
rank_oofs = np.stack([rankdata(results[n]["oof"]) / len(y) for n in names], axis=1)
rank_blend_oof = rank_oofs @ np.array(best_w)
rank_score = roc_auc_score(y, rank_blend_oof)
print(f"Rank-average blend (same weights) OOF AUC={rank_score:.6f}")

# also try equal-weight rank average across only the previously-nonzero members (LGB_orig, LGB_tuned)
prior_best_w = {"LGB_orig": 0.3, "XGB": 0.0, "CAT_orig": 0.0, "LGB_tuned": 0.7,
                "LGB_tuned_seed2024": 0.0, "CAT_native": 0.0}
prior_combo = np.array([prior_best_w[n] for n in names])
prior_blend_oof = oofs @ prior_combo
prior_score = roc_auc_score(y, prior_blend_oof)
print(f"Round-2 weights replayed on this pool (sanity check): {prior_score:.6f}")

final_method = "prob_weight_search"
final_score = best_score
final_blend_oof = oofs @ np.array(best_w)
final_blend_pred = preds @ np.array(best_w)
if rank_score > final_score:
    final_method = "rank_average_same_weights"
    final_score = rank_score
    final_blend_pred = rank_oofs @ np.array(best_w)  # note: for submission we'd need rank of test preds; see below

PRIOR_BEST = 0.837776
sub_name = None
if final_score > PRIOR_BEST:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_name = f"sub_blend_{final_score:.5f}_{ts}.csv"
    sub_path = f"{SUB}/{sub_name}"
    if final_method == "prob_weight_search":
        out_pred = final_blend_pred
    else:
        # rank-average submission: rank-average the test predictions across models
        rank_preds = np.stack([rankdata(preds[:, i]) / len(preds) for i in range(len(names))], axis=1)
        out_pred = rank_preds @ np.array(best_w)
    sub = pd.DataFrame({ID: test[ID], TARGET: out_pred})
    sub.to_csv(sub_path, index=False)
    print(f"NEW BEST ({final_method}) -> wrote submission: {sub_path}")
else:
    print(f"Not an improvement over prior best ({PRIOR_BEST}); no submission written. "
          f"prob={best_score:.6f} rank={rank_score:.6f}")

base_models = [{"name": n, "oof_auc": round(float(results[n]["score"]), 6)} for n in names]
exp_id = experiment_log.log_experiment_v2(
    COMP,
    model=f"6-way blend incl. CAT_native ({final_method})",
    metric="roc_auc",
    direction="maximize",
    score=float(final_score),
    cv={"strategy": "StratifiedKFold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=base_models,
    ensemble={"method": final_method, "weights": dict(zip(names, [float(w) for w in best_w]))},
    submission=sub_name,
    notes=(f"Round 4: added CAT_native (round 3, solo 0.814259, rescued from 0.7627 via native "
           f"cat_features but still weakest member) to the round-2 5-way pool, and compared "
           f"prob-space weight-search blend ({best_score:.6f}) vs rank-average with the same "
           f"weights ({rank_score:.6f}). Weight search: {dict(zip(names, [round(float(w),3) for w in best_w]))}. "
           f"Prior best (round 2): 0.837776."),
)
print(f"Logged experiment #{exp_id}")
