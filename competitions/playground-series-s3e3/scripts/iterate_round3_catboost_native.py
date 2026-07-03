"""Phase B iteration, round 3: CatBoost retry with native categorical handling.

Open item from STATUS.md: CatBoost badly underperformed (0.7627, exp #3) using the
same label/frequency-encoded categoricals as LGB/XGB. CatBoost's headline feature is
native cat_features handling (ordered target statistics) — untried so far. This round
builds a CatBoost-specific feature set that keeps the 7 categorical columns as raw
strings (passed via cat_features) instead of pre-encoding them, and uses
allow_writing_files=False + explicit thread_count per the s3e11 lesson (catboost_info
file writes can stall in sandboxed background execution).

If CatBoost still lands < 0.80 here, per instructions we respect the zero-weight
verdict and drop it permanently (do not retry again in future iterations).
"""
import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))

_spec = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

COMP = "competitions/playground-series-s3e3"
DATA = f"{COMP}/data"
TARGET, ID = "Attrition", "id"
N_SPLITS, SEED = 5, 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

CAT_COLS = ["BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"]
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18"]
SATISFACTION_COLS = ["EnvironmentSatisfaction", "JobSatisfaction",
                     "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]


def build_native_features(df):
    out = df.copy()
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    # Keep categoricals as raw strings (native CatBoost handling), no label/freq encoding
    for c in CAT_COLS:
        out[c] = out[c].astype(str)
    # Same numeric engineered features as build_features(), minus overtime_joblevel
    # (which depended on OverTime_le — replace with raw OverTime string interaction unnecessary,
    # CatBoost cat x numeric interactions are learned internally from ordered boosting).
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


Xtr_full = build_native_features(train)
Xte_full = build_native_features(test)
FEATS = [c for c in Xtr_full.columns if c not in ("id", "Attrition")]
cat_feature_idx = [FEATS.index(c) for c in CAT_COLS]

X = Xtr_full[FEATS].copy()
y = train[TARGET].to_numpy(np.int64)
Xtest = Xte_full[FEATS].copy()
print(f"n_features={len(FEATS)}  cat_features={CAT_COLS}  train={X.shape}  test={Xtest.shape}")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y))


def run_cat_native(depth, l2_leaf_reg, learning_rate):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y)); pred = np.zeros(len(Xtest))
    for f, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=2000,
            learning_rate=learning_rate, depth=depth, l2_leaf_reg=l2_leaf_reg,
            random_seed=SEED, thread_count=4, verbose=False,
            allow_writing_files=False, cat_features=cat_feature_idx,
        )
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_feature_idx),
              eval_set=Pool(X.iloc[va], y[va], cat_features=cat_feature_idx),
              early_stopping_rounds=150, verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
        print(f"  [CAT_native d={depth} l2={l2_leaf_reg}] fold{f} AUC={roc_auc_score(y[va], oof[va]):.5f}")
    return oof, pred


t0 = time.time()
configs = [
    dict(depth=5, l2_leaf_reg=8.0, learning_rate=0.03),   # same reg as exp#3 CAT
    dict(depth=4, l2_leaf_reg=12.0, learning_rate=0.03),  # tighter, matching LGB's shallow-strong-reg pattern
    dict(depth=3, l2_leaf_reg=16.0, learning_rate=0.05),  # very shallow + strong reg
]
best = None
for cfg in configs:
    oof, pred = run_cat_native(**cfg)
    score = roc_auc_score(y, oof)
    print(f"config {cfg}: OOF AUC={score:.6f}")
    if best is None or score > best[0]:
        best = (score, cfg, oof, pred)

elapsed = time.time() - t0
best_score, best_cfg, best_oof, best_pred = best
print(f"\nBest CAT_native config: {best_cfg}  OOF AUC={best_score:.6f}  ({elapsed:.1f}s total)")
print(f"Compare vs exp #3 CAT (encoded, no native cat): 0.762684")

exp_id = experiment_log.log_experiment_v2(
    COMP,
    model="CatBoost native categorical retry (allow_writing_files=False)",
    metric="roc_auc",
    direction="maximize",
    score=float(best_score),
    cv={"strategy": "StratifiedKFold", "n_splits": N_SPLITS, "seed": SEED},
    features=FEATS,
    base_models=[{"name": "CAT_native_best", "oof_auc": round(float(best_score), 6)}],
    ensemble={"method": "single_model", "weights": {"CAT_native_best": 1.0}},
    submission=None,
    notes=(f"Open item retry: native cat_features={CAT_COLS} passed directly to CatBoost "
           f"(raw strings, no label/freq pre-encoding) instead of the label+freq-encoded "
           f"features used by LGB/XGB/exp#3-CAT. Swept 3 depth/reg configs (best: {best_cfg}). "
           f"Result {best_score:.6f} vs exp #3 encoded-CAT 0.762684 "
           f"({'RESCUED' if best_score >= 0.80 else 'still <0.80, dropping CatBoost permanently per protocol'})."),
)
print(f"Logged experiment #{exp_id}")
