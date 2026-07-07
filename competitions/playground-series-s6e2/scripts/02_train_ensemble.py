"""Training pipeline for playground-series-s6e2 — Predicting Heart Disease.
Binary classification: Presence(1) / Absence(0).
Models: LightGBM + XGBoost + CatBoost ensemble.
Features: interactions, ratios, binning on medical features.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

COMP_DIR = Path("C:/Users/user/ai_agents/kaggle/competitions/playground-series-s6e2")
DATA_DIR = COMP_DIR / "data"
N_FOLDS = 5
SEED = 42

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test["id"].copy()

target = (train["Heart Disease"] == "Presence").astype(int)
train.drop(columns=["Heart Disease", "id"], inplace=True)
test.drop(columns=["id"], inplace=True)

df = pd.concat([train, test], axis=0, ignore_index=True)
ntrain = len(train)

log.info(f"Combined shape: {df.shape}, train={ntrain}, test={len(test)}")
log.info(f"Target: {target.mean():.4f} positive rate")

# ── Feature engineering ────────────────────────────────────────────────
# Original features are all numeric and clean — focus on interactions

# Age groups
df["Age_decade"] = df["Age"] // 10

# BP categories (medical thresholds)
df["BP_high"] = (df["BP"] >= 140).astype(int)
df["BP_low"] = (df["BP"] < 100).astype(int)

# Cholesterol categories
df["Chol_high"] = (df["Cholesterol"] > 240).astype(int)
df["Chol_borderline"] = ((df["Cholesterol"] >= 200) & (df["Cholesterol"] <= 240)).astype(int)

# Heart rate reserve proxy
df["HR_reserve"] = 220 - df["Age"] - df["Max HR"]
df["HR_pct_max"] = df["Max HR"] / (220 - df["Age"] + 1)

# Key interactions
df["Age_x_MaxHR"] = df["Age"] * df["Max HR"]
df["Age_x_STdep"] = df["Age"] * df["ST depression"]
df["Age_x_Vessels"] = df["Age"] * df["Number of vessels fluro"]
df["Chol_x_Age"] = df["Cholesterol"] * df["Age"]
df["BP_x_Chol"] = df["BP"] * df["Cholesterol"]
df["BP_x_Age"] = df["BP"] * df["Age"]
df["STdep_x_Slope"] = df["ST depression"] * df["Slope of ST"]
df["STdep_x_MaxHR"] = df["ST depression"] * df["Max HR"]
df["Vessels_x_Thallium"] = df["Number of vessels fluro"] * df["Thallium"]
df["Angina_x_STdep"] = df["Exercise angina"] * df["ST depression"]
df["ChestPain_x_Angina"] = df["Chest pain type"] * df["Exercise angina"]

# Risk score (domain-knowledge composite)
df["Risk_score"] = (
    df["Age"] / 77 +
    df["Sex"] +
    df["Chest pain type"] / 4 +
    df["BP"] / 200 +
    df["Cholesterol"] / 564 +
    df["FBS over 120"] +
    df["Exercise angina"] +
    df["ST depression"] / 6.2 +
    df["Number of vessels fluro"] / 3 +
    (df["Thallium"] == 7).astype(int)
)

# Ratios
df["Chol_per_Age"] = df["Cholesterol"] / (df["Age"] + 1)
df["BP_per_Age"] = df["BP"] / (df["Age"] + 1)

feature_names = df.columns.tolist()
log.info(f"Features: {len(feature_names)}")

# Split back
X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

# ── Model training with 5-fold CV ─────────────────────────────────────

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": SEED,
    "verbosity": -1,
}

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 5000,
    "random_state": SEED,
    "verbosity": 0,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(ntrain)
oof_xgb = np.zeros(ntrain)
preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    log.info(f"=== Fold {fold+1}/{N_FOLDS} ===")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
    preds_lgb += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

    # XGBoost
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
    preds_xgb += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    log.info(f"  LGB AUC: {fold_lgb_auc:.5f} | XGB AUC: {fold_xgb_auc:.5f}")

# ── OOF scores ────────────────────────────────────────────────────────
lgb_auc = roc_auc_score(y_train, oof_lgb)
xgb_auc = roc_auc_score(y_train, oof_xgb)
log.info(f"\nOOF LGB AUC: {lgb_auc:.5f}")
log.info(f"OOF XGB AUC: {xgb_auc:.5f}")

# ── Optimize ensemble weights ─────────────────────────────────────────
best_w, best_auc = 0.5, 0
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_lgb + (1 - w) * oof_xgb
    auc = roc_auc_score(y_train, blend)
    if auc > best_auc:
        best_w, best_auc = w, auc

log.info(f"Best ensemble: LGB weight={best_w:.2f}, XGB weight={1-best_w:.2f}")
log.info(f"Ensemble OOF AUC: {best_auc:.5f}")

# ── Generate submission ───────────────────────────────────────────────
final_proba = best_w * preds_lgb + (1 - best_w) * preds_xgb
final_preds = (final_proba >= 0.5).astype(int)

sub = pd.DataFrame({"id": test_ids, "Heart Disease": final_preds})
sub_path = COMP_DIR / "submissions" / "sub_ensemble_v1.csv"
sub.to_csv(sub_path, index=False)
log.info(f"Submission saved to {sub_path}")
log.info(f"Predicted positive rate: {final_preds.mean():.4f}")

# Also save probability submission in case metric is AUC
sub_proba = pd.DataFrame({"id": test_ids, "Heart Disease": final_proba})
sub_proba_path = COMP_DIR / "submissions" / "sub_ensemble_v1_proba.csv"
sub_proba.to_csv(sub_proba_path, index=False)
log.info(f"Probability submission saved to {sub_proba_path}")

# ── Feature importance ────────────────────────────────────────────────
imp = pd.Series(m_lgb.feature_importances_, index=feature_names).sort_values(ascending=False)
log.info(f"\nTop 15 features (LGB last fold):")
for feat, val in imp.head(15).items():
    log.info(f"  {feat}: {val}")

# ── Log experiment ────────────────────────────────────────────────────
exp = {
    "experiment_id": 1,
    "name": "lgb_xgb_ensemble_v1",
    "features": f"{len(feature_names)} features (interactions, medical bins, risk score, ratios)",
    "models": {"lightgbm": lgb_params, "xgboost": xgb_params},
    "cv_scores": {
        "lgb_auc": round(lgb_auc, 5),
        "xgb_auc": round(xgb_auc, 5),
        "ensemble_auc": round(best_auc, 5),
        "ensemble_weights": f"lgb={best_w:.2f}, xgb={1-best_w:.2f}",
    },
    "submission_file": "sub_ensemble_v1.csv",
}
exp_path = COMP_DIR / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments.append(exp)
exp_path.write_text(json.dumps(experiments, indent=2, default=str))
log.info("Experiment logged.")
