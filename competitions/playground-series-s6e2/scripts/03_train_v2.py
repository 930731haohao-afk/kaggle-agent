"""v2: LGB + XGB + CatBoost + more features + tuned params.
Target: maximize AUC-ROC. Submit probabilities.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

COMP_DIR = Path("C:/Users/user/ai_agents/kaggle/competitions/playground-series-s6e2")
DATA_DIR = COMP_DIR / "data"
N_FOLDS = 10
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

# ── Feature engineering ────────────────────────────────────────────────
# Age
df["Age_decade"] = df["Age"] // 10
df["Age_sq"] = df["Age"] ** 2

# BP
df["BP_high"] = (df["BP"] >= 140).astype(int)
df["BP_low"] = (df["BP"] < 100).astype(int)
df["BP_sq"] = df["BP"] ** 2

# Cholesterol
df["Chol_high"] = (df["Cholesterol"] > 240).astype(int)
df["Chol_borderline"] = ((df["Cholesterol"] >= 200) & (df["Cholesterol"] <= 240)).astype(int)
df["Chol_sq"] = df["Cholesterol"] ** 2

# Heart rate
df["HR_reserve"] = 220 - df["Age"] - df["Max HR"]
df["HR_pct_max"] = df["Max HR"] / (220 - df["Age"] + 1)
df["MaxHR_sq"] = df["Max HR"] ** 2

# ST depression
df["STdep_sq"] = df["ST depression"] ** 2
df["STdep_cat"] = pd.cut(df["ST depression"], bins=[-1, 0, 1, 2, 7], labels=[0, 1, 2, 3]).astype(int)

# Interactions
df["Age_x_MaxHR"] = df["Age"] * df["Max HR"]
df["Age_x_STdep"] = df["Age"] * df["ST depression"]
df["Age_x_Vessels"] = df["Age"] * df["Number of vessels fluro"]
df["Age_x_Thallium"] = df["Age"] * df["Thallium"]
df["Age_x_ChestPain"] = df["Age"] * df["Chest pain type"]
df["Age_x_ExAngina"] = df["Age"] * df["Exercise angina"]
df["Chol_x_Age"] = df["Cholesterol"] * df["Age"]
df["Chol_x_BP"] = df["Cholesterol"] * df["BP"]
df["Chol_x_MaxHR"] = df["Cholesterol"] * df["Max HR"]
df["BP_x_Age"] = df["BP"] * df["Age"]
df["BP_x_MaxHR"] = df["BP"] * df["Max HR"]
df["STdep_x_Slope"] = df["ST depression"] * df["Slope of ST"]
df["STdep_x_MaxHR"] = df["ST depression"] * df["Max HR"]
df["STdep_x_Angina"] = df["ST depression"] * df["Exercise angina"]
df["Vessels_x_Thallium"] = df["Number of vessels fluro"] * df["Thallium"]
df["Vessels_x_STdep"] = df["Number of vessels fluro"] * df["ST depression"]
df["ChestPain_x_Angina"] = df["Chest pain type"] * df["Exercise angina"]
df["ChestPain_x_STdep"] = df["Chest pain type"] * df["ST depression"]
df["Sex_x_Age"] = df["Sex"] * df["Age"]
df["Sex_x_ChestPain"] = df["Sex"] * df["Chest pain type"]

# Risk score
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
df["MaxHR_per_Age"] = df["Max HR"] / (df["Age"] + 1)
df["STdep_per_BP"] = df["ST depression"] / (df["BP"] + 1)

# Thallium flags
df["Thallium_7"] = (df["Thallium"] == 7).astype(int)
df["Thallium_6"] = (df["Thallium"] == 6).astype(int)

feature_names = df.columns.tolist()
log.info(f"Features: {len(feature_names)}")

X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

# ── 3-Model stacking with 10-fold CV ─────────────────────────────────

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

n_models = 3
model_names = ["LGB", "XGB", "CatBoost"]
oof_all = np.zeros((ntrain, n_models))
preds_all = np.zeros((len(X_test), n_models))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    log.info(f"=== Fold {fold+1}/{N_FOLDS} ===")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM — tuned
    m_lgb = lgb.LGBMClassifier(
        objective="binary", metric="auc", learning_rate=0.02,
        num_leaves=80, max_depth=-1, min_child_samples=30,
        subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.05, reg_lambda=0.5,
        n_estimators=5000, random_state=SEED, verbosity=-1,
    )
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof_all[val_idx, 0] = m_lgb.predict_proba(X_val)[:, 1]
    preds_all[:, 0] += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

    # XGBoost — tuned
    m_xgb = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", learning_rate=0.02,
        max_depth=7, min_child_weight=30, subsample=0.75,
        colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=0.5,
        n_estimators=5000, random_state=SEED, verbosity=0,
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_all[val_idx, 1] = m_xgb.predict_proba(X_val)[:, 1]
    preds_all[:, 1] += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    # CatBoost
    m_cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, subsample=0.8,
        eval_metric="AUC", random_seed=SEED, verbose=0,
        early_stopping_rounds=150,
    )
    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_all[val_idx, 2] = m_cat.predict_proba(X_val)[:, 1]
    preds_all[:, 2] += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

    aucs = [roc_auc_score(y_val, oof_all[val_idx, i]) for i in range(n_models)]
    log.info(f"  " + " | ".join(f"{model_names[i]} AUC={aucs[i]:.5f}" for i in range(n_models)))

# ── OOF scores ────────────────────────────────────────────────────────
log.info("\n=== OOF Scores ===")
for i, name in enumerate(model_names):
    auc = roc_auc_score(y_train, oof_all[:, i])
    log.info(f"  {name}: AUC={auc:.5f}")

# ── Optimize ensemble weights ─────────────────────────────────────────
from scipy.optimize import minimize

def neg_auc(weights):
    w = np.array(weights)
    w = w / w.sum()
    blend = (oof_all * w[None, :]).sum(axis=1)
    return -roc_auc_score(y_train, blend)

result = minimize(
    neg_auc,
    x0=np.ones(n_models) / n_models,
    method="SLSQP",
    bounds=[(0.0, 1.0)] * n_models,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)
best_weights = np.array(result.x)
best_auc = -result.fun

log.info(f"\nOptimized ensemble weights:")
for i, name in enumerate(model_names):
    log.info(f"  {name}: {best_weights[i]:.4f}")
log.info(f"Ensemble OOF AUC: {best_auc:.5f}")

# ── Generate submission ───────────────────────────────────────────────
final_proba = (preds_all * best_weights[None, :]).sum(axis=1)

sub = pd.DataFrame({"id": test_ids, "Heart Disease": final_proba})
sub_path = COMP_DIR / "submissions" / "sub_3model_v2.csv"
sub.to_csv(sub_path, index=False)
log.info(f"\nSubmission saved to {sub_path}")
log.info(f"Predicted mean proba: {final_proba.mean():.4f}")

# ── Log experiment ────────────────────────────────────────────────────
exp = {
    "experiment_id": 2,
    "name": "lgb_xgb_catboost_v2",
    "features": f"{len(feature_names)} features (more interactions, squared, thallium flags)",
    "models": model_names,
    "n_folds": N_FOLDS,
    "cv_scores": {
        "ensemble_auc": round(best_auc, 5),
        "weights": {name: round(w, 4) for name, w in zip(model_names, best_weights)},
    },
    "submission_file": "sub_3model_v2.csv",
}
exp_path = COMP_DIR / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments.append(exp)
exp_path.write_text(json.dumps(experiments, indent=2, default=str))
log.info("Experiment logged.")
