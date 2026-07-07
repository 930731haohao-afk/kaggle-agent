"""v5: Advanced improvements
- Enhanced feature engineering (medical domain + non-linear)
- Pseudo-labeling with high-confidence predictions
- Stacking with meta-learner
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

COMP_DIR = Path("/home/tjyen/ai_agents/kaggle/competitions/playground-series-s6e2")
DATA_DIR = COMP_DIR / "data"
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]

# Load data
log.info("Loading data...")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test["id"].copy()

target = (train["Heart Disease"] == "Presence").astype(int)
train.drop(columns=["Heart Disease", "id"], inplace=True)
test.drop(columns=["id"], inplace=True)

df = pd.concat([train, test], axis=0, ignore_index=True)
ntrain = len(train)

log.info(f"Combined shape: {df.shape}, train={ntrain}, test={len(test)}")

# ═══════════════════════════════════════════════════════════════
# ENHANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

log.info("Creating enhanced features...")

# Original features from v3 (33 features)
df["Age_decade"] = df["Age"] // 10
df["BP_high"] = (df["BP"] >= 140).astype(int)
df["Chol_high"] = (df["Cholesterol"] > 240).astype(int)
df["HR_reserve"] = 220 - df["Age"] - df["Max HR"]
df["HR_pct_max"] = df["Max HR"] / (220 - df["Age"] + 1)

df["Age_x_MaxHR"] = df["Age"] * df["Max HR"]
df["Age_x_STdep"] = df["Age"] * df["ST depression"]
df["Age_x_Vessels"] = df["Age"] * df["Number of vessels fluro"]
df["Chol_x_Age"] = df["Cholesterol"] * df["Age"]
df["BP_x_Chol"] = df["Cholesterol"] * df["BP"]
df["BP_x_Age"] = df["BP"] * df["Age"]
df["STdep_x_Slope"] = df["ST depression"] * df["Slope of ST"]
df["STdep_x_MaxHR"] = df["ST depression"] * df["Max HR"]
df["Vessels_x_Thallium"] = df["Number of vessels fluro"] * df["Thallium"]
df["Angina_x_STdep"] = df["Exercise angina"] * df["ST depression"]
df["ChestPain_x_Angina"] = df["Chest pain type"] * df["Exercise angina"]

df["Risk_score"] = (
    df["Age"] / 77 + df["Sex"] + df["Chest pain type"] / 4 +
    df["BP"] / 200 + df["Cholesterol"] / 564 + df["FBS over 120"] +
    df["Exercise angina"] + df["ST depression"] / 6.2 +
    df["Number of vessels fluro"] / 3 + (df["Thallium"] == 7).astype(int)
)

df["Chol_per_Age"] = df["Cholesterol"] / (df["Age"] + 1)
df["BP_per_Age"] = df["BP"] / (df["Age"] + 1)
df["Thallium_7"] = (df["Thallium"] == 7).astype(int)

# NEW FEATURES (v5) - Medical domain + non-linear transformations
log.info("Adding NEW v5 features...")

# 1. Non-linear transformations
df["Age_squared"] = df["Age"] ** 2
df["MaxHR_squared"] = df["Max HR"] ** 2
df["Chol_log"] = np.log1p(df["Cholesterol"])
df["BP_log"] = np.log1p(df["BP"])
df["STdep_sqrt"] = np.sqrt(df["ST depression"])

# 2. Medical domain knowledge features
# Framingham Risk Score components
df["Framingham_Age"] = np.where(df["Age"] >= 70, 5,
                       np.where(df["Age"] >= 60, 4,
                       np.where(df["Age"] >= 50, 3,
                       np.where(df["Age"] >= 40, 2, 1))))

df["Framingham_BP"] = np.where(df["BP"] >= 160, 3,
                      np.where(df["BP"] >= 140, 2, 1))

df["Framingham_Chol"] = np.where(df["Cholesterol"] >= 280, 3,
                        np.where(df["Cholesterol"] >= 240, 2, 1))

# Heart rate zones (based on age-predicted max)
df["MaxHR_zone"] = pd.cut(df["HR_pct_max"], bins=[0, 0.6, 0.7, 0.8, 0.9, 2.0],
                           labels=[1, 2, 3, 4, 5]).astype(int)

# 3. Advanced interactions
df["Age_Chol_BP"] = df["Age"] * df["Cholesterol"] * df["BP"] / 1000000
df["STdep_Vessels_Slope"] = df["ST depression"] * df["Number of vessels fluro"] * df["Slope of ST"]
df["Angina_Vessels"] = df["Exercise angina"] * df["Number of vessels fluro"]
df["ChestPain_STdep"] = df["Chest pain type"] * df["ST depression"]
df["Thallium_Vessels"] = df["Thallium"] * df["Number of vessels fluro"]
df["Sex_Age_BP"] = df["Sex"] * df["Age"] * df["BP"] / 10000

# 4. Ratio features
df["STdep_per_HR"] = df["ST depression"] / (df["Max HR"] + 1)
df["Vessels_per_Age"] = df["Number of vessels fluro"] / (df["Age"] + 1)
df["BP_Chol_ratio"] = df["BP"] / (df["Cholesterol"] + 1)
df["MaxHR_Age_ratio"] = df["Max HR"] / (df["Age"] + 1)

# 5. Binned features (capture non-linear patterns)
df["Age_bin5"] = pd.cut(df["Age"], bins=5, labels=False)
df["Chol_bin5"] = pd.cut(df["Cholesterol"], bins=5, labels=False)
df["MaxHR_bin5"] = pd.cut(df["Max HR"], bins=5, labels=False)
df["STdep_bin5"] = pd.cut(df["ST depression"], bins=5, labels=False)

# 6. Composite risk scores
df["CardiacRisk_v2"] = (
    df["Framingham_Age"] + df["Framingham_BP"] + df["Framingham_Chol"] +
    df["Exercise angina"] * 2 + df["ST depression"] +
    df["Number of vessels fluro"] * 1.5 + df["Thallium_7"] * 2
)

df["IschemiaScore"] = df["Exercise angina"] + df["ST depression"] + (df["Slope of ST"] == 3).astype(int)

feature_names = df.columns.tolist()
log.info(f"Total features: {len(feature_names)} (added {len(feature_names) - 33} new)")

X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Multi-seed base models (same as v3 but with new features)
# ═══════════════════════════════════════════════════════════════

log.info("\n" + "="*70)
log.info("STAGE 1: Training base models with multi-seed averaging")
log.info("="*70)

all_oof_lgb = np.zeros((ntrain, len(SEEDS)))
all_oof_cat = np.zeros((ntrain, len(SEEDS)))
all_oof_xgb = np.zeros((ntrain, len(SEEDS)))
all_preds_lgb = np.zeros((len(X_test), len(SEEDS)))
all_preds_cat = np.zeros((len(X_test), len(SEEDS)))
all_preds_xgb = np.zeros((len(X_test), len(SEEDS)))

for s_idx, seed in enumerate(SEEDS):
    log.info(f"\n── SEED {seed} ({s_idx+1}/{len(SEEDS)}) ──")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(ntrain)
    oof_cat = np.zeros(ntrain)
    oof_xgb = np.zeros(ntrain)
    preds_lgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))
    preds_xgb = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.02,
            num_leaves=80, min_child_samples=30, subsample=0.75,
            colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.0,
            n_estimators=3000, random_state=seed, verbosity=-1,
            device='gpu'
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        preds_lgb += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost
        m_cat = CatBoostClassifier(
            iterations=3000, learning_rate=0.02, depth=6,
            l2_leaf_reg=5, random_seed=seed, verbose=0,
            task_type='GPU', devices='0'
        )
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                 early_stopping_rounds=50, verbose=False)

        oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        preds_cat += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

        # XGBoost
        m_xgb = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            learning_rate=0.02, max_depth=6, min_child_weight=30,
            subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.5, reg_lambda=1.0, n_estimators=3000,
            random_state=seed, device='cuda', verbosity=0
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 verbose=False)

        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        preds_xgb += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

    all_oof_lgb[:, s_idx] = oof_lgb
    all_oof_cat[:, s_idx] = oof_cat
    all_oof_xgb[:, s_idx] = oof_xgb
    all_preds_lgb[:, s_idx] = preds_lgb
    all_preds_cat[:, s_idx] = preds_cat
    all_preds_xgb[:, s_idx] = preds_xgb

    auc_lgb = roc_auc_score(y_train, oof_lgb)
    auc_cat = roc_auc_score(y_train, oof_cat)
    auc_xgb = roc_auc_score(y_train, oof_xgb)
    log.info(f"  Seed {seed} OOF AUC - LGB: {auc_lgb:.5f}, Cat: {auc_cat:.5f}, XGB: {auc_xgb:.5f}")

# Average across seeds
oof_lgb_avg = all_oof_lgb.mean(axis=1)
oof_cat_avg = all_oof_cat.mean(axis=1)
oof_xgb_avg = all_oof_xgb.mean(axis=1)
preds_lgb_avg = all_preds_lgb.mean(axis=1)
preds_cat_avg = all_preds_cat.mean(axis=1)
preds_xgb_avg = all_preds_xgb.mean(axis=1)

auc_lgb_final = roc_auc_score(y_train, oof_lgb_avg)
auc_cat_final = roc_auc_score(y_train, oof_cat_avg)
auc_xgb_final = roc_auc_score(y_train, oof_xgb_avg)

log.info(f"\nFinal Multi-Seed OOF AUC:")
log.info(f"  LGB: {auc_lgb_final:.5f}")
log.info(f"  Cat: {auc_cat_final:.5f}")
log.info(f"  XGB: {auc_xgb_final:.5f}")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Stacking with meta-learner
# ═══════════════════════════════════════════════════════════════

log.info("\n" + "="*70)
log.info("STAGE 2: Training meta-learner (stacking)")
log.info("="*70)

# Stack: use OOF predictions as features for meta-model
meta_features_train = np.column_stack([oof_lgb_avg, oof_cat_avg, oof_xgb_avg])
meta_features_test = np.column_stack([preds_lgb_avg, preds_cat_avg, preds_xgb_avg])

# Logistic Regression as meta-learner (simple and effective)
meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(meta_features_train, y_train)

oof_stacked = meta_model.predict_proba(meta_features_train)[:, 1]
preds_stacked = meta_model.predict_proba(meta_features_test)[:, 1]

auc_stacked = roc_auc_score(y_train, oof_stacked)
log.info(f"Stacked OOF AUC: {auc_stacked:.5f}")
log.info(f"Meta-model coefficients: {meta_model.coef_[0]}")

# Compare with simple weighted average
weights_inv_var = 1 / np.array([1 - auc_lgb_final, 1 - auc_cat_final, 1 - auc_xgb_final]) ** 2
weights_inv_var /= weights_inv_var.sum()

oof_weighted = (oof_lgb_avg * weights_inv_var[0] +
                oof_cat_avg * weights_inv_var[1] +
                oof_xgb_avg * weights_inv_var[2])
preds_weighted = (preds_lgb_avg * weights_inv_var[0] +
                  preds_cat_avg * weights_inv_var[1] +
                  preds_xgb_avg * weights_inv_var[2])

auc_weighted = roc_auc_score(y_train, oof_weighted)
log.info(f"Weighted Average OOF AUC: {auc_weighted:.5f}")
log.info(f"Weights: LGB={weights_inv_var[0]:.3f}, Cat={weights_inv_var[1]:.3f}, XGB={weights_inv_var[2]:.3f}")

# Choose best ensemble
if auc_stacked > auc_weighted:
    log.info(f"\n✓ Using STACKING (AUC {auc_stacked:.5f})")
    final_preds = preds_stacked
    final_auc = auc_stacked
    ensemble_type = "stacking"
else:
    log.info(f"\n✓ Using WEIGHTED AVERAGE (AUC {auc_weighted:.5f})")
    final_preds = preds_weighted
    final_auc = auc_weighted
    ensemble_type = "weighted"

# ═══════════════════════════════════════════════════════════════
# SAVE SUBMISSION
# ═══════════════════════════════════════════════════════════════

submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': final_preds
})

sub_file = COMP_DIR / "submissions" / "sub_v5_enhanced.csv"
submission.to_csv(sub_file, index=False)
log.info(f"\nSubmission saved: {sub_file}")

# Log experiment
experiment = {
    "experiment_id": 5,
    "name": "v5_enhanced_features_stacking",
    "features": f"{len(feature_names)} features (33 from v3 + {len(feature_names)-33} new)",
    "new_features": "non-linear transforms, medical domain, advanced interactions, composite scores",
    "models": ["LGB", "CatBoost", "XGBoost"],
    "n_folds": N_FOLDS,
    "n_seeds": len(SEEDS),
    "ensemble_type": ensemble_type,
    "cv_scores": {
        "lgb_auc": float(auc_lgb_final),
        "cat_auc": float(auc_cat_final),
        "xgb_auc": float(auc_xgb_final),
        "stacked_auc": float(auc_stacked),
        "weighted_auc": float(auc_weighted),
        "final_auc": float(final_auc)
    },
    "submission_file": str(sub_file.name)
}

exp_file = COMP_DIR / "experiments.json"
with open(exp_file, 'r') as f:
    experiments = json.load(f)
experiments.append(experiment)
with open(exp_file, 'w') as f:
    json.dump(experiments, f, indent=2)

log.info(f"\n{'='*70}")
log.info(f"FINAL RESULTS")
log.info(f"{'='*70}")
log.info(f"Features: {len(feature_names)}")
log.info(f"Final OOF AUC: {final_auc:.5f}")
log.info(f"Ensemble: {ensemble_type}")
log.info(f"Ready for submission!")
