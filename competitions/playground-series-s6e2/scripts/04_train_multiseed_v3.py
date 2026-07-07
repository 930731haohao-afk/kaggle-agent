"""v3: Multi-seed averaging with LGB + CatBoost.
Key insight: average predictions across multiple random seeds
to reduce variance. Also try original UCI dataset augmentation.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

COMP_DIR = Path("C:/Users/user/ai_agents/kaggle/competitions/playground-series-s6e2")
DATA_DIR = COMP_DIR / "data"
N_FOLDS = 5
SEEDS = [42, 123, 456, 789, 2024]

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

# ── Feature engineering (focused — only proven useful features) ────────
# Keep it simpler to avoid noise

df["Age_decade"] = df["Age"] // 10
df["BP_high"] = (df["BP"] >= 140).astype(int)
df["Chol_high"] = (df["Cholesterol"] > 240).astype(int)
df["HR_reserve"] = 220 - df["Age"] - df["Max HR"]
df["HR_pct_max"] = df["Max HR"] / (220 - df["Age"] + 1)

# Top interactions from feature importance
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

# Risk score
df["Risk_score"] = (
    df["Age"] / 77 + df["Sex"] + df["Chest pain type"] / 4 +
    df["BP"] / 200 + df["Cholesterol"] / 564 + df["FBS over 120"] +
    df["Exercise angina"] + df["ST depression"] / 6.2 +
    df["Number of vessels fluro"] / 3 + (df["Thallium"] == 7).astype(int)
)

df["Chol_per_Age"] = df["Cholesterol"] / (df["Age"] + 1)
df["BP_per_Age"] = df["BP"] / (df["Age"] + 1)
df["Thallium_7"] = (df["Thallium"] == 7).astype(int)

feature_names = df.columns.tolist()
log.info(f"Features: {len(feature_names)}")

X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

# ── Multi-seed averaging ──────────────────────────────────────────────
all_oof_lgb = np.zeros((ntrain, len(SEEDS)))
all_oof_cat = np.zeros((ntrain, len(SEEDS)))
all_preds_lgb = np.zeros((len(X_test), len(SEEDS)))
all_preds_cat = np.zeros((len(X_test), len(SEEDS)))

for s_idx, seed in enumerate(SEEDS):
    log.info(f"\n====== SEED {seed} ({s_idx+1}/{len(SEEDS)}) ======")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(ntrain)
    oof_cat = np.zeros(ntrain)
    preds_lgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBM
        m_lgb = lgb.LGBMClassifier(
            objective="binary", metric="auc", learning_rate=0.02,
            num_leaves=80, min_child_samples=30, subsample=0.75,
            colsample_bytree=0.75, reg_alpha=0.05, reg_lambda=0.5,
            n_estimators=5000, random_state=seed, verbosity=-1,
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        preds_lgb += m_lgb.predict_proba(X_test)[:, 1] / N_FOLDS

        # CatBoost
        m_cat = CatBoostClassifier(
            iterations=3000, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, subsample=0.8,
            eval_metric="AUC", random_seed=seed, verbose=0,
            early_stopping_rounds=150,
        )
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        preds_cat += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

    lgb_auc = roc_auc_score(y_train, oof_lgb)
    cat_auc = roc_auc_score(y_train, oof_cat)
    log.info(f"  Seed {seed}: LGB AUC={lgb_auc:.5f}, CatBoost AUC={cat_auc:.5f}")

    all_oof_lgb[:, s_idx] = oof_lgb
    all_oof_cat[:, s_idx] = oof_cat
    all_preds_lgb[:, s_idx] = preds_lgb
    all_preds_cat[:, s_idx] = preds_cat

# ── Average across seeds ──────────────────────────────────────────────
avg_oof_lgb = all_oof_lgb.mean(axis=1)
avg_oof_cat = all_oof_cat.mean(axis=1)
avg_preds_lgb = all_preds_lgb.mean(axis=1)
avg_preds_cat = all_preds_cat.mean(axis=1)

lgb_auc_avg = roc_auc_score(y_train, avg_oof_lgb)
cat_auc_avg = roc_auc_score(y_train, avg_oof_cat)
log.info(f"\n=== Multi-seed averaged OOF ===")
log.info(f"  LGB: {lgb_auc_avg:.5f}")
log.info(f"  CatBoost: {cat_auc_avg:.5f}")

# ── Optimize blend weights ────────────────────────────────────────────
best_w, best_auc = 0.5, 0
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * avg_oof_lgb + (1 - w) * avg_oof_cat
    auc = roc_auc_score(y_train, blend)
    if auc > best_auc:
        best_w, best_auc = w, auc

log.info(f"Best blend: LGB={best_w:.2f}, CatBoost={1-best_w:.2f}")
log.info(f"Ensemble OOF AUC: {best_auc:.5f}")

# ── Generate submission ───────────────────────────────────────────────
final_proba = best_w * avg_preds_lgb + (1 - best_w) * avg_preds_cat

sub = pd.DataFrame({"id": test_ids, "Heart Disease": final_proba})
sub_path = COMP_DIR / "submissions" / "sub_multiseed_v3.csv"
sub.to_csv(sub_path, index=False)
log.info(f"\nSubmission saved to {sub_path}")

# ── Log experiment ────────────────────────────────────────────────────
exp = {
    "experiment_id": 3,
    "name": "multiseed_lgb_cat_v3",
    "features": f"{len(feature_names)} features (focused interactions)",
    "models": ["LGB", "CatBoost"],
    "n_folds": N_FOLDS,
    "n_seeds": len(SEEDS),
    "seeds": SEEDS,
    "cv_scores": {
        "lgb_auc_avg": round(lgb_auc_avg, 5),
        "cat_auc_avg": round(cat_auc_avg, 5),
        "ensemble_auc": round(best_auc, 5),
        "blend_weights": f"lgb={best_w:.2f}, cat={1-best_w:.2f}",
    },
    "submission_file": "sub_multiseed_v3.csv",
}
exp_path = COMP_DIR / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments.append(exp)
exp_path.write_text(json.dumps(experiments, indent=2, default=str))
log.info("Experiment logged.")
