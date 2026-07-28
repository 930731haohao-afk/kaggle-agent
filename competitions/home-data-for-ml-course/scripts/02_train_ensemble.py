"""Full training pipeline for home-data-for-ml-course.
Features: ordinal encoding, interaction features, area aggregates.
Models: LightGBM + XGBoost ensemble with OOF-optimized weights.
Target: log1p(SalePrice), metric: RMSLE.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

COMP_DIR = Path("C:/Users/user/ai_agents/kaggle/competitions/home-data-for-ml-course")
DATA_DIR = COMP_DIR / "data"
N_FOLDS = 5
SEED = 42

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test["Id"].copy()

target = np.log1p(train["SalePrice"])
train.drop(columns=["SalePrice", "Id"], inplace=True)
test.drop(columns=["Id"], inplace=True)

df = pd.concat([train, test], axis=0, ignore_index=True)
ntrain = len(train)

log.info(f"Combined shape: {df.shape}, train={ntrain}, test={len(test)}")

# ── Feature engineering ────────────────────────────────────────────────

# 1. Ordinal quality mappings
qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
for col in qual_cols:
    df[col] = df[col].map(qual_map).fillna(0).astype(int)

# Other ordinal mappings
df["BsmtExposure"] = df["BsmtExposure"].map({"No": 1, "Mn": 2, "Av": 3, "Gd": 4}).fillna(0).astype(int)
df["BsmtFinType1"] = df["BsmtFinType1"].map({"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}).fillna(0).astype(int)
df["BsmtFinType2"] = df["BsmtFinType2"].map({"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}).fillna(0).astype(int)
df["GarageFinish"] = df["GarageFinish"].map({"Unf": 1, "RFn": 2, "Fin": 3}).fillna(0).astype(int)
df["Fence"] = df["Fence"].map({"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).fillna(0).astype(int)
df["Functional"] = df["Functional"].map({"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).fillna(8).astype(int)
df["LotShape"] = df["LotShape"].map({"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}).fillna(0).astype(int)
df["PavedDrive"] = df["PavedDrive"].map({"N": 0, "P": 1, "Y": 2}).fillna(0).astype(int)
df["CentralAir"] = df["CentralAir"].map({"N": 0, "Y": 1}).fillna(0).astype(int)
df["Street"] = df["Street"].map({"Grvl": 0, "Pave": 1}).fillna(0).astype(int)
df["LandSlope"] = df["LandSlope"].map({"Sev": 1, "Mod": 2, "Gtl": 3}).fillna(0).astype(int)

# 2. Fill NAs for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

# 3. Aggregate / interaction features
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"] + df["GarageArea"]
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["IsRemod"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)

# Quality x Area interactions
df["OverallQual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
df["OverallQual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
df["ExterQual_x_TotalSF"] = df["ExterQual"] * df["TotalSF"]
df["KitchenQual_x_GrLivArea"] = df["KitchenQual"] * df["GrLivArea"]
df["GarageQual_x_GarageArea"] = df["GarageQual"] * df["GarageArea"]
df["BsmtQual_x_TotalBsmtSF"] = df["BsmtQual"] * df["TotalBsmtSF"]

# Neighborhood median price (from train only)
# We'll use target-encoding-like approach but only from train
neigh_median = target.groupby(df.iloc[:ntrain]["Neighborhood"]).median()
df["Neighborhood_MedianPrice"] = df["Neighborhood"].map(neigh_median).fillna(target.median())

# 4. Label encode remaining categoricals
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
log.info(f"Label encoding {len(cat_cols)} categorical columns: {cat_cols}")
for col in cat_cols:
    df[col] = df[col].fillna("Missing")
    df[col] = df[col].astype("category").cat.codes

# 5. Split back
X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

feature_names = X_train.columns.tolist()
log.info(f"Features: {len(feature_names)}")

# ── Model training with 5-fold CV ─────────────────────────────────────

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": SEED,
    "verbosity": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": SEED,
    "verbosity": 0,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(ntrain)
oof_xgb = np.zeros(ntrain)
preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    log.info(f"=== Fold {fold+1}/{N_FOLDS} ===")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    preds_lgb += model_lgb.predict(X_test) / N_FOLDS

    # XGBoost
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(X_test) / N_FOLDS

    fold_lgb_rmse = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    fold_xgb_rmse = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    log.info(f"  LGB RMSE(log): {fold_lgb_rmse:.5f} | XGB RMSE(log): {fold_xgb_rmse:.5f}")

# ── OOF scores (RMSLE = RMSE of log1p predictions) ────────────────────
lgb_rmsle = np.sqrt(mean_squared_error(y_train, oof_lgb))
xgb_rmsle = np.sqrt(mean_squared_error(y_train, oof_xgb))
log.info(f"\nOOF LGB RMSLE: {lgb_rmsle:.5f}")
log.info(f"OOF XGB RMSLE: {xgb_rmsle:.5f}")

# ── Optimize ensemble weights ─────────────────────────────────────────
best_w, best_score = 0.5, 999
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * oof_lgb + (1 - w) * oof_xgb
    score = np.sqrt(mean_squared_error(y_train, blend))
    if score < best_score:
        best_w, best_score = w, score

log.info(f"Best ensemble: LGB weight={best_w:.2f}, XGB weight={1-best_w:.2f}")
log.info(f"Ensemble OOF RMSLE: {best_score:.5f}")

# ── Generate submission ───────────────────────────────────────────────
final_preds_log = best_w * preds_lgb + (1 - best_w) * preds_xgb
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame({"Id": test_ids, "SalePrice": final_preds})
sub_path = COMP_DIR / "submissions" / "sub_ensemble_v1.csv"
sub.to_csv(sub_path, index=False)
log.info(f"Submission saved to {sub_path}")
log.info(f"Prediction stats: mean={final_preds.mean():.0f}, median={np.median(final_preds):.0f}, min={final_preds.min():.0f}, max={final_preds.max():.0f}")

# ── Log experiment ────────────────────────────────────────────────────
exp = {
    "experiment_id": 1,
    "name": "lgb_xgb_ensemble_v1",
    "features": f"{len(feature_names)} features (ordinal encoding, interactions, area aggregates, neighborhood median)",
    "models": {"lightgbm": lgb_params, "xgboost": xgb_params},
    "cv_scores": {
        "lgb_rmsle": round(lgb_rmsle, 5),
        "xgb_rmsle": round(xgb_rmsle, 5),
        "ensemble_rmsle": round(best_score, 5),
        "ensemble_weights": f"lgb={best_w:.2f}, xgb={1-best_w:.2f}",
    },
    "submission_file": "sub_ensemble_v1.csv",
}
exp_path = COMP_DIR / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments.append(exp)
exp_path.write_text(json.dumps(experiments, indent=2, default=str))
log.info("Experiment logged.")
