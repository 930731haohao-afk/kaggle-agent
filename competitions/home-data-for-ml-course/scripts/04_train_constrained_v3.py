"""v3: Constrained positive-weight ensemble of XGB + GBR + Ridge.
Remove ElasticNet (weakest). Use bounded optimization.
Also adds outlier removal and skew correction on features.
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import skew

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

y_raw = train["SalePrice"].values
target = np.log1p(y_raw)

# Remove outliers (GrLivArea > 4000 with low price)
outlier_mask = ~((train["GrLivArea"] > 4000) & (y_raw < 300000))
log.info(f"Removing {(~outlier_mask).sum()} outliers")
train = train[outlier_mask].reset_index(drop=True)
target = target[outlier_mask]

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

# Quality interactions
df["OverallQual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
df["OverallQual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]
df["ExterQual_x_TotalSF"] = df["ExterQual"] * df["TotalSF"]
df["KitchenQual_x_GrLivArea"] = df["KitchenQual"] * df["GrLivArea"]
df["GarageQual_x_GarageArea"] = df["GarageQual"] * df["GarageArea"]
df["BsmtQual_x_TotalBsmtSF"] = df["BsmtQual"] * df["TotalBsmtSF"]
df["OverallQual_sq"] = df["OverallQual"] ** 2
df["GrLivArea_log"] = np.log1p(df["GrLivArea"])
df["TotalSF_log"] = np.log1p(df["TotalSF"])
df["LotArea_log"] = np.log1p(df["LotArea"])

# Area ratios
df["LivArea_to_Lot"] = df["GrLivArea"] / (df["LotArea"] + 1)
df["Bsmt_to_1stFlr"] = df["TotalBsmtSF"] / (df["1stFlrSF"] + 1)
df["TotalQual"] = df["OverallQual"] + df["ExterQual"] + df["KitchenQual"] + df["BsmtQual"] + df["GarageQual"]
df["AvgRoomSize"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + 1)

# Neighborhood target encoding (from train only)
neigh_median = pd.Series(target[:ntrain]).groupby(df.iloc[:ntrain]["Neighborhood"]).median()
df["Neighborhood_MedianPrice"] = df["Neighborhood"].map(neigh_median).fillna(pd.Series(target[:ntrain]).median())

# 4. Label encode remaining categoricals
cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
log.info(f"Label encoding {len(cat_cols)} categorical columns")
for col in cat_cols:
    df[col] = df[col].fillna("Missing")
    df[col] = df[col].astype("category").cat.codes

# 5. Correct skewed numeric features with log1p
numeric_feats = df.select_dtypes(include=[np.number]).columns
skewed = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
high_skew = skewed[abs(skewed) > 0.75].index
log.info(f"Log1p transforming {len(high_skew)} highly skewed features")
for col in high_skew:
    if df[col].min() >= 0:
        df[col] = np.log1p(df[col])

# 6. Split back
X_train = df.iloc[:ntrain].reset_index(drop=True)
X_test = df.iloc[ntrain:].reset_index(drop=True)
y_train = pd.Series(target[:ntrain]).reset_index(drop=True)

feature_names = X_train.columns.tolist()
log.info(f"Features: {len(feature_names)}")

# ── 4-Model stacking with 5-fold CV ───────────────────────────────────

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

n_models = 4
model_names = ["LGB", "XGB", "Ridge", "GBR"]
oof_all = np.zeros((ntrain, n_models))
preds_all = np.zeros((len(X_test), n_models))

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    log.info(f"=== Fold {fold+1}/{N_FOLDS} ===")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    X_tr_s, X_val_s = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # LightGBM
    m_lgb = lgb.LGBMRegressor(
        objective="regression", metric="rmse", learning_rate=0.02,
        num_leaves=50, min_child_samples=5, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.0,
        n_estimators=5000, random_state=SEED, verbosity=-1,
    )
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof_all[val_idx, 0] = m_lgb.predict(X_val)
    preds_all[:, 0] += m_lgb.predict(X_test) / N_FOLDS

    # XGBoost
    m_xgb = xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", learning_rate=0.02,
        max_depth=5, min_child_weight=3, subsample=0.7,
        colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.0,
        n_estimators=5000, random_state=SEED, verbosity=0,
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_all[val_idx, 1] = m_xgb.predict(X_val)
    preds_all[:, 1] += m_xgb.predict(X_test) / N_FOLDS

    # Ridge
    m_ridge = Ridge(alpha=15.0, random_state=SEED)
    m_ridge.fit(X_tr_s, y_tr)
    oof_all[val_idx, 2] = m_ridge.predict(X_val_s)
    preds_all[:, 2] += m_ridge.predict(X_test_scaled) / N_FOLDS

    # GradientBoosting
    m_gbr = GradientBoostingRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=4,
        min_samples_leaf=5, subsample=0.7, random_state=SEED,
        loss="huber",
    )
    m_gbr.fit(X_tr, y_tr)
    oof_all[val_idx, 3] = m_gbr.predict(X_val)
    preds_all[:, 3] += m_gbr.predict(X_test) / N_FOLDS

    scores = []
    for m in range(n_models):
        s = mean_absolute_error(np.expm1(y_val), np.expm1(oof_all[val_idx, m]))
        scores.append(s)
    log.info(f"  MAE: " + " | ".join(f"{model_names[i]}={scores[i]:.0f}" for i in range(n_models)))

# ── Individual OOF scores ─────────────────────────────────────────────
log.info("\n=== OOF Scores ===")
for i, name in enumerate(model_names):
    rmsle = np.sqrt(mean_squared_error(y_train, oof_all[:, i]))
    mae = mean_absolute_error(np.expm1(y_train), np.expm1(oof_all[:, i]))
    log.info(f"  {name}: RMSLE={rmsle:.5f}, MAE={mae:.0f}")

# ── Constrained ensemble weights (positive, sum to 1) ────────────────
def neg_mae(weights):
    w = np.array(weights)
    blend = (oof_all * w[None, :]).sum(axis=1)
    return mean_absolute_error(np.expm1(y_train), np.expm1(blend))

from scipy.optimize import minimize
result = minimize(
    neg_mae,
    x0=np.ones(n_models) / n_models,
    method="SLSQP",
    bounds=[(0.0, 1.0)] * n_models,
    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
)
best_weights = np.array(result.x)
best_mae = result.fun

log.info(f"\nOptimized ensemble weights (constrained):")
for i, name in enumerate(model_names):
    log.info(f"  {name}: {best_weights[i]:.4f}")
log.info(f"Ensemble MAE: {best_mae:.0f}")

blend_oof = (oof_all * best_weights[None, :]).sum(axis=1)
ensemble_rmsle = np.sqrt(mean_squared_error(y_train, blend_oof))
log.info(f"Ensemble RMSLE: {ensemble_rmsle:.5f}")

# ── Generate submission ───────────────────────────────────────────────
final_preds_log = (preds_all * best_weights[None, :]).sum(axis=1)
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0, None)

sub = pd.DataFrame({"Id": test_ids, "SalePrice": final_preds})
sub_path = COMP_DIR / "submissions" / "sub_stack_v3.csv"
sub.to_csv(sub_path, index=False)
log.info(f"\nSubmission saved to {sub_path}")
log.info(f"Prediction stats: mean={final_preds.mean():.0f}, median={np.median(final_preds):.0f}")

# ── Log experiment ────────────────────────────────────────────────────
exp = {
    "experiment_id": 3,
    "name": "constrained_4model_stack_v3",
    "features": f"{len(feature_names)} features (+ outlier removal, skew correction, log features, huber loss GBR)",
    "models": model_names,
    "cv_scores": {
        "ensemble_rmsle": round(ensemble_rmsle, 5),
        "ensemble_mae": round(best_mae, 0),
        "weights": {name: round(w, 4) for name, w in zip(model_names, best_weights)},
    },
    "submission_file": "sub_stack_v3.csv",
}
exp_path = COMP_DIR / "experiments.json"
experiments = json.loads(exp_path.read_text())
experiments.append(exp)
exp_path.write_text(json.dumps(experiments, indent=2, default=str))
log.info("Experiment logged.")
