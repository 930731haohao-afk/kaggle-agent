"""Phase-B iteration Round 1 for s3e20 (Rwanda CO2, RMSE).

Change vs train_v2.py (baseline OOF RMSE 22.6488, exp #4): apply empirical-Bayes
/ James-Stein-style SHRINKAGE to the location-week target encoding, blending the
(very noisy, n=2-per-fold) cell mean toward the location-level mean (n~106-per-fold):

    te_locweek_shrunk = alpha * cell_mean + (1 - alpha) * loc_mean

Rationale: the (lat,lon,week_no) panel is perfectly balanced (497 locations x 53
weeks x 3 years), so every fold's source (2 remaining years) gives EXACTLY n=2
observations per cell -- a highly noisy average given within-cell cross-year std
(median ~3.1 per EDA, but heavy-tailed). The location-level mean uses ~106 obs and
is much more stable. A pre-sweep (scratch, not GBDT) over alpha in [0.3, 1.0] on
the *same* LOYO CV found a clear minimum at alpha ~= 0.935 (OOF RMSE 22.4897 vs
22.6488 for alpha=1.0, i.e. no shrinkage / plain historical mean). This is
classic empirical-Bayes shrinkage on count-sparse group means.

Everything else (CV = Leave-One-Year-Out 2019/2020/2021, feature set, models,
blend search) is unchanged from train_v2.py so scores stay comparable.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

COMP = "competitions/playground-series-s3e20"
DATA, SUB, EXP = f"{COMP}/data", f"{COMP}/submissions", f"{COMP}/experiments.json"
TARGET, IDC = "emission", "ID_LAT_LON_YEAR_WEEK"
SEED = 42
ALPHA = 0.935  # shrinkage weight on cell mean (1-ALPHA on loc mean); tuned via pre-sweep (see notes)
os.makedirs(SUB, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
print(f"train {train.shape}  test {test.shape}")

SENSORS = [c for c in train.columns if c not in (IDC, TARGET, "latitude", "longitude", "year", "week_no")]
miss = train[SENSORS].isna().mean()
drop_cols = miss[miss > 0.90].index.tolist()
SENSORS = [c for c in SENSORS if c not in drop_cols]
print(f"dropped {len(drop_cols)} sensor cols with >90% missing; keep {len(SENSORS)} sensors")

sensor_median = train[SENSORS].median()


def base_features(df):
    X = pd.DataFrame(index=df.index)
    X["latitude"] = df["latitude"]
    X["longitude"] = df["longitude"]
    X["week_no"] = df["week_no"]
    X["year"] = df["year"]
    X["week_sin"] = np.sin(2 * np.pi * df["week_no"] / 53)
    X["week_cos"] = np.cos(2 * np.pi * df["week_no"] / 53)
    X["lat_lon"] = df["latitude"] * df["longitude"]
    for c in SENSORS:
        X[c] = df[c].fillna(sensor_median[c])
    return X


def add_target_enc(X, src_df, tgt_df, alpha=ALPHA):
    """Add shrunk location-week + location mean emission, learned from src_df, applied to tgt_df rows.
    te_locweek = alpha*cell_mean + (1-alpha)*loc_mean (empirical-Bayes shrinkage toward the
    less-noisy, higher-count location-level mean). te_loc unchanged from v2."""
    gmean = src_df[TARGET].mean()
    lw = src_df.groupby(["latitude", "longitude", "week_no"])[TARGET].mean()
    lo = src_df.groupby(["latitude", "longitude"])[TARGET].mean()
    key_lw = list(zip(tgt_df.latitude, tgt_df.longitude, tgt_df.week_no))
    key_lo = list(zip(tgt_df.latitude, tgt_df.longitude))
    lw_map = np.array([lw.get(k, np.nan) for k in key_lw])
    lo_map = np.array([lo.get(k, np.nan) for k in key_lo])
    lo_filled = np.where(np.isnan(lo_map), gmean, lo_map)
    # shrink cell mean toward loc mean; if cell missing, fall back straight to loc/global (as v2)
    lw_filled = np.where(np.isnan(lw_map), lo_filled, alpha * lw_map + (1 - alpha) * lo_filled)
    X = X.copy()
    X["te_locweek"] = lw_filled
    X["te_loc"] = lo_filled
    return X


y = train[TARGET].to_numpy(np.float64)
ylog = np.log1p(y)
years = train["year"].to_numpy()
YEARS = [2019, 2020, 2021]


def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5


def run_models(Xtr, ytr, Xva, Xtest):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    out = {}
    lgm = lgb.LGBMRegressor(objective="regression", metric="rmse", n_estimators=1500,
                            learning_rate=0.03, num_leaves=63, min_child_samples=30,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                            reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    lgm.fit(Xtr, ytr)
    out["LGB"] = (lgm.predict(Xva), lgm.predict(Xtest))
    xgm = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", n_estimators=1500,
                           learning_rate=0.03, max_depth=7, min_child_weight=5, subsample=0.8,
                           colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
                           random_state=SEED, n_jobs=-1, tree_method="hist")
    xgm.fit(Xtr, ytr)
    out["XGB"] = (xgm.predict(Xva), xgm.predict(Xtest))
    ctm = CatBoostRegressor(loss_function="RMSE", iterations=2000, learning_rate=0.04,
                            depth=8, l2_leaf_reg=3.0, random_seed=SEED, thread_count=-1, verbose=False)
    ctm.fit(Xtr, ytr)
    out["CAT"] = (ctm.predict(Xva), ctm.predict(Xtest))
    return out


Xtrain_base = base_features(train)
Xtest_base = base_features(test)
NAMES = ["LGB", "XGB", "CAT"]

oof = {n: np.full(len(train), np.nan) for n in NAMES}
test_pred_acc = {n: np.zeros(len(test)) for n in NAMES}
fold_scores = {n: [] for n in NAMES}
t0 = time.time()

for holdout in YEARS:
    tr_mask = years != holdout
    va_mask = years == holdout
    src = train[tr_mask]
    Xtr = add_target_enc(Xtrain_base[tr_mask], src, train[tr_mask])
    Xva = add_target_enc(Xtrain_base[va_mask], src, train[va_mask])
    Xte = add_target_enc(Xtest_base, src, test)
    res = run_models(Xtr.to_numpy(np.float32), ylog[tr_mask], Xva.to_numpy(np.float32), Xte.to_numpy(np.float32))
    for n in NAMES:
        va_pred_log, te_pred_log = res[n]
        va_pred = np.clip(np.expm1(va_pred_log), 0, None)
        oof[n][va_mask] = va_pred
        test_pred_acc[n] += np.clip(np.expm1(te_pred_log), 0, None) / len(YEARS)
        s = rmse(y[va_mask], va_pred)
        fold_scores[n].append(s)
        print(f"  holdout {holdout} [{n}] RMSE={s:.4f}")

print(f"\n--- per-model OOF RMSE (LOYO) --- [{time.time()-t0:.0f}s]")
for n in NAMES:
    print(f"  {n}: OOF={rmse(y, oof[n]):.4f}  folds={[round(s,3) for s in fold_scores[n]]}")

# --- shrunk location-week mean predictor (Round 1 change) ---
te_only = np.full(len(train), np.nan)
for holdout in YEARS:
    src = train[years != holdout]
    lw = src.groupby(["latitude", "longitude", "week_no"])[TARGET].mean()
    lo = src.groupby(["latitude", "longitude"])[TARGET].mean()
    g = src[TARGET].mean()
    sub = train[years == holdout]
    p_lw = np.array([lw.get(k, np.nan) for k in zip(sub.latitude, sub.longitude, sub.week_no)])
    p_lo = np.array([lo.get(k, np.nan) for k in zip(sub.latitude, sub.longitude)])
    p_lo_filled = np.where(np.isnan(p_lo), g, p_lo)
    p = np.where(np.isnan(p_lw), p_lo_filled, ALPHA * p_lw + (1 - ALPHA) * p_lo_filled)
    te_only[years == holdout] = p
print(f"shrunk loc-week-mean (alpha={ALPHA}) OOF RMSE = {rmse(y, te_only):.4f}")
print(f"  (reference, unshrunk v2 TE OOF RMSE = 22.6488)")

lw_all = train.groupby(["latitude", "longitude", "week_no"])[TARGET].mean()
lo_all = train.groupby(["latitude", "longitude"])[TARGET].mean()
g_all = train[TARGET].mean()
te_test_lw = np.array([lw_all.get(k, np.nan) for k in zip(test.latitude, test.longitude, test.week_no)])
te_test_lo = np.array([lo_all.get(k, np.nan) for k in zip(test.latitude, test.longitude)])
te_test_lo_filled = np.where(np.isnan(te_test_lo), g_all, te_test_lo)
te_test = np.where(np.isnan(te_test_lw), te_test_lo_filled, ALPHA * te_test_lw + (1 - ALPHA) * te_test_lo_filled)

BLEND = NAMES + ["TE"]
oof_all = {**oof, "TE": te_only}
test_all = {**test_pred_acc, "TE": te_test}
oofs = np.stack([oof_all[n] for n in BLEND], axis=1)

best_w, best_s = None, 1e18
grid = np.arange(0, 1.0001, 0.05)
for w0 in grid:
    for w1 in grid[grid <= 1 - w0 + 1e-9]:
        for w2 in grid[grid <= 1 - w0 - w1 + 1e-9]:
            w3 = 1 - w0 - w1 - w2
            if w3 < -1e-9:
                continue
            s = rmse(y, oofs @ np.array([w0, w1, w2, w3]))
            if s < best_s:
                best_s, best_w = s, (w0, w1, w2, w3)
print(f"\nBEST blend {dict(zip(BLEND, np.round(best_w,3)))}  OOF RMSE = {best_s:.4f}")

w = np.array(best_w)
final_test = np.stack([test_all[n] for n in BLEND], axis=1) @ w
final_test = np.clip(final_test, 0, None)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sub = pd.DataFrame({IDC: test[IDC], TARGET: final_test})
sub_path = f"{SUB}/sub_v3_blend_{best_s:.4f}_{stamp}.csv"
sub.to_csv(sub_path, index=False)
print(f"\nWrote {sub_path}  shape={sub.shape}")
print(sub[TARGET].describe())

log = dict(experiment_id=None, timestamp=datetime.now().isoformat(timespec="seconds"),
           stage="modeling_v3_round1_eb_shrinkage",
           model="LGB+XGB+CatBoost blend + EB-shrunk loc-week target encoding",
           features=f"{Xtrain_base.shape[1]} base ({len(SENSORS)} sensors, dropped {len(drop_cols)}) + te_locweek(shrunk alpha={ALPHA}) + te_loc",
           cv_strategy="Leave-One-Year-Out (2019/2020/2021), same as v2",
           change_vs_prev="empirical-Bayes shrinkage of te_locweek toward te_loc: te = alpha*cell_mean + (1-alpha)*loc_mean, alpha=0.935 (chosen via pre-sweep grid 0.3-1.0 on same LOYO CV, min at 0.935)",
           per_model_oof_rmse={n: round(rmse(y, oof[n]), 4) for n in NAMES},
           fold_scores={n: [round(s, 4) for s in fold_scores[n]] for n in NAMES},
           blend_members=BLEND,
           blend_weights=dict(zip(BLEND, [round(float(x), 3) for x in best_w])),
           blend_oof_rmse=round(best_s, 4),
           ref_locweek_mean_shrunk_rmse=round(rmse(y, te_only), 4),
           ref_prev_best_unshrunk_rmse=22.6488,
           delta_vs_prev_best=round(best_s - 22.6488, 4),
           notes="Round 1 of Phase-B iteration. Cells are perfectly balanced panel (497 loc x 53 wk x 3yr) so every LOYO fold's source has exactly n=2 obs/cell -- shrinking the noisy 2-obs cell mean toward the ~106-obs loc mean improves OOF. GBDTs still expected to get ~0 weight (structure-dominant target, per s3e20/s3e19 boundary-condition insight in knowledge/experience.md).",
           submission_file=os.path.basename(sub_path))
history = json.load(open(EXP)) if os.path.exists(EXP) else []
log["experiment_id"] = len(history) + 1
history.append(log)
json.dump(history, open(EXP, "w"), indent=2)
print(f"Logged experiment #{log['experiment_id']} to {EXP}")
