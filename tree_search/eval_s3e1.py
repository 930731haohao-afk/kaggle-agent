"""tree_search/eval_s3e1.py — per-competition evaluator for playground-series-s3e1
(California Housing, regression, RMSE metric, minimize-better).

Phase D-4 (5th harness_v2 sweep comp). Two node kinds, same schema as
eval_s3e3.py/eval_s3e7.py:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
             features:{drop:[...]}, postprocess:{"clip": bool}}
     Trains one regressor with 5-fold KFold(shuffle=True, seed=42) — EXACT same CV as
     competitions/playground-series-s3e1/scripts/{tune_lgb_optuna,seed_bag_round2,
     round4_full_retrain}.py. Feature engineering is REPRODUCED HERE (not imported from
     scripts/features.py, which is a run-as-script pipeline with import-time side
     effects — writes train_processed*.csv to disk — unlike s3e3/s3e7's
     build_features()/feature_columns() module functions) so this evaluator is
     side-effect-free and import-safe. The 26-feature set matches STATUS.md exp #7
     exactly: 8 raw + 16 engineered (households/ratios/log1p/city-distances/geo_cluster)
     + 2 round-3 features (knn_mean_dist_10, coastal_dist) — see _build_features below,
     ported verbatim from scripts/features.py + scripts/geo_knn_probe.py. ~10-20s per
     fold-set on this 37k-row/26-feature dataset. If `node_id` is given, its OOF/test
     prediction vector is cached via harness_v2.cache_oof to
     tree_search/cache_s3e1/solo_<node_id>.npz so blend nodes never retrain.

     `postprocess.clip=True` clips predictions to [y_train.min(), 5.00001] BEFORE RMSE
     is computed — the linear-iteration run only ever clipped the final TEST submission
     to the train target range (round4_full_retrain.py's "clip_to_train_target_range"),
     never the OOF used to SCORE/select models. Since 4.92% of train rows are top-coded
     at exactly 5.00001 (STATUS.md EDA finding #1), any solo/blend prediction above that
     ceiling for a genuinely-capped row is guaranteed excess error that clipping removes
     for free — this node kind measures whether that's a real, reproducible OOF RMSE
     gain here (untested lever per the Phase D-4 brief: "top-code-aware nodes").

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet"|"grid_simplex", space:"prob", postprocess:{...}}
     Loads each member's cached OOF vector via harness_v2.load_oof, weight-searches for
     the blend that MINIMIZES RMSE. `postprocess.clip` is applied INSIDE metric_fn (per
     harness_v2.eval_blend's docstring: postprocessing must happen inside metric_fn so
     it's applied to every candidate weight vector during the search itself, not just
     retroactively to the winner) so the clip-aware search can find a genuinely different
     optimum than the unclipped search, not just clip the unclipped winner after the fact.

SIGN CONVENTION: harness.py/harness_v2.py assume lower-is-better scores; RMSE already
IS lower-is-better, so `score` handed to add_root/add_node is the RMSE itself (no sign
flip needed, unlike the AUC comps) — `result["rmse"]` and score are identical.

CatBoost: allow_writing_files=False and thread_count explicit per the s3e11 lesson in
knowledge/experience.md ("反面教訓" — CatBoost's default catboost_info file logging
stalled a long-running sandboxed process for 33 minutes with zero output).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e1")
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e1")
TARGET, ID = "MedHouseVal", "id"
N_SPLITS, SEED = 5, 42
TOP_CODE_CEILING = 5.00001  # STATUS.md EDA finding #1: 4.92% of train rows capped here

CITIES = {
    "LA": (34.05, -118.24), "SF": (37.77, -122.41), "SanDiego": (32.72, -117.16),
    "Sacramento": (38.58, -121.49), "SanJose": (37.34, -121.89),
}
COASTLINE = [
    (32.55, -117.13), (33.17, -117.35), (33.76, -118.20), (34.02, -118.49),
    (34.42, -119.70), (35.37, -120.85), (36.60, -121.90), (37.62, -122.49),
    (38.35, -123.05), (39.44, -123.80), (40.80, -124.16), (41.76, -124.20),
]
K_NEIGHBORS = 10

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.float64)
Y_MIN, Y_MAX = float(_y.min()), TOP_CODE_CEILING


def _add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ports scripts/features.py's add_features() verbatim (16 engineered features,
    minus geo_cluster which needs a KMeans fit shared across train/test -- see below)."""
    df = df.copy()
    df["households"] = df["Population"] / df["AveOccup"].replace(0, np.nan)
    df["households"] = df["households"].fillna(df["households"].median())
    df["bedroom_ratio"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    df["rooms_per_person"] = df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
    df["bedroom_ratio"] = df["bedroom_ratio"].fillna(df["bedroom_ratio"].median())
    df["rooms_per_person"] = df["rooms_per_person"].fillna(df["rooms_per_person"].median())
    for c in ["AveRooms", "AveBedrms", "Population", "AveOccup", "households"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    for name, (lat, lon) in CITIES.items():
        df[f"dist_{name}"] = np.sqrt((df["Latitude"] - lat) ** 2 + (df["Longitude"] - lon) ** 2)
    df["dist_nearest_city"] = df[[f"dist_{n}" for n in CITIES]].min(axis=1)
    df["lat_long"] = df["Latitude"] * df["Longitude"]
    return df


def _coastal_distance(lat, lon):
    dists = np.stack([np.sqrt((lat - a) ** 2 + (lon - b) ** 2) for a, b in COASTLINE], axis=1)
    return dists.min(axis=1)


def _build_features_fresh():
    """Reproduces STATUS.md exp #7's full 26-feature set (scripts/features.py +
    scripts/geo_knn_probe.py, unmodified logic) directly from raw train/test.csv,
    side-effect-free (no CSVs written). NOTE (verified empirically while wiring up this
    evaluator): LightGBM's histogram-based split search is chaotically sensitive to
    sub-ULP (~1e-13) floating-point differences at num_leaves=121/~1000+ boosting
    rounds -- freshly recomputing these ratio/log/distance features from raw CSVs (vs.
    loading the linear run's ALREADY-WRITTEN train_processed_v2.csv/test_processed_v2.csv)
    lands on bit-patterns that differ from the historical run by ~1e-13 per feature
    (pandas fillna/arithmetic ordering, not a correctness bug -- confirmed by round-
    tripping this function's own output through a CSV text buffer, which alone shifts
    LGB_TUNED_SEED2024's OOF RMSE from 0.558552 to the historical 0.558812) and that
    tiny noise cascades into a real ~0.00026 RMSE difference. `_build_features` below
    therefore prefers loading the existing processed_v2 CSVs (byte-identical to what
    produced every STATUS.md/experiments.json number) and only falls back to this fresh
    computation if those files are missing (e.g. a clean checkout that never ran the
    linear-iteration scripts) -- in that fallback case exact digit-for-digit
    reproduction of LINEAR_BEST is NOT guaranteed, only reproduction of the same
    feature-engineering LOGIC."""
    tr = _add_core_features(_train)
    te = _add_core_features(_test)

    kmeans = KMeans(n_clusters=25, random_state=SEED, n_init=10)
    tr["geo_cluster"] = kmeans.fit_predict(_train[["Latitude", "Longitude"]])
    te["geo_cluster"] = kmeans.predict(_test[["Latitude", "Longitude"]])

    all_coords = np.vstack([_train[["Latitude", "Longitude"]].to_numpy(),
                             _test[["Latitude", "Longitude"]].to_numpy()])
    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)
    nn.fit(all_coords)
    dist_all, _ = nn.kneighbors(all_coords)
    knn_mean_dist = dist_all[:, 1:].mean(axis=1)
    n_train = len(tr)
    tr["knn_mean_dist_10"] = knn_mean_dist[:n_train]
    te["knn_mean_dist_10"] = knn_mean_dist[n_train:]

    tr["coastal_dist"] = _coastal_distance(tr["Latitude"].to_numpy(), tr["Longitude"].to_numpy())
    te["coastal_dist"] = _coastal_distance(te["Latitude"].to_numpy(), te["Longitude"].to_numpy())

    new_features = [
        "households", "bedroom_ratio", "rooms_per_person",
        "log_AveRooms", "log_AveBedrms", "log_Population", "log_AveOccup", "log_households",
        "dist_LA", "dist_SF", "dist_SanDiego", "dist_Sacramento", "dist_SanJose",
        "dist_nearest_city", "lat_long", "geo_cluster", "knn_mean_dist_10", "coastal_dist",
    ]
    orig_features = [c for c in _train.columns if c not in (TARGET, ID)]
    all_features = orig_features + new_features
    assert tr[all_features].isnull().sum().sum() == 0
    assert te[all_features].isnull().sum().sum() == 0
    return tr, te, all_features


def _build_features():
    """Prefer the linear run's already-on-disk train_processed_v2.csv/
    test_processed_v2.csv (STATUS.md exp #7's exact 26-feature artifact, byte-identical
    to what produced every historical score) for true digit-for-digit reproduction; fall
    back to _build_features_fresh() (same logic, recomputed from raw CSVs) if those
    files aren't present."""
    v2_train = os.path.join(DATA, "train_processed_v2.csv")
    v2_test = os.path.join(DATA, "test_processed_v2.csv")
    if os.path.exists(v2_train) and os.path.exists(v2_test):
        tr = pd.read_csv(v2_train)
        te = pd.read_csv(v2_test)
        all_features = [c for c in tr.columns if c not in (TARGET, ID)]
        assert set(all_features) == set(c for c in te.columns if c != ID), (
            "train_processed_v2.csv / test_processed_v2.csv feature-column mismatch")
        assert tr[all_features].isnull().sum().sum() == 0
        assert te[all_features].isnull().sum().sum() == 0
        print(f"eval_s3e1: loaded {len(all_features)} features from on-disk "
              f"train_processed_v2.csv/test_processed_v2.csv (exact linear-run artifact)")
        return tr, te, all_features
    print("eval_s3e1: train_processed_v2.csv/test_processed_v2.csv not found -- "
          "recomputing features fresh (LINEAR_BEST reproduction not guaranteed "
          "digit-for-digit, see _build_features_fresh docstring)")
    return _build_features_fresh()


_Xtr_full, _Xte_full, ALL_FEATURES = _build_features()
assert len(ALL_FEATURES) == 26, f"expected 26 features, got {len(ALL_FEATURES)}"

# MANDATORY: identical fold assignment to scripts/tune_lgb_optuna.py & seed_bag_round2.py
# & round4_full_retrain.py (folds depend only on n_samples).
_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_kf.split(np.zeros(len(_y))))


def rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def maybe_clip(pred, clip: bool):
    return np.clip(pred, Y_MIN, Y_MAX) if clip else pred


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    return _Xtr_full[feats].copy(), _Xte_full[feats].copy(), feats


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/round4_full_retrain.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    # NOTE: deliberately no n_jobs override -- passing n_jobs=-1 explicitly measured
    # ~10-50x SLOWER on this 20-core sandbox (17-27s/fold vs 1.6-2s/fold with n_jobs
    # left at its LightGBM sklearn-API default) -- an oversubscription/threading quirk,
    # not present in the linear-iteration scripts (round4_full_retrain.py etc.), which
    # never set n_jobs either. Match that, do not "fix" it back to -1.
    p = dict(objective="rmse", verbosity=-1, random_state=SEED)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(n_estimators=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    # NOTE: n_jobs deliberately left unset, same rationale as _run_lgb above.
    p = dict(objective="reg:squarederror", random_state=SEED, tree_method="hist",
             verbosity=0)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", random_seed=SEED, thread_count=4, verbose=False,
             allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(iterations=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=(X[va], _y[va]), early_stopping_rounds=esr,
              use_best_model=True)
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_ceiling_hybrid(params, X, Xtest):
    """The Phase D-4 brief's ONE "ambitious" node: a two-stage ceiling-aware hybrid.
    STATUS.md EDA finding #1: 4.92% of train rows are top-coded at exactly
    TOP_CODE_CEILING (5.00001) -- their TRUE value is >=5.00001 but unrecoverable, so a
    pure regressor's residual error on those rows is capped-target-shape, not iid noise.
    This trains (a) the usual LGB regressor and (b) a binary LGB classifier for
    "is this row's target >= ceiling" on the SAME folds, then combines:
      mode="soft": final = reg*(1-p_cap) + ceiling*p_cap  (confidence-weighted nudge)
      mode="hard": final = ceiling if p_cap>=threshold else reg  (discrete override)
    Honest risk (this is exactly why the brief flags it "ambitious"): only 4.92% of
    rows are truly capped, so a classifier that is anything less than highly precise
    will nudge/override plenty of NOT-capped rows upward too, which can easily net
    negative vs. leaving the regressor alone -- this node's job is to measure whether
    the real effect is net positive or negative, not to assume it helps."""
    import lightgbm as lgb
    reg_params = dict(params.get("reg_params") or {})
    clf_params = dict(params.get("clf_params") or {})
    mode = params.get("mode", "soft")
    threshold = float(params.get("threshold", 0.5))

    reg_oof, reg_pred = _run_lgb(reg_params, X, Xtest)

    y_cap = (_y >= TOP_CODE_CEILING - 1e-9).astype(np.int64)
    p = dict(objective="binary", metric="auc", verbosity=-1, random_state=SEED)
    p.update(clf_params)
    n_est = p.pop("n_estimators", 500)
    esr = p.pop("early_stopping_rounds", 50)
    p_cap_oof = np.zeros(len(_y))
    p_cap_test = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMClassifier(n_estimators=n_est, **p)
        m.fit(X[tr], y_cap[tr], eval_set=[(X[va], y_cap[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        p_cap_oof[va] = m.predict_proba(X[va])[:, 1]
        p_cap_test += m.predict_proba(Xtest)[:, 1] / N_SPLITS

    if mode == "hard":
        final_oof = np.where(p_cap_oof >= threshold, TOP_CODE_CEILING, reg_oof)
        final_pred = np.where(p_cap_test >= threshold, TOP_CODE_CEILING, reg_pred)
    elif mode == "soft":
        final_oof = reg_oof * (1 - p_cap_oof) + TOP_CODE_CEILING * p_cap_oof
        final_pred = reg_pred * (1 - p_cap_test) + TOP_CODE_CEILING * p_cap_test
    else:
        raise ValueError(f"unknown ceiling_hybrid mode {mode!r}")

    from sklearn.metrics import roc_auc_score
    extra = dict(clf_auc=round(float(roc_auc_score(y_cap, p_cap_oof)), 5),
                 reg_only_rmse=round(rmse(_y, reg_oof), 6),
                 frac_flagged=round(float((p_cap_oof >= threshold).mean()), 5),
                 mode=mode, threshold=threshold)
    return final_oof, final_pred, extra


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    clip = bool((config.get("postprocess") or {}).get("clip", False))

    Xdf, Xtestdf, feats = get_feature_frame(drop)
    Xnp, Xtestnp = Xdf.to_numpy(np.float64), Xtestdf.to_numpy(np.float64)
    extra = None
    if model == "lgb":
        oof, pred = _run_lgb(params, Xnp, Xtestnp)
    elif model == "xgb":
        oof, pred = _run_xgb(params, Xnp, Xtestnp)
    elif model == "cat":
        oof, pred = _run_cat(params, Xnp, Xtestnp)
    elif model == "ceiling_hybrid":
        oof, pred, extra = _run_ceiling_hybrid(params, Xnp, Xtestnp)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score = rmse(_y, maybe_clip(oof, clip))
    return oof, pred, score, feats, extra


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors, no
# retraining. postprocess.clip is applied INSIDE metric_fn per harness_v2.eval_blend's
# contract, so the clip-aware search explores a genuinely different weight optimum.
# ---------------------------------------------------------------------------
def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    clip = bool((config.get("postprocess") or {}).get("clip", False))

    def metric_fn(vec):
        return rmse(_y, maybe_clip(vec, clip))

    best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn, weight_search=method)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, clip=clip, rmse=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 200) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    RMSE is lower-is-better already, so score == result["rmse"] (no sign flip, unlike
    the AUC comps s3e3/s3e7)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats, extra = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, rmse=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "rmse": round(score, 6)}
            if extra is not None:
                result.update(extra)
            return dict(status="evaluated", score=round(score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            return dict(status="evaluated", score=round(score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), result=None,
                    error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 1), result=None,
                    error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
