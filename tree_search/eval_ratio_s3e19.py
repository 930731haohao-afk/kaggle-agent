"""tree_search/eval_s3e19.py — per-competition evaluator for playground-series-s3e19
(Forecast Mini-course Sales, regression, SMAPE metric, minimize-better).

Phase D-5 (the TIME-SERIES sweep comp). Two node kinds, same schema convention as
eval_s3e1.py/eval_s3e3.py/eval_s3e7.py:

  1. solo  — {kind:"solo", model:"lgb"|"cat", params:{...}, features:{drop:[...]},
             postprocess:{"scale": float} | {"auto_scale": true}}
     Trains one regressor on log1p(num_sold), predictions inverted with expm1 before
     SMAPE scoring — EXACT same CV as competitions/playground-series-s3e19/scripts/
     {train,train_seedbag2,train_optuna}.py: TimeSeriesSplit(n_splits=5) sliced on the
     1826 UNIQUE dates (not row-level), expanding-window train / strictly-later
     validation block per fold. Feature frame is loaded DIGIT-EXACTLY from the linear
     run's own on-disk train_processed.csv/test_processed.csv (scripts/features.py's
     already-written artifact) rather than recomputed, so every base-model score this
     evaluator produces is byte-identical to STATUS.md/experiments.json's numbers
     (verified empirically before this driver started searching — see run_s3e19.py's
     docstring for the digit-for-digit reproduction check). Falls back to recomputing
     the same feature-engineering LOGIC via scripts/features.py's pure functions
     (add_calendar_features/encode_categoricals — NOT build(), which has the disk-write
     side effect) only if those processed CSVs are missing.

     ⚠️ CRITICAL OOF-index subtlety (the task brief's own warning): TimeSeriesSplit on
     1826 unique dates with n_splits=5 produces 5 folds of ~304 validation dates each,
     covering the LAST 5*304=1520 dates (114000 of 136950 rows); the FIRST ~306 dates
     (22950 rows) are only ever a training block, never a validation block, in ANY
     fold, so they have no OOF prediction. `IDX` (module-level, computed once from the
     union of all 5 folds' validation masks) is the ONLY set of rows every score in
     this module is computed over — this exactly matches scripts/train.py's own
     `fold_mask_any` mask (verified: 114000 covered / 22950 uncovered, matching the
     5*304*75-series arithmetic). Solo OOF arrays are cached at FULL length
     (len(train)=136950) with zeros outside IDX so `harness_v2.eval_blend`'s
     `oofs @ w` matrix-multiply works unmodified across solo members; every metric_fn
     in this module slices by IDX before calling `smape` so blend weight search is
     scored on the IDENTICAL row set the linear run used, never on the padded zeros.

     `postprocess.auto_scale=True` is this run's untried "global multiplier" lever
     (task brief: "a global multiplier node is cheap and linear never tried it") — a
     SEPARATE, more honest idea than the already-REJECTED ratio-decomposition
     (STATUS.md Round 1, RD weight-searched to 0): auto_scale does a small 1-D grid
     search over a flat scalar multiplier applied to the WHOLE prediction vector,
     picking whichever multiplier minimizes OOF SMAPE — no country/date-conditional
     structure at all, just testing whether the blend/solo output is systematically
     mis-leveled. Applied INSIDE the scoring path (both solo and blend) so it's fairly
     compared against the unscaled baseline at every candidate.

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet"|"grid_simplex", postprocess:{...}}
     Loads each member's cached OOF vector via harness_v2.load_oof, weight-searches for
     the blend that MINIMIZES SMAPE on the IDX-masked rows. `postprocess.auto_scale` is
     applied INSIDE metric_fn (per harness_v2.eval_blend's contract: postprocessing
     must happen inside metric_fn so it's applied to every candidate weight vector
     during the search itself, not just retroactively to the winner) — the grid search
     over scale happens for EVERY candidate weight vector, so the weight search itself
     is scale-aware, not just the final winner.

SIGN CONVENTION: harness.py/harness_v2.py assume lower-is-better scores; SMAPE already
IS lower-is-better, so `score` handed to add_root/add_node is the SMAPE itself (no sign
flip needed, same as the RMSE comps s3e1/s3e14, unlike the AUC comps s3e3/s3e7).

CatBoost: allow_writing_files=False and thread_count=4 explicit per the s3e11 lesson in
knowledge/experience.md ("反面教訓" — CatBoost's default catboost_info file logging
stalled a long-running sandboxed process for 33 minutes with zero output) — measured
empirically here too: thread_count=4 CatBoost fold-5 fit (largest fold, ~114k train
rows) took ~25s, well within budget; thread_count=-1 is NOT used (oversubscription
risk, same rationale as eval_s3e1.py's n_jobs note for LGB/XGB, applied defensively to
CatBoost as well even though it wasn't directly measured to hang here).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e19-v5-ratio")
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_ratio_s3e19")
TARGET, ID = "num_sold", "id"
N_SPLITS, SEED = 5, 42
N_THREADS = 4      # matches CatBoost's existing thread_count=4; env-pinned below to avoid
                   # BLAS oversubscription now that LightGBM also pins its thread count

import os as _os  # noqa: E402
_os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
_os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
_os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))

FEATURE_COLS = [
    "year", "month", "day", "dow", "day_of_year", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end", "is_new_year",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "country_cat", "store_cat", "product_cat",
]
FEATURE_COLS = FEATURE_COLS + ["year_c", "is_holiday"]  # v5 operator columns
CAT_FEATURES_ALL = ["country_cat", "store_cat", "product_cat"]

SCALE_GRID = np.round(np.arange(0.85, 1.201, 0.01), 3)  # auto_scale search grid


# ---------------------------------------------------------------------------
# data loading — prefer on-disk train_processed.csv/test_processed.csv (byte-identical
# to what produced every STATUS.md/experiments.json number); fall back to recomputing
# the same feature-engineering LOGIC via scripts/features.py's pure functions (no disk
# writes) if those files are missing.
# ---------------------------------------------------------------------------
def _load_processed():
    tr_path = os.path.join(DATA, "train_processed.csv")
    te_path = os.path.join(DATA, "test_processed.csv")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = pd.read_csv(tr_path, parse_dates=["date"])
        te = pd.read_csv(te_path, parse_dates=["date"])
        print(f"eval_s3e19: loaded processed CSVs from disk (exact linear-run artifact), "
              f"train={tr.shape} test={te.shape}")
        return tr, te
    print("eval_s3e19: train_processed.csv/test_processed.csv not found -- recomputing "
          "features fresh via scripts/features.py's pure functions (digit-for-digit "
          "reproduction of LINEAR_BEST not guaranteed, only same logic)")
    sys.path.insert(0, os.path.join(_COMP_DIR, "scripts"))
    import features as feat_mod  # noqa: E402
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
    te_raw = pd.read_csv(os.path.join(DATA, "test.csv"))
    tr = feat_mod.add_calendar_features(tr_raw)
    te = feat_mod.add_calendar_features(te_raw)
    tr, te = feat_mod.encode_categoricals(tr, te)
    tr["log_num_sold"] = np.log1p(tr["num_sold"])
    return tr, te


_train, _test = _load_processed()
for _c in CAT_FEATURES_ALL:
    _train[_c] = _train[_c].astype("category")
    _test[_c] = _test[_c].astype("category")

_y = _train[TARGET].to_numpy(np.float64)

# ---- v5 target transform (ratio_log) -------------------------------------------------
# Fit on the covariate-normalized target so the test-period LEVEL comes from the
# covariate instead of from a piecewise-constant model that cannot extrapolate.
_COV_TR = _train["cov_level"].to_numpy(np.float64)
_COV_TE = _test["cov_level"].to_numpy(np.float64)
assert np.all(_COV_TR > 0) and np.all(_COV_TE > 0), "covariate must be strictly positive"
_Y_RATIO = np.log(_train[TARGET].to_numpy(np.float64) / _COV_TR)


def _invert_va(p, va_mask):
    return np.exp(p) * _COV_TR[va_mask]


def _invert_test(p):
    return np.exp(p) * _COV_TE
# -----------------------------------------------------------------------------------
_y_log = _train["log_num_sold"].to_numpy(np.float64) if "log_num_sold" in _train.columns \
    else np.log1p(_y)
assert np.allclose(_y_log, np.log1p(_y), atol=1e-9), "log_num_sold != log1p(num_sold)"


# ---------------------------------------------------------------------------
# MANDATORY: identical TimeSeriesSplit fold assignment to scripts/train.py's
# get_time_folds() -- TimeSeriesSplit(n_splits=5) sliced on the 1826 SORTED UNIQUE
# dates, then each fold's train/val DATE SETS are broadcast back to row masks via
# isin(). Verified digit-for-digit against scripts/train.py before this evaluator
# started being used for search: 5 folds x 304 val dates x 75 series = 22800 val rows
# each, union (IDX) = 114000 covered / 22950 uncovered (first ~306 dates, train-only
# in every fold).
# ---------------------------------------------------------------------------
def _build_folds():
    unique_dates = np.sort(_train["date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []
    for tr_idx, va_idx in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr_idx])
        va_dates = set(unique_dates[va_idx])
        tr_mask = _train["date"].isin(tr_dates).to_numpy()
        va_mask = _train["date"].isin(va_dates).to_numpy()
        folds.append((tr_mask, va_mask))
    return folds


_FOLDS = _build_folds()
IDX = np.zeros(len(_train), dtype=bool)
for _tr_mask, _va_mask in _FOLDS:
    IDX |= _va_mask
assert IDX.sum() == 114000, f"expected 114000 OOF-covered rows, got {IDX.sum()}"
N_OOF = int(IDX.sum())


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    ratio = np.where(denom == 0, 0.0, diff / denom)
    return float(100.0 * np.mean(ratio))


_Y_OOF_TRUE = _y[IDX]  # cached once -- every score in this module compares against this


def best_scale(pred_full: np.ndarray):
    """Grid-search the flat scalar multiplier (SCALE_GRID) that minimizes IDX-masked
    SMAPE for this prediction vector. Returns (best_scale, best_score)."""
    best_s, best_sc = None, None
    base = pred_full[IDX]
    for sc in SCALE_GRID:
        s = smape(_Y_OOF_TRUE, base * sc)
        if best_s is None or s < best_s:
            best_s, best_sc = s, sc
    return float(best_sc), float(best_s)


def maybe_postprocess(pred_full: np.ndarray, postprocess: dict):
    """Applies a FIXED scale (postprocess["scale"], default 1.0) OR, if
    postprocess.get("auto_scale") is truthy, grid-searches the best scale on the fly.
    Returns (scored_full_vector_at_IDX_only_is_meaningful, score, scale_used)."""
    pp = postprocess or {}
    if pp.get("auto_scale"):
        sc, _ = best_scale(pred_full)
    else:
        sc = float(pp.get("scale", 1.0))
    scored = pred_full[IDX] * sc
    return float(smape(_Y_OOF_TRUE, scored)), sc


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature selection
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(FEATURE_COLS)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in FEATURE_COLS if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    cat_feats = [c for c in CAT_FEATURES_ALL if c in feats]
    return _train[feats], _test[feats], feats, cat_feats


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/train.py / scripts/train_seedbag2.py exactly,
# generalized over an arbitrary params dict and feature subset)
# ---------------------------------------------------------------------------
def _run_lgb(params, Xdf, Xtestdf):
    import lightgbm as lgb
    # Determinism pinned 2026-07-30 (readiness-audit finding): LGBM's multi-threaded
    # histogram build is not reproducible across processes by default, which can flip
    # champion selection on a 2e-5 difference (scores are stored to 6 dp). N_THREADS
    # matches the env pinning at module load so no oversubscription risk is reintroduced.
    p = dict(objective="rmse", verbosity=-1, random_state=SEED,
             deterministic=True, force_row_wise=True, num_threads=N_THREADS)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_train))
    pred = np.zeros(len(Xtestdf))
    for tr_mask, va_mask in _FOLDS:
        X_tr, X_va = Xdf[tr_mask], Xdf[va_mask]
        y_tr, y_va = _Y_RATIO[tr_mask], _Y_RATIO[va_mask]
        m = lgb.LGBMRegressor(n_estimators=n_est, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va_mask] = _invert_va(m.predict(X_va), va_mask)
        pred += _invert_test(m.predict(Xtestdf)) / N_SPLITS
    return oof, pred


def _run_cat(params, Xdf, Xtestdf, cat_feats):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", random_seed=SEED, thread_count=4, verbose=False,
             allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_train))
    pred = np.zeros(len(Xtestdf))
    for tr_mask, va_mask in _FOLDS:
        X_tr, X_va = Xdf[tr_mask], Xdf[va_mask]
        y_tr, y_va = _Y_RATIO[tr_mask], _Y_RATIO[va_mask]
        m = CatBoostRegressor(iterations=n_est, cat_features=cat_feats, **p)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=esr)
        oof[va_mask] = _invert_va(m.predict(X_va), va_mask)
        pred += _invert_test(m.predict(Xtestdf)) / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    postprocess = config.get("postprocess") or {}

    Xdf, Xtestdf, feats, cat_feats = get_feature_frame(drop)
    if model == "lgb":
        oof, pred = _run_lgb(params, Xdf, Xtestdf)
    elif model == "cat":
        oof, pred = _run_cat(params, Xdf, Xtestdf, cat_feats)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score, scale_used = maybe_postprocess(oof, postprocess)
    extra = dict(n_feats=len(feats), scale_used=round(scale_used, 3))
    return oof, pred, score, feats, extra


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors, no
# retraining. postprocess.auto_scale is applied INSIDE metric_fn per harness_v2.
# eval_blend's contract, so the scale-aware search explores a genuinely different
# weight optimum than the unscaled search, not just a post-hoc rescale of the winner.
# ---------------------------------------------------------------------------
def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    postprocess = config.get("postprocess") or {}

    def metric_fn(vec_full):
        s, _ = maybe_postprocess(vec_full, postprocess)
        return s

    best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn, weight_search=method)
    blended = oofs @ best_w
    _, scale_used = maybe_postprocess(blended, postprocess)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, scale_used=round(scale_used, 3),
                smape=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 200) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    SMAPE is lower-is-better already, so score == result["smape"] (no sign flip, same
    convention as the RMSE comps)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, _feats, extra = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, smape=score)
            wall = time.time() - t0
            result = {"n_feats": extra["n_feats"], "smape": round(score, 6),
                      "scale_used": extra["scale_used"]}
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
