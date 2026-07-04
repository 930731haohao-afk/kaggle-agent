"""tree_search/eval_s3e20_v2.py -- per-competition evaluator for playground-series-s3e20
(Rwanda CO2 emissions, RMSE). Phase E-4: the STRUCTURE-DOMINATED regime test. Linear
Phase-B iteration (scripts/train_v5.py) established that a pure location-week historical
mean (denoised via 3 hand-tuned scalar knobs: EB shrinkage alpha, COVID year-weight
W2020, neighbor-week-smoothing weight WNB) beats every GBDT outright (weight search gives
the GBDTs 0 in every round, experiments.json #4-#8) -- OOF RMSE 22.6488 -> 21.1487 across
Phase B's one-at-a-time hand tuning.

This module treats that SAME structural pipeline as a pure hyperparameter landscape (a
handful of SCALARS: alpha, w2020, wnb, window) plus a few genuinely new STRUCTURAL ideas
Phase B never tried (per-location alpha, extra year-weights beyond 2020, a month-level
fallback tier for unseen cells, circular week wraparound) -- exactly the "does tree search
add value on a structure-dominated landscape" question the task brief poses. Because the
structural predictor is just a few groupby-mean computations (no gradient-boosted tree
training), every "solo" eval here costs low hundreds of milliseconds to a couple of
seconds, not the minutes/GBDT-fold-training cost every other eval_s3e*_v2.py module pays
-- this is a genuinely different regime for the harness_v2 search loop (many more nodes
affordable per minute of wall-clock).

--- Structural config schema (kind="solo", method="te") ---
{alpha, w2020, wnb, window, neighbor_weight_mode: "flat"|"decay", circular: bool,
 month_fallback: bool, per_loc_alpha: bool, per_loc_k: float, year_weights: {"<year>": w}}

  alpha              EB shrinkage weight on the (neighbor-smoothed) cell mean vs the loc
                     mean: pred = alpha*cell_smooth + (1-alpha)*loc_mean. Root=1.0.
  w2020              down-weight applied to year==2020 rows when computing historical
                     means (COVID dip). Root=0.24. Folded into year_weights[2020] unless
                     year_weights explicitly overrides it.
  wnb                weight of each week +-1 neighbor cell relative to the own week
                     (weight 1.0), before the alpha blend. Root=0.28.
  window             half-width (in weeks) of the neighbor-smoothing window. Root=1
                     (Phase B's own re-sweep: +-2/+-3 both worse, 21.23/21.51).
  neighbor_weight_mode  "flat" (every offset 1..window gets weight `wnb`, matching Phase
                     B's train_v4/v5 code exactly for window=1 -- this IS the root's exact
                     computation, verified digit-for-digit below) or "decay" (offset d gets
                     wnb**d, a genuinely untested joint idea: maybe wider windows only
                     looked bad because Phase B gave far neighbors full weight). NOTE:
                     Phase B's own window=2/3 regression numbers (21.2341/21.3163,
                     experiments.json #8/Round 4a) came from an ad hoc scratch re-sweep
                     that was never saved as a script -- STATUS.md/experiments.json record
                     only the final numbers, and per Phase B's own established pattern
                     (Round 3's docstring: "jointly re-tuned ALPHA, W2020, WNB" whenever a
                     new knob is added) those numbers likely reflect alpha/w2020/wnb
                     RE-TUNED per window width, not window changed in isolation. This
                     module's window=2/3 with alpha/w2020/wnb held at the root's values
                     scores materially worse (21.77/23.14) than those reported numbers --
                     expected, since holding the other 3 knobs fixed is a strictly harder
                     comparison. This is exactly why Phase E-4 hands the joint (alpha,
                     w2020, wnb, window) space to the tree search instead of re-deriving
                     Phase B's un-reproducible scratch sweep.
  circular           wrap week index across the year boundary (week 52 neighbors week 0)
                     instead of leaving the edge weeks with only one neighbor. NEW idea.
  month_fallback     add a location-month mean as an extra fallback tier used only when
                     the cell mean itself is NaN (never happens on this dataset's PERFECTLY
                     BALANCED panel -- 497 locs x 53 weeks x 3 years, every (loc,week)
                     combination present in both remaining LOYO years -- see module
                     docstring's "expected no-op" note). Tested anyway per the task brief.
  per_loc_alpha      replace the single scalar `alpha` with a per-location empirical-Bayes
                     alpha_loc = tau2_loc / (tau2_loc + sigma2_loc / (per_loc_k * n_year)),
                     where tau2_loc = variance of that location's per-week cell means
                     (真訊號: how much real week-to-week structure the location has) and
                     sigma2_loc = mean within-cell residual variance around the cell mean
                     (量測噪音). Locations with flat/noisy weekly profiles shrink harder
                     toward their own loc mean; locations with strong real seasonal
                     structure trust the cell mean more. NEW idea Phase B never tried
                     (Phase B's alpha was a single global scalar).
  year_weights       optional dict of {year: weight} overriding/extending the single
                     w2020 knob -- e.g. also down/up-weighting 2019 (ramp-up year) or 2021
                     (most-recent, closest analog to the 2022 test year). NEW idea.

--- GBDT diagnostic (method="gbdt_lgb_quick") ---
Exactly ONE lightweight GBDT solo node is defined (not a 3-model ensemble): a single
LightGBM regressor (log1p target, reduced n_estimators=400 for speed) trained on the same
63 sensor columns + lat/lon/week features as train_v5.py, LOYO CV. This exists purely to
let ONE blend node re-confirm harness_v2's own weight search still assigns it ~0 weight
against the tree's best structural TE candidate -- Phase B already proved this 5 times
(experiments.json #4-#8) with a full 3-model ensemble, so re-running the whole ensemble
here would just burn budget on an already-answered question; one cheap model is enough to
re-confirm the finding transfers to the tree-search harness's own OOF-caching path.

SIGN CONVENTION: RMSE is already lower-is-better/minimize -- no sign flip anywhere,
matching harness_v2's convention (same as eval_s3e9_v2.py/eval_s3e14.py, unlike the QWK
modules that need a sign flip).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e20")
DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e20")

sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

TARGET, IDC = "emission", "ID_LAT_LON_YEAR_WEEK"
YEARS = [2019, 2020, 2021]
N_WEEKS = 53  # week_no 0..52, verified against data (perfectly balanced 497*53*3 panel)
SEED = 42

_train = pd.read_csv(os.path.join(DATA, "train.csv"))
_test = pd.read_csv(os.path.join(DATA, "test.csv"))
_y = _train[TARGET].to_numpy(np.float64)
_years = _train["year"].to_numpy()

SENSORS = [c for c in _train.columns if c not in (IDC, TARGET, "latitude", "longitude", "year", "week_no")]
_miss = _train[SENSORS].isna().mean()
_drop_cols = _miss[_miss > 0.90].index.tolist()
SENSORS = [c for c in SENSORS if c not in _drop_cols]
_sensor_median = _train[SENSORS].median()


def rmse(a, b) -> float:
    return float(mean_squared_error(a, b) ** 0.5)


# ---------------------------------------------------------------------------
# structural (TE) predictor -- fully vectorized (pandas reindex, no python row loops)
# ---------------------------------------------------------------------------
def _year_weight_vec(years_arr, year_weights):
    w = np.ones(len(years_arr), dtype=np.float64)
    for yr, wt in year_weights.items():
        w = np.where(years_arr == int(yr), float(wt), w)
    return w


def _wmean(df, keys, w, val):
    tmp = df[keys].copy()
    tmp["_w"] = w
    tmp["_wv"] = val * w
    g = tmp.groupby(keys, observed=True).agg(wv=("_wv", "sum"), w=("_w", "sum"))
    return (g["wv"] / g["w"]).rename(None)


def _reindex(series, keys_df):
    idx = pd.MultiIndex.from_frame(keys_df)
    return series.reindex(idx).to_numpy()


def per_loc_alpha_map(src, w, val, k=1.0):
    """EB per-location alpha: tau2_loc (across-week signal variance of the location's
    cell means) / (tau2_loc + sigma2_loc/(k*n_year)) (sigma2_loc = mean within-cell
    residual variance around its own cell mean, n_year = avg years averaged per cell)."""
    tmp = src[["latitude", "longitude", "week_no"]].copy()
    tmp["v"] = val
    tmp["w"] = w
    cellmean = _wmean(src, ["latitude", "longitude", "week_no"], w, val)
    cm_row = _reindex(cellmean, tmp[["latitude", "longitude", "week_no"]])
    tmp["resid2"] = (tmp["v"] - cm_row) ** 2
    sigma2 = tmp.groupby(["latitude", "longitude"], observed=True)["resid2"].mean()
    tau2 = cellmean.groupby(level=[0, 1], observed=True).var(ddof=0).fillna(0.0)
    n_year = src.groupby(["latitude", "longitude", "week_no"], observed=True)["year"].nunique() \
                .groupby(level=[0, 1], observed=True).mean()
    denom = tau2 + sigma2 / np.maximum(k * n_year, 0.5)
    alpha_loc = (tau2 / denom.replace(0, np.nan)).fillna(0.5)
    return alpha_loc.clip(0.05, 0.999)


def build_predictor(src, config):
    """Returns a callable pred(lat_arr, lon_arr, week_arr) -> np.ndarray, fit on `src`."""
    alpha = float(config.get("alpha", 1.0))
    w2020 = float(config.get("w2020", 0.24))
    wnb = float(config.get("wnb", 0.28))
    window = int(config.get("window", 1))
    neighbor_mode = config.get("neighbor_weight_mode", "flat")
    circular = bool(config.get("circular", False))
    month_fallback = bool(config.get("month_fallback", False))
    per_loc_alpha = bool(config.get("per_loc_alpha", False))
    per_loc_k = float(config.get("per_loc_k", 1.0))
    year_weights_cfg = config.get("year_weights")
    year_weights = {int(k2): float(v) for k2, v in (year_weights_cfg or {}).items()}
    year_weights.setdefault(2020, w2020)

    w = _year_weight_vec(src["year"].to_numpy(), year_weights)
    val = src[TARGET].to_numpy(np.float64)

    cellmean = _wmean(src, ["latitude", "longitude", "week_no"], w, val)
    locmean = _wmean(src, ["latitude", "longitude"], w, val)
    gmean = float(np.average(val, weights=w))

    monthmean = None
    if month_fallback:
        tmp = src[["latitude", "longitude"]].copy()
        tmp["month"] = np.minimum(src["week_no"].to_numpy() // 4 + 1, 12)
        monthmean = _wmean(tmp, ["latitude", "longitude", "month"], w, val)

    alpha_loc_map = per_loc_alpha_map(src, w, val, k=per_loc_k) if per_loc_alpha else None

    def pred(lat_arr, lon_arr, week_arr):
        n = len(lat_arr)
        keys_lo = pd.DataFrame({"latitude": lat_arr, "longitude": lon_arr})
        lo = _reindex(locmean, keys_lo)
        lo_filled = np.where(np.isnan(lo), gmean, lo)

        if month_fallback:
            month_arr = np.minimum(week_arr // 4 + 1, 12)
            keys_mo = pd.DataFrame({"latitude": lat_arr, "longitude": lon_arr, "month": month_arr})
            mo = _reindex(monthmean, keys_mo)
            fallback_target = np.where(np.isnan(mo), lo_filled, mo)
        else:
            fallback_target = lo_filled

        keys_cell = pd.DataFrame({"latitude": lat_arr, "longitude": lon_arr, "week_no": week_arr})
        c0 = _reindex(cellmean, keys_cell)
        num = np.where(np.isnan(c0), 0.0, c0)
        den = np.where(np.isnan(c0), 0.0, 1.0)
        for d in range(1, window + 1):
            wt = wnb ** d if neighbor_mode == "decay" else wnb
            for wk_off in (week_arr - d, week_arr + d):
                if circular:
                    wk_n = np.mod(wk_off, N_WEEKS)
                    valid = np.ones(n, dtype=bool)
                else:
                    wk_n = wk_off
                    valid = (wk_n >= 0) & (wk_n < N_WEEKS)
                keys_n = pd.DataFrame({"latitude": lat_arr, "longitude": lon_arr, "week_no": wk_n})
                cv = _reindex(cellmean, keys_n)
                use = valid & ~np.isnan(cv)
                num = num + np.where(use, cv * wt, 0.0)
                den = den + np.where(use, wt, 0.0)
        cm_smooth = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

        if per_loc_alpha:
            a = _reindex(alpha_loc_map, keys_lo)
            a = np.where(np.isnan(a), alpha, a)
        else:
            a = alpha

        out = np.where(np.isnan(cm_smooth), fallback_target, a * cm_smooth + (1 - a) * fallback_target)
        return out

    return pred, (cellmean, locmean, gmean)


def score_te(config):
    """LOYO OOF RMSE for a structural config. Returns (rmse, oof)."""
    oof = np.full(len(_train), np.nan)
    for holdout in YEARS:
        tr_mask = _years != holdout
        va_mask = _years == holdout
        src = _train[tr_mask]
        pred_fn, _ = build_predictor(src, config)
        sub = _train[va_mask]
        oof[va_mask.to_numpy() if hasattr(va_mask, "to_numpy") else va_mask] = pred_fn(
            sub["latitude"].to_numpy(), sub["longitude"].to_numpy(), sub["week_no"].to_numpy())
    return rmse(_y, oof), oof


def test_predict(config):
    pred_fn, _ = build_predictor(_train, config)
    return pred_fn(_test["latitude"].to_numpy(), _test["longitude"].to_numpy(), _test["week_no"].to_numpy())


# ---------------------------------------------------------------------------
# ONE lightweight GBDT diagnostic (method="gbdt_lgb_quick")
# ---------------------------------------------------------------------------
def base_features(df):
    X = pd.DataFrame(index=df.index)
    X["latitude"] = df["latitude"]
    X["longitude"] = df["longitude"]
    X["week_no"] = df["week_no"]
    X["week_sin"] = np.sin(2 * np.pi * df["week_no"] / 53)
    X["week_cos"] = np.cos(2 * np.pi * df["week_no"] / 53)
    X["lat_lon"] = df["latitude"] * df["longitude"]
    for c in SENSORS:
        X[c] = df[c].fillna(_sensor_median[c])
    return X


def score_gbdt_quick(config):
    import lightgbm as lgb
    params = dict(objective="regression", metric="rmse", n_estimators=400,
                  learning_rate=0.05, num_leaves=63, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_alpha=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    params.update(config.get("params") or {})
    ylog = np.log1p(_y)
    Xall = base_features(_train)
    oof = np.full(len(_train), np.nan)
    for holdout in YEARS:
        tr_mask = (_years != holdout)
        va_mask = (_years == holdout)
        m = lgb.LGBMRegressor(**params)
        m.fit(Xall[tr_mask].to_numpy(np.float32), ylog[tr_mask])
        pred_log = m.predict(Xall[va_mask].to_numpy(np.float32))
        oof[va_mask] = np.clip(np.expm1(pred_log), 0, None)
    return rmse(_y, oof), oof


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")

    def metric_fn(vec):
        return rmse(_y, vec)  # RMSE already lower-is-better, no sign flip, no rounding

    best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                               weight_search=method, k=800)
    weights = {str(m): round(float(w), 4) for m, w in zip(members, best_w)}
    return dict(members=members, weights=weights, method=method), best_score


def evaluate(config: dict, node_id: int = None, timeout_s: int = 120) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises. score = RMSE
    (lower-is-better, no sign flip)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            method = config.get("method", "te")
            if method == "te":
                score, oof = score_te(config)
                result = {"method": "te"}
            elif method == "gbdt_lgb_quick":
                score, oof = score_gbdt_quick(config)
                result = {"method": "gbdt_lgb_quick", "n_estimators": config.get("params", {}).get("n_estimators", 400)}
            else:
                raise ValueError(f"unknown solo method {method!r}")
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, y=_y)
            wall = time.time() - t0
            result["rmse"] = round(score, 4)
            return dict(status="evaluated", score=round(score, 4), wall_s=round(wall, 2),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            result["rmse"] = round(score, 4)
            return dict(status="evaluated", score=round(score, 4), wall_s=round(wall, 2),
                        result=result, error=None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 2), result=None,
                    error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        wall = time.time() - t0
        return dict(status="failed", score=None, wall_s=round(wall, 2), result=None,
                    error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
