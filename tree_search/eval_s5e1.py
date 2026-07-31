"""tree_search/eval_s5e1.py — per-competition evaluator for playground-series-s5e1
(Forecasting Sticker Sales, regression, MAPE metric, minimize-better).

Fourth competition in the TASK-TS-FUTURE firing class. Deliberately mirrors eval_s3e19.py's
structure — same node schema, same TimeSeriesSplit-on-unique-dates CV, same log1p target and
expm1 inversion — because s5e1 is the same synthetic country x store x product panel and its
v5 arms are meant to be read side by side with s3e19's. Three things genuinely differ, each
forced by this competition rather than chosen:

  1. METRIC IS MAPE, NOT SMAPE. MAPE divides by the actual, so unlike SMAPE it is unbounded
     and asymmetric in the direction most people do not expect: over-prediction is punished
     harder than under-prediction of the same absolute size, and the optimal point forecast
     sits below the conditional mean. It also weights low-volume rows enormously — these 90
     series span two orders of magnitude (Norway's yearly total ~7.3M against Kenya's ~74k),
     so a 2-unit error on a Kenya row (typical value 5-11) costs 20-40% while the same error
     on a Norway row (~180) costs ~1%. Division by zero is not a risk: the observed minimum
     of any non-null target is exactly 5.0.

  2. TRUNCATED TARGETS ARE ALREADY GONE. scripts/features.py drops the 8871 null-target rows
     (3.85%) before writing train_processed.csv, because they are a low-value truncation floor
     rather than missing-at-random data — see that module's docstring for the evidence and for
     why they are dropped rather than imputed. This evaluator therefore loads 221259 rows, not
     230130. The treatment is in the feature builder precisely so every arm inherits it
     identically: no operator changes WHICH rows are null (null/covariate is still null), so
     the same rows are absent from the baseline, the join_feature arm and the ratio_target arm.

  3. THREE-YEAR TEST HORIZON. Test is 2017-2019 against train 2010-2016, where s3e19 and
     sep-2022 both extrapolate a single year. This is the longest horizon in the firing class
     and is why this competition is the sharpest test of the operator-form question.

NODE SCHEMA (identical to eval_s3e19.py):

  1. solo  — {kind:"solo", model:"lgb"|"cat", params:{...}, features:{drop:[...]},
             postprocess:{...}}
  2. blend — {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"|"nnls"}

OOF-index subtlety, same as s3e19: TimeSeriesSplit(n_splits=5) over the unique dates leaves
the earliest block as training-only in every fold, so it never receives an OOF prediction.
`IDX` is the union of all folds' validation masks and is the ONLY row set any score here is
computed over. Its size is derived from the folds rather than hardcoded — eval_s3e19.py
originally asserted a literal 114000, which is exactly the kind of competition-specific
constant that breaks when the file is reused.
"""
from __future__ import annotations

import os
import signal
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s5e1")
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402,F401  (imported for schema parity with the other evaluators)
import eval_support as esup  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s5e1")
TARGET, ID = "num_sold", "id"
N_SPLITS, SEED = 5, 42
# Capabilities the seed driver may rely on (tree_search/seed_from_ledger.py):
SUPPORTS = {"per_fold_target_encoding": True, "linear_family": True}
N_THREADS = 4      # matches CatBoost's thread_count=4; env-pinned below so LightGBM's
                   # determinism pin does not reintroduce BLAS oversubscription

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
CAT_FEATURES_ALL = ["country_cat", "store_cat", "product_cat"]

SCALE_GRID = np.round(np.arange(0.85, 1.201, 0.01), 3)  # auto_scale search grid


# ---------------------------------------------------------------------------
# data loading — prefer the on-disk processed CSVs (the artifact every reported number
# comes from); fall back to recomputing the same LOGIC via scripts/features.py's pure
# functions, never build(), which writes to disk.
# ---------------------------------------------------------------------------
def _load_processed():
    tr_path = os.path.join(DATA, "train_processed.csv")
    te_path = os.path.join(DATA, "test_processed.csv")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = pd.read_csv(tr_path, parse_dates=["date"])
        te = pd.read_csv(te_path, parse_dates=["date"])
        print(f"eval_s5e1: loaded processed CSVs from disk, train={tr.shape} test={te.shape}")
        return tr, te
    print("eval_s5e1: processed CSVs not found -- recomputing features via "
          "scripts/features.py's pure functions")
    sys.path.insert(0, os.path.join(_COMP_DIR, "scripts"))
    import features as feat_mod  # noqa: E402
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
    te_raw = pd.read_csv(os.path.join(DATA, "test.csv"))
    tr = feat_mod.add_calendar_features(tr_raw)
    te = feat_mod.add_calendar_features(te_raw)
    tr, te = feat_mod.encode_categoricals(tr, te)
    tr, _ = feat_mod.drop_truncated(tr)        # same treatment as the on-disk path
    tr["log_num_sold"] = np.log1p(tr[TARGET])
    return tr, te


_train, _test = _load_processed()
for _c in CAT_FEATURES_ALL:
    _train[_c] = _train[_c].astype("category")
    _test[_c] = _test[_c].astype("category")

_y = _train[TARGET].to_numpy(np.float64)
_y_log = _train["log_num_sold"].to_numpy(np.float64) if "log_num_sold" in _train.columns \
    else np.log1p(_y)
assert np.allclose(_y_log, np.log1p(_y), atol=1e-9), "log_num_sold != log1p(num_sold)"
assert not np.isnan(_y).any(), (
    "null targets reached the evaluator; scripts/features.py is supposed to have dropped "
    "them, and every arm depends on that happening identically")
assert _y.min() >= 5.0, f"observed target below the 5.0 truncation floor: {_y.min()}"


# ---------------------------------------------------------------------------
# folds: TimeSeriesSplit over the UNIQUE dates, not rows -- 90 series share each date, so a
# row-level split would put the same day on both sides of the boundary.
# ---------------------------------------------------------------------------
def _build_folds():
    unique_dates = np.sort(_train["date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []
    for tr_idx, va_idx in tscv.split(unique_dates):
        tr_dates = set(unique_dates[tr_idx])
        va_dates = set(unique_dates[va_idx])
        folds.append((_train["date"].isin(tr_dates).to_numpy(),
                      _train["date"].isin(va_dates).to_numpy()))
    return folds


_FOLDS = _build_folds()
IDX = np.zeros(len(_train), dtype=bool)
for _tr_mask, _va_mask in _FOLDS:
    IDX |= _va_mask
N_OOF = int(IDX.sum())
# derived, not hardcoded: the earliest date block is training-only in every fold and so has
# no OOF prediction. Asserting a literal row count here is what made eval_s3e19.py
# non-reusable.
_expected = sum(int(m.sum()) for _, m in _FOLDS)
assert N_OOF == _expected, f"fold validation masks overlap: {N_OOF} != {_expected}"
assert 0 < N_OOF < len(_train), f"implausible OOF coverage {N_OOF}/{len(_train)}"
print(f"eval_s5e1: {len(_FOLDS)} folds, OOF covers {N_OOF}/{len(_train)} rows "
      f"({N_OOF / len(_train) * 100:.1f}%)")


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, in percent. The competition's official metric.

    No zero-guard branch is needed on this data (minimum observed actual is 5.0) but one is
    kept anyway: a silent divide-by-zero would surface as inf/nan deep inside a search rather
    than here.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true)
    ratio = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom))
    return float(100.0 * np.mean(ratio))


_Y_OOF_TRUE = _y[IDX]  # cached once -- every score in this module compares against this


def best_scale(pred_full: np.ndarray):
    """Grid-search the flat multiplier minimizing IDX-masked MAPE. Returns (scale, score)."""
    best_s, best_sc = None, None
    base = pred_full[IDX]
    for sc in SCALE_GRID:
        s = mape(_Y_OOF_TRUE, base * sc)
        if best_s is None or s < best_s:
            best_s, best_sc = s, sc
    return float(best_sc), float(best_s)


def maybe_postprocess(pred_full: np.ndarray, postprocess: dict):
    """Scale, then clip, then round -- all optional, all inside the metric so every fold and
    every blend candidate is scored on the post-processed vector.

    Accepts both vocabularies: this module's own `auto_scale`/`scale`, and the dossier
    operator contract's `global_scale`/`round_to_int`/`clip_min`/`clip_max`. Accepting only
    the former is the mismatch that silently dropped s3e19's requested post-processing until
    2026-07-30; see knowledge/injection_operators.md.
    """
    pp = postprocess or {}
    gs = pp.get("global_scale")
    if gs == "auto" or pp.get("auto_scale"):
        sc, _ = best_scale(pred_full)
    elif gs is not None:
        sc = float(gs)
    else:
        sc = float(pp.get("scale", 1.0))
    scored = pred_full[IDX] * sc
    if pp.get("clip_min") is not None or pp.get("clip_max") is not None:
        scored = np.clip(scored, pp.get("clip_min"), pp.get("clip_max"))
    if pp.get("round_to_int"):
        scored = np.round(scored)
    return float(mape(_Y_OOF_TRUE, scored)), sc


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature frame
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    feats = [c for c in FEATURE_COLS if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    cat_feats = [c for c in CAT_FEATURES_ALL if c in feats]
    return _train[feats], _test[feats], feats, cat_feats


# ---------------------------------------------------------------------------
# solo-model runners
# ---------------------------------------------------------------------------
def _run_lgb(params, Xdf, Xtestdf, te_cfg=None, cat_feats=None):
    import lightgbm as lgb
    # Determinism pinned from the start here (eval_s3e19.py had to be retrofitted): LGBM's
    # multi-threaded histogram build is not reproducible across processes by default, which
    # can flip champion selection on a 2e-5 difference when scores are stored to 6 dp.
    p = dict(objective="rmse", verbosity=-1, random_state=SEED,
             deterministic=True, force_row_wise=True, num_threads=N_THREADS)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_train))
    pred = np.zeros(len(Xtestdf))
    for tr_mask, va_mask in _FOLDS:
        X_tr, X_va = Xdf[tr_mask], Xdf[va_mask]
        Xte_fold = Xtestdf
        if te_cfg:
            X_tr, X_va, Xte_fold = esup.per_fold_target_encode(
                X_tr, X_va, Xtestdf, _y_log[tr_mask],
                te_cfg.get("columns"), te_cfg.get("smoothing", esup.DEFAULT_SMOOTHING),
                cat_fallback=cat_feats)
        y_tr, y_va = _y_log[tr_mask], _y_log[va_mask]
        m = lgb.LGBMRegressor(n_estimators=n_est, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va_mask] = np.expm1(m.predict(X_va))
        pred += np.expm1(m.predict(Xte_fold)) / N_SPLITS
    return oof, pred


def _run_cat(params, Xdf, Xtestdf, cat_feats, te_cfg=None):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", random_seed=SEED, thread_count=N_THREADS, verbose=False,
             allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_train))
    pred = np.zeros(len(Xtestdf))
    for tr_mask, va_mask in _FOLDS:
        X_tr, X_va = Xdf[tr_mask], Xdf[va_mask]
        Xte_fold = Xtestdf
        if te_cfg:
            X_tr, X_va, Xte_fold = esup.per_fold_target_encode(
                X_tr, X_va, Xtestdf, _y_log[tr_mask],
                te_cfg.get("columns"), te_cfg.get("smoothing", esup.DEFAULT_SMOOTHING),
                cat_fallback=cat_feats)
        y_tr, y_va = _y_log[tr_mask], _y_log[va_mask]
        m = CatBoostRegressor(iterations=n_est, cat_features=cat_feats, **p)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=esr)
        oof[va_mask] = np.expm1(m.predict(X_va))
        pred += np.expm1(m.predict(Xte_fold)) / N_SPLITS
    return oof, pred


def _run_lin(params, Xdf, Xtestdf, cat_feats):
    """Sparse-linear family (Ridge on one-hot design) — unlocks encoding scheme
    onehot_sparse. Same OOF/pred contract and log-space semantics as the GBDT runners."""
    return esup.run_linear(params, Xdf, Xtestdf, cat_feats, _FOLDS, _y_log,
                           len(_train), invert=np.expm1)


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    postprocess = config.get("postprocess") or {}
    te_cfg = config.get("target_encoding") or None

    Xdf, Xtestdf, feats, cat_feats = get_feature_frame(drop)
    if model == "lgb":
        oof, pred = _run_lgb(params, Xdf, Xtestdf, te_cfg=te_cfg, cat_feats=cat_feats)
    elif model == "cat":
        oof, pred = _run_cat(params, Xdf, Xtestdf, cat_feats, te_cfg=te_cfg)
    elif model == "lin":
        oof, pred = _run_lin(params, Xdf, Xtestdf, cat_feats)
    else:
        raise ValueError(f"unknown model type {model!r}")

    score, scale_used = maybe_postprocess(oof, postprocess)
    extra = dict(n_feats=len(feats), scale_used=round(scale_used, 3))
    if te_cfg:
        extra["target_encoding"] = te_cfg.get("columns") or cat_feats
        esup.write_attestation(_COMP_DIR, "target_encoding", {
            "columns": te_cfg.get("columns") or cat_feats,
            "smoothing": te_cfg.get("smoothing", esup.DEFAULT_SMOOTHING),
            "fold_aligned": True})
    if postprocess:
        esup.write_attestation(_COMP_DIR, "postprocess", {
            "honored_params": sorted(postprocess), "scale_used": round(scale_used, 3)})
    if model == "lin":
        esup.write_attestation(_COMP_DIR, "linear_family", {"model": "ridge"})
    return oof, pred, score, feats, extra


# ---------------------------------------------------------------------------
# blend node: weight search over cached member OOF vectors, no retraining
# ---------------------------------------------------------------------------
def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    postprocess = config.get("postprocess") or {}

    # postprocessing must happen INSIDE metric_fn so it applies to every candidate weight
    # vector rather than retroactively to the winner -- harness_v2.eval_blend's contract
    def metric_fn(vec_full):
        s, _ = maybe_postprocess(vec_full, postprocess)
        return s

    best_w, best_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                              weight_search=method, seed=SEED)
    blended = oofs @ best_w
    _, scale_used = maybe_postprocess(blended, postprocess)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, scale_used=round(scale_used, 3),
                mape=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 200) -> dict:
    import time
    t0 = time.time()
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, _feats, extra = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mape=score)
            return {"score": round(float(score), 6), "status": "evaluated",
                    "wall_s": round(time.time() - t0, 2), "result": extra}
        elif kind == "blend":
            info, score = evaluate_blend(config)
            return {"score": round(float(score), 6), "status": "evaluated",
                    "wall_s": round(time.time() - t0, 2), "result": info}
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return {"score": None, "status": "timeout", "wall_s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001 -- a failed node is recorded, never hidden
        return {"score": None, "status": "failed", "wall_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
