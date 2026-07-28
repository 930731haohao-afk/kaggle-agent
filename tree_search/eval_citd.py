"""tree_search/eval_citd.py -- per-competition evaluator for cat-in-the-dat
(binary classification, ROC-AUC, MAXIMIZE-better, no post-processing).

Node kinds (same schema as eval_s6e2.py):
  1. solo  -- {kind:"solo", model:"lr"|"lgb_te"|"lgb_cat", params:{...},
               features:{variant:"base"|"pairs"|"pairs_ord5"}}
     "lr"      : LogisticRegression on sparse OHE (variant controls interaction crosses)
     "lgb_te"  : LightGBM on int frame + fold-safe smoothed target encoding
                 (byte-identical to competitions/cat-in-the-dat/scripts/members.py lgb_te)
     "lgb_cat" : LightGBM native categorical (members.py lgb_cat)
  2. blend -- handled by the driver via harness_v3.eval_blend on this module's CACHE_DIR.

Folds: fixed 5-fold StratifiedKFold(seed=42) loaded from data_proc/folds.npy --
IDENTICAL to the linear-stage members (scripts/members.py), so cached-OOF pool reuse
is digit-exact. SIGN CONVENTION: harness wants lower-is-better -> score = -AUC;
result["auc"] carries the human-readable AUC.

Threads capped at 10 (shared machine); LGBM deterministic+force_row_wise.
"""
import os
import signal
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_COMP = os.path.join(_REPO, "competitions", "cat-in-the-dat")
PROC = os.path.join(_COMP, "data_proc")
CACHE_DIR = os.path.join(_HERE, "cache_citd")
os.makedirs(CACHE_DIR, exist_ok=True)
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

THREADS = 10
N_FOLDS = 5
_y = np.load(os.path.join(PROC, "y.npy"))
_folds = np.load(os.path.join(PROC, "folds.npy"))


def auc(y_true, y_score) -> float:
    return roc_auc_score(y_true, y_score)


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature matrices (lazy, disk-cached)
# ---------------------------------------------------------------------------
_MAT = {}
PAIR_COLS = ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4",
             "nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
             "ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "day", "month"]
ORD5_PARTNERS = ["day", "month", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4",
                 "ord_0", "ord_1", "ord_2", "ord_3", "ord_4"]


def _cross_matrix(pairs):
    """Sparse indicator matrix for a list of column pairs, over train+test rows."""
    gtr = pd.read_parquet(os.path.join(PROC, "gbdt_train.parquet"))
    gte = pd.read_parquet(os.path.join(PROC, "gbdt_test.parquet"))
    n_tr, n_te = len(gtr), len(gte)
    blocks_tr, blocks_te = [], []
    for a, b in pairs:
        ca = np.concatenate([gtr[a].values, gte[a].values]).astype(np.int64)
        cb = np.concatenate([gtr[b].values, gte[b].values]).astype(np.int64)
        combo = ca * (cb.max() + 1) + cb
        codes, _ = pd.factorize(combo)
        k = codes.max() + 1
        rows = np.arange(len(codes))
        m = sparse.csr_matrix((np.ones(len(codes), np.float32), (rows, codes)),
                              shape=(len(codes), k))
        blocks_tr.append(m[:n_tr])
        blocks_te.append(m[n_tr:])
    return sparse.hstack(blocks_tr).tocsr(), sparse.hstack(blocks_te).tocsr()


def get_lr_matrices(variant="base"):
    if variant in _MAT:
        return _MAT[variant]
    base_tr = sparse.load_npz(os.path.join(PROC, "ohe_train.npz"))
    base_te = sparse.load_npz(os.path.join(PROC, "ohe_test.npz"))
    if variant == "base":
        _MAT[variant] = (base_tr, base_te)
        return _MAT[variant]
    cache_tr = os.path.join(PROC, f"cross_{variant}_train.npz")
    cache_te = os.path.join(PROC, f"cross_{variant}_test.npz")
    if os.path.exists(cache_tr) and os.path.exists(cache_te):
        cr_tr, cr_te = sparse.load_npz(cache_tr), sparse.load_npz(cache_te)
    else:
        if variant == "pairs":
            pairs = list(combinations(PAIR_COLS, 2))
        elif variant == "pairs_ord5":
            pairs = list(combinations(PAIR_COLS, 2)) + [("ord_5", c) for c in ORD5_PARTNERS]
        else:
            raise ValueError(f"unknown variant {variant!r}")
        cr_tr, cr_te = _cross_matrix(pairs)
        sparse.save_npz(cache_tr, cr_tr)
        sparse.save_npz(cache_te, cr_te)
    X_tr = sparse.hstack([base_tr, cr_tr]).tocsr()
    X_te = sparse.hstack([base_te, cr_te]).tocsr()
    _MAT[variant] = (X_tr, X_te)
    return _MAT[variant]


# ---------------------------------------------------------------------------
# solo runners
# ---------------------------------------------------------------------------
def _run_lr(params, variant):
    p = dict(C=0.1, solver="lbfgs", max_iter=2000, tol=1e-5)
    p.update(params)
    X_tr, X_te = get_lr_matrices(variant)
    oof = np.zeros(len(_y))
    pred = np.zeros(X_te.shape[0])
    for f in range(N_FOLDS):
        tr, va = np.where(_folds != f)[0], np.where(_folds == f)[0]
        m = LogisticRegression(**p)
        m.fit(X_tr[tr], _y[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
        pred += m.predict_proba(X_te)[:, 1] / N_FOLDS
    return oof, pred


LGB_BASE = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50, num_threads=THREADS, seed=42,
    deterministic=True, force_row_wise=True, verbosity=-1,
)
CAT_COLS = [f"nom_{i}" for i in range(10)] + ["day", "month"]


def _run_lgb_cat(params):
    import lightgbm as lgb
    p = dict(LGB_BASE)
    p.update({k: v for k, v in params.items() if k not in ("num_boost_round",)})
    nbr = params.get("num_boost_round", 3000)
    df_tr = pd.read_parquet(os.path.join(PROC, "gbdt_train.parquet"))
    df_te = pd.read_parquet(os.path.join(PROC, "gbdt_test.parquet"))
    use = [c for c in df_tr.columns if c not in ("day_sin", "day_cos", "month_sin", "month_cos")]
    df_tr, df_te = df_tr[use], df_te[use]
    oof = np.zeros(len(_y))
    pred = np.zeros(len(df_te))
    for f in range(N_FOLDS):
        tr, va = np.where(_folds != f)[0], np.where(_folds == f)[0]
        dtr = lgb.Dataset(df_tr.iloc[tr], _y[tr], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(df_tr.iloc[va], _y[va], categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(df_tr.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(df_te, num_iteration=m.best_iteration) / N_FOLDS
    return oof, pred


def _add_te(df_tr, df_te, cols, tr_idx, va_idx, smoothing):
    out_va, out_te = {}, {}
    prior = _y[tr_idx].mean()
    for c in cols:
        g = pd.DataFrame({"k": df_tr[c].iloc[tr_idx].values, "y": _y[tr_idx]}).groupby("k")["y"].agg(["mean", "count"])
        te_map = (g["mean"] * g["count"] + prior * smoothing) / (g["count"] + smoothing)
        out_va[f"te_{c}"] = df_tr[c].iloc[va_idx].map(te_map).fillna(prior).values
        out_te[f"te_{c}"] = df_te[c].map(te_map).fillna(prior).values
    return out_va, out_te


def _run_lgb_te(params):
    import lightgbm as lgb
    p = dict(LGB_BASE)
    smoothing = params.get("smoothing", 20)
    p.update({k: v for k, v in params.items() if k not in ("num_boost_round", "smoothing")})
    nbr = params.get("num_boost_round", 3000)
    df_tr = pd.read_parquet(os.path.join(PROC, "gbdt_train.parquet"))
    df_te = pd.read_parquet(os.path.join(PROC, "gbdt_test.parquet"))
    te_cols = [f"nom_{i}" for i in range(10)] + ["ord_5", "day", "month"]
    base_cols = [c for c in df_tr.columns if c not in [f"nom_{i}" for i in range(5, 10)]]
    oof = np.zeros(len(_y))
    pred = np.zeros(len(df_te))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(_folds != f)[0], np.where(_folds == f)[0]
        sub = _folds[tr_idx]
        Xtr = df_tr[base_cols].iloc[tr_idx].copy()
        for c in te_cols:
            Xtr[f"te_{c}"] = np.nan
        for sf in sorted(set(sub)):
            s_tr = tr_idx[sub != sf]
            s_va = tr_idx[sub == sf]
            va_map, _ = _add_te(df_tr, df_te.iloc[:1], te_cols, s_tr, s_va, smoothing)
            for c in te_cols:
                Xtr.loc[df_tr.index[s_va], f"te_{c}"] = va_map[f"te_{c}"]
        va_map, te_map_ = _add_te(df_tr, df_te, te_cols, tr_idx, va_idx, smoothing)
        Xva = df_tr[base_cols].iloc[va_idx].copy()
        Xte = df_te[base_cols].copy()
        for c in te_cols:
            Xva[f"te_{c}"] = va_map[f"te_{c}"]
            Xte[f"te_{c}"] = te_map_[f"te_{c}"]
        low_cats = [f"nom_{i}" for i in range(5)]
        dtr = lgb.Dataset(Xtr, _y[tr_idx], categorical_feature=low_cats)
        dva = lgb.Dataset(Xva, _y[va_idx], categorical_feature=low_cats, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=nbr, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = m.predict(Xva, num_iteration=m.best_iteration)
        pred += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
    return oof, pred


def evaluate_solo(config):
    model = config["model"]
    params = dict(config.get("params") or {})
    variant = (config.get("features") or {}).get("variant", "base")
    if model == "lr":
        oof, pred = _run_lr(params, variant)
    elif model == "lgb_te":
        oof, pred = _run_lgb_te(params)
    elif model == "lgb_cat":
        oof, pred = _run_lgb_cat(params)
    else:
        raise ValueError(f"unknown model {model!r}")
    return oof, pred, auc(_y, oof)


# ---------------------------------------------------------------------------
# dispatch (evaluate contract for eval_solo_subprocess)
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"auc": round(score, 6)}, error=None)
        raise ValueError(f"eval_citd handles solo only in-subprocess, got {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
