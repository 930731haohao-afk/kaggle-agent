"""tree_search/eval_s4e11_run2.py -- evaluator for playground-series-s4e11
(Depression, binary classification, ACCURACY metric, MAXIMIZE-better).

Accuracy is a *discretized decision* metric, so the decision threshold is part of the
model, not a reporting detail. Every score this module returns is the accuracy AFTER an
exactly-optimal OOF threshold (`features.best_threshold_acc`), recomputed inside the
metric function -- so every fold, every candidate weight vector and every blend node is
compared post-threshold (the s3e16 raw-vs-rounded mirror-failure rule).

Node kinds:
  1. solo  -- {kind:"solo", model:"lgb"|"xgb"|"cat"|"lr", params:{...},
               features:{drop:[...]}, te:[<raw col>, ...]}
     Trains on the SAME canonical 5-fold StratifiedKFold(shuffle=True, seed=42) as
     competitions/.../scripts/features.py:make_folds (imported, not re-implemented).
     `te` requests fold-aligned target encoding of the named raw columns: computed inside
     each fold from that fold's TRAINING rows only, using the model's own folds -- the
     s4e1 lesson (an independent-seed encoding split is a leakage path, not a safeguard).
     With node_id given, OOF/test predictions are cached via harness_v2.cache_oof so
     blend nodes never retrain.

  2. blend -- {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"}
     Weight-searches cached member OOFs for the ACCURACY-MAXIMIZING mix.

SIGN CONVENTION: the harness assumes LOWER-is-better, so every score handed back is
`-accuracy`. `result["acc"]` carries the real, higher-is-better accuracy -- use THAT for
reporting.

HONESTY CAVEAT built into the design: a blend node's weights AND its threshold are fitted
on the same OOF vector they are scored on, so blend node scores are optimistic by
construction (measured at +0.000817 on this competition's first 5-way blend). The
champion is therefore NOT the tree's top node by raw score -- scripts/05_honest_select.py
re-fits weights and threshold leave-fold-out and picks by that number.
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s4e11")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
from features import (  # noqa: E402
    best_threshold_acc, build_all, encode_target, make_folds,
)
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s4e11_run2")
os.makedirs(CACHE_DIR, exist_ok=True)
N_SPLITS, SEED, NT = 5, 42, 10

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = encode_target(_train)
_FOLDS = make_folds(_y)
_Xtr_full, _Xte_full, ALL_FEATURES, CAT_COLS = build_all(_train, _test)

# raw columns available for fold-aligned target encoding
TE_SOURCES = ["City", "Profession", "Degree", "Name"]
_raw_tr = {c: _train[c].astype("object").fillna("MISSING").to_numpy() for c in TE_SOURCES}
_raw_te = {c: _test[c].astype("object").fillna("MISSING").to_numpy() for c in TE_SOURCES}

LGB_BASE = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                num_leaves=63, max_depth=-1, min_child_samples=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
                verbose=-1, num_threads=NT, deterministic=True, force_row_wise=True,
                seed=SEED)
XGB_BASE = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, min_child_weight=10, random_state=SEED, n_jobs=NT,
                tree_method="hist")
CAT_BASE = dict(loss_function="Logloss", eval_metric="Logloss", learning_rate=0.05,
                depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
                allow_writing_files=False, thread_count=NT)


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def accuracy(y_true, p) -> float:
    """The competition metric, with its own postprocessing (threshold) inside."""
    return best_threshold_acc(np.asarray(y_true), np.asarray(p))[0]


# ---------------------------------------------------------------------------
# feature frame (+ fold-aligned target encoding)
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    cats = [c for c in CAT_COLS if c not in drop]
    return _Xtr_full[feats].copy(), _Xte_full[feats].copy(), feats, cats


def _te_columns(te_cols, tr_idx, smoothing=20.0):
    """Smoothed target encoding computed from tr_idx rows ONLY (the model's own fold).

    Returns (train_matrix over ALL rows, test_matrix). Rows outside tr_idx are encoded
    with the mapping learned on tr_idx, which is exactly what the held-out fold needs.
    """
    prior = float(_y[tr_idx].mean())
    tr_out = np.empty((len(_y), len(te_cols)), dtype=np.float32)
    te_out = np.empty((len(_test), len(te_cols)), dtype=np.float32)
    for j, c in enumerate(te_cols):
        keys = _raw_tr[c]
        df = pd.DataFrame({"k": keys[tr_idx], "y": _y[tr_idx]})
        agg = df.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + smoothing * prior) / (agg["count"] + smoothing)
        tr_out[:, j] = pd.Series(keys).map(enc).fillna(prior).to_numpy(np.float32)
        te_out[:, j] = pd.Series(_raw_te[c]).map(enc).fillna(prior).to_numpy(np.float32)
    return tr_out, te_out


# ---------------------------------------------------------------------------
# model runners
# ---------------------------------------------------------------------------
def _fit_fold_lgb(p, Xtr, ytr, Xva, yva, Xte, cats):
    import lightgbm as lgb
    n_rounds = p.pop("num_boost_round", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr,
                      free_raw_data=False)
    m = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(esr, verbose=False),
                             lgb.log_evaluation(0)])
    return (m.predict(Xva, num_iteration=m.best_iteration),
            m.predict(Xte, num_iteration=m.best_iteration))


def _fit_fold_xgb(p, Xtr, ytr, Xva, yva, Xte, cats):
    import xgboost as xgb
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
    m.fit(Xtr.to_numpy(np.float32), ytr,
          eval_set=[(Xva.to_numpy(np.float32), yva)], verbose=False)
    return (m.predict_proba(Xva.to_numpy(np.float32))[:, 1],
            m.predict_proba(Xte.to_numpy(np.float32))[:, 1])


def _fit_fold_cat(p, Xtr, ytr, Xva, yva, Xte, cats):
    from catboost import CatBoostClassifier
    n_est = p.pop("iterations", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    idx = [Xtr.columns.get_loc(c) for c in cats]
    A, B, C = Xtr.copy(), Xva.copy(), Xte.copy()
    for c in cats:
        A[c] = A[c].astype(int)
        B[c] = B[c].astype(int)
        C[c] = C[c].astype(int)
    m = CatBoostClassifier(iterations=n_est, cat_features=idx, **p)
    m.fit(A, ytr, eval_set=(B, yva), early_stopping_rounds=esr, verbose=False)
    return m.predict_proba(B)[:, 1], m.predict_proba(C)[:, 1]


def _fit_fold_lr(p, Xtr, ytr, Xva, yva, Xte, cats):
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    C = p.pop("C", 0.5)
    num_cols = [c for c in Xtr.columns if c not in cats]

    def design(fit_frames, frames):
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
        ohe.fit(pd.concat([f[cats] for f in fit_frames], axis=0).astype(str))
        med = pd.concat([f[num_cols] for f in fit_frames], axis=0).median()
        sc = StandardScaler().fit(
            pd.concat([f[num_cols] for f in fit_frames], axis=0).fillna(med))
        out = []
        for f in frames:
            A = ohe.transform(f[cats].astype(str))
            miss = f[num_cols].isna().astype(np.float32).to_numpy()
            D = np.hstack([sc.transform(f[num_cols].fillna(med)).astype(np.float32), miss])
            out.append(sparse.hstack([A, sparse.csr_matrix(D)]).tocsr())
        return out

    Mtr, Mva, Mte = design([Xtr], [Xtr, Xva, Xte])
    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    m.fit(Mtr, ytr)
    return m.predict_proba(Mva)[:, 1], m.predict_proba(Mte)[:, 1]


_RUNNERS = {"lgb": (_fit_fold_lgb, LGB_BASE), "xgb": (_fit_fold_xgb, XGB_BASE),
            "cat": (_fit_fold_cat, CAT_BASE), "lr": (_fit_fold_lr, {})}


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    Xdf, Xtedf, feats, cats = get_feature_frame(feat_cfg.get("drop", []))
    model = config["model"]
    if model not in _RUNNERS:
        raise ValueError(f"unknown model type {model!r}")
    runner, base = _RUNNERS[model]
    te_cols = list(config.get("te") or [])
    bad = [c for c in te_cols if c not in TE_SOURCES]
    if bad:
        raise ValueError(f"unknown te column(s): {bad}")

    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtedf))
    for tr, va in _FOLDS:
        A, B, C = Xdf.iloc[tr], Xdf.iloc[va], Xtedf
        if te_cols:
            te_tr, te_te = _te_columns(te_cols, tr)  # fold-aligned, fitted on tr only
            names = [f"te_{c}" for c in te_cols]
            A = pd.concat([A.reset_index(drop=True),
                           pd.DataFrame(te_tr[tr], columns=names)], axis=1)
            B = pd.concat([B.reset_index(drop=True),
                           pd.DataFrame(te_tr[va], columns=names)], axis=1)
            C = pd.concat([C.reset_index(drop=True),
                           pd.DataFrame(te_te, columns=names)], axis=1)
        p = dict(base)
        p.update(config.get("params") or {})
        pv, pt = runner(p, A, _y[tr], B, _y[va], C, cats)
        oof[va] = pv
        pred += pt / N_SPLITS

    score = accuracy(_y, oof)
    n_feats = len(feats) + len(te_cols)
    return oof, pred, score, n_feats


# ---------------------------------------------------------------------------
# blend node
# ---------------------------------------------------------------------------
def _neg_acc(vec):
    return -accuracy(_y, vec)


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    best_w, best_neg, _oofs = hv2.eval_blend(CACHE_DIR, members, _neg_acc,
                                             weight_search=method)
    best_score = -best_neg
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, acc=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 1200) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises.

    score is `-accuracy` (harness sign convention). result["acc"] carries the real,
    higher-is-better accuracy."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, n_feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, acc=score)
            thr = best_threshold_acc(_y, oof)[1]
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"n_feats": n_feats, "acc": round(score, 6),
                                "thr": round(float(thr), 4)}, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
