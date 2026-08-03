"""tree_search/eval_s4e11.py -- per-competition evaluator for playground-series-s4e11
(Depression, binary classification, Accuracy metric, maximize-better, THRESHOLD-based).

Tier4 harness_v3 driver support. Same two-node-kind schema as eval_s4e1.py/eval_s3e5_v2.py:

  1. solo  -- {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]}}
     Trains one classifier with the SAME canonical 5-fold StratifiedKFold(Depression,
     shuffle=True, seed=42) as scripts/04_train_blend.py / scripts/05_iterate.py (via
     scripts/features.py's make_folds -- imported, not re-implemented). If `node_id` is
     given, OOF/test predictions are cached to tree_search/cache_s4e11/solo_<node_id>.npz
     via harness_v2.cache_oof so blend nodes never retrain.

  2. blend -- {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet"}
     Loads cached member OOFs and weight-searches (dispatches to harness_v2.eval_blend)
     for the accuracy-maximizing combination -- ALWAYS scored through
     best_threshold_accuracy (never on raw prob), matching this comp's post-processing
     rule (features.py docstring / knowledge/experience.md's "decide on the
     post-processed score, never the raw one").

SIGN CONVENTION: harness.py/harness_v2.py/harness_v3.py assume lower-is-better scores;
Accuracy is maximize-better, so every score handed to add_root/add_node is `-accuracy`.
`result["accuracy"]` carries the real, human-readable, higher-is-better accuracy.

Reuses competitions/playground-series-s4e11/scripts/features.py UNMODIFIED (28-feature
Feb-2026 set, label-encoding only) so scores are directly comparable to tier2
(scripts/04_train_blend.py) and tier3 (scripts/05_iterate.py) numbers.
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
from features import make_folds, build_all, best_threshold_accuracy, TARGET, ID  # noqa: E402
import harness_v2 as hv2  # noqa: E402
import eval_support as esup  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s4e11")
os.makedirs(CACHE_DIR, exist_ok=True)
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.int64)
_FOLDS = make_folds(_y)  # canonical folds -- IDENTICAL to tier2/tier3
_Xtr_full, _Xte_full, ALL_FEATURES = build_all(_train, _test, _y, _FOLDS)


def acc(y_true, prob) -> float:
    """Threshold-optimized accuracy -- the ONLY decision metric in this comp (see
    features.py's best_threshold_accuracy). Returns accuracy only (drops threshold) for
    call sites that just need a scalar score."""
    a, _ = best_threshold_accuracy(y_true, prob)
    return float(a)


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
# solo-model runners (mirrors scripts/04_train_blend.py / 05_iterate.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63,
              max_depth=-1, min_child_samples=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=SEED)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X[tr], label=_y[tr])
        dva = lgb.Dataset(X[va], label=_y[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05,
              max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
              random_state=SEED, n_jobs=-1, tree_method="hist")
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 50)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostClassifier
    p = dict(loss_function="Logloss", eval_metric="Logloss", learning_rate=0.05, depth=7,
              l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
              thread_count=20)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2500))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(iterations=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=(X[va], _y[va]), early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def maybe_postprocess(prob_full, postprocess, honored=None):
    """Threshold-inside-the-metric, dossier vocabulary (threshold / round_to_int /
    clip_min / clip_max). Defaults reproduce acc()'s historical path digit-for-digit:
    threshold="auto" runs the same best_threshold_accuracy sweep, and with no clip
    requested the probability vector is untouched. round_to_int is acknowledged but a
    no-op here: the threshold sweep already emits hard 0/1 for submission, which is the
    behavior the dossier asks that key to guarantee. Until 2026-07-31 this evaluator had
    no mechanism reading the dossier's postprocess request (the audit's NOT_IMPLEMENTED
    case for s4e11).

    Pass a set as `honored` to collect the keys this function ACTUALLY read. The runtime
    attestation used to report `sorted(postprocess)` -- every key the dossier sent,
    read or not -- so a key with no mechanism here (or a plain typo) came back attested
    as honoured, which is the exact failure the attestation exists to catch. Recording
    the keys at the point of use is the only version that cannot drift from the code
    (2026-08-03 audit)."""
    pp = postprocess or {}
    seen = honored if honored is not None else set()
    v = np.asarray(prob_full, dtype=np.float64)
    if pp.get("clip_min") is not None or pp.get("clip_max") is not None:
        v = np.clip(v, pp.get("clip_min"), pp.get("clip_max"))
        seen.update(k for k in ("clip_min", "clip_max") if pp.get(k) is not None)
    thr = pp.get("threshold", "auto")
    if "threshold" in pp:
        seen.add("threshold")
    if thr == "auto":
        score = acc(_y, v)
    else:
        score = float(((v >= float(thr)).astype(np.float64) == _y).mean())
    if "round_to_int" in pp:
        seen.add("round_to_int")  # guaranteed by the sweep's hard 0/1 submission output
    return score


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    X, Xtest = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)

    if model == "lgb":
        oof, pred = _run_lgb(params, X, Xtest)
    elif model == "xgb":
        oof, pred = _run_xgb(params, X, Xtest)
    elif model == "cat":
        oof, pred = _run_cat(params, X, Xtest)
    else:
        raise ValueError(f"unknown model type {model!r}")

    postprocess = config.get("postprocess") or {}
    honored = set()
    score = maybe_postprocess(oof, postprocess, honored)
    if postprocess:
        # Attest what was READ, and name what was not, so a silently-ignored key shows up
        # as ignored instead of hiding inside honored_params (2026-08-03 audit).
        esup.write_attestation(_COMP_DIR, "postprocess", {
            "honored_params": sorted(honored),
            "ignored_params": sorted(set(postprocess) - honored)})
    return oof, pred, score, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node
# ---------------------------------------------------------------------------
def _neg_acc(vec):
    return -acc(_y, vec)


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    # METRIC SYMMETRY (v5.3, 2026-08-03 audit). Solo nodes are scored through
    # maybe_postprocess, so a dossier-fixed threshold binds them; blends used to ignore
    # `postprocess` entirely and always take the best of an 81-point threshold sweep. The
    # search then compared the two kinds with one min(). Measured here: the asymmetry is
    # worth 5e-5 to 1e-4, while this tree's top-five score gaps are 2.9e-5 to 7.9e-5 -- the
    # difference between the two rulers exceeded the difference between the candidates, so
    # a blend could win for reasons unrelated to model quality. The blend metric now reads
    # the same postprocess block, INSIDE the weight search, so every candidate weight vector
    # is scored on the vector that would actually be submitted.
    postprocess = config.get("postprocess") or {}
    metric_fn = (lambda vec: -maybe_postprocess(vec, postprocess)) if postprocess else _neg_acc
    best_w, best_neg, _oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn, weight_search=method)
    best_score = -best_neg
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, accuracy=round(best_score, 6),
                postprocess=postprocess or None), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 180) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    score is `-accuracy` (harness sign convention: lower-is-better). result["accuracy"]
    is the real, human-readable, higher-is-better threshold-optimized accuracy -- use
    THAT for all reporting/printouts."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, accuracy=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "accuracy": round(score, 6)}
            return dict(status="evaluated", score=round(-score, 6), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            wall = time.time() - t0
            return dict(status="evaluated", score=round(-score, 6), wall_s=round(wall, 1),
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
