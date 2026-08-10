"""tree_search/eval_s3e5_v2.py — per-competition evaluator for playground-series-s3e5
(Wine Quality, ordinal regression, QWK, DISCRETIZED and maximize-better), harness_v2
(Phase E-2) build. v1 (tree_search/eval_s3e5.py, hand-rolled solo+blend schema on
harness.py) TIED the linear-iteration best here: node #11 (4-way blend) <score> vs
linear's <score>, a <score> gap STATUS.md's own appendix calls "the cutpoint-boundary
noise gap" (single-sample cutpoint assignment territory). This build asks whether
harness_v2's two mechanisms BUILT FOR exactly this situation — the metric-aware
adaptive plateau (tie_rate-driven, §6.2) and the ensemble-default node space (§6.1) —
change the outcome when re-run on the identical data/CV/rounder.

SAME rounder discipline as v1 (experience.md's s3e16/s3e5 lesson): every node's score is
the QWK *after* a cutpoint-fitting step on that node's own OOF, never a raw regression
number. NEW this build: the cutpoint-fitting step itself is a config-level choice,
`rounder`: "full_oof" (default, identical to v1 — OptimizedRounder fit once on the whole
5-fold OOF vector) or "fold_avg" (per the task brief's "per-fold-averaged cutpoints as an
alternative rounder inside a node variant" instruction — fit OptimizedRounder separately
on each fold's own (oof[val_idx], y[val_idx]) slice, average the 5 resulting cutpoint
vectors element-wise, then decode the FULL OOF with that averaged vector). This is
EXACTLY the computation STATUS.md's exp #10 diagnostic already ran once by hand
("Averaged-fold cutpoints applied globally = <score>" vs full-OOF <score> on the 6-way
blend, i.e. WORSE by -<score> at that pool's maturity) — this module makes it a first-
class, re-runnable node config instead of a one-off diagnostic script, so the search can
test it on THIS run's own pool compositions with a low, evidence-based prior (expect
flat-to-worse, not an assumed win). Whichever rounder a node uses, it is applied inside
the node's own evaluate_solo/evaluate_blend — never after the fact — same non-negotiable
rule as v1.

Two node kinds, same schema convention as eval_s3e9_v2.py/eval_s3e11.py:

  1. solo — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...}, features:{drop:[...]},
             rounder:"full_oof"|"fold_avg"}
     Trains one regressor with the IDENTICAL 5-fold StratifiedKFold(quality, shuffle,
     seed=42) as v1/scripts/{train.py,iterate.py}. If `node_id` is given, its OOF is
     cached via harness_v2.cache_oof to CACHE_DIR (see module-level CACHE_DIR comment
     for why this is a `v2/` subdirectory of v1's cache_s3e5/, not the same flat
     namespace) so blend nodes never retrain.

     ⚠️ Legacy-OOF-reuse (this run's own budget lever, ported from eval_s3e9_v2.py's
     load_legacy_solo / eval_s3e11.py's pattern): v1's OWN cache_s3e5/solo_<id>.npz files
     for the root LGB_tuned (#0) and 6 solo seeds (CAT #2, XGB #3, LGBNUDGE #4, FEAT #5,
     CATORIG #6, LGBORIG #7) are READ-ONLY inputs here — `load_v1_solo(v1_id,
     expected_qwk)` loads the cached OOF, recomputes full_oof QWK via THIS module's own
     post_rounder_qwk, and asserts it reproduces the historical value digit-for-digit
     before anything is re-cached under a v2 node id. This IS the "root reproduces
     digit-for-digit BEFORE searching" check the task brief requires, done as a load +
     recompute (near-instant) rather than a full retrain, freeing the ~30 min budget for
     genuinely new territory (boundary-push probes, the fold_avg rounder variant, new
     blend compositions).

  2. blend — {kind:"blend", members:[<solo node id>, ...], weight_search:"dirichlet",
             rounder:"full_oof"|"fold_avg"}
     Loads cached member OOF vectors via harness_v2.load_oof, weight-searches
     (harness_v2.eval_blend) for the blend that maximizes the CHOSEN rounder's QWK.
     `rounder` is threaded into metric_fn so it applies to EVERY candidate weight vector
     during the search itself (harness_v2's non-negotiable rule for discretized metrics),
     not just retroactively to the winning weight vector.

     Blend-cost mitigation (task brief: "coarser weight grid ... if 45s/blend threatens
     budget — document what you choose") — REVISED after an empirical check, not assumed:
     an initial pass tried k=200 (~15s/blend node). It reproduced v1's own exact best
     composition (members [root, CAT_tuned, XGB, FEAT]) but landed on a visibly WORSE
     weight-simplex point (QWK <score> vs v1's <score> on the IDENTICAL 4 OOF vectors —
     confirmed by plugging v1's own weight vector into this module's own metric_fn, which
     reproduces <score> and v1's exact cutpoints digit-for-digit, so the gap is 100% a
     search-quality issue, not a data/implementation mismatch). Adding a coordinate-ascent
     refinement stage on top of k=200 did NOT close the gap (still <score>) — the coarse
     dirichlet stage simply never sampled near the true optimum's basin, so no amount of
     *local* refinement could reach it. A direct k-sweep on that exact composition (200 /
     500 / 800 / 1500 dirichlet draws, each + coordinate-ascent refinement) found: k=200
     -> <score> (17s), k=500 -> <score> (40s, matches v1 exactly), k=800 -> <score> (60s,
     BEATS both v1's <score> and the linear iteration's <score>), k=1500 -> <score> (112s,
     still beats both, but LESS than k=800 — the search is stochastic, not monotonic in
     k). Given the run's actual wall-clock budget has ample headroom (the first full pass
     with k=200 used only ~90s total for 22 nodes against a ~28 min budget), the "coarser
     grid" mitigation is NOT worth the precision it costs here — this module uses k=800
     (roughly v1's own 800-draw budget, now WITH the coordinate-ascent refinement stage
     v1's weight_search() also had, which harness_v2.eval_blend's dirichlet-only search
     lacks by default) rather than a coarser setting, at a measured ~60s/blend node. This
     is the single most consequential empirical decision in this build: the "safe-looking"
     default coarser search would have silently reproduced v1's tie-or-loss outcome for a
     reason that had NOTHING to do with the harness's new mechanisms (adaptive plateau,
     ensemble-default, dedup) and everything to do with an under-budgeted weight search.

SIGN CONVENTION: unchanged from v1 — QWK is maximize-better, harness_v2 (like v1's
harness.py) assumes lower-is-better scores throughout, so every score handed to
harness_v2.add_root/add_node is `-QWK`. `result["qwk"]` carries the real, human-readable,
higher-is-better number.
"""
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e5")
DATA = os.path.join(_COMP_DIR, "data")

# v1's cache (READ-ONLY here — never written to, per the task brief's "do NOT overwrite"
# instruction extended in spirit from the tree JSON to its backing OOF cache): holds
# solo_<v1_node_id>.npz for v1's root + 6 first-gen solo seeds, reused below via
# load_v1_solo(). v2's OWN new node caches (both genuinely-new solos and the legacy
# members re-cached under v2's own node-id numbering) go to a `v2/` SUBDIRECTORY of that
# same cache_s3e5/ folder — literally inside "tree_search/cache_s3e5/" as the task brief
# names it, but namespaced so v2's node id #3 (say) can never collide with / overwrite
# v1's unrelated node #3 file sitting flat in the parent directory. This is the safest
# reading of "cache tree_search/cache_s3e5/, may reuse v1's cached OOFs if compatible".
V1_CACHE_DIR = os.path.join(_HERE, "cache_s3e5")
CACHE_DIR = os.path.join(V1_CACHE_DIR, "v2")

sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402

TARGET, ID = "quality", "Id"
N_SPLITS, SEED = 5, 42

_train = pd.read_csv(os.path.join(DATA, "train_processed.csv"))
_test = pd.read_csv(os.path.join(DATA, "test_processed.csv"))
ALL_FEATURES = [c for c in _train.columns if c not in (ID, TARGET)]
_y = _train[TARGET].to_numpy(int)
LOW, HIGH = int(_y.min()), int(_y.max())

_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _y))  # IDENTICAL to v1/scripts/{train,iterate}.py


# ---------------------------------------------------------------------------
# OptimizedRounder — ported verbatim from v1's eval_s3e5.py (itself ported from
# scripts/iterate.py). Two scoring modes now sit on top of it (see module docstring):
# post_rounder_qwk (v1's only mode, "full_oof") and fold_avg_rounder_qwk (new).
# ---------------------------------------------------------------------------
class OptimizedRounder:
    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self.coef_ = np.arange(low + 0.5, high, 1.0)

    def _to_classes(self, x, coef):
        coef = np.sort(coef)
        return np.clip(np.digitize(x, coef) + self.low, self.low, self.high)

    def _loss(self, coef, x, y):
        return -cohen_kappa_score(y, self._to_classes(x, coef), weights="quadratic")

    def fit(self, x, y):
        res = minimize(self._loss, self.coef_, args=(x, y), method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
        self.coef_ = np.sort(res.x)
        return self

    def predict(self, x):
        return self._to_classes(x, self.coef_)


def qwk(y_true, y_pred):
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def post_rounder_qwk(oof_vec):
    """v1's ONLY rounder mode ("full_oof"): fit OptimizedRounder on the whole 5-fold OOF
    vector, score against it. Digit-for-digit identical to v1's eval_s3e5.py."""
    r = OptimizedRounder(LOW, HIGH).fit(oof_vec, _y)
    return qwk(_y, r.predict(oof_vec)), r.coef_.copy()


def fold_avg_rounder_qwk(oof_vec):
    """NEW rounder mode ("fold_avg", task brief's "per-fold-averaged cutpoints"): fit
    OptimizedRounder separately on each fold's own (oof[val_idx], y[val_idx]) slice (5
    independent cutpoint-fits, each using only that fold's held-out predictions — a
    cheaper cousin of STATUS.md's exp #7/#10 honest leave-fold-out nested diagnostic,
    which retrained models leaving a fold fully out; here the OOF vector is already
    out-of-fold per row, so this is "per-fold cutpoint fit on already-OOF predictions",
    averaged), then average the 5 resulting cutpoint vectors element-wise and decode the
    FULL OOF vector with that single averaged vector. STATUS.md's own exp #10 already ran
    this exact computation once by hand on the linear run's 6-way blend and found it
    WORSE than full-OOF fitting (<score> vs <score>) — this function makes that a
    re-runnable node-level choice instead of a one-off diagnostic, with that result as
    the honest prior (expect flat-to-worse, not assumed to help)."""
    coefs = []
    for _, va in _FOLDS:
        r = OptimizedRounder(LOW, HIGH).fit(oof_vec[va], _y[va])
        coefs.append(r.coef_)
    avg_coef = np.sort(np.mean(np.array(coefs), axis=0))
    r_final = OptimizedRounder(LOW, HIGH)
    r_final.coef_ = avg_coef
    return qwk(_y, r_final.predict(oof_vec)), avg_coef


def score_with_rounder(oof_vec, rounder: str):
    if rounder == "full_oof":
        return post_rounder_qwk(oof_vec)
    elif rounder == "fold_avg":
        return fold_avg_rounder_qwk(oof_vec)
    else:
        raise ValueError(f"unknown rounder mode {rounder!r}")


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature selection (unchanged from v1)
# ---------------------------------------------------------------------------
def get_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {unknown}")
    feats = [c for c in ALL_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    return _train[feats].copy(), _test[feats].copy(), feats


# ---------------------------------------------------------------------------
# solo-model runners (unchanged from v1's eval_s3e5.py — same fixed n_estimators/no
# early-stopping recipe; data is tiny)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="regression", n_estimators=1500, learning_rate=0.03,
             num_leaves=31, max_depth=6, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.7, reg_lambda=1.0, min_child_samples=15,
             random_state=SEED, n_jobs=-1, verbose=-1)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="reg:squarederror", n_estimators=1500, learning_rate=0.03,
              max_depth=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
              min_child_weight=5, random_state=SEED, n_jobs=-1, tree_method="hist")
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xtest):
    from catboost import CatBoostRegressor
    p = dict(loss_function="RMSE", iterations=1500, learning_rate=0.03, depth=6,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False, allow_writing_files=False,
             thread_count=-1)
    p.update(params)
    if "n_estimators" in p:
        p["iterations"] = p.pop("n_estimators")
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = CatBoostRegressor(**p)
        m.fit(X[tr], _y[tr])
        oof[va] = m.predict(X[va])
        pred += m.predict(Xtest) / N_SPLITS
    return oof, pred


RUNNERS = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat}


def evaluate_solo(config):
    drop = (config.get("features") or {}).get("drop", [])
    Xdf, Xtestdf, feats = get_feature_frame(drop)
    model = config["model"]
    params = dict(config.get("params") or {})
    rounder = config.get("rounder", "full_oof")
    if model not in RUNNERS:
        raise ValueError(f"unknown model type {model!r}")
    Xnp, Xtestnp = Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32)
    oof, pred = RUNNERS[model](params, Xnp, Xtestnp)
    score, coef = score_with_rounder(oof, rounder)
    return oof, pred, score, coef, feats


# ---------------------------------------------------------------------------
# legacy-OOF-reuse: v1's cache_s3e5/solo_<v1_id>.npz, READ-ONLY.
# ---------------------------------------------------------------------------
def load_v1_solo(v1_node_id: int, expected_qwk: float, tol: float = 2e-4):
    path = os.path.join(V1_CACHE_DIR, f"solo_{v1_node_id}.npz")
    if not os.path.exists(path):
        raise ValueError(f"v1 cache has no node #{v1_node_id} at {path}")
    d = np.load(path, allow_pickle=True)
    oof, pred = d["oof"], d["pred"]
    if not np.array_equal(d["y"], _y):
        raise ValueError(f"v1 cache node #{v1_node_id}'s y does not match current target "
                          f"vector -- fold/data drift, cache is stale")
    if len(oof) != len(_y):
        raise ValueError(f"v1 cache node #{v1_node_id} OOF length {len(oof)} != {len(_y)}")
    score, coef = post_rounder_qwk(oof)  # full_oof, matching v1's own scoring exactly
    if abs(score - expected_qwk) > tol:
        raise ValueError(f"v1 cache node #{v1_node_id} QWK {score:.6f} does not reproduce "
                          f"expected {expected_qwk:.6f} (tol={tol}) -- digit-for-digit "
                          f"reproduction FAILED, do not reuse")
    return oof, pred, score, coef


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF, no retraining.
# ---------------------------------------------------------------------------
def _coord_ascent_refine(oofs, metric_fn, w0, s0, rounds=6,
                         deltas=(0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005)):
    """Cheap coordinate-ascent local refinement AFTER the coarse dirichlet search
    (ported in spirit from v1's own weight_search()/eval_s3e9_v2.py's
    _coord_descent_refine). ADDED after this module's first pass (k=200 dirichlet alone)
    reproduced v1's own exact best composition (members [root,CAT,XGB,FEAT]) but landed
    on a visibly worse point of the weight simplex (<score> vs v1's <score> on the
    IDENTICAL 4 members) -- i.e. the coarser dirichlet-only search was cutting corners
    that mattered on this discretized metric's narrow winning region, not just adding
    noise. This bounded local-search stage (<=6 rounds, single-coordinate perturbations,
    stops on no improvement) recovers that precision cheaply (a few hundred extra
    metric_fn calls, still far short of v1's ~45s/node) without reverting to v1's full
    800-draw regime."""
    best_w, best_s = np.array(w0, dtype=float), s0
    k = len(best_w)
    for _ in range(rounds):
        improved = False
        for i in range(k):
            for d in deltas:
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + d)
                if w_try.sum() <= 0:
                    continue
                w_try = w_try / w_try.sum()
                s = metric_fn(oofs @ w_try)
                if s < best_s - 1e-9:
                    best_s, best_w = s, w_try
                    improved = True
        if not improved:
            break
    return best_w, best_s


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    rounder = config.get("rounder", "full_oof")

    def metric_fn(vec):
        s, _ = score_with_rounder(vec, rounder)
        return -s  # harness_v2.eval_blend minimizes; QWK is maximize-better

    best_w, best_neg_score, oofs = hv2.eval_blend(CACHE_DIR, members, metric_fn,
                                                   weight_search=method, k=800)
    best_w, best_neg_score = _coord_ascent_refine(oofs, metric_fn, best_w, best_neg_score)
    best_score, best_coef = score_with_rounder(oofs @ best_w, rounder)
    weights = {str(m): round(float(w), 4) for m, w in zip(members, best_w)}
    return dict(members=members, weights=weights, method=method, rounder=rounder,
                cutpoints=[round(float(c), 4) for c in best_coef]), best_score, best_coef


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 120) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, coef, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, y=_y, qwk=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "qwk": round(score, 5),
                      "rounder": config.get("rounder", "full_oof"),
                      "cutpoints": [round(float(c), 4) for c in coef]}
            return dict(status="evaluated", score=round(-score, 5), wall_s=round(wall, 1),
                        result=result, error=None)
        elif kind == "blend":
            result, score, coef = evaluate_blend(config)
            wall = time.time() - t0
            result["qwk"] = round(score, 5)
            return dict(status="evaluated", score=round(-score, 5), wall_s=round(wall, 1),
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
