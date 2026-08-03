"""tree_search/eval_s6e2_run2.py — per-competition evaluator for playground-series-s6e2
(Heart Disease, synthetic tabular binary classification, ROC-AUC, maximize-better).

LANE ISOLATION NOTE: tree_search/ already contains eval_s6e2.py, run_s6e2_v3.py and
cache_s6e2/ from a PREVIOUS run of this same competition. Under this run's strict
isolation rule those files must not be read or reused, so this module and its driver use
the `_run2` suffix and the separate cache directory tree_search/cache_s6e2_run2/
throughout. Nothing here reads the older artifacts.

Two node kinds, the harness's general-tabular default schema:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
              features:{version:"raw"|"v1"|"v2"|"v3", drop:[...]},
              cat_native: bool, seed: int}
     Trains one classifier with StratifiedKFold(5, shuffle=True, random_state=42) on the
     0/1-encoded target — byte-identical folds to
     competitions/playground-series-s6e2/scripts/{02_baseline,03_feature_race,04_tune_lgb,
     05_members}.py, so every score in this tree is directly comparable to the linear
     iteration's experiments.json.
     Reuses competitions/playground-series-s6e2/scripts/features.py UNMODIFIED.
     With `node_id`, the OOF/test vectors are cached via harness_v2.cache_oof to
     cache_s6e2_run2/solo_<node_id>.npz so blend nodes never retrain.

  2. blend — {kind:"blend", members:[<solo node id>...], weight_search:"dirichlet",
              space:"prob"|"rank"}
     Weight-searches over cached member OOFs. HONEST-REPORTING CAVEAT: this is a
     FULL-OOF FITTED blend score and is therefore optimistic. The driver re-scores the
     final winner with leave-fold-out weight fitting (the same procedure as
     scripts/06_blend.py) before anything is submitted.

Comp-local facts that shaped the node space (Stage 1 EDA + the linear run):
  * 630,000 rows produce 630,000 DISTINCT feature vectors — no duplicate-row label
    noise, so no memorisation/target-encoding lever exists (honest OOF full-vector
    group-rate AUC = 0.500000 exactly).
  * every engineered feature version LOST to the raw 13 columns (raw 0.955170 vs v1
    0.955056 / v2 0.955029 / v3 0.954724). The versions stay reachable in the node space
    so the search can re-test them under different model params, but `raw` is the root.
  * the entire model family spans ~0.0008 AUC (0.9547-0.9556), so gains here are small
    by construction and the honest-reporting rules matter more than usual.

SIGN CONVENTION: the harness assumes lower-is-better; AUC is maximize-better, so every
score handed back is `-AUC`. result["auc"] carries the real, human-readable AUC.

CatBoost: allow_writing_files=False + explicit thread_count per the s3e11 lesson in
knowledge/experience.md (default catboost_info file logging stalled a sandboxed run 33
minutes with zero output).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s6e2")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
import features as F  # noqa: E402
import harness_v2 as hv2  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_s6e2_run2")
N_SPLITS, SEED, NTHREAD = 5, 42, 10

_train, _test, _y = F.load(_COMP_DIR)
_FOLDS = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED).split(np.zeros(len(_y)), _y))

_FRAME_CACHE = {}


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def auc(y, p):
    return float(roc_auc_score(y, p))


def fast_auc(y, p):
    """Rank-sum ROC-AUC, tie-safe. Used ONLY inside the blend weight-search inner loop.

    sklearn's roc_auc_score costs ~0.10s on this 630k-row vector, so a 1500-draw
    Dirichlet search cost 178s per blend node in the smoke test — more than a whole
    5-fold LightGBM training. This computes the same quantity from one argsort
    (~0.04s). Every score that is REPORTED or STORED still goes through
    `auc()`/roc_auc_score, so the numbers in the tree remain directly comparable to
    the linear run's experiments.json; fast_auc only decides which weight vector wins.
    """
    n = len(p)
    order = np.argsort(p, kind="quicksort")
    ps, ys = p[order], y[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(ps[1:], ps[:-1], out=first[1:])
    idx = np.flatnonzero(first)
    counts = np.diff(np.append(idx, n))
    if counts.max() > 1:                       # ties present -> average their ranks
        ranks = np.repeat((2.0 * idx + counts - 1) / 2.0 + 1.0, counts)
    else:
        ranks = np.arange(1.0, n + 1.0)
    n_pos = float(ys.sum())
    n_neg = float(n) - n_pos
    return float((ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def get_frames(version, drop, cat_native):
    """Build (X, Xtest, feature_names, cat_cols) for a feature version, memoised."""
    key = (version, tuple(sorted(drop)), bool(cat_native))
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    X = F.build(_train, version)
    Xt = F.build(_test, version)
    if drop:
        keep = [c for c in X.columns if c not in set(drop)]
        X, Xt = X[keep], Xt[keep]
    cats = [c for c in F.cat_features(version) if c in X.columns] if cat_native else []
    _FRAME_CACHE[key] = (X, Xt, list(X.columns), cats)
    return _FRAME_CACHE[key]


def _run_lgb(params, X, Xt, cats, seed):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", verbose=-1, num_threads=NTHREAD,
             deterministic=True, force_row_wise=True, seed=seed, bagging_freq=1,
             n_estimators=3000)
    p.update(params)
    if cats:
        X, Xt = X.copy(), Xt.copy()
        for c in cats:
            uniq = sorted(set(X[c].unique()) | set(Xt[c].unique()))
            dt = pd.CategoricalDtype(categories=uniq)
            X[c] = X[c].astype(dt)
            Xt[c] = Xt[c].astype(dt)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xt))
    for tr, va in _FOLDS:
        m = lgb.LGBMClassifier(**p)
        m.fit(X.iloc[tr], _y[tr], eval_set=[(X.iloc[va], _y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xt, seed):
    from xgboost import XGBClassifier
    p = dict(n_estimators=3000, tree_method="hist", eval_metric="auc",
             early_stopping_rounds=100, n_jobs=NTHREAD, random_state=seed,
             objective="binary:logistic")
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xt))
    for tr, va in _FOLDS:
        m = XGBClassifier(**p)
        m.fit(X.iloc[tr], _y[tr], eval_set=[(X.iloc[va], _y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xt)[:, 1] / N_SPLITS
    return oof, pred


def _run_cat(params, X, Xt, cats, seed):
    from catboost import CatBoostClassifier, Pool
    p = dict(iterations=3000, loss_function="Logloss", eval_metric="AUC",
             random_seed=seed, thread_count=NTHREAD, allow_writing_files=False,
             od_type="Iter", od_wait=100, verbose=False)
    p.update(params)
    Xc, Xtc = X.copy(), Xt.copy()
    for c in cats:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)
    cat_idx = [Xc.columns.get_loc(c) for c in cats]
    te_pool = Pool(Xtc, cat_features=cat_idx)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtc))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(**p)
        m.fit(Pool(Xc.iloc[tr], _y[tr], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[va], _y[va], cat_features=cat_idx),
              use_best_model=True)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        pred += m.predict_proba(te_pool)[:, 1] / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat = config.get("features") or {}
    version = feat.get("version", "raw")
    drop = feat.get("drop", [])
    cat_native = bool(config.get("cat_native", False))
    model = config["model"]
    params = dict(config.get("params") or {})
    seed = int(config.get("seed", SEED))

    X, Xt, feats, cats = get_frames(version, drop, cat_native)
    if model == "lgb":
        oof, pred = _run_lgb(params, X, Xt, cats, seed)
    elif model == "xgb":
        oof, pred = _run_xgb(params, X, Xt, seed)
    elif model == "cat":
        oof, pred = _run_cat(params, X, Xt, cats, seed)
    else:
        raise ValueError(f"unknown model type {model!r}")
    return oof, pred, auc(_y, oof), feats


def _neg_auc(vec):
    return -auc(_y, vec)


BLEND_K = 300          # Dirichlet draws; the harness default is 800, see the note below
ASCENT_ROUNDS = 6
ASCENT_STEPS = (0.20, 0.08, 0.03, 0.01)


def _weight_search(oofs, k=BLEND_K, seed=SEED):
    """Dirichlet draws + coordinate-ascent refinement, scored with fast_auc.

    Deviation from harness_v3's DEFAULT_BLEND_K=800, declared loudly rather than
    silently (07_tree_search.md §5/§6): k is 300 here because this competition's OOF
    vectors are 630,000 long, so each candidate evaluation is ~40ms even with fast_auc.
    The coordinate-ascent refinement (which is what E-2 found actually breaks the
    coarse-grid ties, not raw draw count) is kept at full strength, and the ascent
    explores far more finely around the optimum than 500 extra random draws would.
    """
    rng = np.random.default_rng(seed)
    n = oofs.shape[1]
    cands = [np.eye(n)[i] for i in range(n)]
    cands.append(np.full(n, 1.0 / n))
    cands += list(rng.dirichlet(np.ones(n), size=k))
    best_w, best_s = None, -1e18
    for w in cands:
        s = fast_auc(_y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    # coordinate ascent
    w = np.asarray(best_w, dtype=np.float64)
    for step in ASCENT_STEPS:
        for _ in range(ASCENT_ROUNDS):
            improved = False
            for i in range(n):
                for sign in (1.0, -1.0):
                    c = w.copy()
                    c[i] = max(0.0, c[i] + sign * step)
                    tot = c.sum()
                    if tot <= 0:
                        continue
                    c /= tot
                    s = fast_auc(_y, oofs @ c)
                    if s > best_s + 1e-10:
                        best_s, w, improved = s, c, True
            if not improved:
                break
    return w, best_s


def evaluate_blend(config):
    from scipy.stats import rankdata
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    space = config.get("space", "prob")

    raw = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
    if space == "prob":
        oofs = raw
    elif space == "rank":
        n_rows, n_mem = raw.shape
        oofs = np.stack([rankdata(raw[:, i]) / n_rows for i in range(n_mem)], axis=1)
    else:
        raise ValueError(f"unknown blend space {space!r}")

    best_w, _ = _weight_search(oofs)
    # re-score the winner with sklearn so the stored number is exactly comparable to
    # every solo node and to the linear run's experiments.json
    best_score = auc(_y, oofs @ best_w)
    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, space=space, auc=round(best_score, 6),
                weight_search_k=BLEND_K,
                fitting="FULL-OOF FITTED (optimistic) — see driver caveats"), best_score


def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises.

    score is `-AUC` (harness sign convention). result["auc"] is the real AUC."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1),
                        result={"n_feats": len(feats), "auc": round(score, 6)}, error=None)
        elif kind == "blend":
            result, score = evaluate_blend(config)
            return dict(status="evaluated", score=round(-score, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
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
