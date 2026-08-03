"""tree_search/eval_s6e2_lane3.py — per-competition evaluator for playground-series-s6e2
(Heart Disease, synthetic tabular binary classification, ROC-AUC, maximize-better).

Written fresh for this run. Earlier s6e2 artifacts exist in this directory
(eval_s6e2.py / run_s6e2_v3.py / eval_s6e2_run2.py / run_s6e2_run2.py / cache_s6e2*);
they are previous runs of THIS competition and were deliberately NOT opened, so this
module and its cache directory carry the `_lane3` suffix to stay disjoint from them.

Node schema (same two kinds as every other eval_*.py in this directory):

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
              variant:"raw"|"raw_pruned"|"domain"|"domain_only_hr",
              cat_declare: bool}
     Trains one classifier under StratifiedKFold(5, shuffle, seed=42) on the binary
     target — the IDENTICAL fold object used by competitions/playground-series-s6e2/
     scripts/common.py, so tree-search scores are bit-comparable with this run's linear
     iteration (experiments #1-#16). Reuses that comp's scripts/features.py unmodified.

  2. blend — handled by the driver via harness_v3.eval_blend_with_cost_guard against
     this module's CACHE_DIR; nothing to do here.

OOF REUSE (07_tree_search.md driver checklist item 2): the linear iteration already
trained 15 arms and cached their OOF/test vectors under
competitions/playground-series-s6e2/oof/<arm>.npz. `_KNOWN_ARMS` maps a canonical config
to its arm name plus the AUC logged in experiments.json; when a proposed config matches,
the cached vectors are reloaded and the AUC RECOMPUTED, and the reuse is accepted only if
it agrees with the logged value to 6 decimal places — otherwise it falls through to a
real retrain. This is what makes the root's digit-for-digit verification meaningful
rather than circular.

SIGN CONVENTION: the harness minimizes, ROC-AUC maximizes, so `score = -auc`.
`result["auc"]` carries the human-readable value.

CatBoost: allow_writing_files=False + explicit thread_count, per the s3e11 lesson in
knowledge/experience.md (default catboost_info file logging stalled a sandboxed run 33
minutes with zero output).
"""
from __future__ import annotations

import os
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

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s6e2_lane3")
LINEAR_OOF = os.path.join(_COMP_DIR, "oof")
TARGET, ID = "Heart Disease", "id"
N_SPLITS, SEED, N_THREADS = 5, 42, 10

os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = (_train[TARGET] == "Presence").astype(int).to_numpy()
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _y))

BASE_LGB = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
    min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.0, lambda_l2=1.0, max_bin=255,
    num_threads=N_THREADS, seed=SEED, deterministic=True, force_row_wise=True, verbose=-1,
)
BASE_XGB = dict(
    objective="binary:logistic", eval_metric="auc", tree_method="hist", max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
    reg_lambda=1.0, nthread=N_THREADS, seed=SEED, max_bin=256,
)
BASE_CAT = dict(
    loss_function="Logloss", eval_metric="AUC", depth=8, learning_rate=0.05,
    l2_leaf_reg=3.0, iterations=4000, random_seed=SEED, thread_count=N_THREADS,
    allow_writing_files=False, verbose=0,
)

# Declared search space -> feeds harness_v3.boundary_candidates.
SEARCH_SPACE = {
    "lgb": {"num_leaves": (4, 255), "learning_rate": (0.01, 0.15),
            "min_child_samples": (10, 600), "feature_fraction": (0.4, 1.0),
            "bagging_fraction": (0.5, 1.0), "lambda_l1": (0.0, 10.0),
            "lambda_l2": (0.0, 20.0), "max_bin": (63, 511)},
    "xgb": {"max_depth": (3, 10), "learning_rate": (0.01, 0.15),
            "min_child_weight": (5, 300), "subsample": (0.5, 1.0),
            "colsample_bytree": (0.4, 1.0), "reg_lambda": (0.0, 20.0)},
    "cat": {"depth": (3, 8), "learning_rate": (0.02, 0.15), "l2_leaf_reg": (0.5, 20.0)},
}

# arm name -> (model, variant, cat_declare, params delta vs BASE_*, AUC logged in experiments.json)
_KNOWN_ARMS = {
    "lgb_raw":          ("lgb", "raw", False, {}, 0.955199),
    "lgb_raw_cat":      ("lgb", "raw", True, {}, 0.955198),
    "lgb_raw_pruned":   ("lgb", "raw_pruned", False, {}, 0.955119),
    "lgb_domain":       ("lgb", "domain", False, {}, 0.955036),
    "lgb_domain_hr":    ("lgb", "domain_only_hr", False, {}, 0.955146),
    "lgb_deep":         ("lgb", "raw", False,
                         {"num_leaves": 255, "learning_rate": 0.03, "min_child_samples": 40}, 0.954882),
    "lgb_shallow":      ("lgb", "raw", False,
                         {"num_leaves": 16, "learning_rate": 0.05, "lambda_l2": 3.0}, 0.955369),
    "lgb_leaves8":      ("lgb", "raw", False, {"num_leaves": 8, "lambda_l2": 3.0}, 0.955434),
    "lgb_leaves31":     ("lgb", "raw", False, {"num_leaves": 31, "lambda_l2": 3.0}, 0.955304),
    "lgb_shallow_cat":  ("lgb", "raw", True, {"num_leaves": 16, "lambda_l2": 3.0}, 0.955382),
    "lgb_shallow_lr02": ("lgb", "raw", False,
                         {"num_leaves": 16, "learning_rate": 0.02, "lambda_l2": 3.0}, 0.955412),
    "xgb_d5":           ("xgb", "raw", False, {"max_depth": 5, "min_child_weight": 50}, 0.955264),
    "xgb_d4":           ("xgb", "raw", False,
                         {"max_depth": 4, "min_child_weight": 50, "learning_rate": 0.04}, 0.955343),
    "cat_d6":           ("cat", "raw", True, {"depth": 6}, 0.955454),
    "cat_d4":           ("cat", "raw", True, {"depth": 4, "learning_rate": 0.06}, 0.955540),
}


def core(cfg: dict) -> dict:
    """Canonical / hashable stored form of a config (drives the harness's dedup hash)."""
    if cfg.get("kind") == "blend":
        return {"kind": "blend", "members": sorted(cfg["members"]),
                "space": cfg.get("space", "rank"),
                "weight_search": cfg.get("weight_search", "dirichlet")}
    return {"kind": "solo", "model": cfg["model"], "variant": cfg.get("variant", "raw"),
            "cat_declare": bool(cfg.get("cat_declare", False)),
            "params": {k: cfg["params"][k] for k in sorted(cfg.get("params", {}))}}


def _match_known(cfg: dict):
    c = core(cfg)
    for arm, (model, variant, catdec, delta, logged) in _KNOWN_ARMS.items():
        if (c["model"] == model and c["variant"] == variant
                and c["cat_declare"] == catdec
                and c["params"] == {k: delta[k] for k in sorted(delta)}):
            return arm, logged
    return None, None


def _reuse(arm: str, logged: float):
    p = os.path.join(LINEAR_OOF, f"{arm}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    oof, pred = d["oof"], d["test"]
    auc = float(roc_auc_score(_y, oof))
    if round(auc, 6) != round(logged, 6):
        return None          # digit verification failed -> retrain for real
    return oof, pred, auc


def _run_lgb(params, X, Xte, cat_cols):
    import lightgbm as lgb
    p = {**BASE_LGB, **params}
    oof = np.zeros(len(_y))
    te = np.zeros(len(Xte))
    iters = []
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], _y[tr], categorical_feature=cat_cols or "auto",
                          free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va], _y[va], reference=dtr,
                          categorical_feature=cat_cols or "auto", free_raw_data=False)
        b = lgb.train(p, dtr, num_boost_round=6000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        te += b.predict(Xte, num_iteration=b.best_iteration) / len(_FOLDS)
        iters.append(int(b.best_iteration))
    return oof, te, iters


def _run_xgb(params, X, Xte, cat_cols):
    import xgboost as xgb
    p = {**BASE_XGB, **params}
    oof = np.zeros(len(_y))
    te = np.zeros(len(Xte))
    iters = []
    dte = xgb.DMatrix(Xte, nthread=N_THREADS)
    for tr, va in _FOLDS:
        dtr = xgb.DMatrix(X.iloc[tr], _y[tr], nthread=N_THREADS)
        dva = xgb.DMatrix(X.iloc[va], _y[va], nthread=N_THREADS)
        b = xgb.train(p, dtr, num_boost_round=6000, evals=[(dva, "va")],
                      early_stopping_rounds=200, verbose_eval=False)
        rng = (0, b.best_iteration + 1)
        oof[va] = b.predict(dva, iteration_range=rng)
        te += b.predict(dte, iteration_range=rng) / len(_FOLDS)
        iters.append(int(b.best_iteration))
    return oof, te, iters


def _run_cat(params, X, Xte, cat_cols):
    from catboost import CatBoostClassifier, Pool
    p = {**BASE_CAT, **params}
    Xs, Xts = X.copy(), Xte.copy()
    for c in (cat_cols or []):
        Xs[c] = Xs[c].astype(int).astype(str)
        Xts[c] = Xts[c].astype(int).astype(str)
    idx = [Xs.columns.get_loc(c) for c in (cat_cols or [])]
    oof = np.zeros(len(_y))
    te = np.zeros(len(Xts))
    iters = []
    pool_te = Pool(Xts, cat_features=idx)
    for tr, va in _FOLDS:
        m = CatBoostClassifier(**p)
        m.fit(Pool(Xs.iloc[tr], _y[tr], cat_features=idx),
              eval_set=Pool(Xs.iloc[va], _y[va], cat_features=idx),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict_proba(Pool(Xs.iloc[va], cat_features=idx))[:, 1]
        te += m.predict_proba(pool_te)[:, 1] / len(_FOLDS)
        iters.append(int(m.get_best_iteration()))
    return oof, te, iters


_RUNNERS = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat}


def evaluate(config: dict, node_id=None, timeout_s: float = None) -> dict:
    """Solo-node evaluator. Returns the standard
    {"score", "status", "wall_s", "result", "error"} contract; score = -AUC."""
    t0 = time.time()
    try:
        if config.get("kind") != "solo":
            raise ValueError("eval_s6e2_lane3.evaluate handles kind='solo' only, got "
                             f"{config.get('kind')!r}")
        variant = config.get("variant", "raw")
        X = F.build(_train, variant)
        Xte = F.build(_test, variant)
        cat_cols = F.cat_indices(list(X.columns)) if config.get("cat_declare") else []

        arm, logged = _match_known(config)
        reused = _reuse(arm, logged) if arm else None
        if reused is not None:
            oof, te, auc = reused
            iters = None
            reuse_note = f"reused cached OOF of linear-iteration arm '{arm}' (digit-verified)"
        else:
            oof, te, iters = _RUNNERS[config["model"]](config.get("params", {}), X, Xte, cat_cols)
            auc = float(roc_auc_score(_y, oof))
            reuse_note = None

        if node_id is not None:
            hv2.cache_oof(CACHE_DIR, node_id, oof, pred=te)
        fold_aucs = [float(roc_auc_score(_y[va], oof[va])) for _tr, va in _FOLDS]
        return dict(score=-auc, status="evaluated", wall_s=round(time.time() - t0, 2),
                    result=dict(auc=auc, fold_aucs=[round(f, 6) for f in fold_aucs],
                                fold_std=round(float(np.std(fold_aucs)), 6),
                                best_iters=iters, n_features=int(X.shape[1]),
                                reused=reuse_note),
                    error=None)
    except Exception as exc:  # noqa: BLE001 -- harness contract is "never raise"
        return dict(score=None, status="failed", wall_s=round(time.time() - t0, 2),
                    result=None, error=f"{type(exc).__name__}: {exc}")


def auc_of(blended_oof) -> float:
    """metric_fn for blend nodes: the harness minimizes, so return -AUC."""
    return -float(roc_auc_score(_y, blended_oof))


def y() -> np.ndarray:
    return _y
