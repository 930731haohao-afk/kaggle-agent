"""tree_search/eval_s3e3.py — per-competition evaluator for playground-series-s3e3
(Employee Attrition, binary classification, ROC-AUC metric, maximize-better).

Phase D-2 (first harness_v2 sweep comp). Two node kinds, same schema as
eval_s3e14.py/eval_s3e5.py:

  1. solo  — {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
             features:{drop:[...], native_cat:bool}}
     Trains one classifier with 5-fold StratifiedKFold(Attrition, shuffle, seed=42) —
     EXACT same CV as competitions/playground-series-s3e3/scripts/train_v2.py and
     iterate_round*.py. Cheap (~1-5s on this 1677-row dataset). If `node_id` is given,
     its OOF/test prediction vector is cached via harness_v2.cache_oof to
     tree_search/cache_s3e3/solo_<node_id>.npz so blend nodes never retrain.
     `features.native_cat=True` (model="cat" only) switches to the raw-string
     categorical feature set from iterate_round3/4 (cat_features passed to CatBoost
     natively instead of label/freq pre-encoded columns).

  2. blend — {kind:"blend", members:[<solo node id>, ...],
             weight_search:"dirichlet", space:"prob"|"rank"}
     Loads each member's cached OOF vector via harness_v2.load_oof (recommendation #1's
     OOF-cache contract), weight-searches for the blend that maximizes ROC-AUC. Two
     search spaces: "prob" (default) dispatches straight to harness_v2.eval_blend (the
     ensemble-default node-space machinery this sweep is testing); "rank" is a small
     local re-implementation of the same dirichlet search operating on a
     rank-transformed member matrix (STATUS.md's own "next idea": re-check whether the
     linear-iteration exp#7 rank-average-beats-prob-blend result (+<score>) holds up
     or is noise). No retraining either way — blend nodes cost a fraction of a second.

SIGN CONVENTION: harness.py/harness_v2.py assume lower-is-better scores; AUC is
maximize-better, so every score this module hands to add_root/add_node is `-AUC`.
`result["auc"]` (and all prints in run_s3e3.py) carry the real, human-readable AUC.

Reuses competitions/playground-series-s3e3/scripts/features.py UNMODIFIED for the
label/freq-encoded feature set (build_features/feature_columns/fit_encoders) so scores
are directly comparable to STATUS.md's numbers (root tuned-LGB solo ~<score>,
linear-iteration best 6-way blend, rank-average, <score>).
"""
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "playground-series-s3e3")
_SCRIPTS_DIR = os.path.join(_COMP_DIR, "scripts")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _HERE)
from features import build_features, feature_columns, fit_encoders  # noqa: E402
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s3e3")
TARGET, ID = "Attrition", "id"
N_SPLITS, SEED = 5, 42

CAT_COLS = ["BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"]
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18"]  # constant, zero-variance
SATISFACTION_COLS = ["EnvironmentSatisfaction", "JobSatisfaction",
                      "RelationshipSatisfaction", "WorkLifeBalance", "JobInvolvement"]

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[TARGET].to_numpy(np.int64)

# --- label/freq-encoded feature set (features.py, unmodified) ---
_enc = fit_encoders(_train)
_Xtr_full = build_features(_train, _enc)
_Xte_full = build_features(_test, _enc)
ALL_FEATURES = feature_columns(_Xtr_full)


# --- native (raw-string categorical) feature set, ported verbatim from
# iterate_round3_catboost_native.py / iterate_round4_add_catnative_blend.py's
# build_native_features -- CatBoost-only, cat_features passed via Pool at fit time. ---
def _build_native(df):
    out = df.copy()
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")
    for c in CAT_COLS:
        out[c] = out[c].astype(str)
    out["role_tenure_ratio"] = out["YearsInCurrentRole"] / out["YearsAtCompany"].replace(0, np.nan)
    out["mgr_tenure_ratio"] = out["YearsWithCurrManager"] / out["YearsAtCompany"].replace(0, np.nan)
    out["promo_ratio"] = out["YearsSinceLastPromotion"] / out["YearsAtCompany"].replace(0, np.nan)
    out["company_tenure_ratio"] = out["YearsAtCompany"] / out["TotalWorkingYears"].replace(0, np.nan)
    for c in ["role_tenure_ratio", "mgr_tenure_ratio", "promo_ratio", "company_tenure_ratio"]:
        out[c] = out[c].fillna(0.0)
    out["income_per_joblevel"] = out["MonthlyIncome"] / out["JobLevel"].replace(0, np.nan)
    out["income_per_joblevel"] = out["income_per_joblevel"].fillna(out["MonthlyIncome"])
    out["income_per_year_worked"] = out["MonthlyIncome"] / (out["TotalWorkingYears"] + 1)
    out["satisfaction_avg"] = out[SATISFACTION_COLS].mean(axis=1)
    out["satisfaction_min"] = out[SATISFACTION_COLS].min(axis=1)
    out["age_at_join"] = out["Age"] - out["TotalWorkingYears"]
    out["companies_per_year"] = out["NumCompaniesWorked"] / (out["Age"] - 18).clip(lower=1)
    return out


_Xtr_native_full = _build_native(_train)
_Xte_native_full = _build_native(_test)
NATIVE_FEATURES = [c for c in _Xtr_native_full.columns if c not in (ID, TARGET)]

# MANDATORY: identical fold assignment to scripts/train_v2.py & iterate_round*.py (folds
# depend only on _y / n_samples, so this is byte-identical regardless of which feature
# set -- encoded or native -- a given node trains on).
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
_FOLDS = list(_skf.split(np.zeros(len(_y)), _y))


def auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


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


def get_native_feature_frame(drop=None):
    drop = set(drop or [])
    unknown = drop - set(NATIVE_FEATURES)
    if unknown:
        raise ValueError(f"unknown native feature(s) in drop list: {unknown}")
    feats = [c for c in NATIVE_FEATURES if c not in drop]
    if not feats:
        raise ValueError("drop list removed every native feature")
    cat_names = [c for c in CAT_COLS if c in feats]
    return _Xtr_native_full[feats].copy(), _Xte_native_full[feats].copy(), feats, cat_names


# ---------------------------------------------------------------------------
# solo-model runners (mirrors scripts/train_v2.py / iterate_round*.py exactly)
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xtest):
    import lightgbm as lgb
    p = dict(objective="binary", metric="auc", n_jobs=-1, verbose=-1, random_state=SEED)
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = lgb.LGBMClassifier(n_estimators=n_est, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])],
              callbacks=[lgb.early_stopping(esr, verbose=False)])
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def _run_xgb(params, X, Xtest):
    import xgboost as xgb
    p = dict(objective="binary:logistic", n_jobs=-1, random_state=SEED, eval_metric="auc")
    p.update(params)
    n_est = p.pop("n_estimators", 2000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtest))
    for tr, va in _FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X[tr], _y[tr], eval_set=[(X[va], _y[va])], verbose=False)
        oof[va] = m.predict_proba(X[va])[:, 1]
        pred += m.predict_proba(Xtest)[:, 1] / N_SPLITS
    return oof, pred


def _run_cat(params, Xdf, Xtestdf, cat_names):
    from catboost import CatBoostClassifier, Pool
    p = dict(loss_function="Logloss", eval_metric="AUC", random_seed=SEED,
             thread_count=4, verbose=False, allow_writing_files=False)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 2000))
    esr = p.pop("early_stopping_rounds", 150)
    cat_idx = [Xdf.columns.get_loc(c) for c in cat_names] if cat_names else []
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtestdf))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(iterations=n_est, **p)
        m.fit(Pool(Xdf.iloc[tr], _y[tr], cat_features=cat_idx),
              eval_set=Pool(Xdf.iloc[va], _y[va], cat_features=cat_idx),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(Xdf.iloc[va])[:, 1]
        pred += m.predict_proba(Xtestdf)[:, 1] / N_SPLITS
    return oof, pred


def evaluate_solo(config):
    feat_cfg = config.get("features") or {}
    drop = feat_cfg.get("drop", [])
    model = config["model"]
    params = dict(config.get("params") or {})
    native_cat = bool(feat_cfg.get("native_cat", False))

    if model == "cat" and native_cat:
        Xdf, Xtestdf, feats, cat_names = get_native_feature_frame(drop)
        oof, pred = _run_cat(params, Xdf, Xtestdf, cat_names)
    elif model == "cat":
        Xdf, Xtestdf, feats = get_feature_frame(drop)
        oof, pred = _run_cat(params, Xdf, Xtestdf, [])
    elif model == "lgb":
        Xdf, Xtestdf, feats = get_feature_frame(drop)
        oof, pred = _run_lgb(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    elif model == "xgb":
        Xdf, Xtestdf, feats = get_feature_frame(drop)
        oof, pred = _run_xgb(params, Xdf.to_numpy(np.float32), Xtestdf.to_numpy(np.float32))
    else:
        raise ValueError(f"unknown model type {model!r}")

    score = auc(_y, oof)
    return oof, pred, score, feats


# ---------------------------------------------------------------------------
# ensemble (blend) node: weight-search over harness_v2-CACHED member OOF vectors, no
# retraining. "prob" space dispatches straight to harness_v2.eval_blend (recommendation
# #1's generic ensemble machinery -- the actual thing this sweep is testing); "rank" is
# a small local re-implementation of the same dirichlet search on a rank-transformed
# member matrix (per-member preprocessing that eval_blend's post-hoc linear-combine
# design doesn't support, needed for the STATUS.md "next idea" rank-vs-prob re-check).
# ---------------------------------------------------------------------------
def _neg_auc(vec):
    return -auc(_y, vec)


def _rank_weight_search(oofs, seed=42, k=1500):
    """Dirichlet search identical in structure to harness_v2.eval_blend's, but scores
    directly on AUC (maximize) over an already rank-transformed member matrix."""
    n = oofs.shape[1]
    rng = np.random.default_rng(seed)
    candidates = [np.eye(n)[i] for i in range(n)]
    candidates.append(np.full(n, 1.0 / n))
    candidates += list(rng.dirichlet(np.ones(n), size=k))
    best_w, best_s = None, -1e18
    for w in candidates:
        s = auc(_y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=max(k // 3, 50)):
        s = auc(_y, oofs @ w)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    method = config.get("weight_search", "dirichlet")
    space = config.get("space", "prob")

    if space == "prob":
        best_w, best_neg, oofs = hv2.eval_blend(CACHE_DIR, members, _neg_auc, weight_search=method)
        best_score = -best_neg
    elif space == "rank":
        oofs_raw = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)
        n = len(_y)
        oofs = np.stack([rankdata(oofs_raw[:, i]) / n for i in range(oofs_raw.shape[1])], axis=1)
        if method != "dirichlet":
            raise ValueError(f"rank-space blend only supports weight_search='dirichlet', got {method!r}")
        best_w, best_score = _rank_weight_search(oofs)
    else:
        raise ValueError(f"unknown blend space {space!r}")

    return dict(members=members, weights=[round(float(w), 4) for w in best_w],
                method=method, space=space, auc=round(best_score, 6)), best_score


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 120) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises: a timeout or any
    exception is captured as status="failed" so the tree-search loop can keep going.

    score is `-AUC` (harness sign convention: lower-is-better). result["auc"] is the
    real, human-readable, higher-is-better AUC -- use THAT for all reporting/printouts."""
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
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, auc=score)
            wall = time.time() - t0
            result = {"n_feats": len(feats), "auc": round(score, 6)}
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
