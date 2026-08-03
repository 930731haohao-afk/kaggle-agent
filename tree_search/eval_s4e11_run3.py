"""tree_search/eval_s4e11_run3.py -- per-competition evaluator for playground-series-s4e11
(Depression, binary classification, ACCURACY metric, MAXIMIZE-better).

Written fresh for this run. It deliberately does not read, import or reuse any artifact of
a previous run of this competition (lane isolation + SKILL.md's "never read the answers
this competition already produced" rule).

Two node kinds, the standard v2/v3 schema:

  1. solo  -- {kind:"solo", model:"lgb"|"xgb"|"cat", params:{...},
               features:{drop:[...], variant:{...}}}
     Trains one classifier on the canonical StratifiedKFold(5, shuffle, seed=42) folds
     from competitions/.../scripts/features.py (imported, never re-implemented).
     `features.variant` is forwarded to features.build() as keyword arguments, so a
     feature-set ablation is just another node config.

  2. blend -- {kind:"blend", members:[<solo node id>...], weight_search:"dirichlet"}
     Loads the members' cached OOFs; no retraining.

THE METRIC, AND WHY IT IS NOT THE OBVIOUS ONE.
Accuracy scores hard labels, so the decision threshold is a fitted parameter of the model
and a blend's weights are another. Fitting either on the same OOF vector that then scores
them is optimistic -- the effect that flipped the ranking on aug-2022 (fitted-weight blend
"won" by +0.001 AUC in-sample and lost to the root solo once weights were re-fit
leave-fold-out). Every score this module hands to the harness is therefore HONEST:

  * solo  -- for each fold, the threshold is chosen on the other four folds' OOF rows and
             applied to this fold.
  * blend -- for each fold, BOTH the weights and the threshold are fitted on the other
             four folds and applied to this fold.

`result["insample_acc"]` carries the corresponding in-sample number, so every node also
reports how optimistic it would have looked; `result["optimism"]` is the difference.

SIGN CONVENTION: the harness assumes LOWER-is-better; accuracy is MAXIMIZE-better, so the
score handed to add_root/add_node is `-honest_accuracy`. `result["acc"]` is the real,
human-readable, higher-is-better number -- use THAT for reporting.
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
import features as F  # noqa: E402
import harness_v2 as hv2  # noqa: E402

DATA = os.path.join(_COMP_DIR, "data")
CACHE_DIR = os.path.join(_HERE, "cache_s4e11_run3")
os.makedirs(CACHE_DIR, exist_ok=True)
NUM_THREADS = 10
SEED = F.SEED

_train = pd.read_csv(f"{DATA}/train.csv")
_test = pd.read_csv(f"{DATA}/test.csv")
_y = _train[F.TARGET].astype(int)
_FOLDS = F.make_folds(_y)
_YNP = _y.to_numpy()

_VARIANT_CACHE = {}

DEFAULT_VARIANT = dict(rare_min_count=10, keep_block_originals=True, add_counts=True,
                       add_interactions=False, use_name=True)


class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


# ---------------------------------------------------------------------------
# feature frames (variant-cached: building is deterministic and target-free)
# ---------------------------------------------------------------------------
def get_feature_frame(feat_cfg):
    feat_cfg = feat_cfg or {}
    variant = dict(DEFAULT_VARIANT)
    variant.update(feat_cfg.get("variant") or {})
    key = tuple(sorted(variant.items()))
    if key not in _VARIANT_CACHE:
        Xtr, Xte, _, cols, cats = F.build(_train, _test, **variant)
        _VARIANT_CACHE[key] = (Xtr, Xte, cols, cats)
    Xtr, Xte, cols, cats = _VARIANT_CACHE[key]

    drop = set(feat_cfg.get("drop") or [])
    unknown = drop - set(cols)
    if unknown:
        raise ValueError(f"unknown feature(s) in drop list: {sorted(unknown)}")
    feats = [c for c in cols if c not in drop]
    if not feats:
        raise ValueError("drop list removed every feature")
    cat_feats = [c for c in cats if c not in drop]
    return Xtr[feats], Xte[feats], feats, cat_feats


# ---------------------------------------------------------------------------
# solo runners
# ---------------------------------------------------------------------------
def _run_lgb(params, X, Xte, cat_feats):
    import lightgbm as lgb
    p = dict(objective="binary", metric="binary_logloss", learning_rate=0.03,
             num_leaves=63, min_child_samples=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=1.0,
             verbose=-1, num_threads=NUM_THREADS, deterministic=True,
             force_row_wise=True, seed=SEED)
    p.update(params)
    n_rounds = p.pop("num_boost_round", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xte))
    for tr, va in _FOLDS:
        dtr = lgb.Dataset(X.iloc[tr], label=_y.iloc[tr], categorical_feature=cat_feats)
        dva = lgb.Dataset(X.iloc[va], label=_y.iloc[va], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(esr, verbose=False),
                                 lgb.log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(Xte, num_iteration=m.best_iteration) / len(_FOLDS)
    return oof, pred


def _run_xgb(params, X, Xte, cat_feats):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.03,
             max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
             reg_lambda=1.0, min_child_weight=10, random_state=SEED,
             n_jobs=NUM_THREADS, tree_method="hist", enable_categorical=True,
             max_cat_to_onehot=1)
    p.update(params)
    n_est = p.pop("n_estimators", 3000)
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xte))
    for tr, va in _FOLDS:
        m = xgb.XGBClassifier(n_estimators=n_est, early_stopping_rounds=esr, **p)
        m.fit(X.iloc[tr], _y.iloc[tr], eval_set=[(X.iloc[va], _y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / len(_FOLDS)
    return oof, pred


def _run_cat(params, X, Xte, cat_feats):
    from catboost import CatBoostClassifier
    # CatBoost rejects NaN inside cat_features (experience.md, s6e7) -- the feature builder
    # already folds every missing categorical level into "__RARE__", but cast to str so a
    # pandas `category` dtype cannot smuggle one back in.
    Xc = X.copy()
    Xtec = Xte.copy()
    for c in cat_feats:
        Xc[c] = Xc[c].astype(str)
        Xtec[c] = Xtec[c].astype(str)
    p = dict(loss_function="Logloss", eval_metric="Logloss", learning_rate=0.05, depth=6,
             l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
             allow_writing_files=False, thread_count=NUM_THREADS)
    p.update(params)
    n_est = p.pop("iterations", p.pop("n_estimators", 3000))
    esr = p.pop("early_stopping_rounds", 100)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtec))
    for tr, va in _FOLDS:
        m = CatBoostClassifier(iterations=n_est, cat_features=cat_feats, **p)
        m.fit(Xc.iloc[tr], _y.iloc[tr], eval_set=(Xc.iloc[va], _y.iloc[va]),
              early_stopping_rounds=esr, verbose=False)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        pred += m.predict_proba(Xtec)[:, 1] / len(_FOLDS)
    return oof, pred


def evaluate_solo(config):
    X, Xte, feats, cat_feats = get_feature_frame(config.get("features"))
    model = config["model"]
    params = dict(config.get("params") or {})
    runner = {"lgb": _run_lgb, "xgb": _run_xgb, "cat": _run_cat}.get(model)
    if runner is None:
        raise ValueError(f"unknown model type {model!r}")
    oof, pred = runner(params, X, Xte, cat_feats)
    honest, insample, thr = F.honest_accuracy(oof, _YNP, _FOLDS)
    return oof, pred, honest, insample, thr, feats


# ---------------------------------------------------------------------------
# blend node -- weights AND threshold fitted leave-fold-out
# ---------------------------------------------------------------------------
def _fit_weights(oofs_sub, y_sub, k=800, ascent_rounds=3, rng=None):
    """Dirichlet search + coordinate ascent, maximizing best-threshold accuracy."""
    rng = rng or np.random.default_rng(SEED)
    n = oofs_sub.shape[0]

    def sc(w):
        return F.best_threshold(oofs_sub.T @ w, y_sub)[1]

    cands = [np.full(n, 1.0 / n)]
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        cands.append(e)
    cands.extend(rng.dirichlet(np.ones(n), size=k))
    scores = [sc(w) for w in cands]
    best_i = int(np.argmax(scores))
    w, s = np.asarray(cands[best_i], float), scores[best_i]

    # coordinate ascent (E-2: coarse Dirichlet grids silently tie without refinement)
    for _ in range(ascent_rounds):
        improved = False
        for i in range(n):
            for delta in (0.08, 0.04, 0.02, -0.02, -0.04, -0.08):
                w2 = w.copy()
                w2[i] = max(0.0, w2[i] + delta)
                tot = w2.sum()
                if tot <= 0:
                    continue
                w2 /= tot
                s2 = sc(w2)
                if s2 > s + 1e-12:
                    w, s, improved = w2, s2, True
        if not improved:
            break
    return w, s


def evaluate_blend(config):
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    oofs = np.vstack([hv2.load_oof(CACHE_DIR, m) for m in members])
    k = int(config.get("k", 800))

    # in-sample: weights and threshold fitted on the whole OOF vector
    w_in, acc_in = _fit_weights(oofs, _YNP, k=k)
    thr_in, _ = F.best_threshold(oofs.T @ w_in, _YNP)

    # honest: for each fold, weights AND threshold come from the other four folds
    correct = 0
    for _, va in _FOLDS:
        mask = np.ones(len(_YNP), dtype=bool)
        mask[va] = False
        w_f, _ = _fit_weights(oofs[:, mask], _YNP[mask], k=k)
        t_f, _ = F.best_threshold(oofs[:, mask].T @ w_f, _YNP[mask])
        p_va = oofs[:, va].T @ w_f
        correct += int(((p_va >= t_f).astype(np.int64) == _YNP[va]).sum())
    honest = correct / len(_YNP)

    result = dict(members=list(members),
                  weights=[round(float(x), 4) for x in w_in],
                  threshold=round(float(thr_in), 4),
                  acc=round(honest, 6), insample_acc=round(acc_in, 6),
                  optimism=round(acc_in - honest, 6), k=k)
    return result, honest


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config: dict, node_id: int = None, timeout_s: int = 900) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises.

    score is `-honest_accuracy` (harness sign convention: lower-is-better).
    result["acc"] is the real, higher-is-better honest accuracy -- report THAT.
    """
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    if have_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, honest, insample, thr, feats = evaluate_solo(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, acc=honest)
            result = {"n_feats": len(feats), "acc": round(honest, 6),
                      "insample_acc": round(insample, 6),
                      "optimism": round(insample - honest, 6),
                      "insample_thr": round(thr, 4)}
            return dict(status="evaluated", score=round(-honest, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        elif kind == "blend":
            result, honest = evaluate_blend(config)
            return dict(status="evaluated", score=round(-honest, 6),
                        wall_s=round(time.time() - t0, 1), result=result, error=None)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - the evaluator must never kill the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
