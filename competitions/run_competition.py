"""Generic tabular pipeline for Kaggle Playground competitions.

Usage:
    uv run python3 competitions/run_competition.py <competition-name>

Reads competitions/<name>/config.yaml for: target_column, id_column,
evaluation_metric, problem_type. Auto-handles regression vs classification,
matches the boosting objective to the metric, runs a 5-fold LGB/XGB/CatBoost
ensemble with OOF weight search, writes a submission, and logs to experiments.json.

Supported metrics: rmse, mae, rmsle, msle, auc/roc_auc, logloss/log_loss,
qwk/kappa, smape, accuracy. Falls back sensibly by problem_type otherwise.
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, cohen_kappa_score, log_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

SEED, N_SPLITS = 42, 5


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = (np.abs(y) + np.abs(p))
    return np.mean(np.where(d == 0, 0.0, 2 * np.abs(p - y) / d)) * 100


def rmse(y, p):
    return mean_squared_error(y, p) ** 0.5


def rmsle(y, p):
    p = np.clip(p, 0, None)
    return mean_squared_error(np.log1p(np.clip(y, 0, None)), np.log1p(p)) ** 0.5


def qwk(y, p):
    return cohen_kappa_score(np.round(y).astype(int), np.round(p).astype(int), weights="quadratic")


# metric -> (kind, direction, fn, higher_is_better)
def resolve_metric(name, ptype):
    m = (name or "").lower().replace("-", "_")
    table = {
        "rmse": ("reg", rmse, False), "root_mean_squared_error": ("reg", rmse, False),
        "mae": ("reg", mean_absolute_error, False),
        "rmsle": ("reg_log", rmsle, False), "msle": ("reg_log", rmsle, False),
        "root_mean_squared_log_error": ("reg_log", rmsle, False),
        "smape": ("reg", smape, False),
        "auc": ("clf", roc_auc_score, True), "roc_auc": ("clf", roc_auc_score, True),
        "logloss": ("clf", log_loss, False), "log_loss": ("clf", log_loss, False),
        "qwk": ("ord", qwk, True), "kappa": ("ord", qwk, True),
        "quadratic_weighted_kappa": ("ord", qwk, True),
        "accuracy": ("clf_label", accuracy_score, True),
    }
    if m in table:
        return (m,) + table[m]
    # fallback by problem type
    if ptype and "class" in ptype:
        return ("auc", "clf", roc_auc_score, True)
    return ("rmse", "reg", rmse, False)


def _expand_dates(Xtr, Xte, cols):
    """Detect date-like string columns, expand into numeric parts, drop original."""
    for c in list(cols):
        s = Xtr[c].astype(str)
        if not (("date" in c.lower()) or s.str.match(r"\d{4}-\d{2}-\d{2}").mean() > 0.8):
            continue
        for frame in (Xtr, Xte):
            d = pd.to_datetime(frame[c], errors="coerce")
            frame[f"{c}_year"] = d.dt.year
            frame[f"{c}_month"] = d.dt.month
            frame[f"{c}_day"] = d.dt.day
            frame[f"{c}_dow"] = d.dt.dayofweek
            frame[f"{c}_doy"] = d.dt.dayofyear
            frame.drop(columns=[c], inplace=True)
        cols.remove(c)
    return Xtr, Xte


def build_features(train, test, target, idc):
    feats = [c for c in train.columns if c not in (idc, target)]
    Xtr, Xte = train[feats].copy(), test[feats].copy()
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]
    Xtr, Xte = _expand_dates(Xtr, Xte, cat)
    feats = list(Xtr.columns)
    for c in cat:
        # label encode using union of categories
        cats = pd.Index(pd.concat([Xtr[c], Xte[c]]).astype(str).unique())
        mp = {v: i for i, v in enumerate(cats)}
        Xtr[c] = Xtr[c].astype(str).map(mp).astype("int32")
        Xte[c] = Xte[c].astype(str).map(mp).astype("int32")
    # simple missing fill (median)
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    return Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32), feats


def main(comp):
    root = os.path.dirname(os.path.abspath(__file__))
    cdir = os.path.join(root, comp)
    cfg = yaml.safe_load(open(os.path.join(cdir, "config.yaml")))
    target = cfg["target_column"]
    idc = cfg["id_column"]
    ptype = cfg.get("problem_type", "")
    mname, kind, mfn, higher = resolve_metric(cfg.get("evaluation_metric"), ptype)
    data = os.path.join(cdir, "data")
    sub_dir = os.path.join(cdir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(data, cfg.get("train_file", "train.csv")))
    test = pd.read_csv(os.path.join(data, cfg.get("test_file", "test.csv")))
    ss = pd.read_csv(os.path.join(data, cfg.get("sample_submission_file", "sample_submission.csv")))
    print(f"[{comp}] metric={mname} kind={kind} target={target} "
          f"train={train.shape} test={test.shape}")

    X, Xtest, feats = build_features(train, test, target, idc)
    y = train[target].to_numpy()
    is_clf = kind in ("clf", "clf_label")
    y_fit = y.astype(float)
    if kind == "reg_log":
        y_fit = np.log1p(np.clip(y.astype(float), 0, None))

    # CV splitter
    if is_clf or kind == "ord":
        splitter = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
        strat = y.astype(int)
        folds = list(splitter.split(X, strat))
    else:
        ybin = pd.qcut(pd.Series(y).rank(method="first"), q=min(10, N_SPLITS * 2), labels=False)
        folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, ybin))

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier, CatBoostRegressor

    def make_models():
        if is_clf:
            return {
                "LGB": lgb.LGBMClassifier(objective="binary", n_estimators=2000, learning_rate=0.03,
                                          num_leaves=63, subsample=0.8, subsample_freq=1,
                                          colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
                                          n_jobs=-1, verbose=-1),
                "XGB": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                         n_estimators=2000, learning_rate=0.03, max_depth=6,
                                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                         random_state=SEED, n_jobs=-1, tree_method="hist"),
                "CAT": CatBoostClassifier(loss_function="Logloss", iterations=2500, learning_rate=0.03,
                                          depth=7, l2_leaf_reg=3.0, random_seed=SEED, verbose=False),
            }
        obj_lgb = "regression_l1" if mname == "mae" else "regression"
        obj_xgb = "reg:absoluteerror" if mname == "mae" else "reg:squarederror"
        loss_cat = "MAE" if mname == "mae" else "RMSE"
        return {
            "LGB": lgb.LGBMRegressor(objective=obj_lgb, n_estimators=2000, learning_rate=0.03,
                                     num_leaves=63, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
                                     n_jobs=-1, verbose=-1),
            "XGB": xgb.XGBRegressor(objective=obj_xgb, n_estimators=2000, learning_rate=0.03,
                                    max_depth=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                    random_state=SEED, n_jobs=-1, tree_method="hist"),
            "CAT": CatBoostRegressor(loss_function=loss_cat, iterations=2500, learning_rate=0.03,
                                     depth=7, l2_leaf_reg=3.0, random_seed=SEED, verbose=False),
        }

    NAMES = ["LGB", "XGB", "CAT"]
    oof = {n: np.zeros(len(y)) for n in NAMES}
    pred = {n: np.zeros(len(Xtest)) for n in NAMES}
    t0 = time.time()
    for f, (tr, va) in enumerate(folds):
        for n, m in make_models().items():
            m.fit(X[tr], y_fit[tr])
            if is_clf:
                ov = m.predict_proba(X[va])[:, 1]
                tv = m.predict_proba(Xtest)[:, 1]
            else:
                ov, tv = m.predict(X[va]), m.predict(Xtest)
                if kind == "reg_log":
                    ov, tv = np.expm1(ov), np.expm1(tv)
            oof[n][va] = ov
            pred[n] += tv / len(folds)

    def score(p):
        if mname == "logloss":
            return mfn(y, np.clip(p, 1e-6, 1 - 1e-6))
        if kind == "clf_label":
            return mfn(y, np.round(p))
        return mfn(y, p)

    per = {n: score(oof[n]) for n in NAMES}
    print(f"  per-model {mname}: " + "  ".join(f"{n}={per[n]:.5f}" for n in NAMES) + f"  [{time.time()-t0:.0f}s]")

    # weight search
    oofs = np.stack([oof[n] for n in NAMES], 1)
    best_w, best_s = None, (-1e18 if higher else 1e18)
    for w0 in np.arange(0, 1.001, 0.1):
        for w1 in np.arange(0, 1.001 - w0, 0.1):
            w2 = 1 - w0 - w1
            if w2 < -1e-9:
                continue
            s = score(oofs @ np.array([w0, w1, w2]))
            if (s > best_s) if higher else (s < best_s):
                best_s, best_w = s, (w0, w1, w2)
    print(f"  BEST blend {dict(zip(NAMES, np.round(best_w,2)))} -> {mname}={best_s:.5f}")

    w = np.array(best_w)
    final = np.stack([pred[n] for n in NAMES], 1) @ w
    # format output per problem
    out_col = ss.columns[1]
    if kind == "ord":
        final = np.clip(np.round(final), y.min(), y.max())
    elif kind == "clf_label":
        final = np.round(final).astype(int)
    elif not is_clf and mname in ("rmsle", "smape", "mae", "rmse"):
        if y.min() >= 0:
            final = np.clip(final, 0, None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({ss.columns[0]: test[idc], out_col: final})
    path = os.path.join(sub_dir, f"sub_generic_{best_s:.5f}_{stamp}.csv")
    sub.to_csv(path, index=False)
    print(f"  wrote {os.path.basename(path)}  ({len(sub)} rows)")

    # log (canonical v2 schema — see docs/superpowers/specs/2026-07-03-kaggle-report-skill-design.md §6)
    exp = os.path.join(cdir, "experiments.json")
    hist = json.load(open(exp)) if os.path.exists(exp) else []
    hist.append(dict(
        schema_version=2,
        experiment_id=len(hist) + 1,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model="generic LGB+XGB+CAT blend",
        metric=mname,
        direction=("maximize" if higher else "minimize"),
        score=round(float(best_s), 5),
        n_features=len(feats),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(float(per[n]), 5)) for n in NAMES],
        ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 2) for x in best_w])),
                      score=round(float(best_s), 5)),
        submission=os.path.basename(path),
    ))
    json.dump(hist, open(exp, "w"), indent=2)
    return dict(comp=comp, metric=mname, score=round(float(best_s), 5),
                weights=dict(zip(NAMES, [round(float(x), 2) for x in best_w])))


if __name__ == "__main__":
    res = main(sys.argv[1])
    print("RESULT " + json.dumps(res))
