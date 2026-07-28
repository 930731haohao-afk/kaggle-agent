"""Member training library: each member produces OOF + test preds, cached as npz.

Usage: uv run python3 competitions/cat-in-the-dat/scripts/members.py <member_name>
Fixed 5-fold StratifiedKFold (seed 42) from data_proc/folds.npy. AUC metric.
Threads capped at 10; LGBM deterministic.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

COMP = Path("competitions/cat-in-the-dat")
PROC = COMP / "data_proc"
MEMBERS = PROC / "members"
MEMBERS.mkdir(exist_ok=True)

y = np.load(PROC / "y.npy")
folds = np.load(PROC / "folds.npy")
N_FOLDS = 5
THREADS = 10


def run_cv(fit_predict, X_tr, X_te):
    """fit_predict(Xtr, ytr, Xva, Xte) -> (va_pred, te_pred). Returns oof, test (fold-avg)."""
    oof = np.zeros(len(y))
    te_pred = np.zeros(X_te.shape[0])
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        t0 = time.time()
        va_p, te_p = fit_predict(X_tr[tr_idx], y[tr_idx], X_tr[va_idx], X_te)
        oof[va_idx] = va_p
        te_pred += te_p / N_FOLDS
        print(f"  fold {f}: auc {roc_auc_score(y[va_idx], va_p):.6f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  OOF AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    return oof, te_pred


def save(name, oof, te_pred):
    np.savez_compressed(MEMBERS / f"{name}.npz", oof=oof, test=te_pred)
    print(f"saved {name}: oof auc {roc_auc_score(y, oof):.6f}")


# ---------- LR on OHE ----------
def lr_ohe(C=0.1, solver="lbfgs", max_iter=2000):
    X_tr = sparse.load_npz(PROC / "ohe_train.npz")
    X_te = sparse.load_npz(PROC / "ohe_test.npz")

    def fp(Xtr, ytr, Xva, Xte):
        m = LogisticRegression(C=C, solver=solver, max_iter=max_iter, tol=1e-5)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]

    return run_cv(fp, X_tr, X_te)


# ---------- LGBM native categorical ----------
LGB_BASE = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50, num_threads=THREADS, seed=42,
    deterministic=True, force_row_wise=True, verbosity=-1,
)
CAT_COLS = [f"nom_{i}" for i in range(10)] + ["day", "month"]


def lgb_cat(params=None, num_boost_round=3000, seed=42):
    import lightgbm as lgb
    p = dict(LGB_BASE)
    if params:
        p.update(params)
    p["seed"] = seed
    df_tr = pd.read_parquet(PROC / "gbdt_train.parquet")
    df_te = pd.read_parquet(PROC / "gbdt_test.parquet")
    use = [c for c in df_tr.columns if c not in ("day_sin", "day_cos", "month_sin", "month_cos")]
    df_tr, df_te = df_tr[use], df_te[use]

    oof = np.zeros(len(y))
    te_pred = np.zeros(len(df_te))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        t0 = time.time()
        dtr = lgb.Dataset(df_tr.iloc[tr_idx], y[tr_idx], categorical_feature=CAT_COLS)
        dva = lgb.Dataset(df_tr.iloc[va_idx], y[va_idx], categorical_feature=CAT_COLS, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = m.predict(df_tr.iloc[va_idx], num_iteration=m.best_iteration)
        te_pred += m.predict(df_te, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {f}: auc {roc_auc_score(y[va_idx], oof[va_idx]):.6f} "
              f"iters {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  OOF AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    return oof, te_pred


# ---------- LGBM with fold-safe target encoding ----------
def add_te(df_tr, df_te, cols, tr_idx, va_idx, smoothing=20):
    """Fold-safe smoothed TE: stats from tr_idx only; returns TE cols for va and test."""
    out_va, out_te = {}, {}
    prior = y[tr_idx].mean()
    for c in cols:
        g = pd.DataFrame({"k": df_tr[c].iloc[tr_idx].values, "y": y[tr_idx]}).groupby("k")["y"].agg(["mean", "count"])
        te_map = (g["mean"] * g["count"] + prior * smoothing) / (g["count"] + smoothing)
        out_va[f"te_{c}"] = df_tr[c].iloc[va_idx].map(te_map).fillna(prior).values
        out_te[f"te_{c}"] = df_te[c].map(te_map).fillna(prior).values
    return out_va, out_te


def lgb_te(params=None, num_boost_round=3000, seed=42, smoothing=20):
    import lightgbm as lgb
    p = dict(LGB_BASE)
    if params:
        p.update(params)
    p["seed"] = seed
    df_tr = pd.read_parquet(PROC / "gbdt_train.parquet")
    df_te = pd.read_parquet(PROC / "gbdt_test.parquet")
    te_cols = [f"nom_{i}" for i in range(10)] + ["ord_5", "day", "month"]
    base_cols = [c for c in df_tr.columns if c not in [f"nom_{i}" for i in range(5, 10)]]

    oof = np.zeros(len(y))
    te_pred = np.zeros(len(df_te))
    for f in range(N_FOLDS):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        t0 = time.time()
        # inner split of training fold for TE on train rows (avoid self-leak): 4 sub-folds
        sub = folds[tr_idx]  # reuse global fold ids (4 distinct values on tr rows)
        Xtr = df_tr[base_cols].iloc[tr_idx].copy()
        for c in te_cols:
            Xtr[f"te_{c}"] = np.nan
        for sf in sorted(set(sub)):
            s_tr = tr_idx[sub != sf]
            s_va = tr_idx[sub == sf]
            va_map, _ = add_te(df_tr, df_te.iloc[:1], te_cols, s_tr, s_va, smoothing)
            for c in te_cols:
                Xtr.loc[df_tr.index[s_va], f"te_{c}"] = va_map[f"te_{c}"]
        va_map, te_map_ = add_te(df_tr, df_te, te_cols, tr_idx, va_idx, smoothing)
        Xva = df_tr[base_cols].iloc[va_idx].copy()
        Xte = df_te[base_cols].copy()
        for c in te_cols:
            Xva[f"te_{c}"] = va_map[f"te_{c}"]
            Xte[f"te_{c}"] = te_map_[f"te_{c}"]
        low_cats = [f"nom_{i}" for i in range(5)]
        dtr = lgb.Dataset(Xtr, y[tr_idx], categorical_feature=low_cats)
        dva = lgb.Dataset(Xva, y[va_idx], categorical_feature=low_cats, reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = m.predict(Xva, num_iteration=m.best_iteration)
        te_pred += m.predict(Xte, num_iteration=m.best_iteration) / N_FOLDS
        print(f"  fold {f}: auc {roc_auc_score(y[va_idx], oof[va_idx]):.6f} "
              f"iters {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  OOF AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    return oof, te_pred


REGISTRY = {
    "lr_c010": lambda: lr_ohe(C=0.1),
    "lr_c005": lambda: lr_ohe(C=0.05),
    "lr_c020": lambda: lr_ohe(C=0.2),
    "lgb_base": lambda: lgb_cat(),
    "lgb_te": lambda: lgb_te(),
}

if __name__ == "__main__":
    name = sys.argv[1]
    print(f"=== {name} ===", flush=True)
    t0 = time.time()
    oof, te = REGISTRY[name]()
    save(name, oof, te)
    print(f"total {time.time()-t0:.0f}s")
