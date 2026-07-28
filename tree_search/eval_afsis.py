"""tree_search/eval_afsis.py — per-competition evaluator for afsis-soil-properties
(Africa Soil Property Prediction, 5-target regression, MCRMSE metric, minimize-better).

p >> n mid-infrared chemometrics: 1157 train rows x 3563 spectral bands (CO2 band
2352-2380 cm-1 dropped) + 15 spatial covariates + Depth; 727 test rows.

Node schema (same two kinds as every other eval_*.py in this directory):

  1. solo — {kind:"solo", model:"krr"|"svr"|"pls"|"lgbpca"|"mean",
             variant:"raw"|"snv"|"sg1"|"sg2"|"sg1w11"|"spatial",
             use_spatial:bool, spatial_weight:float, params:{...}}

     Validation is a NESTED GroupKFold: outer GroupKFold(5) over the 580 hidden site
     groups recovered from the spatial signature (EDA: 1157 rows -> 580 distinct spatial
     signatures, i.e. Topsoil/Subsoil pairs from the same site; a plain KFold leaks the
     site's location across the split), and for KRR an inner GroupKFold(4) over the
     outer-train rows only that picks each target's ridge alpha. Per knowledge/
     experience.md the per-target alpha spread is large (P wants the strongest
     regularization, the rest 10-100x weaker), so alpha is selected PER TARGET; because
     selection only ever sees outer-train rows the reported OOF MCRMSE is honest.

     KRR is solved in the dual with a single eigendecomposition of the training kernel
     per fit, so the whole alpha grid costs nothing extra after the eigh.

  2. blend — {kind:"blend", members:[<solo node id>, ...], per_target:bool,
             weight_search:"dirichlet"}
     Loads members' cached (n,5) OOF matrices and searches non-negative simplex weights.
     per_target=True searches an independent weight vector per target (MCRMSE is a mean
     of 5 independent column RMSEs, so per-target weights are strictly more expressive);
     per_target=False shares one weight vector across all 5.

SIGN CONVENTION: MCRMSE is already lower-is-better, so score == MCRMSE directly (no
sign flip, unlike the AUC comps). result["mcrmse"] carries the same human-readable value.
"""
import os
import signal
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMP_DIR = os.path.join(_REPO_ROOT, "competitions", "afsis-soil-properties")
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_afsis")
GRAM_DIR = os.path.join(CACHE_DIR, "gram")
FEATURES_NPZ = os.path.join(_COMP_DIR, "scripts", "features.npz")
TARGETS = ["Ca", "P", "pH", "SOC", "Sand"]
N_OUTER, N_INNER, SEED = 5, 4, 42
N_THREADS = 10

os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))

_D = np.load(FEATURES_NPZ)
_Y = _D["y"].astype(np.float64)              # (1157, 5)
_GROUPS = _D["groups"]
N_TRAIN, N_TARGETS = _Y.shape
N_TEST = _D["te_raw"].shape[0]
DEFAULT_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


# ---------------------------------------------------------------------------
# folds: outer GroupKFold(5) over site groups, inner GroupKFold(4) within outer-train
# ---------------------------------------------------------------------------
def _group_folds(groups, n_splits, seed):
    """Deterministic shuffled group k-fold: assign whole groups to folds, balancing
    fold sizes. sklearn's GroupKFold has no shuffle/seed, so this is done explicitly."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    sizes = np.array([(groups == g).sum() for g in uniq])
    fold_of = {}
    load = np.zeros(n_splits)
    for i in order:                       # greedy: biggest-remaining to lightest fold
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += sizes[i]
    assign = np.array([fold_of[g] for g in groups])
    return [(np.where(assign != f)[0], np.where(assign == f)[0]) for f in range(n_splits)]


_FOLDS = _group_folds(_GROUPS, N_OUTER, SEED)


# ---------------------------------------------------------------------------
# feature spaces + cached Gram matrices over [train; test]
# ---------------------------------------------------------------------------
_gram_mem = {}


def _feature_matrix(variant, use_spatial, spatial_weight):
    Xtr = _D[f"tr_{variant}"].astype(np.float64)
    Xte = _D[f"te_{variant}"].astype(np.float64)
    if use_spatial and variant != "spatial":
        Str = _D["tr_spatial"].astype(np.float64) * spatial_weight
        Ste = _D["te_spatial"].astype(np.float64) * spatial_weight
        Xtr = np.hstack([Xtr, Str])
        Xte = np.hstack([Xte, Ste])
    return Xtr, Xte


def get_gram(variant, use_spatial, spatial_weight):
    """Gram matrix over the stacked [train; test] rows, cached in memory and on disk.
    Every kernel this module uses (linear / rbf / poly) is a function of the Gram plus
    the row self-norms, so one matmul per feature space serves the whole search."""
    key = f"{variant}_sp{int(bool(use_spatial))}_w{spatial_weight:g}"
    if key in _gram_mem:
        return _gram_mem[key]
    path = os.path.join(GRAM_DIR, f"{key}.npy")
    if os.path.exists(path):
        G = np.load(path)
    else:
        Xtr, Xte = _feature_matrix(variant, use_spatial, spatial_weight)
        Z = np.vstack([Xtr, Xte])
        Z /= np.sqrt(Z.shape[1])          # keep Gram magnitudes O(1) across variants
        G = Z @ Z.T
        os.makedirs(GRAM_DIR, exist_ok=True)
        tmp = path + ".tmp.npy"
        np.save(tmp, G)
        os.replace(tmp, path)
    if len(_gram_mem) > 3:
        _gram_mem.clear()
    _gram_mem[key] = G
    return G


def _sqdist(G, rows, cols):
    d = np.diag(G)
    return np.maximum(d[rows][:, None] + d[cols][None, :] - 2.0 * G[np.ix_(rows, cols)], 0.0)


def _kernel(G, rows, cols, kernel, gamma, degree, coef0):
    if kernel == "linear":
        return G[np.ix_(rows, cols)]
    if kernel == "rbf":
        return np.exp(-gamma * _sqdist(G, rows, cols))
    if kernel == "laplacian":
        return np.exp(-gamma * np.sqrt(_sqdist(G, rows, cols)))
    if kernel == "poly":
        return (gamma * G[np.ix_(rows, cols)] + coef0) ** degree
    raise ValueError(f"unknown kernel {kernel!r}")


def _median_sqdist(G):
    """Median pairwise squared distance among TRAIN rows — the gamma reference scale."""
    rows = np.arange(N_TRAIN)
    sub = rows[::3]                                  # subsample: median is stable
    D = _sqdist(G, sub, sub)
    iu = np.triu_indices(len(sub), k=1)
    return float(np.median(D[iu]))


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mcrmse(y_true, y_pred):
    return float(np.mean([rmse(y_true[:, t], y_pred[:, t]) for t in range(y_true.shape[1])]))


# ---------------------------------------------------------------------------
# KRR: dual solve via one eigendecomposition, whole alpha grid free
# ---------------------------------------------------------------------------
def _krr_fit_predict(Ktr, Y, alphas_per_target, Kpred_list):
    """Ktr (m,m) train kernel, Y (m,5) centered targets, alphas_per_target (5,).
    Returns list of (n_i, 5) predictions, one per kernel block in Kpred_list."""
    w, U = np.linalg.eigh(Ktr)
    UtY = U.T @ Y                                     # (m, 5)
    duals = np.empty_like(UtY)
    for t in range(Y.shape[1]):
        duals[:, t] = UtY[:, t] / (w + alphas_per_target[t])
    C = U @ duals                                     # (m, 5)
    return [Kp @ C for Kp in Kpred_list]


def _krr_alpha_paths(Ktr, Y, alphas, Kval):
    """Predictions for every alpha in the grid, one eigh. Returns (n_alphas, nval, 5)."""
    w, U = np.linalg.eigh(Ktr)
    UtY = U.T @ Y
    KU = Kval @ U
    out = np.empty((len(alphas), Kval.shape[0], Y.shape[1]))
    for i, a in enumerate(alphas):
        out[i] = KU @ (UtY / (w + a)[:, None])
    return out


def _run_krr(params, variant, use_spatial, spatial_weight):
    kernel = params.get("kernel", "rbf")
    gamma_scale = float(params.get("gamma_scale", 1.0))
    degree = int(params.get("degree", 2))
    coef0 = float(params.get("coef0", 1.0))
    alphas = list(params.get("alphas", DEFAULT_ALPHAS))
    fixed_alpha = params.get("alpha", None)

    G = get_gram(variant, use_spatial, spatial_weight)
    gamma = gamma_scale / max(_median_sqdist(G), 1e-12)
    test_rows = np.arange(N_TRAIN, N_TRAIN + N_TEST)

    oof = np.zeros((N_TRAIN, N_TARGETS))
    pred = np.zeros((N_TEST, N_TARGETS))
    chosen = []
    for tr, va in _FOLDS:
        Ytr = _Y[tr]
        ymu = Ytr.mean(axis=0)
        Yc = Ytr - ymu

        if fixed_alpha is not None:
            best_alpha = np.full(N_TARGETS, float(fixed_alpha))
        else:
            # inner GroupKFold on outer-train rows ONLY -> honest per-target alpha
            inner = _group_folds(_GROUPS[tr], N_INNER, SEED + 1)
            err = np.zeros((len(alphas), N_TARGETS))
            for itr, iva in inner:
                gi_tr, gi_va = tr[itr], tr[iva]
                Ki = _kernel(G, gi_tr, gi_tr, kernel, gamma, degree, coef0)
                Kv = _kernel(G, gi_va, gi_tr, kernel, gamma, degree, coef0)
                imu = _Y[gi_tr].mean(axis=0)
                paths = _krr_alpha_paths(Ki, _Y[gi_tr] - imu, alphas, Kv) + imu
                for i in range(len(alphas)):
                    for t in range(N_TARGETS):
                        err[i, t] += np.sum((paths[i][:, t] - _Y[gi_va, t]) ** 2)
            best_alpha = np.array([alphas[int(np.argmin(err[:, t]))] for t in range(N_TARGETS)])
        chosen.append(best_alpha.tolist())

        Ktr = _kernel(G, tr, tr, kernel, gamma, degree, coef0)
        Kva = _kernel(G, va, tr, kernel, gamma, degree, coef0)
        Kte = _kernel(G, test_rows, tr, kernel, gamma, degree, coef0)
        pv, pt = _krr_fit_predict(Ktr, Yc, best_alpha, [Kva, Kte])
        oof[va] = pv + ymu
        pred += (pt + ymu) / N_OUTER
    return oof, pred, {"alpha_per_fold": chosen, "gamma": round(gamma, 6)}


# ---------------------------------------------------------------------------
# SVR (precomputed kernel), PLS, LightGBM-on-PCA, mean baseline
# ---------------------------------------------------------------------------
def _run_svr(params, variant, use_spatial, spatial_weight):
    from sklearn.svm import SVR
    kernel = params.get("kernel", "rbf")
    gamma_scale = float(params.get("gamma_scale", 1.0))
    C = float(params.get("C", 10.0))
    epsilon = float(params.get("epsilon", 0.1))
    degree, coef0 = int(params.get("degree", 2)), float(params.get("coef0", 1.0))

    G = get_gram(variant, use_spatial, spatial_weight)
    gamma = gamma_scale / max(_median_sqdist(G), 1e-12)
    test_rows = np.arange(N_TRAIN, N_TRAIN + N_TEST)

    oof = np.zeros((N_TRAIN, N_TARGETS))
    pred = np.zeros((N_TEST, N_TARGETS))
    for tr, va in _FOLDS:
        Ktr = _kernel(G, tr, tr, kernel, gamma, degree, coef0)
        Kva = _kernel(G, va, tr, kernel, gamma, degree, coef0)
        Kte = _kernel(G, test_rows, tr, kernel, gamma, degree, coef0)
        for t in range(N_TARGETS):
            m = SVR(kernel="precomputed", C=C, epsilon=epsilon, cache_size=1000)
            m.fit(Ktr, _Y[tr, t])
            oof[va, t] = m.predict(Kva)
            pred[:, t] += m.predict(Kte) / N_OUTER
    return oof, pred, {"gamma": round(gamma, 6)}


def _run_pls(params, variant, use_spatial, spatial_weight):
    from sklearn.cross_decomposition import PLSRegression
    nc = int(params.get("n_components", 20))
    Xtr, Xte = _feature_matrix(variant, use_spatial, spatial_weight)
    oof = np.zeros((N_TRAIN, N_TARGETS))
    pred = np.zeros((N_TEST, N_TARGETS))
    for tr, va in _FOLDS:
        m = PLSRegression(n_components=nc, scale=False)
        m.fit(Xtr[tr], _Y[tr])
        oof[va] = m.predict(Xtr[va])
        pred += m.predict(Xte) / N_OUTER
    return oof, pred, {"n_components": nc}


def _run_lgbpca(params, variant, use_spatial, spatial_weight):
    import lightgbm as lgb
    from sklearn.decomposition import PCA
    n_comp = int(params.get("n_components", 40))
    p = dict(objective="regression", metric="rmse", num_threads=N_THREADS, verbose=-1,
             deterministic=True, force_row_wise=True, random_state=SEED,
             learning_rate=params.get("learning_rate", 0.05),
             num_leaves=params.get("num_leaves", 15),
             min_child_samples=params.get("min_child_samples", 20),
             feature_fraction=params.get("feature_fraction", 0.7),
             bagging_fraction=params.get("bagging_fraction", 0.8), bagging_freq=1,
             lambda_l2=params.get("lambda_l2", 1.0))
    n_est = int(params.get("n_estimators", 400))
    Xtr, Xte = _feature_matrix(variant, use_spatial, spatial_weight)
    oof = np.zeros((N_TRAIN, N_TARGETS))
    pred = np.zeros((N_TEST, N_TARGETS))
    for tr, va in _FOLDS:
        pca = PCA(n_components=n_comp, random_state=SEED).fit(Xtr[tr])
        Ztr, Zva, Zte = pca.transform(Xtr[tr]), pca.transform(Xtr[va]), pca.transform(Xte)
        for t in range(N_TARGETS):
            m = lgb.LGBMRegressor(n_estimators=n_est, **p)
            m.fit(Ztr, _Y[tr, t])
            oof[va, t] = m.predict(Zva)
            pred[:, t] += m.predict(Zte) / N_OUTER
    return oof, pred, {"n_components": n_comp}


def _run_mean(params, variant, use_spatial, spatial_weight):
    oof = np.zeros((N_TRAIN, N_TARGETS))
    pred = np.zeros((N_TEST, N_TARGETS))
    for tr, va in _FOLDS:
        mu = _Y[tr].mean(axis=0)
        oof[va] = mu
        pred += mu / N_OUTER
    return oof, pred, {}


_RUNNERS = {"krr": _run_krr, "svr": _run_svr, "pls": _run_pls,
            "lgbpca": _run_lgbpca, "mean": _run_mean}


def evaluate_solo(config):
    model = config["model"]
    if model not in _RUNNERS:
        raise ValueError(f"unknown model type {model!r}")
    variant = config.get("variant", "sg1")
    if f"tr_{variant}" not in _D:
        raise ValueError(f"unknown variant {variant!r}")
    use_spatial = bool(config.get("use_spatial", True))
    spatial_weight = float(config.get("spatial_weight", 1.0))
    oof, pred, info = _RUNNERS[model](dict(config.get("params") or {}),
                                       variant, use_spatial, spatial_weight)
    score = mcrmse(_Y, oof)
    per_target = {TARGETS[t]: round(rmse(_Y[:, t], oof[:, t]), 6) for t in range(N_TARGETS)}
    info.update({"mcrmse": round(score, 6), "per_target": per_target})
    return oof, pred, score, info


# ---------------------------------------------------------------------------
# blend: per-target (default) or shared simplex weights over cached member OOFs
# ---------------------------------------------------------------------------
def _load_member(mid):
    d = np.load(os.path.join(CACHE_DIR, f"solo_{mid}.npz"), allow_pickle=True)
    return d["oof"], d["pred"]


def _weight_search_1d(cols, y, k=hv3.DEFAULT_BLEND_K, seed=SEED):
    """cols (n, n_members) single-target OOF columns -> best simplex weights by RMSE."""
    n = cols.shape[1]
    rng = np.random.default_rng(seed)
    cands = [np.eye(n)[i] for i in range(n)]
    cands.append(np.full(n, 1.0 / n))
    cands += list(rng.dirichlet(np.ones(n), size=k))
    metric = lambda v: rmse(y, v)  # noqa: E731
    best_w, best_s = None, None
    for w in cands:
        s = metric(cols @ w)
        if best_s is None or s < best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=max(k // 3, 50)):
        s = metric(cols @ w)
        if s < best_s:
            best_s, best_w = s, w
    return hv3._coordinate_ascent_refine(cols, metric, best_w, best_s)


def _greedy_weights(cols, y, rounds=60, member_idx=None):
    """Caruana forward selection WITH replacement over `cols` (n, M) for one target.
    Returns (weights, best_rmse) taken at the best prefix, not the last round — with
    replacement the path is not monotone. knowledge/experience.md records this as the
    weight search that actually works on this comp's many-member pools, where a
    Dirichlet(1,..,1) draw over M>>5 members concentrates near uniform and dilutes."""
    n_m = cols.shape[1]
    idxs = np.arange(n_m) if member_idx is None else np.asarray(member_idx)
    counts = np.zeros(n_m)
    cur = np.zeros(len(y))
    k = 0
    best_s, best_counts = None, counts.copy()
    for _ in range(rounds):
        cand = (cur[:, None] * k + cols[:, idxs]) / (k + 1)
        errs = np.sqrt(np.mean((cand - y[:, None]) ** 2, axis=0))
        j = idxs[int(np.argmin(errs))]
        cur = (cur * k + cols[:, j]) / (k + 1)
        k += 1
        counts[j] += 1
        s = float(errs.min())
        if best_s is None or s < best_s:
            best_s, best_counts = s, counts.copy()
    w = best_counts / max(best_counts.sum(), 1.0)
    return w, rmse(y, cols @ w)


def _bagged_greedy_weights(cols, y, rounds=30, n_bags=20, frac=0.5, seed=SEED):
    """Caruana-style bagged greedy: each bag greedily selects from a random subset of
    members, weights are averaged across bags. experience.md: bagging lowers the greedy
    search's own optimism (+0.0146 vs +0.0186) but does not always win on the nested
    score — both must be tried and the search must decide."""
    rng = np.random.default_rng(seed)
    n_m = cols.shape[1]
    k = max(2, int(round(frac * n_m)))
    acc = np.zeros(n_m)
    for b in range(n_bags):
        sub = rng.choice(n_m, size=k, replace=False)
        w, _ = _greedy_weights(cols, y, rounds=rounds, member_idx=sub)
        acc += w
    w = acc / max(acc.sum(), 1e-12)
    return w, rmse(y, cols @ w)


def _weight_search(cols, y, method, seed=SEED):
    if method == "dirichlet":
        return _weight_search_1d(cols, y, seed=seed)
    if method == "greedy":
        return _greedy_weights(cols, y, rounds=60)
    if method == "bagged_greedy":
        return _bagged_greedy_weights(cols, y, seed=seed)
    if method == "greedy_ascent":
        w, s = _greedy_weights(cols, y, rounds=60)
        return hv3._coordinate_ascent_refine(cols, lambda v: rmse(y, v), w, s)
    raise ValueError(f"unknown weight_search {method!r}")


def evaluate_blend(config):
    members = list(config.get("members") or [])
    if len(members) < 2:
        raise ValueError(f"blend node needs >=2 members, got {members!r}")
    per_target = bool(config.get("per_target", True))
    method = config.get("weight_search", "dirichlet")
    loaded = [_load_member(m) for m in members]
    oofs = np.stack([o for o, _ in loaded], axis=1)       # (n, M, 5)
    preds = np.stack([p for _, p in loaded], axis=1)      # (ntest, M, 5)

    blend_oof = np.zeros((N_TRAIN, N_TARGETS))
    blend_pred = np.zeros((N_TEST, N_TARGETS))
    weights = {}
    if per_target:
        for t in range(N_TARGETS):
            w, _ = _weight_search(oofs[:, :, t], _Y[:, t], method)
            blend_oof[:, t] = oofs[:, :, t] @ w
            blend_pred[:, t] = preds[:, :, t] @ w
            weights[TARGETS[t]] = [round(float(x), 4) for x in w]
    else:
        flat = oofs.transpose(0, 2, 1).reshape(N_TRAIN * N_TARGETS, len(members))
        yflat = _Y.reshape(-1)
        w, _ = _weight_search(flat, yflat, method)
        blend_oof = np.einsum("nmt,m->nt", oofs, w)
        blend_pred = np.einsum("nmt,m->nt", preds, w)
        weights["all"] = [round(float(x), 4) for x in w]

    score = mcrmse(_Y, blend_oof)
    per = {TARGETS[t]: round(rmse(_Y[:, t], blend_oof[:, t]), 6) for t in range(N_TARGETS)}
    info = {"members": members, "per_target": per_target, "weight_search": method,
            "weights": weights, "mcrmse": round(score, 6), "per_target_rmse": per}
    return blend_oof, blend_pred, score, info


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
class EvalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EvalTimeout()


def evaluate(config: dict, node_id: int = None, timeout_s: int = 600) -> dict:
    """config -> {status, score, wall_s, result, error}. Never raises. score is MCRMSE
    (already lower-is-better, no sign flip)."""
    t0 = time.time()
    have_alarm = hasattr(signal, "SIGALRM")
    old = None
    if have_alarm:
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_s))
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            oof, pred, score, info = evaluate_solo(config)
        elif kind == "blend":
            oof, pred, score, info = evaluate_blend(config)
        else:
            raise ValueError(f"unknown node kind {kind!r}")
        if node_id is not None:
            hv2.cache_oof(CACHE_DIR, node_id, oof, pred=pred, mcrmse=score)
        return dict(status="evaluated", score=round(score, 6),
                    wall_s=round(time.time() - t0, 1), result=info, error=None)
    except EvalTimeout:
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"timeout>{timeout_s}s")
    except Exception as e:  # noqa: BLE001 - evaluator must never crash the search loop
        return dict(status="failed", score=None, wall_s=round(time.time() - t0, 1),
                    result=None, error=f"{type(e).__name__}: {e}")
    finally:
        if have_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
