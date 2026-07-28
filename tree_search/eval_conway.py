"""tree_search/eval_conway.py — per-competition evaluator for conway-s-reverse-game-of-life
(reverse Game of Life, per-cell binary MAE, minimize).

Fold-0 proxy evaluation (fold-0 of the fixed 5-fold stratified-by-delta split, seed 42):
full 5-fold NN evals are ~8.4 min each, unaffordable for a tree search; fold-0 rows are
12.5k boards = 5M cells so metric noise is tiny. Search-grade solo defaults (epochs 10,
synth_n 50k) are ~3 min. The full-grade linear-protocol members (cnn_v1: 20ep/100k synth,
lgb_v1: per-delta 7x7-window LGB) are seeded via cached-OOF reuse at zero cost.

Node kinds:
  solo cnn — {kind:"solo", model:"cnn", params:{channels,depth,residual,epochs,batch,lr,
              wd,synth_n,d4_aug,tta,per_delta,seed}}
  solo lgb — {kind:"solo", model:"lgb", params:{window,n_estimators,num_leaves,
              learning_rate,min_child_samples}}
  reuse    — {kind:"solo", model:"reuse", params:{name:"cnn_v1"|"lgb_v1"}}
              loads competitions/.../scripts/cache/preds_<name>.npz fold-0 slice.
  blend    — {kind:"blend", members:[node ids], per_delta_weights:bool,
              thresh_sweep:bool}
              weight-search minimizing MAE@0.5; optional per-delta independent weights
              (afsis per-target-weights prior) and per-delta threshold sweep afterwards
              (metric includes the thresholding postprocess, per 07_tree_search.md §2).
  refine   — {kind:"refine", params:{source:<node id>, lam, n_steps, focus_frac,
              focus_k, seed}}
              forward-consistency greedy hill-climb: flip cells of the thresholded
              prediction to reduce hamming(evolve(start,delta), stop) + lam * prior
              cost from the source node's probabilities. Comp-local structural lever:
              the stop board is a hard constraint the CNN only uses statistically.

Scores handed to the harness are MAE (lower better) — no sign flip needed.
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_REPO_ROOT, "competitions", "conway-s-reverse-game-of-life", "scripts")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, _HERE)
os.chdir(_REPO_ROOT)  # cnn_lib/common use repo-root-relative paths

import common  # noqa: E402
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402

CACHE_DIR = os.path.join(_HERE, "cache_conway")
os.makedirs(CACHE_DIR, exist_ok=True)

_a = common.load_arrays()
S, P, DELTA = _a["S"], _a["P"], _a["delta"]
FOLDS = common.get_folds(DELTA)
F0 = np.where(FOLDS == 0)[0]           # fold-0 rows, ascending
S0 = S[F0]                              # (n0,20,20) true starts
P0 = P[F0]
D0 = DELTA[F0]
N0 = len(F0)
Y0 = S0.reshape(-1).astype(np.int8)     # flat truth for blend metric


def mae_of_probs(flat_probs: np.ndarray, thresh: float = 0.5) -> float:
    return float(((flat_probs > thresh).astype(np.int8) != Y0).mean())


def _metric_fn(vec):
    return mae_of_probs(vec)


# ---------------------------------------------------------------------------
# solo: cnn / lgb / reuse
# ---------------------------------------------------------------------------
def _eval_cnn(params):
    import cnn_lib
    cfg = {**cnn_lib.DEFAULT_CFG, **params}
    oof_mae, scores, oof = cnn_lib.run_cv(cfg, name="_tsnode", folds_to_run=[0], save=False)
    return oof[F0].reshape(-1).astype(np.float32), oof_mae, cfg


def _eval_lgb(params):
    import lightgbm as lgb
    from numpy.lib.stride_tricks import sliding_window_view
    w = int(params.get("window", 7)) // 2
    lp = dict(objective="binary",
              learning_rate=float(params.get("learning_rate", 0.1)),
              num_leaves=int(params.get("num_leaves", 63)),
              min_child_samples=int(params.get("min_child_samples", 100)),
              subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
              n_estimators=int(params.get("n_estimators", 100)),
              num_threads=10, deterministic=True, force_row_wise=True,
              random_state=42, verbosity=-1)

    def feats(board):
        n = len(board)
        pad = np.pad(board, ((0, 0), (w, w), (w, w)))
        win = sliding_window_view(pad, (2 * w + 1, 2 * w + 1), axis=(1, 2))
        f = win.reshape(n, 400, (2 * w + 1) ** 2).astype(np.uint8)
        rows = np.tile(np.repeat(np.arange(20, dtype=np.uint8), 20), (n, 1))[..., None]
        cols = np.tile(np.tile(np.arange(20, dtype=np.uint8), 20), (n, 1))[..., None]
        return np.concatenate([f, rows, cols], axis=2).reshape(n * 400, -1)

    tr = np.where(FOLDS != 0)[0]
    oof0 = np.zeros((N0, 20, 20), np.float32)
    for d in range(1, 6):
        trd = tr[DELTA[tr] == d]
        clf = lgb.LGBMClassifier(**lp)
        clf.fit(feats(P[trd]), S[trd].reshape(-1))
        m = D0 == d
        oof0[m] = clf.predict_proba(feats(P0[m]))[:, 1].reshape(-1, 20, 20)
    flat = oof0.reshape(-1).astype(np.float32)
    return flat, mae_of_probs(flat), lp


def _eval_reuse(params):
    name = params["name"]
    npz = np.load(f"{common.CACHE}/preds_{name}.npz")
    with open(f"{common.CACHE}/score_{name}.json") as fh:
        meta = json.load(fh)
    flat = npz["oof"][F0].reshape(-1).astype(np.float32)
    sc = mae_of_probs(flat)
    # digit-verify against the stored fold-0 score
    stored = round(float(meta["fold_scores"][0]), 6)
    assert round(sc, 6) == stored, f"reuse {name}: {sc:.6f} != stored {stored:.6f}"
    return flat, sc, meta.get("cfg", {})


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------
def _weight_search(oofs, truth, k=300, seed=42):
    """Dirichlet search + coordinate ascent, minimizing MAE@0.5. oofs (n, m)."""
    rng = np.random.RandomState(seed)
    m = oofs.shape[1]
    best_w, best_s = np.ones(m) / m, None
    best_s = float(((oofs.mean(1) > 0.5).astype(np.int8) != truth).mean())
    for i in range(k):
        w = rng.dirichlet(np.ones(m))
        s = float((((oofs @ w) > 0.5).astype(np.int8) != truth).mean())
        if s < best_s:
            best_w, best_s = w, s
    # coordinate ascent
    for _ in range(4):
        improved = False
        for j in range(m):
            for step in (0.1, -0.1, 0.05, -0.05):
                w = best_w.copy()
                w[j] = max(0.0, w[j] + step)
                if w.sum() == 0:
                    continue
                w /= w.sum()
                s = float((((oofs @ w) > 0.5).astype(np.int8) != truth).mean())
                if s < best_s - 1e-9:
                    best_w, best_s, improved = w, s, True
        if not improved:
            break
    return best_w, best_s


def evaluate_blend(config):
    members = config["members"]
    if len(members) < 2:
        raise ValueError("blend needs >=2 members")
    oofs = np.stack([hv2.load_oof(CACHE_DIR, m) for m in members], axis=1)  # (N0*400, m)
    per_delta = bool(config.get("per_delta_weights", False))
    thresh_sweep = bool(config.get("thresh_sweep", False))
    cell_delta = np.repeat(D0, 400)

    weights = {}
    blended = np.zeros(oofs.shape[0], np.float32)
    if per_delta:
        for d in range(1, 6):
            m = cell_delta == d
            w, _ = _weight_search(oofs[m], Y0[m], seed=42 + d)
            blended[m] = oofs[m] @ w
            weights[str(d)] = [round(float(x), 4) for x in w]
    else:
        w, _ = _weight_search(oofs, Y0)
        blended = (oofs @ w).astype(np.float32)
        weights["all"] = [round(float(x), 4) for x in w]

    thresholds = {}
    if thresh_sweep:
        pred = np.zeros_like(Y0)
        for d in range(1, 6):
            m = cell_delta == d
            cand = np.arange(0.35, 0.66, 0.01)
            scs = [((blended[m] > t).astype(np.int8) != Y0[m]).mean() for t in cand]
            t = float(cand[int(np.argmin(scs))])
            thresholds[str(d)] = round(t, 2)
            pred[m] = (blended[m] > t).astype(np.int8)
        score = float((pred != Y0).mean())
    else:
        score = mae_of_probs(blended)
    return blended, score, dict(members=members, weights=weights,
                                per_delta_weights=per_delta,
                                thresholds=thresholds or None)


# ---------------------------------------------------------------------------
# refine: forward-consistency hill climb
# ---------------------------------------------------------------------------
def evolve(b: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = b.copy()
    for step in range(5):
        out = np.where((d > step)[:, None, None], common.life_step(out), out)
    return out


def refine_probs(probs3, stop, delta, lam=0.35, n_steps=1200, focus_frac=0.8,
                 focus_k=120, seed=42, ret_boards=False):
    """probs3 (n,20,20) -> refined binary boards (n,20,20). Greedy per-board flips."""
    rng = np.random.RandomState(seed)
    n = len(probs3)
    p = probs3.clip(1e-6, 1 - 1e-6).astype(np.float64)
    q = np.log(p / (1 - p))                       # logit; flip cost uses this
    s = (p > 0.5).astype(np.int8)
    order = np.argsort(np.abs(q).reshape(n, 400), axis=1)  # uncertain first
    fwd = evolve(s, delta)
    err = (fwd != stop).sum((1, 2)).astype(np.float64)
    prior_pen = np.zeros(n)
    bi = np.arange(n)
    for t in range(n_steps):
        if rng.rand() < focus_frac:
            ci = order[bi, rng.randint(0, focus_k, n)]
        else:
            ci = rng.randint(0, 400, n)
        r, c = ci // 20, ci % 20
        s2 = s.copy()
        s2[bi, r, c] = 1 - s2[bi, r, c]
        err2 = (evolve(s2, delta) != stop).sum((1, 2)).astype(np.float64)
        dprior = (2 * s[bi, r, c] - 1) * q[bi, r, c] * lam
        acc = (err2 - err + dprior) < 0
        if acc.any():
            s[acc] = s2[acc]
            err[acc] = err2[acc]
            prior_pen[acc] += dprior[acc]
    return s if ret_boards else s


def evaluate_refine(config):
    prm = config.get("params") or {}
    src = prm["source"]
    flat = hv2.load_oof(CACHE_DIR, src)
    probs3 = flat.reshape(N0, 20, 20)
    s = refine_probs(probs3, P0, D0,
                     lam=float(prm.get("lam", 0.35)),
                     n_steps=int(prm.get("n_steps", 1200)),
                     focus_frac=float(prm.get("focus_frac", 0.8)),
                     focus_k=int(prm.get("focus_k", 120)),
                     seed=int(prm.get("seed", 42)))
    score = float((s.reshape(-1) != Y0).mean())
    # cache refined boards as pseudo-probs so blends can consume them
    return s.reshape(-1).astype(np.float32), score, dict(source=src, **prm)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def evaluate(config, node_id=None, timeout_s=None):
    t0 = time.time()
    try:
        kind = config.get("kind", "solo")
        if kind == "solo":
            model = config["model"]
            params = dict(config.get("params") or {})
            if model == "cnn":
                flat, score, meta = _eval_cnn(params)
            elif model == "lgb":
                flat, score, meta = _eval_lgb(params)
            elif model == "reuse":
                flat, score, meta = _eval_reuse(params)
            else:
                raise ValueError(f"unknown model {model!r}")
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, flat)
            result = dict(mae=round(score, 6), model=model, meta=meta)
        elif kind == "blend":
            flat, score, result = evaluate_blend(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, flat.astype(np.float32))
        elif kind == "refine":
            flat, score, result = evaluate_refine(config)
            if node_id is not None:
                hv2.cache_oof(CACHE_DIR, node_id, flat)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        return dict(score=float(score), status="ok", wall_s=round(time.time() - t0, 2),
                    result=result, error=None)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(score=None, status="failed", wall_s=round(time.time() - t0, 2),
                    result=None, error=f"{e}\n{traceback.format_exc()[-1500:]}")
