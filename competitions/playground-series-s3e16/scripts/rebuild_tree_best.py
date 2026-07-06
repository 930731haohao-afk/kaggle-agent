"""rebuild_tree_best.py — Rebuild the tree-search winning blend (node #15 in
experiments_tree.json, rounded OOF MAE 1.33563) END TO END and emit a REAL test
submission file for Kaggle LB validation.

Why this script exists: the winning blend was found purely as an OOF-space weight
search over 8 CACHED member OOF vectors (tree_search/eval_s3e16_v2.py's evaluate_blend,
harness_v2.eval_blend) — no test predictions were ever produced for this exact
composition (5 of the 8 members were reused read-only from Phase B's
scripts/cache/*.npz which DO carry test preds; 3 were freshly trained by
eval_s3e16_v2.py and cached to tree_search/cache_s3e16/solo_<id>.npz, which ALSO
carry test preds — but nobody ever combined them into a submission file). This
script retrains every one of the 8 members from scratch, using the IDENTICAL
5-fold StratifiedKFold-on-binned-Age split (seed=42) and the IDENTICAL
hyperparameters recorded in experiments_tree.json, as an independent end-to-end
verification (not just a copy of the cache) — then blends the freshly-produced
test predictions with node #15's exact weights, rounds, clips, and writes the
submission.

Node #15 composition (competitions/playground-series-s3e16/experiments_tree.json):
  member id | model                              | weight  | solo rounded MAE
  --------- | ---------------------------------- | ------- | ----------------
  0  (root) | LGB   (regression_l1)               | 0.0198  | 1.33885
  1         | XGB   (reg:absoluteerror)           | 0.0256  | 1.34160
  2         | CAT   (MAE loss)                    | 0.5061  | 1.33846
  3         | LGB_tuned (Optuna fold0-proxy)      | 0.0075  | 1.33979
  4         | LGB_tuned_seed2024 (seed-bagged)    | 0.0626  | 1.33914
  5         | LGBBOUND (boundary-pushed lr=0.005) | 0.1117  | 1.33950
  6         | TWEEDIE (tweedie-objective LGB)     | 0.0004  | 1.37467
  8         | FEATPRUNE (LGB, drop `Weight`)      | 0.2662  | 1.34025

GATE (must pass before any submission file is written):
  1. Each member's reconstructed OOF vector must match its cached OOF vector to
     within max-abs-diff <= 1e-9 (digit-for-digit reproduction).
  2. The blended + rounded + clipped OOF MAE must equal 1.33563 (rounded to 5
     decimals) exactly, using the RECOVERED full-precision weight vector (see below).
  3. The recovered weights must round(4) to the tree-stored weights member-for-member.
If any check fails, the script STOPS and prints the discrepancy — no submission
file is written from mismatched models.

Weight-recovery note (why not just use the weights stored in experiments_tree.json):
evaluate_blend stored node #15's weights ROUNDED TO 4 DECIMALS (`round(float(w), 4)`;
they sum to 0.9999). Blending with those display-precision weights gives rounded OOF
MAE 1.335796, NOT 1.33563 (observed in this script's first run — the ~5e-5 per-member
weight error flips a handful of samples across the 0.5 rounding boundary). The exact
full-precision weight vector is therefore RECOVERED by replaying the winning node's own
fully deterministic weight search over the gate-verified reconstructed OOF stack, in
the config's member order [0,1,2,3,4,5,6,8]:
  harness_v2.eval_blend's "dirichlet" path — rng=np.random.default_rng(42): the n unit
  vectors + the uniform blend + k=800 Dirichlet(1,..,1) draws, then a concentrated
  refinement round of k//3 draws around the coarse best — followed by
  eval_s3e16_v2._coord_ascent_refine (rounds=6, deltas ±0.05/±0.02/±0.01/±0.005),
  every candidate scored on MAE(clip(round(blend), 1, None)).
Verified offline before this run: this replay recovers best rounded MAE 1.3356335...
(= 1.33563 at 5 dp), raw 1.35712, and weights that round(4) to the stored tree weights
exactly.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

warnings_ignored = True
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMP_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_COMP_DIR))
DATA = os.path.join(_COMP_DIR, "data")
SUB_DIR = os.path.join(_COMP_DIR, "submissions")
LEGACY_CACHE_DIR = os.path.join(_HERE, "cache")
TREE_CACHE_DIR = os.path.join(_REPO_ROOT, "tree_search", "cache_s3e16")
TREE_JSON = os.path.join(_COMP_DIR, "experiments_tree.json")

sys.path.insert(0, _HERE)
from features import build_features, feature_columns  # noqa: E402

TARGET, ID = "Age", "id"
N_SPLITS, SEED = 5, 42
GATE_TOL = 1e-9
TARGET_ROUNDED_MAE = 1.33563
WINNING_NODE_ID = 15

# ---------------------------------------------------------------------------
# Data + folds — IDENTICAL to scripts/pool_lib.py and tree_search/eval_s3e16_v2.py
# ---------------------------------------------------------------------------
_train = pd.read_csv(os.path.join(DATA, "train.csv"))
_test = pd.read_csv(os.path.join(DATA, "test.csv"))
_h_med = _train.loc[_train["Height"] > 0, "Height"].median()
_Xtr_full = build_features(_train, height_median=_h_med)
_Xte_full = build_features(_test, height_median=_h_med)
ALL_FEATURES = feature_columns(_Xtr_full)
_y = _train[TARGET].to_numpy(np.float64)
_test_ids = _test[ID]
AGE_LOW = int(_y.min())

_ybin = np.where(_y >= 20, 20, _y).astype(int)
_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(_skf.split(np.zeros(len(_y)), _ybin))


def raw_mae(oof_vec) -> float:
    return float(mean_absolute_error(_y, oof_vec))


def round_clip_mae(oof_vec) -> float:
    return float(mean_absolute_error(_y, np.clip(np.round(oof_vec), AGE_LOW, None)))


def feature_frame(drop):
    drop = set(drop or [])
    feats = [c for c in ALL_FEATURES if c not in drop]
    Xnp = _Xtr_full[feats].to_numpy(np.float32)
    Xtestnp = _Xte_full[feats].to_numpy(np.float32)
    return Xnp, Xtestnp, feats


# ---------------------------------------------------------------------------
# Trainers — mirror scripts/pool_lib.py (LGB/XGB/CAT base members) and
# tree_search/eval_s3e16_v2.py's _run_lgb (LGB base+override merge pattern) exactly.
# ---------------------------------------------------------------------------
LGB_RUNTIME_BASE = dict(objective="regression_l1", metric="mae", n_estimators=3000,
                         learning_rate=0.02, num_leaves=63, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                         reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, n_jobs=-1,
                         verbose=-1)


def train_lgb(params, Xnp, Xtestnp):
    import lightgbm as lgb
    p = dict(LGB_RUNTIME_BASE)
    p.update(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtestnp))
    for tr, va in FOLDS:
        m = lgb.LGBMRegressor(**p)
        m.fit(Xnp[tr], _y[tr], eval_set=[(Xnp[va], _y[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict(Xnp[va])
        pred += m.predict(Xtestnp) / N_SPLITS
    return oof, pred


def train_xgb(params, Xnp, Xtestnp):
    import xgboost as xgb
    p = dict(params)
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtestnp))
    for tr, va in FOLDS:
        m = xgb.XGBRegressor(**p)
        m.fit(Xnp[tr], _y[tr], eval_set=[(Xnp[va], _y[va])], verbose=False)
        oof[va] = m.predict(Xnp[va])
        pred += m.predict(Xtestnp) / N_SPLITS
    return oof, pred


def train_cat(params, Xnp, Xtestnp):
    from catboost import CatBoostRegressor, Pool
    oof = np.zeros(len(_y))
    pred = np.zeros(len(Xtestnp))
    for tr, va in FOLDS:
        m = CatBoostRegressor(**params, thread_count=-1, verbose=False,
                               allow_writing_files=False)
        m.fit(Pool(Xnp[tr], _y[tr]), eval_set=Pool(Xnp[va], _y[va]),
              early_stopping_rounds=200, verbose=False)
        oof[va] = m.predict(Xnp[va])
        pred += m.predict(Xtestnp) / N_SPLITS
    return oof, pred


# ---------------------------------------------------------------------------
# Member specs — pulled verbatim from experiments_tree.json nodes 0,1,2,3,4,5,6,8
# (verified against the tree JSON at script-start, see load_and_check_tree()).
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(objective="reg:absoluteerror", n_estimators=3000, learning_rate=0.02,
                   max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                   reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1,
                   eval_metric="mae", early_stopping_rounds=150)
CAT_PARAMS = dict(loss_function="MAE", eval_metric="MAE", iterations=4000,
                   learning_rate=0.03, depth=7, l2_leaf_reg=5.0, random_seed=42)

MEMBERS = [
    dict(node_id=0, name="LGB (root)", runner=train_lgb, drop=[],
         params=dict(objective="regression_l1", metric="mae", n_estimators=3000,
                     learning_rate=0.02, num_leaves=63, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, random_state=42),
         cache_kind="legacy", cache_name="LGB"),
    dict(node_id=1, name="XGB", runner=train_xgb, drop=[], params=XGB_PARAMS,
         cache_kind="legacy", cache_name="XGB"),
    dict(node_id=2, name="CAT", runner=train_cat, drop=[], params=CAT_PARAMS,
         cache_kind="legacy", cache_name="CAT"),
    dict(node_id=3, name="LGB_tuned", runner=train_lgb, drop=[],
         params=dict(learning_rate=0.010194593774108831, num_leaves=61,
                     min_child_samples=30, subsample=0.8094065093571273,
                     colsample_bytree=0.7216637100997246, reg_alpha=0.1131044878505013,
                     reg_lambda=2.563832295704767, n_estimators=3000, random_state=42),
         cache_kind="legacy", cache_name="LGB_tuned"),
    dict(node_id=4, name="LGB_tuned_seed2024", runner=train_lgb, drop=[],
         params=dict(learning_rate=0.010194593774108831, num_leaves=61,
                     min_child_samples=30, subsample=0.8094065093571273,
                     colsample_bytree=0.7216637100997246, reg_alpha=0.1131044878505013,
                     reg_lambda=2.563832295704767, n_estimators=3000, random_state=2024),
         cache_kind="legacy", cache_name="LGB_tuned_seed2024"),
    dict(node_id=5, name="LGBBOUND", runner=train_lgb, drop=[],
         params=dict(learning_rate=0.005, num_leaves=61, min_child_samples=30,
                     subsample=0.8094065093571273, colsample_bytree=0.7216637100997246,
                     reg_alpha=0.1131044878505013, reg_lambda=2.563832295704767,
                     n_estimators=3000, random_state=42),
         cache_kind="tree", cache_name="solo_5"),
    dict(node_id=6, name="TWEEDIE", runner=train_lgb, drop=[],
         params=dict(num_leaves=61, min_child_samples=30, subsample=0.8094065093571273,
                     colsample_bytree=0.7216637100997246, reg_alpha=0.1131044878505013,
                     reg_lambda=2.563832295704767, n_estimators=3000, random_state=42,
                     objective="tweedie", tweedie_variance_power=1.3, metric="mae",
                     learning_rate=0.02),
         cache_kind="tree", cache_name="solo_6"),
    dict(node_id=8, name="FEATPRUNE", runner=train_lgb, drop=["Weight"],
         params=dict(objective="regression_l1", metric="mae", n_estimators=3000,
                     learning_rate=0.02, num_leaves=63, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                     reg_alpha=1.0, reg_lambda=2.0, random_state=42),
         cache_kind="tree", cache_name="solo_8"),
]


def recover_exact_weights(oof_stack):
    """Replay node #15's own weight search, bit-for-bit: harness_v2.eval_blend's
    'dirichlet' path (rng=default_rng(42), unit vectors + uniform + k=800 draws, then a
    concentrated k//3 refinement round) followed by eval_s3e16_v2._coord_ascent_refine
    (rounds=6, deltas ±0.05/±0.02/±0.01/±0.005, improvement threshold 1e-9), every
    candidate scored on the rounded metric. Fully deterministic -> recovers the exact
    full-precision weight vector that produced 1.33563."""
    n = oof_stack.shape[1]
    k = 800
    rng = np.random.default_rng(42)
    candidates = [np.eye(n)[i] for i in range(n)]
    candidates.append(np.full(n, 1.0 / n))
    candidates += list(rng.dirichlet(np.ones(n), size=k))
    best_w, best_s = None, None
    for w in candidates:
        s = round_clip_mae(oof_stack @ w)
        if best_s is None or s < best_s:
            best_s, best_w = s, w
    conc = np.clip(best_w, 1e-3, None) * 200.0
    for w in rng.dirichlet(conc, size=max(k // 3, 50)):
        s = round_clip_mae(oof_stack @ w)
        if s < best_s:
            best_s, best_w = s, w

    # coordinate-ascent refinement (identical to eval_s3e16_v2._coord_ascent_refine)
    best_w = np.array(best_w, dtype=float)
    deltas = (0.05, -0.05, 0.02, -0.02, 0.01, -0.01, 0.005, -0.005)
    for _ in range(6):
        improved = False
        for i in range(n):
            for d in deltas:
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + d)
                if w_try.sum() <= 0:
                    continue
                w_try = w_try / w_try.sum()
                s = round_clip_mae(oof_stack @ w_try)
                if s < best_s - 1e-9:
                    best_s, best_w = s, w_try
                    improved = True
        if not improved:
            break
    return best_w, best_s


def load_cached_oof(cache_kind, cache_name):
    if cache_kind == "legacy":
        path = os.path.join(LEGACY_CACHE_DIR, f"{cache_name}.npz")
    else:
        path = os.path.join(TREE_CACHE_DIR, f"{cache_name}.npz")
    d = np.load(path)
    return d["oof"]


def load_and_check_tree():
    """Sanity-check MEMBERS + winning weights against experiments_tree.json node #15
    before doing any (expensive) training, so a stale hand-transcription of the
    weights/params is caught immediately."""
    tree = json.load(open(TREE_JSON))
    by_id = {n["id"]: n for n in tree["nodes"]}
    node15 = by_id[WINNING_NODE_ID]
    assert node15["config"]["kind"] == "blend"
    tree_members = sorted(node15["config"]["members"])
    our_members = sorted(m["node_id"] for m in MEMBERS)
    assert tree_members == our_members, (
        f"member id mismatch: tree node #15 has {tree_members}, script has {our_members}")
    tree_weights = node15["config"]["result"]["weights"]
    tree_rounded = node15["config"]["result"]["mae_rounded"]
    assert abs(tree_rounded - TARGET_ROUNDED_MAE) < 1e-9, (
        f"experiments_tree.json node #15 mae_rounded={tree_rounded} != "
        f"expected {TARGET_ROUNDED_MAE}")
    for m in MEMBERS:
        node = by_id[m["node_id"]]
        assert node["config"]["kind"] == "solo"
        assert node["config"]["model"] == ("cat" if m["name"] == "CAT" else
                                            ("xgb" if m["name"] == "XGB" else "lgb"))
        tree_params = node["config"]["params"]
        # For legacy-reused members (XGB/CAT/LGB/LGB_tuned/LGB_tuned_seed2024),
        # experiments_tree.json's stored "params" is documentation-only metadata
        # (the identifying hyperparams), while the ACTUAL training call (pool_lib.py's
        # train_xgb/train_cat/train_lgb) bakes in a few extra runtime-only keys
        # (n_jobs, eval_metric, early_stopping_rounds, subsample_freq, verbose) not
        # present in that metadata dict. So the check here is "every key the tree
        # records must match this script's value exactly" (subset containment), not
        # dict equality -- any genuine mismatch on a shared key still fails loudly.
        mismatched = {k: (tree_params[k], m["params"].get(k)) for k in tree_params
                      if m["params"].get(k) != tree_params[k]}
        assert not mismatched, (
            f"param mismatch for node #{m['node_id']} ({m['name']}): "
            f"{mismatched} (tree vs script)")
        tree_drop = (node["config"].get("features") or {}).get("drop", [])
        assert tree_drop == m["drop"], (
            f"feature-drop mismatch for node #{m['node_id']}: tree={tree_drop} "
            f"vs script={m['drop']}")
    print(f"[check] experiments_tree.json node #{WINNING_NODE_ID} verified: "
          f"members={tree_members}, target rounded MAE={tree_rounded}")
    return tree_weights


def main():
    t_start = time.time()
    print("=== rebuild_tree_best.py: reconstructing tree-search winning blend (node "
          f"#{WINNING_NODE_ID}, target rounded OOF MAE {TARGET_ROUNDED_MAE}) ===\n")
    tree_weights = load_and_check_tree()

    oofs = {}
    preds = {}
    gate_rows = []
    for m in MEMBERS:
        nid, name = m["node_id"], m["name"]
        print(f"--- training member #{nid} ({name}) ---")
        t0 = time.time()
        Xnp, Xtestnp, feats = feature_frame(m["drop"])
        oof, pred = m["runner"](m["params"], Xnp, Xtestnp)
        dt = time.time() - t0
        r_raw, r_rounded = raw_mae(oof), round_clip_mae(oof)
        cached_oof = load_cached_oof(m["cache_kind"], m["cache_name"])
        max_abs_diff = float(np.max(np.abs(oof - cached_oof)))
        ok = max_abs_diff <= GATE_TOL
        gate_rows.append(dict(node_id=nid, name=name, raw_mae=r_raw, rounded_mae=r_rounded,
                               max_abs_diff=max_abs_diff, gate_ok=ok, wall_s=dt,
                               n_feats=len(feats)))
        print(f"    reconstructed raw={r_raw:.6f} rounded={r_rounded:.6f} "
              f"| max|diff vs cache|={max_abs_diff:.3e} | gate={'PASS' if ok else 'FAIL'} "
              f"| {dt:.1f}s")
        if not ok:
            print(f"\nGATE FAILED for member #{nid} ({name}): reconstructed OOF does not "
                  f"match cached OOF within tol={GATE_TOL}. STOPPING -- no submission "
                  f"file will be written.")
            sys.exit(1)
        oofs[nid] = oof
        preds[nid] = pred

    # --- recover node #15's exact full-precision weights by replaying its own
    #     deterministic weight search over the gate-verified reconstructed OOFs ---
    members_order = [m["node_id"] for m in MEMBERS]  # [0,1,2,3,4,5,6,8], config order
    oof_stack = np.stack([oofs[nid] for nid in members_order], axis=1)
    pred_stack = np.stack([preds[nid] for nid in members_order], axis=1)

    w, blend_rounded = recover_exact_weights(oof_stack)
    blend_raw = raw_mae(oof_stack @ w)
    print(f"\n=== recovered blend (weight-search replay, member order {members_order}) ===")
    print(f"full-precision weights = {dict(zip(members_order, np.round(w, 6)))}")
    print(f"reconstructed blend raw MAE     = {blend_raw:.6f}")
    print(f"reconstructed blend rounded MAE = {blend_rounded:.6f}  (target {TARGET_ROUNDED_MAE})")

    weights_r4_ok = all(
        abs(round(float(wi), 4) - tree_weights[str(nid)]) < 1e-12
        for nid, wi in zip(members_order, w))
    score_ok = round(blend_rounded, 5) == TARGET_ROUNDED_MAE
    print(f"recovered weights round(4) == tree-stored weights: "
          f"{'PASS' if weights_r4_ok else 'FAIL'}")
    print(f"\nFINAL GATE: {'PASS' if (score_ok and weights_r4_ok) else 'FAIL'}")
    if not (score_ok and weights_r4_ok):
        if not score_ok:
            print(f"Blended rounded OOF MAE {round(blend_rounded, 5)} != "
                  f"{TARGET_ROUNDED_MAE}. STOPPING -- no submission file will be written.")
        if not weights_r4_ok:
            print(f"Recovered weights {dict(zip(members_order, np.round(w, 4)))} do not "
                  f"round to tree-stored {tree_weights}. STOPPING.")
        sys.exit(1)

    # --- test predictions: blend, round, clip (SAME convention as pool_lib.write_submission) ---
    blend_pred = pred_stack @ w
    sub_pred = np.clip(np.round(blend_pred), AGE_LOW, None)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB_DIR, f"sub_tree_best_1.33563_{stamp}.csv")
    sub = pd.DataFrame({ID: _test_ids, TARGET: sub_pred.astype(int)})
    sub.to_csv(sub_path, index=False)

    # --- format validation vs sample_submission.csv ---
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    assert len(sub) == len(sample), (len(sub), len(sample))
    assert (sub[ID].to_numpy() == sample[ID].to_numpy()).all(), "id order mismatch"
    assert sub[TARGET].min() >= AGE_LOW
    assert np.array_equal(sub[TARGET].to_numpy(), sub[TARGET].to_numpy().astype(int))

    total_wall = time.time() - t_start
    print(f"\nWrote submission: {sub_path}")
    print(f"rows={len(sub)}  {TARGET} range=[{sub[TARGET].min()}, {sub[TARGET].max()}]")
    print(f"Total wall time: {total_wall:.1f}s")

    # --- summary json for STATUS.md / audit trail ---
    summary = dict(
        winning_node_id=WINNING_NODE_ID,
        target_rounded_mae=TARGET_ROUNDED_MAE,
        reconstructed_blend_raw_mae=round(blend_raw, 5),
        reconstructed_blend_rounded_mae=round(blend_rounded, 5),
        recovered_weights={str(nid): float(wi) for nid, wi in zip(members_order, w)},
        recovered_weights_round4_match_tree=True,
        gate_pass=True,
        members=gate_rows,
        submission_file=os.path.basename(sub_path),
        total_wall_s=round(total_wall, 1),
    )
    summary_path = os.path.join(_HERE, "rebuild_tree_best_result.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
