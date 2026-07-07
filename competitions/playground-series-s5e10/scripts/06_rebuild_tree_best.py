"""06_rebuild_tree_best.py -- Rebuild the s5e10 stage-4 tree-search winner END TO END
and emit a REAL test submission file, gated by digit-for-digit OOF reproduction
(s3e16/s4e1/s4e11 rebuild precedent: reads the winner from experiments_tree_v3.json
instead of hardcoding a node id, and handles both solo and blend winners).

Why this script exists: blend nodes in the stage-4 search are scored purely as
OOF-space weight searches over CACHED member OOF vectors (tree_search/eval_s5e10.py
evaluate_blend -> harness eval_blend); no test predictions were ever combined into a
submission for the winning composition. This script:

  1. Loads competitions/playground-series-s5e10/experiments_tree_v3.json, takes the
     global-best evaluated node (min score; scores ARE clipped-OOF RMSE, so min =
     best -- no sign flip in this comp).
  2. RETRAINS every member (or the winner itself if solo) from the exact config
     stored in the tree, via tree_search/eval_s5e10.py's evaluate_solo --
     byte-identical features.py + make_folds, the same entry point the search used.
  3. GATE 1: each member's reconstructed OOF must match its cached OOF
     (tree_search/cache_s5e10/solo_<id>.npz) to max-abs-diff <= 1e-9.
  4. For a blend winner: replays the winning node's own fully deterministic weight
     search (harness_v3.eval_blend defaults: dirichlet k=800 seed=42 + coordinate
     ascent), scored through the SAME clip-embedding metric_fn
     (`eval_s5e10.rmse(y, vec)`) the search itself used, over the gate-verified
     reconstructed OOFs to recover the FULL-PRECISION weight vector.
  5. GATE 2: recovered blend OOF RMSE (round 6) must equal the tree-stored RMSE
     exactly, and recovered weights must round(4) to the tree-stored weights
     member-for-member.
  6. Only then: blends the freshly-produced test predictions with the full-precision
     weights, applies the comp's metric-aware post-processing (clip to the target's
     [0,1] range -- RMSE submits continuous values, no threshold/rounding), and
     writes submissions/sub_tree_best_<rmse>_<stamp>.csv.

If any gate fails, the script STOPS with a nonzero exit and no submission is written.

Run from repo root:
  uv run python3 competitions/playground-series-s5e10/scripts/06_rebuild_tree_best.py
"""
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMP_DIR = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_COMP_DIR))
_TREE_SEARCH_DIR = os.path.join(_REPO_ROOT, "tree_search")
TREE_JSON = os.path.join(_COMP_DIR, "experiments_tree_v3.json")
TREE_CACHE_DIR = os.path.join(_TREE_SEARCH_DIR, "cache_s5e10")
REBUILD_CACHE_DIR = os.path.join(_HERE, "cache", "rebuild_stage4")  # under gitignored scripts/cache/
SUB_DIR = os.path.join(_COMP_DIR, "submissions")
DATA = os.path.join(_COMP_DIR, "data")
GATE_TOL = 1e-9

sys.path.insert(0, _TREE_SEARCH_DIR)
import harness_v2 as hv2  # noqa: E402
import harness_v3 as hv3  # noqa: E402


def _load_eval_module():
    """Import tree_search/eval_s5e10.py by file path (not a package module). Its
    module-level code loads train/test and builds the canonical 27-feature frame +
    folds once -- the same initialization the search's own subprocess evals ran."""
    path = os.path.join(_TREE_SEARCH_DIR, "eval_s5e10.py")
    spec = importlib.util.spec_from_file_location("eval_s5e10_rebuild", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t_start = time.time()
    tree = json.load(open(TREE_JSON))
    scored = [n for n in tree["nodes"] if n["status"] == "evaluated" and n["score"] is not None]
    winner = min(scored, key=lambda n: n["score"])   # scores are clipped-OOF RMSE
    target_rmse = round(winner["score"], 6)
    kind = winner["config"].get("kind", "solo")
    node_results = tree.get("search_state", {}).get("node_results", {})
    stored_result = node_results.get(str(winner["id"])) or {}
    print(f"=== 06_rebuild_tree_best: winner = node #{winner['id']} ({kind}), "
          f"target OOF RMSE {target_rmse:.6f} ===")
    print(f"mutation: {winner['mutation'][:120]}")

    print("\n[init] importing eval_s5e10 (loads data + builds canonical features/folds)...")
    ev = _load_eval_module()
    by_id = {n["id"]: n for n in tree["nodes"]}

    if kind == "solo":
        member_ids = [winner["id"]]
    else:
        member_ids = list(winner["config"]["members"])  # stored sorted by core()

    # --- retrain every member + GATE 1 (per-member OOF digit reproduction) ---
    os.makedirs(REBUILD_CACHE_DIR, exist_ok=True)
    oofs, preds, gate_rows = {}, {}, []
    for mid in member_ids:
        node = by_id[mid]
        assert node["config"].get("kind", "solo") == "solo", (
            f"member #{mid} is not a solo node -- unsupported composition")
        t0 = time.time()
        oof, pred, score, feats = ev.evaluate_solo(node["config"])
        dt = time.time() - t0
        cached = np.load(os.path.join(TREE_CACHE_DIR, f"solo_{mid}.npz"))
        max_abs_oof = float(np.max(np.abs(oof - cached["oof"])))
        max_abs_pred = float(np.max(np.abs(pred - cached["pred"])))
        ok = max_abs_oof <= GATE_TOL
        gate_rows.append(dict(node_id=mid, model=node["config"]["model"],
                              rmse=round(score, 6), max_abs_diff_oof=max_abs_oof,
                              max_abs_diff_pred=max_abs_pred, gate_ok=ok,
                              wall_s=round(dt, 1), n_feats=len(feats)))
        print(f"[member #{mid}] {node['config']['model']} RMSE={score:.6f} "
              f"| max|dOOF|={max_abs_oof:.2e} max|dPRED|={max_abs_pred:.2e} "
              f"| gate={'PASS' if ok else 'FAIL'} | {dt:.1f}s")
        if not ok:
            print(f"\nGATE 1 FAILED for member #{mid}: reconstructed OOF does not match "
                  f"the cached OOF the search scored (tol={GATE_TOL}). STOPPING.")
            sys.exit(1)
        oofs[mid], preds[mid] = oof, pred
        hv2.cache_oof(REBUILD_CACHE_DIR, mid, oof, pred=pred, rmse=score)

    # --- recover weights (blend) or take solo directly ---
    if kind == "blend":
        print(f"\n[replay] harness_v3.eval_blend defaults (dirichlet k=800 seed=42 + "
              f"coordinate ascent) over {len(member_ids)} gate-verified fresh OOFs, "
              f"scored through the SAME clip-embedding metric the search used...")
        best_w, best_s, _ = hv3.eval_blend(REBUILD_CACHE_DIR, member_ids,
                                           lambda vec: ev.rmse(ev._y, vec))
        blend_rmse = round(best_s, 6)
        stored_weights = stored_result.get("weights")
        w_r4 = [round(float(w), 4) for w in best_w]
        print(f"recovered blend OOF RMSE = {blend_rmse:.6f} (target {target_rmse:.6f})")
        print(f"recovered weights round(4) = {w_r4}")
        print(f"tree-stored weights        = {stored_weights}")
        score_ok = blend_rmse == target_rmse
        weights_ok = (stored_weights is not None
                      and w_r4 == [round(float(x), 4) for x in stored_weights])
        print(f"GATE 2: score {'PASS' if score_ok else 'FAIL'} | weights "
              f"{'PASS' if weights_ok else 'FAIL'}")
        if not (score_ok and weights_ok):
            print("STOPPING -- no submission written from a mismatched replay.")
            sys.exit(1)
        oof_stack = np.stack([oofs[m] for m in member_ids], axis=1)
        pred_stack = np.stack([preds[m] for m in member_ids], axis=1)
        final_oof = oof_stack @ np.asarray(best_w, dtype=float)
        final_pred_raw = pred_stack @ np.asarray(best_w, dtype=float)
        final_weights = {str(m): float(w) for m, w in zip(member_ids, best_w)}
    else:
        solo_rmse = gate_rows[0]["rmse"]
        assert solo_rmse == target_rmse, (
            f"solo winner OOF RMSE {solo_rmse} != tree-stored {target_rmse} -- STOPPING")
        final_oof = oofs[winner["id"]]
        final_pred_raw = preds[winner["id"]]
        final_weights = {str(winner["id"]): 1.0}

    # --- re-check the final OOF score through the comp's post-processing (clip) ---
    final_rmse = ev.rmse(ev._y, final_oof)
    print(f"\n[final] reconstructed final OOF RMSE (clip[0,1]) = {final_rmse:.6f} "
          f"(target {target_rmse:.6f})")
    assert round(final_rmse, 6) == target_rmse, (
        f"re-derived final OOF RMSE {final_rmse} != tree-stored {target_rmse} -- STOPPING")
    final_pred = np.clip(final_pred_raw, 0.0, 1.0)

    # --- write + validate submission ---
    test_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=["id"])["id"]
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(SUB_DIR, f"sub_tree_best_{target_rmse:.5f}_{stamp}.csv")
    sub = pd.DataFrame({"id": test_ids, "accident_risk": final_pred})
    assert list(sub.columns) == list(sample.columns), (sub.columns, sample.columns)
    assert len(sub) == len(sample), (len(sub), len(sample))
    assert (sub["id"].to_numpy() == sample["id"].to_numpy()).all(), "id order mismatch"
    assert sub["accident_risk"].between(0, 1).all(), "predictions must lie in [0,1]"
    assert sub["accident_risk"].notna().all(), "NaN in predictions"
    sub.to_csv(sub_path, index=False)
    total_wall = time.time() - t_start
    print(f"\nALL GATES PASSED. Wrote submission: {sub_path}")
    print(f"rows={len(sub)}  pred stats: min={final_pred.min():.4f} "
          f"max={final_pred.max():.4f} mean={final_pred.mean():.4f}")
    print(f"Total wall: {total_wall:.1f}s")

    summary = dict(winner_node_id=winner["id"], kind=kind, target_oof_rmse=target_rmse,
                   members=gate_rows, recovered_weights=final_weights, gate_pass=True,
                   submission_file=os.path.basename(sub_path),
                   total_wall_s=round(total_wall, 1))
    out = os.path.join(_HERE, "rebuild_tree_best_result.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {out}")


if __name__ == "__main__":
    main()
