"""Stage 5 for afsis-soil-properties: honest scoring of the tree-search champion,
submission generation, and experiment logging.

Three numbers are reported and they mean different things:
  * champion OOF MCRMSE — blend weights fit on the SAME OOF they are scored on, so it
    carries greedy-weight optimism (experience.md quantifies that at ~+0.016-0.018 for
    this comp's forward-selection search).
  * leave-fold-out MCRMSE — the identical blend PROCEDURE re-fit five times, each time
    on four outer folds' OOF rows and scored on the held-out fold. This is the honest
    number for the whole pipeline and is what STATUS.md reports.
  * per-member solo scores — already honest (nested inner-CV alpha selection).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tree_search")
import harness as hv1      # noqa: E402
import harness_v3 as hv3   # noqa: E402
import eval_afsis as ev    # noqa: E402

COMP_DIR = "competitions/afsis-soil-properties"
TREE_PATH = f"{COMP_DIR}/experiments_tree_v3.json"
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)


def leave_fold_out_blend(members, method, per_target=True):
    """Re-fit the blend weights five times, each on 4/5 of the OOF rows, and score on
    the held-out fold. Caveat kept explicit in the write-up: a member's OOF row inside
    the weight-fitting portion still comes from a model that trained on the held-out
    fold's rows, so this bounds the WEIGHT-fitting optimism, not every source of it."""
    oofs = np.stack([np.load(f"{ev.CACHE_DIR}/solo_{m}.npz")["oof"] for m in members],
                    axis=1)
    out = np.zeros((ev.N_TRAIN, ev.N_TARGETS))
    for tr, va in ev._FOLDS:
        if per_target:
            for t in range(ev.N_TARGETS):
                w, _ = ev._weight_search(oofs[tr][:, :, t], ev._Y[tr, t], method)
                out[va, t] = oofs[va][:, :, t] @ w
        else:
            flat = oofs[tr].transpose(0, 2, 1).reshape(len(tr) * ev.N_TARGETS, len(members))
            w, _ = ev._weight_search(flat, ev._Y[tr].reshape(-1), method)
            out[va] = np.einsum("nmt,m->nt", oofs[va], w)
    return ev.mcrmse(ev._Y, out), out


def main():
    tree = hv3.load_search_state(TREE_PATH)
    gb = hv1.global_best(tree)
    cfg = gb["config"]
    res = tree["search_state"]["driver_state"]["node_results"][str(gb["id"])]
    print(f"champion #{gb['id']} kind={cfg.get('kind')} OOF MCRMSE={gb['score']}")
    print(f"  mutation: {gb['mutation'][:150]}")

    # re-evaluate the champion from cache and verify it reproduces digit-for-digit
    check = ev.evaluate(cfg, node_id=None)
    assert round(check["score"], 6) == round(gb["score"], 6), \
        f"champion did not reproduce: {check['score']} vs {gb['score']}"
    print(f"  champion reproduces from cache: {check['score']:.6f}")

    members = cfg["members"]
    method = cfg.get("weight_search", "dirichlet")
    per_target = bool(cfg.get("per_target", True))
    honest, honest_oof = leave_fold_out_blend(members, method, per_target)
    optimism = honest - gb["score"]
    print(f"  leave-fold-out (honest) MCRMSE = {honest:.6f}  "
          f"(greedy-weight optimism +{optimism:.6f})")
    per_t_honest = {ev.TARGETS[t]: round(ev.rmse(ev._Y[:, t], honest_oof[:, t]), 6)
                    for t in range(ev.N_TARGETS)}
    print("  honest per-target RMSE:", per_t_honest)

    # best honest solo, for the "what did the ensemble buy" line
    solos = [n for n in tree["nodes"] if n["status"] == "evaluated"
             and n["config"].get("kind") == "solo" and n["score"] is not None]
    best_solo = min(solos, key=lambda n: n["score"])
    print(f"  best solo #{best_solo['id']} MCRMSE={best_solo['score']:.6f} "
          f"({best_solo['mutation'][:80]})")

    # ---- submission ----
    d = np.load(f"{ev.CACHE_DIR}/solo_{gb['id']}.npz", allow_pickle=True)
    pred = d["pred"]
    assert pred.shape == (ev.N_TEST, ev.N_TARGETS), pred.shape
    sample = pd.read_csv(f"{COMP_DIR}/data/sample_submission.csv")
    test_ids = pd.read_csv(f"{COMP_DIR}/scripts/test_ids.csv")["PIDN"]
    assert list(test_ids) == list(sample["PIDN"]), "test row order != sample_submission order"

    sub = pd.DataFrame({"PIDN": sample["PIDN"]})
    for t, name in enumerate(ev.TARGETS):
        sub[name] = pred[:, t]
    sub = sub[list(sample.columns)]
    assert sub.shape == sample.shape, (sub.shape, sample.shape)
    assert not sub.isnull().any().any()
    os.makedirs(f"{COMP_DIR}/submissions", exist_ok=True)
    sub.to_csv(f"{COMP_DIR}/submissions/submission_tree_v3_node{gb['id']}.csv", index=False)
    sub.to_csv(f"{COMP_DIR}/submission.csv", index=False)
    print(f"\nsubmission written: {sub.shape}")
    print(sub.describe().T[["mean", "std", "min", "max"]].round(3))
    print("train target ranges for comparison:")
    print(pd.DataFrame(ev._Y, columns=ev.TARGETS).describe().T[["mean", "std", "min", "max"]].round(3))

    # ---- experiment log ----
    n_eval = sum(1 for n in tree["nodes"] if n["status"] == "evaluated")
    experiment_log.log_experiment_v2(
        COMP_DIR,
        model=f"tree-search champion: per-target greedy blend of {len(members)} members",
        metric="MCRMSE", direction="minimize", score=honest,
        cv=dict(scheme="nested GroupKFold(5) over 580 spatial-signature site groups; "
                       "blend weights re-fit leave-one-outer-fold-out",
                n_splits=ev.N_OUTER, seed=ev.SEED, per_target_rmse=per_t_honest),
        ensemble=dict(method=f"per-target {method} (Caruana forward selection with "
                             f"replacement, 60 rounds)",
                      n_members=len(members), members=members,
                      weights=res.get("weights"),
                      oof_score_weights_fit_in_sample=round(gb["score"], 6),
                      weight_fitting_optimism=round(optimism, 6)),
        base_models=[dict(node_id=n["id"], score=n["score"], mutation=n["mutation"][:120])
                     for n in sorted(solos, key=lambda n: n["score"])[:10]],
        submission=f"submissions/submission_tree_v3_node{gb['id']}.csv",
        notes=(f"Stage 4 tree search (harness_v3, {n_eval} evaluated nodes, "
               f"{sum(1 for n in tree['nodes'] if n['config'].get('kind') == 'blend')} blend "
               f"/ {len(solos)} solo). Champion came from the MANDATORY explore-burst "
               f"kitchen-sink blend. Beat the linear-iteration best (0.433219 in-sample) "
               f"at evaluation {tree.get('evals_to_match_linear_best')}. "
               f"OOF-only: no Kaggle LB validation (competition closed 2014)."))

    with open(f"{COMP_DIR}/scripts/state/final.json", "w") as f:
        json.dump(dict(champion_node=gb["id"], champion_config=cfg,
                       oof_mcrmse=gb["score"], honest_mcrmse=round(honest, 6),
                       optimism=round(optimism, 6), per_target_honest=per_t_honest,
                       best_solo=dict(node=best_solo["id"], score=best_solo["score"]),
                       n_members=len(members), n_evaluated=n_eval), f, indent=1)
    print("\nfinal state saved.")


if __name__ == "__main__":
    main()
