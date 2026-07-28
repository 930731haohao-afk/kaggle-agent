"""Stage 3 (linear iteration protocol) for afsis-soil-properties.

Establishes the naive/linear baselines, a first solo pool over the spectral
preprocessing variants and model families, and one greedy per-target blend — the
precondition the skill sets before tree search may take over (references/07_tree_search.md
§1). Every config is evaluated through tree_search/eval_afsis.py, so the best solo found
here is bit-identical to the tree-search root.

Solo node ids 9000+ are reserved for this linear pool (tree search uses 0+).
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "tree_search")
import eval_afsis as ev  # noqa: E402

COMP_DIR = "competitions/afsis-soil-properties"
_spec = importlib.util.spec_from_file_location(
    "experiment_log", ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
experiment_log = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment_log)

POOL = [
    # naive + linear baselines
    ("mean", dict(kind="solo", model="mean", variant="sg1", use_spatial=False)),
    ("ridge_linear_sg1", dict(kind="solo", model="krr", variant="sg1", use_spatial=True,
                              params=dict(kernel="linear"))),
    ("ridge_linear_snv", dict(kind="solo", model="krr", variant="snv", use_spatial=True,
                              params=dict(kernel="linear"))),
    # KRR-RBF across preprocessing variants (experience.md: kernelization is the big lever)
    ("krr_rbf_raw", dict(kind="solo", model="krr", variant="raw", use_spatial=True,
                         params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_rbf_snv", dict(kind="solo", model="krr", variant="snv", use_spatial=True,
                         params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_rbf_sg1", dict(kind="solo", model="krr", variant="sg1", use_spatial=True,
                         params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_rbf_sg2", dict(kind="solo", model="krr", variant="sg2", use_spatial=True,
                         params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_rbf_sg1w11", dict(kind="solo", model="krr", variant="sg1w11", use_spatial=True,
                            params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_rbf_sg1_nospatial", dict(kind="solo", model="krr", variant="sg1", use_spatial=False,
                                   params=dict(kernel="rbf", gamma_scale=1.0))),
    ("krr_lap_sg1", dict(kind="solo", model="krr", variant="sg1", use_spatial=True,
                         params=dict(kernel="laplacian", gamma_scale=1.0))),
    ("krr_poly2_snv", dict(kind="solo", model="krr", variant="snv", use_spatial=True,
                           params=dict(kernel="poly", degree=2, gamma_scale=1.0, coef0=1.0))),
    # diverse families (experience.md: weak-but-decorrelated members still earn weight)
    ("svr_rbf_sg1", dict(kind="solo", model="svr", variant="sg1", use_spatial=True,
                         params=dict(kernel="rbf", gamma_scale=1.0, C=10.0, epsilon=0.1))),
    ("pls_sg1", dict(kind="solo", model="pls", variant="sg1", use_spatial=True,
                     params=dict(n_components=20))),
    ("pls_snv", dict(kind="solo", model="pls", variant="snv", use_spatial=True,
                     params=dict(n_components=25))),
    ("lgbpca_sg1", dict(kind="solo", model="lgbpca", variant="sg1", use_spatial=True,
                        params=dict(n_components=40))),
    ("krr_rbf_spatialonly", dict(kind="solo", model="krr", variant="spatial", use_spatial=False,
                                 params=dict(kernel="rbf", gamma_scale=1.0))),
]


def main():
    results = {}
    for i, (name, cfg) in enumerate(POOL):
        nid = 9000 + i
        r = ev.evaluate(cfg, node_id=nid, timeout_s=900)
        results[name] = dict(node_id=nid, config=cfg, **r)
        if r["status"] == "evaluated":
            pt = r["result"]["per_target"]
            print(f"#{nid} {name:24s} MCRMSE {r['score']:.6f}  {r['wall_s']:6.1f}s  "
                  f"Ca {pt['Ca']:.3f} P {pt['P']:.3f} pH {pt['pH']:.3f} "
                  f"SOC {pt['SOC']:.3f} Sand {pt['Sand']:.3f}", flush=True)
        else:
            print(f"#{nid} {name:24s} FAILED: {r['error']}", flush=True)

    ok = {k: v for k, v in results.items() if v["status"] == "evaluated"}
    best_name = min(ok, key=lambda k: ok[k]["score"])
    print(f"\nbest solo: {best_name} {ok[best_name]['score']:.6f}")

    # first blend: every non-degenerate member, per-target weights
    members = [v["node_id"] for k, v in ok.items() if k != "mean"]
    bcfg = dict(kind="blend", members=sorted(members), per_target=True,
                weight_search="dirichlet")
    rb = ev.evaluate(bcfg, node_id=9500, timeout_s=900)
    print(f"blend(all {len(members)}) MCRMSE {rb['score']:.6f}  {rb['wall_s']:.1f}s")
    print("weights:", json.dumps(rb["result"]["weights"], indent=1))

    experiment_log.log_experiment_v2(
        COMP_DIR, model="mean-predictor", metric="MCRMSE", direction="minimize",
        score=ok["mean"]["score"],
        cv=dict(scheme="nested GroupKFold(5) over 580 spatial-signature site groups",
                n_splits=ev.N_OUTER, seed=ev.SEED),
        notes="Stage 3 naive baseline: per-fold target mean.")
    experiment_log.log_experiment_v2(
        COMP_DIR, model="Ridge (linear kernel, dual) on SG-d1+SNV spectra + spatial",
        metric="MCRMSE", direction="minimize", score=ok["ridge_linear_sg1"]["score"],
        cv=dict(scheme="nested GroupKFold(5) outer / GroupKFold(4) inner per-target alpha",
                n_splits=ev.N_OUTER, seed=ev.SEED),
        notes="Stage 3 linear baseline. Alpha selected per target on outer-train rows only.")
    experiment_log.log_experiment_v2(
        COMP_DIR, model=f"solo pool ({len(ok)} configs), best = {best_name}",
        metric="MCRMSE", direction="minimize", score=ok[best_name]["score"],
        cv=dict(scheme="nested GroupKFold(5) outer / GroupKFold(4) inner per-target alpha",
                n_splits=ev.N_OUTER, seed=ev.SEED),
        base_models=[dict(name=k, node_id=v["node_id"], score=v["score"]) for k, v in
                     sorted(ok.items(), key=lambda kv: kv[1]["score"])],
        notes="Stage 3 solo pool over preprocessing variants x kernels x model families.")
    if rb["status"] == "evaluated":
        experiment_log.log_experiment_v2(
            COMP_DIR, model="per-target weighted blend of full linear solo pool",
            metric="MCRMSE", direction="minimize", score=rb["score"],
            cv=dict(scheme="nested GroupKFold(5); blend weights fit on the same OOF",
                    n_splits=ev.N_OUTER, seed=ev.SEED),
            ensemble=dict(method="per-target dirichlet + coordinate ascent",
                          members=bcfg["members"], weights=rb["result"]["weights"]),
            notes="Stage 3 first blend -> satisfies the tree-search entry condition "
                  "(baseline solo + >=1 blend). Weights fit on the full OOF, so this "
                  "number carries greedy-weight optimism (~+0.016 per experience.md).")

    os.makedirs(f"{COMP_DIR}/scripts/state", exist_ok=True)
    with open(f"{COMP_DIR}/scripts/state/linear_pool.json", "w") as f:
        json.dump({k: dict(node_id=v["node_id"], config=v["config"], score=v["score"],
                           status=v["status"]) for k, v in results.items()}
                  | {"_blend": dict(node_id=9500, config=bcfg, score=rb["score"],
                                    status=rb["status"], result=rb["result"])}, f, indent=1)
    print("\nlinear-stage state saved.")


if __name__ == "__main__":
    main()
