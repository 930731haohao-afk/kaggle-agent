"""Rewrite experiments.json with the full run history in chronological order,
using the canonical logger. Scores are the ones printed by each iteration
script (see scripts/*.py and the session transcript).
"""
import importlib.util
import json
import os

COMP = "/home/tjyen/ai_agents/kaggle/competitions/afsis-soil-properties"
CV = "GroupKFold-5 on spatial-signature site (seed 42)"

_e = importlib.util.spec_from_file_location(
    "experiment_log",
    "/home/tjyen/ai_agents/kaggle/.claude/skills/kaggle-agent/assets/utils/experiment_log.py")
el = importlib.util.module_from_spec(_e)
_e.loader.exec_module(el)

final = json.load(open(f"{COMP}/scripts/final_blend_result.json"))
os.remove(f"{COMP}/experiments.json")

el.log_experiment_v2(
    COMP, model="mean predictor (naive baseline)", metric="MCRMSE",
    direction="minimize", score=1.02421,
    cv={"strategy": CV, "folds": 5},
    notes="Per-fold training-mean per target. Reference point for all later scores.")

el.log_experiment_v2(
    COMP, model="Ridge (SVD, per-target alpha) on raw spectra, CO2 band dropped",
    metric="MCRMSE", direction="minimize", score=0.48877,
    cv={"strategy": CV, "folds": 5},
    features=["3563 MIR absorbance cols (m7497.96-m599.76 minus 2352-2380 CO2 band)"],
    notes="Best single preprocessing variant of round 1; per-target RMSE "
          "Ca 0.3935 P 0.9609 pH 0.3589 SOC 0.3600 Sand 0.3706. "
          "Adding the 15 spatial cols + Depth gives 0.48668.")

el.log_experiment_v2(
    COMP, model="7-variant Ridge per-target greedy blend", metric="MCRMSE",
    direction="minimize", score=0.47787,
    cv={"strategy": CV, "folds": 5, "note": "full-OOF weight fit (optimistic)"},
    ensemble={"type": "per-target greedy forward selection with replacement",
              "members": ["raw", "sg1", "sg2", "snv", "snv_sg1", "raw_spatial", "sg1_spatial"]},
    notes="Round 1. Equal-weight blend of the same 7 was only 0.48777 — weights must be searched.")

el.log_experiment_v2(
    COMP, model="16-member blend (Ridge + PLS + SVR-RBF)", metric="MCRMSE",
    direction="minimize", score=0.47338,
    cv={"strategy": CV, "folds": 5,
        "note": "NESTED score; full-OOF weight fit was 0.45515 (optimism +0.01823)"},
    ensemble={"type": "per-target greedy forward selection", "n_members_pool": 16},
    postprocess=["per-target shrink calibration (b=0.96-1.04) — noise-level, +0.0004, dropped"],
    notes="Round 2. SVR members earn weight despite weak solo scores (svr_raw solo 0.684, "
          "still 0.20 weight on Ca) — solo strength is not the criterion for pool membership.")

el.log_experiment_v2(
    COMP, model="Kernel Ridge RBF (per-variant), solo", metric="MCRMSE",
    direction="minimize", score=0.45244,
    cv={"strategy": CV, "folds": 5,
        "note": "hyperparameters picked per target on the full OOF — optimistic; "
                "honest inner-CV version of the same model scores 0.46661"},
    notes="Rounds 3-4. KRR-RBF on SNV + Savitzky-Golay 1st-derivative (window 41) is the "
          "strongest single family, beating the best plain Ridge (0.48668) by 0.034. "
          "Kernelising is a bigger lever than any preprocessing choice.")

el.log_experiment_v2(
    COMP, model="52-member blend (leaky-hyperparameter pool)", metric="MCRMSE",
    direction="minimize", score=0.44861,
    cv={"strategy": CV, "folds": 5,
        "note": "NESTED blend weights, but member hyperparameters still chosen on full OOF"},
    ensemble={"type": "bagged greedy (25 bags, 50% member subsample)", "n_members_pool": 52},
    notes="Round 5. Method (plain vs bagged greedy) chosen by nested score, not full-OOF score.")

el.log_experiment_v2(
    COMP, model="log-shifted and winsorized target KRR variants", metric="MCRMSE",
    direction="minimize", score=0.46725,
    cv={"strategy": CV, "folds": 5},
    notes="Round 5, diagnostic. Targets are heavily right-skewed (P 7.45, Ca 4.71, SOC 2.45) "
          "but log-shifted fits are worse than raw-target fits on every variant "
          "(krrlog best 0.47216 vs krr 0.45244); winsorizing at p99 also worse (0.46725). "
          "The metric is plain RMSE, so squashing the tail costs more than it buys. Not used.")

el.log_experiment_v2(
    COMP, model="H_lap_* Laplacian-kernel KRR members added to honest pool",
    metric="MCRMSE", direction="minimize", score=0.44903,
    cv={"strategy": CV, "folds": 5, "note": "NESTED"},
    notes="Round 7, REVERTED. Solo 0.485-0.558, and the 36-member pool's nested score is "
          "worse than the 32-member pool's 0.44817. L1 distance geometry adds no usable "
          "diversity here.")

eid = el.log_experiment_v2(
    COMP,
    model="FINAL: honest-pool 32-member per-target greedy blend (KRR-RBF / Ridge / SVR / PLS)",
    metric="MCRMSE", direction="minimize", score=final["nested"],
    cv={"strategy": CV, "folds": 5, "fold_scores": final["nested_fold_scores"],
        "note": "score = NESTED (leave-fold-out blend-weight fit); full-OOF weight fit gives "
                f"{final['full_oof']:.5f} (optimism +{final['nested']-final['full_oof']:.5f}). "
                "Every member's hyperparameters are selected by inner CV on outer-train rows "
                "only, so the pool carries no selection leakage."},
    base_models=[{"name": k, "weights_per_target": v} for k, v in final["weights"].items()],
    ensemble={"type": "per-target greedy forward selection with replacement",
              "n_members_pool": 32, "n_members_used": len(final["weights"])},
    submission="submission.csv",
    notes="Per-target full-OOF RMSE: " +
          " ".join(f"{k} {v}" for k, v in final["per_target_full_oof"].items()) +
          ". P is ~2.8x the error of every other target and is the binding constraint on MCRMSE.")
print("rewrote experiments.json, final experiment id", eid)
