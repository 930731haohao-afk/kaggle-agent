# afsis-soil-properties — STATUS

**Final CV: MCRMSE 0.444076** (honest, leave-one-outer-fold-out re-fit of the blend weights; the same
blend scores 0.428772 when its weights are fit in-sample on the full OOF). Validation is a nested
GroupKFold(5) over the 580 hidden site groups recovered from the spatial signature, with each target's
ridge alpha selected on outer-train rows only.
Champion: per-target greedy (Caruana forward-selection-with-replacement) blend of 64 solo nodes —
kernel ridge / SVR over five spectral preprocessing variants, plus PLS and LightGBM-on-PCA for diversity.
Produced by the Stage 4 tree search (harness_v3, 80 evaluated nodes, 74 s), which beat the linear-iteration
best at evaluation 17. Submission: `submission.csv` (727 rows, PIDN order per sample_submission.csv).

---

## Pipeline as run

| Stage | What happened |
|---|---|
| 0 Setup | `config.yaml` already present; data/ symlinks to the official files. |
| 1 EDA | `scripts/eda.py`. 1157x3600 train / 727 test, zero nulls. **580 distinct spatial signatures across 1157 rows** — 565 sites contribute a Topsoil/Subsoil pair, so a plain KFold leaks site location. Zero train/test site overlap. Targets are heavily skewed (P 7.45, Ca 4.71). |
| 2 Features | `scripts/features.py`. CO2 band (2352–2380 cm⁻¹, 15 columns) dropped; five spectral preprocessing variants (raw, SNV, SG-d1+SNV, SG-d2+SNV, SG-d1(w11)+SNV) plus z-scored spatial covariates + Depth; site-group labels saved for GroupKFold. |
| 3 Linear iteration | `scripts/baseline.py` — 16-config solo pool + one full-pool per-target blend, all through `tree_search/eval_afsis.py`. |
| 4 Tree search | `tree_search/run_afsis_v3.py` (harness_v3), 80 evaluated nodes / 67 solo / 13 blend. |
| 5 Submission | `scripts/finalize.py` — honest re-scoring, `submission.csv`, experiment log. |

## Scores

| Model | MCRMSE | Note |
|---|---|---|
| Per-fold target mean | 1.024945 | naive floor |
| Ridge, linear kernel, SG-d1 spectra + spatial | 0.507098 | linear baseline |
| KRR-RBF, SG-d1 + spatial | 0.475432 | kernelization is worth ~0.032 over linear ridge |
| SVR-RBF, SG-d1 + spatial (best linear-stage solo, = tree root) | 0.471939 | honest |
| Best tree-search solo (KRR-RBF on SNV, spatial_weight 0.25) | 0.469086 | honest |
| Linear-stage blend, 15 members | 0.433219 | weights fit in-sample |
| **Tree-search champion, 64 members** | **0.428772** | weights fit in-sample |
| **Tree-search champion, leave-fold-out weights** | **0.444076** | **the honest number** |

Solo scores are honest by construction (alpha chosen by an inner GroupKFold(4) that only ever sees
outer-train rows). Blend scores are not, until the weights are re-fit out-of-fold — the measured
weight-fitting optimism here is **+0.015304**, squarely inside the +0.0146…+0.0186 band
`knowledge/experience.md` predicts for this search.

## Tree search — honest reporting (07_tree_search.md §6)

- **evals-to-beat**: evaluation **17** of 80 first went below the linear-iteration best (0.433219).
- **Budget / stopping**: hard cap, 80/80 evaluated nodes. `stop_reason = "hard budget cap reached"` —
  the post-burst patience counter never fired, because the burst was still improving when the cap hit.
  The budget was raised from the documented default 60 to 80 for one specific reason: 15 of the first 16
  nodes are zero-compute cached-OOF re-imports of the linear pool, so ~60 nodes of real search remain.
- **Explore burst**: mandatory, injected at n_eval=70. It produced the champion (#73, the kitchen-sink
  blend of all 64 solos) — the same pattern the reference documents, where every post-exploit gain came
  from the injected kitchen-sink blend rather than a long-shot solo.
- **Sanity gate**: fired once. Burst seed #71 (`EXPL_HIGHREG`, linear kernel on raw spectra, heavy ridge
  only) scored 0.686914 against a bound of 0.549675 and its lineage was plateaued immediately, so no
  children were trained off it.
- **Cost guard**: never fired (no blend evaluation approached the 45 s threshold; the whole 80-node
  search took 74 s).
- **Backtracks / plateaus**: 19 lineages plateaued, all through the normal "expansion budget exhausted"
  or "mutation space exhausted" paths; the full log is in
  `experiments_tree_v3.json → search_state.backtrack_log`.
- **Dedup rejections**: 12. Not alarming for a 15-lineage node space, but concentrated in the blend
  lineage, where several (top-k, method) plan entries collapse to the same member list once the pool
  stops growing.
- **OOF-only caveat**: this competition closed in 2014, so **nothing here is LB-validated**. Every score
  above is an OOF/nested-CV number.
- **Small-sample caveat**: the champion beats the second-best blend (#75, same 64 members, Dirichlet
  weights) by 0.000127 — that margin is noise, not signal. What is *not* noise is the gap to the
  16-member blends (~0.0043) and to the best solo (~0.040).

## The one real finding: member diversity, not weight search

The first tree-search run re-seeded only 4 of the 16 linear-pool solos, so its blend lineage could only
mix models the search itself had generated. Its kitchen-sink blend scored **0.449717 — worse than the
linear blend it was supposed to beat.** Re-seeding the entire linear pool as first-generation lineages,
with no other change to node kinds or budget, moved the champion to 0.428772. That run-1 tree is archived
at `experiments_tree_v3_run1_diluted_blend.json` and logged as experiment #6.

The weight-search method turned out to be second-order. At 64 members: greedy 0.428772, Dirichlet
0.428899, bagged greedy 0.430371. At 16 members all three land within 0.000006 of each other. My initial
hypothesis — that Dirichlet draws dilute over a large member set — is not what the data shows; pool
composition explains essentially the whole gap.

Per-target weights, by contrast, matter a lot: 0.433071 with per-target weights vs 0.452840 with one
shared weight vector over the same 16 members.

## Files

- `scripts/eda.py`, `scripts/features.py`, `scripts/baseline.py`, `scripts/finalize.py`
- `tree_search/eval_afsis.py` (node evaluator), `tree_search/run_afsis_v3.py` (driver)
- `experiments.json` (6 entries), `experiments_tree_v3.json` (search tree), `scripts/state/final.json`
- `submission.csv` / `submissions/submission_tree_v3_node73.csv`
