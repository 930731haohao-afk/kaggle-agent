# A Controlled Ablation Study of a Hybrid LLM+AutoML Agent on Kaggle Tabular Competitions (Technical Report Draft)

> **Draft**: this document synthesizes the entire project into a technical report for research communication, to be later refined into a formal report / paper draft.
> All quantitative results are taken from reproducible raw logs and pass automatic consistency verification (see the cited files).

## Abstract

We evaluate whether a hybrid agent, "Claude Code (large language model agent) + gradient-boosted-tree AutoML (LightGBM/XGBoost/
CatBoost)," can autonomously complete the full data-science pipeline of a Kaggle tabular competition, and use **controlled
ablation** to quantify the marginal contribution of each autonomous capability. The approach is to solve the same competition once under each incremental configuration—(1) no-skill
baseline, (2) adding the structured competition-workflow skill, (3) adding linear self-iteration and a cross-competition experience library, (4) adding tree search (the candidate tree
replacing the linear single path), (5) adding external-literature idea injection—and then compare the scores. Across 15 competitions (10 same-season Season 3 main
benchmark + 5 cross-Season 4–6, five metrics), the incremental ladder **holds in every competition, with no competition regressing overall at any stage**;
the gains concentrate on competitions where "the data hides structural insight." The cross-season recipe still holds but with converging magnitudes. A paired
bootstrap on the five cross-season competitions shows that the total improvement of the end-to-end ladder is **statistically significant**, but individual inter-stage steps can fall within the OOF estimation noise
in smaller-sample competitions. On ERA's second pillar (external-idea injection), we report honestly: the priors were originally write-only (a constructive no-op),
we **built a real injection mechanism and empirically tested it**, and the conclusion is "the mechanism is feasible, but external injection yields no measurable systematic
gain (null) on the test competitions"—a rigorous negative result. This work corresponds to the two pillars of the ERA system by Aygün et al. (2026).

## 1. Introduction

Kaggle tabular competitions are a practical benchmark for evaluating autonomous data-science agents. Existing work mostly reports "whether the end-to-end pipeline can achieve
a good score," and rarely provides **per-capability controlled attribution**. The research question of this paper is: **for each added autonomous capability, how much does the score improve,
is it statistically significant, and under what conditions does it fail?** There are four contributions: (a) a five-stage ablation protocol + a fully automatic,
numerically traceable report-generation pipeline; (b) cross-competition evidence over 15 competitions and five metrics; (c) an honest calibration of cross-season generalization and statistical significance;
(d) the mechanism construction for ERA's second pillar (external-knowledge injection), together with the rigorous negative result of "mechanism feasible, marginal value null."

## 2. Alignment with ERA of Aygün et al.

The ERA system of Aygün et al. (2026, Nature) has two pillars: replacing linear single-path optimization with **candidate tree search**,
and injecting **external knowledge** to guide the search. This work's Stage 4 (custom tree search: candidate tree + backtracking + budget phase machine)
corresponds to the first pillar; Stage 5 (literature idea bank + injection hook + plateau injection) targets the second pillar. Section 6 reports: the second-pillar
prior was originally write-only (J-3); after we wired it up and empirically tested it, the honest conclusion is "mechanism feasible, marginal value null"—
the most honest entry in aligning with ERA.

## 3. Method

**Five-stage ablation**: Stage 1 no-skill baseline → 2 the kaggle-agent skill's structured six stages (EDA→CV design
→features→modeling→metric-aware post-processing→submission) → 3 linear self-iteration (experience-library priors, Optuna, seed bagging)
→ 4 tree search (the candidate tree replaces the linear single path) → 5 external-idea injection. All stages within a competition share the same set of folds, so that scores
are directly comparable.

**Tree search** (custom, evolved over three versions): a node is a candidate configuration (solo or blend), scored by OOF score; after a lineage
plateaus it backtracks, with an explore burst (kitchen-sink mega-blend); the budget phase machine schedules broad-first-then-deep and stops automatically.

**Report generation and dual gates**: report numbers are extracted from the raw logs by a program and verified by a **numeric-traceability gate** (every
number in the report must be traceable to the fact table); submittable competitions additionally have an **OOF bit-by-bit reproducibility gate** (all members of the best solution are retrained
from scratch and their OOF is compared bit-by-bit against the search cache before test predictions are generated). The whole pipeline is managed by uv and reproducible.

## 4. Experimental Setup

15 Kaggle Playground competitions: 10 in Season 3 (S3, same-season weekend batch, main benchmark) + 5 across Seasons 4–6
(S4–S6). Five official metrics: RMSE, ROC-AUC, accuracy, QWK, R². Cross-validation is always 5-fold (choosing
stratified/binned/time-series/random by the data). Local scores are always OOF; submittable competitions are cross-checked with the leaderboard. Per-competition data specifications,
the master experiment table are in each competition's ML-spec report (docs/ml_specs/md/playground-series-*.md); reproduction is each competition's scripts/ (04/05/06) + tree_search/ driver, and the cross-competition results table is benchmark_results/rerun_three_way.csv.

## 5. Results

**(R1) The incremental ladder holds in every competition**: Stages 1→4 of the 15 competitions are all positive or flat, with no competition regressing overall at any stage
(project-brief objective three: cross-competition evidence that performance does not regress).

**(R2) The gain distribution reflects structure**: large gains concentrate on structural-insight competitions (s3e20 +25.7%, s3e5 +19.2%); competitions where
the baseline is already near the ceiling generally have increments <1%. The agent's value lies in "finding and exploiting structure," not indiscriminate stacking.

**(R3) Real leaderboard validation**: s3e16 is the only competition with a real LB comparison; the tree-search version improves over the linear version on the Public and
Private boards **in the same direction on both**, with the CV↔LB gap consistent across the two submissions—external reliability evidence.

**(R4) Cross-season generalization**: the ladder still holds in every one of the five cross-season competitions (five metrics), with magnitudes converging relative to S3.

**(R5) Statistical significance (paired bootstrap)**: the **total improvement of the end-to-end ladder (base blend→
tree-search winner) exceeds the OOF estimation noise and is statistically significant** in all five cross-season competitions; but individual inter-stage steps (especially the tree-search step) fall
within the noise in the two smaller-sample competitions (s4e1, s4e11)—their resolvability increases with the number of validation rows n. Honest conclusion: trust
"the cumulative ladder," and do not over-claim any single 0.0x% step (see docs/statistical_rigor for details).

## 6. Analysis and Cross-Competition Findings

**(F1) Experience-library priors must be checked competition by competition—the same prior can yield opposite conclusions**: the interaction-term prior was
confirmed by controlled checks on s5e10 and s6e2 but rejected on s6e1; the class-imbalance weighting prior has different applicable boundaries across metric competitions. Priors must be transferred together with
their conditions of applicability.

**(F2) The metric decides post-processing, not the appearance of the target**: an integer target pairs with rounding (s3e16), an ordinal one with cut-point optimization
(s3e5); but although the s5e10 target falls on a 0.01 grid, snapping to the grid actually hurts RMSE (squared error wants the conditional mean).

**(F3) The winning shape of the tree search varies with the data**: mega-blend, lineage blend, and single model each have competitions where they won
—a fixed playbook would bet on the wrong shape, whereas the search finds the right one on the spot.

**(F4) Reproducibility discipline**: the cross-competition best solutions all pass the OOF bit-by-bit reproducibility gate; after fixing LightGBM's cross-process
non-determinism, bit-level reproducibility with max|dOOF|=0 is achieved.

**(F5, negative/architectural) External injection: the mechanism can be built, but the marginal value is null**: first, a single-variable comparison (harness_v4
toggling only the injection mode) found that the search trees of off and ext were **bit-for-bit identical** (a constructive 0)—the root cause being that the return of `suggest_priors`
was only written into `tree['priors']` for logging and **no search operator read it**, so experience-library knowledge was actually guided through the driver via
**hand-written seeds** rather than automatic plumbing. Subsequently, **a real injection mechanism was built** (`idea_injection.py`: read
idea_bank → filter and deduplicate → translate into search candidates → tag provenance, plateau-triggered, capped count) and empirically tested at the Stage 4
convergence point over **15 competitions** with cached OOF (rank averaging + a generic stacking translator covering non-AUC competitions):
**external injection never produced a candidate that beat Stage 4—Stage 5 = Stage 4 (flat across all 15 competitions)**; the process caught and removed two false positives
(s3e7's weak baseline, and CV noise already excluded from s3e20's squeeze). **Why null**: (1) Stage 4's OOF weight search is already
near-optimal on the member pool, and ideas that can be injected cleanly (rank/stacking) merely recombine the same members and cannot beat it; (2) stacking is more
flexible → overfits CV noise and does not align with non-squared-error metrics (MAE/SMAPE competitions are clearly worse); (3) useful
domain ideas were long ago hand-written into the search by the author; (4) the signal was already squeezed by the first four stages down to the noise floor; (5) ideas that could bring new signal
(adversarial validation, etc.) require new features/retraining and are data-dependent. **External knowledge is valuable only when it introduces new signal the search missed, and
these mature tabular problems have almost no missed structure left to pick up. This does not negate the value of the Stage 4 tree search** (its gain comes from the search mechanism itself).
Reproducible: `docs/scripts/stage5_sweep.py`; see docs/phase_j_j3_findings for details.

## 7. Limitations

- **CV-only**: most competitions (especially cross-season) are closed, and except for s3e16 only local OOF is available (+ some prior-season LB
  anchors), lacking a current real LB comparison.
- **Statistical scope**: the R5 bootstrap measures only the sampling variance of the OOF estimate, **not** the variance of rerunning the search (changing seed/folds/Optuna)—
  the latter requires retraining and is not done here; so the CI is a lower bound on uncertainty, and individual "significant" results may still be seed-fragile.
- **Second-pillar marginal value null** (F5): the injection mechanism was built and empirically tested, but external injection yields no measurable systematic
  gain on the test competitions; this is a rigorous negative result, not a mechanism defect.
- **Single-agent, single-driver mode**: each competition's driver prior seeds are hand-written, not systematically auto-generated.

## 8. Conclusion and Future Work

Incremental autonomous capabilities bring consistent (no competition regresses) and **end-to-end statistically significant** gains across 15 competitions and five metrics,
with the gain sizes reflecting the data's structural exploitability; the cross-season case still holds but converges. ERA's second pillar (external injection) has had its mechanism built
and empirically tested, with the honest conclusion "mechanism feasible, marginal value null" (F5). Remaining work: a study of search variance under multiple seeds/folds
(upgrading "significant" from an OOF lower bound to rerun robustness), real LB validation on open competitions, broadening the evaluation
beyond tabular problems; and if the second pillar is to be explored further, implementing external ideas of the "require retraining/new features" kind (adversarial validation, frequency encoding),
though the consistent pattern so far points to diminishing returns.

---

**Reproducibility and provenance**: each competition's ML-spec report (docs/ml_specs/md/playground-series-*.md), cross-competition results table
(benchmark_results/rerun_three_way.csv), results brief (docs/PROJECT_BRIEF), statistical rigor (docs/statistical_rigor
+ docs/scripts/bootstrap_ci.py), J-3 findings (docs/phase_j_j3_findings), benchmark fact table
(docs/benchmark_facts.json + docs/scripts/build_benchmark_table.py). Methodology and the five stages are in docs/pipeline_stages_detail.md.
