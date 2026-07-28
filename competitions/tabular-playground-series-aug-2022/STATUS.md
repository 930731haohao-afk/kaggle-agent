# STATUS — tabular-playground-series-aug-2022

**FINAL: equal-weight LR pair (prior5 LR + prior5+measurement_9 LR), per-group rank-pp, OOF AUC 0.591645** (honest LOFO-selection view 0.591204; root solo 0.591248 — all candidates within a ±0.0005 noise band).
Pipeline: EDA → per-code z-scale + Huber m17 imputation + NA flags → LR/LGB baselines → harness_v3 tree search (60 nodes) → honest leave-fold-out audit rejected all fitted-weight blends → unfitted 2-member average submitted.
`submission.csv` = `submissions/final_root_plus_node17_equalweight.csv` (20,775 rows, per-test-group rank-normalized). OOF-only, not LB-validated.

---

## Run report (2026-07-28)

### Setup
- 26,570 train / 20,775 test. Binary failure (21.3%), metric ROC-AUC.
- **Train product codes A–E, test F–I: fully disjoint** → CV = leave-one-group-out GroupKFold on product_code (5 folds). Attributes constant within code → unusable.
- Decision metric includes post-processing: pooled AUC after **per-group rank normalization** (prior-run validated +0.0007; direction confirmed across all configs this run).

### Features (scripts/prep.py)
- Per-code z-scaling of loading + measurement_0..17 (codes disjoint, no target used → no leakage).
- measurement_17 imputed per code by HuberRegressor on its top-4 correlated measurements; other NaNs per-code median.
- NA indicator flags (m3_na: fail 0.160 when missing; m5_na: 0.254; base 0.213).

### Linear stage (experiments #1–#3)
- LR C=0.01 on prior-run 5-feature set [loading, measurement_17, m3_na, m5_na, measurement_2]: **0.591248** (reproduces prior-run 0.59131).
- Shallow LGB (35 feats): 0.58174 — LR > LGB confirmed again on disjoint groups.
- LR+LGB weight search: degenerate w_lr=1.0 (prior "blend is noise on AUC" confirmed).

### Tree search (harness_v3, tree_search/run_aug22_v3.py, experiments_tree_v3.json)
- 60 nodes evaluated, wall ~7 min. Root digit-verified 0.591248. Full linear pool reseeded (cached OOF).
- Lineages: LRC (C sweep + boundary push), LRFEAT (feature-set search), LGBREG, SEEDBAG, BLEND + mandatory explore burst.
- Best solo: #17 LR prior5+measurement_9 = 0.591447. Pre-burst blend #23 = 0.591648. Kitchen-sink mega-blend #52 (38 members) = **0.591955 in-sample**.
- evals-to-beat-linear-best: eval 7 (seed blend #6, 0.591249, marginal). Dedup rejections: 0. Cost guard: never fired. Burst sanity gate: fired once (EXPL_DEEPLGB #53, 0.551 — correctly killed). Backtrack log in experiments_tree_v3.json.

### Honest audit (experiments #4–#6) — the decisive step
- Mega-blend #52 leave-fold-out weight refit: **0.590944 < root 0.591248** → its +0.0007 was pure weight-fitting optimism (+0.001011 measured). Rejected.
- Loose rule-based 22-member equal-weight family: 0.590510. Rejected (weak members drag unfitted average).
- LOFO-selection over 5 candidates: picks root+#17 in 4/5 folds, honest score 0.591204 ≈ root. No candidate robustly beats the root solo.
- Final: unfitted equal-weight [root, #17] — 0.591645 in-sample, downside bounded at noise level, variance-reduction rationale (seed-bag analog).

### Caveats
- OOF-only; no LB anchor (competition closed for this benchmark run).
- All gains beyond the root LR are within CV noise (fold AUCs range 0.586–0.600); the honest conclusion is that this competition's ceiling with these features is ~0.591–0.592 and member/weight selection cannot be trusted below ±0.0005.
