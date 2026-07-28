# cat-in-the-dat — STATUS

**Final CV (OOF AUC): 0.803272** — tree-search champion, 38-member kitchen-sink blend (LR-family-dominated), 5-fold StratifiedKFold seed 42; honest leave-fold-out weight re-fit 0.803231.
**Submission**: `submission.csv` (200k rows, id/target per sample_submission). OOF-only — not LB-validated (competition closed-era benchmark run).
**Pipeline**: EDA → OHE/TE prep → linear protocol (LR-OHE solo 0.803103 + blend 0.803169) → harness_v3 tree search 60/60 nodes → champion mega-blend from mandatory explore burst.

---

## Run summary (2026-07-28)

### Data / EDA
- 300k train / 200k test, 23 all-categorical features, target rate 0.306, no missing, no dupes.
- High-card noms (nom_5–9, up to 11,981 levels) nearly fully shared train/test (nom_9: only 144 unseen test rows).
- Ordinals' alphabetical order ≈ target-rate order (corr 0.994/0.994/0.971) → ord_3/4/5 mapped alphabetically.
- CV: StratifiedKFold(5, seed=42) — no groups/time structure.

### Linear protocol (Stage 3)
| member | OOF AUC |
|---|---|
| LR on full OHE (16,552 cols), C=0.1 | **0.803103** |
| LR C=0.2 / C=0.05 | 0.802699 / 0.801734 |
| LGBM + fold-safe TE (smoothing 20) | 0.797328 |
| LGBM native categorical | 0.776214 |
| Greedy blend (LR 0.75/0.25) | 0.803169 |

Encoding benchmark confirmed: sparse OHE + LogisticRegression beats GBDT by ~0.005–0.027 AUC.

### Tree search (Stage 4, harness_v3, 60/60 nodes)
- Root lr_c010 digit-verified 0.803103. Full linear pool reseeded as cached-OOF first-gen nodes (afsis lesson).
- Lineages: LRC (C grid 0.07–0.5, best C=0.12 → 0.803169), LRPAIRS (interaction crosses — **hurt everywhere**, best 0.796539), LGBTE (best 0.798865), SEEDBAG, BLEND.
- Champion: explore-burst kitchen-sink mega-blend (node #55) of all 38 solos → **0.803272**; 17 nonzero weights, top: lr_C0.12 0.372, lr_C0.10 0.166, lr_C0.20 0.140.
- evals-to-beat-linear-best: eval #10 (first BLEND seed 0.803226 > 0.803169).
- Honest reporting: dedup_rejections=10; cost-guard coarsened every large blend (k=800→200, ~46–60s, logged); burst sanity gate both long-shots PASS; backtrack log in experiments_tree_v3.json. In-sample weight optimism +0.000041 (0.803272 → 0.803231 leave-fold-out). All conclusions OOF-only, no LB anchor.

### Was tree search worth it here?
Gain over linear blend: +0.000103 (0.803169 → 0.803272), noise-level per fold variance — consistent with the "AUC 混合層只有噪音級差異" prior. The search's value here was negative-result mapping (interactions dead, GBDT ceiling ~0.799, C optimum flat in [0.1, 0.15]), confirming plain OHE+LR is essentially the ceiling for this feature space at this budget.

### Files
- `scripts/`: eda.py, prep.py, members.py, blend.py, make_submission.py
- `tree_search/eval_citd.py`, `tree_search/run_citd_v3.py`, tree at `experiments_tree_v3.json`
- experiments.json: 5 experiments (schema v2)
