# tabular-playground-series-jan-2022 — STATUS

**FINAL: structural Ridge solo (ridge_hol), honest expanding-year CV SMAPE 4.415985 (fold-2017 4.6457 / fold-2018 4.1862, round-to-int).**
**Tree search (harness_v3, 60 nodes) found a 37-member mega-blend at OOF 4.40502, but the LOFO honest gate (weights refit leave-year-out) showed +0.032 optimism — every fitted blend loses to the root solo, so the solo was submitted.**
**submission.csv = full-train (2015-2018) refit of the root config, 2019 preds rounded to int, clipped ≥1; range [100, 2907], mean 414.0.**

## Pipeline (2026-07-28 benchmark run, kaggle-agent skill)

### Stage 1-2: EDA + features
All structural assumptions from the prior tpsjan22 run (knowledge/experience.md) re-confirmed on data:
- Store ratio KaggleRama/KaggleMart constant 1.742 every year; product shares constant across years AND countries; dow effect identical across stores/countries.
- total(country,year)/gdp_pc nearly equal across countries (14.22/14.48/14.87/14.96) — only a common year drift remains → `log_gdp` (World Bank NY.GDP.PCAP.CD, fetched live incl. 2019) + linear `year_c` in the design matrix.
- Holiday effects: per-holiday-name dummies (English names via `holidays` pkg, pooled across FI/NO/SE), shift window −5..+10.

### Stage 3: baselines (expanding-year folds: train≤2016→val 2017, train≤2017→val 2018)
| model | fold-2017 | fold-2018 | pooled |
|---|---|---|---|
| ridge_base (Fourier4, no holidays) | 10.875 | 10.738 | 10.807 |
| ridge_fp (+Fourier8×product) | 7.358 | 6.685 | 7.022 |
| **ridge_hol (+holiday-name dummies)** | **4.646** | **4.186** | **4.415985** |
| lgb_gdp (log(num_sold/gdp_pc) target) | 6.627 | 6.032 | 6.329 |
| grid blend ridge+lgb | — | — | 4.4160 (weight → 1.0 ridge) |

### Stage 4: tree search (references/07_tree_search.md, tree_search/harness_v3.py)
- Driver `tree_search/run_tpsjan22_v3.py`, evaluator `tree_search/eval_tpsjan22.py`, tree at `experiments_tree_v3.json`. 60/60 nodes (44 solo / 16 blend), wall 91 s.
- Root digit-verified at 4.415985. 9 lineages: ALPHA, FOURK, HOLWIN, PERPROD, GDPOFF, DOWPROD, RECENT, LGBDIV, BLEND; explore burst injected EXPL_HUGEK / EXPL_OLS / EXPL_COMBO / EXPL_LGBBIG / EXPL_MEGABLEND.
- **No structural mutation beat the root solo** (best solo child 4.417218, ALPHA lineage). All apparent gains were fitted-weight blends: #17 (7-member, 4.411229), #48 mega-blend (4.406131), #59 champion (37-member, 4.40502).
- evals-to-match-linear-best: 1 (root IS the linear champion). Dedup rejections: 1. Cost-guard fired: 0 times. Burst sanity gate killed EXPL_COMBO (4.690) and EXPL_LGBBIG (6.429). Stop reason: hard budget cap 60/60 (patience path never fired, consistent with all prior runs).
- Backtrack log: 16 events, all "3 consecutive non-improving children" plateaus + 2 sanity-gate kills + 1 all-plateaued reopen (full detail in experiments_tree_v3.json search_state.backtrack_log).

### Honest audit (final gate)
Weights refit leave-year-out (fit on 2017 → score 2018, and vice versa), per the aug-2022 lesson:
| candidate | full-OOF | honest LOFO | optimism |
|---|---|---|---|
| champion #59 (37 members) | 4.405020 | 4.437388 | +0.032 |
| megablend #48 | 4.406131 | 4.432727 | +0.027 |
| blend #17 (7 members) | 4.411229 | 4.431641 | +0.020 |
| seed blend #9 (4 members) | 4.415542 | 4.417713 | +0.002 |
| **root solo** | **4.415985** | **4.415985** | 0 |

→ Root solo wins. The blend "gains" were OOF-noise fits over near-homogeneous Ridge variants (true diversity member LGB is 2 SMAPE points worse and earns ~0 honest weight).

### Caveats
- Scores are OOF-only (expanding-year folds); not LB-validated in this run (competition closed; API submission for pre-2022 playgrounds typically rejected).
- With only 2 validation years, the LOFO honest gate itself is high-variance (fit-on-2017→2018 half was much better than the reverse); the conservative solo choice is the point.
