# Three-agent benchmark: score comparison and analysis

Generated 2026-07-24 20:09; NVIDIA gap-fill for s6e7 / spaceship-titanic / home-data added 2026-07-27 (now 17/17 three-way). Scope: competitions my agent ran with the July pipeline. All 17 rows are three-way. `local` = each agent's own validation score; `pub`/`priv` = Kaggle public/private leaderboard (late submissions, huangweihaohuang account); `PR` = percent of public-leaderboard teams at or below our entry (higher is better). s3e20 has no LB row (late submission disabled by Kaggle). home-data 'mine local' is in raw-dollar units (not comparable to the log-RMSLE columns) — use its LB columns.

## Score table

| Comp | Metric | Mine local | NV local | AIDE local | Mine pub / priv (PR) | NV pub / priv (PR) | AIDE pub / priv (PR) | Local winner (naive) | LB winner (naive) | **Rank (private, best first)** | **Separated pairs** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s3e1 ⚠ | RMSE (min) | 0.556329 | 0.5086 | 0.54952 | 0.56083 / 0.55987 (63.3) | 0.55332 / 0.5555 (93.2) | 0.55863 / 0.55343 (68.6) | nvidia | aide | AIDE, NV, mine | AIDE>mine; NV>mine |
| s3e3 | AUC (max) | 0.845051 | 0.8758 | 0.8463 | 0.89262 / 0.87303 (29.4) | 0.93526 / 0.89537 (66.8) | 0.88484 / 0.86863 (26.4) | nvidia | nvidia | NV, mine, AIDE | all three |
| s3e5 ⚠ | QWK (max) | 0.56769 | 0.5478 | 0.57654 | 0.57921 / 0.59743 (78.8) | 0.57515 / 0.56974 (75.7) | 0.62064 / 0.59073 (92.0) | aide | mine | mine, AIDE, NV | mine>NV |
| s3e7 | AUC (max) | 0.900455 | 0.9014 | 0.89963 | 0.91116 / 0.90257 (74.3) | 0.9123 / 0.90458 (78.1) | 0.90934 / 0.90109 (67.1) | nvidia | nvidia | NV, mine, AIDE | all three |
| s3e9 | RMSE (min) | 12.070034 | 12.026 | 12.0526 | 11.84951 / 12.29871 (50.6) | 11.83399 / 12.22913 (59.1) | 11.86657 / 12.30667 (44.5) | nvidia | nvidia | NV, mine, AIDE | NV>mine; NV>AIDE |
| s3e11 | RMSLE (min) | 0.29528 | 0.2956 | 0.2941 | 0.29534 / 0.29597 (67.4) | 0.29559 / 0.29624 (66.6) | 0.29389 / 0.29445 (73.3) | aide | aide | AIDE, mine, NV | all three |
| s3e14 | MAE (min) | 340.35572 | 337.28 | 339.8059 | 341.26711 / 332.4356 (76.1) | 338.36524 / 330.71616 (89.2) | 342.44918 / 332.31556 (71.2) | nvidia | nvidia | NV, AIDE, mine | NV>mine |
| s3e16 | MAE (min) | 1.33563 | 1.3387 | 1.3369 | 1.34315 / 1.33859 (84.2) | 1.33566 / 1.34001 (96.4) | 1.3392 / 1.34224 (91.4) | mine | mine | mine, NV, AIDE | NV>AIDE |
| s3e19 ⚠ | SMAPE (min) | 9.75707 | 13.72 | 11.1944 | 50.22497 / 50.45537 (28.5) | 47.52145 / 48.26558 (41.4) | 48.99617 / 49.61325 (37.7) | mine | nvidia | NV, AIDE, mine | all three |
| s3e20 | RMSE (min) | 21.14870 | 19.648 | 21.17676 | — | — | — | nvidia | — | — (no private LB) | — |
| s4e11 | Accuracy (max) | 0.940235 | 0.9399 | 0.94055 | 0.94125 / 0.94043 (50.7) | 0.94136 / 0.94076 (52.8) | 0.94184 / 0.94056 (62.1) | aide | nvidia | NV, AIDE, mine | NV>mine |
| s5e10 | RMSE (min) | 0.055968 | 0.05609 | 0.055994 | 0.05551 / 0.05576 (85.7) | 0.05562 / 0.05588 (59.4) | 0.05555 / 0.05579 (78.9) | mine | mine | mine, AIDE, NV | all three |
| s6e1 | R2 (max) | 0.787183 | 0.7863 | 0.7868 | 8.69338 / 8.71733 (75.2) | 8.7048 / 8.73239 (69.5) | 8.69449 / 8.72191 (74.8) | mine | mine | mine, AIDE, NV | all three |
| s6e2 | AUC (max) | 0.955529 | 0.9554 | 0.95551 | 0.95367 / 0.95516 (78.8) | 0.95348 / 0.95508 (63.0) | 0.95364 / 0.95516 (76.5) | mine | mine | mine, AIDE, NV | none |
| s6e7 ⚠ | balanced_accuracy (max) | 0.950049 | 0.95007 | 0.94981 | 0.94985 / — (66.1) | 0.95015 / — | 0.94982 / — (64.8) | mine | nvidia (pub only) | — (no private LB) | — |
| spaceship-titanic | accuracy (max) | 0.814333 | 0.80202 | 0.8156 | 0.80383 / — (61.4) | 0.79728 / — | 0.805 / — (68.5) | aide | aide (pub only) | — (no private LB) | — |
| home-data-for-ml-course | rmsle (min) | 14321.293122 | 0.10595 | 0.11031 | 12907.49283 / — (98.2) | 13683.25686 / — | 12887.15456 / — (98.3) | aide | aide (pub only) | — (no private LB) | — |

> Leaderboard cells read **public / private (percentile)**. Public and private are
> disjoint random subsets of the same test set scored on the same submission, so the pair
> also shows how far a score travels between samples — the basis of the paired test below.
> ⚠ = AIDE's run was fed another agent's intermediates; that row's AIDE figures are void.
> **Rank** orders the three agents by private score, best first — a plain ordering, with no
> claim that neighbours differ. **Separated pairs** lists only the orderings that survive the
> paired test below; anything absent is unresolved at this resolution. The two are kept apart
> because the separation relation is not transitive — on s3e14 and s4e11 neither neighbouring
> pair separates, yet the first and third do, which no single ranking chain can express.
> Three competitions have no private leaderboard: s3e20 (late submission closed) and the two
> getting-started comps (spaceship-titanic, home-data), which are rolling public-only.

**LB wins — uncontaminated set (12 comps; excludes s3e1/s3e5/s3e19/s6e7 where the harness leaked inputs into AIDE's workspace, s3e20 for no LB):** NVIDIA 6 · mine 3 · AIDE 3
**LB wins — all 17 (inherits the contamination defect, see bottom section):** NVIDIA 8 · mine 4 · AIDE 4
**Local-score wins (16 comps; home-data excluded — unit mismatch):** NVIDIA 7 · mine 5 · AIDE 4
**Head-to-head mine vs AIDE on LB:** mine 8 · AIDE 5 (of 13 three-way-original rows)

> Tally updated 2026-07-27 after the NVIDIA gap-fill. The three added comps went NVIDIA 1 (s6e7) / AIDE 2 (spaceship-titanic, home-data), which moved NVIDIA from a tie into a clear LB lead.

## Agent profiles — why the scores differ

### My agent (staged pipeline + tree search)
- **Approach:** 5-stage pipeline (ingest → EDA → features → modeling → submission) driven by Claude Code, plus a tree-search layer (~24–29 nodes/comp in the July runs) with an experience library and priors.
- **Budget/time:** hours per competition (July overnight runs); comparable node budget to AIDE@20.
- **Rigor:** deterministic LightGBM settings, OOF reproduction gates, no-fabrication rules. These governance constraints cost some raw score but make the reported numbers trustworthy — reflected in smaller local→LB drops than AIDE's.

### NVIDIA agent (kernel reproduction)
- **Approach:** finds and reproduces the best public kernel for each competition (repro_type mostly 'verbatim') — effectively distilled human-expert solutions, not independent problem solving.
- **Budget/time:** minutes per competition once the kernel is found.
- **Profile:** highest peaks (top 3.6% on s3e16) because it inherits community-tuned solutions, but high variance (bottom 8% on s4e1) — quality depends entirely on which kernel exists and still runs.

### AIDE (aideml, 20-step, Claude Sonnet via subscription shim)
- **Approach:** greedy solution-tree search; every node is a full write-run-score cycle. 5 drafts then improve/debug; no EDA stage, no validation governance; optimizes exactly one number.
- **Budget/time:** 20 nodes/comp (its default), 13 min–3.2 h per comp (median ≈ 1 h under load); ~1,000 LLM calls ≈ 5.3M tokens for the whole study — zero API cost via the subscription shim.
- **Profile:** strong on smooth standard tabular tasks; its self-reported bests are optimistically biased (max over ~20 noisy validation estimates + greedy selection), so local wins can evaporate on the private LB — measured here as the local-vs-LB winner gap.

## Key explanations for observed differences

1. **Selection bias:** AIDE reports max-of-20 noisy CV estimates; my agent reports gated, reproduced scores. Same true skill would still show AIDE ahead locally and even on the public LB, with regression on private — exactly the observed pattern.
2. **Budget scaling:** AIDE@5 lost 14:2 to my agent locally; AIDE@20 closed most gaps and takes comps outright (e.g. s3e1 private LB). Node budget, not scaffold cleverness, drove most of that swing.
3. **Knowledge source:** NVIDIA's peaks come from human community solutions; AIDE and my agent search from scratch. On comps with strong public kernels NVIDIA is hard to beat; where kernels are weak (s4e1) it collapses.
4. **Validation quality is a differentiator:** s3e19 exposed broken local validation in both my and NVIDIA's runs (CV 4.3/13.7 → LB 50/48). Only LB submission catches this class of error — motivating LB verification as a standard step.

---

## ⚠ Validity defect found 2026-07-27 — harness fed AIDE another agent's intermediates (4 competitions)

**This is a defect in my benchmark harness, not a flaw in AIDE.** `run_comp.py` passes
`copy_data=True` with `data_dir=competitions/<comp>/data`, which copies that entire directory
into the workspace's `input/`. But that directory is **my own agent's working directory**: on
2026-07-03 my pipeline cached processed feature tables, tuned-parameter JSONs and OOF arrays
there alongside the official download (file mtimes separate cleanly — official files
2025-12-31, my intermediates 2026-07-03). AIDE was then handed the merged directory and told
it was the competition data. Nothing in AIDE's prompt or data preview distinguishes an
official file from a cached one, and auxiliary files are common in real competitions, so
reading them was reasonable behaviour on AIDE's part. The fault is entirely upstream.

Audit of all 17 input trees found foreign files in 6 competitions; in **4 of them AIDE's
champion solution actually reads them**:

| Comp | Foreign files AIDE's champion loads | What it inherited |
|---|---|---|
| s3e1 | `train_processed_v2.csv`, `test_processed_v2.csv`, `round1_best_lgb_params.json` | 28-column engineered feature set (raw data has 10) **plus my agent's Optuna-tuned LightGBM hyperparameters** |
| s3e5 | `train_processed.csv`, `test_processed.csv` | 23-column engineered feature set (raw has 13) |
| s3e19 | `train_processed.csv`, `test_processed.csv` | 24-column engineered feature set (raw has 6) |

| s6e7 | 7 OOF archives loaded as `./input/{name}.npz` (`oof_cat`, `oof_et`, `oof_lgbm`, `oof_lgbm_seed2024`, `oof_lgbm_tuned`, `oof_mlp`, `oof_xgb`) | **the entire model.** AIDE never trained a base learner from raw features — all 20 steps were stacking on another agent's out-of-fold predictions |

Clean (foreign files present in `input/` but never opened by the champion): s3e14
(`oof_v4.npz`), s3e20.

Audit method note: s6e7 was initially scored clean by a filename grep, because its champion
builds the path dynamically (`f"./input/{name}.npz"` over a list of stems) rather than
naming files literally. The audit was redone by extracting every `./input/...` path each
champion opens and subtracting the three official filenames — that is the check that holds.

**Consequence.** For s3e1, s3e5, s3e19 and s6e7 the AIDE column does not measure AIDE. Feature
engineering is the bulk of the work on these tabular episodes, and on s3e1 the
hyperparameter search was inherited outright, so those four AIDE scores — including the
s3e1 private-LB result — must be treated as void, not merely optimistic. s6e7 is the most
severe: with no base learner of its own, that run measures how well AIDE reweights another
agent's ensemble. The honest three-way count is over the 12 uncontaminated episodes until
these are re-run from raw data.

The fix is already in place for future runs: benchmark runs now read an isolated
competition root (`bench-comps/`, via `AIDE_COMP_ROOT`) containing only `config.yaml` and a
symlink to the official data, so no agent can see another's intermediates.

---

## Cross-cutting AIDE failure modes (from the 17 per-competition reports, 2026-07-27)

Writing every AIDE run up against its own journal surfaced failure patterns that are
invisible in a score table. These are properties of the agent, not of any one competition.

**1. It diagnoses correctly and then ignores the diagnosis.** The clearest case is s3e16:
AIDE ran nested-CV checks at steps 16 and 18, measured its calibration at 1.3396/1.3412,
wrote in its own report that the in-sample figure was optimistic — and still selected the
in-sample node. Private LB (1.34224) vindicated the estimate it discarded. Same shape on
s5e10, where steps 13 and 14 re-proposed the same over-budget CatBoost search back to back
*after* the agent had written a correct post-mortem on the cost, and on s3e3, where it
identified its own calibration leak at step 18 and kept the leaky variant because the honest
fixes scored lower.

**2. Timeouts dominate the compute bill and the draft policy never learns.** Single steps
consumed 70 %, 49 % and 90 % of total execution time on s3e1/s3e7/s3e9; 52.6 % on s5e10;
56.3 % on s6e1; 45.3 % on s6e2; 59.0 % on home-data. Repeated `ModuleNotFoundError` on
PyTorch burned 7 steps in batch 1 alone — the same unavailable import re-proposed across
steps because nothing feeds the failure back into drafting.

**3. No verification gate anywhere.** Every champion metric is one execution's printout.
There is no replay, no rebuild, no seed check — so a lucky split is indistinguishable from a
real gain, which is precisely how the s3e16 and s3e19 selections went wrong (s3e19's private
score is 4.4x its local estimate, the whole improvement having been measured on a single
2021 holdout).

**4. No EDA phase, and its self-reports inherit the gap.** AIDE's auto-generated `report.md`
files contain factual errors about their own data: s3e11 calls a skew-0.019 target
"right-skewed"; s3e14 justifies a log transform for "right-skew" on a target with skew
-0.29; s3e16 describes "~8" calibration boundaries where its code fits 28; s6e1 claims
"fourteen" iterations against 20 executed. **Consequence for this benchmark: AIDE's own
report.md cannot be used as a numeric source.** All figures in the per-competition reports
come from `facts_aide.json`, extracted deterministically from `journal.json` by
`aideml/collect_aide_facts.py`, and conflicts with AIDE's prose are flagged in place.

**5. When it wins, the win is often one node.** s3e11's entire advantage came from a
bug-fix node that incidentally introduced native categorical handling (86 % of the run's
total improvement); s3e5's from `OptimizedRounder` at step 2 (75.29 %); s3e16's from
"round predictions to integers" in a 22-second node (77 %); s6e7's champion is itself a
post-mortem node that deleted the agent's own polynomial expansion. The search rarely
compounds — it finds one lever and then plateaus, frequently re-scoring an identical value
for many consecutive steps.

### Full audit scope (completed 2026-07-27)

The contamination finding above came from auditing the 17 benchmark competitions. That scope
was too narrow, so every run by every agent was then re-audited against Kaggle's own file
manifests (`bench-comps/manifests/`), not against filename guesses:

| Agent | Runs audited | Unofficial reads | Verdict |
|---|---|---|---|
| AIDE | 42 | 5 | 4 real (s3e1, s3e5, s3e19, s6e7); afsis cleared |
| NVIDIA | 19 scripts / 18 comps | 1 | the discarded `contaminated-leak` run only — already excluded |
| my-agent | 45 | 0 | never reads another lane's tree |

Two candidates were cleared rather than counted, because a manifest mismatch is not by itself
evidence of contamination:

- **afsis-soil-properties** — AIDE reads `training.csv` and `sorted_test.csv`, which are absent
  from the manifest (it lists `train.zip` / `test.zip`). `unzip -l` confirms those are exactly
  the files inside the official archives. Official content, merely unzipped.
- **NVIDIA's `test_pred_fold{fold}.npy`** — the lane's own intermediate, and it belongs to the
  us-patent run already discarded for a prompt leak (the run whose stray process later had to
  be killed). Not part of any reported score.

So the four void AIDE cells are the complete set, confirmed across all 106 runs rather than
inferred from the benchmark subset.

---

## Correction 2026-07-27 — three "Mine local" figures were diagnostic runs

The aggregator that built this table took each competition's best experiment by score. My
agent's `experiments.json` also records **deliberately-leaky diagnostics** — runs whose
purpose is to isolate *why* a score moved, explicitly annotated "not used for submission" —
and on three competitions that diagnostic was the best-scoring entry, so it was reported as
the agent's local result.

| Comp | was reported | honest best | what the diagnostic was |
|---|---|---|---|
| s3e19 | 4.28142 | **9.75707** | shuffled 5-fold KFold on a time series, run to test whether the gap came from the CV scheme or the features |
| s3e5 | 0.57066 | **0.56769** | (QWK, max) |
| s3e20 | 21.0589 | **21.1487** | |

s3e19 is the visible one: SMAPE 4.28 local against 50.46 on the leaderboard is a 11.8x gap,
where NVIDIA and AIDE sit at 3.5x and 4.4x. Under the honest time-based number (9.757) the
ratio is 5.2x, in line with the other two — the remaining gap is the genuine difficulty of
extrapolating a year forward, which hit all three agents.

Note what this is *not*: the agent did not cheat itself. It ran time-based CV for every
submission, ran the shuffled-KFold variant purely as a diagnostic, and labelled it as such
in its own log. The defect is in the reporting layer, which had no notion of an experiment
being excluded from selection. Local winners were recomputed after the fix; none changed.

---

## Tie-aware standings — paired test on the public/private split

Counting a win whenever one score is numerically higher treats 0.00003 like 0.03, so every
comparison needs an error scale. The scale is taken from the data, not assumed.

**A first attempt was wrong and is worth recording.** I used each agent's own public-private
gap as the threshold. That gap is dominated by a component *shared* across agents — which
rows happened to land in each subset makes everyone's score move together. On s5e10 the three
lanes drifted +0.00025, +0.00026 and +0.00024: a spread of 0.00002 around a common shift of
0.00025. Using the shift as the threshold inflates the noise by an order of magnitude, and
duly returned eight three-way ties out of ten — an artefact of an unpaired scale applied to a
paired question.

**The evidence that the drift is shared.** Each agent's score moves from public to
private because the two subsets differ in difficulty — and they move *together*:

| Comp | Metric | mine drift | NVIDIA drift | AIDE drift | shared shift | spread between agents |
|---|---|---|---|---|---|---|
| s3e1 | RMSE | -0.00096 | +0.00218 | -0.00520 | -0.00133 | 0.00738 |
| s3e3 | AUC | -0.01959 | -0.03989 | -0.01621 | -0.02523 | 0.02368 |
| s3e5 | QWK | +0.01822 | -0.00541 | -0.02991 | -0.00570 | 0.04813 |
| s3e7 | AUC | -0.00859 | -0.00772 | -0.00825 | -0.00819 | 0.00087 ✅ |
| s3e9 | RMSE | +0.44920 | +0.39514 | +0.44010 | +0.42815 | 0.05406 ✅ |
| s3e11 | RMSLE | +0.00063 | +0.00065 | +0.00056 | +0.00061 | 0.00009 ✅ |
| s3e14 | MAE | -8.83151 | -7.64908 | -10.13362 | -8.87140 | 2.48454 ✅ |
| s3e16 | MAE | -0.00456 | +0.00435 | +0.00304 | +0.00094 | 0.00891 |
| s3e19 | SMAPE | +0.23040 | +0.74413 | +0.61708 | +0.53054 | 0.51373 |
| s4e11 | Accuracy | -0.00082 | -0.00060 | -0.00128 | -0.00090 | 0.00068 |
| s5e10 | RMSE | +0.00025 | +0.00026 | +0.00024 | +0.00025 | 0.00002 ✅ |
| s6e1 | R2 | +0.02395 | +0.02759 | +0.02742 | +0.02632 | 0.00364 ✅ |
| s6e2 | AUC | +0.00149 | +0.00160 | +0.00152 | +0.00154 | 0.00011 ✅ |

✅ marks competitions where the spread between agents is under half the shared shift, i.e.
the drift is almost entirely common. s5e10 is the clearest: the three lanes move +0.00025,
+0.00026 and +0.00024 — a shared shift of 0.00025 with only 0.00002 of disagreement. Judging
a 0.00003 gap against a 0.00025 threshold declares a tie; judging it against the 0.00002 that
actually separates the agents does not. s3e11, s6e2, s3e7, s6e1, s3e9 and s3e14 show the same
pattern (s4e11 at 0.00068 spread against a 0.00090 shift is borderline). The three competitions with disagreeing signs (s3e1, s3e5, s3e16) are the ones where
no pairing survives the test.

**The correct scale is how much the *gap between two agents* moves between the two subsets.**
For a pair (A, B) let `d_pub` and `d_priv` be their score differences on each subset. The
ordering is called decisive only when both hold:

1. `sign(d_priv) == sign(d_pub)` — the ordering reproduces on a disjoint sample; and
2. `|d_priv| > |d_priv − d_pub|` — the gap exceeds how far the gap itself travels.

The shared drift cancels in `d`, so what remains is the noise that actually threatens the
comparison. Applied to the ten uncontaminated competitions with a leaderboard (three pairwise
contests each, thirty in total):

| Agent | Won | Lost | Tied | Win rate excluding ties |
|---|---|---|---|---|
| NVIDIA | 9 | 6 | 5 | 60% |
| mine | 7 | 6 | 7 | 54% |
| AIDE | 4 | 8 | 8 | 33% |

Per-competition pairings:

| Comp | mine vs NVIDIA | mine vs AIDE | NVIDIA vs AIDE |
|---|---|---|---|
| s3e3 | NVIDIA | mine | NVIDIA |
| s3e7 | NVIDIA | mine | NVIDIA |
| s3e9 | NVIDIA | ≈ | NVIDIA |
| s3e11 | mine | AIDE | AIDE |
| s3e14 | NVIDIA | ≈ | tie |
| s3e16 | ≈ | tie | NVIDIA |
| s4e11 | NVIDIA | ≈ | tie |
| s5e10 | mine | mine | AIDE |
| s6e1 | mine | mine | AIDE |
| s6e2 | ≈ | tie | ≈ |

Twenty of thirty contests are decisive, so the field does separate — the earlier eight-tie
result was measurement error, not a property of the agents. NVIDIA leads, my agent is close
behind, AIDE trails; s6e2 is inseparable at every pairing (`AIDE≈mine≈NV`), and there mine and AIDE also happen to score identically at 0.95516.

Note on s6e1: its `direction` column reads `max` (local R²) while its leaderboard columns are
a minimise-direction error metric on a different scale. The pairing above uses minimise for
the leaderboard, matching the recorded `lb_winner`.

Two caveats. This captures test-set sampling noise but **not** agent run-to-run variance,
which for AIDE is likely larger — its code is re-sampled from an LLM at every step. Schedule
item 8 measures that. And decisiveness is not magnitude: NVIDIA's edge on s3e7 is 0.00201 in
AUC, real but worth almost nothing in practice.
---

## Finding — vote count is the wrong selection signal, and it is NVIDIA's binding constraint

NVIDIA's method is "reproduce the strongest public kernel", with strength read off
**vote count**. Votes measure popularity and pedagogy, not score, and the two come apart
exactly where it costs the most.

| Comp | Kernel reproduced | Votes | Result |
|---|---|---|---|
| spaceship-titanic | `gusthema/spaceship-titanic-with-tfdf` | 7859 | 0.79728 — last of three; a getting-started TFDF tutorial |
| s3e19 | `tumpanjawat/s3e19-course-eda-fe-lightgbm` | — | 48.27 against a 4.67 leaderboard top; an EDA/course notebook |
| s3e3 | `chunweishen/ps-s03e03-ensembling` | 89 | best of three; a low-vote but genuinely competitive ensemble |

The 89-vote kernel beat the field; the 7859-vote kernel lost to both other agents. Vote count
inverted the ranking.

**This is not an inability to use external data.** NVIDIA does bring the original source
dataset when its chosen kernel does — s3e3 (IBM HR Attrition), s3e7, s3e14 and s3e16 all load
the parent dataset the synthetic competition was generated from. On s3e19 it did not, because
the kernel it selected states in its own header that it uses none. The capability is present;
the selection step wastes it.

s3e19 also shows the second failure in the same step: **reproduction without verification.**
That kernel's header claims "2nd-place paddykb uses same idea", and NVIDIA reproduced the
skeleton faithfully — model the daily total, then disaggregate by country/store/product
shares. But paddykb scored 5.19 and this reproduction scored 48.27. The structure was copied;
the year-level scaling that actually decides the competition was not, and nothing in the
method compares the reproduction's score against the source's claimed standing to catch it.

The leaderboard shows why that gap is so expensive here. s3e19 is bimodal, not a gradient:
265 of 1174 teams scored under 10, 570 landed between 40 and 60, and the 10th-to-25th
percentile jumps from 6.1 to 22.8. All three agents sit in the lower cluster (48.3 / 49.6 /
50.5). The paired test does separate them there, but it is separating degrees of the same
miss — that ranking should not be read as a capability difference.

**For agent design this is the more useful framing.** "Pick by votes" is a concrete mechanism
that could be replaced — by the poster's competition rank, by the score claimed in the kernel,
by cross-checking a reproduction against the source's reported number — whereas "lacks
judgement" is not actionable. AIDE and my agent have the opposite problem: neither ever
considers external data at all, so on s3e19 they were never in a position to select badly.

**This weakness is documented, not fixed — deliberately.** The five later NVIDIA lanes
(phase 9a) select by vote count exactly as the original seventeen did. NVIDIA and AIDE serve
this benchmark as fixed yardsticks for the user's own agent; improving a yardstick mid-study
would make the before/after numbers incomparable and quietly turn the benchmark into an
optimisation of the reference methods. The skill itself ships a score-based selection tool
(`fetch_top_kernel_scores.py`) that these runs do not use — so the gap is in the executed
convention, and it stays constant across all 22 NVIDIA lanes by design.

---

## s3e19 — AIDE's clean re-run scored *better* locally and no better on the leaderboard

The four contaminated competitions were re-run from the manifest-verified clean root. Three
behaved as expected: without another agent's engineered features, AIDE's local score got
slightly worse. s3e19 went the other way, and the reason is worth recording.

| | contaminated run | clean re-run |
|---|---|---|
| local SMAPE | 11.1944 | **4.3718** |
| private LB | 49.61325 | 48.34131 |
| local-to-LB ratio | 4.4× | **11.1×** |

Removing the contamination made the local number look 2.6× better while the leaderboard moved
by 1.3 points. The improvement was entirely in the measurement.

**Where it came from.** The step-by-step trajectory jumps in a single move:

```
step 0-10 : 18.42 … 22.20 … 18.42     (validation is time-aware)
step 11   : 4.41                       (<- KFold(shuffle=True) introduced)
step 12-19: 4.40, 4.37, 4.40, 4.49     (nine steps polishing the leaked metric)
```

Step 11's plan is a bug-fix — LightGBM's newer API rejected the custom SMAPE objective the
previous step used — and while rewriting the training loop it also replaced the validation
split with `KFold(n_splits=5, shuffle=True, random_state=42)`. The SMAPE definition is
byte-identical before and after, so the 4× is entirely the split.

**Confirmed by direct measurement**, same features and same model, changing only the split:

| Split | SMAPE |
|---|---|
| `KFold(shuffle=True)` | 4.56 |
| train 2017-2020 → validate 2021 | 20.41 |

A 4.5× gap. The competition's test set is calendar year 2022, so shuffling rows lets the model
see a series' neighbouring days while predicting the days between them — interpolation inside
a series it has already memorised, not the forecast the competition scores. (An earlier
hypothesis that the `(country, product, store)` target encoding carried the leak was tested
and rejected: that feature alone scores 14.43 shuffled versus 14.85 time-split.)

**What this says about the agent.** AIDE has no notion of which validation scheme the task
demands. The switch arrived as collateral in an unrelated bug-fix, its own metric improved
fourfold, and nothing in the search questioned a jump that large — the remaining nine steps
tuned hyperparameters against the leaked number. This is the same failure the s3e16 report
documents from the other direction: there AIDE *measured* its optimism with nested CV, wrote
it down, and selected the optimistic node anyway. The agent can produce the diagnostic; it has
no mechanism that lets a diagnostic overrule a score.

Worth noting my own agent hit the identical trap on this competition and handled it
differently: its experiments.json contains a shuffled-KFold run at SMAPE 4.2814 explicitly
labelled "DIAGNOSTIC: not used for submission", run to isolate whether a scheme change or the
features explained a gap. Same leak, deliberately quarantined. It was our reporting layer, not
the agent, that later mistook that number for a result.
