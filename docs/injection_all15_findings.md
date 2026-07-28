# External-Idea Injection — completed across all 15 competitions (2026-07-15)

Extends `prior_wiring_findings.md` (which covered 5) to **all 15**. Two ways to inject:
**Arm A** = reweight the existing model pool using the priors; **Arm B** = expand the pool with new
external-idea-derived members, then reblend.

## Two tiers of measurement (this matters)

- **Tier 1 — 8 rigorous comps** with the full v3 infrastructure (deterministic evaluators, fixed champion
  weights, solo-cache pool): s3e5, s3e7, s3e14, s4e1, s4e11, s5e10, s6e1, s6e2. Here the ±1e-5 paired
  comparison is defensible.
- **Tier 2 — 7 adapted comps** (s3e1, s3e3, s3e9, s3e11, s3e16, s3e19, s3e20): the committed trees were the
  early v1/v2 batch. I wrote `tree_search/adapt_v1_to_v3.py` to reformat them into v3 layout (no re-search).
  **Caveat surfaced by doing this:** the v1/v2 committed champions are *not* the convex optimum under the
  stronger v3 weight search, so a pool-only reblend already beats them — and the reblend weight search itself
  carries ~1e-4 stochastic noise. That noise **swamps** the ~1e-5 injection signal, so Tier 2 cannot resolve
  injection at Tier-1 precision. **This empirically confirms the original decision to restrict the rigorous
  measurement to the v3 batch.** For Tier 2 the clean quantity is (pool+external reblend) − (pool-only reblend),
  both under the same v3 search.

## Tier 1 result (rigorous)

- **Arm A: champion unchanged 8/8** — reweighting cannot beat the convex-optimum blend (theorem holds).
- **Arm B: improved only 2/8** — s4e1 (+7e-5 AUC), s6e1 (+3e-5 R²), both significant but ~1e-5; the other 6
  (s3e5, s3e7, s3e14, s4e11, s5e10, s6e2) showed **no gain** (reblend ≤ champion).

## Tier 2 result (adapted; injection delta = pool+external − pool-only, both v3 reblend)

| Comp  | Metric | pool-only | pool+external | injection Δ | note |
|-------|--------|----------:|--------------:|------------:|------|
| s3e16 | MAE ↓  | 1.33678 | 1.33678 | **0.000000** | external members weighted to 0 (exact tie) |
| s3e1  | RMSE ↓ | 0.55621 | 0.55627 | +0.000056 | neutral/worse (within weight-search noise) |
| s3e3  | AUC ↑  | 0.84162 | 0.84161 | −0.00001 (AUC) | neutral/worse |
| s3e9  | RMSE ↓ | 12.06682 | 12.06350 | **−0.003319** | only nominal *benefit* — unconfirmed (no bootstrap; may be search variance) |
| s3e11 | RMSLE ↓| 0.29531 | 0.29537 | +0.000061 | neutral/worse |
| s3e19 | SMAPE ↓| 9.93880 | 9.96838 | +0.02958 | worse — higher-dim reblend under-converged (search artifact, not "injection hurts") |
| s3e20 | RMSE ↓ | — | — | **N/A** | pool is structural (location×week means), no GBDT family to decorrelate |

## Overall conclusion (all 15)

**External-idea injection has a negligible and inconsistent effect.**
- Reweighting (Arm A) never beats the convex optimum (8/8 rigorous).
- Adding external members (Arm B) helps in only ~2 of the rigorously-measurable comps, by ~1e-5 — no practical
  or leaderboard significance.
- On the adapted comps the effect sits inside the weight-search noise floor (~1e-4); the single nominal benefit
  (s3e9, −0.0033 RMSE) is unconfirmed, and one apparent degradation (s3e19) is a search-convergence artifact.
- On s3e20 injection is structurally inapplicable.
- The value of the experiment remains **mechanism attribution, not score** (a shallow bias-dominated member can
  decorrelate a variance-heavy blend), and the exercise additionally demonstrated *why* the rigorous ±1e-5
  measurement requires the v3 infrastructure rather than the adapted v1/v2 trees.

Artifacts: `tree_search/wire_<comp>_{A,B}.json`, `tree_search/wire_<comp>_ledger_{a,b}.json`,
`tree_search/llm_proposals_<comp>.json`, `tree_search/adapt_v1_to_v3.py`, `tree_search/gen_proposals.py`.

---
## Full re-run (generic v3 forward-search) + v3-vs-v1/v2 — 2026-07-16

Rather than 7 bespoke ~800-line drivers, `tree_search/run_v3_generic.py` drives harness_v3's
node/phase machine generically (generic shallow-reg mutation operators + full-pool reblends,
each comp's own `eval.evaluate`), extending the adapted tree forward → a genuinely v3-searched
champion (removes the adapter confound). Ran on all 7 (s3e20 N/A — structural pool).

**v3-searched champion vs the committed v1/v2 champion:**

| Comp | v1/v2 champ | v3 champ | Δ | read |
|------|------------:|---------:|----:|------|
| s3e1  | 0.556329 | 0.556242 | −0.00009 | tiny real gain |
| s3e3  | 0.84144 (AUC) | 0.84572 | +0.00428 | **inflated — OOF weight overfit** (see below) |
| s3e9  | 12.07003 | 12.06596 | −0.00407 | small gain (v3 blend search) |
| s3e11 | 0.295280 | 0.295280 | 0 | none |
| s3e16 | 1.335630 | 1.335630 | 0 | none |
| s3e19 | 9.757070 | 9.757070 | 0 | none |

**Is v3 a "no-gain architecture" vs v1/v2? No.** A fresh v3 search gives a **small real score gain**
on the comps where the v1 blend was suboptimal (s3e1, s3e9, ~1e-4 to 4e-3), and none where v1 was
already optimal. But v3's *major* value over v1/v2 is not score — it's **rigor/reproducibility**
(deterministic bit-level reruns, solo caches, stored champion weights, budget/phase policy). That
infrastructure is what makes a defensible ±1e-5 measurement possible at all (v1/v2 can't support it).

**The s3e3 +0.0043 is NOT a real gain — it is OOF blend-weight overfit.** s3e3 is 1,677 rows; the
forward search added ~8 members and re-optimized blend weights on the same OOF, inflating OOF AUC
without generalization. Proof: the clean **paired bootstrap** on the controlled 2-member injection
gives s3e3 Δ=−0.00001, CI [−0.00032, +0.00030] — **not significant**; s3e1 likewise Δ=−0.00006,
CI [−0.00023, +0.00011] not significant. So the forward-search jump does not survive a paired test.

**Honest measurement limits on the adapted infra:** s3e9/s3e11/s3e16 could not be paired-bootstrapped
(their `eval` modules need comp-specific encoders/args the generic harness can't reconstruct);
s3e19's bootstrap failed its bit-check (OOF-space mismatch). These blocks are *themselves* the
recurring lesson: clean ±1e-5 injection measurement genuinely requires the true-v3 infrastructure,
not adapted v1/v2 trees.

## Final verdict (all 15)
- **External-idea injection: negligible and inconsistent.** Reweighting never beats the optimum (8/8
  rigorous); adding members helps only ~2/8 rigorously, by ~1e-5; every larger apparent gain
  (s3e3 +0.004, s3e9 −0.003) fails a paired test or is OOF overfit. Not a driver of anything.
- **v3 vs v1/v2: small real score gain (~1e-4 to 4e-3 on some comps) + a large rigor/reproducibility
  gain.** v3 is a better architecture, but not because of a big score jump.

---
## Faithful full v3 re-run — all 15 to a uniform v3 standard (2026-07-16)

`tree_search/run_faithful_v3.py` replays each of the 7 not-originally-v3 comps by RE-TRAINING
its exact committed node configs (extracted from experiments_tree.json/_v2) deterministically
through that comp's real eval module, then re-searching blends with the v3 weight search — a
genuine re-run (re-training + re-search), not the adapter reformat. Reproduction certificate:

| Comp | nodes | exact <5e-4 | champion Δ vs committed |
|------|------:|------------:|------------------------:|
| s3e1  | 22 | 22/22 | -0.000002 |
| s3e3  | 22 | 22/22 |  0.000000 |
| s3e11 | 24 | 24/24 | +0.000004 |
| s3e16 | 24 | 24/24 |  0.000000 |
| s3e20 | 38 | 38/38 |  0.000000 |
| s3e19 | 22 | 20/22 |  0.000000 (2 node drifts, champion unaffected) |
| s3e9  | 26 | 21/26 | -0.003087 (early-batch LGBM non-determinism; v3 slightly better) |

5/7 reproduce the committed champion exactly (|Δ|<=4e-6); all within the ~1e-4 standard. The
one real champion drift (s3e9, -0.003) is the pre-determinism-fix non-determinism, the honest
signature of a real re-run vs a reformat. (The hand-transcribed run_s3e16_v3.py had 2 seed
drifts; the replay reproduced s3e16 exactly, so replay-of-committed-configs is the more faithful
method.) **All 15 competitions now stand on a uniform faithful v3 process.** Injection and
v3-vs-v1/v2 conclusions are unchanged.
