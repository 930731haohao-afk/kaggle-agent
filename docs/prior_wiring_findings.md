# Prior-Wiring Experiment: three-arm comparison × five competitions

> **Question answered**: the priors computed by `suggest_priors` were never read by the search (write-only, the root cause of the Stage 5 no-op).
> After wiring — what can mechanical-template translation (arm A) and LLM on-the-fly translation (arm B) each bring?
> Date 2026-07-13 (s3e5/s6e1 pilots + same-day cross-season three-event expansion).
> Code: `tree_search/prior_wiring.py` (translators), `run_s3e5_wire(_b).py`, `run_s6e1_wire.py`, `run_wire_v3.py` (generic), `wire_bootstrap_v3.py`.

## Design

**Paired three-arm**, same event, same folds, same cache, with the OFF tree never modified:

| Arm | Candidate source | Implementation |
|---|---|---|
| OFF | historical hand-written seeds only | the submitted historical tree itself (= the true control where priors are not delivered) |
| A mechanical | + template-translated prior candidates | `wire_priors()`: each prior is dispositioned into a candidate/satisfied/policy/n-a/unmapped ledger (delivery proof); the executable ones are translated into configs marked `[PRIOR-INT/EXT-xx]`; deterministic, no RNG |
| B LLM | + LLM subagent proposals | the subagent reads "priors + full search state + contract" → frozen replay (propose-once) → dynamic-whitelist validation (only parameter keys that appeared in the tree are allowed) → evaluate `[PRIOR-LLM-xx]` → a **pre-declared** full-pool re-blend |

Significance: paired bootstrap B=10,000 (champion weights taken from `search_state.node_results` or v2 config.result, with the metric recomputed and **bit-checked** against the tree score; see the honest boundary for s4e11).

## Five-event summary table

| Event | n | Metric | Arm A (mechanical) | Arm B reblend vs champion | paired bootstrap (improvement) | LLM member weight |
|---|---|---|---|---|---|---|
| s3e5 | 2,056 | QWK | champion unchanged | +0.00051 (nominal new champion) | CI [−0.0064, +0.0130] **not significant** (low power) | 24.4% |
| s6e1 | 630,000 | R² | champion unchanged | +0.000029 (new champion) | **[+0.000005, +0.000054] significant** ✅ | 29.8% (shallow XGB = largest single weight) |
| s4e1 | 165,034 | AUC | champion unchanged | +0.000071 (new champion) | **[+0.000022, +0.000119] significant** ✅ | 23.6% (shallow XGB #61=0.187) |
| s4e11 | 140,700 | Accuracy | champion unchanged | −0.000215 (no change) | [−0.00049, +0.00006] not significant | **0.1% (all zeroed out)** |
| s5e10 | 517,754 | RMSE | champion unchanged | −0.000004 (no change) | [−0.000007, −0.000001] (marginally significant **worse**) | 10.4% |

**s5e10's NNLS diagnostic (RMSE has a closed-form solution, post-hoc/exploratory)**: the **true convex optimum of the 39-member expanded pool = 0.055964 < champion 0.055968**,
giving the LLM members **19.1%** weight (shallow CAT #53=0.17); post-hoc bootstrap improvement +0.0000038, CI [+0.0000017, +0.0000058] significant. → Arm B's "not passing" on s5e10 is a
**39-dimensional Dirichlet weight search failing to find the bowl bottom** (an approximation failure), not the LLM members having no value. The same kind of approximation failure also appears in
s6e1-A (the 30-member blend below the 21-member champion) and s4e11-B (the 47-member reblend below the champion — the champion solution is within the feasible region but was not searched).

## Conclusions

1. **The write-only hole is sealed, all five events have a disposition ledger.** Prior delivery rate 100%, every [PRIOR-*] node traceable to the prior that generated it.
2. **Arm A: champion unchanged in 5/5 events** — reweighting the same pool (plus conservative template-reachable variants) cannot beat the convex-optimum champion, and the NNLS theorem is verified in all five events. Honest limitation of the template library: most generic process-type priors are unmapped (s6e1/s4e1/s4e11/s5e10 each ~20/27 entries), and mechanical rules cannot translate "suggestion"-type knowledge.
3. **Arm B (LLM pool expansion), across 5 events: 2 events with statistically significant improvement (s6e1 R², s4e1 AUC), 1 event low-power nominally positive (s3e5), 2 events not passing (s4e11, s5e10)**; of which s5e10 was confirmed by the NNLS closed-form diagnostic to have real value in the LLM members, the failure lying in the weight-search approximation; s4e11 (accuracy, the only threshold-type metric) is the cleanest "ineffective" event — the weight search zeroed out all LLM members.
4. **A recurring winning archetype**: the "shallow and heavily-regularized diverse-library model" (bias-dominated shallow learner) took the largest weight among LLM members in both s6e1 and s4e1, and s5e10's NNLS solution also reused the same archetype (shallow CAT) — "using a bias-dominated member to decorrelate a variance-dominated pool" is a transferable recipe, worth writing back into the experience library.
5. **Engineering recommendation (derived from 3)**: the blend weights for RMSE-family competitions should replace/verify Dirichlet with the **NNLS closed-form solution** (high-dimensional approximation error measured at ~1e-5 magnitude, enough to eat up the real gain); other metrics can use "Dirichlet + champion warm-start" to prevent the degeneracy of "failing to search back to a known champion."
6. **Meaning for the plan**: the second pillar (idea injection) is upgraded from "honest null (not delivered)" to "**delivered and producing statistically significant gain in 2/5 events** (and 1 more event with a true value under the closed-form diagnostic)" — the absolute magnitude of the gain is extremely small (1e-5~1e-4 level), with no practical LB significance, but the mechanism and the theoretical prediction (NNLS escape clause) are fully self-consistent.

## Honest boundary

- Each event's arm B makes only **one pre-declared** reblend comparison; s5e10's NNLS test is **post-hoc exploratory** and not counted toward confirmatory conclusions.
- s4e11's recomputed accuracy (0.940021) differs from the tree-stored value (0.940014) by 7e-6 — the tie-handling of the threshold sweep differs slightly from the evaluator; both are below the champion, and the conclusion is unaffected.
- All CV-only (all five competitions are closed), with no real-LB corroboration.
- Once frozen, the LLM proposals can be replayed and evaluated; the proposals themselves are not reproducible (only replayable).
- The gain magnitudes of the two significant events (+3e-5 R², +7e-5 AUC) are negligible in any practical sense; the value of this experiment lies in **mechanism attribution** (prior delivery → pool expansion → decorrelation → convex optimum shifts up), not in the score.
