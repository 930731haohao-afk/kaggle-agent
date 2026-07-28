# Idea Recombination (recombination / Stage 5.2) findings

> Plan §4/§6: "hand the core concepts of two high-scoring candidate solutions to the LLM to compare and recombine into a new solution." Date 2026-07-08.

## Mechanism (built)

- **Mechanical version**: `harness_v4.recombine` (J-2b, commit 0330b70) — inherits the stronger parent's model/hyperparameters + the union of member/feature families; pure, testable.
- **LLM version**: LLM recombination operator (subagent) — compares the core concepts of two parents and proposes a recombination candidate; **propose-once, frozen and replayable** (the LLM proposes once, freezes into a config, after which evaluation is reproducible). This is the plan's required "hand to the LLM to compare and recombine," and is this project's only **LLM-in-loop mutation** (filling the core gap marked in aygun_comparison).

## Pilot measurement (s6e1, R2 higher is better)

Recombine the **champion node#27** (21-member mega-blend, R2 0.787183) × the **best single model node#13** (single LGB, R2 0.786684).

**Result: redundant, no systematic gain.** Independently verified (not trusting the subagent blindly):

| Comparison | R2 | Note |
|---|---|---|
| champion (given Dirichlet weights) | 0.787183 | — |
| **NNLS convex optimum (blend/simplex family)** | **0.787183** | **bit-for-bit match** — the Dirichlet search already converged to the true convex optimum |
| OLS unconstrained stacking | 0.787192 | only +9e-6, and requires negative weights (min_w −0.092) |

- **Root cause (verified)**: member OOF correlation 0.998–1.000, **error-residual correlation 0.993–0.998** — all are GBDT variants on the same 22 features/same folds, nearly duplicate, with no diversity to squeeze; the ensemble is only +0.0005 over the best single model, already fully consumed by the convex optimum.
- The negative-weight OLS's +9e-6 is noise-level (fold sampling itself wobbles ~6e-4, about 100× larger), requiring negative weights is fragile on a bounded target [0,100], and the existing blend nodes (Dirichlet/grid_simplex, both on the simplex = convex) cannot express it.

## Conclusion

> **Recombination is structurally redundant under this project's architecture**: the tree search's mega-blend + convex weight search **is already the optimal combination of the whole pool's high-scoring nodes**
> (the champion = the NNLS convex optimum of the member pool, proven bit-for-bit), and the champion already includes the best single model. Recombining two high-scoring solutions = reweighting/subsetting a nearly-duplicate pool,
> with no systematic gain. **To break past 0.787183 requires injecting a new base learner decorrelated from the existing pool** (non-GBDT:
> NN/kNN/linear, new features/representations, target transforms, pseudo-labeling) — that is the work of a new base model, not recombining the existing pool.

This is **homologous** with the Stage 5 external-injection null (the pool is already a near-optimal combination, signal at the noise floor), a rigorous negative result at the same honesty level.
**ERA's second pillar (external injection + recombination): both mechanisms are built, both measured with no measurable systematic gain** — but both give a mechanistic explanation of "why,"
and point out that the true EV lies in "injecting a decorrelated new base model."

**Reproducible**: LLM recombination operator proposal (frozen) + `docs/scripts/` (or scratchpad) `eval_recombine_s6e1.py` (NNLS/OLS/Dirichlet comparison, reproducing the champion bit-for-bit).
