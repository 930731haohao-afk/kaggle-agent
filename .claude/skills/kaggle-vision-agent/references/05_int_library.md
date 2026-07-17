# Stage V5 — Distil to the vision [INT] library

After the run (or at any major milestone), distil what was LEARNED into
`knowledge/vision_experience.md` — the vision analogue of `knowledge/experience.md`. This is how
the discovery-first agent compounds: this run's discoveries become the next run's priors.

## House rules (identical to the tabular library)

- **Every entry carries score evidence**: `competition, exp #N, scoreA -> scoreB`. No evidence, no
  entry — anecdotes are not knowledge.
- Entries are `[INT]` by definition (self-discovered, in-project). If external ideas get injected
  later (opt-in stage), they are marked `[EXT]` and live separately — never blur the two.
- Negative results are first-class: "tried X, hurt/no effect" with the numbers prevents re-trying
  dead ends next run (the tabular 反面教訓 section proved its worth).
- Organize by matchable section headers (metric / data type / technique keywords), so the same
  keyword-matching prior mechanism (`suggest_priors`) can consume this file later without changes.

## What to distil from each stage

- **V2 discovery**: which backbone family fit which data type; probe-vs-finetune ranking agreement
  observed this run (append to the derisk evidence); which augmentation family won and for what
  data; LR regime; whether LightGBM-on-embeddings beat the linear head.
- **V3 fine-tune**: epochs-to-convergence, whether seed bagging paid, config-vs-config deltas.
- **V4 ensemble**: solo-vs-blend delta, TTA delta, whether calibration helped, blend weights
  (which members earned weight — family diversity or raw strength?).
- **CV honesty**: CV<->LB gap for this competition; any grouping/leakage findings.

## Template

```markdown
### <topic keyword header, e.g. "digits/grayscale small images">
- <finding, one sentence, actionable>. | Evidence: <comp>, exp #<N>, <A> -> <B>.
```
