# Bug Report: External-Idea "Injection" Never Actually Entered the Search

> **Delivery target**: advisor/supervisor. **Dates**: discovered 2026-07-08, fixed 2026-07-13.
> Full technical text: `docs/phase_j_j3_findings.md` (discovery), `docs/prior_wiring_findings.md` (fix and validation).

## One sentence

The Stage 5 "external-idea injection" pipeline was **only half done**: the ideas were fetched and logged, but the tree search
**never read** this record from start to finish — so all prior measurements of "injection has no effect" actually measured
"**the ideas were simply not delivered**," not "the ideas are useless."

## Analogy

The consultant (idea bank) wrote a recommendation and filed it in the company's cabinet (`tree['priors']`),
but the kitchen menu (the candidate list the search will try) **was already printed**, and the plating process
had no "check the cabinet" step. The dishes went out as usual, business ran as usual — the recommendation affected not a single dish.

## What the bug is (data flow)

```
suggest_priors() fetches matching priors
        │
        ▼
tree["priors"] ← written, printed, saved          ✅ this half is fine
        │
        ✗  ←──────── break point: no program reads this field
        │
search candidates (seeds, mutations, expansions)   ← all come from "hand-written" code in the driver
```

The `[PRIOR P18]` markers appearing in the search code are **hand-written comment strings by the author**
(the author manually referenced priors when writing seeds), not evidence of an automatic pipeline — they made the pipeline
"look like" it was working, delaying the bug's discovery.

## How it was confirmed (two lines of evidence)

1. **Code level (grep)**: across all search engines (harness v2/v3/v4) and all drivers,
   the number of read points of `tree["priors"]` = **0**. Write-only.
2. **Experiment level (single variable, bit-for-bit comparison)**: on s3e3, re-running the entire search with only "injection on/off" toggled
   (same seed, same folds, same budget) — the two trees are **bit-for-bit identical** (22 nodes max|Δscore| = 0,
   18 OOF vectors max|Δ| = 0). If the 2 extra external ideas were read anywhere, they would necessarily change
   something; complete identity is a constructive proof of "not read."

## Scope of impact

- **The old Stage 5 conclusion needs rewriting**: the correct interpretation of "external injection is an honest null (no measurable gain)"
  in prior reports is "**the injection mechanism was not connected and could not be measured**."
- **The Stage 1–4 conclusions are unaffected**: the tree search's gains come from the candidate tree/backtracking/weight search themselves,
  unrelated to this pipeline; the experience library's knowledge did affect the search — but via the path of "the author manually transcribing it into seeds,"
  not automatic injection.
- Lesson: **"written and logged" does not equal "consumed."** Pipeline acceptance must verify the consumption end
  (in this case, ultimately caught by the single-variable test of "toggle injection on/off → output bit-for-bit identical").

## Fix status (summary)

On 2026-07-13 the consumption end was wired up (both a mechanical-template and an LLM translator), and a three-arm comparison over 5 competitions
validated "delivery" and its effect; see `docs/prior_wiring_findings.md` for details. This document delivers only the bug itself.
