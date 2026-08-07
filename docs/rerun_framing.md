# Framing for the 20-competition re-run — approved wording

Drafted 2026-08-07, per the discussion of what the new numbers measure. To be inserted into
REPORT_v3.tex (Methods and Limitations) when the re-run's numbers replace the current ones.
The numbers themselves are placeholders until the run completes.

## The paragraph (Methods)

> The my-agent results in this report come from a re-run of all 20 competitions under the
> architecture frozen at commit `<RERUN_COMMIT>` — after a three-round adversarial audit
> (131-agent architecture gate; three fix-and-re-verify cycles) repaired 14 blocker-class
> defects, several of which affected scores directly (a silently void ensemble track on
> sep-2022, champion selection dropping legitimate winners, two disagreeing blend scorers).
> The AIDE and NVIDIA yardsticks remain **frozen at their original runs**: the comparison is
> a corrected agent against unchanged references. Two consequences follow, and we state them
> rather than let the table imply otherwise. First, the corrections were made with all 20
> outcomes visible, so the my-agent rate is a **re-measurement of a corrected architecture,
> not a clean out-of-sample estimate**; adding or removing a defect fix can move it, and no
> significance test can launder that. Second, my-agent runs with a knowledge library
> distilled from its own prior encounters with related tasks. Entry-level self-exclusion
> (`knowledge/task_priors_for.py`) removes every prior whose evidence rests on the
> competition being solved — including the whole entry when its evidence set empties — and
> removes the other agents' recorded results for every competition. What remains is
> generalization from *other* tasks: legitimately accumulated capability, but a different
> quantity from meeting each competition fresh, and one the frozen yardsticks do not possess.
> The three-way comparison should be read accordingly: a knowledge-accumulating agent,
> corrected in-sample, against two frozen single-encounter references.

## The sentence (Limitations, if space is short)

> The re-run numbers are a re-measurement of a corrected architecture on a fixed competition
> set with outcomes previously observed — not an out-of-sample estimate — and my-agent
> carries cross-task knowledge (self-excluded at entry level per competition) that the frozen
> AIDE/NVIDIA references do not.

## What must be true before insertion

- `<RERUN_COMMIT>` is the commit the run launched from, recorded by the launcher.
- Every my-agent number in the tables comes from the re-run; no mixing with pre-correction
  numbers in the same column.
- The old numbers move to an appendix table labelled as superseded, not deleted — the delta
  between the two runs is itself evidence about how much the defects cost.
