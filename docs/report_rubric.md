# Report Scoring Rubric

> The acceptance baseline for the plan's Goal 2 + §5 report validation 1. Checks item by item whether each competition report **fully covers the competition purpose and
> the five workflow components**, and assesses **clarity and reproducibility**. Each item is scored 0–2, for a maximum of 20.
> Acceptance threshold: **total ≥ 16 and no single item scoring 0**.

| # | Checklist item | 0 points | 1 point | 2 points |
|---|---|---|---|---|
| 1 | **Competition purpose — what** | Missing problem definition or target column | Present but incomplete | Problem type, data, target column, metric all present (§1) |
| 2 | **Competition purpose — why** | Metric motivation not explained | Generic | Metric clearly linked to practical need / data characteristics (§1) |
| 3 | **Data spec** | Missing | Partial | Row count / column count / missingness / special rules all present (§3.1) |
| 4 | **Model spec** | Missing | Only lists the model | Member models + blend/ensemble config + weights (§3.2) |
| 5 | **Training spec** | Missing | Missing key items | CV scheme / folds / seed / key hyperparameters (§3.3) |
| 6 | **Inference procedure** | Missing | Partial | Post-processing + submission format/file (§3.4) |
| 7 | **Evaluation metric** | Missing | Only gives the definition | Definition + CV + LB (or honestly notes the competition is closed) (§3.5) |
| 8 | **Clarity** | Hard to read / jargon-laden | Adequate | Coherent narrative, plain-language terms, clear tables |
| 9 | **Reproducibility (instructions + traceability)** | No reproduction instructions | Instructions incomplete or numbers without provenance | §7 step-by-step instructions complete + all numbers traceable (verify passed) |
| 10 | **Major decisions reconstructable from report only** | Cannot reconstruct without reading the code | Some decisions reconstructable | The key decision chain (model selection / post-processing / ladder) is reconstructable from the report alone (§4 trajectory + turning points) |

## Scoring Method

- An **independent LLM agent** (not the report's author) scores each report, returning per-item scores + a one-sentence rationale + a total.
- For reports below the threshold (<16 or any item at 0), list the **specific missing items** and feed them back for report revision.
- Manual spot-check: 2–3 of the LLM scoring results are manually re-reviewed (the plan's "then a human spot-checks").

## Output

- Per-competition scoring table: `docs/report_validation.md` (rubric scores + missing items).
- Combined with §5 report validation 2 (code consistency) and 3 (report-only reproduction test) to form the complete report-validation evidence.
