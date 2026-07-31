# Stage 0.5 — Problem Dossier (upstream knowledge injection)

## Purpose

Answer "what kind of problem is this?" BEFORE deep EDA or any modeling. The benchmark's
largest my-agent failure (s3e19: all three agents in the lower mode of a bimodal leaderboard,
48–50 vs. top 4.67) was a problem-identification miss — the needed judgment "this task requires
external GDP data" exists at this stage, not at the modeling or blend stage. Post-hoc injection
at the blend stage measured ~1e-5 gains (`docs/injection_all15_findings.md`); the same knowledge
applied upstream on tps-jan-2022 moved CV SMAPE by whole points (4.1793 with GDP vs. AIDE's
6.1551 without). Inject knowledge where the leverage is.

## When

After Stage 0 (setup) and before Stage 1 (EDA). Cheap: reading + reasoning only, no training.

## Inputs

- Competition description / rules / metric definition (official pages or `config.yaml`)
- Data schema: column names, dtypes, ranges — a `head`/`describe` of each file is enough here
- `knowledge/task_priors.md` — the task-level prior library ([TASK-*] entries)

**Forbidden inputs**: competition-specific discussion threads, public kernels, or write-ups.
Reproducing competition-specific solutions is the NVIDIA lane's frozen method, not ours; the
dossier must be derivable from the problem statement plus *task-type* knowledge only.

## Procedure

1. Classify the task family (time-series forecast / iid tabular regression / binary
   classification / ordinal / multi-target / structured output / spectral / image / NLP).
2. Determine the train–test relationship: is the test set a **future time window**? A disjoint
   group? An iid split? State the evidence (date columns, id ranges, description wording). If
   undeterminable from the description, write `"unknown"` — Stage 1 EDA must then resolve it.
3. Match against `knowledge/task_priors.md`: list every [TASK-*] entry whose trigger fits.
4. Derive the split policy from the match (this pre-empts the shuffled-KFold-on-time-series
   trap: 4.5× optimism bias measured, 4.56 vs. 20.41 SMAPE on s3e19).
5. Assess external-data need: does the target plausibly depend on covariates absent from the
   provided files (macro indicators, calendars, geography)? Pick candidates ONLY from the
   whitelist in `task_priors.md`, each with a join key and a leakage rule.
6. Express every injection idea as a **typed operator** from
   `knowledge/injection_operators.md` — that file is the shared vocabulary of this stage and
   the execution layer. Free-text ideas are not executable and get silently dropped (this
   cost s3e19 its whole point: a correct "GDP as a level covariate" judgment reached an
   execution layer that could only append feature columns, and a GBDT cannot extrapolate a
   feature outside its training range). **Decision rule: if the test window lies outside the
   training range and the covariate carries the level, emit BOTH `join_feature` and
   `ratio_target` (or `log_offset`) as separate arms and race them; state in the rationale which
   one you would pick and why, but do not drop either. Reason: the extrapolation argument favours
   `ratio_target`, and on both competitions measured so far the leaderboard favoured
   `join_feature` — see the form verdict in `knowledge/task_priors.md` TASK-TS-FUTURE.** Anything you want that the operator set cannot
   express goes in `not_recorded`.
7. **Open pre-registrations are binding.** If the dossier fires external-data need on a
   country-panel time-series task (TASK-TS-FUTURE class), check `docs/preregistrations/` for
   REGISTERED hypotheses: copy each one's prediction into this dossier's `preregistration`
   field and commit it BEFORE any model is fitted or any leaderboard is consulted. The
   prediction never changes what runs — both forms still race; see
   `docs/preregistrations/horizon_length_form_selector.md`.
8. Write the dossier (schema below) to `competitions/<comp>/dossier.json`.

## Output schema — `dossier.json`

```json
{
  "task_family": "time-series forecast",
  "test_window": {"relation": "future", "evidence": "test dates 2021-01..2021-12 strictly after train 2015..2020"},
  "split_policy": {"scheme": "time-based (year < y vs year == y) or TimeSeriesSplit", "forbidden": "shuffled KFold"},
  "metric": {"name": "SMAPE", "traps": ["asymmetric near zero", "scale-free: per-series normalization matters"]},
  "external_data": {
    "needed": "likely",
    "candidates": [
      {"source": "World Bank GDP per capita", "join_key": "country x year", "leakage_rule": "only values dated <= prediction year"}
    ]
  },
  "injection_ideas": [
    {"operator": "ratio_target",
     "params": {"source": "worldbank:gdp_per_capita",
                "join": {"keys": ["country", "year"], "lag": 0},
                "space": "log", "carry_forward": true},
     "rationale": "test year is outside the training range; the covariate must carry the level because a GBDT cannot extrapolate it as a feature",
     "source_prior": "experience.md GDP recipe (evidence: tpsjan22 exp #2/#3, -2.6 SMAPE)"},
    {"operator": "trend_term", "params": {"unit": "year", "degree": 1, "centered": true},
     "rationale": "common year drift shared by all series", "source_prior": "TASK-TS-FUTURE"},
    {"operator": "flag_feature",
     "params": {"source": "holidays", "join": {"keys": ["country", "date"]}, "as": ["is_holiday"]},
     "rationale": "daily retail-like panel", "source_prior": "TASK-TS-CALENDAR"}
  ],
  "matched_task_priors": ["TASK-TS-FUTURE", "TASK-TS-CALENDAR"],
  "not_recorded": []
}
```

Fields that cannot be determined are listed in `not_recorded` — never guessed.

## Downstream consumption

- **Stage 1 (EDA)** must *verify* the dossier's `test_window` and `external_data.needed`
  hypotheses against the actual data — the dossier is a prior, not a conclusion. Record
  confirmations/refutations back into `dossier.json` under `"eda_verdict"`.
- **Stage 2 (features)** reads `injection_ideas` and dispatches each typed operator; every
  idea is either realized or recorded in `injection_ledger.json` with a reason — a run that
  drops an idea silently is the failure this contract exists to prevent. Implementation:
  `external_data/` module — `sources.py` (whitelisted fetchers, cached with snapshot dates)
  and `join.py` (`merge_year_safe` / `merge_period_safe` / `merge_holiday_flags`, LeakageError on violation,
  JSON audit logs). See `external_data/README.md`.
- **Tree search (v5 seeding)**: operators that change the data or the target (`ratio_target`,
  `log_offset`, `join_feature`) become raced lanes at small budget — an idea that changes the
  data gets its own lane; operators that only change the model (`postprocess`, `sample_weight`)
  become seed lineages inside the main tree. See `references/07_tree_search.md`.
- **Experience library write-back**: after the competition, any dossier hypothesis that was
  confirmed with a score delta becomes a new evidence-backed entry in `knowledge/experience.md`
  or a trigger refinement in `knowledge/task_priors.md`.
