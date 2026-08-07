# Stage 0.5 — Problem Dossier (upstream knowledge injection)

## Purpose

Answer "what kind of problem is this?" BEFORE deep EDA or any modeling. The benchmark's
largest my-agent failure was a problem-identification miss on a country-panel task: the needed
judgment "this task requires external macro data" exists at this stage, not at the modeling or
blend stage, and missing it left the run in the wrong mode of a bimodal leaderboard. Post-hoc
injection at the blend stage measured ~1e-5 gains (archived findings report — do not open during a run); the
same knowledge applied upstream moved CV error by whole points. Inject knowledge where the
leverage is. (Specific scores and competition names are deliberately absent here: this file is
read on every benchmark run, and a motivating example that names a benchmark competition hands
that competition's re-run its own recorded outcome — 2026-08-07 audit.)

## When

After Stage 0 (setup) and before Stage 1 (EDA). Cheap: reading + reasoning only, no training.

## Inputs

- Competition description / rules / metric definition (official pages or `config.yaml`)
- Data schema: column names, dtypes, ranges — a `head`/`describe` of each file is enough here
- The task-level prior library ([TASK-*] entries), read via
  `python3 knowledge/task_priors_for.py <comp>` — rendered from `knowledge/knowledge_base.json`; there is no raw prose file to read (step 3)

**Forbidden inputs**: competition-specific discussion threads, public kernels, or write-ups.
Reproducing competition-specific solutions is the NVIDIA lane's frozen method, not ours; the
dossier must be derivable from the problem statement plus *task-type* knowledge only.

## Procedure

1. Classify the task family (time-series forecast / iid tabular regression / binary
   classification / ordinal / multi-target / structured output / spectral / image / NLP).
2. Determine the train–test relationship: is the test set a **future time window**? A disjoint
   group? An iid split? State the evidence (date columns, id ranges, description wording). If
   undeterminable from the description, write `"unknown"` — Stage 1 EDA must then resolve it.
3. Match against the task-prior library — but read it through the filter, NEVER raw:

   ```bash
   python3 knowledge/task_priors_for.py <competition-slug>            # the priors you may use
   python3 knowledge/task_priors_for.py <competition-slug> --report   # what was withheld, and why
   ```

   List every [TASK-*] entry whose trigger fits, from that output only. **There is no raw
   prose library to read**: the knowledge lives in `knowledge/knowledge_base.json` as
   structured facts whose prose is competition-free by contract, and the command above RENDERS
   your competition's view — evidence from the competition being solved is excluded by exact
   set arithmetic, an entry whose admissible evidence empties is dropped whole (the Action
   line *is* the distilled answer), cross-agent evidence is never rendered, and aggregates are
   recomputed from the surviving facts. If `--report` shows an entry was dropped, that is the
   mechanism working: you had no such prior before you solved that competition.
4. Derive the split policy from the match (this pre-empts the shuffled-KFold-on-time-series
   trap: a controlled split experiment on a benchmark panel measured a 4.5× optimism bias from the forbidden split — see the filtered prior library's TASK-TS-FUTURE evidence).
5. **Rules gate FIRST, before any external-data thinking.** A verdict may ALREADY be
   recorded: if `competitions/<comp>/rules_verdict.json` already exists, READ it and move on —
   it is an operator-resolved human reading, and re-running the gate overwrites it with
   `conflict` (every real rules page carries generic restrictive words in non-data clauses),
   silently shutting off a lane the human already opened (2026-08-10 audit). Only when no
   verdict exists: save the competition's rules text to a file and run

   ```bash
   python3 external_data/rules_gate.py competitions/<comp>/rules.txt \
       --competition <comp> --config-flag <true|false> \
       --record-to competitions/<comp>/rules_verdict.json
   ```

   (omit `--config-flag` if `config.yaml` has no `external_data_allowed`; exit status is the
   verdict — 0 permitted, 3 forbidden, 4 conflict, 5 unstated). On a CONFLICT, stop this line
   of work and flag it for a human — do not resolve it yourself and do not re-run the gate
   hoping for a different answer. The gate
   requires a QUOTE and a section reference, treats silence as forbidden (absence of a
   prohibition is not a permission), and reports a disagreement between `config.yaml` and the
   rules text as a CONFLICT to verify rather than picking a side — the recorded case where the
   local flag said no and rules Section 7.C said yes. If the gate does not return
   `permitted`, external data is off for this competition: say so in the dossier and stop
   this line of work. A run that ignores the rules is disqualified whatever it scores.

6. Assess external-data need: does the target plausibly depend on covariates absent from the
   provided files (macro indicators, calendars, geography, reference tables)? Pick candidates
   from the whitelist in the filtered library (and `external_data/admitted_sources.json`), each
   with a join key and a leakage rule.

   **If the task needs external data but NO admitted source fits**, you may propose a new one
   instead of giving up — the vocabulary grows through a procedure, not by improvisation.
   Write `docs/source_proposals/<key>.json` with the fields
   `external_data/source_registry.py:SourceSpec` requires (url, publisher, join key class,
   join/value columns, leakage rule, licence, evidence that the source predates the
   competition and was not authored by a participant, and a FALSIFIABLE pre-registration of
   what it is expected to be worth and how that will be judged), implement the fetcher in
   `external_data/sources.py` following the existing `(frame, meta)` contract, then run
   `external_data/admit_source.py <proposal>` — it decides, you do not. Refusal is a normal
   outcome and its reason is printed; do not work around it. Two rules are structural rather
   than advisory: the host must be a listed general-purpose reference publisher, and a
   data-sharing platform is excluded outright because a file there may be a competitor's
   dataset containing the answer, which no leakage test can detect.
7. Express every injection idea as a **typed operator** from the operator vocabulary, read
   through the same filter as the priors:

   ```bash
   python3 knowledge/task_priors_for.py <competition-slug> --ops
   ```

   That vocabulary (rendered from `knowledge/knowledge_base.json`) is the shared contract of
   this stage and the execution layer; the raw file's evidence tables carry per-competition
   results, so during a benchmark run it is read only through the filter — every operator row
   survives, but an evidence cell naming this competition is withheld. Free-text ideas are not executable and get silently dropped (this
   cost a benchmark competition its whole point: a correct "macro covariate as a level" judgment reached an
   execution layer that could only append feature columns, and a GBDT cannot extrapolate a
   feature outside its training range). **Decision rule: if the test window lies outside the
   training range and the covariate carries the level, emit BOTH `join_feature` and
   `ratio_target` (or `log_offset`) as separate arms and race them; state in the rationale which
   one you would pick and why, but do not drop either. Reason: the extrapolation argument favours
   `ratio_target`, and on both competitions measured so far the leaderboard favoured
   `join_feature` — see the form verdict in TASK-TS-FUTURE (via the filtered library).** Anything you want that the operator set cannot
   express goes in `not_recorded`.
8. **Open pre-registrations are binding — read them through the filter.** If the dossier
   fires external-data need on a country-panel time-series task (TASK-TS-FUTURE class), read
   the registered hypotheses via

   ```bash
   python3 knowledge/task_priors_for.py <competition-slug> --prereg
   ```

   — **never `docs/preregistrations/` raw during a benchmark run.** The registration files
   carry an evidence table naming benchmark competitions with their private scores and
   winning forms, so a raw read hands a re-run of those competitions its own recorded answer
   (2026-08-07 audit: this was the fourth unfiltered door). The filtered output keeps every
   registered hypothesis and its protocol. Copy each open prediction into this dossier's
   `preregistration` field and commit it BEFORE any model is fitted or any leaderboard is
   consulted. The prediction never changes what runs — both forms still race.
9. Write the dossier (schema below) to `competitions/<comp>/dossier.json`.

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
     "source_prior": "experience.md GDP recipe (cite the filtered library's own evidence line here — this example deliberately names no competition or score, because this file is read on every run: 2026-08-07 audit)"},
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
  or a structured evidence item / trigger refinement in `knowledge/knowledge_base.json` (never prose appended to a file — prose write-back is the leak class the renderer exists to close).
