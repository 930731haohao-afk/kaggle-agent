# External-data module (v5 Plan C)

Whitelisted external sources with leakage-guarded joins. Consumed by Stage 0.5
dossiers (`injection_ideas.data_level`) and by v5 tree-search data lanes.

## Rules

1. Only sources in the whitelist table of `knowledge/task_priors.md` may be
   fetched (`sources.WB_INDICATORS`, holiday calendars, ISO tables).
2. Every join enforces: external value date/year <= prediction row date/year − lag.
   `lag=0` = same-period values (historical-solution convention); `lag=1` =
   strictly-available variant. Violations raise `LeakageError`.
3. Unmapped countries are reported, never guessed (extend `WB_NAME_ALIASES` or
   record in dossier `not_recorded`).
4. Every fetch is cached under `cache/` with the source snapshot date; every
   join can write a JSON audit log. Both make joins replayable.
5. Check competition rules permit external data before use.

## Usage

```python
from external_data.sources import fetch_worldbank, fetch_holidays, resolve_iso3
from external_data.join import merge_year_safe, merge_holiday_flags

mapping, unmatched = resolve_iso3(df["country"].unique().tolist())
gdp, meta = fetch_worldbank("gdp_per_capita", 2015, 2021)
df2, report = merge_year_safe(df, gdp, df_country="country", df_year="year",
                              out_col="gdp_pc", mapping=mapping, lag=0,
                              log_path="scripts/join_gdp.log.json", meta=meta)
```

`selftest.py` covers fetch, alias resolution, lag semantics, and the leakage guard.
