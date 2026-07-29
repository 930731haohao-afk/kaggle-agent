# External-data module (v5 Plan C)

Whitelisted external sources with leakage-guarded joins. Consumed by Stage 0.5
dossiers (`injection_ideas.data_level`) and by v5 tree-search data lanes.

## Rules

1. Only sources in the whitelist table of `knowledge/task_priors.md` may be
   fetched: 8 World Bank indicators (`WB_INDICATORS`), ECB monthly FX vs EUR
   (12 currencies, `ECB_FX_CURRENCIES`), OWID COVID compact (4 metrics,
   country x date; ~178 MB one-time cache — first fetch requires
   `allow_download=True`), holiday calendars, ISO tables.
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

Joins: `merge_year_safe` (yearly) and `merge_period_safe` (freq="M" or "D")
enforce the lag rule; the leakage invariant raises `LeakageError` and numeric
time columns must declare their unit (`month_ordinal`/`day_ordinal`) or the
join refuses with TypeError. Finer-grained external data requires an explicit
`coarsen` choice. `merge_holiday_flags` is an exact-date calendar lookup — no
lag concept applies (future holidays are not leakage). All reports include
unmatched keys, keys absent from the external table, mapping collisions, and
collapsed/duplicate counts.

Snapshots: WB = `lastupdated`; ECB = latest period in the response; OWID =
upstream `Last-Modified` (or max data date); holidays = package version pin
(offline, not cached).

`selftest.py` covers fetch, alias resolution (incl. the alias branch), year/
month/day lag semantics, the numeric-time input contract, the coarsen contract,
NaN fallback, a direct unit test of the leakage invariant, mapping-collision
reporting, order preservation, and the big-download guard (never fetches the
178 MB COVID file in tests).
