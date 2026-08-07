# Injection Operators — the shared vocabulary of the judgment and execution layers

Stage 0.5 (the dossier) decides *what* to inject; the variant builders and evaluators
decide *how*. Between 07-29 and 07-30 those two layers spoke different languages and the
cost was measurable: on s3e19 the dossier correctly proposed World Bank GDP "as a
country-level scale covariate", the experience library already held the operational recipe
(`target = log(num_sold / gdp_pc)`, −2.6 SMAPE, evidence tpsjan22 exp #2/#3, an entry that
even names s3e19 as the case it dissolves) — and the execution layer could only append a
column to `FEATURE_COLS`. A GBDT cannot extrapolate a feature outside its training range,
so the 2022 test year was predicted at 0.96× the 2021 level while the data implied
+5..+33% growth, and the submission scored 48.24 SMAPE where the recipe targets single
digits. The judgment was right and the plumbing silently truncated it.

This file is the contract that prevents a repeat. **A dossier may only propose operators
listed here. Anything it wants that is not here goes in `not_recorded` — never in prose
that the execution layer will quietly ignore.** Every run reports a coverage ledger:
ideas proposed, ideas realized, ideas unrealized with a reason.

---

## Operator set

### `join_feature`
Add whitelisted external column(s) as model features.
```json
{"operator": "join_feature", "params": {"source": "worldbank:gdp_per_capita",
 "join": {"keys": ["country", "year"], "lag": 0}, "as": ["gdp_pc"]}}
```
Use when the covariate's *training-range values* carry signal. **Do not use alone for a
level covariate when the test window lies outside the training range** — trees are
piecewise-constant and cannot extrapolate; use `ratio_target` or `log_offset` instead.

**Join key classes (2026-08-03).** `join_feature` routes on the SOURCE's declared key class,
not on an assumption that every join is country×year: `country_year` / `country_date` /
`currency_month` / `date` are as-of merges with a declared lag, while `lookup` is a static
attribute table keyed by a code column (`params.join.on` names the competition's key column)
with no time axis and therefore no lag. The class is recorded in
`external_data/admitted_sources.json` when the source is admitted; unresolved keys are
reported in the ledger rather than filled.

### `ratio_target`
Divide the target by an external level covariate, fit on the ratio, multiply back at
predict time. This is the operator that extrapolates: the test-period level comes from the
covariate, not from the model.
```json
{"operator": "ratio_target", "params": {"source": "worldbank:gdp_per_capita",
 "join": {"keys": ["country", "year"], "lag": 0}, "space": "log",
 "carry_forward": true}}
```
`space: "log"` fits `log(y / c)` and inverts with `exp(pred) * c` (the jan-2022 recipe);
`space: "linear"` fits `y / c`. `carry_forward` reuses the last available covariate value
when the test period has none published yet — required whenever the test year post-dates
the source's coverage, and it must be logged.

**Covariate must be a constant-price / PPP series — never a current-price one.** A ratio
target multiplies the covariate into the prediction, so a nominal series propagates exchange
rate and inflation moves as though they were demand moves. Measured on the s3e19 countries,
2021 → 2022:

| series | Japan | Argentina | cross-country spread |
|---|---|---|---|
| `NY.GDP.PCAP.CD` (current USD) | −14.5% | +30.0% | **14.6 pp** |
| `NY.GDP.PCAP.KD` (constant 2015 USD) | +1.8% | +5.8% | 3.0 pp |
| `NY.GDP.PCAP.PP.KD` (PPP, constant) | +1.8% | +5.8% | 3.0 pp |

Use `gdp_per_capita_const` or `gdp_per_capita_ppp`. `apply.py` files a ledger entry if a
current-price indicator reaches this operator, and `covariate_volatility_check` flags any
covariate whose cross-group moves exceed 8 pp.

**Diagnostic (checked automatically, `apply.proportionality_check`) — RECORDED, NOT A GATE
(demoted 2026-07-31).** It was originally a precondition: dispersion of `total(group, period)/c`
across groups had to be small for the operator to be licensed. A pre-registered test killed
that: s5e1 violated it in 6 of 7 years (dispersion 0.05–0.21) and `ratio_target` still WON the
form race decisively (private MAPE 0.12417 vs join_feature's 0.15626, paired-significant), the
exact outcome the prediction's falsifier named. The check still runs and its verdict is still
written to the ledger — it describes the data honestly — but it must not block or select the
operator. Historical readings:

| Competition | dispersion by year | verdict | measured effect |
|---|---|---|---|
| s3e19 | 0.014 / 0.011 / 0.010 / 0.006 / 0.017 | proportional throughout | CV 10.148 → **7.793** |
| sep-2022 | 0.011 / 0.011 / 0.009 / **0.466** (2020) | breaks in the COVID year | CV 11.348 → 11.807 (**worse**); with 2020 down-weighted to 0.3, 11.515 |

A violation is recorded in the ledger, nothing more. (Historical note: on sep-2022 the
2020 down-weighting remedy improved CV and worsened the private LB — one more local signal
that misranked. And s5e1's win came with NO remedy applied despite six violated years.)

**Open caveat, and the reason the leaderboard adjudicates sep-2022.** A CV whose folds all
sit inside the training range cannot see this operator's actual advantage, which is
extrapolation to an unseen period; worse, sep-2022's last folds land on the broken 2020 while
the real test year is 2021, so CV scores the operator twice on its worst case. The
precondition should therefore be evaluated over *the periods the score depends on*, not
merely over all training periods — a refinement pending the sep-2022 leaderboard verdict.

### `log_offset`
Keep the target in log space and subtract `log(covariate)` as a fixed-elasticity offset —
algebraically `ratio_target` with `space: "log"`, exposed separately because linear models
implement it by dropping the column from the design matrix rather than by transforming `y`.
```json
{"operator": "log_offset", "params": {"source": "worldbank:gdp_per_capita",
 "join": {"keys": ["country", "year"], "lag": 0}}}
```

### `trend_term`
Add an explicit linear (or polynomial) time term so a common drift shared by all series can
be extrapolated. Pairs with `ratio_target`: the covariate carries the cross-sectional level,
the trend term carries the common year drift.
```json
{"operator": "trend_term", "params": {"unit": "year", "degree": 1, "centered": true}}
```
Evidence that this matters: ablating `year_c` on tpsjan22 cost 4.83 → 6.02 on the 2017 fold.

### `flag_feature`
Join deterministic indicator columns (holiday calendars, regime flags). No leakage concept
applies to a deterministic future calendar.
```json
{"operator": "flag_feature", "params": {"source": "holidays",
 "join": {"keys": ["country", "date"]}, "as": ["is_holiday"],
 "window": {"before": 5, "after": 10}, "per_name": true}}
```
`per_name` + a displacement window is the stronger form (tpsjan22: pooled flag → per-name
window, 5.46 → 4.19).

### `sample_weight`
Re-weight or flag an anomalous regime instead of deleting it.
```json
{"operator": "sample_weight", "params": {"predicate": "year in [2020, 2021]",
 "weight": 0.5, "also_flag": true}}
```

### `encoding`
How categorical columns enter the model. **Routed by scheme (07-30)**: `count`, `ordinal` and
`crosses` involve no target, so `apply.py` computes them as added columns in the data layer —
there is no fold boundary for a target-free transform to protect. `native` is the evaluators'
default and is recorded as advisory. `target` and `onehot_sparse` are refused with a reason:
the first needs per-fold computation inside the evaluator, the second a sparse linear family.
```json
{"operator": "encoding", "params": {"scheme": "target", "columns": ["surname"],
 "smoothing": 20.0}}
```
`scheme` ∈ {native, ordinal, count, onehot_sparse, target, crosses}, each with its own evidence:

| scheme | when | evidence |
|---|---|---|
| `native` | the strong default for GBDTs | [withheld — the original evidence cited another lane's run, which is not a legitimate input under the isolation protocol; treat as an unvalidated strong default] |
| `onehot_sparse` | all-categorical data, as a linear **member** in the pool | cat-in-the-dat: sparse OHE + logistic regression beat the GBDT outright |
| `count` | high cardinality, no target involved | no leakage path |
| `ordinal` | genuinely ordered levels only | |
| `target` | high-cardinality with signal, **fold-aligned only** | s4e1 (below) |
| `crosses` | tree members only | cat-in-the-dat: explicit crosses **hurt** an already-saturated sparse linear model |

**`target` carries a hard requirement.** The encoding must be computed inside each fold from
that fold's training rows, using *the model's own folds*. Computing it on an independent fold
split with a different seed looks safer and is in fact another leakage path: on s4e1 that
inflated AUC to 0.89653, and aligning the folds brought the same recipe back to ~0.8937. The
operator therefore sets `fold_aligned: true` and
`requires_evaluator_support: per_fold_target_encoding`; an evaluator that cannot honour it must
file a ledger entry rather than approximate.

### `objective`
Match the training loss to what the metric actually charges. Config-only: it changes what the
search proposes, not the data, so it is emitted as a seed node config — and since 07-30
`tree_search/seed_from_ledger.py` translates that emission into one seed node per model family
with `forbid_params` removed. Before that the emission had no reader (see the ledger note below).
```json
{"operator": "objective", "params": {"metric_family": "mae",
 "drop_imbalance_weighting": true}}
```
`metric_family` ∈ {mae, smape, rmse, rmsle, auc, accuracy, qwk}. The mapping carries its own
evidence: AUC is a ranking metric, so imbalance weighting perturbs the loss surface without
improving the ranking (s3e3 exp #3: 0.81901 → 0.83292 on removal; s4e1 confirms the sign at
165k rows, 0.893235 → 0.893650); ordinal targets take a regression head plus an
OptimizedRounder rather than a multiclass head (s3e5 exp #2→#3: QWK 0.47191 → 0.52687);
RMSLE means RMSE on log1p targets, so transform the target and keep L2.

### `blend_member`
Add one bias-dominated member to a variance-heavy pool as a decorrelator — added to the pool,
never replacing an existing member. Config-only.
```json
{"operator": "blend_member", "params": {"archetype": "shallow_regularized", "depth": 3,
 "weight_search": "nnls_then_dirichlet"}}
```
This is the transferable prototype from the prior-wiring experiments: the shallow,
heavily-regularized diverse-library member took the largest LLM-member weight in both s6e1
(+3e-5 R²) and s4e1 (+7e-5 AUC), and s5e10's NNLS diagnostic reused the same shape. For
RMSE-family metrics verify the weights with the NNLS closed form — the Dirichlet search's
~1e-5 approximation error is large enough to eat the gain.

### `split_policy`
Set the cross-validation split. Not optional: the dossier's split decision is binding on
the evaluator, and an evaluator that cannot honor it must fail loudly.
```json
{"operator": "split_policy", "params": {"scheme": "time_expanding",
 "time_col": "date", "n_splits": 5, "forbid": ["shuffled_kfold"]}}
```

### `postprocess`
Transforms applied **inside the metric function**, so every CV fold and every blend
candidate is scored on the post-processed vector.
```json
{"operator": "postprocess", "params": {"round_to_int": true, "clip_min": 0,
 "global_scale": "auto"}}
```
**Evaluator-owned, and that only means something where the evaluator implements this exact
vocabulary (2026-07-30 finding + fix).** `apply.py` books this operator "advisory: the
evaluator owns it" unconditionally, because the dispatcher works on dataframes and has no
evaluator to check against at that point. Until today `eval_s3e19.py`/`eval_sep22.py`'s
`maybe_postprocess` only recognized its own internal `auto_scale`/`scale` keys -- a name
mismatch against this operator's `round_to_int`/`clip_min`/`clip_max`/`global_scale`, so a
competition asking for rounding got nothing rounded, silently. Fixed for the two evaluators
with a live v5 arm (`global_scale` now aliases `auto_scale`; `round_to_int`/`clip_min`/
`clip_max` are implemented, applied scale-then-clip-then-round). **Not fixed generally**:
s5e10's dossier also emits this operator (`clip_min:0, clip_max:1, round_to_int:false`) and
its evaluator has no `maybe_postprocess` at all -- no live arm consumes it yet, so nothing
reported is affected, but the same silent-mismatch risk exists for any future arm on a
competition whose evaluator wasn't specifically checked. Verifying this per-competition, the
way `make_v5_arm.py` should but does not yet, is outstanding work.

---

## Coverage ledger (required output)

Every v5 run writes `injection_ledger.json` next to its tree:

```json
{"proposed": 7, "realized": 5, "unrealized": [
   {"operator": "sample_weight", "reason": "evaluator has no per-row weight hook"},
   {"operator": "flag_feature", "params_subset": "per_name windows",
    "reason": "generator supports boolean flags only"}]}
```

A run whose ledger has `unrealized` entries is still valid, but the report must state what
was not executed. Silence is the failure mode this file exists to prevent.

**The ledger reproduced that failure itself, and the fix is why it now has four buckets
(2026-07-30).** `realized` used to mix three different things: columns `apply.py` wrote,
configs emitted for a driver to seed, and decisions the evaluator owns. The middle group had no
reader at all — `plan["node_configs"]` was written and never consumed — so `objective`,
`blend_member` and `encoding` were booked realized while changing nothing about a run. A
consumer trace (not a vocabulary count) found it. The buckets are now `realized` (data-layer
only, and `realized_count` counts only these), `emitted_config` (a driver must confirm
consumption in `injection_consumed.json`), `advisory` (another layer owns it; verify there), and
`unrealized`. **Rule: never quote a coverage number without saying which bucket it is.**
