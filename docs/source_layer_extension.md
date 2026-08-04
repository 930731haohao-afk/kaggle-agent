# Generalizing the External-Data Source Layer — 2026-08-03

## The question this answers

The injection layer had a property worth keeping and a limitation worth removing, and they
were the same property.

**Worth keeping:** the dossier may name only whitelisted sources. Every joined column
therefore has a stated origin, a stated leakage rule, and a pinned vintage. That is what lets
a reported number be traced back to the data it used.

**The limitation:** every whitelisted source was keyed by *country and time* — World Bank
indicators, ECB reference rates, national holiday calendars. So the layer fired on exactly
the competitions shaped like a country panel and on nothing else. The architecture generalized;
the vocabulary did not. A patent-matching task whose `context` column is an opaque CPC code,
a chemistry task with an element reference table, a retail task keyed by store rather than
country — the dispatcher had no way to express any of them, and the whitelist had nothing to
offer.

The obvious fix (let the agent fetch whatever it judges useful) destroys the property that made
the layer trustworthy. What follows is the procedure that grows the vocabulary without it.

## The dangerous class, stated plainly

On Kaggle the most valuable "external data" for a given competition is frequently *a dataset
another competitor assembled* — occasionally containing the test labels themselves. **No
leakage test can catch that.** Nothing is dated wrong; the answer is simply present. A
temporal as-of join with a correct lag will happily join the answer.

So the defence cannot be a check. It has to be structural, and it is two rules:

1. The host must be a **general-purpose reference publisher** — a statistics agency, a
   standards body, a central bank. `ALLOWED_DOMAINS` in
   [`external_data/source_registry.py`](../external_data/source_registry.py) is a set-membership
   test, not a judgment call.
2. Data-sharing platforms are **excluded outright**, listed explicitly in `FORBIDDEN_DOMAINS`:
   Kaggle, GitHub (including `raw.githubusercontent.com`), HuggingFace, Zenodo, figshare,
   Mendeley, Drive, Dropbox. Zenodo is a legitimate archive and is still excluded, because
   anyone may publish a derived dataset there.

Widening `ALLOWED_DOMAINS` is a deliberate reviewed act — it changes what the agent can reach.
Adding a *dataset from an already-listed host* is what the admission procedure automates.

## Join key classes

The key class decides which merge function may run and which leakage rule applies. Until now
it lived in a prose column of the whitelist table and in the dispatcher's hardcoded call to
`merge_year_safe`. It is now explicit:

| class | shape | leakage rule |
|---|---|---|
| `country_year` | one value per (country, year) | `lag=N`, as-of merge |
| `country_date` | one value per (country, date) | `lag=N`, as-of merge |
| `currency_month` | one value per (currency, month) | `lag=N`, as-of merge |
| `date` | one value per date, no entity key | `lag=N`, as-of merge |
| `lookup` | static attribute table keyed by a code/id | `static` — no time axis exists |

`lookup` is the class that breaks the country-panel ceiling. A CPC code, an ISO code, a product
id: there is no period to be on the wrong side of, so the leakage rule is `static` and a
declared lag is itself a refusal reason — a lag there would imply a temporal guarantee the
data cannot carry. Symmetrically, a temporal class claiming `static` is refused.

## The admission procedure

`docs/source_proposals/<key>.json` → `external_data/admit_source.py` → (on pass, with
`--admit`) `external_data/admitted_sources.json`. Five gates, in order, each refusing with a
stated reason:

1. **Structural** (no network, no data): host in `ALLOWED_DOMAINS`; key class known and its
   leakage rule consistent with it; provenance evidence present.
2. **Leakage invariant**: synthetic prediction rows are built spanning the source's own period
   range and joined through the declared rule; every joined value must respect the lag. Tested
   against the proposal's *own fetched frame*, so a fetcher that flattens dates fails here.
3. **Snapshot determinism**: two fetches must agree on the vintage, and the vintage pins into
   `sources.lock.json`. A source whose content moves cannot back a reported number — the World
   Bank rebasing case (07-30) rewrites every historical year, so "the same indicator, refetched"
   is silently a different series.
4. **Coverage**: the fetcher must report which requested keys it could not supply. `unmatched`
   may be empty — that is a fine answer — but it must be *present*. This gate exists because of
   `flag_feature`, where countries with no calendar silently received all-zero flags while the
   ledger booked the operator realized.
5. **Reachability**: the dispatcher must have a route that realizes this key class for this
   source. Added after the seam regression caught admission and dispatch disagreeing (below).

### The pre-registration requirement

`expected_value` is not documentation. It is a pre-registration in the same spirit as
`docs/preregistrations/`: a source admitted without a stated expectation can be declared useful
after the fact whatever it did. The gate requires a **predicted magnitude** (a number, even a
range) *and* a **stated adjudication rule** (the comparison that would prove it wrong).

This gate found its own hole during construction. An early version checked only length, and
`expected_value = "should help"` passed — eleven characters that predict nothing. It now
requires a digit and an adjudication keyword. Two refusal classes were added for it.

## The first admitted source: CPC titles

`cpc:titles` — the official Cooperative Patent Classification code→title table, published
jointly by the EPO and USPTO. Chosen deliberately as the *first* non-panel source: no country,
no date, no time axis at all, on a competition shape (`us-patent-phrase-to-phrase-matching`) the
old layer could not have expressed.

Its pre-registration: replacing the bare CPC code with its official title text should help a
text-pairing model that currently sees the context as an opaque token; predicted effect on
Pearson r **between +0.000 and +0.010**, judged against a matched no-external arm at equal
budget. A negative effect falsifies the premise that the code is uninformative as a token.

Cached vintage `CPC-2024.01`, 17 real section/subclass titles seeded, network off by default.

Adversarial variants of the same real table were refused: a `raw.githubusercontent.com` mirror
(data-sharing platform), a Kaggle dataset copy, a HuggingFace copy, a Zenodo record, and an
unlisted university host.

## The rules gate runs first

[`external_data/rules_gate.py`](../external_data/rules_gate.py) decides whether the competition
permits external data *before* the dossier considers any source. Three properties, each with a
case behind it:

- **A decision requires a quote and a section reference**, recorded next to the dossier, so a
  later reader can check the reading rather than trust it.
- **Silence resolves to forbidden.** The absence of a prohibition is not a permission.
- **A config/rules disagreement is a `conflict`, not a pick.** This is the s3e19 case: the
  conservative setup-time `config.yaml` flag said external data was not allowed, and the
  official rules (Section 7.C) said it was. The correct behaviour was to refuse to guess and
  flag it for verification — which is what the module now encodes as a regression test.

It does not fetch. A gate that silently downloads makes its decision depend on network state at
an arbitrary moment, and offline reproducibility of the decision matters more than convenience.

## What Stage 0.5 now does

[`00_problem_dossier.md`](../.claude/skills/kaggle-agent/references/00_problem_dossier.md)
step 5 is the rules gate (hard stop on `forbidden`/`unstated`/`conflict`); step 6 assesses need
against the admitted vocabulary, and — new — when the task needs external data and no admitted
source fits, the agent may **write a proposal and implement the fetcher**, then hand it to
`admit_source.py`.

**The agent may propose and implement; it may not admit.** Refusal is a normal outcome, its
reason is printed, and working around it is not permitted. That asymmetry is the whole design:
the vocabulary grows through a procedure rather than through improvisation, and the growth is
adjudicated by rules that were written before the proposal was.

## Seam bugs found by the end-to-end regression

Each module had a selftest and each passed. The failures were all at the *seams*, which is why
[`external_data/selftest_pipeline.py`](../external_data/selftest_pipeline.py) exists — it walks
a source from a rules verdict all the way to joined columns in an injection ledger, then walks
the same seams with inputs that must not pass. 38 checks.

**1. An empty reference table joined silently.** `merge_lookup_safe` checked that the declared
columns existed but not that any rows did. A fetcher returning nothing — cache miss, network
off, upstream schema change — produced an all-NaN column that reads downstream as *"the source
was useless"* rather than as *"the fetch failed"*. Those are opposite conclusions and the run
could not tell them apart. Now refused.

**2. A total key-space miss joined silently, by a different route.** A non-empty reference whose
keys do not overlap the competition's (wrong column, wrong code vintage, string-vs-int keys)
produced the same all-NaN column. Partial misses are normal and are reported; a *total* miss is
a wiring error and now raises with both key samples in the message.

**3. Admission and dispatch disagreed about what was reachable.** This is the worst of the
three. The admission gate would happily admit an ECB or NOAA temporal source, but the
dispatcher's temporal path fetches only `worldbank:*` — so the registry would advertise a
source, a dossier would emit it in good faith, and the run would book an unrealized idea with
an obscure reason, mid-competition. A registry that can lie about what is available is worse
than a shorter registry. Fixed by `apply.py:dispatch_route()` plus admission gate 5, so the
disagreement is caught at admission time where the missing fetcher is a two-line change.

One reported "bug" turned out to be correct-by-design and the *test* was wrong: a `join_feature`
naming a column the competition lacks does not raise — it adds no column, leaves both frames
untouched, and lands in the ledger with the exact exception. The other operators still run. That
is the ledger contract working, not a swallowed error.

## What the adversarial verification found — 2026-08-04

Four independent auditors attacked the finished layer along different lenses (smuggling a
forbidden source, silent degradation, integration seams, the rules gate), and every finding was
handed to a separate verifier instructed to refute it and required to reproduce it before it
counted. **15 findings survived; 13 were refuted.** All 15 are fixed, and each is now a case in
`selftest_pipeline.py` (48 checks).

**The rules gate failed open.** This was the worst of them by a distance. `_PERMIT_PATTERNS`
contained the bare substring `allowed to use external data`, which matches inside the single
commonest Kaggle prohibition — *"Participants are **not** allowed to use external data"* — and
no forbid pattern covered that wording. The gate the whole subsystem treats as its hard stop
returned `permitted` and cited the prohibiting clause as its evidence. Four related holes came
with it: the forbid list missed Kaggle's canonical *"you may not use data other than the
Competition Data"* family, bolded negation and table rows; `_find` took the first match anywhere
in the document, so one permission-shaped sentence in an FAQ outvoted a binding prohibition; a
clause permitting *pretrained models* was read as permitting joined data; and matching against a
lowercased copy while slicing the original text let a `permitted` verdict be recorded with an
empty quote, violating the module's own stated invariant. The matcher is now clause-scoped and
negation-aware, collects all hits rather than the first, treats a pretrained-only permission as a
conflict, and matches case-insensitively on the original text. Ten prohibition phrasings that
used to open the gate are regression cases; four real permissions must still be recognised, with
a non-empty quote, or the gate would be useless in the other direction.

**`flag_feature` was still committing the exact defect the coverage gate was written to
prevent.** `check_coverage`'s docstring cites countries with no calendar silently receiving
all-zero flags — and the runtime path still did it: the branch discarded `merge_holiday_flags`'
report and never read `meta["unmatched"]`, so those rows got a systematically zero feature while
the ledger booked the operator realized and left `unrealized` empty. Worse, `merge_holiday_flags`
normalized the competition's dates but never the *calendar's*, so a calendar whose dates were
strings, `datetime.date` objects, or tz-aware Timestamps matched nothing at all — an all-zero
column with a clean report, on any panel. Both fixed; a zero-flag total across the whole panel is
now a raised error, because it is a join failure and never a plausible calendar.

**The dispatcher would silently substitute a different source.** `flag_feature` never read
`params["source"]` — it unconditionally fetched holidays — and my own new `dispatch_route`
blessed *any* `date`-class key. Together those meant a date-class source could pass the
reachability gate and then come back as holiday flags under the requested column name, with the
ledger naming `holidays` as the origin. Both ends now refuse.

**The leakage gate could not refuse anything.** `usable` was built as
`[p for p in ps if p <= row_period - lag]` and then tested with
`any(p > row_period - lag for p in usable)` — false by construction. `violations` was always
zero, so gate 2 reduced to "a period column exists with at least two values", and a source
declaring `lag=500` over a one-year span passed. It now checks what actually matters: that the
source can *supply* a value under its own declared rule.

**Three smaller ones.** `merge_lookup_safe` used "the first value column is null" as a proxy for
"the key did not resolve", which undercounted matches, named resolved keys as unresolved, and
could trigger the new total-miss abort on a join where every key matched; it now carries an
explicit match indicator and reports per-column nulls, so a second value column that arrives
entirely empty is recorded as unrealized instead of booked as a feature. A missing join key
crashed the whole join from inside the *report builder*, because `sorted()` on a list mixing
strings with NA raises. And `ext_key` was read by `apply.py` but was not a `SourceSpec` field, so
re-admitting a source deleted it and a proposal declaring it died with an uncaught `TypeError`
rather than a stated refusal.

**What the refuted thirteen say.** Most were attacks on the domain gate — userinfo tricks,
punycode, open redirects, `file://`, arbitrary module import through the fetch spec. They were
refuted because the gate's design already handles them or because the attack could not reach the
gate. The pattern in what *survived* is the same one this project keeps rediscovering: not
"someone broke in", but "something was produced, recorded, or asserted, and nothing checked that
it was consumed or honoured".

## What did not change

- No existing behaviour. A source absent from the registry keeps the `country_year` default, so
  every previously admitted source dispatches exactly as before, and `external_data/selftest.py`
  (the panel-join invariant suite) passes unchanged.
- No published benchmark number. This is capability and safety work on the agent, not a
  re-measurement.
- The whitelist itself is not now open. It grew by exactly one source, through the procedure,
  and the procedure refused nineteen classes of proposal along the way.

## Files

| file | role |
|---|---|
| `external_data/source_registry.py` | domain whitelist, key classes, `SourceSpec`, structural gates |
| `external_data/admit_source.py` | 5 admission gates, CLI, selftest (19 refusal classes) |
| `external_data/rules_gate.py` | competition-rules gate, s3e19 conflict regression |
| `external_data/join.py` | `merge_lookup_safe`; `merge_holiday_flags` relaxed for date-only calendars |
| `external_data/sources.py` | `fetch_cpc_titles` |
| `external_data/apply.py` | key-class dispatch, `_apply_lookup_join`, `dispatch_route` |
| `external_data/selftest_pipeline.py` | end-to-end seam regression, 38 checks |
| `docs/source_proposals/cpc_titles.json` | the first proposal |
| `external_data/admitted_sources.json` | the registry every consumer reads |

Reproduce every check:

```bash
python3 external_data/rules_gate.py            # rules gate
python3 external_data/admit_source.py --selftest
python3 external_data/selftest_pipeline.py     # end-to-end seams
VIRTUAL_ENV= uv run python3 external_data/selftest.py   # panel-join invariants (needs holidays)
python3 external_data/admit_source.py docs/source_proposals/cpc_titles.json  # dry-run the real proposal
```
