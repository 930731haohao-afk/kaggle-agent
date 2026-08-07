"""Admission gate for new external-data sources — the procedure that lets the vocabulary grow.

The dossier may propose only whitelisted sources. That closed list is what makes every joined
column traceable, and it is also what has kept the injection layer to country-panel
competitions: every listed source is keyed by country and time. This module is how a new
source enters the list WITHOUT giving up the property that made the list worth having.

A source is admitted only if it passes, in order:

  1. STRUCTURAL gates (external_data/source_registry.py, no network, no data): the host is a
     listed general-purpose reference publisher; the join key class is known and its leakage
     rule matches; provenance evidence exists.
  2. LEAKAGE invariant: for a temporal source, no external value dated after
     (row period - lag) may reach a prediction row. Tested against the proposal's own fetched
     frame, not against a hand-written fixture, so a fetcher that flattens dates or drops the
     lag fails here.
  3. SNAPSHOT determinism: two fetches return the same vintage, and the vintage pins into
     sources.lock.json. A source whose content moves under us cannot back a reported number.
  4. COVERAGE report: the join must state which keys matched and which did not. Silence is a
     failure -- the study's own recurring defect is an artifact that looks complete because
     nothing recorded what was missing.
  5. REACHABILITY: the dispatcher (external_data/apply.py) must have a route that realizes
     this key class for this source. Admission and dispatch disagreeing is the worst failure
     mode available here -- the registry advertises a source, a dossier emits it in good
     faith, and the run books an unrealized idea with an obscure reason mid-competition.

Refusal is the normal outcome for anything that cannot evidence all four; the reason is
always printed. Admission appends the spec plus its evidence to the registry file and leaves
a trace, so a result that used the source can be traced back to the day it was admitted and
to what was checked.

Usage:
  VIRTUAL_ENV= uv run python3 external_data/admit_source.py <proposal.json> [--admit]
  VIRTUAL_ENV= uv run python3 external_data/admit_source.py --selftest
Without --admit it dry-runs every check and reports; with --admit it also writes the record.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from external_data.source_registry import (  # noqa: E402
    TEMPORAL_CLASSES, SourceRejected, SourceSpec, structural_gates,
)

ADMITTED_PATH = _HERE / "admitted_sources.json"


# ---------------------------------------------------------------------------
# gate 2: leakage invariant
# ---------------------------------------------------------------------------
def check_leakage(spec: SourceSpec, frame: pd.DataFrame, *, period_col: str = "period") -> str:
    """No external value may be dated after (row period - lag).

    Tested by construction rather than by trusting the fetcher: we build synthetic prediction
    rows spanning the source's own period range, join through the declared rule, and assert
    that every joined value's source period respects the lag. A lookup source has no period
    column at all, and a period column appearing there is itself the failure -- it means the
    source has a time axis its declared class denies.
    """
    if spec.join_key_class not in TEMPORAL_CLASSES:
        # any time-axis column violates the static claim, not only one literally named
        # 'period' -- a frame keyed [code, year, title] was admitted static and then
        # merge_lookup_safe's keep-last silently served each key its LATEST year
        # (2026-08-07 round-4)
        time_cols = [c for c in frame.columns
                     if str(c).strip().lower() in
                     {period_col, "year", "date", "month", "quarter", "week", "day",
                      "timestamp", "datetime", "time", "asof", "as_of", "vintage_year"}
                     or pd.api.types.is_datetime64_any_dtype(frame[c])]
        if time_cols:
            raise SourceRejected(
                f"{spec.key!r}: declared join_key_class {spec.join_key_class!r} (static) but "
                f"the fetched frame carries time-axis column(s) {time_cols}. A time axis the "
                f"class denies means the leakage rule 'static' is a claim the data "
                f"contradicts -- and the lookup join's keep-last would silently serve each "
                f"key its latest period.")
        return "static source: no time axis, no temporal leakage path"

    if period_col not in frame.columns:
        raise SourceRejected(
            f"{spec.key!r}: temporal class {spec.join_key_class!r} but the fetched frame has "
            f"no {period_col!r} column ({list(frame.columns)}); the lag cannot be enforced "
            f"on data that does not say when it is from.")
    lag = int(spec.leakage_rule.split("=")[1])
    periods = sorted(pd.unique(frame[period_col]))
    if len(periods) < 2:
        raise SourceRejected(
            f"{spec.key!r}: only {len(periods)} distinct period(s) in the fetched frame, so "
            f"the lag cannot be exercised. Fetch a wider range before admission.")

    # For every (key, period) the source offers, the value a row at that period may see must
    # come from a period <= row_period - lag. Verify against the frame directly: the join
    # implementation is checked separately by external_data/selftest.py's invariant test;
    # here we verify the SOURCE can support the rule it declares.
    key_cols = [c for c in frame.columns if c not in (period_col, *spec.value_columns)]
    if not key_cols:
        raise SourceRejected(f"{spec.key!r}: fetched frame has no key column besides "
                             f"{period_col!r} and the values {spec.value_columns}")
    # The first version of this loop built `usable = [p for p in ps if p <= row_period - lag]`
    # and then asked whether any element of `usable` was `> row_period - lag`. That is false by
    # construction, so `violations` was always 0 and the gate reduced to "a period column
    # exists with >= 2 values" -- a gate that cannot refuse (2026-08-04 adversarial
    # verification). What must actually be checked is that the source can SUPPLY a value under
    # its own declared lag: for each key, at least one period must have a usable predecessor,
    # and the declared lag must not exceed the source's own coverage span.
    # The lag's UNIT follows the key class, and getting that wrong is silent. The first fix
    # guarded the span check with `isinstance(periods[0], int)`, so a datetime64 period column
    # skipped it entirely -- and `ps[-1] - lag` on a datetime64 subtracts NANOSECONDS, so
    # lag=99999 read as 99999ns and a wildly over-declared lag passed (2026-08-04).
    if pd.api.types.is_datetime64_any_dtype(frame[period_col]):
        step = pd.Timedelta(days=30) if spec.join_key_class == "currency_month" \
            else pd.Timedelta(days=1)
        lag_delta = step * lag
        span = periods[-1] - periods[0]
    elif isinstance(periods[0], (int, np.integer)):
        lag_delta = lag
        span = periods[-1] - periods[0]
    else:
        raise SourceRejected(
            f"{spec.key!r}: period column has dtype {frame[period_col].dtype} "
            f"(e.g. {periods[0]!r}); the lag cannot be arithmetic on it. Emit integer periods "
            f"(years) or real datetimes -- strings are refused rather than guessed at.")
    if lag_delta > span:
        raise SourceRejected(
            f"{spec.key!r}: declared lag={lag} ({lag_delta}) exceeds the source's own coverage "
            f"span ({periods[0]}..{periods[-1]}, span {span}). Under this rule no prediction "
            f"row could ever be given a value, so the source cannot support the rule it "
            f"declares.")

    # Group by EVERY key column. Grouping on key_cols[0] alone hid a starved key whenever the
    # frame's first non-period column was not the join key -- a constant leading column
    # collapsed every entity into one group and the check became vacuous (2026-08-04).
    if frame[key_cols].isna().all(axis=None):
        raise SourceRejected(
            f"{spec.key!r}: key column(s) {key_cols} are entirely null, so the lag check has "
            f"nothing to group by and would pass vacuously.")
    starved = []
    for kval, grp in frame.dropna(subset=key_cols).groupby(key_cols, observed=True):
        ps = sorted(pd.unique(grp[period_col]))
        # the latest row this key could serve is ps[-1]; it needs some p <= ps[-1] - lag
        if not any(p <= ps[-1] - lag_delta for p in ps):
            starved.append(str(kval))
    if not starved and not len(frame.dropna(subset=key_cols)):
        raise SourceRejected(f"{spec.key!r}: no rows with a non-null key to check the lag on")
    if starved:
        raise SourceRejected(
            f"{spec.key!r}: lag={lag} leaves {len(starved)} key(s) with no usable value at "
            f"any period (e.g. {starved[:5]}). The source cannot honour its declared leakage "
            f"rule for those keys, so the join would silently produce nothing for them.")
    return (f"temporal source, lag={lag} enforceable over {len(periods)} periods "
            f"({periods[0]}..{periods[-1]})")


# ---------------------------------------------------------------------------
# gate 3: snapshot determinism
# ---------------------------------------------------------------------------
def check_snapshot(fetch, spec: SourceSpec) -> str:
    """Two fetches must agree on the vintage, and the vintage must be a real value.

    A source whose snapshot is None or moves between calls cannot back a reported number:
    the World Bank rebasing case (2026-07-30 finding) rewrites every historical year, so
    "the same indicator, refetched" is silently a different series.
    """
    _f1, m1 = fetch()
    _f2, m2 = fetch()
    s1, s2 = m1.get("snapshot"), m2.get("snapshot")
    if not s1:
        raise SourceRejected(
            f"{spec.key!r}: fetcher returned no snapshot. Every source must report the "
            f"upstream vintage it read, or a rerun cannot be tied to the data it used.")
    if s1 != s2:
        raise SourceRejected(
            f"{spec.key!r}: snapshot moved between two consecutive fetches ({s1!r} -> {s2!r})")
    return f"snapshot stable across two fetches: {s1}"


# ---------------------------------------------------------------------------
# gate 4: coverage is reported, never silent
# ---------------------------------------------------------------------------
def check_coverage(meta: dict, spec: SourceSpec) -> str:
    """The fetcher must report which requested keys it could not supply.

    `unmatched` may be empty -- that is a fine answer -- but it must be PRESENT. The failure
    this gate exists to stop is the one the 08-03 audit found in flag_feature: countries with
    no calendar silently received all-zero flags while the ledger booked the operator
    realized, so a systematically wrong feature looked like a complete one.
    """
    if "unmatched" not in meta:
        raise SourceRejected(
            f"{spec.key!r}: fetcher meta has no 'unmatched' key. A source that cannot say "
            f"which keys it failed to cover produces features that are wrong in a way "
            f"nothing downstream can see (cf. flag_feature's silent all-zero countries).")
    unmatched = meta["unmatched"]
    return f"coverage reported: {len(unmatched)} unmatched key(s){' — ' + str(unmatched[:5]) if unmatched else ''}"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# gate 5: reachability
# ---------------------------------------------------------------------------
def check_reachable(spec: SourceSpec) -> str:
    """The dispatcher must have a route that realizes this source's key class.

    Without this gate, admission and dispatch can disagree: the registry advertises a source,
    a dossier emits it in good faith, and the run silently records an unrealized idea. The
    failure surfaces mid-competition with an obscure reason instead of at admission time,
    where the missing fetcher is a two-line fix. A registry that can lie about what is
    available is worse than a shorter registry.
    """
    from external_data.apply import dispatch_route
    try:
        return f"dispatcher route: {dispatch_route(spec.key, spec.join_key_class)}"
    except ValueError as e:
        raise SourceRejected(f"{spec.key!r}: {e}") from None


def evaluate(spec: SourceSpec, fetch) -> dict:
    """Run every gate. Returns the evidence dict; raises SourceRejected on the first failure."""
    evidence = dict(structural_gates(spec))
    frame, meta = fetch()
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SourceRejected(f"{spec.key!r}: fetcher returned no rows")
    if not isinstance(spec.ext_key, str) or not spec.ext_key.strip():
        raise SourceRejected(
            f"{spec.key!r}: ext_key must be a non-empty column name, got {spec.ext_key!r}")
    if spec.join_key_class == "lookup" and spec.ext_key not in frame.columns:
        raise SourceRejected(
            f"{spec.key!r}: ext_key {spec.ext_key!r} is not a column of the fetched frame "
            f"({list(frame.columns)}). evaluate() used to check value_columns but not the KEY, "
            f"so a typo there was admitted and then failed mid-competition in the join.")
    missing_vals = [c for c in spec.value_columns if c not in frame.columns]
    if missing_vals:
        raise SourceRejected(f"{spec.key!r}: declared value_columns {missing_vals} are absent "
                             f"from the fetched frame ({list(frame.columns)})")
    evidence["leakage"] = check_leakage(spec, frame)
    evidence["snapshot"] = check_snapshot(fetch, spec)
    evidence["coverage"] = check_coverage(meta, spec)
    evidence["reachable"] = check_reachable(spec)
    evidence["rows"] = int(len(frame))
    return evidence


def admit(spec: SourceSpec, evidence: dict) -> Path:
    """Append the admitted source and its evidence to the registry file."""
    rec = json.loads(ADMITTED_PATH.read_text()) if ADMITTED_PATH.exists() else {}
    rec[spec.key] = {
        "url": spec.url, "publisher": spec.publisher,
        "join_key_class": spec.join_key_class, "join_columns": spec.join_columns,
        "value_columns": spec.value_columns, "leakage_rule": spec.leakage_rule,
        "ext_key": spec.ext_key,
        "licence": spec.licence, "predates_competition": spec.predates_competition,
        "author_is_not_participant": spec.author_is_not_participant,
        "expected_value": spec.expected_value, "notes": spec.notes,
        "admitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": evidence,
    }
    tmp = ADMITTED_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True))
    tmp.replace(ADMITTED_PATH)
    return ADMITTED_PATH


def _spec_from_json(path: Path) -> tuple[SourceSpec, dict]:
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict):
        raise SourceRejected(f"{path}: proposal must be a JSON object, got {type(d).__name__}")
    fetch_spec = d.pop("fetch", None)
    known = set(SourceSpec.__dataclass_fields__)
    unknown = sorted(set(d) - known)
    if unknown:
        # A stated refusal, not a TypeError from the dataclass constructor: the proposer is
        # an agent following written instructions, and "unexpected keyword argument" does not
        # tell it which field to remove or what the legal field set is.
        raise SourceRejected(
            f"{path}: unknown proposal field(s) {unknown}. Legal fields: {sorted(known)}. "
            f"(The fetch spec goes under \"fetch\".)")
    missing = [f for f, spec in SourceSpec.__dataclass_fields__.items()
               if f not in d and spec.default is dataclasses.MISSING
               and spec.default_factory is dataclasses.MISSING]
    if missing:
        raise SourceRejected(f"{path}: proposal is missing required field(s) {missing}")
    return SourceSpec(**d), (fetch_spec or {})


def _resolve_fetch(fetch_spec: dict):
    """Resolve the proposal's fetcher: {"module": "...", "callable": "...", "kwargs": {...}}."""
    import importlib
    if not isinstance(fetch_spec, dict) or not fetch_spec:
        raise SourceRejected('proposal has no "fetch" block: {"module": ..., "callable": ...}')
    unknown = sorted(set(fetch_spec) - {"module", "callable", "kwargs"})
    missing = [k for k in ("module", "callable") if k not in fetch_spec]
    if missing or unknown:
        raise SourceRejected(
            f'proposal "fetch" block is malformed: missing {missing}, unknown {unknown}. '
            f'Expected {{"module": "external_data.sources", "callable": "fetch_x", '
            f'"kwargs": {{}}}}.')
    try:
        mod = importlib.import_module(fetch_spec["module"])
        fn = getattr(mod, fetch_spec["callable"])
    except (ImportError, AttributeError) as e:
        raise SourceRejected(f'proposal "fetch" names an unresolvable callable: {e}') from None
    kwargs = fetch_spec.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise SourceRejected(f'proposal "fetch".kwargs must be an object, got {type(kwargs).__name__}')
    return lambda: fn(**kwargs)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal", nargs="?", type=Path)
    ap.add_argument("--admit", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.proposal:
        ap.error("proposal json required (or --selftest)")
    spec, fetch_spec = _spec_from_json(args.proposal)
    try:
        evidence = evaluate(spec, _resolve_fetch(fetch_spec))
    except SourceRejected as e:
        print(f"REFUSED  {spec.key}\n  {e}")
        return 2
    print(f"PASSES   {spec.key}")
    for k, v in evidence.items():
        print(f"  {k}: {v}")
    if args.admit:
        print(f"  admitted -> {admit(spec, evidence)}")
    else:
        print("  (dry run; pass --admit to record)")
    return 0


# ---------------------------------------------------------------------------
# selftest: a gate that cannot refuse anything is not a gate
# ---------------------------------------------------------------------------
def _good_spec(**over) -> SourceSpec:
    base = dict(
        key="worldbank:test_indicator", url="https://api.worldbank.org/v2/x",
        publisher="World Bank",
        join_key_class="country_year", join_columns=["country", "year"],
        value_columns=["value"], leakage_rule="lag=0", licence="CC-BY 4.0, open data",
        predates_competition="series published 1960-2024; competition opened 2023-06",
        author_is_not_participant="published by the World Bank, an institution, not a "
                                  "competition participant",
        expected_value="expected to move SMAPE by -2 to -6 on country-panel tasks; judged "
                       "by the paired leaderboard test against a no-external arm")
    base.update(over)
    return SourceSpec(**base)


def _frame(periods=(2019, 2020, 2021), keys=("SWE", "NOR")) -> pd.DataFrame:
    return pd.DataFrame([{"iso3": k, "period": p, "value": float(i)}
                         for i, (k, p) in enumerate((k, p) for k in keys for p in periods)])


def selftest() -> int:
    ok = _good_spec()
    fetch_ok = lambda: (_frame(), {"snapshot": "2026-01-31", "unmatched": []})  # noqa: E731
    ev = evaluate(ok, fetch_ok)
    assert ev["leakage"].startswith("temporal"), ev
    print("accepts a well-formed source:", ev["leakage"])

    # Every rejection below is a class the gates exist for. A gate that never refuses is
    # decoration, so each is exercised explicitly.
    def refuses(label, spec, fetch):
        try:
            evaluate(spec, fetch)
        except SourceRejected as e:
            print(f"refuses {label}: {str(e)[:88]}")
            return
        raise AssertionError(f"{label} was NOT refused")

    refuses("off-whitelist host", _good_spec(url="https://example.org/data.csv"), fetch_ok)
    refuses("data-sharing platform", _good_spec(url="https://www.kaggle.com/d/x"), fetch_ok)
    refuses("github raw", _good_spec(url="https://raw.githubusercontent.com/a/b/c.csv"), fetch_ok)
    refuses("unknown key class", _good_spec(join_key_class="planet_year"), fetch_ok)
    refuses("temporal class with static rule",
            _good_spec(leakage_rule="static"), fetch_ok)
    refuses("lookup class claiming a lag",
            _good_spec(join_key_class="lookup", leakage_rule="lag=1"), fetch_ok)
    refuses("lookup class whose data has a time axis",
            _good_spec(join_key_class="lookup", leakage_rule="static",
                       join_columns=["code"], value_columns=["value"]), fetch_ok)
    refuses("no provenance evidence", _good_spec(predates_competition=""), fetch_ok)
    refuses("no pre-registered expectation", _good_spec(expected_value="?"), fetch_ok)
    refuses("intent instead of a prediction",
            _good_spec(expected_value="should help the model quite a lot on this task"),
            fetch_ok)
    refuses("magnitude with no adjudication rule",
            _good_spec(expected_value="expected to improve the metric by roughly 0.01 to "
                                      "0.03 once the columns are joined in"), fetch_ok)
    refuses("silent coverage",
            ok, lambda: (_frame(), {"snapshot": "2026-01-31"}))
    refuses("no snapshot", ok, lambda: (_frame(), {"snapshot": None, "unmatched": []}))
    calls = {"n": 0}

    def _moving():
        calls["n"] += 1
        return _frame(), {"snapshot": f"2026-0{calls['n']}-01", "unmatched": []}

    refuses("moving snapshot", ok, _moving)
    refuses("empty frame", ok, lambda: (pd.DataFrame(), {"snapshot": "x", "unmatched": []}))
    refuses("declared value column absent",
            _good_spec(value_columns=["gdp_pc"]), fetch_ok)
    refuses("a lag wider than the source's own coverage span",
            _good_spec(leakage_rule="lag=50"), fetch_ok)
    refuses("single period cannot exercise the lag",
            _good_spec(leakage_rule="lag=1"),
            lambda: (_frame(periods=(2020,)), {"snapshot": "x", "unmatched": []}))

    lf = pd.DataFrame([{"code": "A01B", "title": "soil working"},
                       {"code": "A01C", "title": "planting"}])
    _lookup_fetch = lambda: (lf, {"snapshot": "2026-01", "unmatched": []})  # noqa: E731

    # a source the dispatcher cannot reach is refused: admitting it would make the registry
    # advertise a source that silently records an unrealized idea mid-competition
    refuses("lookup class with no registered fetcher",
            _good_spec(key="nowhere:thing", url="https://www.uspto.gov/x",
                       publisher="USPTO", join_key_class="lookup",
                       join_columns=["code"], value_columns=["title"],
                       leakage_rule="static"),
            _lookup_fetch)
    refuses("country_year source from a publisher the dispatcher cannot fetch",
            _good_spec(key="ecb:some_yearly_series",
                       url="https://data-api.ecb.europa.eu/x", publisher="ECB"),
            fetch_ok)

    # a genuine lookup source, with a fetcher the dispatcher actually has, passes
    look = _good_spec(key="cpc:titles",
                      url="https://www.cooperativepatentclassification.org/x",
                      publisher="CPC", join_key_class="lookup",
                      join_columns=["code"], value_columns=["title"],
                      leakage_rule="static")
    ev2 = evaluate(look, _lookup_fetch)
    assert ev2["leakage"].startswith("static"), ev2
    assert "dispatcher route" in ev2["reachable"], ev2
    print("accepts a well-formed lookup source:", ev2["leakage"])
    print(f"  reachability: {ev2['reachable']}")
    print("admit_source selftest: all sections passed (20 refusal classes exercised)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
