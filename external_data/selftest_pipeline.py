"""End-to-end regression for the source-extension pipeline.

The individual modules each have a selftest; this one exists because the failure mode that
worries me is not a broken module but a broken SEAM. The layer only earns trust if a source
that enters at one end (a rules verdict, then a proposal file) comes out the other end as
joined columns in an injection ledger, and if a source that should be refused cannot reach
the other end by ANY of the paths.

Order below is the order Stage 0.5 runs the gates:

  1. rules gate       -- may this competition use external data at all?
  2. vocabulary       -- does an admitted source fit? (registry lookup)
  3. proposal         -- if not, propose one; the admission gate decides, not the proposer
  4. dispatch         -- the dossier's typed operator reaches the right merge for its key class
  5. join             -- the merge preserves rows/order and reports what it could not resolve
  6. ledger           -- realized and unrealized are both recorded, never silently dropped
  7. validator        -- the contract checker sees registry-admitted keys as legal

and then the adversarial half: the same seams, walked by inputs that must NOT pass.

Run:  python3 external_data/selftest_pipeline.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from external_data import rules_gate  # noqa: E402
from external_data.admit_source import _resolve_fetch, _spec_from_json, evaluate  # noqa: E402
from external_data.apply import _admitted_sources, _key_class_of, apply_operators  # noqa: E402
from external_data.join import LeakageError, merge_lookup_safe  # noqa: E402
from external_data.source_registry import SourceRejected, SourceSpec, structural_gates  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROPOSAL = REPO / "docs" / "source_proposals" / "cpc_titles.json"

_checks = 0


def ok(msg: str) -> None:
    global _checks
    _checks += 1
    print(f"  [{_checks:02d}] {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _spec_from_proposal() -> SourceSpec:
    spec, _ = _spec_from_json(PROPOSAL)
    return spec


def _fetch_from_proposal():
    _, fetch_spec = _spec_from_json(PROPOSAL)
    return _resolve_fetch(fetch_spec)


# ---------------------------------------------------------------------------
# 1-2. the gates a competition passes through before any source is considered
# ---------------------------------------------------------------------------
def stage_rules_and_vocabulary() -> None:
    section("Stage 0.5 seam 1: rules gate runs before anything else")

    permit = ("7. EXTERNAL DATA\n7.C External data is allowed provided it is publicly "
              "available and free to all participants.\n")
    v = rules_gate.gate(permit)
    assert v.allows_external_data() and v.section and v.quote
    ok(f"permitted with a citable clause (section {v.section!r})")

    v = rules_gate.gate("1. OVERVIEW\nPredict the target; scored by RMSE.\n")
    assert not v.allows_external_data() and v.verdict == "unstated"
    ok("a rules text that never mentions external data does NOT open the gate")

    v = rules_gate.gate(permit, config_flag=False)
    assert v.verdict == "conflict" and not v.allows_external_data()
    ok("config/rules disagreement is a conflict, gate stays shut (s3e19 regression)")

    section("Stage 0.5 seam 2: the admitted vocabulary is visible to every consumer")
    admitted = _admitted_sources()
    assert "cpc:titles" in admitted, admitted
    ok(f"apply.py sees the registry: {sorted(admitted)}")
    assert _key_class_of("cpc:titles") == "lookup"
    ok("dispatcher resolves cpc:titles to the lookup key class")
    assert _key_class_of("worldbank:gdp_per_capita") == "country_year"
    ok("a source absent from the registry keeps the country_year default (no behaviour change)")

    from external_data.validate_dossier import _admitted_source_keys
    assert "cpc:titles" in _admitted_source_keys()
    ok("dossier validator accepts registry-admitted keys as legal source names")


# ---------------------------------------------------------------------------
# 3. proposal -> admission
# ---------------------------------------------------------------------------
def stage_admission() -> None:
    section("Stage 0.5 seam 3: a proposal is adjudicated, not self-approved")
    spec = _spec_from_proposal()
    ev = structural_gates(spec)
    assert ev["domain"] and ev["key_class"] and ev["provenance"]
    ok(f"the on-disk proposal passes the structural gates ({ev['key_class']})")

    evidence = evaluate(spec, _fetch_from_proposal())
    assert evidence["rows"] > 0 and evidence["snapshot"], evidence
    ok(f"full admission reproduces offline: rows={evidence['rows']}, "
       f"snapshot={evidence['snapshot'][:34]!r}")

    # The proposer cannot promote itself by writing admitted=True into the proposal file:
    # that field is not what the consumers read, and the gates run regardless.
    forged = json.loads(PROPOSAL.read_text())
    forged.pop("fetch")
    forged |= {"admitted": True, "admission_evidence": {"domain": "trust me"}}
    forged["url"] = "https://raw.githubusercontent.com/someone/cpc/main/titles.csv"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "forged.json"
        p.write_text(json.dumps(forged))
        forged_spec, _ = _spec_from_json(p)
    assert forged_spec.admitted is True  # the flag is set...
    try:
        evaluate(forged_spec, _fetch_from_proposal())
    except SourceRejected as e:
        assert "data-sharing platform" in str(e), e
        ok("...but a proposal that marks ITSELF admitted is still refused on the domain rule")
    else:
        raise AssertionError("a self-admitted github-hosted proposal passed the gates")

    # and the registry file, which IS what consumers read, is written only by admit()
    assert "should-not-exist" not in _admitted_sources()
    ok("the registry consumers read is written by admit(), never by the proposal file")


# ---------------------------------------------------------------------------
# 4-6. dispatch -> join -> ledger, on a shape the layer could not previously handle
# ---------------------------------------------------------------------------
def stage_dispatch_join_ledger() -> None:
    section("Stage 0.5 seams 4-6: dispatch, join and ledger on a NON-PANEL task")
    # us-patent-phrase-to-phrase-matching shape: no country, no date, no panel at all.
    train = pd.DataFrame({
        "id": [f"t{i}" for i in range(6)],
        "anchor": ["abatement", "abatement", "active catalyst", "el display",
                   "wood article", "zzz thing"],
        "target": ["forest", "eliminate", "catalyst", "oled", "lumber", "nothing"],
        "context": ["A47", "A47", "B01", "G02", "B27", "ZZZZ"],
        "score": [0.5, 0.75, 0.25, 1.0, 0.5, 0.0],
    })
    test = pd.DataFrame({
        "id": [f"s{i}" for i in range(3)],
        "anchor": ["abatement", "el display", "unknown"],
        "target": ["reduce", "screen", "thing"],
        "context": ["A47", "G02", "ZZZZ"],
    })
    ideas = [{"operator": "join_feature",
              "params": {"source": "cpc:titles", "join": {"on": "context"},
                         "as_prefix": "cpc_"},
              "rationale": "the context code is an opaque token to a text model"}]

    tr, te, plan = apply_operators(train, test, ideas, target_col="score")

    assert plan["added_columns"] == ["cpc_title"], plan["added_columns"]
    ok(f"dispatch reached the lookup path: added {plan['added_columns']}")
    assert plan["realized"][0]["join_key_class"] == "lookup"
    ok("ledger records the key class that decided the merge")
    assert len(tr) == len(train) and len(te) == len(test)
    ok(f"row counts preserved ({len(tr)} train / {len(te)} test)")
    assert list(tr["id"]) == list(train["id"])
    ok("row ORDER preserved (a lookup join must not reorder the frame)")
    a47 = tr.loc[tr["context"] == "A47", "cpc_title"].unique().tolist()
    assert len(a47) == 1 and "FURNITURE" in a47[0].upper(), a47
    ok(f"real values joined: A47 -> {a47[0][:44]!r}")
    assert tr.loc[tr["context"] == "ZZZZ", "cpc_title"].isna().all()
    ok("an unknown code carries NaN rather than a silently invented value")
    unresolved = [u for u in plan["unrealized"] if "unresolved" in u["params_subset"]]
    assert unresolved and "ZZZZ" in unresolved[0]["reason"], plan["unrealized"]
    ok(f"the unresolved key is REPORTED in the ledger, not swallowed ({len(unresolved)} entries)")
    assert plan["sources"]["cpc:titles"]
    ok(f"vintage pinned in the ledger: {plan['sources']['cpc:titles'][:32]!r}")


# ---------------------------------------------------------------------------
# adversarial half: the seams that must refuse
# ---------------------------------------------------------------------------
def stage_adversarial() -> None:
    section("adversarial: every refusal path, walked from the outside")
    base = _spec_from_proposal()

    def refused(spec: SourceSpec, expect: str, label: str) -> None:
        try:
            structural_gates(spec)
        except SourceRejected as e:
            assert expect.lower() in str(e).lower(), f"{label}: wrong reason: {e}"
            ok(f"refused {label}")
            return
        raise AssertionError(f"{label} was ADMITTED and should not have been")

    refused(replace(base, url="https://www.kaggle.com/datasets/x/cpc-titles.csv"),
            "data-sharing platform", "a Kaggle dataset (may contain the answer)")
    refused(replace(base, url="https://huggingface.co/datasets/x/cpc/raw/main/t.csv"),
            "data-sharing platform", "a HuggingFace mirror of the same real table")
    refused(replace(base, url="https://zenodo.org/record/1/files/cpc.csv"),
            "data-sharing platform", "a Zenodo archive (anyone may publish there)")
    refused(replace(base, url="https://some-university.edu/cpc.csv"),
            "not in ALLOWED_DOMAINS", "an unlisted host, however respectable")
    refused(replace(base, join_key_class="country_month"),
            "unknown join_key_class", "an invented key class")
    refused(replace(base, join_key_class="country_year", leakage_rule="static"),
            "must be 'lag=N'", "a temporal class claiming a static leakage rule")
    refused(replace(base, leakage_rule="lag=1"),
            "must be exactly 'static'", "a lookup class claiming a temporal lag")
    refused(replace(base, predates_competition=""),
            "needs stated evidence", "a source with no provenance evidence")
    refused(replace(base, expected_value="should help"),
            "falsifiable pre-registration", "a pre-registration that predicts nothing")
    refused(replace(base, expected_value="I expect this to improve the score a great deal, "
                                         "and I will look at the leaderboard afterwards."),
            "predicted magnitude", "a stated intent with no magnitude")
    refused(replace(base, expected_value="Predicted effect between +0.000 and +0.010 Pearson r "
                                         "on the public leaderboard for this competition."),
            "how it will be judged", "a magnitude with no adjudication rule")

    section("adversarial: the join refuses what it cannot do safely")
    df = pd.DataFrame({"code": ["A", "B"], "y": [1, 2]})
    ext = pd.DataFrame({"code": ["A", "A"], "title": ["first", "second"]})
    joined, report = merge_lookup_safe(df, ext, df_key="code", ext_key="code",
                                       value_cols=["title"], out_prefix="x_",
                                       meta={"snapshot": "test"})
    assert len(joined) == len(df) and report["ext_duplicates_dropped"] == 1
    ok("a duplicated lookup key cannot fan out rows; the drop is counted, not hidden")

    try:
        merge_lookup_safe(df, pd.DataFrame({"code": [], "title": []}), df_key="code",
                          ext_key="code", value_cols=["title"], out_prefix="x_",
                          meta={"snapshot": "test"})
    except (ValueError, LeakageError) as e:
        ok(f"an empty lookup frame is refused rather than joined to nothing ({type(e).__name__})")
    else:
        raise AssertionError("empty lookup frame was accepted")

    try:
        merge_lookup_safe(df, ext, df_key="code", ext_key="code", value_cols=["absent"],
                          out_prefix="x_", meta={"snapshot": "test"})
    except (KeyError, ValueError) as e:
        ok(f"a value column the source does not have is refused ({type(e).__name__})")
    else:
        raise AssertionError("absent value column was accepted")

    try:
        merge_lookup_safe(df, pd.DataFrame({"code": ["X", "Y"], "title": ["a", "b"]}),
                          df_key="code", ext_key="code", value_cols=["title"],
                          out_prefix="x_", meta={"snapshot": "test"})
    except ValueError as e:
        assert "wiring error" in str(e), e
        ok("a reference table sharing NO keys is a wiring error, not an all-NaN column")
    else:
        raise AssertionError("a total key-space miss was accepted")

    section("adversarial: an unregistered source cannot be dispatched")
    train = pd.DataFrame({"id": [1, 2], "context": ["A47", "B01"], "score": [0.5, 0.25]})
    test = pd.DataFrame({"id": [3], "context": ["A47"]})
    ideas = [{"operator": "join_feature",
              "params": {"source": "cpc:titles", "join": {"on": "no_such_column"}},
              "rationale": "points at a column the competition does not have"}]
    tr, te, plan = apply_operators(train, test, ideas, target_col="score")
    assert plan["added_columns"] == [] and plan["realized"] == [], plan
    assert plan["unrealized"] and "KeyError" in plan["unrealized"][0]["reason"], plan
    ok("a join key absent from the data adds NO column and lands in the ledger with the "
       "exact exception, so the run continues but the failure is on the record")
    assert list(tr.columns) == list(train.columns) and list(te.columns) == list(test.columns)
    ok("a failed operator leaves the frames untouched (no half-applied join)")

    # A country-panel shape, so the unregistered source reaches the temporal path rather than
    # failing earlier on a missing date column -- the refusal must come from the whitelist.
    panel = pd.DataFrame({"country": ["Sweden"] * 3 + ["Norway"] * 3,
                          "date": ["2019-01-01", "2020-01-01", "2021-01-01"] * 2,
                          "num_sold": [10, 11, 12, 20, 21, 22]})
    panel_te = pd.DataFrame({"country": ["Sweden", "Norway"],
                             "date": ["2022-01-01", "2022-01-01"]})
    ideas = [{"operator": "join_feature",
              "params": {"source": "acme:private_feed",
                         "join": {"keys": ["country", "year"], "lag": 0}},
              "rationale": "a source name that was never admitted"}]
    _, _, plan2 = apply_operators(panel, panel_te, ideas, target_col="num_sold")
    assert plan2["added_columns"] == [], plan2
    reason = plan2["unrealized"][0]["reason"]
    assert "not a whitelisted yearly covariate" in reason, reason
    ok("an unadmitted source name on a valid panel shape is refused by the whitelist itself, "
       "not incidentally by a missing column")

    from external_data.apply import dispatch_route
    try:
        dispatch_route("acme:private_feed", "country_year")
    except ValueError as e:
        assert "no route" in str(e)
        ok("admission and dispatch cannot disagree: an unroutable source fails the "
           "reachability gate before it can enter the registry")
    else:
        raise AssertionError("an unroutable source reported a dispatch route")


def stage_verification_regressions() -> None:
    """The twelve defects the 2026-08-04 adversarial verification confirmed.

    Each of these passed every module selftest that existed at the time. They are kept here,
    walked from the outside, so a later refactor cannot quietly reintroduce one.
    """
    section("regressions: the 2026-08-04 adversarial verification findings")

    # 1-5. the rules gate used to FAIL OPEN on the commonest Kaggle prohibition
    for label, text in [
        ("negated permit", "Participants are not allowed to use external data.\n"),
        ("'not permitted'", "The use of external data is not permitted.\n"),
        ("bolded negation", "External data is **not** allowed.\n"),
        ("table row", "| External data | Not allowed |\n"),
        ("'no data other than'", "You may not use any data other than the Competition Data.\n"),
        ("pretrained-only permission",
         "You are allowed to use external data a pretrained model was trained on, but you "
         "may not join additional data.\n"),
    ]:
        v = rules_gate.gate(text)
        assert not v.allows_external_data(), f"rules gate FAILED OPEN on {label}: {v.verdict}"
    ok("rules gate fails closed on all 6 prohibition phrasings that used to open it")
    v = rules_gate.gate("7.C External data is allowed for all participants.\n")
    assert v.allows_external_data() and v.quote.strip()
    ok("...and still recognises a real permission, with a non-empty quote")

    # 6. a value column that resolved but is null no longer counts as an unresolved key
    ext = pd.DataFrame({"code": ["A", "B"], "title": [None, "b"], "level": [1, 2]})
    joined, rep = merge_lookup_safe(pd.DataFrame({"code": ["A", "B"]}), ext, df_key="code",
                                    ext_key="code", value_cols=["title", "level"],
                                    out_prefix="", meta={"snapshot": "t"})
    assert rep["rows_matched"] == 2 and rep["unresolved_keys"] == [], rep
    assert rep["null_by_column"]["title"] == 1, rep
    ok("a resolved key whose first value is null is counted matched, with per-column nulls")

    # 7. a fully-null second value column is reported, not silently shipped
    ext = pd.DataFrame({"code": ["A", "B"], "title": ["a", "b"], "extra": [None, None]})
    _, rep = merge_lookup_safe(pd.DataFrame({"code": ["A", "B"]}), ext, df_key="code",
                               ext_key="code", value_cols=["title", "extra"],
                               out_prefix="", meta={"snapshot": "t"})
    assert rep["all_null_columns"] == ["extra"], rep
    ok("an entirely-null value column is named in the report")

    # 8. a missing key no longer crashes the report builder
    df = pd.DataFrame({"code": ["A", None]})
    _, rep = merge_lookup_safe(df, pd.DataFrame({"code": ["A"], "title": ["a"]}),
                               df_key="code", ext_key="code", value_cols=["title"],
                               out_prefix="", meta={"snapshot": "t"})
    assert rep["rows_unmatched"] == 1, rep
    ok("a missing join key is reported instead of raising TypeError from sorted()")

    # 9. the calendar side is normalized, so string/date/tz-aware calendars still match
    from external_data.join import merge_holiday_flags
    d = pd.DataFrame({"date": ["2019-01-01", "2019-06-05"]})
    for label, hol in [
        ("string dates", pd.DataFrame({"date": ["2019-01-01"]})),
        ("tz-aware", pd.DataFrame({"date": pd.to_datetime(["2019-01-01"]).tz_localize("UTC")})),
        ("python date objects",
         pd.DataFrame({"date": [__import__("datetime").date(2019, 1, 1)]})),
    ]:
        out, rep = merge_holiday_flags(d, hol, df_country=None, df_date="date")
        assert rep["holiday_rows"] == 1, f"{label}: calendar matched nothing ({rep})"
    ok("a date-only calendar matches whether its dates are strings, dates or tz-aware")

    # 10. the date-class reachability route no longer blesses any key
    from external_data.apply import dispatch_route
    assert dispatch_route("holidays", "date")
    try:
        dispatch_route("noaa:precipitation", "date")
    except ValueError as e:
        assert "no route" in str(e)
        ok("an unimplemented date-class source cannot pass reachability and become holidays")
    else:
        raise AssertionError("dispatch_route blessed an unimplemented date-class source")

    # 11. the leakage gate can actually refuse
    from external_data.admit_source import _good_spec, check_leakage
    frame = pd.DataFrame([{"iso3": k, "period": p, "value": 1.0}
                          for k in ("SWE", "NOR") for p in (2020, 2021)])
    assert check_leakage(_good_spec(leakage_rule="lag=1"), frame)
    try:
        check_leakage(_good_spec(leakage_rule="lag=500"), frame)
    except SourceRejected as e:
        assert "exceeds the source" in str(e)
        ok("the leakage gate refuses a lag wider than the source's coverage (was tautological)")
    else:
        raise AssertionError("the leakage gate accepted lag=500 over a 1-year span")

    # 12. ext_key survives the admission round-trip
    from external_data.admit_source import _spec_from_json
    spec, _ = _spec_from_json(PROPOSAL)
    assert hasattr(spec, "ext_key")
    assert _admitted_sources()["cpc:titles"].get("ext_key") == "code"
    ok("ext_key is a real spec field and survives in the registry apply.py reads")
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.json"
        bad.write_text(json.dumps(json.loads(PROPOSAL.read_text()) | {"typo_field": 1}))
        try:
            _spec_from_json(bad)
        except SourceRejected as e:
            assert "unknown proposal field" in str(e)
            ok("an unknown proposal field is a stated refusal, not an uncaught TypeError")
        else:
            raise AssertionError("an unknown proposal field was accepted")


def main() -> int:
    stage_rules_and_vocabulary()
    stage_admission()
    stage_dispatch_join_ledger()
    stage_adversarial()
    stage_verification_regressions()
    print(f"\nselftest_pipeline: all seams passed ({_checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
