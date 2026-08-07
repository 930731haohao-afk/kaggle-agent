"""tree_search/test_harness_v4_injection.py — Phase J J-2 unit tests for the v4 external-idea
injection hook (harness_v4). DELIBERATELY tiny / parse-level: it only reads knowledge/*.md and
exercises the pooling+dedup logic — it trains NO model, spawns NO subprocess, and finishes in
well under a second, so it can run while the s5e10 v3 search is using the CPU.

Run:  uv run python3 tree_search/test_harness_v4_injection.py
(also importable by pytest — every check is a `test_*` function; the __main__ block is a
zero-dependency runner that prints PASS/FAIL and exits nonzero on any failure.)

Assertions (mapping to the J-2 spec):
  (a) idea_bank.md's [EXT] entries parse (all 24, with ids/bodies/haystacks).
  (b) an [EXT]-ONLY idea absent from [INT] (adversarial validation, EXT-14) surfaces as an
      EXT-tagged prior under a corresponding comp_meta — and no [INT] prior mentions it.
  (c) an [EXT] idea already proven by [INT] (OptimizedRounder/QWK, EXT-21) does NOT re-fire in
      mode='ext' (dedup) even though it WOULD have matched (shown via respect_dedup=False).
  (d) mode='off' injects zero [EXT] and reproduces harness_v3.suggest_priors byte-for-byte.
plus two guardrails: the dedup-suppress set is exactly the 13 documented ids, and an invalid
mode raises.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import harness_v3 as hv3   # noqa: E402  -- for the mode='off' == stage-4-baseline equality check
import harness_v4 as hv4
import os as _os
# the live idea bank is archived (2026-08-10, unguarded per-comp results);
# these tests exercise the PARSER against the archived historical file
hv4.DEFAULT_IDEA_BANK_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    'knowledge', 'archive_pre_structured', 'idea_bank.md')   # noqa: E402

# comp_meta a distribution-shift / adversarial-validation scenario would carry (any metric;
# adversarial validation triggers on the SCENARIO, per idea_bank EXT-14's "任何指標").
CM_ADVERSARIAL = {"metric": "auc", "tags": ["adversarial", "distribution_shift"]}
# a QWK ordinal comp — its ordinal/threshold knowledge is fully [INT]-proven (EXT-21 suppressed).
CM_QWK = {"metric": "qwk", "tags": ["序數目標"]}

# the 13 ids idea_bank.md's tail table marks "已被 [INT] 實證 ... 避免重複觸發"
EXPECTED_SUPPRESS = {"EXT-01", "EXT-02", "EXT-03", "EXT-05", "EXT-06", "EXT-07", "EXT-08",
                     "EXT-10", "EXT-13", "EXT-15", "EXT-16", "EXT-21", "EXT-22"}


def test_a_ext_entries_parse():
    """(a) All 24 [EXT] entries parse, each with an id, a non-empty body and haystack, a
    category, and EXT-14 is the adversarial-validation entry."""
    entries = hv4.parse_idea_bank()
    ids = [e["id"] for e in entries]
    assert len(entries) == 24, f"expected 24 [EXT] entries, parsed {len(entries)}"
    assert ids == [f"EXT-{i:02d}" for i in range(1, 25)], f"ids not 01..24 in order: {ids}"
    for e in entries:
        assert e["lines"], f"{e['id']} parsed with no body bullets"
        assert e["haystack"], f"{e['id']} parsed with empty haystack"
        assert e["category"], f"{e['id']} has no parent category"
    e14 = next(e for e in entries if e["id"] == "EXT-14")
    assert "adversarial" in e14["haystack"] and "對抗驗證" in e14["haystack"], \
        "EXT-14 haystack missing adversarial-validation tokens"
    return f"parsed {len(entries)} [EXT] entries EXT-01..EXT-24; EXT-14 = adversarial validation"


def test_b_ext_only_idea_surfaces():
    """(b) EXT-14 (adversarial validation — [EXT]-unique, nothing in experience.md) surfaces as
    an EXT-tagged prior under an adversarial/shift comp_meta, and NO [INT] prior mentions it."""
    priors = hv4.suggest_priors_v4(CM_ADVERSARIAL, mode="ext", exclude_self=False)
    ext = hv4.priors_by_provenance(priors, "EXT")
    ext_ids = {p["source"] for p in ext}
    assert "EXT-14" in ext_ids, f"EXT-14 did not surface under {CM_ADVERSARIAL}; got {sorted(ext_ids)}"
    p14 = next(p for p in ext if p["source"] == "EXT-14")
    assert p14["provenance"] == "EXT" and "EXT 文獻先驗" in p14["text"], \
        "EXT-14 prior not provenance-tagged as external"
    # genuinely [EXT]-unique: the [INT] channel must not already carry an adversarial lesson
    int_texts = " ".join(p["text"] for p in hv4.priors_by_provenance(priors, "INT"))
    assert "對抗" not in int_texts and "adversarial" not in int_texts.lower(), \
        "adversarial validation unexpectedly present in [INT] — not an [EXT]-unique idea"
    return f"EXT-14 surfaced as EXT prior (source={p14['source']}); absent from {len(hv4.priors_by_provenance(priors,'INT'))} [INT] priors"


def test_c_int_proven_ext_is_deduped():
    """(c) EXT-21 (OptimizedRounder/QWK) is [INT]-proven → suppressed. It WOULD match metric
    qwk (shown with dedup off), but mode='ext' excludes it; the idea is still represented by
    the [INT] QWK priors."""
    suppress = hv4.parse_dedup_suppress_ids()
    assert "EXT-21" in suppress, "EXT-21 not in the dedup-suppress set"
    # would-match check: with dedup OFF, EXT-21 matches qwk
    raw = {p["source"] for p in hv4.suggest_ext_priors(CM_QWK, respect_dedup=False)}
    assert "EXT-21" in raw, f"EXT-21 should match metric=qwk with dedup off; got {sorted(raw)}"
    # with dedup ON (mode='ext'), EXT-21 must NOT re-fire
    priors = hv4.suggest_priors_v4(CM_QWK, mode="ext", exclude_self=False)
    ext_ids = {p["source"] for p in hv4.priors_by_provenance(priors, "EXT")}
    assert "EXT-21" not in ext_ids, f"EXT-21 re-fired despite being [INT]-proven; ext={sorted(ext_ids)}"
    # the idea is not lost — [INT] carries the QWK/ordinal knowledge
    assert hv4.priors_by_provenance(priors, "INT"), "no [INT] priors for a QWK comp — idea lost"
    return f"EXT-21 matched with dedup off but suppressed in mode='ext'; {len(hv4.priors_by_provenance(priors,'INT'))} [INT] QWK priors present"


def test_d_off_mode_injects_no_ext():
    """(d) mode='off' injects zero [EXT] and reproduces harness_v3.suggest_priors exactly."""
    off = hv4.suggest_priors_v4(CM_ADVERSARIAL, mode="off", exclude_self=False)
    assert not hv4.priors_by_provenance(off, "EXT"), "mode='off' leaked [EXT] priors"
    assert all(p["provenance"] == "INT" for p in off), "mode='off' produced non-[INT] priors"
    # exclude_self=False: this test is about v4-vs-v3 channel equality, not about the
    # self-evidence filter added 2026-07-30 (these synthetic comp_meta dicts name no competition).
    baseline = hv3.suggest_priors(CM_ADVERSARIAL, exclude_self=False)
    assert hv4.prior_texts(off) == baseline, \
        "mode='off' flattened texts diverge from harness_v3.suggest_priors (stage-4 baseline perturbed)"
    # and mode='ext' is a strict superset here (adds EXT on top of the same INT)
    ext = hv4.suggest_priors_v4(CM_ADVERSARIAL, mode="ext", exclude_self=False)
    assert hv4.prior_texts(hv4.priors_by_provenance(ext, "INT")) == baseline, \
        "mode='ext' altered the [INT] channel"
    return f"mode='off' == harness_v3.suggest_priors ({len(baseline)} [INT] lines), 0 [EXT]"


def test_e_guardrails():
    """dedup-suppress set is exactly the 13 documented ids; invalid mode raises."""
    assert hv4.parse_dedup_suppress_ids() == EXPECTED_SUPPRESS, \
        f"suppress set mismatch: {sorted(hv4.parse_dedup_suppress_ids())}"
    raised = False
    try:
        hv4.suggest_priors_v4(CM_QWK, mode="bogus", exclude_self=False)
    except ValueError:
        raised = True
    assert raised, "invalid mode did not raise ValueError"
    return f"suppress set == {len(EXPECTED_SUPPRESS)} documented ids; invalid mode raises ValueError"


_TESTS = [
    ("a", "[EXT] entries parse", test_a_ext_entries_parse),
    ("b", "[EXT]-only idea (EXT-14 adversarial) surfaces", test_b_ext_only_idea_surfaces),
    ("c", "[INT]-proven [EXT] (EXT-21 QWK) deduped in mode='ext'", test_c_int_proven_ext_is_deduped),
    ("d", "mode='off' injects no [EXT], == stage-4 baseline", test_d_off_mode_injects_no_ext),
    ("e", "guardrails (suppress set + mode validation)", test_e_guardrails),
]


if __name__ == "__main__":
    t0 = time.time()
    failures = 0
    for tag, name, fn in _TESTS:
        try:
            detail = fn()
            print(f"PASS ({tag}) {name}\n        -> {detail}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL ({tag}) {name}\n        -> {exc}")
        except Exception as exc:  # noqa: BLE001 -- surface any parse/IO error as a failure
            failures += 1
            print(f"ERROR ({tag}) {name}\n        -> {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed in {dt*1000:.0f} ms")
    sys.exit(1 if failures else 0)
