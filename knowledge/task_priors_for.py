"""Render the knowledge library FILTERED for one competition — the only door Stage 0.5 may use.

WHY A RENDERER AND NOT A REDACTOR. Four adversarial verification rounds attacked the previous
design (per-competition scrubbing of the prose files) and each round found a channel the
scrubber could not see: competition names, then bare numbers, then hard-wrapped continuation
lines, then positional aggregates, then verbatim headings, then this module's own docstring.
Redaction of natural language is an unwinnable enumeration game, because information survives
in meaning. The same lesson produced the fail-closed rules gate; this is its knowledge-side
twin:

  - the source of truth is `knowledge/knowledge_base.json`: structured records whose prose
    fields are COMPETITION-FREE by contract (enforced by --selftest), and whose every
    competition-specific fact is an evidence item carrying its comp slug;
  - exclusion is exact set arithmetic on that slug — no pattern matching on prose;
  - every view is REGENERATED from the surviving facts, so a withheld fact leaves no marker,
    no heading, no paragraph shape, and no position for an attacker to read;
  - cross-agent evidence (other lanes' results) is never rendered for any competition;
  - an entry whose admissible evidence empties for a competition is dropped whole — the
    action line IS the distilled answer — unless the entry is marked generic;
  - computed aggregates (the cross-competition form record) are derived at render time from
    surviving rows only, so a member competition never sees a summary its own row shaped.

Usage:
  python3 knowledge/task_priors_for.py <competition-slug>            # task priors
  python3 knowledge/task_priors_for.py <competition-slug> --ops      # operator vocabulary
  python3 knowledge/task_priors_for.py <competition-slug> --prereg   # pre-registrations
  python3 knowledge/task_priors_for.py <competition-slug> --report   # AUDITORS ONLY
  python3 knowledge/task_priors_for.py --selftest                    # AUDITORS ONLY
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
KB_PATH = _HERE / "knowledge_base.json"

# Workspace-derivation suffixes: a repeat/arm/leftover workspace is still solving the base
# competition, so exclusion must key on the base slug.
_WS_SUFFIX = re.compile(r"(\.(repeat|leftover|v5arms)[\w-]*|-v5-[\w-]+)$")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), ord("-"))

_SHORT_ALIASES = {
    "tabular-playground-series-jan-2022": ["tpsjan22", "jan-2022", "jan2022", "tps-jan-2022"],
    "tabular-playground-series-aug-2022": ["tpsaug22", "aug-2022", "aug2022", "tps-aug-2022"],
    "tabular-playground-series-sep-2022": ["tpssep22", "sep-2022", "sep2022", "tps-sep-2022"],
    "afsis-soil-properties": ["afsis"],
    "conway-s-reverse-game-of-life": ["conway"],
    "cat-in-the-dat": ["citd"],
}


def load_kb() -> dict:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _known_comps(kb: dict) -> set:
    out = set()
    for e in kb.get("task_priors", []):
        out.update(ev["comp"] for ev in e.get("evidence", []))
    for op in kb.get("operators", []):
        out.update(ev["comp"] for ev in op.get("evidence", []))
    for r in kb.get("form_race", {}).get("rows", []):
        out.add(r["comp"])
    for p in kb.get("prereg", []):
        for c in p.get("clauses", []):
            out.update(c.get("evidence", []))
    return out


def _all_evidence(kb: dict) -> list:
    """Every evidence item in the base, plus the form-race rows rendered as evidence-like
    dicts, so a caller can ask "what does the base record about competition X?"."""
    out = []
    for e in kb.get("task_priors", []):
        out += e.get("evidence", [])
    for op in kb.get("operators", []):
        out += op.get("evidence", [])
    for r in kb.get("form_race", {}).get("rows", []):
        out.append({"comp": r["comp"], "text": " ".join(str(v) for v in r.values())})
    return out


def canonical(comp: str, kb: dict | None = None) -> str:
    """Any spelling of a competition -> the canonical slug evidence items use.

    Handles workspace-derived names (repeat runs, v5 arms), short forms, unicode dashes and
    the playground-series prefix. An unknown competition canonicalizes to its own base name,
    which matches no evidence and therefore excludes nothing — correct, because a new
    competition's facts enter the base carrying its own slug and then match exactly.
    """
    c = (comp or "").translate(_DASHES).strip().lower()
    c = c.replace("_", "-")                      # underscore spellings of known slugs
    if "/" in c:                                  # path forms: competitions/<slug>
        c = c.rstrip("/").rsplit("/", 1)[-1]
    prev = None
    while c and c != prev:
        prev = c
        c = _WS_SUFFIX.sub("", c)
    comps = _known_comps(kb) if kb else set(_SHORT_ALIASES)
    for canon in comps | set(_SHORT_ALIASES):
        if c == canon:
            return canon
        short = canon.replace("playground-series-", "").replace("tabular-", "")
        if c == short or c in _SHORT_ALIASES.get(canon, []):
            return canon
        if c == f"playground-series-{canon}" or f"playground-series-{c}" == canon:
            return canon
    # AMBIGUITY REFUSAL. A name that EXTENDS a known slug but resolved to nothing is far more
    # likely an unanticipated workspace derivation than a new competition -- and failing open
    # here serves the base competition its own facts with no warning (the .v5arms case,
    # 2026-08-07 round-5). A genuinely new competition shares no known slug as a prefix.
    for canon in comps | set(_SHORT_ALIASES):
        short = canon.replace("playground-series-", "").replace("tabular-", "")
        for stem in (canon, short):
            # a TRUNCATION of a known slug is as ambiguous as an extension, and failing open
            # served the truncated workspace the full library (2026-08-10 round-7)
            # The digit-sibling rule applies symmetrically: playground-series-s3e1 is a real
            # slug that happens to be a PREFIX of s3e11/s3e19, so a digit at the split point
            # means numbered siblings, not truncation (2026-08-10 round-7).
            if (stem and len(c) >= 6 and stem.startswith(c) and c != stem
                    and not stem[len(c)].isdigit()):
                raise ValueError(
                    f"competition spelling {comp!r} is a truncation of the known slug "
                    f"{canon!r}. Refusing to guess: pass the full slug.")
            if stem and c.startswith(stem) and c != stem:
                # A DIGIT continuation is a numbered sibling competition (s4e1 -> s4e11,
                # s5e1 -> s5e10), not a workspace derivation: refusing it crashed the s4e11
                # lane's only knowledge door (2026-08-10 round-6). Derivations always start
                # with a separator, which the branches above already recognize.
                if c[len(stem)].isdigit():
                    continue
                raise ValueError(
                    f"competition spelling {comp!r} extends the known slug {canon!r} but "
                    f"matches no recognized derivation pattern. Refusing to guess: if this "
                    f"is a workspace of {canon!r}, add its suffix to _WS_SUFFIX; if it is a "
                    f"new competition, rename it so no known slug is its prefix.")
    return c


def _admissible(evidence: list, excl: str) -> list:
    return [e for e in evidence
            if not e.get("cross_agent") and e.get("comp") != excl]


def _short(comp: str) -> str:
    return comp.replace("playground-series-", "").replace("tabular-", "")


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------
def render_priors(kb: dict, comp: str) -> tuple[str, list[dict]]:
    excl = canonical(comp, kb)
    out, report = [kb["priors_header"], ""], []

    for e in kb["task_priors"]:
        adm = _admissible(e.get("evidence", []), excl)
        had_own = any(ev.get("comp") == excl for ev in e.get("evidence", []))
        only_cross = (not adm and bool(e.get("evidence"))
                      and all(ev.get("cross_agent") for ev in e["evidence"]))
        if not adm and not e.get("generic") and not only_cross:
            report.append({"action": "dropped_entry"})
            continue
        out.append(f"## {e['id']} — {e['title']}")
        out.append(f"- **Trigger**: {e['trigger']}")
        out.append(f"- **Action**: {e['action']}")
        for note in e.get("generic_notes", []):
            out.append(f"- {note}")
        if adm:
            for ev in adm:
                out.append(f"- **Evidence** ({_short(ev['comp'])}): {ev['text']}")
            if had_own:
                report.append({"action": "trimmed"})
        elif only_cross:
            out.append("- **Evidence**: withheld — the recorded evidence cited another "
                       "lane's run, inadmissible under the isolation protocol. Treat the "
                       "action as an unvalidated prior.")
        else:
            out.append("- **Evidence**: none of ours yet — [GEN], unvalidated.")

        if e["id"] == "TASK-TS-FUTURE":
            rows = [r for r in kb["form_race"]["rows"] if r["comp"] != excl]
            if len(rows) >= 2:
                n = len(rows)
                beat = sum(1 for r in rows if r.get("external_beat_baseline"))
                cv_right = sum(1 for r in rows if r.get("cv_pick") == r.get("winner"))
                out.append(f"- **Cross-competition form record** (measured on {n} "
                           f"competitions; any evidence from the competition being solved "
                           f"is excluded): external arms beat their no-external baselines "
                           f"in {beat}/{n}; local CV picked the eventual form winner in "
                           f"{cv_right}/{n} — race both arms, always.")
                for r in rows:
                    # horizon_years is DELIBERATELY not rendered: it is the selector
                    # variable of the open registration, and the reader knows its own
                    # horizon from its data -- printing the others completes the map
                    # (2026-08-10 round-7).
                    out.append(f"  - {_short(r['comp'])}: `{r['winner']}` beat "
                               f"`{r['loser']}`, {r['scores']}.")
            if any(r["comp"] == excl for r in kb["form_race"]["rows"]):
                report.append({"action": "row_withheld"})
        out.append("")

    out.append("---")
    out.append("")
    out.append(kb["whitelist_md"])
    return "\n".join(out).rstrip() + "\n", report


def render_ops(kb: dict, comp: str) -> tuple[str, list[dict]]:
    excl = canonical(comp, kb)
    out, report = [kb["operators_header"], "", "---", "", "## Operator set", ""], []
    for op in kb["operators"]:
        out.append(f"### `{op['name']}`")
        out.append(op["doc"])
        adm = _admissible(op.get("evidence", []), excl)
        if adm:
            out.append("")
            for ev in adm:
                out.append(f"- **Evidence** ({_short(ev['comp'])}): {ev['text']}")
        if any(ev.get("comp") == excl for ev in op.get("evidence", [])):
            report.append({"action": "trimmed"})
        out.append("")
    out.append("---")
    out.append("")
    out.append(kb["ledger_md"])
    return "\n".join(out).rstrip() + "\n", report


def render_prereg(kb: dict, comp: str) -> tuple[str, list[dict]]:
    excl = canonical(comp, kb)
    out, report = [], []
    for p in kb["prereg"]:
        # A registration renders WHOLE or NOT AT ALL, and "not at all" is decided by
        # membership, not by survival. Round 7 asked whether a clause's evidence list
        # EMPTIED after exclusion; a clause carrying two evidence competitions therefore
        # stayed visible for BOTH of them, and the clause text states the private-
        # leaderboard winner those competitions recorded -- each was served its own answer
        # (2026-08-10 round-8). The right question is whether the reader is cited at all:
        # its own result cannot be a premise of a hypothesis it is about to be tested on.
        if any(canonical(e, kb) == excl
               for c in p["clauses"] for e in c.get("evidence", [])):
            report.append({"action": "registration_withheld"})
            # Render NOTHING -- not even a notice. Any marker is a signal. Emptiness is a
            # neutral, ordinary state: no other surface may assert that a registration
            # exists (selftest CONTRACT 7), --report is audit-gated, and Stage 0.5 reads an
            # empty view as "no open registration applies to you".
            continue
        out.append(f"# Pre-registration: {p['id']}")
        out.append("")
        out.append(f"**Status: {p['status']}**")
        out.append("")
        out.append("## Why this registration exists")
        out.append("")
        out.append(p["why"])
        out.append("")
        out.append(f"## Hypothesis ({p['id']})")
        out.append("")
        out.append(p["hypothesis_intro"])
        out.append("")
        # Every clause renders: the reader is cited by none of them, so the whole
        # registration is other competitions' evidence. There is no partial-render path any
        # more -- a clause set with a hole is a complement puzzle with one piece missing
        # (2026-08-07 round-5, 2026-08-10 round-7), and the only safe partial is silence.
        for text in [c["text"] for c in p["clauses"]]:
            out.append(f"- {text};")
        out.append("")
        out.append("## Protocol (binding)")
        out.append("")
        out.append(p["protocol"])
        out.append("")
        out.append("## Pre-committed consequences")
        out.append("")
        out.append(p["consequences"])
        out.append("")
        out.append("## Log")
        out.append("")
        for line in p.get("log", []):
            out.append(f"- {line}")
    return "\n".join(out).rstrip() + "\n", report


# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("comp", nargs="?", help="competition slug, any spelling")
    ap.add_argument("--ops", action="store_true", help="operator vocabulary view")
    ap.add_argument("--prereg", action="store_true", help="pre-registrations view")
    ap.add_argument("--report", action="store_true",
                    help="AUDITORS ONLY (needs KAGGLE_KB_AUDIT=1): how many items were "
                         "withheld for this competition")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        # AUDITOR-GATED, like --report. The contracts assert WHICH entries are withheld and
        # what they say -- the three form-race member slugs, that afsis's withheld entry is
        # TASK-SPECTRAL and mentions Savitzky, that conway's is TASK-STRUCT-OUT, the
        # H-HORIZON thresholds whose lanes render as silence. Round 8 removed a hardcoded
        # per-competition score table from this very file, recording the reason as "the
        # filter itself becoming the door it exists to close"; it fixed contract 2 by
        # deriving its needles and left 3, 4, 5 and 8 hardcoded (round 11). Until those are
        # derived too, the tool that verifies the withholding may not be run by the party it
        # withholds from.
        if os.environ.get("KAGGLE_KB_AUDIT") != "1":
            print("--selftest is for auditors: its contracts name the entries they verify "
                  "are withheld. Re-run with KAGGLE_KB_AUDIT=1.", file=sys.stderr)
            return 2
        return selftest()
    if not args.comp:
        ap.error("competition slug required (or --selftest)")
    kb = load_kb()
    fn = render_prereg if args.prereg else (render_ops if args.ops else render_priors)
    md, rep = fn(kb, args.comp)
    if args.report:
        # A COUNTING ORACLE, not a view. "1 item(s) withheld" tells the reader that a fact
        # about itself was removed -- and once it knows something is missing it can reason
        # about the complement, which is the whole channel rounds 4-8 kept re-opening. The
        # count is genuinely useful to a human auditor diffing views against the base, so it
        # stays behind an env gate rather than being deleted (2026-08-10 round-8).
        if os.environ.get("KAGGLE_KB_AUDIT") != "1":
            print("--report is an auditor's tool: it discloses that facts about this "
                  "competition were withheld, which is itself a deduction channel. "
                  "Re-run with KAGGLE_KB_AUDIT=1 if you are auditing, not solving.",
                  file=sys.stderr)
            return 2
        kind = "prereg" if args.prereg else ("operator" if args.ops else "task-prior")
        # COUNTS ONLY. Naming the withheld entry plus its reason was a deduction oracle: an
        # entry name under a known hypothesis reconstructs the withheld content
        # (2026-08-07 round-5). Auditors diff the views against knowledge_base.json directly.
        print(f"# {kind} exclusions for {args.comp}\n")
        print(f"{len(rep)} item(s) withheld for this competition." if rep
              else "(nothing withheld)")
        return 0
    print(md)
    return 0


# ---------------------------------------------------------------------------
def selftest() -> int:
    kb = load_kb()
    known = sorted(_known_comps(kb))
    agents = re.compile(r"\bAIDE\b|\bNVIDIA\b", re.I)

    # --- CONTRACT 1: all prose fields are competition-free and agent-free ---------------
    def prose_fields():
        yield "priors_header", kb["priors_header"]
        yield "whitelist_md", kb["whitelist_md"]
        yield "operators_header", kb["operators_header"]
        yield "ledger_md", kb["ledger_md"]
        for e in kb["task_priors"]:
            yield f"{e['id']}.title", e["title"]
            yield f"{e['id']}.trigger", e["trigger"]
            yield f"{e['id']}.action", e["action"]
            for i, n in enumerate(e.get("generic_notes", [])):
                yield f"{e['id']}.note{i}", n
        for op in kb["operators"]:
            yield f"op.{op['name']}.doc", op["doc"]
        for p in kb["prereg"]:
            for f in ("status", "why", "hypothesis_intro", "protocol", "consequences"):
                yield f"{p['id']}.{f}", p[f]
            for c in p["clauses"]:
                yield f"{p['id']}.clause", c["text"]
            for i, line in enumerate(p.get("log", [])):
                yield f"{p['id']}.log{i}", line
        yield "_schema", kb.get("_schema", "")
        yield "form_race._purpose", kb.get("form_race", {}).get("_purpose", "")

    tokens = set()
    for c in known:
        tokens.add(c)
        tokens.add(_short(c))
        tokens.update(_SHORT_ALIASES.get(c, []))
    bad = []
    n_fields = 0
    for name, text in prose_fields():
        n_fields += 1
        low = str(text).translate(_DASHES).lower()
        for tok in tokens:
            if re.search(rf"(?<![0-9a-z]){re.escape(tok)}(?![0-9a-z])", low):
                bad.append(f"{name} names {tok}")
        if agents.search(str(text)):
            bad.append(f"{name} names another agent")
    assert not bad, "prose fields must be competition-free:\n  " + "\n  ".join(bad[:10])
    print(f"contract 1: every prose field is competition-free and agent-free "
          f"({n_fields} fields x {len(tokens)} tokens)")

    # --- CONTRACT 2: no competition's own facts survive its view -----------------------
    # The needles are DERIVED from the base, never listed here. A hardcoded table of
    # per-competition scores made this very file a leak: it sits in knowledge/, a lane can
    # read it, and the table named ten competitions next to their own numbers -- the filter
    # itself becoming the door it exists to close (2026-08-10 round-8, found by
    # benchmark_infra/verify_clean_slate.py). Deriving is also strictly stronger: needles
    # cannot go stale when the base grows.
    _num = re.compile(r"\d+\.\d{2,}")
    by_comp: dict[str, set] = {}
    for ev in _all_evidence(kb):
        c = canonical(ev.get("comp", ""), kb)
        by_comp.setdefault(c, set()).update(_num.findall(ev.get("text", "")))
    # A needle is a number ONLY this competition's evidence carries. A figure two
    # competitions happen to share (0.011 turns up in two dispersion series) is nobody's
    # own fact and matching on it would fail the contract for a coincidence.
    own_needles = {c: sorted(ns - set().union(*(v for k, v in by_comp.items() if k != c)))
                   for c, ns in by_comp.items()}
    own_needles = {c: ns for c, ns in own_needles.items() if ns}
    assert len(own_needles) >= 8, (
        f"fixture drift: only {len(own_needles)} competitions carry numeric evidence")
    for c, needles in own_needles.items():
        for fn in (render_priors, render_ops, render_prereg):
            md, _ = fn(kb, c)
            # whole-number match: "0.01" is a substring of another competition's "0.011"
            hits = [n for n in needles
                    if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", md)]
            assert not hits, f"{fn.__name__}({c}) serves its own facts: {hits}"
            low = md.translate(_DASHES).lower()
            for tok in {c, _short(c), *_SHORT_ALIASES.get(c, [])}:
                assert not re.search(rf"(?<![0-9a-z]){re.escape(tok)}(?![0-9a-z])", low), \
                    f"{fn.__name__}({c}) names the competition ({tok})"
    print(f"contract 2: no competition's own facts or name survive any of its 3 views "
          f"({len(own_needles)} comps checked)")

    # --- CONTRACT 3: positional aggregates exclude the member --------------------------
    for c in ("playground-series-s3e19", "tabular-playground-series-sep-2022",
              "playground-series-s5e1"):
        md, _ = render_priors(kb, c)
        assert "measured on 2 competitions" in md, (
            f"{c}: the form record must be recomputed over the 2 OTHER competitions")
    md, _ = render_priors(kb, "no-such-competition")
    assert "measured on 3 competitions" in md
    print("contract 3: the form-record aggregate is computed from surviving rows only")

    # --- CONTRACT 4: entry-level and clause-level exclusion ----------------------------
    md, _rep = render_priors(kb, "afsis-soil-properties")
    assert "TASK-SPECTRAL" not in md and "Savitzky" not in md
    md, _ = render_priors(kb, "conway-s-reverse-game-of-life")
    assert "TASK-STRUCT-OUT" not in md and "cellular automata" not in md.lower()
    # Registration exclusion is by CITATION, not by survival: a clause with two evidence
    # competitions used to stay visible for both of them (2026-08-10 round-8).
    cited = {canonical(e, kb) for p in kb["prereg"] for c in p["clauses"]
             for e in c.get("evidence", [])}
    assert len(cited) >= 2, "fixture drift: registrations cite fewer than 2 competitions"
    for c in sorted(cited):
        md, rep = render_prereg(kb, c)
        assert md.strip() == "", (
            f"{c} is cited as evidence for a registered clause, so the registration states "
            f"its own recorded outcome; it rendered {len(md)} bytes")
        assert rep, "the withheld registration left no audit record"
    md, _ = render_prereg(kb, "some-new-competition")
    assert "> 1 year" in md and "≤ 1 year" in md
    print("contract 4: entry-level (afsis/conway) and citation-level (prereg) exclusion hold")

    # --- CONTRACT 5: nothing else lost ------------------------------------------------
    md, _ = render_priors(kb, "no-such-competition")
    assert md.count("\n## TASK-") + md.startswith("## TASK-") >= len(kb["task_priors"]) - 1
    n_entries = sum(1 for ln in md.splitlines() if ln.startswith("## TASK-"))
    assert n_entries == len(kb["task_priors"]), (n_entries, len(kb["task_priors"]))
    n_wl = sum(1 for ln in kb["whitelist_md"].splitlines() if ln.startswith("| "))
    assert sum(1 for ln in md.splitlines() if ln.startswith("| ")) >= n_wl
    md, _ = render_priors(kb, "playground-series-s3e1")
    assert "TASK-TS-FUTURE" in md and "s3e19" in md, "sibling slugs over-excluded"
    md, _ = render_ops(kb, "no-such-competition")
    for op in kb["operators"]:
        assert f"### `{op['name']}`" in md, f"operator {op['name']} missing from the view"
    print("contract 5: unrelated competitions lose nothing; the vocabulary stays complete; "
          "siblings are not over-excluded")

    # --- CONTRACT 6: the renderer never emits redaction markers ------------------------
    for c in ("playground-series-s3e19", "afsis-soil-properties", "no-such-competition"):
        md, _ = render_priors(kb, c)
        assert "[withheld" not in md, "redaction markers are the scrub approach's signature"
    print("contract 6: views are regenerated, not redacted (no [withheld markers)")

    # --- CONTRACT 7: no doc pointers in any rendered view ------------------------------
    for c in ("playground-series-s6e1", "playground-series-s4e1", "no-such-comp"):
        for fn in (render_priors, render_ops, render_prereg):
            md, _ = fn(kb, c)
            assert "docs/" not in md, (
                f"{fn.__name__}({c}) points at a repo doc -- pointer indirection defeats "
                f"set-arithmetic exclusion (2026-08-07 round-5)")
    print("contract 7: rendered views contain no doc pointers")

    # --- CONTRACT 8: silence is neutral -------------------------------------------------
    # An empty --prereg is only safe if nothing else a run reads asserts that a registration
    # exists: "a selector is under open pre-registration" + an empty view = the marker the
    # silence was meant to erase (2026-08-10 round-8).
    # EXISTENTIAL claims only. "if --prereg renders one, it binds you" is symmetric and
    # tells the reader nothing; "a selector IS under open pre-registration" does.
    advert = re.compile(r"open pre-?registration|under (an?\s+)?open pre-?regist|"
                        r"pre-?registered test|a registration (is|exists)", re.I)
    for c in sorted(cited):
        for fn in (render_priors, render_ops):
            md, _ = fn(kb, c)
            hit = advert.search(md)
            assert not hit, (
                f"{fn.__name__}({c}) advertises a registration ({hit.group(0)!r}) whose "
                f"view renders empty for this reader -- absence becomes evidence")
    print("contract 8: no view advertises a registration that another view withholds")

    print("task_priors_for selftest: all contracts hold")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
