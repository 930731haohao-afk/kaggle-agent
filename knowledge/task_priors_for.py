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
  python3 knowledge/task_priors_for.py <competition-slug> --report   # what was withheld
  python3 knowledge/task_priors_for.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
KB_PATH = _HERE / "knowledge_base.json"

# Workspace-derivation suffixes: a repeat/arm/leftover workspace is still solving the base
# competition, so exclusion must key on the base slug.
_WS_SUFFIX = re.compile(r"(\.(repeat|leftover)[\w-]*|-v5-[\w-]+)$")
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


def canonical(comp: str, kb: dict | None = None) -> str:
    """Any spelling of a competition -> the canonical slug evidence items use.

    Handles workspace-derived names (repeat runs, v5 arms), short forms, unicode dashes and
    the playground-series prefix. An unknown competition canonicalizes to its own base name,
    which matches no evidence and therefore excludes nothing — correct, because a new
    competition's facts enter the base carrying its own slug and then match exactly.
    """
    c = (comp or "").translate(_DASHES).strip().lower()
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
            report.append({"entry": e["id"], "action": "dropped_entry",
                           "reason": "all admissible evidence rests on this competition"})
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
                report.append({"entry": e["id"], "action": "trimmed",
                               "reason": "evidence from this competition withheld"})
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
                    out.append(f"  - {_short(r['comp'])}: `{r['winner']}` beat "
                               f"`{r['loser']}`, {r['scores']} "
                               f"({r['horizon_years']}-year horizon).")
            if any(r["comp"] == excl for r in kb["form_race"]["rows"]):
                report.append({"entry": "form_race", "action": "row_withheld",
                               "reason": "this competition's own race row"})
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
            report.append({"entry": op["name"], "action": "trimmed",
                           "reason": "evidence from this competition withheld"})
        out.append("")
    out.append("---")
    out.append("")
    out.append(kb["ledger_md"])
    return "\n".join(out).rstrip() + "\n", report


def render_prereg(kb: dict, comp: str) -> tuple[str, list[dict]]:
    excl = canonical(comp, kb)
    out, report = [], []
    for p in kb["prereg"]:
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
        n_withheld = 0
        for c in p["clauses"]:
            surviving = [e for e in c.get("evidence", []) if canonical(e, kb) != excl]
            if not surviving:
                n_withheld += 1
                report.append({"entry": p["id"], "action": "clause_withheld",
                               "reason": "the clause's only evidence is this competition's "
                                         "own run"})
                continue
            out.append(f"- {c['text']};")
        if n_withheld:
            out.append("")
            out.append(f"*({n_withheld} registered clause(s) withheld for this competition: "
                       f"their only evidence is this competition's own recorded run. The "
                       f"registration still binds — register the applicable prediction for "
                       f"the clauses shown.)*")
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
    ap.add_argument("--report", action="store_true", help="print what was withheld instead")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.comp:
        ap.error("competition slug required (or --selftest)")
    kb = load_kb()
    fn = render_prereg if args.prereg else (render_ops if args.ops else render_priors)
    md, rep = fn(kb, args.comp)
    if args.report:
        kind = "prereg" if args.prereg else ("operator" if args.ops else "task-prior")
        print(f"# {kind} exclusions for {args.comp}\n")
        if not rep:
            print("(nothing withheld)")
        for r in rep:
            print(f"- {r['entry']}: {r['action']} — {r['reason']}")
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
    own_needles = {
        "playground-series-s3e19": ["48.497", "52.073", "10.148", "7.793", "48.24",
                                    "4.56", "20.41", "48.3", "bimodal"],
        "tabular-playground-series-sep-2022": ["23.390", "24.091", "11.348", "11.807",
                                               "11.515", "0.466"],
        "playground-series-s5e1": ["0.12417", "0.15626"],
        "afsis-soil-properties": ["0.49517", "0.44817", "0.444076"],
        "conway-s-reverse-game-of-life": ["0.10875", "98.6"],
        "cat-in-the-dat": ["0.80241", "73.1"],
        "tabular-playground-series-jan-2022": ["4.83", "6.02", "5.46", "4.19", "8.43", "5.80"],
        "playground-series-s3e3": ["0.81901", "0.83292"],
        "playground-series-s3e5": ["0.47191", "0.52687"],
        "playground-series-s4e1": ["0.89653", "0.893235", "0.893650"],
    }
    for c, needles in own_needles.items():
        for fn in (render_priors, render_ops, render_prereg):
            md, _ = fn(kb, c)
            hits = [n for n in needles if n in md]
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
    md, _ = render_prereg(kb, "playground-series-s5e1")
    assert "> 1 year" not in md and "&gt; 1 year" not in md, (
        "the >1-year clause survives for the competition that is its only evidence")
    md, _ = render_prereg(kb, "some-new-competition")
    assert "> 1 year" in md and "≤ 1 year" in md
    print("contract 4: entry-level (afsis/conway) and clause-level (prereg) exclusion hold")

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

    print("task_priors_for selftest: all contracts hold")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
