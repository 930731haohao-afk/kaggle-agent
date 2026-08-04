"""Emit the task-prior library FILTERED for one competition — the Stage 0.5 door.

WHY THIS EXISTS. `knowledge/task_priors.md` is read by Stage 0.5 to answer "what kind of
problem is this?". It is also, almost entirely, a residue of the benchmark runs: 7 of its 8
[TASK-*] entries have Evidence resting on competitions in the 20-competition benchmark, and
some of that evidence names the OTHER agents' results outright --

    TASK-CAT-ONLY ... Evidence: cat-in-the-dat — my-agent private AUC 0.80241 (PR 73.1) vs
    NVIDIA 0.77084 ...; s3e11 — AIDE's only outright win came from a node that accidentally
    enabled native categorical handling.

`run_myagent_headless.sh` forbids reading anything under the other agents' directories, but
that constraint is enforced on DIRECTORIES. This file is the path around it: a re-run of s3e11
reads AIDE's winning mechanism without touching AIDE's workspace, and a re-run of
cat-in-the-dat reads its own private score. SKILL.md's HARD RULE is scoped to experience.md
bullets and explicitly places task priors outside it, and 00_problem_dossier.md has no
exclusion instruction at all.

WHY EXCLUSION IS ENTRY-LEVEL AND NOT CITATION-LEVEL. Each entry is Trigger / Action /
Evidence, and the ACTION is the distilled answer. Deleting only the Evidence line removes the
attribution and leaves the answer:

    TASK-SPECTRAL  Trigger: thousands of ordered numeric columns, few rows.
                   Action : Savitzky-Golay-type derivatives, dimensionality reduction,
                            per-target models, strong regularization.
                   Evidence: afsis-soil-properties ...

Those three moves ARE the afsis solution, learned by solving afsis; on a re-run of afsis a
citation-level filter hands the agent the recipe with the serial number filed off. TASK-STRUCT-OUT
is starker still: its trigger ("a grid or sequence governed by explicit rules, e.g. cellular
automata") fires on exactly one competition in existence.

So: drop any line naming the competition, AND drop the whole entry when its Evidence set
empties as a result. The agent is genuinely weaker on those competitions -- which is the point,
because it had no such capability before it solved them. Entries resting on several
competitions keep their Action, since it stays supported by independent evidence.

The project already reached this conclusion once by hand: on 2026-07-30 a clause was struck
from TASK-TS-FUTURE with the note "a prior library that carries it launders exactly the input
class the isolation protocol excludes". That single instance was fixed; the structural version
was not, until now.

Usage:
  python3 knowledge/task_priors_for.py <competition-slug>          # the filtered library
  python3 knowledge/task_priors_for.py <slug> --report             # what was dropped and why
  python3 knowledge/task_priors_for.py --selftest
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PRIORS = _HERE / "task_priors.md"

# Same alias table the other two doors use. Kept in one place per module deliberately: three
# copies of a slug table is how they drift, so each imports from harness_v2 when it can.
try:
    sys.path.insert(0, str(_HERE.parent / "tree_search"))
    from harness_v2 import comp_aliases as _aliases  # noqa: E402
except Exception:  # noqa: BLE001 - stay usable if the harness is not importable
    _ALIASES = {
        "tabular-playground-series-jan-2022": ["tpsjan22", "jan-2022", "jan2022"],
        "tabular-playground-series-aug-2022": ["tpsaug22", "aug-2022", "aug2022"],
        "tabular-playground-series-sep-2022": ["tpssep22", "sep-2022", "sep2022"],
        "afsis-soil-properties": ["afsis"],
        "conway-s-reverse-game-of-life": ["conway"],
        "cat-in-the-dat": ["citd"],
    }

    def _aliases(comp: str) -> list[str]:
        al = {comp} | set(_ALIASES.get(comp, []))
        al.add(comp.replace("playground-series-", "").replace("tabular-", ""))
        return sorted(al, key=len, reverse=True)

# The other agents are never a legitimate input to this stage, whatever competition is being
# solved. A prior that cites them is laundering the comparison itself.
_OTHER_AGENTS = re.compile(r"\bAIDE\b|\bNVIDIA\b", re.I)

# Every token the library uses to cite a competition. Used to ask "does this surviving evidence
# fragment cite ANY competition?" -- a fragment citing none is a continuation of a dropped item.
_ALL_COMP_TOKENS = sorted({
    "s3e1", "s3e3", "s3e5", "s3e7", "s3e9", "s3e11", "s3e14", "s3e16", "s3e19", "s3e20",
    "s4e1", "s4e11", "s5e1", "s5e10", "s6e1", "s6e2",
    "afsis", "afsis-soil-properties", "cat-in-the-dat", "citd",
    "conway", "conway-s-reverse-game-of-life",
    "tps-aug-2022", "tpsaug22", "aug-2022", "tps-jan-2022", "tpsjan22", "jan-2022",
    "tps-sep-2022", "tpssep22", "sep-2022",
    "spaceship-titanic", "house-prices", "home-data",
}, key=len, reverse=True)


def _names(text: str, aliases: list[str]) -> bool:
    low = text.lower()
    for a in aliases:
        if re.search(rf"(?<![0-9a-z]){re.escape(a.lower())}(?![0-9a-z])", low):
            return True
    return False


def _parse(md: str) -> list[dict]:
    """Split into [{name, header, fields:[(label, text)]}] — one dict per [TASK-*] entry."""
    out, cur, preamble = [], None, []
    for line in md.splitlines():
        h = re.match(r"^##\s+(TASK-[\w-]+)(.*)$", line)
        if h:
            cur = {"name": h.group(1), "header": line, "fields": []}
            out.append(cur)
            continue
        if cur is None:
            preamble.append(line)
            continue
        f = re.match(r"^-\s+\*\*(\w+)\*\*:\s*(.*)$", line)
        if f:
            cur["fields"].append([f.group(1), f.group(2)])
        elif cur["fields"] and line.strip():
            cur["fields"][-1][1] += "\n" + line
        elif line.strip():
            cur["fields"].append(["_free", line])
    return out, "\n".join(preamble)


def filter_priors(md: str, comp: str) -> tuple[str, list[dict]]:
    """Return (filtered markdown, drop report)."""
    entries, preamble = _parse(md)
    aliases = _aliases(comp)
    kept_md, report = [preamble.rstrip()], []

    for e in entries:
        kept_fields, dropped_lines = [], []
        for label, text in e["fields"]:
            # Segment-granular, NOT line-granular. task_priors.md is hard-wrapped, so an
            # evidence item routinely spans three lines: dropping only the line that carries
            # the competition name leaves orphan continuations ("an 80-node search; tree
            # search on this comp: 0.44817 -> 0.444076") that name nothing, look like
            # surviving evidence, and kept TASK-SPECTRAL alive on an afsis run.
            flat = " ".join(ln.strip() for ln in text.split("\n") if ln.strip())
            segs = [s for s in re.split(r"(?<=[.;])\s+", flat) if s.strip()]
            keep = [s for s in segs if not (_names(s, aliases) or _OTHER_AGENTS.search(s))]
            dropped_lines += [s for s in segs if s not in keep]
            if label.lower() == "evidence":
                # An Evidence field survives only if something in it cites a competition OTHER
                # than the one being solved. A fragment citing nothing is a continuation of a
                # dropped item, not independent support.
                if not any(_names(s, _ALL_COMP_TOKENS) for s in keep):
                    keep = []
            if keep:
                kept_fields.append((label, " ".join(keep)))
            elif label.lower() != "evidence":
                # Trigger/Action never survive on their own if they named the competition.
                dropped_lines.append(f"[{label} removed entirely]")

        has_evidence = any(lb.lower() == "evidence" and tx.strip() for lb, tx in kept_fields)
        if not has_evidence:
            # THE ENTRY-LEVEL RULE. Its evidence emptied, so the Action it licenses is
            # supported only by the competition being solved: drop the whole entry.
            report.append({"entry": e["name"], "action": "dropped_entry",
                           "reason": "every Evidence line named this competition (or another "
                                     "agent's result), so the Action rests on it alone",
                           "dropped": dropped_lines})
            continue
        if dropped_lines:
            report.append({"entry": e["name"], "action": "trimmed",
                           "reason": "some evidence named this competition; the Action stays "
                                     "supported by independent evidence",
                           "dropped": dropped_lines})
        kept_md.append("")
        kept_md.append(e["header"])
        for lb, tx in kept_fields:
            kept_md.append(tx if lb == "_free" else f"- **{lb}**: {tx}")

    return "\n".join(kept_md).rstrip() + "\n", report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("comp", nargs="?", help="competition slug, e.g. playground-series-s3e11")
    ap.add_argument("--report", action="store_true", help="print what was dropped, not the library")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.comp:
        ap.error("competition slug required (or --selftest)")
    md, rep = filter_priors(PRIORS.read_text(encoding="utf-8"), args.comp)
    if args.report:
        print(f"# task-prior exclusions for {args.comp}\n")
        if not rep:
            print("(nothing dropped: no prior names this competition or another agent)")
        for r in rep:
            print(f"## {r['entry']} — {r['action'].upper()}")
            print(f"   {r['reason']}")
            for d in r["dropped"][:6]:
                print(f"   - dropped: {d[:150]}")
            print()
        return 0
    print(md)
    return 0


def selftest() -> int:
    md = PRIORS.read_text(encoding="utf-8")
    full, _ = filter_priors(md, "no-such-competition-xyz")
    n_full = full.count("\n## TASK-")

    # afsis: TASK-SPECTRAL rests on afsis alone -> the whole entry must go, Action included
    out, rep = filter_priors(md, "afsis-soil-properties")
    assert "TASK-SPECTRAL" not in out, "TASK-SPECTRAL survived an afsis run"
    assert "Savitzky" not in out, "the afsis ACTION survived — citation-level filtering only"
    assert any(r["entry"] == "TASK-SPECTRAL" and r["action"] == "dropped_entry" for r in rep), rep
    print(f"afsis: TASK-SPECTRAL dropped whole (Action included), {out.count(chr(10) + '## TASK-')}"
          f"/{n_full} entries remain")

    # conway: same shape
    out, rep = filter_priors(md, "conway-s-reverse-game-of-life")
    assert "TASK-STRUCT-OUT" not in out, "TASK-STRUCT-OUT survived a conway run"
    assert "cellular automata" not in out.lower()
    print("conway: TASK-STRUCT-OUT dropped whole")

    # the other agents' results must never appear, whatever competition is being solved
    for comp in ("playground-series-s3e7", "playground-series-s6e2", "no-such-competition-xyz"):
        out, _ = filter_priors(md, comp)
        assert not _OTHER_AGENTS.search(out), f"AIDE/NVIDIA result leaked into {comp}'s priors"
    print("no AIDE/NVIDIA result survives for ANY competition, including unrelated ones")

    # a multi-source entry keeps its Action: TASK-TS-FUTURE rests on 4 competitions
    out, rep = filter_priors(md, "playground-series-s3e19")
    assert "TASK-TS-FUTURE" in out, "a multi-source entry was dropped for one competition"
    assert not _names(out, _aliases("playground-series-s3e19")), "s3e19 still named in its own priors"
    print("s3e19: TASK-TS-FUTURE trimmed but kept (evidence from 3 other competitions survives)")

    # no competition sees its own name anywhere in its filtered library
    comps = ["playground-series-s3e1", "playground-series-s3e11", "playground-series-s3e19",
             "playground-series-s5e1", "playground-series-s5e10", "playground-series-s6e1",
             "cat-in-the-dat", "afsis-soil-properties", "conway-s-reverse-game-of-life",
             "tabular-playground-series-jan-2022", "tabular-playground-series-sep-2022",
             "playground-series-s3e14", "playground-series-s3e16"]
    for c in comps:
        out, _ = filter_priors(md, c)
        assert not _names(out, _aliases(c)), f"{c} is still named in its own filtered priors"
    print(f"none of {len(comps)} benchmark competitions is named in its own filtered library")

    # sibling slugs must NOT be over-excluded (s3e1 vs s3e19/s3e11/s3e14/s3e16)
    out, _ = filter_priors(md, "playground-series-s3e1")
    assert "s3e19" in out or "s3e11" in out, "sibling competitions were over-excluded from s3e1"
    print("sibling slugs survive (s3e1 does not exclude s3e19/s3e11) — no over-exclusion")

    print("task_priors_for selftest: all sections passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
