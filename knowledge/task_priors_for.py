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


def _parse(md: str) -> tuple[list[dict], str, list[dict]]:
    """Split the file into TASK entries, the preamble, and every OTHER section.

    Returns (task_entries, preamble, other_sections). The third value exists because a
    `##` heading that is not a TASK entry -- "## External-data whitelist (Stage 0.5 / Plan C)"
    and its 7-row source table -- used to be absorbed as trailing free text into whichever
    TASK entry preceded it, and was therefore deleted whenever THAT entry was dropped. Result:
    7 whitelist rows in, 0 out, for all 20 competitions, while 00_problem_dossier.md step 6
    instructs the agent to pick external sources from exactly that table in the filtered
    output. A section this filter does not understand must pass through untouched, never be
    annexed by a neighbour (2026-08-07 re-verification).
    """
    tasks, others, cur, preamble = [], [], None, []
    for line in md.splitlines():
        h2 = re.match(r"^##\s+(.*)$", line)
        if h2:
            name = h2.group(1).strip()
            if re.match(r"^TASK-[\w-]+", name):
                cur = {"name": name.split()[0], "header": line, "fields": []}
                tasks.append(cur)
            else:
                cur = {"header": line, "lines": []}
                others.append(cur)
            continue
        if cur is None:
            preamble.append(line)
            continue
        if "lines" in cur:                       # a non-TASK section: verbatim, no filtering
            cur["lines"].append(line)
            continue
        f = re.match(r"^-\s+\*\*(\w+)\*\*:\s*(.*)$", line)
        if f:
            cur["fields"].append([f.group(1), f.group(2)])
        elif cur["fields"] and line.strip():
            cur["fields"][-1][1] += "\n" + line
        elif line.strip():
            cur["fields"].append(["_free", line])
    return tasks, "\n".join(preamble), others


def filter_priors(md: str, comp: str) -> tuple[str, list[dict]]:
    """Return (filtered markdown, drop report)."""
    entries, preamble, others = _parse(md)
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

            # TWO DIFFERENT OPERATIONS, kept apart. Conflating them made the filter delete
            # 3 of 8 entries for EVERY competition, including ones unrelated to any prior
            # (2026-08-07 re-verification):
            #
            #   self-exclusion  -- a sentence naming the competition being solved is that
            #                      competition's own answer. It is removed, and if that empties
            #                      the evidence the whole entry goes, because the Action it
            #                      licenses then rests on the competition alone.
            #   cross-agent     -- a sentence citing AIDE's or NVIDIA's result must not reach
            #                      any run, but its absence says nothing about whether OUR
            #                      evidence supports the Action. It must never, by itself,
            #                      condemn the entry.
            # SUBJECT TRACKING. Prose evidence names its competition once and then continues
            # for two or three sentences that name nothing -- "s5e1 violated it in 6 of 7
            # years ... The opposite happened: ratio_target won decisively (private MAPE
            # 0.12417 vs 0.15626)." A per-sentence name match drops the first and SERVES the
            # rest, which handed an s5e1 re-run its own private score and its own
            # pre-registered form verdict. A sentence citing no competition belongs to the
            # last one that did (2026-08-07 re-verification).
            self_named, cross_agent, keep = [], [], []
            subject_is_self = False
            for s in segs:
                cites = _names(s, _ALL_COMP_TOKENS)
                if cites:                       # this sentence sets the subject
                    subject_is_self = _names(s, aliases)
                # else: subject carries over from the previous sentence
                if subject_is_self:
                    self_named.append(s)
                elif _OTHER_AGENTS.search(s):
                    cross_agent.append(s)
                else:
                    keep.append(s)
            dropped_lines += self_named + cross_agent
            emptied_by_self = False
            if label.lower() == "evidence" and (self_named or cross_agent):
                # Evidence survives only if something in it cites a competition OTHER than the
                # one being solved; a fragment citing nothing is a continuation of a dropped
                # item, not independent support.
                #
                # Gated on "something was actually dropped from this field". Orphan fragments
                # only exist where a sibling segment was removed. Applied unconditionally, this
                # rule also condemned TASK-IMBALANCED, whose evidence reads "none of ours yet
                # -- [GEN], unvalidated": an honestly-labelled generic prior containing no
                # competition's answer at all, which is precisely what this filter exists to
                # preserve (2026-08-07 re-verification).
                if not any(_names(s, _ALL_COMP_TOKENS) for s in keep):
                    emptied_by_self = bool(self_named)
                    keep = []
            if keep:
                kept_fields.append((label, " ".join(keep)))
            elif label.lower() == "evidence" and not emptied_by_self and cross_agent:
                # Everything that supported this Action was another agent's result. The Action
                # itself may still be sound generic knowledge, so the entry stays -- but the
                # gap is STATED, not papered over, because an unsupported prior a reader
                # believes is evidence-backed is worse than one that admits it is not.
                kept_fields.append((label, "[withheld — the evidence for this entry cited "
                                           "another agent's result, which is not a legitimate "
                                           "input to this stage. Treat the Action as an "
                                           "unvalidated generic prior.]"))
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

    # Non-TASK sections pass through verbatim. They carry no per-competition evidence to
    # exclude -- the whitelist is a source vocabulary, not a result -- and the filter has no
    # business editing a section whose shape it does not model.
    for sec in others:
        kept_md.append("")
        kept_md.append(sec["header"])
        kept_md.extend(sec["lines"])

    return "\n".join(kept_md).rstrip() + "\n", report


def filter_ops(md: str, comp: str) -> tuple[str, list[dict]]:
    """Per-competition view of knowledge/injection_operators.md.

    That file is the OPERATOR VOCABULARY -- the shared contract between Stage 0.5 and the
    execution layer -- so structure is never deleted: every heading, table row and paragraph
    survives. What gets redacted, in place and visibly, is any sentence or evidence cell whose
    subject is the competition being solved (its own answer) or another agent's run. Deleting
    a vocabulary row would make conforming dossiers unwritable; serving the row's evidence
    would hand the re-run its own result. Redaction with a stated reason does neither
    (2026-08-07: the audit showed filtering task_priors.md alone defeats the string, not the
    mechanism -- this file was still read raw by Stage 0.5 step 7).
    """
    aliases = _aliases(comp)
    out_lines, report = [], []

    def scrub(text: str, where: str) -> str:
        segs = [s for s in re.split(r"(?<=[.;])\s+", text) if s.strip()] or [text]
        kept, subject_is_self = [], False
        for s in segs:
            if _names(s, _ALL_COMP_TOKENS):
                subject_is_self = _names(s, aliases)
            if subject_is_self:
                report.append({"where": where, "dropped": s.strip()[:140]})
                continue
            if _OTHER_AGENTS.search(s):
                report.append({"where": where, "dropped": s.strip()[:140]})
                continue
            kept.append(s)
        if len(kept) == len(segs):
            return text
        return (" ".join(kept) + " [withheld — named this competition or another agent]"
                if kept else "[withheld — the evidence named this competition or another "
                             "agent; treat as unvalidated]")

    def flush_para(buf: list, where: str):
        """Prose is scrubbed per PARAGRAPH, not per line: the file is hard-wrapped, so the
        line naming the competition and the line carrying its score are different lines, and
        per-line subject tracking reset between them -- "the submission scored 48.24 SMAPE"
        survived an s3e19 filter because "s3e19" sat two wraps earlier (2026-08-07)."""
        if not buf:
            return
        joined = " ".join(ln.strip() for ln in buf)
        cleaned = scrub(joined, where)
        out_lines.append(cleaned)
        buf.clear()

    para: list = []
    for i, line in enumerate(md.splitlines()):
        if line.startswith("|") and line.count("|") >= 3 and "---" not in line:
            flush_para(para, f"para before line {i + 1}")
            cells = line.split("|")
            # If ANY cell names the competition (or an agent), every DATA cell in the row is
            # that competition's reading: redacting only the naming cell left "0.014 / 0.011
            # / ..." sitting beside a "[withheld]" label (2026-08-07). Keep the first
            # non-empty cell -- the vocabulary label -- and withhold the rest.
            row_named = any(_names(c, aliases) or _OTHER_AGENTS.search(c) for c in cells)
            first_cell = next((c.strip() for c in cells if c.strip()), "")
            if row_named and _names(first_cell, aliases):
                # The row's LABEL is the competition itself -- a per-competition data row in a
                # historical-readings table, not vocabulary. There is nothing to preserve.
                report.append({"where": f"table row {i + 1}",
                               "dropped": f"entire row keyed by the competition ({first_cell})"})
                continue
            if row_named:
                kept_label = False
                for j, c in enumerate(cells):
                    if c.strip() and not kept_label:
                        kept_label = True          # the operator/label cell survives
                    elif c.strip():
                        report.append({"where": f"table row {i + 1}", "dropped": c.strip()[:140]})
                        cells[j] = " [withheld — this row's evidence named the competition " \
                                   "being solved or another agent] "
                        # one marker is enough; blank the rest
                        cells[j + 1:] = ["" if cc.strip() else cc for cc in cells[j + 1:]]
                        break
                out_lines.append("|".join(cells))
            else:
                out_lines.append(line)
        elif line.strip().startswith(("#", "```")) or not line.strip():
            flush_para(para, f"para before line {i + 1}")
            out_lines.append(line)
        else:
            para.append(line)
    flush_para(para, "final para")
    return "\n".join(out_lines) + "\n", report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("comp", nargs="?", help="competition slug, e.g. playground-series-s3e11")
    ap.add_argument("--report", action="store_true", help="print what was dropped, not the library")
    ap.add_argument("--ops", action="store_true",
                    help="emit the filtered OPERATOR vocabulary (injection_operators.md) "
                         "instead of the task-prior library")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.comp:
        ap.error("competition slug required (or --selftest)")
    src = (_HERE / "injection_operators.md") if args.ops else PRIORS
    fn = filter_ops if args.ops else filter_priors
    md, rep = fn(src.read_text(encoding="utf-8"), args.comp)
    if args.report:
        print(f"# {'operator-vocabulary' if args.ops else 'task-prior'} exclusions "
              f"for {args.comp}\n")
        if not rep:
            print("(nothing dropped: no prior names this competition or another agent)")
        for r in rep:
            if "where" in r:
                print(f"- {r['where']}: {r['dropped']}")
                continue
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

    # --- BASELINE MUST COME FROM THE RAW FILE ------------------------------------------
    # This used to measure n_full from filter_priors(md, "no-such-competition-xyz") -- the
    # filter's own output -- so "4/5 entries remain" was measured against an already-degraded
    # yardstick and could not see that 3 of 8 entries were being dropped for EVERY
    # competition, related or not (2026-08-07 re-verification).
    n_raw_entries = md.count("\n## TASK-")
    n_raw_whitelist = sum(1 for ln in md.splitlines() if ln.startswith("| "))
    assert n_raw_entries >= 8 and n_raw_whitelist >= 7, (n_raw_entries, n_raw_whitelist)

    # --- NON-TASK SECTIONS MUST SURVIVE VERBATIM ---------------------------------------
    # The external-data whitelist lives under "## External-data whitelist", which does not
    # match the TASK-* parser pattern. It was being absorbed as trailing free text into the
    # preceding entry and deleted with it -- 7 rows in, 0 rows out, for all 20 competitions,
    # while 00_problem_dossier.md step 6 tells the agent to pick sources from that very
    # table in the filtered output. A capability deletion, and a silent one.
    for comp in ("playground-series-s3e19", "playground-series-s5e1", "cat-in-the-dat",
                 "no-such-competition-xyz"):
        out, _ = filter_priors(md, comp)
        rows = sum(1 for ln in out.splitlines() if ln.startswith("| "))
        assert rows == n_raw_whitelist, (
            f"{comp}: external-data whitelist lost {n_raw_whitelist - rows} of "
            f"{n_raw_whitelist} rows; Stage 0.5 step 6 reads that table")
        assert "External-data whitelist" in out, f"{comp}: the whitelist heading is gone"
    print(f"non-TASK sections survive intact ({n_raw_whitelist} whitelist rows, all comps)")

    # --- AN UNRELATED COMPETITION MUST LOSE NOTHING ------------------------------------
    full, _ = filter_priors(md, "no-such-competition-xyz")
    n_full = full.count("\n## TASK-")
    assert n_full == n_raw_entries, (
        f"a competition unrelated to every prior lost {n_raw_entries - n_full} of "
        f"{n_raw_entries} entries; exclusion must be driven by the competition being solved, "
        f"not by whether an entry happens to cite another agent")
    print(f"an unrelated competition keeps all {n_full} entries")

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

    # --- AND NOT NAMED IS NOT THE SAME AS NOT PRESENT ----------------------------------
    # The library is hard-wrapped prose: an evidence item names its competition once and then
    # continues for two more sentences that name nothing. Filtering on "does this sentence
    # contain the slug" drops the first sentence and serves the rest -- so an s5e1 re-run still
    # read its own private leaderboard score and its own pre-registered form verdict, which is
    # exactly the decision Stage 0.5 is supposed to make blind (2026-08-07 re-verification).
    leaks = {
        "playground-series-s5e1": ["0.12417", "0.15626"],
        "playground-series-s3e19": ["10.148", "7.793"],
        "tabular-playground-series-sep-2022": ["11.515", "0.466"],
        "afsis-soil-properties": ["0.49517", "0.44817"],
    }
    for c, needles in leaks.items():
        out, _ = filter_priors(md, c)
        present = [n for n in needles if n in out]
        assert not present, (
            f"{c}: its own recorded numbers {present} survive the filter -- the sentence "
            f"carrying them does not repeat the competition's name, so a name-match filter "
            f"cannot see it")
    print(f"no competition's own recorded NUMBERS survive either ({len(leaks)} checked)")

    # --- THE FILTER MUST COVER EVERY DOOR STAGE 0.5 IS TOLD TO OPEN --------------------
    # Filtering task_priors.md while Stage 0.5 step 7 orders injection_operators.md read raw
    # defeats the string, not the mechanism: that file's evidence tables carried AIDE's s3e11
    # result and cat-in-the-dat's own two scores, and SKILL.md's Stage 0.5 key-actions line
    # still pointed at the raw priors file (2026-08-07 re-verification).
    ops = (_HERE / "injection_operators.md").read_text(encoding="utf-8")
    assert not _OTHER_AGENTS.search(ops), (
        "knowledge/injection_operators.md still cites another agent's result; Stage 0.5 "
        "step 7 orders that file read raw, so the isolation hole is open one file over")
    skill = (_HERE.parent / ".claude/skills/kaggle-agent/SKILL.md").read_text(encoding="utf-8")
    import re as _re
    for m in _re.finditer(r"^.*task_priors\.md.*$", skill, _re.M):
        line = m.group(0)
        assert "task_priors_for" in line or "HARD RULE" in line or "filter" in line.lower(), (
            f"SKILL.md still points at the raw priors file with no filter mention: {line!r}")
    print("the other two doors are covered: injection_operators.md carries no cross-agent "
          "result, SKILL.md nowhere points at the raw file unqualified")

    # --- THE OPS FILTER ITSELF ---------------------------------------------------------
    for comp, own in [("cat-in-the-dat", ["beat the GBDT outright", "crosses **hurt**"]),
                      ("playground-series-s3e19", ["10.148", "7.793", "48.24"]),
                      ("playground-series-s5e1", ["0.12417"])]:
        fout, _ = filter_ops(ops, comp)
        hits = [n for n in own if n in fout]
        assert not hits, f"{comp}: own results {hits} survive the --ops filter"
        assert not _names(fout, _aliases(comp)), \
            f"{comp}: its own slug survives in the --ops output"
        # the vocabulary itself must be intact: same table rows, same headings
        assert fout.count("\n#") == ops.count("\n#"), f"{comp}: --ops deleted a heading"
        n_rows_raw = sum(1 for ln in ops.splitlines() if ln.startswith("|"))
        n_rows_out = sum(1 for ln in fout.splitlines() if ln.startswith("|"))
        n_comp_keyed = sum(1 for ln in ops.splitlines()
                           if ln.startswith("|") and
                           _names(ln.split("|")[1] if ln.count("|") > 1 else "", _aliases(comp)))
        assert n_rows_out == n_rows_raw - n_comp_keyed, \
            f"{comp}: --ops deleted a VOCABULARY row (data rows keyed by the competition " \
            f"itself are the only legitimate deletions: {n_comp_keyed})"
    fout, _ = filter_ops(ops, "no-such-competition-xyz")
    assert "withheld" not in fout.replace("[withheld — the original evidence cited another "
                                          "lane's run, which is not a legitimate input under "
                                          "the isolation protocol; treat as an unvalidated "
                                          "strong default]", ""), \
        "--ops withheld something for a competition unrelated to every entry"
    print("--ops: own results withheld, vocabulary structurally intact, unrelated comps "
          "lose nothing")

    # sibling slugs must NOT be over-excluded (s3e1 vs s3e19/s3e11/s3e14/s3e16)
    out, _ = filter_priors(md, "playground-series-s3e1")
    assert "s3e19" in out or "s3e11" in out, "sibling competitions were over-excluded from s3e1"
    print("sibling slugs survive (s3e1 does not exclude s3e19/s3e11) — no over-exclusion")

    print("task_priors_for selftest: all sections passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
