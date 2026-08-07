"""Experience-library retrieval gate: query before you experiment, and leave a trace.

Motivating incidents (main report, open problems): while decomposing the s3e19 residual we
re-derived a result the library already held, because nothing forces a query before an
experiment; and the library simultaneously held a correct entry and an over-claiming
neighbour -- we retrieved the optimistic one. The library's weak link is retrieval, not
content. This tool is the retrieval half of the fix: a ranked query whose results (or
explicit emptiness) are recorded into the experiment log, plus an audit mode that reports
experiments carrying no query trace.

Ranking rule: entries whose evidence field carries a score delta outrank keyword-only hits
(the library's own admission rule -- "no score delta, not admitted" -- applied at read
time), and section headers scope the search by metric/data-type before free keywords.

Usage:
  VIRTUAL_ENV= uv run python3 knowledge/query_library.py --query "SMAPE 時序 外推" [--top 5]
  VIRTUAL_ENV= uv run python3 knowledge/query_library.py --audit competitions/<comp>/experiments.json
  VIRTUAL_ENV= uv run python3 knowledge/query_library.py --selftest
Audit exits non-zero if any experiment lacks a `library_query` field (legacy files: the
audit reports counts; the gate is binding on NEW experiments per references/04_modeling.md).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
LIBRARY = _HERE / "experience.md"
EVIDENCE_PAT = re.compile(r"證據\s*[::]")
DELTA_PAT = re.compile(r"\d[\d.,]*\s*(?:→|->)\s*\d[\d.,]*")


_COMP_CITATION_ALIASES = {
    "tabular-playground-series-jan-2022": ["tpsjan22", "jan-2022", "jan2022"],
    "tabular-playground-series-aug-2022": ["tpsaug22", "aug-2022", "aug2022"],
    "tabular-playground-series-sep-2022": ["tpssep22", "sep-2022", "sep2022"],
    "afsis-soil-properties": ["afsis"],
    "conway-s-reverse-game-of-life": ["conway"],
    "cat-in-the-dat": ["citd"],
}


def _comp_aliases(comp: str | None) -> list[str]:
    """Every token the library might cite `comp` by, longest first (mirrors harness_v2)."""
    if not comp:
        return []
    al = {comp}
    al.update(_COMP_CITATION_ALIASES.get(comp, []))
    al.add(comp.replace("playground-series-", "").replace("tabular-", ""))
    return sorted(al, key=len, reverse=True)


def _names_comp(text: str, aliases: list[str]) -> bool:
    """Token-boundary match: 's3e1' must not match s3e19/s3e11, nor 's5e1' match s5e10."""
    for a in aliases:
        if re.search(rf"(?<![0-9a-z]){re.escape(a.lower())}(?![0-9a-z])", text):
            return True
    return False


def parse_library(path: Path = LIBRARY) -> list[dict]:
    """Flatten the library into entries: {section, text, has_evidence, has_delta}.

    Continuation lines attach to the bullet above them. The library is hard-wrapped and its
    own write-back convention puts "  | 證據: ..." on its own line, so a parser that keeps
    only "-"-prefixed lines detaches a bullet from its evidence -- and the self-exclusion in
    query() then cannot see which competition the bullet cites, serving it verbatim to that
    competition's re-run (2026-08-07 re-verification).
    """
    entries, section = [], ""
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        stripped = line.strip()
        if stripped.startswith("-"):
            text = stripped.lstrip("- ").strip()
            if text:
                entries.append({"section": section, "text": text})
        elif stripped and entries and not stripped.startswith("```"):
            entries[-1]["text"] += " " + stripped
    for e in entries:
        e["has_evidence"] = bool(EVIDENCE_PAT.search(e["text"]))
        e["has_delta"] = bool(DELTA_PAT.search(e["text"]))
    return entries


def query(terms: list[str], entries: list[dict], top: int = 5,
          comp: str | None = None, allow_self: bool = False) -> list[dict]:
    """Evidence-delta-ranked retrieval, with the same self-exclusion suggest_priors enforces.

    `comp` is REQUIRED in normal use. This module was the second, unfiltered door into the
    library: suggest_priors refuses to run without a competition name, while query() took no
    competition argument at all, so a run could satisfy the binding retrieval gate and read its
    own recorded answers all the way through (2026-08-04 architecture gate). Exclusion is
    scoped to the whole entry, not just its 證據 tail.
    """
    if comp is None and not allow_self:
        raise ValueError(
            "query() needs comp=<competition slug> so entries naming the competition being "
            "solved can be excluded -- the HARD RULE in SKILL.md. Pass allow_self=True only "
            "for a deliberately unfiltered read (e.g. a post-hoc audit).")
    aliases = _comp_aliases(comp) if (comp and not allow_self) else []
    scored, dropped = [], []
    for e in entries:
        hay = (e["section"] + " " + e["text"]).lower()
        if aliases and _names_comp(hay, aliases):
            dropped.append(e)
            continue
        hits = sum(1 for t in terms if t.lower() in hay)
        if hits == 0:
            continue
        score = (hits, e["has_delta"], e["has_evidence"], -len(e["text"]))
        scored.append((score, e, hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    if dropped:
        print(f"[query_library] excluded {len(dropped)} entry/entries naming {comp!r} itself "
              f"(self-exclusion); {min(len(scored), top)} returned")
    return [{"section": e["section"], "hits": h, "evidence_backed": e["has_delta"],
             "text": e["text"][:240]} for _, e, h in scored[:top]]


def audit(exp_path: Path, comp: str | None = None) -> dict:
    """Check the retrieval gate: every experiment has a trace, AND no trace recorded a hit
    that names the competition being solved.

    The trace-presence half was all this checked before, so a run could be fully "compliant"
    -- every experiment carrying a library_query -- while having read its own answers through
    the unfiltered query() door, and the audit output ("missing: []") would be quoted as proof
    of isolation (2026-08-04 architecture gate). `comp` defaults to the workspace directory
    name, which is where the slug already lives.
    """
    d = json.loads(exp_path.read_text())
    exps = d if isinstance(d, list) else d.get("experiments", [])
    missing = [e.get("experiment_id", i) for i, e in enumerate(exps)
               if not e.get("library_query")]
    comp = comp or Path(exp_path).parent.name
    aliases = _comp_aliases(comp)
    self_hits = []
    for i, e in enumerate(exps):
        hits = e.get("library_hits")
        if not hits or hits == "none":
            continue
        blob = json.dumps(hits, ensure_ascii=False).lower() if not isinstance(hits, str) else hits.lower()
        if _names_comp(blob, aliases):
            self_hits.append(e.get("experiment_id", i))
    return {"file": str(exp_path), "competition": comp, "n_experiments": len(exps),
            "with_query_trace": len(exps) - len(missing), "missing": missing,
            "self_citing_hits": self_hits,
            "verdict": "clean" if not missing and not self_hits else "VIOLATION"}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--comp", help="competition slug; entries naming it are excluded")
    ap.add_argument("--allow-self", action="store_true",
                    help="deliberately unfiltered read (post-hoc audits only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        entries = parse_library()
        assert len(entries) > 30, f"library parse suspiciously small: {len(entries)}"
        deltas = [e for e in entries if e["has_delta"]]
        assert deltas, "no evidence-delta entries parsed -- format drift?"
        try:
            query(["SMAPE"], entries)
        except ValueError:
            pass
        else:
            raise AssertionError("query() ran without a competition — the second door is open")
        r = query(["SMAPE", "時序"], entries, comp="playground-series-s3e1")
        assert r and r[0]["evidence_backed"], r[:1]
        # the retrieval-failure regression: querying for total-extrapolation must surface
        # the entry that identifies it as the method-agnostic bottleneck (the one we
        # failed to retrieve on s3e19)
        r = query(["總量", "外推"], entries, comp="playground-series-s3e1")
        assert any("總量" in x["text"] for x in r), "s3e19 regression query found nothing"
        empty = query(["zzz-no-such-term"], entries, allow_self=True)
        assert empty == []
        a = audit(_HERE.parent / "competitions/playground-series-s3e19/experiments.json")
        assert a["n_experiments"] > 0 and "self_citing_hits" in a and "verdict" in a
        s19 = query(["SMAPE", "時序", "外推"], entries, comp="playground-series-s3e19")
        al19 = _comp_aliases("playground-series-s3e19")
        assert not any(_names_comp((x["section"] + " " + x["text"]).lower(), al19) for x in s19), s19
        print(f"query_library selftest: all sections passed "
              f"({len(entries)} entries, {len(deltas)} evidence-backed)")
        return 0

    if args.audit:
        a = audit(args.audit, args.comp)
        print(json.dumps(a, ensure_ascii=False, indent=2))
        return 1 if a["verdict"] != "clean" else 0

    if not args.query:
        ap.error("--query, --audit, or --selftest required")
    for r in query(args.query, parse_library(), args.top,
                   comp=args.comp, allow_self=args.allow_self):
        print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
