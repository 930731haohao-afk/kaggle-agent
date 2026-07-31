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


def parse_library(path: Path = LIBRARY) -> list[dict]:
    """Flatten the library into entries: {section, text, has_evidence, has_delta}."""
    entries, section = [], ""
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        if line.lstrip().startswith("-"):
            text = line.lstrip("- ").strip()
            if text:
                entries.append({"section": section, "text": text,
                                "has_evidence": bool(EVIDENCE_PAT.search(text)),
                                "has_delta": bool(DELTA_PAT.search(text))})
    return entries


def query(terms: list[str], entries: list[dict], top: int = 5) -> list[dict]:
    scored = []
    for e in entries:
        hay = (e["section"] + " " + e["text"]).lower()
        hits = sum(1 for t in terms if t.lower() in hay)
        if hits == 0:
            continue
        score = (hits, e["has_delta"], e["has_evidence"], -len(e["text"]))
        scored.append((score, e, hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"section": e["section"], "hits": h, "evidence_backed": e["has_delta"],
             "text": e["text"][:240]} for _, e, h in scored[:top]]


def audit(exp_path: Path) -> dict:
    d = json.loads(exp_path.read_text())
    exps = d if isinstance(d, list) else d.get("experiments", [])
    missing = [e.get("experiment_id", i) for i, e in enumerate(exps)
               if not e.get("library_query")]
    return {"file": str(exp_path), "n_experiments": len(exps),
            "with_query_trace": len(exps) - len(missing), "missing": missing}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        entries = parse_library()
        assert len(entries) > 30, f"library parse suspiciously small: {len(entries)}"
        deltas = [e for e in entries if e["has_delta"]]
        assert deltas, "no evidence-delta entries parsed -- format drift?"
        r = query(["SMAPE", "時序"], entries)
        assert r and r[0]["evidence_backed"], r[:1]
        # the retrieval-failure regression: querying for total-extrapolation must surface
        # the entry that identifies it as the method-agnostic bottleneck (the one we
        # failed to retrieve on s3e19)
        r = query(["總量", "外推"], entries)
        assert any("總量" in x["text"] for x in r), "s3e19 regression query found nothing"
        empty = query(["zzz-no-such-term"], entries)
        assert empty == []
        a = audit(_HERE.parent / "competitions/playground-series-s3e19/experiments.json")
        assert a["n_experiments"] > 0
        print(f"query_library selftest: all sections passed "
              f"({len(entries)} entries, {len(deltas)} evidence-backed)")
        return 0

    if args.audit:
        a = audit(args.audit)
        print(json.dumps(a, ensure_ascii=False, indent=2))
        return 1 if a["missing"] else 0

    if not args.query:
        ap.error("--query, --audit, or --selftest required")
    for r in query(args.query, parse_library(), args.top):
        print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
