"""Competition-rules gate — the hard stop before any external data is fetched.

Order of gates in Stage 0.5 is deliberate: this one runs FIRST, before the dossier's
external-data judgment and before any source is proposed, because a competition that forbids
external data makes the rest of the question moot and a run that ignores the rules is
disqualified regardless of its score.

WHY IT IS NOT AN LLM READING THE RULES AND DECIDING. The study's own s3e19 case is the
motivating example: a conservative setup-time `config.yaml` flag said external data was not
allowed while the task prior's evidence implied otherwise. The dossier's correct behaviour
there was to REFUSE TO GUESS and flag the conflict for verification; the official rules
(Section 7.C) settled it in favour of permission. This module encodes that behaviour:

  - a decision requires a QUOTE and a section reference from the rules text, recorded in the
    dossier, so a later reader can check the reading rather than trust it;
  - ambiguity resolves to FORBIDDEN, never to permitted;
  - a disagreement between the local config flag and the rules text is reported as a
    CONFLICT to be verified, not silently resolved by whichever source the code happens to
    consult first.

The rules text is supplied by the caller (fetched from the competition page or pasted from
the rules tab). This module does not fetch: a gate that silently downloads makes the decision
depend on network state at an arbitrary moment, and offline reproducibility of the decision
matters more here than convenience.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Phrases that grant permission, and phrases that withhold it. Both lists exist because the
# absence of a prohibition is NOT a permission: a rules text that never mentions external data
# leaves the question open, and open resolves to forbidden.
#
# MATCHING IS CLAUSE-SCOPED AND NEGATION-AWARE (2026-08-04). The first version searched a
# lowercased copy of the whole document for the first pattern hit, and one of its permit
# patterns was the bare substring "allowed to use external data" -- which matches inside the
# single most common Kaggle prohibition, "Participants are NOT allowed to use external data".
# The gate meant to be the subsystem's hard stop therefore FAILED OPEN on that wording, and
# cited the prohibiting clause as its permission. Three changes follow from that:
#
#   1. every permit hit is re-read inside its own clause and discarded if the clause negates it;
#   2. ALL hits are collected, not the first, so one stray permission-shaped sentence in an FAQ
#      cannot outvote a binding prohibition elsewhere -- both present is a conflict;
#   3. matching runs case-insensitively on the ORIGINAL text, never on a lowercased copy, since
#      str.lower() is not length-preserving in Unicode and the offsets drifted (a 'permitted'
#      verdict could be recorded with an empty quote, violating the module's own invariant).
_PERMIT_PATTERNS = [
    r"external data (?:is|are) allowed",
    r"use of external data is allowed",
    r"(?:you|participants|entrants|teams) may use (?:publicly available )?external data",
    r"external data:?\s*(?:is\s+)?permitted",
    r"publicly available external data (?:is|are) allowed",
    r"allowed to use (?:publicly available )?external data",
    r"external data\s*[:|\-–]\s*\**\s*(?:allowed|permitted|yes)\b",
]
_FORBID_PATTERNS = [
    # "is not allowed" / "is **not** permitted" / "| External data | Not allowed |"
    r"external data\b[^.\n|]{0,40}?\**\s*not\s*\**\s*(?:allowed|permitted)",
    r"external data\s*[:|\-–]\s*\**\s*(?:not allowed|not permitted|prohibited|forbidden|no)\b",
    r"external data (?:is|are) (?:prohibited|forbidden|disallowed)",
    r"use of external data is (?:not allowed|not permitted|prohibited|forbidden)",
    r"no external data",
    r"(?:may|must) not use (?:any )?(?:additional |outside |third[- ]party )?external data",
    r"not (?:allowed|permitted) to use (?:any )?(?:publicly available )?external data",
    # the "provided data only" family, which never says the words "external data"
    r"(?:may|must|can) not use (?:any )?data other than",
    r"(?:no|any) data other than the (?:competition|provided|training) data",
    r"only the (?:provided|competition) data may be used",
    r"restricted to the (?:provided|competition) data",
    r"solely (?:from|on) the (?:provided|competition) data",
]

# Words that flip a permission when they appear in the clause BEFORE the matched phrase.
_NEGATORS = re.compile(
    r"\b(?:not|never|cannot|can't|won't|may\s+not|must\s+not|shall\s+not|"
    r"prohibit(?:ed|s)?|forbid(?:den|s)?|disallow(?:ed|s)?|barred|no)\b|\bnot\b", re.I)

# A permission whose SUBJECT is a pretrained model is not a permission to join external data.
# The two are routinely granted separately ("pretrained weights are fine; do not join extra
# data"), and conflating them is how a run gets disqualified while believing it was compliant.
_PRETRAINED = re.compile(
    r"\bpre[-\s]?trained\b|\bpretrained\b|\bmodel weights\b|\bfoundation model\b|"
    r"\bcheckpoints?\b|\bembeddings? (?:model|weights)\b", re.I)

# Clause boundaries: sentence enders, newlines, and the pipes/bullets of tables and lists.
_CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+|\s*\|\s*|\s+[•\-–]\s+")

# A section heading near the decisive sentence, so the record can cite where it came from.
_SECTION_RE = re.compile(r"(?:^|\n)\s*((?:section\s+)?\d+(?:\.\d+)*\.?\s*[A-Z]?\.?)\s", re.I)


@dataclass
class RulesVerdict:
    verdict: str            # "permitted" | "forbidden" | "conflict" | "unstated"
    quote: str = ""
    section: str = ""
    config_flag: bool | None = None
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    def allows_external_data(self) -> bool:
        """Only an explicit, quoted permission opens the gate."""
        return self.verdict == "permitted"


def _clause_around(text: str, start: int, end: int) -> tuple[str, int]:
    """The clause containing [start, end), and the offset where that clause begins.

    Negation is scoped to a clause because "External data is allowed." and "External data is
    allowed only for the practice competition; for this one it is not." differ by a clause
    boundary, and a whole-document scan cannot tell them apart.
    """
    left = 0
    for m in _CLAUSE_SPLIT.finditer(text[:start]):
        left = m.end()
    right = len(text)
    m = _CLAUSE_SPLIT.search(text, end)
    if m:
        right = m.start()
    return text[left:right], left


def _hits(patterns: list[str], text: str, *, drop_negated: bool) -> list[dict]:
    """Every match of any pattern, each re-read inside its own clause.

    `drop_negated` is set for the permit list: a permission phrase preceded by a negator in
    the same clause is not a permission, it is the prohibition that phrase appears inside.
    Returned in document order so the recorded quote is the first real occurrence.
    """
    out: list[dict] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            clause, clause_start = _clause_around(text, m.start(), m.end())
            before = text[clause_start:m.start()]
            negated = bool(_NEGATORS.search(before))
            if drop_negated and negated:
                continue
            secs = _SECTION_RE.findall(text[:m.start()])
            out.append({
                "pattern": pat,
                "start": m.start(),
                "clause": clause.strip(),
                "quote": text[max(0, m.start() - 220):m.end() + 220].strip(),
                "section": secs[-1].strip() if secs else "",
                "pretrained_subject": bool(_PRETRAINED.search(clause)),
            })
    out.sort(key=lambda h: h["start"])
    return out


def evaluate_rules(rules_text: str, *, config_flag: bool | None = None) -> RulesVerdict:
    """Decide whether external data is permitted, with the evidence for the decision.

    `config_flag` is the competition workspace's own setup-time flag, when one exists. It is
    never allowed to decide on its own: it is compared against the rules text, and a
    disagreement is surfaced as a conflict for a human (or a documented rules citation) to
    settle -- the s3e19 behaviour.
    """
    if not (rules_text or "").strip():
        return RulesVerdict(
            "unstated", config_flag=config_flag,
            detail="no rules text supplied; the gate cannot grant permission it has not read")

    permits = _hits(_PERMIT_PATTERNS, rules_text, drop_negated=True)
    forbids = _hits(_FORBID_PATTERNS, rules_text, drop_negated=False)

    if permits and forbids:
        p, f = permits[0], forbids[0]
        return RulesVerdict(
            "conflict", quote=f"PERMIT: {p['quote']}\n---\nFORBID: {f['quote']}",
            section=f"{p['section']} / {f['section']}", config_flag=config_flag,
            detail=f"the rules text contains {len(permits)} permission-shaped and "
                   f"{len(forbids)} prohibition-shaped clause(s); resolve by reading the full "
                   f"clauses before any fetch -- do not pick one")
    if forbids:
        f = forbids[0]
        v = RulesVerdict("forbidden", quote=f["quote"], section=f["section"],
                         config_flag=config_flag,
                         detail="external data is prohibited by the quoted clause")
    elif permits:
        p = permits[0]
        # A permission whose clause is about pretrained models is not a permission to JOIN
        # external data. The two are routinely granted separately, and reading one as the
        # other is how a run gets disqualified while believing it was compliant.
        if all(h["pretrained_subject"] for h in permits):
            return RulesVerdict(
                "conflict", quote=p["quote"], section=p["section"], config_flag=config_flag,
                detail="every permission-shaped clause found also concerns pretrained models "
                       "or model weights. A clause permitting pretrained weights is not a "
                       "clause permitting extra data to be joined; read the rules yourself "
                       "before any fetch.")
        clean = [h for h in permits if not h["pretrained_subject"]]
        p = clean[0]
        if not p["quote"].strip():
            return RulesVerdict(
                "unstated", config_flag=config_flag,
                detail="a permission matched but no quote could be extracted; this gate does "
                       "not grant permission it cannot evidence.")
        v = RulesVerdict("permitted", quote=p["quote"], section=p["section"],
                         config_flag=config_flag,
                         detail="external data is permitted by the quoted clause")
    else:
        return RulesVerdict(
            "unstated", config_flag=config_flag,
            detail="the rules text does not address external data. Absence of a prohibition "
                   "is not a permission: unstated resolves to forbidden, and a competition "
                   "whose rules are silent needs a human reading before any fetch.")

    if config_flag is not None and config_flag != v.allows_external_data():
        return RulesVerdict(
            "conflict", quote=v.quote, section=v.section, config_flag=config_flag,
            detail=f"config.yaml says external_data_allowed={config_flag} but the rules text "
                   f"reads {v.verdict!r}. This is the s3e19 case: flag it for verification "
                   f"rather than letting either source win by default.")
    return v


def gate(rules_text: str, *, config_flag: bool | None = None,
         record_to: str | Path | None = None) -> RulesVerdict:
    """Run the gate and, when asked, write the decision and its evidence next to the dossier.

    The record is the point: a run that used external data must be able to show the clause it
    relied on, and a run that did not must be able to show why it stopped.
    """
    v = evaluate_rules(rules_text, config_flag=config_flag)
    v.evidence = {"verdict": v.verdict, "section": v.section, "quote": v.quote[:600],
                  "config_flag": config_flag, "detail": v.detail}
    if record_to:
        p = Path(record_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(v.evidence, indent=2, ensure_ascii=False))
        tmp.replace(p)
    return v


# ---------------------------------------------------------------------------
def selftest() -> int:
    permit_text = ("6. DATA\n6.A You may use the competition data.\n"
                   "7. EXTERNAL DATA\n7.C External data is allowed provided it is publicly "
                   "available and free to all participants.\n")
    forbid_text = ("5. RULES\n5.B Use of external data is prohibited; submissions must be "
                   "produced from the provided training data only.\n")
    silent_text = "1. OVERVIEW\nPredict the target. Submissions are scored by RMSE.\n"

    v = gate(permit_text)
    assert v.verdict == "permitted" and v.allows_external_data(), v
    assert "7" in v.section, v.section
    print(f"permits with a citation: section {v.section!r}")

    v = gate(forbid_text)
    assert v.verdict == "forbidden" and not v.allows_external_data(), v
    print(f"forbids with a citation: section {v.section!r}")

    v = gate(silent_text)
    assert v.verdict == "unstated" and not v.allows_external_data(), v
    print("silence does NOT grant permission")

    v = gate("", config_flag=True)
    assert not v.allows_external_data(), v
    print("no rules text cannot grant permission even with a permissive config flag")

    # the s3e19 regression: config says no, the rules say yes -> conflict, not a silent pick
    v = gate(permit_text, config_flag=False)
    assert v.verdict == "conflict" and not v.allows_external_data(), v
    assert "s3e19" in v.detail
    print("s3e19 regression: config/rules disagreement surfaces as a conflict, gate stays shut")

    v = gate(forbid_text, config_flag=True)
    assert v.verdict == "conflict" and not v.allows_external_data(), v
    print("the reverse disagreement is also a conflict")

    # both clauses present -> conflict
    v = gate(permit_text + forbid_text)
    assert v.verdict == "conflict" and not v.allows_external_data(), v
    print("a text containing both clauses is a conflict, not a coin flip")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rules_verdict.json"
        gate(permit_text, record_to=p)
        rec = json.loads(p.read_text())
        assert rec["verdict"] == "permitted" and rec["quote"], rec
    print("decision and evidence recorded to disk")

    # -----------------------------------------------------------------------
    # 2026-08-04 regression battery. Every case below was found by an adversarial
    # audit of the first version; the first two FAILED OPEN, which on a real
    # competition means a disqualified run that believed it was compliant.
    # -----------------------------------------------------------------------
    must_not_permit = [
        ("negated permit, the commonest Kaggle prohibition",
         "3. DATA\n3.B Participants are not allowed to use external data of any kind.\n"),
        ("negated permit, publicly-available variant",
         "You are not allowed to use publicly available external data in this competition.\n"),
        ("'not permitted' wording the forbid list used to miss",
         "4. ELIGIBILITY\n4.A The use of external data is not permitted.\n"),
        ("bolded negation inside a markdown clause",
         "## Rules\nExternal data is **not** allowed for this competition.\n"),
        ("table row",
         "| Rule | Value |\n| External data | Not allowed |\n| Pretrained models | Allowed |\n"),
        ("'no data other than' family, never says 'external data'",
         "7. You may not use any data other than the Competition Data to develop and test "
         "your models.\n"),
        ("'solely from the provided data'",
         "Submissions must be produced solely from the provided data.\n"),
        ("a pretrained-model permission is not an external-data permission",
         "9. MODELS\n9.A You are allowed to use external data that a pretrained model was "
         "trained on, but you may not join any additional data to the Competition Data.\n"),
        ("clause-scoped: permission belongs to a different competition",
         "1. This rule is copied from the practice competition, where external data is "
         "allowed. For THIS competition the use of external data is prohibited.\n"),
        ("silence",
         "1. OVERVIEW\nPredict the target. Submissions are scored by RMSE.\n"),
    ]
    for label, text in must_not_permit:
        v = gate(text)
        assert not v.allows_external_data(), f"FAILED OPEN on {label!r}: {v.verdict} / {v.quote}"
        print(f"does not open the gate: {label} -> {v.verdict}")

    # ...and the permissions that must still be recognised, or the gate is useless
    must_permit = [
        ("plain permission", "7.C External data is allowed provided it is publicly available.\n"),
        ("'you may use' form", "5. You may use external data so long as it is freely available "
                               "to all participants.\n"),
        ("'permitted' form", "2.A External data: permitted.\n"),
        ("table row, allowed", "| External data | Allowed |\n"),
    ]
    for label, text in must_permit:
        v = gate(text)
        assert v.allows_external_data(), f"failed to recognise a permission ({label}): {v.verdict}"
        assert v.quote.strip(), f"permitted with an empty quote ({label})"
        print(f"opens the gate with a quote: {label}")

    # a stray permission elsewhere in the document must NOT outvote a prohibition
    v = gate("FAQ: in most Kaggle competitions external data is allowed.\n\n"
             "6. RULES\n6.B For this competition, external data is not allowed.\n")
    assert v.verdict == "conflict" and not v.allows_external_data(), v
    print("a stray permission elsewhere in the document produces a conflict, not a permission")

    # Unicode: str.lower() is not length-preserving, which used to shift the quote offsets
    v = gate("İ" * 50 + "\n7.C External data is allowed for all participants.\n")
    assert v.allows_external_data() and "External data is allowed" in v.quote, v
    print("quote offsets survive non-length-preserving Unicode casing")

    print("rules_gate selftest: all sections passed")
    return 0


def main(argv: list[str]) -> int:
    """CLI, because Stage 0.5 is instructed to run this gate on a real competition.

    Reads the rules text from a file or stdin -- never from the network, for the reason in the
    module docstring. Exit status is the verdict: 0 permitted, 3 forbidden, 4 conflict,
    5 unstated. A non-zero status means external data is off for this competition.
    """
    import argparse
    ap = argparse.ArgumentParser(
        description="Decide whether a competition permits external data, with the evidence.")
    ap.add_argument("rules_file", nargs="?", type=Path,
                    help="file holding the competition's rules text; '-' or omitted reads stdin")
    ap.add_argument("--config-flag", choices=("true", "false"), default=None,
                    help="the workspace config.yaml's external_data_allowed, if it has one; "
                         "a disagreement with the rules text is reported as a conflict")
    ap.add_argument("--record-to", type=Path, default=None,
                    help="write the decision and its evidence here (next to dossier.json)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    if args.rules_file and str(args.rules_file) != "-":
        text = args.rules_file.read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    flag = None if args.config_flag is None else (args.config_flag == "true")
    v = gate(text, config_flag=flag, record_to=args.record_to)
    print(f"VERDICT  {v.verdict.upper()}"
          f"{'  (external data is OFF for this competition)' if not v.allows_external_data() else ''}")
    if v.section:
        print(f"  section: {v.section}")
    if v.quote:
        print(f"  quote:   {v.quote[:400]}")
    print(f"  detail:  {v.detail}")
    if args.record_to:
        print(f"  recorded -> {args.record_to}")
    return {"permitted": 0, "forbidden": 3, "conflict": 4, "unstated": 5}[v.verdict]


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
