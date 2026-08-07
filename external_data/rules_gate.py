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
# 2026-08-04, SECOND REVISION, and the reason is worth stating because it is a design change
# rather than another patch. The first version failed open on "Participants are NOT allowed to
# use external data" (a permit pattern matching inside its own negation). The second version
# added clause-scoped negation -- and an adversarial probe then broke it with a single line
# break: "Participants are not\nallowed to use external data", i.e. hard-wrapped rules text,
# which is exactly what pasting from the rules tab produces. It also fell to "Q: Are
# participants allowed to use external data? A: No.", to "allowed ... in the practice
# competition, not in this one", and to a real rules page whose binding prohibition was phrased
# "prohibited from using external data sources" while an obsolete quoted permission sat two
# sections later.
#
# The lesson is that enumerating how a rules page can say "no" is not a winnable game, and the
# costs are wildly asymmetric: a wrong "forbidden" costs us external data on one competition, a
# wrong "permitted" costs a DISQUALIFICATION. So the gate no longer tries to be clever. It is
# fail-closed by construction:
#
#   permitted  REQUIRES  (a) an explicit, clause-scoped, non-negated, non-interrogative
#                            permission, AND
#                        (b) ZERO restrictive signals anywhere in the entire document, AND
#                        (c) ZERO pretrained/model-weight language anywhere in the document, AND
#                        (d) no scope limiter in the permitting clause.
#   anything else resolves to forbidden / conflict / unstated -- all of which keep the gate shut.
#
# (b) is the load-bearing rule. It does not need to know WHICH prohibition is binding; the mere
# presence of restrictive language anywhere means a human must read the page. That converts an
# unwinnable enumeration problem into a winnable one: the restrictive vocabulary only has to be
# broad, never complete, because breadth costs a human read and narrowness costs the run.
_EXT = r"external\s+data(?:sets?)?|external\s+data\s+sources?|outside\s+data|third[-\s]party\s+data"

_PERMIT_PATTERNS = [
    rf"(?:{_EXT})\s+(?:is|are)\s+allowed",
    rf"use of (?:{_EXT}) is allowed",
    rf"(?:you|participants|entrants|teams) may use (?:publicly available )?(?:{_EXT})",
    rf"(?:{_EXT}):?\s*(?:is\s+)?permitted",
    rf"publicly available (?:{_EXT}) (?:is|are) allowed",
    rf"allowed to use (?:publicly available )?(?:{_EXT})",
    rf"(?:{_EXT})\s*[:|\-–]\s*\**\s*(?:allowed|permitted|yes)\b",
]

# Restrictive signals. ANY occurrence anywhere in the document makes "permitted" unreachable.
# Deliberately broad: a false hit costs a human reading the rules, a miss costs the run.
_RESTRICTIVE = [
    r"\bprohibit(?:ed|s|ion)?\b", r"\bforbid(?:den|s)?\b", r"\bban(?:ned|s)?\b",
    r"\bbarred\b", r"\bdisallow(?:ed|s)?\b", r"\bnot\s+(?:be\s+)?(?:allowed|permitted)\b",
    r"\bmay\s+not\b", r"\bmust\s+not\b", r"\bshall\s+not\b", r"\bcannot\b", r"\bcan\s?not\b",
    r"\bdo\s+not\s+use\b", r"\bnever\s+use\b",
    rf"\bno\s+(?:{_EXT})\b", r"\bno\s+(?:additional|other|outside|third[-\s]party)\s+data\b",
    r"\bdata\s+other\s+than\b", r"\bother\s+than\s+the\s+(?:competition|provided|training)\s+data\b",
    r"\bonly\s+the\s+(?:provided|competition|training)\s+data\b",
    r"\brestricted\s+to\s+the\b", r"\b(?:competition|provided|training)\s+data\s+alone\b",
    r"\bsolely\s+(?:from|on|using)\s+the\b", r"\bstrictly\s+(?:prohibited|forbidden)\b",
]

# A permission whose SUBJECT is a pretrained model is not a permission to join external data.
# The two are routinely granted separately ("pretrained weights are fine; do not join extra
# data"), and conflating them is how a run gets disqualified while believing it was compliant.
# Checked DOCUMENT-WIDE, not clause-locally: the probe defeated the clause-local version with a
# semicolon ("...allowed to use external data; specifically, pretrained weights are fine...").
_PRETRAINED = re.compile(
    r"\bpre[-\s]?trained\b|\bpretrained\b|\bmodel weights\b|\bfoundation model\b|"
    r"\bcheckpoints?\b|\bbackbones?\b|\bimagenet\b|\btransfer learning\b|"
    r"\bembeddings? (?:model|weights)\b", re.I)

# Scope limiters: a permission that applies somewhere else, or only under a condition, is not a
# permission here. ("allowed only in the sandbox arena", "allowed in the practice competition")
_SCOPE_LIMITER = re.compile(
    r"\bonly\b|\bunless\b|\bexcept\b|\bpractice competition\b|\bsandbox\b|\bprevious season\b|"
    r"\blast (?:year|season)\b|\bdo not carry over\b|\bno longer\b|\bformerly\b", re.I)

# Words that flip a permission when they appear before the matched phrase. Bare "no" is NOT here:
# it wrongly negated "There is no restriction on data sources: external data is allowed."
_NEGATORS = re.compile(
    r"\b(?:not|never|cannot|can'?t|won'?t|nor|neither|"
    r"prohibit(?:ed|s)?|forbid(?:den|s)?|disallow(?:ed|s)?|barred|banned)\b", re.I)

# Clause boundaries: sentence enders, colons, blank lines, table pipes and line-initial bullets.
# A single newline is NOT a boundary -- rules text is hard-wrapped, and treating a wrap as a
# clause end is precisely what let "not\nallowed to use external data" read as a permission.
_CLAUSE_SPLIT = re.compile(r"(?<=[.;!?:])\s+|\n\s*\n|\s*\|\s*|(?:^|\n)\s*[•*]\s+")

# Hard-wrap normalizer: join a line to the next when the break is cosmetic (no sentence end, no
# blank line, next line does not start a new numbered section or bullet).
_HARD_WRAP = re.compile(r"(?<![.;!?:])\n(?!\s*\n)(?!\s*(?:\d+[.)]|[•*\-]|\|))[ \t]*")


def _unwrap(text: str) -> str:
    """Undo cosmetic line wrapping so clause scoping sees whole sentences."""
    return _HARD_WRAP.sub(" ", text)

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

    `drop_negated` is set for the permit list: a permission phrase negated in its own clause is
    not a permission, it is the prohibition that phrase appears inside. Negation is checked in
    the clause AND in a raw window immediately before the match, because a table pipe or a
    stray boundary can sit between the negator and the phrase ("not | allowed to use...").
    """
    out: list[dict] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            clause, clause_start = _clause_around(text, m.start(), m.end())
            before_clause = text[clause_start:m.start()]
            before_raw = text[max(0, m.start() - 60):m.start()]
            after_clause = clause[m.end() - clause_start:] if m.end() > clause_start else ""
            negated = bool(_NEGATORS.search(before_clause) or _NEGATORS.search(before_raw)
                           # "allowed in the practice competition, NOT in this one"
                           or _NEGATORS.search(after_clause))
            # A permission inside a QUESTION is not a permission; the answer is elsewhere.
            interrogative = clause.rstrip().endswith("?") or clause.lstrip().lower().startswith(("q:", "q."))
            if drop_negated and (negated or interrogative):
                continue
            secs = _SECTION_RE.findall(text[:m.start()])
            out.append({
                "pattern": pat,
                "start": m.start(),
                "clause": clause.strip(),
                "quote": text[max(0, m.start() - 220):m.end() + 220].strip(),
                "section": secs[-1].strip() if secs else "",
                "scope_limited": bool(_SCOPE_LIMITER.search(clause)),
            })
    out.sort(key=lambda h: h["start"])
    return out


def _restrictive_signals(text: str) -> list[str]:
    """Every restrictive phrase anywhere in the document, verbatim.

    Presence of ANY of these makes "permitted" unreachable. The gate does not try to decide
    which clause binds -- that is a reading, and a reading is what it is refusing to do.
    """
    found = []
    for pat in _RESTRICTIVE:
        for m in re.finditer(pat, text, re.I):
            found.append(m.group(0).strip())
    return sorted(set(found), key=str.lower)


def evaluate_rules(rules_text: str, *, config_flag: bool | None = None) -> RulesVerdict:
    """Decide whether external data is permitted, with the evidence for the decision.

    `config_flag` is the competition workspace's own setup-time flag, when one exists. It is
    never allowed to decide on its own: it is compared against the rules text, and a
    disagreement is surfaced as a conflict for a human (or a documented rules citation) to
    settle -- the s3e19 behaviour.
    """
    v = _read_rules(rules_text, config_flag)
    # The config flag NEVER decides on its own, and this comparison must sit on EVERY path:
    # an early return that skips it silently removes the s3e19 behaviour, which is the whole
    # reason this module exists (caught 2026-08-04 when the fail-closed rewrite bypassed it).
    if config_flag is not None and config_flag != v.allows_external_data():
        return RulesVerdict(
            "conflict", quote=v.quote, section=v.section, config_flag=config_flag,
            detail=f"config.yaml says external_data_allowed={config_flag} but the rules text "
                   f"reads {v.verdict!r}. This is the s3e19 case: flag it for verification "
                   f"rather than letting either source win by default.")
    return v


def _read_rules(rules_text: str, config_flag: bool | None) -> RulesVerdict:
    """The rules reading alone, with no config-flag arbitration -- see evaluate_rules."""
    if not (rules_text or "").strip():
        return RulesVerdict(
            "unstated", config_flag=config_flag,
            detail="no rules text supplied; the gate cannot grant permission it has not read")

    text = _unwrap(rules_text)
    permits = _hits(_PERMIT_PATTERNS, text, drop_negated=True)
    restrictive = _restrictive_signals(text)
    pretrained = sorted({m.group(0) for m in _PRETRAINED.finditer(text)}, key=str.lower)

    # RULE (b), the load-bearing one: any restrictive language anywhere shuts the gate.
    if restrictive:
        f_quote = ""
        m = re.search(_RESTRICTIVE[0] if not restrictive else re.escape(restrictive[0]), text, re.I)
        if m:
            f_quote = text[max(0, m.start() - 220):m.end() + 220].strip()
            secs = _SECTION_RE.findall(text[:m.start()])
            f_sec = secs[-1].strip() if secs else ""
        else:
            f_sec = ""
        if permits:
            p = permits[0]
            return RulesVerdict(
                "conflict", quote=f"PERMIT: {p['quote']}\n---\nRESTRICTIVE: {f_quote}",
                section=f"{p['section']} / {f_sec}", config_flag=config_flag,
                detail=f"the page contains {len(permits)} permission-shaped clause(s) AND "
                       f"restrictive language {restrictive[:6]}. Which one binds is a reading, "
                       f"and this gate does not make readings: a human must resolve it before "
                       f"any fetch.")
        return RulesVerdict(
            "forbidden", quote=f_quote, section=f_sec, config_flag=config_flag,
            detail=f"restrictive language present: {restrictive[:6]}. No unnegated permission "
                   f"was found alongside it.")

    if not permits:
        return RulesVerdict(
            "unstated", config_flag=config_flag,
            detail="the rules text does not address external data. Absence of a prohibition "
                   "is not a permission: unstated resolves to forbidden, and a competition "
                   "whose rules are silent needs a human reading before any fetch.")

    # RULE (c): pretrained-model language anywhere makes the SUBJECT of the permission
    # ambiguous. "External data is allowed only in the form of ImageNet-trained backbones" is
    # a permission to load weights, not to join rows, and no regex can reliably tell which
    # sentence means which.
    if pretrained:
        return RulesVerdict(
            "conflict", quote=permits[0]["quote"], section=permits[0]["section"],
            config_flag=config_flag,
            detail=f"a permission was found, but the page also uses pretrained-model language "
                   f"{pretrained[:5]}. A clause permitting pretrained weights is not a clause "
                   f"permitting extra data to be joined, and telling them apart is a reading. "
                   f"Resolve by hand before any fetch.")

    # RULE (d): a permission that applies elsewhere, or only conditionally, is not one here.
    clean = [h for h in permits if not h["scope_limited"]]
    if not clean:
        return RulesVerdict(
            "conflict", quote=permits[0]["quote"], section=permits[0]["section"],
            config_flag=config_flag,
            detail="every permission found is scope-limited (only / unless / except / a "
                   "different competition or season). A conditional permission needs the "
                   "condition checked by a human, not by this gate.")

    p = clean[0]
    if not p["quote"].strip():
        return RulesVerdict(
            "unstated", config_flag=config_flag,
            detail="a permission matched but no quote could be extracted; this gate does "
                   "not grant permission it cannot evidence.")
    v = RulesVerdict("permitted", quote=p["quote"], section=p["section"],
                     config_flag=config_flag,
                     detail="external data is permitted by the quoted clause, and the page "
                            "contains no restrictive, pretrained-model, or scope-limiting "
                            "language anywhere")

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
        # --- 2026-08-04 SECOND revision: every input that defeated the FIRST fix ------------
        ("the same sentence, hard-wrapped (what pasting the rules tab produces)",
         "3. DATA\n3.B Participants are not\nallowed to use external data of any kind.\n"),
        ("negation AFTER the permitting phrase",
         "7.C External data is allowed in the practice competition, not in this one."),
        ("negator split by a dash parenthetical",
         "4. Participants are not - under any circumstances - allowed to use external data."),
        ("'nor' and 'prohibited from', neither an old negator",
         "6.B Entrants are prohibited from joining outside sources; nor are they allowed to "
         "use external data."),
        ("permission scoped to another arena, prohibition here",
         "2. External data is allowed only in the sandbox arena; unless the host says "
         "otherwise, entrants here must not go outside the Competition Data."),
        ("the permission is a QUESTION; the answer is no",
         "Q: Are participants allowed to use external data? A: No."),
        ("the same, as a table row", "| Are teams allowed to use external data? | No |"),
        ("a table pipe between the negator and the phrase",
         "Participants are not | allowed to use external data."),
        ("a realistic page: binding prohibition plus an obsolete quoted permission",
         "1. OVERVIEW\nPredict the target.\n\n4.B Participants are prohibited from using "
         "external data sources of any kind when developing or testing their models.\n\n"
         "5.A The rules of the 2025 season stated that entrants may use external data. Those "
         "rules do not carry over.\n"),
        ("'do not use'", "Do not use external data."),
        ("'barred from'", "Entrants are barred from using external data."),
        ("plural 'external datasets' defeats a word boundary",
         "The use of external datasets is strictly prohibited."),
        ("'banned'", "External data is banned."),
        ("'Competition Data alone', never says the words 'external data'",
         "Models must be trained on the Competition Data alone."),
        ("pretrained: the permission is for backbones, no keyword in the clause",
         "8.A External data is allowed only in the form of ImageNet-trained backbones. You "
         "may not add any other data to the training set."),
        ("pretrained: a semicolon ends the clause before the word 'pretrained'",
         "4.B You are allowed to use external data; specifically, weights from a pretrained "
         "model are fine, but you may not join any additional data."),
        ("pretrained: an untagged FAQ hit satisfied the old all() quantifier",
         "9.A You are allowed to use external data that a pretrained model was trained on, "
         "but you may not join any additional data to the Competition Data.\nFAQ: in Kaggle "
         "competitions external data is allowed when it is public."),
        ("pretrained: markdown table", "| External data | Allowed | pretrained weights only |"),
        ("pretrained: 'External data: permitted' then scoped to weights",
         "2.A External data: permitted. This means pretrained model weights only; you must "
         "not join any additional data."),
        ("transfer learning without the word 'pretrained'",
         "You may use external data that a model was trained on before the competition "
         "(transfer learning)."),
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
        # the fail-CLOSED regression the same probe found: "no restriction" is not a negator
        ("'There is no restriction on data sources'",
         "7.C There is no restriction on data sources: external data is allowed provided it "
         "is publicly available."),
        ("a permission that is itself hard-wrapped",
         "7. EXTERNAL DATA\n7.C External data is allowed provided it is publicly\navailable "
         "and free to all participants.\n"),
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
    ap.add_argument("--competition", default=None,
                    help="slug this verdict is recorded FOR; consumers refuse a verdict "
                         "recorded for a different competition")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    if args.rules_file and str(args.rules_file) != "-":
        text = args.rules_file.read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    flag = None if args.config_flag is None else (args.config_flag == "true")
    v = gate(text, config_flag=flag, record_to=None)
    if args.competition:
        v.evidence["competition"] = args.competition
    if args.record_to:
        p_out = Path(args.record_to)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = p_out.with_suffix(p_out.suffix + ".tmp")
        tmp.write_text(json.dumps(v.evidence, indent=2, ensure_ascii=False))
        tmp.replace(p_out)
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
