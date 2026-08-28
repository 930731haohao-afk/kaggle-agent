"""
Heuristic number-traceability check: every significant number in REPORT.md
must exist in facts.json (spec §9.1). Exit 0 = clean; exit 1 = suspects found.

Usage:
    uv run python3 …/verify_report.py <REPORT.md> <facts.json>
"""
import json
import pathlib
import re
import sys

# A minus sign counts only when not preceded by an alphanumeric char or dot
# (the dash in 2026-07-03 / sub-2026 is a separator, not a minus sign)
_NUM = re.compile(r"(?<![\w.])-(?:\d+\.\d+|\d{3,})|\d+\.\d+|\d{3,}")
# Thousands-separator comma: between a digit and "exactly 3 digits + boundary" → remove (74,051 → 74051)
_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")
# Markdown section numbers (section structure, not data numbers): `### 2.1`, `#### 2.2a.` etc.
# at the start of heading lines — section-tree numbers are not facts.json data and are not
# bound by Hard Rule 1 (cf. the existing year exemption).
_HEADING_NUM = re.compile(r"^(#{1,6}[ \t]+)\d+(?:\.\d+)*[a-z]?\.?(?=[ \t])", re.M)
# In-body section cross-references (section structure, not data numbers): 「見第 2.1 節」
# 「第 3、2.2a 節」「見節 4、2.5」「見 2.2a 節」「見 2.2a 小節」 etc. — same exemption
# rationale as above; only the section-number tokens directly adjacent to 「第」/「節」/「見」/
# 「小節」 are exempted, without consuming any other surrounding numbers.
_TOKEN = r"\d+(?:\.\d+)*[a-z]?"
_TOKENLIST = rf"{_TOKEN}(?:\s*[、/,]\s*(?:and\s+)?{_TOKEN})*"
_SECTION_REF = re.compile(
    rf"第\s*{_TOKENLIST}(?=\s*節)"      # 「第 2.1、2.2a 節」
    rf"|(?<=節)\s*{_TOKENLIST}"          # 「見節 4、2.5」 (節 comes first, no 「第」)
    rf"|見\s*{_TOKENLIST}(?=\s*節)"      # 「見 2.2a 節」 (no 「第」)
    rf"|{_TOKEN}(?=\s*小節)"             # 「見(本節末的)? 2.2a 小節」
    # English section cross-references (section structure, not data numbers) — bilingual exemption
    # so translated reports keep passing: "§2.1", "Section 2.10", "Sections 3.1 and 3.2",
    # "sub-section 2.2a". Only the section number tokens are removed, not surrounding data.
    rf"|§\s*{_TOKENLIST}"                                       # 「§2.1」「§2.1, 2.2」
    rf"|(?i:sections?)\s*{_TOKENLIST}"                          # 「Section 2.10」「Sections 3.1 and 3.2」
    rf"|(?i:sub-?sections?)\s*{_TOKENLIST}"                     # 「sub-section 2.2a」
)


def _normalize(text: str) -> str:
    """Remove thousands-separator commas so 74,051 matches 74051 in facts."""
    return _GROUP_COMMA.sub("", text)


def _collect_values(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_values(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for tok in _NUM.findall(_normalize(obj)):
            out.add(float(tok))


def find_suspects(report_text: str, facts: dict) -> list:
    known = set()
    _collect_values(facts, known)
    # A derived number whose formula is spelled out inside a fenced code block counts as
    # proven, so the same number may also appear in tables/body text (e.g. the percentages
    # in the Section 5 "relative improvement" column, whose formulas are listed in that
    # section's calculation block). This does not loosen the gate on made-up numbers —
    # code-block content is already exempted wholesale; this merely lets a number proven
    # in a code block be cited by a table.
    code = "\n".join(re.findall(r"```.*?```", report_text, flags=re.S))
    for tok in _NUM.findall(_normalize(code)):
        known.add(float(tok))
    variants = set()
    for v in known:
        variants.add(v)
        for nd in range(1, 7):          # allow variants rounded to 1–6 decimal places
            variants.add(round(v, nd))
    text = re.sub(r"```.*?```", "", report_text, flags=re.S)   # code-block exemption
    text = _HEADING_NUM.sub(lambda m: m.group(1), text)         # section-number exemption (headings)
    text = _SECTION_REF.sub("", text)                           # section-number exemption (in-body cross-refs)
    text = _normalize(text)                                     # thousands-separator normalization
    suspects = []
    for tok in _NUM.findall(text):
        # Stage-marker exemption (2026-07-07): sub-stages use the bare 1.1/3.2 numeric form,
        # identical in shape to a decimal. Only "single digit.single digit" is exempted;
        # every real score in this project has ≥3 decimal places, so the false-pass risk is minimal.
        if re.fullmatch(r"\d\.\d", tok):
            continue
        if re.fullmatch(r"(19|20)\d{2}", tok):                  # year exemption
            continue
        if float(tok) in variants:
            continue
        if tok not in suspects:
            suspects.append(tok)
    return suspects


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: verify_report.py <REPORT.md> <facts.json>")
    report = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    facts = json.load(open(sys.argv[2]))
    suspects = find_suspects(report, facts)
    if suspects:
        print("FAIL — numbers not traceable to facts.json:")
        for s in suspects:
            print(f"  {s}")
        sys.exit(1)
    print("OK — all numbers traceable to facts.json")


if __name__ == "__main__":
    main()
