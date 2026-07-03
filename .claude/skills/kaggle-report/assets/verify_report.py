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

# 負號僅在前面不是英數字/點時才算(2026-07-03、sub-2026 的 dash 是分隔符,不是負號)
_NUM = re.compile(r"(?<![\w.])-(?:\d+\.\d+|\d{3,})|\d+\.\d+|\d{3,}")
# 千分位逗號:夾在數字與「剛好 3 位數字 + 邊界」之間 → 移除(74,051 → 74051)
_GROUP_COMMA = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _normalize(text: str) -> str:
    """移除千分位逗號,使 74,051 與 facts 中的 74051 對得上。"""
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
    variants = set()
    for v in known:
        variants.add(v)
        for nd in range(1, 7):          # 容許四捨五入到 1–6 位小數的變體
            variants.add(round(v, nd))
    text = re.sub(r"```.*?```", "", report_text, flags=re.S)   # code block 豁免
    text = _normalize(text)                                     # 千分位正規化
    suspects = []
    for tok in _NUM.findall(text):
        if re.fullmatch(r"(19|20)\d{2}", tok):                  # 年份豁免
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
