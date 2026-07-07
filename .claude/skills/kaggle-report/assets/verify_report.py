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
# Markdown 節次編號(章節結構,非資料數字):標題行開頭的 `### 2.1`、`#### 2.2a.` 等 —
# 章節樹編號本身不是 facts.json 的資料,不受 Hard Rule 1 拘束(cf. 既有的年份豁免)。
_HEADING_NUM = re.compile(r"^(#{1,6}[ \t]+)\d+(?:\.\d+)*[a-z]?\.?(?=[ \t])", re.M)
# 內文節次交叉引用(章節結構,非資料數字):「見第 2.1 節」「第 3、2.2a 節」「見節 4、2.5」
# 「見 2.2a 節」「見 2.2a 小節」等 — 同上豁免理由;僅豁免緊鄰「第」/「節」/「見」/「小節」
# 的編號本身,不吃掉周圍任何其他數字。
_TOKEN = r"\d+(?:\.\d+)*[a-z]?"
_TOKENLIST = rf"{_TOKEN}(?:\s*[、/,]\s*{_TOKEN})*"
_SECTION_REF = re.compile(
    rf"第\s*{_TOKENLIST}(?=\s*節)"      # 「第 2.1、2.2a 節」
    rf"|(?<=節)\s*{_TOKENLIST}"          # 「見節 4、2.5」(節在前,無「第」)
    rf"|見\s*{_TOKENLIST}(?=\s*節)"      # 「見 2.2a 節」(無「第」)
    rf"|{_TOKEN}(?=\s*小節)"             # 「見(本節末的)? 2.2a 小節」
)


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
    text = _HEADING_NUM.sub(lambda m: m.group(1), text)         # 節次編號豁免(標題)
    text = _SECTION_REF.sub("", text)                           # 節次編號豁免(內文交叉引用)
    text = _normalize(text)                                     # 千分位正規化
    suspects = []
    for tok in _NUM.findall(text):
        # 階段記號豁免(2026-07-07):子階段採 1.1/3.2 純數字型,與小數同形。
        # 僅豁免「單位數.單位數」;本專案真實分數皆 ≥3 位小數,誤放風險極低。
        if re.fullmatch(r"\d\.\d", tok):
            continue
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
