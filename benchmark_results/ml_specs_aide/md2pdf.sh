#!/usr/bin/env bash
# Convert a markdown report to PDF: python-markdown → HTML → weasyprint(主)/chromium(備援).
# 自動加入:目錄(h2+h3,注入第一個 h2 之前)+ 每頁頁尾頁碼(CSS @page)。
# 目錄各節頁碼與頁尾頁碼需要 weasyprint(target-counter / @bottom-center);
# chromium 備援仍會產出 PDF 與目錄清單,但沒有頁碼。
# Usage: bash md2pdf.sh <REPORT.md> [out.pdf]
# exit 0 = PDF 已寫出;2 = 兩引擎皆不可用(.md 為交付物);3 = 引擎回報成功但檔案不存在
set -euo pipefail

MD="$1"
PDF="${2:-${MD%.md}.pdf}"
CSS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/report_style.css"
HTML="${MD%.md}.tmp.html"   # 不用隱藏檔名:snap chromium 讀不到部分隱藏路徑
WLOG="${MD%.md}.tmp.weasy.log"
trap 'rm -f "$HTML" "$WLOG"' EXIT

uv run python3 - "$MD" "$CSS" "$HTML" <<'PY'
import pathlib, sys
import markdown
try:
    from markdown.extensions.toc import slugify_unicode as _slug
except ImportError:  # 舊版 markdown 無 unicode slug:CJK 標題 id 退化但仍可用
    from markdown.extensions.toc import slugify as _slug
md_path, css_path, html_path = sys.argv[1:4]
md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc"],
    extension_configs={"toc": {"slugify": _slug, "toc_depth": "2-3"}})
body = md.convert(pathlib.Path(md_path).read_text(encoding="utf-8"))
toc = (getattr(md, "toc", "") or "").strip()
# 目錄注入第一個 h2 之前;h2 少於 3 個的短文件不加目錄
i = body.find("<h2")
if i != -1 and body.count("<h2") >= 3 and toc:
    nav = '<nav class="report-toc"><div class="toc-title">Contents</div>' + toc + "</nav>"
    body = body[:i] + nav + body[i:]
css = pathlib.Path(css_path).read_text(encoding="utf-8")
pathlib.Path(html_path).write_text(
    f"<!doctype html><html><head><meta charset='utf-8'>"
    f"<style>{css}</style></head><body>{body}</body></html>",
    encoding="utf-8")
PY

# 主引擎 weasyprint:唯一支援目錄頁碼(target-counter)與頁尾頁碼(@bottom-center)的引擎
if uv run --with weasyprint python3 - "$HTML" "$PDF" 2>"$WLOG" <<'PY'
import sys
from weasyprint import HTML
HTML(filename=sys.argv[1]).write_pdf(sys.argv[2])
PY
then
  if [ -s "$PDF" ]; then
    echo "wrote $PDF"
    exit 0
  fi
fi
echo "WARN: weasyprint failed (last 3 log lines below) — falling back to chromium (PDF will lack page numbers)" >&2
tail -3 "$WLOG" >&2 || true

# `|| true` is required: under `set -e`, if both `command -v` calls fail,
# the assignment's command substitution would abort the script here (exit 1,
# no WARN) before the `if` below ever runs — that breaks the spec §10
# soft-fail contract (exit 2 + WARN, keep the .md as deliverable).
CHROME="$(command -v chromium || command -v chromium-browser || true)"
if [ -z "$CHROME" ]; then
  echo "WARN: no PDF engine found — skipping PDF, REPORT.md is the deliverable" >&2
  exit 2
fi
"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$PDF" "file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")" 2>/dev/null
# snap chromium can exit 0 WITHOUT writing the PDF when the output path is
# outside its sandbox (e.g. this session's /tmp scratchpad) — verify the file
# actually exists and is non-empty before declaring success. exit 3
# distinguishes this from exit 2 (no engine at all).
if [ ! -s "$PDF" ]; then
  echo "ERROR: engine exited 0 but no PDF was written at $PDF (sandbox path restriction?)" >&2
  exit 3
fi
echo "wrote $PDF"
