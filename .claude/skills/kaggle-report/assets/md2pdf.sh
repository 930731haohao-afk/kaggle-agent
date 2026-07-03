#!/usr/bin/env bash
# Convert a markdown report to PDF via python-markdown + chromium headless.
# Usage: bash md2pdf.sh <REPORT.md> [out.pdf]
set -euo pipefail

MD="$1"
PDF="${2:-${MD%.md}.pdf}"
CSS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/report_style.css"
HTML="${MD%.md}.tmp.html"   # 不用隱藏檔名:snap chromium 讀不到部分隱藏路徑

uv run python3 - "$MD" "$CSS" "$HTML" <<'PY'
import pathlib, sys
import markdown
md_path, css_path, html_path = sys.argv[1:4]
body = markdown.markdown(
    pathlib.Path(md_path).read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"])
css = pathlib.Path(css_path).read_text(encoding="utf-8")
pathlib.Path(html_path).write_text(
    f"<!doctype html><html><head><meta charset='utf-8'>"
    f"<style>{css}</style></head><body>{body}</body></html>",
    encoding="utf-8")
PY

# `|| true` is required: under `set -e`, if both `command -v` calls fail,
# the assignment's command substitution would abort the script here (exit 1,
# no WARN) before the `if` below ever runs — that breaks the spec §10
# soft-fail contract (exit 2 + WARN, keep the .md as deliverable).
CHROME="$(command -v chromium || command -v chromium-browser || true)"
if [ -z "$CHROME" ]; then
  rm -f "$HTML"
  echo "WARN: chromium not found — skipping PDF, REPORT.md is the deliverable" >&2
  exit 2
fi
"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$PDF" "file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")" 2>/dev/null
rm -f "$HTML"
echo "wrote $PDF"
