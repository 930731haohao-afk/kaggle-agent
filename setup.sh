#!/usr/bin/env bash
# =============================================================================
# setup.sh — Kaggle AI Agent one-command environment build + self-check
#            (the lightweight alternative to Docker)
#
# Goal: let anyone rebuild, on their own machine, a Python environment
#       bit-consistent with this project from the frozen uv.lock, and verify
#       on the spot that the core ML stack imports — no Docker required.
#
# Usage:  bash setup.sh          # build the core environment + self-check
#         bash setup.sh --torch  # additionally install torch (image/NLP
#                                #   competitions only, see notes below)
#
# Full reproduction instructions: REPRODUCE.md.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- colors (disabled when not a TTY) ---------------------------------------
if [ -t 1 ]; then
    G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'
else
    G=''; Y=''; R=''; B=''; N=''
fi
ok()   { echo "${G}✔${N} $*"; }
warn() { echo "${Y}⚠${N} $*"; }
die()  { echo "${R}✗${N} $*" >&2; exit 1; }
step() { echo; echo "${B}==> $*${N}"; }

WITH_TORCH=0
[ "${1:-}" = "--torch" ] && WITH_TORCH=1

# ---- 1. find uv -------------------------------------------------------------
step "1/5 check uv"
if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    die "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh
     then restart the shell or run  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
ok "uv version $($UV --version)"

# ---- 2. platform hint (arm64 is this project's known variable) --------------
step "2/5 platform check"
ARCH="$(uname -m)"
OS="$(uname -s)"
echo "    platform: $OS / $ARCH"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ok "arm64: same as this project's development machine; uv.lock should work as is"
else
    warn "not arm64: uv.lock will select different wheels. The core ML packages
     (lightgbm/xgboost/catboost/sklearn) ship prebuilt x86_64 wheels too, so this is
     usually fine; if a package needs on-the-spot compilation and fails, it is most
     likely a missing compiler or system dev library — install what the error names."
fi

# ---- 3. uv sync: deterministic build from uv.lock ---------------------------
step "3/5 build the environment (uv sync --inexact, versions frozen by uv.lock)"
# Needs Python 3.13 (pyproject: >=3.13,<3.14); uv downloads it automatically.
# --inexact: only add what the lockfile lists, never remove packages outside it
#            (torch is installed separately outside the lock — a plain uv sync
#            would delete it; see the pyproject.toml note and REPRODUCE.md).
$UV sync --inexact
ok "uv sync complete (.venv ready)"

# ---- 4. self-check: core ML stack imports + versions ------------------------
step "4/5 self-check: core ML stack"
$UV run python3 - <<'PY'
import importlib, sys
mods = ["numpy","pandas","sklearn","lightgbm","xgboost","catboost","optuna","yaml","tqdm"]
bad = []
print(f"    python {sys.version.split()[0]}")
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"    OK  {m:12s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"    ERR {m:12s} {e}")
        bad.append(m)
if bad:
    raise SystemExit(f"import failures: {bad}")
PY
ok "core stack fully importable"

# ---- 5. optional: torch (image/NLP only) / PDF backend hints ----------------
step "5/5 optional dependency hints"

# torch is not in uv.lock (triton resolution issue); the core tabular results
# do not need it.
if [ "$WITH_TORCH" = "1" ]; then
    warn "installing torch (GPU nightly; may be slow or need another index on arm64/no-CUDA)"
    $UV pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130 \
        || warn "torch install failed — only image/NLP competitions need it; core tabular reproduction is unaffected"
elif $UV run python3 -c "import torch" >/dev/null 2>&1; then
    ok "torch already present ($($UV run python3 -c 'import torch;print(torch.__version__)' 2>/dev/null))"
else
    echo "    torch not installed (not in uv.lock). The core tabular benchmark and the"
    echo "    tree search use only lightgbm/xgboost/catboost and do not need torch."
    echo "    Image/NLP competitions do; when needed run:  bash setup.sh --torch"
fi

# PDF: optional backend for report generation
echo
if $UV run python3 -c "import weasyprint" >/dev/null 2>&1; then
    ok "weasyprint available (PDFs get a TOC with page numbers)"
elif command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1; then
    warn "no weasyprint, but chromium found: md2pdf.sh degrades to chromium (PDF without page numbers)"
else
    warn "neither weasyprint nor chromium: report .md files still build, but no PDF.
     For PDFs install one of:  uv pip install weasyprint   or a system chromium"
fi

# ---- done -------------------------------------------------------------------
echo
ok "${B}Environment ready.${N}"
echo
echo "Next steps (details in REPRODUCE.md):"
echo "  • run the test suite:        uv run pytest -q"
echo "  • rebuild the benchmark:     uv run python3 docs/scripts/build_benchmark_table.py"
echo "  • reproduce one best solution: each competition's scripts/ (04→05→06_rebuild_tree_best.py) + the tree_search/ driver; see REPRODUCE.md"
echo "  • use the agent:             trigger the kaggle-agent skill in Claude Code"
