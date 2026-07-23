#!/usr/bin/env bash
# =============================================================================
# setup.sh — Kaggle AI Agent 一鍵環境建置 + 自檢(Docker 的輕量替代)
#
# 目標:讓別人在自己的機器上,用凍結的 uv.lock 重建與本專案「位元級一致」的
#       Python 環境,並當場驗證核心 ML 堆疊可用——不需要 Docker。
#
# 用法:  bash setup.sh          # 建置核心環境 + 自檢
#         bash setup.sh --torch  # 額外裝 torch(僅影像/NLP 競賽需要,見下方說明)
#
# 完整重現流程見 REPRODUCE.md。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- 顏色(非 TTY 時關閉)---------------------------------------------------
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

# ---- 1. 找到 uv ------------------------------------------------------------
step "1/5 檢查 uv"
if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    die "找不到 uv。請先安裝:curl -LsSf https://astral.sh/uv/install.sh | sh
     然後重開 shell 或執行  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
ok "uv 版本 $($UV --version)"

# ---- 2. 平台提示(arm64 是本專案的已知變數)-------------------------------
step "2/5 平台檢查"
ARCH="$(uname -m)"
OS="$(uname -s)"
echo "    平台:$OS / $ARCH"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    ok "arm64:與本專案開發機相同,uv.lock 應直接可用"
else
    warn "非 arm64:uv.lock 的 wheel 選擇會不同。核心 ML 套件(lightgbm/xgboost/
     catboost/sklearn)在 x86_64 也有預編譯 wheel,通常無礙;若某套件需現場編譯而
     失敗,多半是缺編譯器或系統開發庫,依錯誤訊息補裝即可。"
fi

# ---- 3. uv sync:從 uv.lock 建置(決定性)---------------------------------
step "3/5 建置環境(uv sync --inexact,依 uv.lock 凍結版本)"
# 需要 Python 3.13(pyproject: >=3.13,<3.14);uv 會自動下載對應版本。
# --inexact:只補齊鎖檔內套件,不刪除鎖檔外套件(torch 是鎖檔外另裝的,
#            預設 uv sync 會把它清掉——見 pyproject.toml 註記與 REPRODUCE.md)。
$UV sync --inexact
ok "uv sync 完成(.venv 已就緒)"

# ---- 4. 自檢:核心 ML 堆疊匯入 + 版本 --------------------------------------
step "4/5 自檢:核心 ML 堆疊"
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
    raise SystemExit(f"匯入失敗:{bad}")
PY
ok "核心堆疊全部可匯入"

# ---- 5. 選配:torch(僅影像/NLP 競賽)/ PDF 後端提示 ----------------------
step "5/5 選配相依提示"

# torch 不在 uv.lock 內(triton 解析問題),核心表格結果不需要它。
if [ "$WITH_TORCH" = "1" ]; then
    warn "安裝 torch(GPU nightly;arm64/無 CUDA 時可能較久或需改 index)"
    $UV pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu130 \
        || warn "torch 安裝失敗——影像/NLP 競賽才需要,核心表格重現不受影響"
elif $UV run python3 -c "import torch" >/dev/null 2>&1; then
    ok "torch 已在環境中($($UV run python3 -c 'import torch;print(torch.__version__)' 2>/dev/null))"
else
    echo "    torch 未安裝(不在 uv.lock)。核心的 15 場表格 benchmark 與樹搜尋"
    echo "    只用 lightgbm/xgboost/catboost,不需要 torch。"
    echo "    影像/NLP 競賽才需要,屆時執行:  bash setup.sh --torch"
fi

# PDF:報告產生的選配後端
echo
if $UV run python3 -c "import weasyprint" >/dev/null 2>&1; then
    ok "weasyprint 可用(PDF 含目錄頁碼)"
elif command -v chromium >/dev/null 2>&1 || command -v chromium-browser >/dev/null 2>&1; then
    warn "無 weasyprint,但有 chromium:md2pdf.sh 會降級用 chromium(PDF 無頁碼)"
else
    warn "無 weasyprint 也無 chromium:報告 .md 仍可產,但無法出 PDF。
     要 PDF 請裝其一:uv pip install weasyprint  或  系統安裝 chromium"
fi

# ---- 完成 ------------------------------------------------------------------
echo
ok "${B}環境就緒。${N}"
echo
echo "接下來(詳見 REPRODUCE.md):"
echo "  • 跑測試驗證:      uv run pytest -q"
echo "  • 重建 benchmark:   uv run python3 docs/scripts/build_benchmark_table.py"
echo "  • 重現單場最佳解:   各場 scripts/(04→05→06_rebuild_tree_best.py)+ tree_search/ driver;見 REPRODUCE.md"
echo "  • 用 agent:         在 Claude Code 觸發 kaggle-agent skill"
