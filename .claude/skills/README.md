# Kaggle Agent Skills — 封裝與安裝說明

本目錄是本專案的三個 Claude Code skill,可整組複製到別的 Claude Code 環境使用。
三者分工如下,通常一起安裝。

## 三個 skill

| skill | 作用 | 觸發時機 |
|-------|------|----------|
| **kaggle-agent** | 競賽核心流程:EDA→CV 設計→特徵→建模(LGB/XGB/CAT)→評估→提交 六階段操作手冊 | 開始/接手一場 Kaggle 表格競賽 |
| **kaggle-agent-self-improvement** | 在 kaggle-agent 之上加「線性自我迭代 + 跨競賽經驗庫」(經驗庫先驗、Optuna、seed bagging、反思回退) | 要對一場競賽做逐輪自我改進、跨賽累積經驗 |
| **kaggle-report** | 從實驗原始紀錄自動產生結構化報告:`collect.py` 抽數字→`verify_report.py` 數字追溯閘門→`md2pdf.sh` 出含目錄/頁碼 PDF | 一場競賽跑完、要產出可審查可重現的分析報告 |

**關係**:`kaggle-agent-self-improvement` 是 `kaggle-agent` 的超集(references/assets 幾乎相同,
多一份 `07_self_improvement.md`)。若只要基本流程用前者;要自我迭代+經驗庫用後者。`kaggle-report`
獨立、與前兩者搭配使用。

## 安裝

1. **複製 skill**:把要用的 skill 目錄整個複製到目標環境的 `.claude/skills/` 下,例如
   ```bash
   cp -r .claude/skills/kaggle-agent          <目標專案>/.claude/skills/
   cp -r .claude/skills/kaggle-report          <目標專案>/.claude/skills/
   # 要自我迭代版就再複製 kaggle-agent-self-improvement
   ```
   Claude Code 會自動偵測 `.claude/skills/<name>/SKILL.md` 並依其 `description`/觸發詞啟用。

2. **Python 環境(以 uv 管理)**:skill 的腳本以 `uv run python3` 執行。目標專案需有
   `pyproject.toml` 含下列相依,`uv sync` 即可:
   ```
   核心建模:lightgbm xgboost catboost scikit-learn optuna
   資料處理:pandas numpy pyyaml tqdm
   Kaggle:  kaggle(CLI;提交用)
   ```
   完整清單見本專案 `pyproject.toml`。

3. **報告 PDF 相依(僅 kaggle-report 需要)**:`md2pdf.sh` 首選 **weasyprint**(唯一支援目錄
   頁碼 target-counter + 頁尾),備援 **chromium**;另需 python 的 **markdown** 套件。三者缺
   weasyprint 時會自動降級用 chromium(無頁碼)。

4. **Kaggle 憑證(僅提交需要)**:以環境變數注入,**絕不落檔**:
   ```bash
   export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
   ```
   詳見專案根 `CLAUDE.md` 的安全規則。

## 目錄結構(每個 skill 內)

```
<skill>/
├── SKILL.md              # 定義:name、description、觸發詞、流程總覽
├── references/           # 各階段詳細操作說明(01_setup … 07_*)
└── assets/
    ├── templates/        # eda / feature / train / submit 腳本骨架
    └── utils/            # data_loader / evaluation / experiment_log / kaggle_auth
```
kaggle-report 的 `assets/` 另含 `collect.py`、`verify_report.py`、`md2pdf.sh`、
`report_style.css`、`report_template.md`、`eda_summary.py`。

## 版本與相依環境

- skill 源檔已納入本 repo 版控;`__pycache__/`、`*.pyc` 由 `.gitignore` 排除。
- 隨機性:建模腳本固定 seed;跨進程逐位重現需固定 LightGBM `deterministic=True,
  force_row_wise=True, num_threads=<固定>`(見 `knowledge/experience.md`)。
- 本機為 arm64 Linux;上列純 Python/有 arm64 wheel 的套件可 `uv sync` 裝起,無需 Docker。
