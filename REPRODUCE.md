# 重現說明(REPRODUCE)

本專案交付以**輕量、可重現**為目標,取代 Docker 容器化(理由見文末)。核心作法:
凍結的 `uv.lock` + 一支 `setup.sh` + 本說明。任何人照這份走,就能在自己的機器上
重建與本專案一致的環境並重現結果。

---

## 0. 前置需求

| 需求 | 說明 |
|------|------|
| **uv** | 唯一硬需求。`curl -LsSf https://astral.sh/uv/install.sh \| sh`,裝完 `export PATH="$HOME/.local/bin:$PATH"`。 |
| **Python 3.13** | 不必自己裝——`uv sync` 會依 `.python-version`(3.13)自動下載對應版本。 |
| 平台 | 開發機為 **arm64 Linux(Ubuntu 24.04)**。x86_64 亦可(見「已知限制」)。 |
| Kaggle 憑證 | **僅下載資料/提交時需要**;純重現本地 CV 分數不需要。 |
| torch | **僅影像/NLP 競賽需要**,核心表格結果不需要。 |

---

## 1. 一鍵建置 + 自檢

```bash
bash setup.sh
```

`setup.sh` 會依序:找到 uv → 印平台/arm64 提示 → `uv sync`(依 `uv.lock` 凍結版本
建置 `.venv`)→ 自檢核心 ML 堆疊可匯入並印版本 → 提示 torch / PDF 後端狀態。
全程冪等,可重複執行。

要連影像/NLP 用的 torch 一起裝:

```bash
bash setup.sh --torch
```

預期自檢輸出(版本以 uv.lock 為準):

```
python 3.13.x
OK  numpy  2.4.2   OK  pandas 3.0.x   OK  sklearn 1.8.0
OK  lightgbm 4.6.0  OK  xgboost 3.2.x  OK  catboost 1.2.10
OK  optuna 4.9.0    OK  yaml 6.0.3     OK  tqdm 4.67.3
```

---

## 2. 重現結果(由淺到深)

```bash
# (a) 全套測試——最快的整體健康檢查
uv run pytest -q

# (b) 重建跨場 benchmark 事實表(從各場 experiments.json 抽取,含一致性 assert)
uv run python3 docs/scripts/build_benchmark_table.py

# (c) 驗證任一報告的數字全部可追溯(閘門,exit 0 = 通過)
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    docs/SUMMARY_REPORT.md docs/benchmark_facts.json

# (d) 重現單場最佳解(含 OOF 逐位重現閘門,通過才產提交檔),以 s5e10 為例
uv run python3 competitions/playground-series-s5e10/scripts/06_rebuild_tree_best.py

# (e) 由 Markdown 產生含目錄/頁碼的 PDF
bash .claude/skills/kaggle-report/assets/md2pdf.sh \
    docs/SUMMARY_REPORT.md docs/SUMMARY_REPORT.pdf
```

> (b)–(e) 從 `experiments.json` / 已快取的 OOF 出發,**不需要競賽原始資料**。
> (d) 若要從原始資料重跑,需先下載該場 `data/`(見第 3 節);各場「從原始資料到
> 最佳解」的完整指令見該場 `REPORT.md` 第 7 節。

---

## 3. 資料與憑證(僅需要下載/提交時)

```bash
# Kaggle token 一律以環境變數注入,絕不落檔(見 CLAUDE.md 安全規則)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')

# 下載某場資料到該場 data/(已 gitignore,不入版控)
uv run kaggle competitions download -c playground-series-s5e10 \
    -p competitions/playground-series-s5e10/data
```

---

## 4. 已知限制(相對於 Docker 的取捨)

輕量替代凍結了 **Python 版本 + 所有套件版本**(`uv.lock`),但**不凍結作業系統與
系統函式庫**。實務影響:

- **arm64(開發機)**:`uv.lock` 直接可用,零風險。
- **x86_64**:核心 ML 套件在 x86_64 也有預編譯 wheel,通常無礙。若某套件在你的
  平台沒有 wheel 而需現場編譯失敗,多半是缺編譯器或系統開發庫,依錯誤補裝即可。
- **torch 不在 `uv.lock`**:因 triton 依賴解析問題另行安裝(`setup.sh --torch`),
  且僅影像/NLP 競賽需要。核心的 15 場表格 benchmark 與樹搜尋不依賴它。
- **PDF 後端**:`weasyprint`(首選,含目錄頁碼)或 `chromium`(備援,無頁碼);
  兩者皆無時報告 `.md` 仍可產,只是不出 PDF。

---

## 為什麼不用 Docker

目標(可重現交付)兩條路都能達成,權衡如下:

| | Docker 映像 | **本方案(uv.lock + setup.sh)** |
|---|---|---|
| 重現強度 | 最強(連 OS/系統庫凍結) | 強(凍結 Python 與全部套件版本) |
| arm64 風險 | 高:乾淨容器裡從頭裝 ML 套件,易踩 arm64 無 wheel → 現場編譯失敗 | 低:以「已知會動」的環境為基準 |
| 建置/交付 | 慢、image GB 級 | 快、交付物為數個文字檔 |
| 對方需要 | 裝 Docker | 裝 uv(單一執行檔) |

開發機為 arm64,ML 套件在乾淨 arm64 容器內建置有真實失敗風險;本方案以已跑通的
環境為基準,對「在類似機器上重現」的實際場景已足夠,故採之。
