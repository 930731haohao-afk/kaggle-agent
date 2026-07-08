# Kaggle AI Agent — 混合式自主競賽 agent

用「Claude Code(LLM 推理)+ AutoML 工具(LightGBM/XGBoost/CatBoost)」組成的混合式 AI
agent,自主走完 Kaggle 表格競賽的完整流程(理解題目 → EDA → CV 設計 → 特徵 → 建模 →
集成 → 提交),並以**受控消融**量化每項自主能力的貢獻。方法對齊 Aygün 等人(2026,
Nature)的 ERA 系統。

## 從哪開始讀

| 想看 | 讀這份 |
|------|--------|
| 5 分鐘看懂全案成果 | [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)(成果簡報) |
| 共同方法、工具、專業術語、階段定義 | [docs/PREFACE.md](docs/PREFACE.md)(前言) |
| 15 場逐階段對照與跨場結論 | [docs/SUMMARY_REPORT.md](docs/SUMMARY_REPORT.md)(總結報告) |
| 研究式綜合(方法+結果+誠實限制) | [docs/TECH_REPORT.md](docs/TECH_REPORT.md)(技術報告草稿) |
| 單一場競賽的完整分析 | `competitions/playground-series-<賽>/REPORT.md` |
| agent 怎麼操作競賽 | [.claude/skills/kaggle-agent/SKILL.md](.claude/skills/kaggle-agent/SKILL.md) |

深入研究文件:統計顯著性 [docs/statistical_rigor.md](docs/statistical_rigor.md);外部注入
歸因發現 [docs/phase_j_j3_findings.md](docs/phase_j_j3_findings.md)。

每份報告都有對應的 PDF(含目錄與頁碼)。

## 核心方法:五階段消融

同一場競賽解多次、一次多給一項能力,再比較分數,就能逐段隔離每項能力的貢獻:

1. **階段 1** 無 skill 基線 → 2. **+ kaggle-agent skill**(結構化六階段)→
3. **+ 線性自我迭代**(經驗庫先驗、Optuna、seed bagging)→
4. **+ 樹搜尋**(候選樹取代線性單路徑;ERA 第一支柱)→
5. **+ 外部想法注入**(文獻想法庫;ERA 第二支柱,注入 hook 建置完成;J-3 歸因發現先驗尚未被搜尋消費,接線進行中——見 docs/phase_j_j3_findings.md)

## 主要成果

- **15 場競賽**(10 場同季 S3 主 benchmark + 5 場跨季 S4–S6、五種指標)的階段 1→4 階梯
  **場場成立**,無一場在任一階整體倒退。
- 唯一有真實 Kaggle 排行榜對照的 s3e16,樹搜尋版兩榜同方向改善——外部信度證據。
- 跨季配方仍成立(增幅收斂);跨場最佳解全數通過 OOF 逐位重現閘門(部分達位元級)。

## repo 結構

```
competitions/playground-series-<賽>/   各場工作區:data/(gitignore)、scripts/、
                                        experiments.json(原始紀錄)、facts.json、
                                        REPORT.md/pdf、STATUS.md、config.yaml
tree_search/                            自建樹搜尋:harness_v2/v3/v4 + 各場 driver
                                        run_<賽>_v3.py + eval_<賽>.py
knowledge/                              experience.md(內部經驗庫 [INT])、
                                        idea_bank.md(外部想法庫 [EXT])
.claude/skills/                         kaggle-agent、kaggle-agent-self-improvement、
                                        kaggle-report 三個 skill
docs/                                   前言、總結報告、成果簡報、實驗日誌、
                                        benchmark_facts.json + build_benchmark_table.py
CLAUDE.md                               專案指令(含 uv 套件管理、Kaggle CLI 設定)
```

## 如何重現

所有 Python 以 **uv** 執行(見 CLAUDE.md)。

```bash
# 重建跨場 benchmark 事實表(從各場 experiments.json 抽取,含一致性 assert)
uv run python3 docs/scripts/build_benchmark_table.py

# 驗證任一報告的數字全部可追溯(閘門,exit 0 = 通過)
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    docs/SUMMARY_REPORT.md docs/benchmark_facts.json

# 由 Markdown 產生含目錄/頁碼的 PDF
bash .claude/skills/kaggle-report/assets/md2pdf.sh docs/SUMMARY_REPORT.md docs/SUMMARY_REPORT.pdf

# 重現單場最佳解(含 OOF 逐位重現閘門,通過才產提交檔),以 s5e10 為例
uv run python3 competitions/playground-series-s5e10/scripts/06_rebuild_tree_best.py
```

各場「從原始資料到最佳解」的完整指令見該場 REPORT.md 第 7 節。

## 如何使用 agent

在 Claude Code 中觸發 kaggle-agent skill(觸發詞與流程見
[.claude/skills/kaggle-agent/SKILL.md](.claude/skills/kaggle-agent/SKILL.md)),指向一個
競賽工作區即可;人在每一階段都能介入、覆寫決策。報告產生見 kaggle-report skill。

三個 skill 的**封裝、安裝步驟與相依**見 [.claude/skills/README.md](.claude/skills/README.md)。

## 資料與憑證

- 競賽資料放各場 `data/`,已 gitignore,不入版控。
- Kaggle API token 一律以環境變數 `KAGGLE_API_TOKEN` 注入,**絕不落檔**(見 CLAUDE.md 安全規則)。
