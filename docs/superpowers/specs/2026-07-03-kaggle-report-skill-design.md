# 設計文件:`kaggle-report` skill — 競賽報告自動生成

- **日期**:2026-07-03
- **狀態**:已與使用者確認設計,待實作規劃
- **出處**:暑期實習計畫書(`~/Desktop/Kaggle_AI.pdf`)目標一(報告自動生成)、目標二(報告品質)、目標四(決策可追溯);第三階段(第 4–5 週)報告生成模組

## 1. 問題

現有 kaggle agent 跑完競賽只留下 `experiments.json`、`STATUS.md`、腳本;「做了什麼、為何這樣做」的報告需人工事後撰寫。此外 `experiments.json` 的 schema 不一致(至少三種寫法並存:skill 流程手刻 dict、`run_competition.py` 手刻 dict、無人使用的 `experiment_log.py` 正式格式),自動彙整無可靠依據。

## 2. 已確認的設計決定

| 決定點 | 選擇 | 理由 |
|---|---|---|
| 產生機制 | **混合式**:Python 決定性抽取數字,LLM 只寫敘述 | 同時滿足「數字可信可重現」(目標二/四)與「what/why 有內容」(目標一);純 LLM 會抄錯數字,純模板寫不出 why |
| 輸出格式 | **Markdown 為主 + PDF**(REPORT.md 放競賽資料夾,另轉 PDF) | 對齊計畫書「markdown/PDF 報告草稿」;MD 可版控可 diff |
| schema 策略 | **定正規格式 + 容錯讀取**:升級 `experiment_log.py` 為 v2 正規 schema、SKILL.md 強制使用;`collect.py` 以容錯 adapter 讀舊資料 | 治本(未來一致)且不動歷史紀錄(零遷移風險) |
| rubric 位置 | 內建於 skill 的 `references/rubric.md` | 慣例做法 |

## 3. 架構與資料流

```
使用者:「幫 <competition> 產報告」
        │
        ▼
[1] collect.py(Python,決定性)
    輸入:competitions/<name>/{config.yaml, experiments.json, STATUS.md?}
    輸出:facts.json(所有數字/表格/實驗軌跡;缺漏欄位明確標 missing)
        │
        ▼
[2] agent 撰寫敘述(LLM)
    讀 facts.json + references/report_structure.md
    只寫 what/why 敘述;數字表格自 facts.json 原樣嵌入,禁止改寫
    輸出:competitions/<name>/REPORT.md
        │
        ▼
[3] rubric 自我檢核(LLM)
    逐項核對 references/rubric.md,缺漏即補
        │
        ▼
[4] md2pdf(決定性)
    REPORT.md → HTML(report_style.css)→ chromium headless --print-to-pdf
    輸出:competitions/<name>/REPORT.pdf
```

單一資料流原則:**LLM 不直接接觸 experiments.json 的數字;一切數字經由 facts.json。**

## 4. 檔案佈局

```
.claude/skills/kaggle-report/
├── SKILL.md                    # 流程定義、觸發條件、行為守則(禁改寫數字等)
├── references/
│   ├── report_structure.md     # 報告 8 節結構 + 各節內容規格
│   └── rubric.md               # 驗收檢核表(對應目標二)
└── assets/
    ├── collect.py              # 事實抽取器(含 3 種舊格式容錯 adapter)
    ├── report_template.md      # REPORT.md 骨架(節標題 + 佔位符)
    ├── report_style.css        # PDF 樣式(需支援中文)
    └── md2pdf.sh               # markdown → PDF 轉換腳本
```

另修改既有檔案(治本部分):

- `.claude/skills/kaggle-agent/assets/utils/experiment_log.py` → **v2 schema**(見 §6)
- `.claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py` → 同步
- 兩個 kaggle-agent SKILL.md → 增加硬規則:「實驗紀錄一律呼叫 `experiment_log.py`,禁止手刻 dict」
- `competitions/run_competition.py` → 改用 v2 schema 寫入

## 5. 報告結構(8 節)

計畫書五大流程元件 + 三個補強:

1. **競賽目的** — what(解什麼問題)+ why(為何重要);由 config.yaml 的 description/notes 與評估指標反推
2. **資料規格** — 列數、欄位、型別、缺失、分布重點、train/test 一致性
3. **模型規格** — 模型清單、ensemble/blend 權重、選型理由
4. **訓練規格** — CV 方案、fold 數、seed、objective、關鍵超參
5. **推論程序** — 預測後處理(如 round/clip)、submission 格式與驗證
6. **評估指標** — 指標定義、OOF 分數、Public/Private LB、CV↔LB gap 分析
7. **實驗軌跡** — 逐實驗分數演進、突破點(breakthrough)及其成因(目標四)
8. **重現指令** — 從資料下載到 submission 的完整命令序列

## 6. experiments.json v2 正規 schema(草案)

```json
{
  "experiment_id": 3,
  "timestamp": "2026-07-03T09:18:40",
  "model": "LGB+XGB+CAT blend",
  "metric": "mae",
  "direction": "minimize",
  "n_features": 24,
  "features": ["..."],
  "cv": {"scheme": "stratified_kfold_agebin", "n_splits": 5, "seed": 42},
  "base_models": [
    {"name": "LGB", "score": 1.35651, "time_s": 30.8, "params": {}}
  ],
  "ensemble": {"weights": {"LGB": 0.6, "XGB": 0.3, "CAT": 0.1}, "score": 1.35589},
  "postprocess": ["round", "clip(1, 29)"],
  "score": 1.33812,
  "submission": "sub_blend_20260703_091840.csv",
  "leaderboard": {"public": 1.34356, "private": 1.34075, "submitted": "2026-07-03"},
  "notes": "本實驗的動機與變更點(選填)"
}
```

要點:`score` 為該實驗最終代表分數(經後處理);`base_models`/`ensemble`/`leaderboard`/`postprocess` 選填但格式固定;實作時以 v2 `log_experiment()` 的參數簽名為準,此處為欄位意圖。

`collect.py` 的容錯 adapter 需認得三種舊寫法並映射到 v2 欄位:

| 舊寫法 | 辨識特徵 | 映射 |
|---|---|---|
| s3e16 train.py 型 | 有 `base_models` + `blend_oof_mae` | `blend_oof_mae`→ensemble.score;`use_round`→postprocess |
| run_competition.py 型 | 有 `per_model` + `blend_score` | `per_model`→base_models;`blend_score`→ensemble.score |
| experiment_log.py v1 型 | 有 `cv_mean` + `cv_scores` | `cv_mean`→score;`cv_strategy`→cv.scheme |

無法辨識的紀錄:保留原文放入 facts.json 的 `unparsed` 清單並在報告中如實標注,不得丟棄或猜測。

## 7. facts.json(中間層)

`collect.py` 的唯一輸出。內容:競賽 metadata(來自 config.yaml)、正規化後的實驗列表(全部映射到 v2 欄位)、最佳實驗標記、分數軌跡(依時間排序)、素材等級判定、`missing` 清單(報告需要但缺席的欄位)、`unparsed` 清單。確切欄位於實作時定;原則:**報告中出現的每個數字都必須可回溯到 facts.json 的某欄位。**

## 8. 素材不足的處理

`collect.py` 判定素材等級:

- **full**:有 EDA/特徵工程紀錄與敘述素材(如 s3e16)→ 全 8 節
- **baseline-only**:僅通用批次基線紀錄(如 s3e1/3/5/7/9/11/14/19)→ 報告如實寫「本場僅執行通用基線,未進行 EDA 與特徵工程」,節 2 僅列基本形狀、節 7 僅一筆紀錄;**禁止 LLM 編造未發生的分析**

## 9. 驗證(對應計畫書 §5 報告驗證)

1. **數字一致性(自動)**:比對腳本核對 REPORT.md 中每個數字皆存在於 facts.json、facts.json 每個數字皆存在於原始 experiments.json/config.yaml
2. **rubric 檢核(LLM 自檢 + 人工抽查)**:8 節齊備、每節必填欄位齊備、重現指令可執行
3. **測試案例**:s3e16(full)+ 一場批次題(baseline-only),兩份報告皆須過 rubric
4. **回歸**:v2 `log_experiment()` 附單元測試;三種舊格式 adapter 各附一筆真實紀錄的解析測試

## 10. 錯誤處理

- `collect.py`:缺 config.yaml 或 experiments.json → 非零退出 + 明確訊息(缺什麼、去哪補)
- experiments.json 解析失敗(壞 JSON)→ 報錯不產出,不得靜默跳過
- chromium 不可用 → 保留 REPORT.md、跳過 PDF 並明確告知(MD 為主產物,PDF 失敗不算整體失敗)

## 11. 範圍外(YAGNI)

- 不做報告的 DOCX 輸出(需要時人工用既有 docx-js 工具鏈轉)
- 不做跨競賽彙總報告(單場報告先站穩)
- 不做舊 experiments.json 的就地遷移(容錯讀取已覆蓋)
- 不動樹搜尋/自我改進策略(計畫書第四階段,另案)
