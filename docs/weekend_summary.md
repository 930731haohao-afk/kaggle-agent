# 週末自主執行總結(2026-07-03 五晚 → 07-04 六午)

> 使用者下班前授權自主執行 2–3 天有意義的專案工作。實際上以約 19 小時連續自主運行完成了原定計畫(Phase A–C)並超前擴展(Phase D–H)。本文為總覽;所有數字之權威出處為各競賽 `experiments.json`/`experiments_tree*.json` 與 `docs/` 下各報告(皆經 verify_report.py 驗證)。

## 一句話總結

**8 場基線題補齊完整 skill 流程並全數擊敗通用基線;10 場自我改進迭代蒸餾出 52+ 條證據型經驗庫;樹搜尋原型從 v1 迭代到 v3,對線性迭代取得 9 勝 1 精確平,並正式接入 kaggle-agent skill 成為 Stage 4 預設迴圈。**

## 各 Phase 完成內容

| Phase | 內容 | 關鍵成果 |
|---|---|---|
| A(8 單元) | 8 場批次題完整 skill 六階段 + 報告 | 8/8 勝 generic 基線;每場 REPORT.md/PDF 過 verify |
| B(11 單元) | 經驗庫 + 10 場自我改進迭代 | knowledge/experience.md(52 條證據型);9/10 場再改善 |
| C(4 單元) | benchmark 彙總 + 樹搜尋 v1 原型(3 場) | v1:1 勝 1 平 1 負 → 找到 ensemble 節點空間缺口 |
| D(7 單元) | harness v2(4 升級)+ 5 場掃描 | **5/5 勝**;「先驗定下限、在地洞見定上限」模式 |
| E(5 單元) | v2 補完(復仇/平反/酸性/結構)+ 規模實驗 | s3e9 精確追平;s3e5 平轉勝;s3e16 穿越取整邊界;80 節點曲線 → 預算規則 |
| F(3 單元) | harness v3(預算相位機等 6 特性)+ 驗證 + 最終報告 | v3 兩場再刷新紀錄;burst+mega-blend 3 戰 3 勝 |
| G(2 單元) | 收割:樹最佳入帳 experiments.json + 全報告更新 + benchmark tier-4 | 四層對照表;一切成果可從正式紀錄回溯 |
| H(2 單元) | v3 上線三需求(resume/timeout/sanity gate)+ skill 接入 | 82 測試全綠;references/07_tree_search.md;Stage 4 正式升級 |

## Benchmark 四層對照(tier1 generic → tier4 樹搜尋後,相對改善)

s3e20 **+25.70%**、s3e5 **+19.21%**、s3e19 +4.11%(t2→t4)、s3e9 +3.77%(t4=t3 精確平)、s3e3 +3.53%、s3e16 +1.39%、s3e1 +0.95%、s3e11 +0.66%、s3e14 +0.31%、s3e7 +0.18%。
詳表與口徑注意事項見 `docs/benchmark_summary.md`(+PDF)。

## 三個可重現的研究發現(詳見 docs/tree_search_prototype.md)

1. **先驗定下限、在地洞見定上限**:經驗庫先驗讓搜尋不走死路(多場 informed 勝率顯著高於 uninformed),但每場最大的單筆增益幾乎都來自該場自己的未驗證假設(冗餘欄剪枝、top-code clip、auto_scale、深度邊界推)。
2. **強制 explore burst + mega-blend**:三次觸發三次貢獻 100% 的後期增益(E-5、F-2×2)——探索爆發不是選配。
3. **邊界推(boundary-push)**:Optuna 最優卡在搜尋盒邊緣是常態而非例外(≥3 場),把「推過邊界」做成標準 mutation 有實質回報。

## 誠實但書(重要)

- **除 s3e16 有真實 LB 錨點(Public 1.34356/Private 1.34075,提交於樹搜尋之前)外,全部分數皆為本地 CV/OOF**,未提交 Kaggle(未動任何憑證)。樹搜尋 blend 未產出測試集預測,報告中皆標「未提交/CV-only」。
- s3e19 的 9.757 帶 fold-5 double-dip 與 OOF 上擬合縮放之樂觀偏差(紀錄在案);s3e20 的 21.03 blend 為低信度未入帳(入帳的是穩健的 21.0589 純結構節點)。
- v3 的自動停止在實戰兩跑皆由硬上限收場(burst 持續改善屬正確行為),耐心路徑僅在小預算煙霧驗證過。
- Phase B 曾將 s3e20 的 experiments.json 就地遷移 v2(偏離「舊紀錄不改」原則;已驗證無資料損失)。

## 主要交付物索引

- `docs/benchmark_summary.md/.pdf` — 十場四層對照
- `docs/tree_search_prototype.md/.pdf` — 樹搜尋 v1→v3 全弧(15 次執行)
- `docs/scaling_experiment.md/.pdf` — 80 節點規模曲線與預算規則
- `knowledge/experience.md` — 跨競賽經驗庫(證據型)
- `tree_search/harness_v3.py` + 82 綠測試 — 可上線的樹搜尋引擎
- `.claude/skills/*/references/07_tree_search.md` — Stage 4 操作手冊(兩 skill 同步)
- 各競賽資料夾:更新後的 REPORT.md/PDF、STATUS.md、experiments.json(樹最佳已入帳)

## 給下週的建議

1. **給老師看的順序**:benchmark_summary(成果)→ tree_search_prototype(研究)→ 任一場 REPORT.pdf(流程品質)。這直接對應計畫書目標一/二/三/四的證據。
2. 若要拿 LB 驗證樹搜尋增益:挑 s3e16(Late Submission 開放、有歷史錨點),為樹最佳 blend 產測試預測後提交一次即可對照。
3. v3 後續(非急):auto-stop 耐心路徑的實戰驗證;經驗庫 prior 的自動化注入(ERA 想法注入完整版);跨場樹遷移。
4. 計畫書第 4–5 週(報告模組)與第 6–7 週(樹搜尋)的核心工作已於本週末提前完成,時程可重新配置(例如轉向論文 16 場 benchmark 的其餘場次、或報告 rubric 的人工驗收輪)。
