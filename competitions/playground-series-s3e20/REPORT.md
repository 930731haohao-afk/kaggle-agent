# 競賽分析報告:playground-series-s3e20

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:以盧安達境內 497 個地理位置在 2019–2021 年的衛星感測資料(SulphurDioxide、
CarbonMonoxide、Cloud、Aerosol、UV 等量測群組)與時空座標(latitude / longitude / year /
week_no),預測 2022 年各位置每週的二氧化碳排放量 `emission`——本質是「跨年外推」的
時空面板迴歸問題。

**Why**:區域級 CO2 排放估計是碳監測與環境政策的基礎,衛星遙測是缺乏地面監測站地區唯一
可規模化的資料來源。評估指標為 **RMSE(minimize)**:目標高度右偏(median 45.6、mean
81.9、max 3167.8),RMSE 的平方級懲罰使大排放源的誤差主導總分,與「抓準高排放熱點」的
實務需求一致。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e20 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | emission |
| 素材等級 | full |

> **誠實但書(本報告全文適用)**:本競賽為 2023 年已關閉的舊競賽,Late Submission 已關閉,
> 任何預測皆無法上傳;全場所有分數均為本機 Leave-One-Year-Out OOF(CV-only),leaderboard
> 無紀錄(列於 facts.missing)。

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基模型:exp1–2 為主力,exp3 起作為 blend 成員與結構訊號的對照組(權重搜尋最終全數歸零) |
| 自建樹搜尋 harness(v2) | Phase E-4 把結構管線的純量超參當節點空間搜尋,於 node #28 找到 exp9 的純結構最優 |
| Leave-One-Year-Out CV 框架 | 2019/2020/2021 逐年留出的 3 折驗證,exp3 起全程同折 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——發現「emission 對 (位置, 週) 跨年近乎
恆定」的結構訊號、把改進重心從 GBDT 轉向歷史均值去噪、依協定於診斷輪停損;Auto-ML 工具
(權重搜尋、樹搜尋 harness)負責系統化執行組合空間探索,兩者分工互補。本場未使用 Optuna
(結構管線超參以網格 pre-sweep 與樹搜尋調校)。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 79,023 |
| test 列數 | 24,353 |
| 原始欄位數 | 74 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

特徵全為數值型、無類別欄位:時空座標 4 欄 + 衛星感測欄位(Cloud、Aerosol、UV 等群組)。
目標 `emission` 重度右偏(mean 81.940552、median 45.593445、skew 10.173826、max
3167.768),為 log 轉換候選,訓練一律取 log1p;train 無重複列。

> **誠實但書**:facts.eda 記錄欄位數 74;competition.notes 與 STATUS.md 敘述則為「75 個
> 特徵」,兩者口徑不同(前者為 EDA 腳本實際掃描之特徵欄計數)。本報告以 facts.eda 為準。

資料品質重點在缺失:UvAerosolLayerHeight 群組 7 欄缺失 78,584/79,023 列(notes 記為
99.4%),v2 起直接刪除,保留 63 個感測欄;SulphurDioxide/NitrogenDioxide 群組亦有中度
缺失。感測欄位間高共線對眾多(各群組 solar_azimuth_angle 兩兩相關逾 0.98),且單一感測
特徵與目標的相關性極弱——與後續「感測特徵對此目標近乎噪音」的建模結論一致。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:RMSE 無取整/門檻類後處理,各實驗「原始 OOF」即「決策
> 分數」,兩欄同值;exp1–2 為 2026-02-14 舊 session 的單一時間切分結果,與 exp3 起的
> Leave-One-Year-Out CV **不可直接比較**,僅列入軌跡供完整性。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | v1 舊 session | LGB/XGB/Ridge 權重 blend | 102 | 38.507172 | 38.507172 | 否,Ridge 失敗、CV 與後續不可比 |
| 2 | v1 舊 session | LGB/XGB(50/50) | 無紀錄 | 33.205568 | 33.205568 | 否,同上 |
| 3 | v2(LOYO 起點) | LGB/XGB/CAT 三 GBDT blend(0/.75/.25) | 70 + TE 欄 | 28.3424 | 28.3424 | 基線參照(tier1 代理) |
| 4 | v2(TE 入 pool) | 三 GBDT + TE(loc-week 均值)四成員 blend | 70 + TE 欄 | 22.6488 | 22.6488 | 是,權重搜尋給 TE 100% |
| 5 | Phase B R1 | 同 exp4,TE 加經驗貝葉斯收縮(α=0.935) | 70 + TE 欄 | 22.4897 | 22.4897 | 是 |
| 6 | Phase B R2 | 同上,加 COVID 年降權(W2020=0.29,α=0.945) | 70 + TE 欄 | 21.6317 | 21.6317 | 是,線性階段最大單輪增益 |
| 7 | Phase B R3(線性終點) | 同上,加鄰週平滑(WNB=0.28,α 收斂至 1.0) | 70 + TE 欄 | 21.1487 | 21.1487 | 是,Phase-B 最終 |
| 8 | Phase B R4(診斷) | 寬窗平滑 + LGB 殘差模型(皆棄用) | 63 感測 + 時空欄 | 21.1487 | 21.1487 | —(診斷,依協定停止) |
| 9 | Phase E-4 樹搜尋(best) | 純結構 TE_JOINT(node #28) | 無紀錄 | **21.0589** | **21.0589** | 是,本場最佳 |

**best 成員表(exp9,樹搜尋 node #28,純結構單一成員)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| TE_JOINT | —(單一成員,無 blend) | **21.0589** | loc-week 歷史均值管線:alpha=0.99、w2020=0.2、wnb=0.3、window=1,加新軸 year_weights(2019: 0.6/2020: 0.2/2021: 1.2) |

選型脈絡:exp3 的純 GBDT blend 是本場 GBDT 能力上限;exp4 把「loc-week 歷史均值」以獨立
成員放入 pool,權重搜尋直接給它 100%、三個 GBDT 全歸零——確立訊號本質是結構而非感測特徵。
Phase-B(exp5–7)遂只對 TE 本身做三段去噪(收縮、COVID 降權、鄰週平滑),exp8 以寬窗與
殘差模型雙診斷確認結構已飽和可預測訊號後停止線性迭代。

exp9 由樹搜尋 harness 產出:重掃描已手調軸(W2020、WNB)僅誠實打平 Phase-B 既有最優;
增益來自(a)Phase-B 從未試過的新軸 year_weights(異常年降權推廣到 2019/2021),與
(b)JOINT lineage 把全部結構旋鈕一步共同移動,複合出 -0.0898 的總增益。

> **誠實但書**:facts.best(exp9)以 OOF 分數最小選出,為 OOF-only 樹搜尋結果——未產生
> test 預測、無 submission 檔,且本場 Late Submission 已關閉,實務上也無法提交。node #28
> 另有手足 BLEND 節點(21.0332,本報告與 benchmark 均**不採用**):其將診斷用 GBDT 以
> 2.17% 權重混入,僅在 3 折 LOYO CV、權重直接對同一份 OOF 擬合下才勝出,STATUS.md 明確
> 記為低信心、CV 噪音範圍內的邊際發現;採用純結構節點是刻意的保守方法論選擇,非疏漏。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1–2 | time-based 單一切分(2019–2020 train,2021 val) | 無紀錄 | 42 |
| 3–9 | Leave-One-Year-Out(2019/2020/2021) | 3 | 42 |

為何用 LOYO:test 為未來年度(2022),逐年留出忠實模擬「以其他年度歷史預測一整年」的
真實情境;TE 每折僅用其餘訓練年度計算,無洩漏。EDA 的 validation_hint 建議一般 KFold,
但時序外推結構優先於該一般性建議;exp3–9 全程同折,分數可直接比較。

Objective 與關鍵超參:GBDT 以 log1p(emission) 為目標訓練、預測後還原,詳細超參未入
facts.json 結構化欄位(無紀錄,不臆測);結構管線超參有完整紀錄——exp7 為 alpha=1.0、
w2020=0.24、wnb=0.28(網格 pre-sweep 選定),exp9 之聯合最優見 3.2 節 best 成員表。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | submission_ensemble_38.5072_20260214_161647.csv | 否 |
| 2 | 無後處理紀錄 | submission_final_33.2056_20260214_161743.csv | 否 |
| 3 | clip >= 0 | sub_v2_blend_28.3424_20260703_095953.csv | 否 |
| 4 | clip >= 0 | sub_v2_blend_22.6488_20260703_101209.csv | 否 |
| 5 | clip >= 0 | sub_v3_blend_22.4897_20260704_005826.csv | 否 |
| 6 | clip >= 0 | sub_v4_blend_21.6317_20260704_011624.csv | 否 |
| 7 | clip >= 0 | sub_v5_blend_21.1487_20260704_011931.csv | 否 |
| 8 | 無後處理紀錄 | 無紀錄(診斷輪) | 否 |
| 9 | 無後處理紀錄 | 無紀錄(OOF-only,未產生 test 預測) | 否 |

`clip >= 0` 為唯一後處理(排放量不可為負)。submission 檔格式兩欄:id 欄
`ID_LAT_LON_YEAR_WEEK`、目標欄 `emission`。因 Late Submission 已關閉,表中檔案皆為本機
artifact,「已提交」一律為否——包含 exp7 與 exp9 在內的全部 tier 分數均為 CV-only。

### 3.5 評估指標

指標定義:RMSE = 預測誤差平方均值的平方根,對大誤差施以平方級懲罰,越低越好。

| 項目 | OOF RMSE |
|------|----------|
| LGB(exp7 成員) | 32.074 |
| XGB(exp7 成員) | 28.0674 |
| CAT(exp7 成員) | 28.8466 |
| TE(exp7 成員,去噪 loc-week 均值) | 21.1487 |
| Ensemble(exp7,TE 權重 1.0) | 21.1487 |
| TE_JOINT(exp9,樹搜尋 best) | **21.0589** |
| Public / Private LB | 無紀錄(Late Submission 已關閉,未提交) |

本場無排行榜表(leaderboard 為 null,列於 facts.missing),CV↔LB gap 無法計算,不作
推測性比較。exp9 相對線性迭代終點 exp7 之改善:

```
exp7 − exp9:21.1487 − 21.0589 = 0.0898,相對改善 0.0898 / 21.1487 = 0.4246%
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase E-4 | 樹搜尋執行(harness v2) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 20260214_161647 | 38.507172 | v1 舊 session | 102 特徵 LGB/XGB/Ridge blend,Ridge 失敗、單一時間切分 |
| 2 | 20260214_161743 | 33.205568 | v1 舊 session | LGB/XGB 50/50,v1 終點 |
| 3 | 2026-07-03T09:59:53 | 28.3424 | v2(LOYO 起點) | 三 GBDT blend 於 LOYO CV,GBDT 能力上限(tier1 代理) |
| 4 | 2026-07-03T10:12:09 | 22.6488 | v2(TE 入 pool) | TE 成員拿 100% 權重,GBDT 全歸零 |
| 5 | 2026-07-04T00:58:26 | 22.4897 | Phase B R1 | TE 經驗貝葉斯收縮 |
| 6 | 2026-07-04T01:16:24 | 21.6317 | Phase B R2 | COVID 年降權,線性階段最大單輪增益 |
| 7 | 2026-07-04T01:19:31 | 21.1487 | Phase B R3 | 鄰週平滑,線性迭代終點 |
| 8 | 2026-07-04T01:35:00 | 21.1487 | Phase B R4(診斷) | 寬窗、殘差模型皆負向,依協定停止 |
| 9 | 2026-07-04T12:18:33 | **21.0589** | Phase E-4 樹搜尋 | node #28 純結構 JOINT 最優,本場最佳 |

- **突破點 1(exp3→exp4)**:把純 loc-week 歷史均值作為獨立成員放入 blend pool,權重
  搜尋將 100% 給該成員(28.3424 → 22.6488)——「理解資料」勝過模型複雜度的本場定調。
- **突破點 2(exp5→exp6)**:2020 COVID 年降權(W2020=0.29)帶來線性階段最大單輪增益
  (22.4897 → 21.6317);exp7 鄰週平滑再收 21.1487。
- **突破點 3(exp8→exp9)**:樹搜尋以新軸 year_weights 加 JOINT 聯合移動,拿到 Phase-B
  「一次一項」協定結構性難以觸及的複合最優(21.1487 → 21.0589)。

無法解析之紀錄:無(facts.unparsed 為空陣列)。

## 5. 效能對照:四層消融

> **語意澄清(tier1 為代理基線)**:本場為 2023 年已關閉、於 2026-02-14 舊 session 先行
> 完成的舊有競賽,不在週末 10 場通用批次範圍內,從未執行 `run_competition.py`,故**無
> 嚴格意義的 generic-batch 基線**。其最早兩筆實驗(exp1–2)使用單一時間切分、無 CatBoost
> 與 TE,與 exp3 起的 LOYO CV 不可比,**排除而非誤標**為 tier1。下表 tier1(28.3424,
> exp3)是同一 LOYO CV 方案下、TE 尚未作為獨立成員入 pool 前的三 GBDT blend,屬近似
> 對照(proxy);凡以 tier1 為錨點的比較皆應視為近似值。

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(代理:exp3,三 GBDT blend、TE 未入 pool、同 LOYO CV) | 28.3424 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp4,TE 成員拿 100% 權重) | 22.6488 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp7,TE 三段去噪終點) | 21.1487 | 見下方算式 |
| tier4 | + 樹搜尋(exp9,node #28 純結構 JOINT) | **21.0589** | 見下方算式 |

```
tier1→tier2: 28.3424 − 22.6488 = 5.6936,相對改善 5.6936 / 28.3424 = 20.0886%(tier1 為代理值,此列為近似口徑)
tier2→tier3: 22.6488 − 21.1487 = 1.5001,相對改善 1.5001 / 22.6488 = 6.6233%
tier3→tier4: 21.1487 − 21.0589 = 0.0898,相對改善 0.0898 / 21.1487 = 0.4246%
(累計 tier1→tier4:28.3424 − 21.0589 = 7.2835,25.6982%,同受代理口徑限制)
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

特殊場次注記(照 benchmark 慣例):其一,tier1 為代理基線(見上方澄清),`docs/
benchmark_facts.json` 該場 row 之 note 同此口徑。其二,Late Submission 已關閉,tier1–
tier4 四層分數**全部為 CV-only、從未提交**,無排行榜錨點。其三,tier4 採 node #28 的
21.0589 而**明確排除**其手足 BLEND 節點的 21.0332——後者僅在權重直接對同一份 OOF 擬合的
3 折 CV 下勝出,屬低信心的邊際發現(見 3.2 節誠實但書),保守取純結構節點是刻意的方法論
選擇。

## 6. 總結

本場資料的表象是 74 欄衛星感測迴歸,實質是一個結構問題:同一(位置, 週)的 emission 跨年
近乎恆定,而感測特徵與目標相關性極弱。全場單一最大槓桿是 exp4 把 loc-week 歷史均值作為
獨立成員入 pool——權重搜尋給它 100%、三個 GBDT 全歸零(28.3424 → 22.6488),此後 GBDT
只作對照組存在。

關鍵決策是承認這個結論並轉向:Phase-B 不再調模型,改對歷史均值本身做三段去噪(經驗貝葉斯
收縮 → COVID 年降權 → 鄰週平滑,22.6488 → 21.1487),每輪只改一項、全程同折;exp8 以
寬窗與殘差模型雙診斷確認感測特徵連殘差都解釋不了,依協定誠實停損。

樹搜尋(exp9)回答「結構完全主宰的地形上自動搜尋還有無價值」:重掃已手調軸只能打平,
增益來自新軸 year_weights 與 JOINT 聯合移動(21.1487 → 21.0589)——幅度小於模型主導型
競賽,但正是手動「一次一項」協定結構性拿不到的部分。四層消融中 tier2 增益最大、逐層遞減,
與「訊號早已被結構飽和」的診斷一致。

可信度方面須誠實:本場 Late Submission 已關閉,全部分數皆為 3 折 LOYO OOF、無排行榜
錨點;tier1 為代理基線,以其為錨的比較是近似口徑;tier4 刻意排除對同一份 OOF 擬合權重的
21.0332 blend 節點,採用的 21.0589 為純結構、可獨立重算驗證(根節點曾逐位吻合 exp7)的
穩健結果。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 0) 環境(uv 管理;所有 Python 執行皆透過 uv run)
uv sync

# 1) 資料(競賽已關閉,若 API 仍可下載歷史資料)
export KAGGLE_API_TOKEN=<your_token>
uv run kaggle competitions download -c playground-series-s3e20 \
    -p competitions/playground-series-s3e20/data
unzip -o competitions/playground-series-s3e20/data/playground-series-s3e20.zip \
    -d competitions/playground-series-s3e20/data

# 2) v2:LOYO CV + loc-week TE 入 pool + 三 GBDT blend → exp3、exp4
uv run python competitions/playground-series-s3e20/scripts/train_v2.py

# 3) Phase-B 線性迭代(一輪一改動,與 v2 同折可比)→ exp5–exp8
uv run python competitions/playground-series-s3e20/scripts/train_v3.py  # +EB 收縮
uv run python competitions/playground-series-s3e20/scripts/train_v4.py  # +COVID 年降權
uv run python competitions/playground-series-s3e20/scripts/train_v5.py  # +鄰週平滑(Phase-B 終點)
# 產出:competitions/playground-series-s3e20/submissions/sub_v5_blend_*.csv

# 4) Phase E-4 樹搜尋 v2(exp9,best;可中斷續跑,樹狀態存 experiments_tree.json)
uv run python3 tree_search/run_s3e20_v2.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e20
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e20/REPORT.md competitions/playground-series-s3e20/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e20/REPORT.md \
    competitions/playground-series-s3e20/s3e20_REPORT.pdf

# 提交至 Kaggle:本場 Late Submission 已關閉,無提交指令(所有分數皆 CV-only)
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`。
