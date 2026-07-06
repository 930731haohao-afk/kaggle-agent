# 競賽分析報告:playground-series-s3e20

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-04(同日再更新:Phase G-1b 樹搜尋成果入帳,實驗 9)

## 1. 競賽目的

**What**:本競賽(Predict CO2 emissions in Rwanda using satellite data)要求以盧安達境內
497 個地理位置在 2019–2021 年的衛星感測資料(雲層、氣膠、UV 等 75 個欄位)與時空座標
(latitude / longitude / year / week_no),預測 2022 年各位置每週的二氧化碳排放量
(`emission`)。本質上是一個「跨年外推」的時空面板迴歸問題。

**Why**:區域級 CO2 排放估計是碳監測與環境政策的基礎,而衛星遙測是缺乏地面監測站地區
(如盧安達)唯一可規模化的資料來源。評估指標為 RMSE(minimize):目標高度右偏
(median 45.6、mean 81.9、max 3167.8),RMSE 的平方懲罰使大排放源的預測誤差主導總分,
與「抓準高排放熱點」的實務需求一致。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e20 |
| URL | https://www.kaggle.com/competitions/playground-series-s3e20 |
| 問題型別 | regression |
| 評估指標 | rmse(minimize) |
| 目標欄位 | emission |
| ID 欄位 | ID_LAT_LON_YEAR_WEEK |

> ⚠️ 本競賽 Late Submission 已關閉,無法上傳提交;全程僅以本地 CV(OOF)評估,
> leaderboard 無紀錄。

## 2. 流程(how):五大元件

### 2.1 資料規格

- 檔案:`train.csv` / `test.csv` / `sample_submission.csv`。
- 時序結構:train 為 2019–2021 三個年度、test 為 2022 年;`week_no` 0–52。
- 空間結構:latitude / longitude(盧安達境內,497 個唯一位置)。
- 特徵:75 個欄位,多數來自衛星感測(Cloud、Aerosol、UV 量測群組)。
- 缺失:部分特徵群組缺失極重(UvAerosolLayerHeight 群組達 99.4%);v2 起將 >90% 缺失的
  7 個感測欄位刪除,保留 63 個感測欄位(來源:facts.json best.features:
  「70 base (63 sensors, dropped 7)」)。
- 目標分布:重度右偏(median 45.6、mean 81.9、max 3167.8),訓練時取 log1p。
- 特別規則:外部資料不允許、預訓練模型不允許、internet 不允許、每日提交上限 5。
- 關鍵資料性質(EDA 驗證,見 STATUS.md):同一 `(latitude, longitude, week_no)` 的
  emission 跨年近乎恆定 —— 這是全場最重要的結構訊號。

### 2.2 模型規格

**facts.json 目前的 best 是實驗 9——一筆樹搜尋(tree-search)結果**(細節見 2.2a 小節),
而非本節原本描述的 Phase-B 手調終點(exp #7)。兩者皆完整說明如下。

exp #7(`train_v5.py`,Phase-B 手調終點)為 4 成員 blend:三個 GBDT + 一個「去噪後的
location-week 歷史均值」(TE)結構預測器。

| 成員 | OOF RMSE |
|------|----------|
| LGB | 32.074 |
| XGB | 28.0674 |
| CAT | 28.8466 |
| TE (denoised loc-week mean) | 21.1487 |

Ensemble(OOF simplex 粗網格權重搜尋):

| 成員 | 權重 |
|------|------|
| LGB | 0.0 |
| XGB | 0.0 |
| CAT | 0.0 |
| TE | 1.0 |

**Blend OOF RMSE = 21.1487**(等於 TE 單體 —— 權重搜尋把三個 GBDT 全數歸零)。

**選型理由**:v2(exp #4)已發現純 loc-week 歷史均值(22.6488)完勝任何 GBDT
(最佳 GBDT blend 28.3424,exp #3),且權重搜尋將全部權重給予 TE。Phase-B(exp #5–#7)
因此把改進重心放在 TE 本身的三項去噪:
1. **經驗貝葉斯收縮**(exp #5):每折 cell 均值僅 n=2 個觀測,向 ~106 個觀測的
   location 均值收縮(α=0.935)。
2. **COVID 年降權**(exp #6):2020 年排放受疫情壓低,計算歷史均值時將該年權重降至
   W2020=0.29(完全排除比降權更差)。
3. **鄰週平滑**(exp #7):cell 均值與同位置 week±1 的 cell 均值加權平均(WNB=0.28);
   此步驟吸收了第 1 項的收縮(α 重調後收斂至 1.0)。

GBDT 保留在 pool 中僅作對照,每輪權重搜尋均自動淘汰(權重 0)。

#### 2.2a 樹搜尋最佳(best,實驗 9)——本次更新(Phase G-1b)新增

Phase E-4(harness v2,2026-07-04)樹搜尋把 exp #7 手調的 4 個結構純量超參
(alpha/w2020/wnb/window)當成節點空間,測試「結構完全主導、評估成本近乎零」的地形上
樹搜尋是否還有價值。根節點以 `experiments_tree.json` 重新獨立驗算 exp #7 的設定,
逐位吻合 21.1487(見 STATUS.md)。JOINT lineage 把 4 個軸一步同時移動,再疊加一個
Phase-B 從未試過的**新軸 year_weights**(依年度訊噪比差異降權/升權),找到全樹最佳的
**純結構**節點(`experiments_tree.json` node #28,best.base_models,38 個評估節點:
37 solo/1 blend,0 去重拒絕,15 次 backtrack,整場搜尋 wall 僅 33.3s;node #28 本身為
單一 solo 設定,node wall_s=0.07s):

| 參數 | 值 |
|------|-----|
| method | te(location-week 歷史均值 target encoding) |
| alpha(EB 收縮) | 0.99 |
| w2020(COVID 降權) | 0.2 |
| wnb(鄰週平滑權重) | 0.3 |
| window | 1 |
| year_weights[2019] | 0.6 |
| year_weights[2020] | 0.2 |
| year_weights[2021] | 1.2 |

**分數(best.score)**:**21.0589**,較 exp #7(21.1487)改善 -0.0898(約 -0.42% 相對改善,
詳見第 2.5 節程式區塊)。

> **重要:不採用 21.0332 的 blend 版本**(best.notes 明確說明)。node #28 有一個手足
> BLEND 節點(#29,本報告不採用其分數):把此結構節點與一個診斷用 GBDT 混合,GBDT 僅拿到
> 2.17% 權重,OOF RMSE 降至 21.0332,但這是在僅 3 折 Leave-One-Year-Out CV 下、且權重
> 直接對同一份 OOF 擬合所得——STATUS.md 明確記為「低信心...視為 CV 噪音範圍內的邊際發現,
> 非穩健結論」。本報告採用的 21.0589(node #28)才是穩健的純結構結果:JOINT lineage
> 把 4 個已手調軸一步到位共同移動(加上新的 year_weights 軸),達到 Phase-B「一次一項」
> 手動協定結構性難以觸及的複合最優;而個別重掃描已手調軸(w2020、wnb)本身只誠實地
> 「打平」Phase-B 既有最優解,並非額外增益。完整節點/lineage 表與地形對比討論見
> `competitions/playground-series-s3e20/STATUS.md`〈Appendix: Phase E-4 樹搜尋 v2〉。

> **重要澄清**:best.notes 明確記載這是 **OOF-only 搜尋結果——未產生任何 test 預測**
> (facts.json 本筆無 submission 欄位)。本場 Late Submission 已關閉,exp #7 與 exp #9
> 皆未、也無法提交至 Kaggle。

### 2.3 訓練規格

| 項目 | 值 |
|------|-----|
| CV 方案 | Leave-One-Year-Out (2019/2020/2021) |
| 折數 | 3 |
| seed | 42 |

- **為何用此 CV**:test 為未來年度(2022),LOYO 每折保留一整年作驗證,忠實模擬
  「以其他年度歷史預測整年」的真實情境;目標編碼(TE)每折僅用訓練年度計算,無洩漏。
  v3–v5 與 v2 完全同折,分數全程可比。
- GBDT objective:RMSE on log1p(emission),預測後 expm1 還原。
- GBDT 詳細超參:facts.json 無紀錄(與 v2 一致未改動;實際數值見
  `scripts/train_v5.py` 的 `run_models`)。
- TE 超參(exp #7):alpha=1.0、w2020=0.24、wnb=0.28(在同一 LOYO CV 上以網格 pre-sweep 選定)。
- 樹搜尋 best(實驗 9)使用同一 LOYO(3 折、seed 42)CV 方案,超參見 2.2a 節表格。

### 2.4 推論程序

**facts.json 現在的 best 是實驗 9(樹搜尋)**,其 `postprocess`/`submission` 欄位皆未記錄
(OOF-only 結果,未產生 test 預測)。以下描述 exp #7(Phase-B 手調終點)的推論流程:

- 後處理:`clip >= 0`(排放量不可為負;來源:facts.json `experiments[6].postprocess`)。
- 最終 test 預測:TE 以「全部三個訓練年度」重算歷史均值(套用同一組 w2020 / wnb 參數)
  後直接輸出(blend 權重 TE=1.0)。
- Submission:`sub_v5_blend_21.1487_20260704_011931.csv`,格式兩欄:
  `ID_LAT_LON_YEAR_WEEK, emission`(列數同 test 集)。
- 註:競賽已關閉,此檔為 artifact,無法實際上傳;實驗 9(現行 best)未產生任何 submission
  檔——樹搜尋為 OOF-only 搜尋,不可與此檔案混淆。

### 2.5 評估指標

RMSE = 預測誤差平方均值的平方根,對大誤差施以平方級懲罰。

分數總表(exp #7,Phase-B 手調終點,LOYO OOF):

| 項目 | OOF RMSE |
|------|----------|
| LGB | 32.074 |
| XGB | 28.0674 |
| CAT | 28.8466 |
| TE (denoised loc-week mean) | 21.1487 |
| Ensemble(TE=1.0) | 21.1487 |
| 實驗 9 樹搜尋 v2(node #28,純結構,**best**) | **21.0589** |
| Public/Private LB | 無紀錄(Late Submission 已關閉) |

CV↔LB gap:無紀錄(無 LB 分數可比)。

樹搜尋(exp9,best)相對 exp #7 手調終點的改善:

```
21.1487 - 21.0589 = 0.0898
0.0898 / 21.1487 × 100 ≈ 0.4247...%(約 -0.42% 相對改善)
```

## 3. 實驗軌跡

| exp # | timestamp | OOF RMSE | source_format |
|-------|-----------|----------|---------------|
| 1 | 20260214_161647 | 38.507172 | v2 |
| 2 | 20260214_161743 | 33.205568 | v2 |
| 3 | 2026-07-03T09:59:53 | 28.3424 | v2 |
| 4 | 2026-07-03T10:12:09 | 22.6488 | v2 |
| 5 | 2026-07-04T00:58:26 | 22.4897 | v2 |
| 6 | 2026-07-04T01:16:24 | 21.6317 | v2 |
| 7 | 2026-07-04T01:19:31 | **21.1487** | v2 |
| 8 | 2026-07-04T01:35:00 | 21.1487 | v2 |
| 9 | 2026-07-04T12:18:33 | **21.0589** | v2 |

（來源:`facts.trajectory`。最佳分數現為第 9 筆(樹搜尋),即 `facts.best`。）

**突破點**:
- **exp #3 → #4**(28.3424 → 22.6488):把「純 loc-week 歷史均值」加入 blend pool,
  權重搜尋將全部權重給予該成員 —— 確立本場的訊號本質是結構而非感測特徵。
- **exp #6**(22.4897 → 21.6317):單輪最大增益,來自 2020 COVID 年降權(W2020=0.29)。
- **exp #7**(21.6317 → 21.1487):鄰週(week±1)平滑,最終最佳。
- **exp #8**(分數不變,診斷輪):(a) 平滑窗加寬至 ±2/±3 週全部變差;(b) LGB 殘差模型
  (訓練於 y − TE)以任何比例加回皆變差 —— 證明感測特徵連殘差都無法解釋,結構已飽和
  全部可預測訊號。此為刻意保留的負向結果紀錄。
- **exp #9(Phase G-1b,本次更新新增,樹搜尋 best)**:第 9 筆不是線性迭代的延續回合,
  而是 Phase E-4(2026-07-04)以 harness v2 執行的**樹搜尋(tree-search)**結果——來源
  `experiments_tree.json` 的 node #28。樹搜尋把 exp #7 已手調的 4 個結構軸
  (alpha/w2020/wnb/window)當成節點空間,重掃描已調軸(w2020、wnb)僅誠實打平既有最優,
  但(a)一個全新軸 year_weights(異常年降權概念推廣到 2019/2021)獨力貢獻增益、(b)
  JOINT lineage 把 4+1 個軸一步到位共同移動,複合出 -0.0898 的總增益(21.1487→21.0589,
  詳見第 2.2a 節與第 2.5 節程式區塊)。**誠實 CV-only 警語**:此結果為 OOF-only 搜尋產物,
  facts.json 本筆亦無 submission 欄位,**未提交至 Kaggle**(本場 Late Submission 亦已
  關閉,實務上無法提交)。**明確不採用**該節點手足 BLEND 節點的 21.0332(2.17% GBDT
  權重、僅 3 折 CV 直接對同一份 OOF 擬合,STATUS.md 記為低信心、CV 噪音範圍內的邊際發現,
  非穩健結論)——見第 2.2a 節澄清。完整節點鏈、backtrack 記錄、與誠實地形對比討論見
  `competitions/playground-series-s3e20/STATUS.md`〈Appendix: Phase E-4 樹搜尋 v2〉。

無法解析之紀錄:無。

## 4. 重現指令

```bash
# 執行目錄:專案根目錄 /home/tjyen/ai_agents/kaggle(依 CLAUDE.md 使用 uv 管理環境)
# 0) 環境
uv sync

# 1) 資料(競賽已關閉,若 API 仍可下載歷史資料)
export KAGGLE_API_TOKEN=<your_token>
uv run kaggle competitions download -c playground-series-s3e20 \
    -p competitions/playground-series-s3e20/data
unzip -o competitions/playground-series-s3e20/data/playground-series-s3e20.zip \
    -d competitions/playground-series-s3e20/data

# 2) v2 baseline(LOYO CV + loc-week TE + 3 GBDT blend;log 至 experiments.json)
uv run python competitions/playground-series-s3e20/scripts/train_v2.py

# 3) Phase-B 迭代(一輪一改動,分數全程與 v2 同折可比)
uv run python competitions/playground-series-s3e20/scripts/train_v3.py  # +EB 收縮 (alpha=0.935)
uv run python competitions/playground-series-s3e20/scripts/train_v4.py  # +COVID 降權 (W2020=0.29)
uv run python competitions/playground-series-s3e20/scripts/train_v5.py  # +鄰週平滑 (WNB=0.28) → Phase-B 終點
# 產出:competitions/playground-series-s3e20/submissions/sub_v5_blend_*.csv
# (競賽 Late Submission 已關閉,無提交指令)

# 4) Phase E-4 樹搜尋 v2(best,實驗 9,分數 21.0589)— resumable;軌跡存於 experiments_tree.json
uv run python3 tree_search/run_s3e20_v2.py
```
