# 競賽分析報告:playground-series-s3e20

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-04

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

## 2. 資料規格

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

## 3. 模型規格

最佳實驗(exp #7,`train_v5.py`)為 4 成員 blend:三個 GBDT + 一個「去噪後的
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

## 4. 訓練規格

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

## 5. 推論程序

- 後處理:`clip >= 0`(排放量不可為負;來源:facts.json best.postprocess)。
- 最終 test 預測:TE 以「全部三個訓練年度」重算歷史均值(套用同一組 w2020 / wnb 參數)
  後直接輸出(blend 權重 TE=1.0)。
- Submission:`sub_v5_blend_21.1487_20260704_011931.csv`,格式兩欄:
  `ID_LAT_LON_YEAR_WEEK, emission`(列數同 test 集)。
- 註:競賽已關閉,此檔為 artifact,無法實際上傳。

## 6. 評估指標

RMSE = 預測誤差平方均值的平方根,對大誤差施以平方級懲罰。

分數總表(最佳實驗 exp #7,LOYO OOF):

| 項目 | OOF RMSE |
|------|----------|
| LGB | 32.074 |
| XGB | 28.0674 |
| CAT | 28.8466 |
| TE (denoised loc-week mean) | 21.1487 |
| **Ensemble(TE=1.0)** | **21.1487** |
| Public/Private LB | 無紀錄(Late Submission 已關閉) |

CV↔LB gap:無紀錄(無 LB 分數可比)。

## 7. 實驗軌跡

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

**突破點**:
- **exp #3 → #4**(28.3424 → 22.6488):把「純 loc-week 歷史均值」加入 blend pool,
  權重搜尋將全部權重給予該成員 —— 確立本場的訊號本質是結構而非感測特徵。
- **exp #6**(22.4897 → 21.6317):單輪最大增益,來自 2020 COVID 年降權(W2020=0.29)。
- **exp #7**(21.6317 → 21.1487):鄰週(week±1)平滑,最終最佳。
- **exp #8**(分數不變,診斷輪):(a) 平滑窗加寬至 ±2/±3 週全部變差;(b) LGB 殘差模型
  (訓練於 y − TE)以任何比例加回皆變差 —— 證明感測特徵連殘差都無法解釋,結構已飽和
  全部可預測訊號。此為刻意保留的負向結果紀錄。

無法解析之紀錄:無。

## 8. 重現指令

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
uv run python competitions/playground-series-s3e20/scripts/train_v5.py  # +鄰週平滑 (WNB=0.28) → 最終
# 產出:competitions/playground-series-s3e20/submissions/sub_v5_blend_*.csv
# (競賽 Late Submission 已關閉,無提交指令)
```
