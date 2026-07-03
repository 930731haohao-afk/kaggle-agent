# 競賽分析報告:playground-series-s3e5

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽要求依葡萄酒的物理化學量測值(固定酸度、揮發性酸度、檸檬酸、殘糖、氯化物、
游離/總二氧化硫、密度、pH、硫酸鹽、酒精濃度)預測其品質評分 `quality`,分數為 3 至 8 的
序數(ordinal)整數等級。

**Why**:評估指標為 **quadratic_weighted_kappa(QWK,maximize)**。品質等級雖以整數表示,但等級
之間存在順序關係(5 分與 6 分的差距遠小於 3 分與 8 分),QWK 會依「預測與真實等級距離的平方」
懲罰誤差,比單純的準確率或無序的多分類損失更能反映序數目標「越接近真實等級越好」的評分邏輯,
因此是此類序數評分資料的合理指標選擇。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e5 |
| 問題型別 | classification |
| 評估指標 | quadratic_weighted_kappa(maximize) |
| 目標欄位 | quality |

## 2. 資料規格

- 依 `competition.notes` 記載,本場為 Aygun et al. Nature 2026 論文所評測之 Kaggle Playground
  benchmark(Season 3 Episode 5)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。
- 實驗 1(`generic_batch` 通用批次管線)使用 11 個原始欄位(來源:`experiments[0].n_features`)。
- 實驗 2、3(`v2` 手刻管線)在原始欄位基礎上,經特徵工程展開為 21 個特徵(來源:
  `experiments[1].n_features`、`experiments[2].n_features`),欄位清單詳 `facts.json` 之
  `experiments[1].features` / `experiments[2].features`,包含游離/總二氧化硫比值、酸度比值、
  酒精濃度與硫酸鹽/密度的交互作用等衍生特徵。

## 3. 模型規格

本場有三筆實驗紀錄:

**實驗 1(`generic_batch`,通用批次管線,11 特徵)— base models:**

| Model | OOF QWK |
|-------|---------|
| LGB | 0.45223 |
| XGB | 0.46995 |
| CAT | 0.46768 |

Ensemble 權重(來源:`experiments[0].ensemble.weights`):LGB 0.1 / XGB 0.4 / CAT 0.5,
Ensemble 分數(來源:`experiments[0].ensemble.score`):**0.47871**。

**實驗 2(`v2`,21 特徵,同一集成權重但採「四捨五入」後處理)— base models:**

| Model | OOF QWK |
|-------|---------|
| LGB | 0.4522 |
| XGB | 0.46218 |
| CAT | 0.47094 |

Ensemble 權重(來源:`experiments[1].ensemble.weights`):LGB 0.2 / XGB 0.2 / CAT 0.6,
Ensemble 分數(來源:`experiments[1].ensemble.score`):**0.47191**。

**實驗 3(`v2`,21 特徵,採「最佳化分割閾值」後處理,為 `facts.best`)— base models:**

| Model | OOF QWK |
|-------|---------|
| LGB | 0.4522 |
| XGB | 0.46218 |
| CAT | 0.47094 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 0.2 / XGB 0.2 / CAT 0.6,
Ensemble 分數(來源:`experiments[2].ensemble.score`):**0.52687**。此分數為 `facts.best` 所指之
最佳 OOF 分數。

**選型理由**:三個梯度提升樹模型(LightGBM、XGBoost、CatBoost)在表格型資料上普遍穩健,各自的
歸納偏誤不同,故以加權集成降低單一模型的方差。實驗 1、2、3 皆採「迴歸」而非「多分類」建模
`quality`(迴歸輸出為連續值,經後處理轉換為離散等級),理由是 QWK 懲罰的是等級距離的平方,
迴歸模型的連續輸出天生保有等級間的順序關係,而多分類模型不具備此順序資訊。實驗 2 與實驗 3
使用完全相同的特徵集與集成權重,差異僅在於後處理步驟(見節 5),藉此獨立驗證後處理本身帶來的
增益。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold | 5 | 42 |
| 3 | 5fold | 5 | 42 |

(來源:`experiments[].cv`;實驗 1 之 seed 未記錄,故寫「無紀錄」。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項為
「無紀錄」。

**為何用此 CV**:`quality` 為小基數(3–8 共 6 級)且嚴重不平衡的序數目標(依 STATUS.md 記載,
最稀有等級與最常見等級之比例懸殊),若採用一般隨機 K-fold 切分,稀有等級可能在某些 fold 中
樣本過少甚至掛零,使該 fold 的 QWK 計算不穩定。三筆實驗皆改用「依 `quality` 標籤分層抽樣」的
StratifiedKFold,使各 fold 的等級分布與整體一致,讓交叉驗證分數更能反映模型的真實泛化能力。

## 5. 推論程序

依 `facts.best`(實驗 3)之 `postprocess` 欄位記錄兩個步驟:
`OptimizedRounder(cutpoints=[3.642, 4.594, 5.657, 6.196, 7.386])` 與 `clip[3,8]`(來源:
`experiments[2].postprocess`)。即先以最佳化搜尋得到的 5 個分割閾值,將迴歸模型輸出的連續值
轉換為離散等級,再限制於 3 至 8 的合法範圍內。

相較之下,實驗 2 之 `postprocess` 記錄為 `["round_to_nearest_int", "clip[3,8]"]`(來源:
`experiments[1].postprocess`),即單純四捨五入至最近整數再截斷範圍。實驗 2 與實驗 3 使用
完全相同的特徵集與集成權重,僅後處理方式不同,OOF 分數卻由 0.47191(實驗 2)提升至
0.52687(實驗 3)。

三筆實驗之 submission 檔名,僅實驗 1 有記錄:`sub_generic_0.47871_20260703_120341.csv`(來源:
`experiments[0].submission`)。實驗 2、3 之 `submission` 欄位在 `facts.json` 中無紀錄(v2 schema
之 submission 為選填欄位,本次執行時未填入),故本節對實驗 2、3 之提交檔名寫「無紀錄」,不作
額外推測。三者皆對應 `id_column = Id`、`target_column = quality`。

**本場所有實驗均未提交至 Kaggle 排行榜**(此執行環境未設定 Kaggle 憑證,`facts.json` 中
`leaderboard` 欄位為 null,且列於 `facts.missing`)。

## 6. 評估指標

**指標定義**:quadratic_weighted_kappa(QWK)= 觀察一致性與隨機期望一致性之差,並以「預測與
真實等級距離的平方」作為誤差權重,數值愈接近 1 表示模型排序與真實等級愈一致。

| 項目 | 分數 |
|------|------|
| 實驗 1 Ensemble(OOF QWK,generic_batch,11 特徵) | 0.47871 |
| 實驗 2 Ensemble(OOF QWK,四捨五入後處理) | 0.47191 |
| 實驗 3 Ensemble(OOF QWK,最佳化閾值後處理,facts.best) | 0.52687 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

(來源:`experiments[].ensemble.score`;`facts.leaderboard` 為 null,`facts.missing` 列出
`leaderboard`,故 Public/Private LB 皆寫「無紀錄」。)

**CV↔LB gap**:因本場所有實驗皆未提交排行榜,`facts.leaderboard` 不存在,故無法計算 CV↔LB
差值,亦不進行任何推測性比較。

實驗 3 相對於實驗 1(既有基準)之 OOF 分數差值,以內嵌算式呈現:

```
實驗 3 OOF QWK − 實驗 1 OOF QWK = 0.52687 − 0.47871 = 0.04816
```

實驗 3 相對於實驗 2(相同特徵/權重、僅後處理不同)之增益:

```
實驗 3 OOF QWK − 實驗 2 OOF QWK = 0.52687 − 0.47191 = 0.05496
```

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:03:41 | 0.47871 | generic_batch |
| 2 | 2026-07-03T18:48:21 | 0.47191 | v2 |
| 3 | 2026-07-03T18:48:21 | 0.52687 | v2 |

（來源：`facts.trajectory`）

**突破點**:第 3 筆實驗(`v2`,最佳化閾值後處理)的 OOF 分數(0.52687)為三筆中最高,分別較第 2
筆(0.47191)與第 1 筆(既有基準,0.47871)明顯提升(精確差值計算見節 6 之內嵌算式)。第 2、3
筆使用完全相同的特徵集(21 個特徵)與集成權重(LGB 0.2 / XGB 0.2 / CAT 0.6),差異僅在於第 3
筆改用針對 OOF QWK 直接最佳化的分割閾值(`experiments[2].postprocess`),而非第 2 筆單純的
四捨五入(`experiments[1].postprocess`)——顯示本場最大的分數躍升來自後處理步驟本身,而非
特徵集或集成權重的改變。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 實驗 1:通用批次管線(11 特徵,四捨五入後處理)
uv run python3 competitions/run_competition.py playground-series-s3e5

# 實驗 2、3:手刻管線(EDA → 特徵工程 → 訓練/CV/集成 → 兩種後處理比較 → 提交檔產出)
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py

# 提交至 Kaggle(本次執行環境未設定憑證,以下指令供後續有憑證時使用)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e5 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。
