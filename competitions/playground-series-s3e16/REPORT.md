# 競賽分析報告:playground-series-s3e16

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽要求依螃蟹的物理量測值(長度、直徑、高度、整體重量與各部位重量等)預測其
年齡(`Age`)。這是一個以連續數值輸出年齡估計的迴歸問題。

**Why**:評估指標為 **MAE(平均絕對誤差,minimize)**。年齡標籤為正整數且分布右偏(多數個體
年齡集中在較低區間、少數個體年齡偏高),MAE 以「絕對誤差的平均」衡量預測誤差,相較平方誤差
類指標對少數高齡離群樣本更穩健,不會讓少數極端樣本主導損失,因此是此類偏態計數型目標的合理
選擇。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e16 |
| 問題型別 | regression |
| 評估指標 | mae(minimize) |
| 目標欄位 | Age |

## 2. 資料規格

- 訓練集 74,051 列、測試集 49,368 列;共 8 個原始欄位(1 個類別型 `Sex`,其餘 7 個為連續型的
  尺寸/重量量測值)加上目標欄位 `Age`(來源:`competition.notes`)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。
- 實驗 1(`skill_train` 手刻管線)在原始 8 欄位基礎上,經特徵工程展開為 24 個特徵(來源:
  `experiments[0].n_features`、`experiments[0].features`),包含 `Sex` one-hot 展開、`is_infant`
  旗標、各部位重量佔比、`weight_resid`、體積/密度衍生量等(欄位清單詳 `facts.json` 之
  `experiments[0].features`)。
- 實驗 2(`generic_batch` 通用批次管線)僅使用原始 8 個欄位,未套用上述特徵工程(來源:
  `experiments[1].n_features`)。

## 3. 模型規格

本場有兩筆實驗紀錄,分別使用不同的特徵集與集成權重:

**實驗 1(`skill_train`,含特徵工程)— base models:**

| Model | OOF MAE | time_s |
|-------|---------|--------|
| LGB | 1.35651 | 30.8 |
| XGB | 1.35763 | 36.7 |
| CAT | 1.36162 | 23.7 |

Ensemble 權重(來源:`experiments[0].ensemble.weights`):LGB 0.6 / XGB 0.3 / CAT 0.1,
Ensemble 分數(來源:`experiments[0].ensemble.score`):**1.35589**。

**實驗 2(`generic_batch`,通用批次管線)— base models:**

| Model | OOF MAE |
|-------|---------|
| LGB | 1.3623 |
| XGB | 1.35886 |
| CAT | 1.35714 |

Ensemble 權重(來源:`experiments[1].ensemble.weights`):LGB 0.3 / XGB 0.2 / CAT 0.5,
Ensemble 分數(來源:`experiments[1].ensemble.score`):**1.35441**。此分數為 `facts.best` 所指之
最佳 OOF 分數(1.35441 低於實驗 1 的 1.35589)。

**選型理由**:三個梯度提升樹模型(LightGBM、XGBoost、CatBoost)在表格型資料上普遍穩健,各自的
歸納偏誤不同,故以加權集成降低單一模型的方差。實驗 1 額外投入特徵工程(24 個特徵)並針對 MAE
目標調整訓練細節;實驗 2 為通用批次管線的預設跑法,未套用上述特徵工程但集成權重不同,結果 OOF
分數反而略低——惟此結果僅為內部 OOF 比較,實驗 2 並未提交至 Kaggle 排行榜驗證(見節 5、6)。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold_stratified_agebin | 5 | 無紀錄 |
| 2 | 5fold | 5 | 無紀錄 |

(來源:`experiments[].cv`;`facts.missing` 列出 `cv_seed`,故兩實驗之 seed 皆寫「無紀錄」。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項亦為
「無紀錄」。

**為何用此 CV**:年齡標籤為右偏的整數計數型目標,若採用一般隨機 K-fold 切分,高齡樣本較少,可能
造成各 fold 之目標分布不均、驗證分數不穩定。實驗 1 因此改用「依年齡分箱後分層抽樣」的
StratifiedKFold(將高齡樣本合併為單一分箱以避免箱內樣本過少),使各 fold 的年齡分布更一致,
讓交叉驗證分數更能反映模型的真實泛化能力。

## 5. 推論程序

依 `facts.best`(實驗 2,通用批次管線,OOF 分數最低者)之 `postprocess` 欄位未記錄任何後處理
步驟,故此欄寫「無後處理紀錄」。其 submission 檔名為 `sub_generic_1.35441_20260703_115739.csv`
(來源:`experiments[1].submission`),對應 `id_column = id`、`target_column = Age`。

需特別說明:此筆(實驗 2)**並未提交至 Kaggle 排行榜**(`facts.json` 中該實驗無 `leaderboard`
欄位)。實際提交排行榜者為實驗 1,其 `postprocess` 記錄為 `["round"]`(即對預測值四捨五入,
因目標 `Age` 為整數),submission 檔名為 `sub_blend_20260703_091840.csv`(來源:
`experiments[0].submission`),同樣對應 `id_column = id`、`target_column = Age`。此筆之排行榜
結果見節 6。

## 6. 評估指標

**指標定義**:MAE(Mean Absolute Error)= 預測值與真實值絕對差的平均,單位與目標欄位相同
(此處為年齡)。

| 項目 | 分數 |
|------|------|
| 實驗 1 Ensemble(OOF MAE) | 1.35589 |
| 實驗 2 Ensemble(OOF MAE,facts.best) | 1.35441 |
| Public LB(實驗 1 之提交) | 1.34356 |
| Private LB(實驗 1 之提交) | 1.34075 |

(來源:`experiments[].ensemble.score`、`facts.leaderboard`。`facts.leaderboard` 對應的是實驗 1
的提交結果,而非 OOF 分數最低的實驗 2——因為只有實驗 1 實際提交過排行榜。)

**CV↔LB gap**(僅實驗 1 有排行榜可比較,差值以內嵌算式呈現):

```
實驗 1 OOF MAE − Public LB  = 1.35589 − 1.34356 = 0.01233
實驗 1 OOF MAE − Private LB = 1.35589 − 1.34075 = 0.01514
```

OOF 分數略高於(即劣於)Public/Private LB,顯示交叉驗證分數並未過度樂觀高估模型表現,LB 分數
反而比 OOF 更好,CV 具參考價值,可安心依 OOF 排序後續實驗。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T09:18:40 | 1.35589 | skill_train |
| 2 | 2026-07-03T11:57:39 | 1.35441 | generic_batch |

（來源：`facts.trajectory`）

**突破點**:第 2 筆實驗(`generic_batch`,通用批次管線)的 OOF 分數(1.35441)低於第 1 筆
(`skill_train`,含特徵工程,1.35589)。第 2 筆僅使用原始 8 個欄位、未套用第 1 筆的 24 個工程
特徵,但集成權重不同(LGB 0.3 / XGB 0.2 / CAT 0.5,相較第 1 筆之 LGB 0.6 / XGB 0.3 / CAT 0.1
更偏重 CatBoost),使內部 OOF 分數略優於第 1 筆。惟此結果僅為 OOF 內部比較,第 2 筆並未提交
Kaggle 排行榜驗證,無法確認其是否真的更能泛化(見節 5)。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 實驗 1:手刻管線(EDA → 特徵工程 → 訓練/CV/集成)
uv run python3 competitions/playground-series-s3e16/scripts/eda.py
uv run python3 competitions/playground-series-s3e16/scripts/train.py

# 實驗 2:通用批次管線
uv run python3 competitions/run_competition.py playground-series-s3e16

# 提交至 Kaggle(需先設定有效的 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e16 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。
