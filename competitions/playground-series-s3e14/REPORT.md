# 競賽分析報告:playground-series-s3e14

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽要求依野生藍莓田區的環境與授粉相關量測值,預測藍莓產量(`yield`)。這是一個
以連續數值輸出產量估計的迴歸問題。

**Why**:評估指標為 **MAE(平均絕對誤差,minimize)**。以絕對誤差的平均衡量預測誤差,單位與
產量本身一致,直觀反映「平均預測誤差是多少產量單位」,不會像平方誤差類指標般被少數極端樣本
過度放大,適合作為此類連續產量預測任務的評估基準。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e14 |
| 問題型別 | regression |
| 評估指標 | mae(minimize) |
| 目標欄位 | yield |

## 2. 資料規格

本場已完整執行 EDA(`scripts/eda.py`)與特徵工程(`scripts/features.py`),`facts.json` 的
`material_level` 為 `full`,`status_md_present` 為 `true`(競賽層 `STATUS.md` 已建立,記錄 EDA
narrative 發現)。

- 依 `facts.json`,本場共有 3 筆實驗紀錄:第 1 筆(`generic_batch`,通用批次基線)使用
  16 個特徵(來源:`experiments[0].n_features`);第 2 筆(特徵工程 iteration 1)使用 27 個特徵
  (來源:`experiments[1].n_features`);第 3 筆即目前最佳實驗(特徵剪枝 iteration 2)使用
  21 個特徵(來源:`best.n_features` / `experiments[2].n_features`)。
- 原始資料列數/欄位型別、缺漏值計數等具體數值未被 `collect.py` 收錄進 `facts.json`(該檔僅記錄
  competition 中繼資料與 experiments 結構化欄位,不解析 EDA 腳本輸出),故此類細項為
  **無紀錄**。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。

## 3. 模型規格

最佳實驗(`best`,即 `experiments[2]`)的三個 base model 分數:

| Model | OOF MAE |
|-------|---------|
| LGB | 342.02154 |
| XGB | 342.21778 |
| CAT | 343.60196 |

Ensemble 權重(來源:`best.ensemble.weights`,方法 `best.ensemble.method = grid_simplex`):
LGB 0.45 / XGB 0.25 / CAT 0.3,Ensemble 分數(來源:`best.ensemble.score`):**340.75961**。

**選型理由**(來源:`best.notes`,並參照 `STATUS.md` 脈絡):此為第 2 輪自我改進迭代
(iteration 2)——相較於第 1 輪(`experiments[1]`,27 特徵、ensemble 分數 341.02061),本輪
剔除 6 個 LightGBM 特徵重要性趨近於零的雜訊特徵、讓 CatBoost 對低基數環境欄位改用原生類別特徵
處理,並額外嘗試以 Ridge 堆疊(stacking)取代網格搜尋的單純加權混合,兩者擇優——最終仍是網格
搜尋加權混合(`grid_simplex`)勝出。

作為對照,第 1 筆實驗(通用批次基線)的 base model 分數為:

| Model | OOF MAE |
|-------|---------|
| LGB | 345.32118 |
| XGB | 344.13027 |
| CAT | 344.24276 |

（來源:`experiments[0].base_models`,ensemble 權重 LGB 0.3 / XGB 0.3 / CAT 0.4,分數
341.40782,來源:`experiments[0].ensemble`。）

## 4. 訓練規格

| 實驗 | source_format | CV scheme | n_splits | seed |
|------|----------------|-----------|----------|------|
| 1(baseline) | generic_batch | 5fold | 5 | 無紀錄 |
| 2(iteration 1) | v2 | 5fold_kfold | 5 | 42 |
| 3(iteration 2,best) | v2 | 5fold_kfold | 5 | 42 |

（來源:`experiments[].cv`;第 1 筆之 `cv` 物件無 `seed` 欄位,故寫「無紀錄」。）

各 base model 之 objective/超參數(如 learning_rate、num_leaves、depth 等)未記錄於
`experiments[].base_models[]` 對應欄位中,故此項為「無紀錄」。

**為何用此 CV**:第 2、3 筆實驗改採 5-fold KFold(shuffle,seed=42)。依 `STATUS.md` 記載之
EDA 脈絡,目標為連續值且無分組/時間結構,train/test 各欄位分布差異小,故不需分層抽樣(與
s3e16 的離散 Age 目標不同),一般 i.i.d. KFold 即為合適的驗證方案。

## 5. 推論程序

依 `best.postprocess`(來源:`experiments[2].postprocess`),本場最佳實驗記錄了兩項後處理/
選型決策原文:

1. `"snap_to_grid=used (raw=340.75961, snapped=340.71180)"`
2. `"grid_simplex=340.75961 vs ridge_stack=344.04375 -> chose grid_simplex"`

即:預測值最終被「snap 至訓練集中曾觀測到的最近 yield 值」,以及混合方法在網格搜尋加權
(grid_simplex)與 Ridge 堆疊(ridge_stack)間擇優,選定前者。

Submission 檔名為 `sub_blend_v2_340.71180_20260703_193959.csv`(來源:`best.submission`),
對應 `id_column = id`、`target_column = yield`(來源:`competition.id_column` /
`competition.target_column`)。

## 6. 評估指標

**指標定義**:MAE(Mean Absolute Error)= 預測值與真實值絕對差的平均,單位與目標欄位相同
(此處為產量)。

| 實驗 | Model/Ensemble | OOF MAE |
|------|----------------|---------|
| 1(baseline) | LGB | 345.32118 |
| 1(baseline) | XGB | 344.13027 |
| 1(baseline) | CAT | 344.24276 |
| 1(baseline) | Ensemble | 341.40782 |
| 3(best) | LGB | 342.02154 |
| 3(best) | XGB | 342.21778 |
| 3(best) | CAT | 343.60196 |
| 3(best) | Ensemble(後處理前) | 340.75961 |
| 3(best) | Ensemble(snap-to-grid 後,= `best.score`) | 340.711795 |
| Public LB | — | 無紀錄 |
| Private LB | — | 無紀錄 |

（來源:`experiments[0].base_models`/`.ensemble`、`best.base_models`/`.ensemble`/`.score`;
`facts.leaderboard` 為 `null`,`facts.missing` 列出 `leaderboard`,顯示本場未提交至 Kaggle
排行榜,故無法計算 CV↔LB gap。）

本場最佳實驗相對第 1 筆基線的分數差距:

```
341.40782 (experiments[0].score, baseline)
- 340.711795 (best.score, iteration 2)
= 0.696025
```

即最佳實驗較基線有小幅但一致的下降(對絕對誤差指標而言為改善方向)。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:11:46 | 341.40782 | generic_batch |
| 2 | 2026-07-03T19:36:46 | 340.958565 | v2 |
| 3 | 2026-07-03T19:39:59 | 340.711795 | v2 |

（來源:`facts.trajectory`）

**突破點敘述**(來源:`experiments[].notes`):分數呈現逐步下降。第 2 筆(iteration 1)引入
27 個特徵(果實生物性交互項、聚合授粉指標等),並在後處理比較中選用 snap-to-grid,分數自
341.40782 降至 340.958565。第 3 筆(iteration 2)為對第 2 筆的 reflexion:依 LightGBM 特徵重要性
剔除 6 個趨近零貢獻的特徵、CatBoost 改用原生類別特徵處理低基數環境欄位,並比較 Ridge 堆疊
(344.04375,劣於網格混合)後仍採網格加權混合,分數再降至 340.711795,為目前最佳。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e14/scripts/eda.py

# Stage 3: 建模(iteration 1,分數 340.958565,對應 experiments[1]/submission
#   sub_blend_340.95856_20260703_193646.csv)
uv run python3 competitions/playground-series-s3e14/scripts/train.py

# Stage 3: 建模(iteration 2 / 自我改進,分數 340.711795,即 best,對應
#   sub_blend_v2_340.71180_20260703_193959.csv)
uv run python3 competitions/playground-series-s3e14/scripts/train_v2.py
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有指令均透過 `uv run` 以確保套件環境
一致。`scripts/features.py` 由 `train.py`/`train_v2.py` 以模組方式匯入,不需單獨執行。
本場未提交至 Kaggle 排行榜(此環境無 Kaggle 憑證),故無對應的 `kaggle competitions submit`
指令可重現。
