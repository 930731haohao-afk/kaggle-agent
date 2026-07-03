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

- 依 `facts.json`,本場共有 7 筆實驗紀錄:第 1 筆(`generic_batch`,通用批次基線)使用
  16 個特徵(來源:`experiments[0].n_features`);第 2 筆(特徵工程 iteration 1)使用 27 個特徵
  (來源:`experiments[1].n_features`);第 3 筆(特徵剪枝 iteration 2)起至第 7 筆(目前最佳)
  均使用 21 個特徵(來源:`best.n_features` / `experiments[2].n_features` / `experiments[6].n_features`)。
  第 4–7 筆為 Phase B 自我改進迭代(Optuna 調參、blend 成員擴充、isotonic 校準試驗、seed bagging)。
- 原始資料列數/欄位型別、缺漏值計數等具體數值未被 `collect.py` 收錄進 `facts.json`(該檔僅記錄
  competition 中繼資料與 experiments 結構化欄位,不解析 EDA 腳本輸出),故此類細項為
  **無紀錄**。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。

## 3. 模型規格

最佳實驗(`best`,即 `experiments[6]`)的五個 base model 分數:

| Model | OOF MAE |
|-------|---------|
| LGB | 342.02154 |
| LGB_TUNED | 341.68775 |
| XGB | 342.21778 |
| CAT | 343.60196 |
| LGB_SEED2024 | 342.04208 |

Ensemble 權重(來源:`best.ensemble.weights`,方法 `best.ensemble.method = grid_simplex_5way`):
LGB 0.2 / LGB_TUNED 0.15 / XGB 0.2 / CAT 0.25 / LGB_SEED2024 0.2,Ensemble 分數(來源:
`best.ensemble.score`):**340.65207**。

**選型理由**(來源:`best.notes`、`experiments[3..6].notes`,並參照 `STATUS.md` 脈絡):Phase B
自我改進共 4 輪,每輪只改一件事。round 1(`experiments[3]`)以 Optuna fold-0 代理調參 LightGBM,
單模自 342.02154 進步到 341.68775,但直接「替換」原 LGB 使 blend 反退步(340.75961→340.95316,
多樣性流失);round 2(`experiments[4]`)改為「保留原 LGB 並加入調參版 LGB」作為第 4 個成員,
4-way 網格混合勝出(340.62702);round 3(`experiments[5]`)測試對混合輸出做 nested OOF isotonic
校準,isotonic 本身分數 346.99717 明顯更差,被自動否決;round 4(`experiments[6]`,best)再加入
一個 seed 多樣化的 LGB(相同超參、random_state 2024)作為第 5 個成員,取得最佳 340.59891。

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
| 3(iteration 2) | v2 | 5fold_kfold | 5 | 42 |
| 4(Phase B round 1) | v2 | 5fold_kfold | 5 | 42 |
| 5(Phase B round 2) | v2 | 5fold_kfold | 5 | 42 |
| 6(Phase B round 3) | v2 | 5fold_kfold | 5 | 42 |
| 7(Phase B round 4,best) | v2 | 5fold_kfold | 5 | 42 |

（來源:`experiments[].cv`;第 1 筆之 `cv` 物件無 `seed` 欄位,故寫「無紀錄」;第 2–7 筆 seed 均為 42。）

多數 base model 之 objective/超參數未記錄於 `experiments[].base_models[]` 對應欄位中,惟第 4 筆
(`experiments[3].notes`)完整記錄了 Optuna 為 LightGBM 找到的最佳超參:`learning_rate`
0.015389816656608301、`num_leaves` 63、`max_depth` 5、`min_child_samples` 30、`subsample`
0.7899031923134097、`colsample_bytree` 0.6842166782718478、`reg_alpha` 0.012648426946439915、
`reg_lambda` 0.00346212601571565(36 trials、293 秒,採 fold-0 單折代理目標)。此組超參即後續
第 5、7 筆中的 `LGB_TUNED` 成員。

**為何用此 CV**:第 2、3 筆實驗改採 5-fold KFold(shuffle,seed=42)。依 `STATUS.md` 記載之
EDA 脈絡,目標為連續值且無分組/時間結構,train/test 各欄位分布差異小,故不需分層抽樣(與
s3e16 的離散 Age 目標不同),一般 i.i.d. KFold 即為合適的驗證方案。

## 5. 推論程序

依 `best.postprocess`(來源:`experiments[6].postprocess`),本場最佳實驗記錄了一項後處理原文:

1. `"snap_to_grid=used (raw=340.65207, snapped=340.59891)"`

即:5-way 網格混合輸出最終被「snap 至訓練集中曾觀測到的最近 yield 值」,分數自 340.65207
降至 340.59891。snap-to-grid 在本場歷次迭代皆為正向後處理。

作為對照,round 3(`experiments[5].postprocess`)另記錄了一項被否決的後處理決策原文:
`"isotonic=rejected (raw=340.69924, iso=346.99717)"` —— 對混合輸出做 nested OOF isotonic 校準
反而使分數自 340.69924 惡化到 346.99717,因此被自動捨棄。

Submission 檔名為 `sub_blend_v6_340.59891_20260703_211503.csv`(來源:`best.submission`),
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
| 7(best) | LGB | 342.02154 |
| 7(best) | LGB_TUNED | 341.68775 |
| 7(best) | XGB | 342.21778 |
| 7(best) | CAT | 343.60196 |
| 7(best) | LGB_SEED2024 | 342.04208 |
| 7(best) | Ensemble(後處理前) | 340.65207 |
| 7(best) | Ensemble(snap-to-grid 後,= `best.score`) | 340.59891 |
| Public LB | — | 無紀錄 |
| Private LB | — | 無紀錄 |

（來源:`experiments[0].base_models`/`.ensemble`、`best.base_models`/`.ensemble`/`.score`;
`facts.leaderboard` 為 `null`,`facts.missing` 列出 `leaderboard`,顯示本場未提交至 Kaggle
排行榜,故無法計算 CV↔LB gap。）

本場最佳實驗相對第 1 筆基線的分數差距:

```
341.40782 (experiments[0].score, baseline)
- 340.59891 (best.score, Phase B round 4)
= 0.80891
```

Phase B 自我改進相對前一階段最佳(iteration 2)的增量:

```
340.711795 (experiments[2].score, iteration 2)
- 340.59891  (best.score, Phase B round 4)
= 0.112885
```

即最佳實驗較基線有小幅但一致的下降(對絕對誤差指標而言為改善方向),Phase B 4 輪迭代再貢獻
上方第二個算式所示的額外改善。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:11:46 | 341.40782 | generic_batch |
| 2 | 2026-07-03T19:36:46 | 340.958565 | v2 |
| 3 | 2026-07-03T19:39:59 | 340.711795 | v2 |
| 4 | 2026-07-03T21:07:27 | 340.826945 | v2 |
| 5 | 2026-07-03T21:11:27 | 340.627017 | v2 |
| 6 | 2026-07-03T21:12:45 | 340.627017 | v2 |
| 7 | 2026-07-03T21:15:03 | 340.59891 | v2 |

（來源:`facts.trajectory`）

**突破點敘述**(來源:`experiments[].notes`):分數整體呈逐步下降。第 2 筆(iteration 1)引入
27 個特徵並選用 snap-to-grid,分數自 341.40782 降至 340.958565;第 3 筆(iteration 2)剔除 6 個
趨近零貢獻特徵、CatBoost 改用原生類別處理,分數再降至 340.711795。Phase B 4 輪:第 4 筆以 Optuna
調參 LGB 並替換原成員,分數 340.826945(較第 3 筆退步,示範「替換破壞多樣性」);第 5 筆改為
「加入」調參版 LGB 為第 4 成員,分數降至 340.627017(新最佳);第 6 筆試 isotonic 校準被否決,
結果與第 5 筆同分 340.627017;第 7 筆再加入 seed-2024 的 LGB 為第 5 成員,分數降至 340.59891,
為目前最佳。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1: EDA
uv run python3 competitions/playground-series-s3e14/scripts/eda.py

# Stage 3: 建模(iteration 1,分數 340.958565,對應 experiments[1]/submission
#   sub_blend_340.95856_20260703_193646.csv)
uv run python3 competitions/playground-series-s3e14/scripts/train.py

# Stage 3: 建模(iteration 2,分數 340.711795,對應
#   sub_blend_v2_340.71180_20260703_193959.csv)
uv run python3 competitions/playground-series-s3e14/scripts/train_v2.py

# Phase B round 1(Optuna 調參 LGB,分數 340.826945,對應 experiments[3])
uv run python3 competitions/playground-series-s3e14/scripts/train_v3.py

# Phase B round 2(4-way blend,分數 340.627017,對應 experiments[4];
#   會寫出 data/oof_v4.npz 供 round 3/4 重用)
uv run python3 competitions/playground-series-s3e14/scripts/train_v4.py

# Phase B round 3(isotonic 校準試驗,被否決,分數 340.627017,對應 experiments[5];需 oof_v4.npz)
uv run python3 competitions/playground-series-s3e14/scripts/train_v5.py

# Phase B round 4 / best(seed bagging 5-way blend,分數 340.59891,即 best,對應
#   sub_blend_v6_340.59891_20260703_211503.csv;需 oof_v4.npz)
uv run python3 competitions/playground-series-s3e14/scripts/train_v6.py
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有指令均透過 `uv run` 以確保套件環境
一致。`scripts/features.py` 由各 `train*.py` 以模組方式匯入,不需單獨執行;`train_v5.py`/`train_v6.py`
依賴 `train_v4.py` 產生的 `data/oof_v4.npz`,須先執行 round 2。
本場未提交至 Kaggle 排行榜(此環境無 Kaggle 憑證),故無對應的 `kaggle competitions submit`
指令可重現。
