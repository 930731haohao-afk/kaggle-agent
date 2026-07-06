# 競賽分析報告:playground-series-s3e14

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03(2026-07-04 更新:Phase G-1b 樹搜尋成果入帳,實驗 8)

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

## 2. 流程(how):五大元件

### 2.1 資料規格

本場已完整執行 EDA(`scripts/eda.py`)與特徵工程(`scripts/features.py`),`facts.json` 的
`material_level` 為 `full`,`status_md_present` 為 `true`(競賽層 `STATUS.md` 已建立,記錄 EDA
narrative 發現)。

- 依 `facts.json`,本場共有 8 筆實驗紀錄:第 1 筆(`generic_batch`,通用批次基線)使用
  16 個特徵(來源:`experiments[0].n_features`);第 2 筆(特徵工程 iteration 1)使用 27 個特徵
  (來源:`experiments[1].n_features`);第 3 筆(特徵剪枝 iteration 2)起至第 7 筆(線性迭代
  終點)均使用 21 個特徵(來源:`experiments[2].n_features` / `experiments[6].n_features`)。
  第 4–7 筆為 Phase B 自我改進迭代(Optuna 調參、blend 成員擴充、isotonic 校準試驗、seed bagging)。
  **第 8 筆(facts.best,Phase G-1b 樹搜尋)不是線性迭代的延續回合**,細節見第 2.2 節末小節。
- 原始資料列數/欄位型別、缺漏值計數等具體數值未被 `collect.py` 收錄進 `facts.json`(該檔僅記錄
  competition 中繼資料與 experiments 結構化欄位,不解析 EDA 腳本輸出),故此類細項為
  **無紀錄**。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。

### 2.2 模型規格

**facts.json 目前的 best 是 experiment_id=8——一筆樹搜尋(tree-search)結果**(細節見
2.2b 小節),而非本節原本描述的線性迭代終點(experiment_id=7)。兩者皆完整說明如下。

#### 2.2a 線性迭代最佳(experiment_id=7,Phase B round 4 之終點)

五個 base model 分數:

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
一個 seed 多樣化的 LGB(相同超參、random_state 2024)作為第 5 個成員,取得 340.59891
(線性迭代終點)。

#### 2.2b 樹搜尋最佳(best,experiment_id=8)——本次更新(Phase G-1b)新增

Phase F-2(harness v3,2026-07-04,預設自動策略驗證跑)樹搜尋在 experiment_id=7 的相同
21-特徵集上另闢節點空間,於 `experiments_tree_v3.json` 的 node #44 找到 MAE 更低的
**34-way「kitchen-sink」mega-blend**——這是 harness v3 的 explore-burst 機制強制觸發的
長射(long-shot)節點,把整場搜尋累積的全部 34 個 solo 成員一次納入權重搜尋
(第 60-評估硬上限中的第 44 次評估;node wall_s=1.64s,純權重搜尋、零重訓)。

| 模型 | OOF MAE |
|------|---------|
| LGB_ROOT | 342.02154 |
| CAT | 343.60196 |
| XGB | 342.21778 |
| LGBTUNED | 341.68775 |
| SEEDBAG_S2024 | 342.04208 |
| FEAT | 342.02283 |
| REG | 342.93033 |
| XGBTUNED | 341.9943 |
| LGBTUNED_NEW | 341.89372 |
| LGBTUNED_S4000 | 341.63373 |
| LGBTUNED_S4001 | 341.91928 |
| XGBTUNED_S4000 | 342.36635 |
| XGBTUNED_S4001 | 342.1212 |
| XGBTUNED_S4002 | 342.49949 |
| FEAT_NEW2 | 342.28883 |
| FEAT_S4000 | 341.89382 |
| FEAT_S4001 | 342.07229 |
| SEEDBAG_S555 | 341.68944 |
| SEEDBAG_S4000 | 342.05908 |
| SEEDBAG_S4001 | 342.23701 |
| XGB_NEW | 342.521 |
| XGB_S4000 | 342.89275 |
| XGB_S4001 | 342.57559 |
| REG_L2 | 342.85043 |
| REG_LEAVES90 | 343.73934 |
| REG_S4000 | 342.81987 |
| CAT_NEW | 344.30153 |
| CAT_S4000 | 343.88165 |
| CAT_S4001 | 343.99525 |
| EXPL_DART | 6156.32115 |
| EXPL_XT | 344.82504 |
| EXPL_CATDEEP | 345.1579 |
| EXPL_XGBLEAF | 343.21412 |
| EXPL_REGDEEP | 341.99788 |

（來源:`best.base_models`,34 個成員。注意 EXPL_DART 是災難性的離群 solo(OOF MAE
6156.32115),Boosting 用了 DART 模式,是 explore-burst「長射」機制刻意嘗試的高風險想法
之一,並未拖累最終 blend——見下方權重說明。）

**Ensemble**(`best.ensemble`):方法為「harness_v3 explore_burst mega-blend: k=800
dirichlet + coordinate-ascent 權重搜尋,34 個快取 solo OOF 向量、零重訓」。raw blend 分數
`score_raw` = 340.57853,經 snap-to-grid 後(`best.postprocess`)為 **340.35572**
(= `best.score`)。**個別成員權重未被此 harness_v3 樹節點格式記錄下來**(不同於 v2 節點
schema)——`experiments_tree_v3.json` 僅記錄成員清單與 raw/snap 兩個分數。依
`best.ensemble.n_members_weight_over_0.005` = 28,34 個成員中有 28 個保留 >0.005 的權重
(其餘未記錄,不臆測);依 STATUS.md 的敘述性記錄(數值未收錄進 facts.json),4 個 CatBoost
變體合計約 0.22 權重,兩個 explore-burst 長射倖存者 EXPL_CATDEEP(solo 345.1579,權重
約 0.051)與 EXPL_REGDEEP(solo 341.99788,權重約 0.045)亦進入最終 blend。

**選型理由/來源說明**(`best.notes`):這是 Phase F-2 樹搜尋(harness v3 預設自動策略驗證
跑)結果,不是線性迭代的延續回合。較 experiment_id=7(340.59891)改善 -0.24319(約
-0.071% 相對改善,詳見第 2.5 節程式區塊),也優於先前 harness v1-proto 樹的結果(Phase
C-2b,340.52635)。explore-burst(評估 39–44)貢獻了 exploit 階段結束後全部的增益
(340.45150 → 340.35572);本輪 60 次評估在硬預算上限停止,並非透過 patience 自動停止
機制觸發。完整節點鏈、phase machine 行為、與誠實操作記錄(含 6 次災難性 DART 評估)見
`competitions/playground-series-s3e14/STATUS.md`〈Appendix: Phase F-2 harness v3
validation run〉。

> **重要澄清**:`best.notes` 明確記載這是 **OOF-only 搜尋結果——未產生任何 test 預測,亦
> 未提交至 Kaggle**(facts.json 本筆無 submission 欄位)。best 是以 OOF score 最小者選出
> (本場 metric 為 mae,minimize),與是否已提交至 Kaggle 無關;本場(experiment_id=7 與
> experiment_id=8 皆同)未提交至 Kaggle(見第 2.5 節),因此沒有 leaderboard 分數可與 best
> 對照。

作為對照,第 1 筆實驗(通用批次基線)的 base model 分數為:

| Model | OOF MAE |
|-------|---------|
| LGB | 345.32118 |
| XGB | 344.13027 |
| CAT | 344.24276 |

（來源:`experiments[0].base_models`,ensemble 權重 LGB 0.3 / XGB 0.3 / CAT 0.4,分數
341.40782,來源:`experiments[0].ensemble`。）

### 2.3 訓練規格

| 實驗 | source_format | CV scheme | n_splits | seed |
|------|----------------|-----------|----------|------|
| 1(baseline) | generic_batch | 5fold | 5 | 無紀錄 |
| 2(iteration 1) | v2 | 5fold_kfold | 5 | 42 |
| 3(iteration 2) | v2 | 5fold_kfold | 5 | 42 |
| 4(Phase B round 1) | v2 | 5fold_kfold | 5 | 42 |
| 5(Phase B round 2) | v2 | 5fold_kfold | 5 | 42 |
| 6(Phase B round 3) | v2 | 5fold_kfold | 5 | 42 |
| 7(Phase B round 4,線性迭代終點) | v2 | 5fold_kfold | 5 | 42 |
| 8(樹搜尋,best) | v2 | 5fold_kfold | 5 | 42 |

（來源:`experiments[].cv`;第 1 筆之 `cv` 物件無 `seed` 欄位,故寫「無紀錄」;第 2–8 筆 seed 均為 42,與線性迭代同折,分數可直接比較。）

多數 base model 之 objective/超參數未記錄於 `experiments[].base_models[]` 對應欄位中,惟第 4 筆
(`experiments[3].notes`)完整記錄了 Optuna 為 LightGBM 找到的最佳超參:`learning_rate`
0.015389816656608301、`num_leaves` 63、`max_depth` 5、`min_child_samples` 30、`subsample`
0.7899031923134097、`colsample_bytree` 0.6842166782718478、`reg_alpha` 0.012648426946439915、
`reg_lambda` 0.00346212601571565(36 trials、293 秒,採 fold-0 單折代理目標)。此組超參即後續
第 5、7 筆中的 `LGB_TUNED` 成員。

**為何用此 CV**:第 2、3 筆實驗改採 5-fold KFold(shuffle,seed=42)。依 `STATUS.md` 記載之
EDA 脈絡,目標為連續值且無分組/時間結構,train/test 各欄位分布差異小,故不需分層抽樣(與
s3e16 的離散 Age 目標不同),一般 i.i.d. KFold 即為合適的驗證方案。

### 2.4 推論程序

**facts.json 現在的 best 是 experiment_id=8(樹搜尋)**,其 `postprocess` 欄位記錄
`"snap_to_grid (applied inside metric_fn on the raw blend OOF, not just at submission
time)"`——與 experiment_id=7 的差異在於樹搜尋版本於 **metric_fn 內部**對 raw blend OOF
直接做 snap-to-grid 評分,而非僅在 submission 階段套用。

以下為線性迭代終點 experiment_id=7 的後處理紀錄(來源:`experiments[6].postprocess`):

1. `"snap_to_grid=used (raw=340.65207, snapped=340.59891)"`

即:5-way 網格混合輸出最終被「snap 至訓練集中曾觀測到的最近 yield 值」,分數自 340.65207
降至 340.59891。snap-to-grid 在本場歷次迭代(含樹搜尋)皆為正向後處理。

作為對照,round 3(`experiments[5].postprocess`)另記錄了一項被否決的後處理決策原文:
`"isotonic=rejected (raw=340.69924, iso=346.99717)"` —— 對混合輸出做 nested OOF isotonic 校準
反而使分數自 340.69924 惡化到 346.99717,因此被自動捨棄。

**Submission 檔名(best.submission,experiment_id=8)**:**無紀錄**——experiment_id=8 是
樹搜尋(OOF-only)結果,facts.json 本筆未附 submission 欄位,未產生 test 預測、未提交
Kaggle(見 2.2b 節澄清)。線性迭代終點(experiment_id=7)有提交檔案供參考:
`sub_blend_v6_340.59891_20260703_211503.csv`(對應 OOF 340.59891,非目前 best 的
340.35572),對應 `id_column = id`、`target_column = yield`(來源:
`competition.id_column` / `competition.target_column`)。

### 2.5 評估指標

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
| 7(線性迭代終點) | Ensemble(後處理前) | 340.65207 |
| 7(線性迭代終點) | Ensemble(snap-to-grid 後) | 340.59891 |
| 8(樹搜尋,best) | Ensemble(raw,34-way) | 340.57853 |
| 8(樹搜尋,best) | Ensemble(snap-to-grid 後,= `best.score`) | **340.35572** |
| Public LB | — | 無紀錄 |
| Private LB | — | 無紀錄 |

（來源:`experiments[0].base_models`/`.ensemble`、`experiments[6].base_models`/`.ensemble`/
`.score`、`best.ensemble`/`.score`(experiment_id=8);
`facts.leaderboard` 為 `null`,`facts.missing` 列出 `leaderboard`,顯示本場未提交至 Kaggle
排行榜,故無法計算 CV↔LB gap。）

線性迭代終點(exp7)相對第 1 筆基線的分數差距:

```
341.40782 (experiments[0].score, baseline)
- 340.59891 (experiments[6].score, Phase B round 4, 線性迭代終點)
= 0.80891
```

Phase B 自我改進相對前一階段最佳(iteration 2)的增量:

```
340.711795 (experiments[2].score, iteration 2)
- 340.59891  (experiments[6].score, Phase B round 4)
= 0.112885
```

樹搜尋(exp8,best)相對線性迭代終點(exp7)的改善:

```
340.59891 (experiments[6].score, 線性迭代終點)
- 340.35572 (best.score, 樹搜尋 node #44)
= 0.24319
0.24319 / 340.59891 × 100 ≈ 0.0714...%(約 -0.071% 相對改善)
```

樹搜尋(exp8)相對先前 harness v1-proto 樹結果(Phase C-2b,340.52635,未列於本場
experiments.json,見 STATUS.md)的改善:

```
340.52635 - 340.35572 = 0.17063
```

即最佳實驗較基線有小幅但一致的下降(對絕對誤差指標而言為改善方向),Phase B 4 輪迭代再貢獻
上方第二個算式所示的額外改善,樹搜尋(Phase G-1b 入帳)再貢獻第三個算式所示的額外改善。

## 3. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:11:46 | 341.40782 | generic_batch |
| 2 | 2026-07-03T19:36:46 | 340.958565 | v2 |
| 3 | 2026-07-03T19:39:59 | 340.711795 | v2 |
| 4 | 2026-07-03T21:07:27 | 340.826945 | v2 |
| 5 | 2026-07-03T21:11:27 | 340.627017 | v2 |
| 6 | 2026-07-03T21:12:45 | 340.627017 | v2 |
| 7 | 2026-07-03T21:15:03 | 340.59891 | v2 |
| 8 | 2026-07-04T12:17:48 | 340.35572 | v2 |

（來源:`facts.trajectory`。最佳分數現為第 8 筆(樹搜尋),即 `facts.best`。）

**突破點敘述**(來源:`experiments[].notes`):分數整體呈逐步下降。第 2 筆(iteration 1)引入
27 個特徵並選用 snap-to-grid,分數自 341.40782 降至 340.958565;第 3 筆(iteration 2)剔除 6 個
趨近零貢獻特徵、CatBoost 改用原生類別處理,分數再降至 340.711795。Phase B 4 輪:第 4 筆以 Optuna
調參 LGB 並替換原成員,分數 340.826945(較第 3 筆退步,示範「替換破壞多樣性」);第 5 筆改為
「加入」調參版 LGB 為第 4 成員,分數降至 340.627017(新最佳);第 6 筆試 isotonic 校準被否決,
結果與第 5 筆同分 340.627017;第 7 筆再加入 seed-2024 的 LGB 為第 5 成員,分數降至 340.59891,
為線性迭代終點。

**突破點(Phase G-1b,本次更新新增)**:第 8 筆不是線性迭代的延續回合,而是 Phase F-2
(2026-07-04)以 harness v3 執行的**樹搜尋(tree-search)**結果——來源
`experiments_tree_v3.json` 的 node #44,harness v3 的 explore-burst 機制強制觸發的
「kitchen-sink mega-blend」,把整場搜尋累積的全部 34 個 solo 成員一次納入權重搜尋
(60-評估硬上限中的第 44 次評估)。樹搜尋在 exp7 的相同 21-特徵集上另闢節點空間,分數由
340.59891 降至 340.35572(相對改善見上方第 2.5 節程式區塊)。**誠實 CV-only 警語**:此結果
為 OOF-only 搜尋產物——tree_search harness 未產生任何 test 預測檔,facts.json 本筆亦無
submission 欄位,**未提交至 Kaggle**;不可與 exp7 實際提交的 submission 檔案混淆(見第 2.4
節)。完整節點鏈、explore-burst 機制、與誠實操作記錄(含 6 次災難性 DART 評估)見
`competitions/playground-series-s3e14/STATUS.md`〈Appendix: Phase F-2 harness v3
validation run〉。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 4. 重現指令

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

# Phase B round 4(seed bagging 5-way blend,分數 340.59891,線性迭代終點,對應
#   sub_blend_v6_340.59891_20260703_211503.csv;需 oof_v4.npz)
uv run python3 competitions/playground-series-s3e14/scripts/train_v6.py

# Phase F-2 樹搜尋 v3(best,實驗 8,分數 340.35572)— resumable;軌跡存於 experiments_tree_v3.json
uv run python3 tree_search/run_s3e14_v3.py
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有指令均透過 `uv run` 以確保套件環境
一致。`scripts/features.py` 由各 `train*.py` 以模組方式匯入,不需單獨執行;`train_v5.py`/`train_v6.py`
依賴 `train_v4.py` 產生的 `data/oof_v4.npz`,須先執行 round 2。
本場未提交至 Kaggle 排行榜(此環境無 Kaggle 憑證),故無對應的 `kaggle competitions submit`
指令可重現。
