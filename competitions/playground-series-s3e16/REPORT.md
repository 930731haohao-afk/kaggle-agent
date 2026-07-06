# 競賽分析報告:playground-series-s3e16

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06(真實 Kaggle LB 驗證入帳後更新;實驗 5 樹搜尋
> 最佳 blend 已提交排行榜,取得 Public 1.34315 / Private 1.33859,詳見第 3a、5、6、7 節。
> 先前 2026-07-04 版本涵蓋 Phase G-1a 樹搜尋成果入帳、Phase B 迭代 exp 3、4 補記)

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

本場現有 5 筆實驗紀錄。**重要語意澄清(本節通篇適用)**:實驗 1、2 的 `score` 欄位是「原始
（未取整)OOF MAE」(source_format 分別為 `skill_train`/`generic_batch`,`collect.py` 正規化時
直接取用未取整的 blend OOF 值);實驗 3、4、5 的 `score` 欄位則是**取整後(rounded)OOF
MAE**——因目標 `Age` 為整數,取整後 MAE 才是本場實際決策指標(見 STATUS.md 記載,取整前後
分數會反向移動,細節見實驗 5 說明)。`facts.best` 純以數字最小值選出,不區分這兩種語意,
故其挑中的實驗未必是「同一把尺」下最佳者——本節下方逐筆說明,不混淆。

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
Ensemble 分數(來源:`experiments[1].ensemble.score`):**1.35441**(原始 OOF MAE,未取整)。

**選型理由(實驗 1、2)**:三個梯度提升樹模型(LightGBM、XGBoost、CatBoost)在表格型資料上
普遍穩健,各自的歸納偏誤不同,故以加權集成降低單一模型的方差。實驗 1 額外投入特徵工程(24
個特徵)並針對 MAE 目標調整訓練細節;實驗 2 為通用批次管線的預設跑法,未套用上述特徵工程但
集成權重不同,結果原始 OOF 分數反而略低——惟此結果僅為內部原始 OOF 比較,實驗 2 並未提交至
Kaggle 排行榜驗證(見節 5、6)。

**實驗 3(Phase B round 1,加入 Optuna fold0-proxy 調參 LGB 為第 4 個池成員)— base models:**

| Model | OOF MAE |
|-------|---------|
| LGB | 1.35651 |
| XGB | 1.35763 |
| CAT | 1.36162 |
| LGB_tuned | 1.35583 |

Ensemble 權重(來源:`experiments[2].ensemble.weights`):LGB 0.3 / XGB 0.1 / CAT 0.1 /
LGB_tuned 0.5,原始 Ensemble 分數(來源:`experiments[2].ensemble.score`):**1.35541**
(較實驗 1 的 1.35589 略有改善)。但 `experiments[2].score`(取整後 OOF MAE)= **1.3385**,
**劣於**當時既有最佳(實驗 1 取整後 1.33812,未記入結構化欄位,見 STATUS.md)——原始分數
進步、取整分數退步,未被採納。

**實驗 4(Phase B round 2,加入 seed=2024 之調參 LGB 為第 5 個池成員)— base models:**

| Model | OOF MAE |
|-------|---------|
| LGB | 1.35651 |
| XGB | 1.35763 |
| CAT | 1.36162 |
| LGB_tuned | 1.35583 |
| LGB_tuned_seed2024 | 1.35604 |

Ensemble 權重(來源:`experiments[3].ensemble.weights`):LGB 0.2 / XGB 0.1 / CAT 0.1 /
LGB_tuned 0.4 / LGB_tuned_seed2024 0.2,原始 Ensemble 分數(來源:
`experiments[3].ensemble.score`):**1.35533**(本場原始 OOF 最佳)。但
`experiments[3].score`(取整後)= **1.33893**,同樣劣於實驗 1 的取整後 1.33812——連續兩輪
取整分數未改善,依 Phase B 停止準則(連 2 輪無改善即停)於此輪後停止線性迭代。

**實驗 5(Phase E-3 樹搜尋,`facts.best`)**:見 3a 節。

### 3a. 樹搜尋最佳(best, experiment_id=5)——本次更新新增

Phase E-3(harness v2 於「取整整數 MAE」上的酸性測試,2026-07-04)在 `experiments_tree.json`
的 node #15 找到本場目前最佳**取整後**OOF MAE:8-way blend(來源:`best.base_models`):

| 成員 | 權重 | solo 取整後 MAE | 備註 |
|------|------|------------------|------|
| LGB(root) | 0.0198 | 1.33885 | — |
| XGB | 0.0256 | 1.3416 | — |
| CAT | 0.5061 | 1.33846 | 主導權重 |
| LGB_tuned | 0.0075 | 1.33979 | — |
| LGB_tuned_seed2024 | 0.0626 | 1.33914 | — |
| LGBBOUND | 0.1117 | 1.3395 | 邊界推進 LGB,learning_rate=0.005 |
| TWEEDIE | 0.0004 | 1.37467 | 權重近乎歸零,但加入時貢獻本次搜尋單筆最大增益 |
| FEATPRUNE | 0.2662 | 1.34025 | 去除與 Weight 高度共線(r=0.993)之欄位;新成員中權重最高,催生全域最佳節點 |

**Ensemble**(best.ensemble):method = "harness_v2 dirichlet(k=800)+coordinate-ascent
weight search, decided directly on ROUNDED MAE",取整後 score = **1.33563**
(`best.score`);同一組權重的原始(未取整)OOF MAE = **1.35712**(`best.ensemble.score`)
——比實驗 1 的原始 1.35589 更差。

**選型理由/來源說明**(best.notes):這是本場「原始 vs 取整」語意反轉的最強例證——本節點
原始 OOF MAE(1.35712)劣於已提交的線性冠軍(實驗 1,原始 1.35589),但**取整後** OOF MAE
(1.33563)優於線性冠軍的取整後 1.33812(改善 -0.00249)。此為實驗 3/4 失敗模式的鏡像
(那裡:原始進步、取整退步;這裡:原始退步、取整進步)——合起來是「決策必須依取整分數、
不能依原始 OOF」這條規則最完整的正反雙向證據。完整節點鏈、backtrack/dedup 統計、與各
lineage 的誠實歸因見 `competitions/playground-series-s3e16/STATUS.md`〈Appendix — Tree-search
v2(Phase E-3,harness_v2,2026-07-04)〉。

> **更新(2026-07-06)**:best.notes 描述的是樹搜尋當下(2026-07-04)的狀態——彼時確為
> OOF-only、未提交。此後 `scripts/rebuild_tree_best.py` 重建出 test 預測並實際提交
> Kaggle(`best.submission` = `sub_tree_best_1.33563_20260706_115200.csv`),取得
> `best.leaderboard`:**Public 1.34315 / Private 1.33859**(提交:2026-07-06)。此結果
> 已是本場**兩筆真實 LB 提交之一**,與實驗 1 的 Public 1.34356 / Private 1.34075 併陳於
> 第 6 節(含改善算式與 CV↔LB gap 一致性檢查)。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold_stratified_agebin | 5 | 無紀錄 |
| 2 | 5fold | 5 | 無紀錄 |
| 3 | 5fold_stratified_agebin | 5 | 42 |
| 4 | 5fold_stratified_agebin | 5 | 42 |
| 5(best,樹搜尋) | 5fold_stratified_agebin | 5 | 42 |

(來源:`experiments[].cv`;實驗 1、2 之 `cv` 未記錄 seed 欄位,故寫「無紀錄」;實驗 3–5 皆為
`5fold_stratified_agebin`(n_splits=5, seed=42)同一組固定折,樹搜尋(實驗 5)沿用與 Phase B
迭代完全相同的 CV。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項亦為
「無紀錄」(實驗 3 之 Optuna 超參僅記於 `experiments[2].notes` 文字說明:fold-0-proxy、21/50
trials、300s timeout,依 Hard Rule 1 不作為表格數字引用)。

**為何用此 CV**:年齡標籤為右偏的整數計數型目標,若採用一般隨機 K-fold 切分,高齡樣本較少,可能
造成各 fold 之目標分布不均、驗證分數不穩定。實驗 1 因此改用「依年齡分箱後分層抽樣」的
StratifiedKFold(將高齡樣本合併為單一分箱以避免箱內樣本過少),使各 fold 的年齡分布更一致,
讓交叉驗證分數更能反映模型的真實泛化能力。

## 5. 推論程序

`facts.best`(實驗 5,樹搜尋)之 `postprocess`(來源:`best.postprocess`):`["round"]`
(對取整後 MAE 做決策,見第 3a 節)。**Submission 檔名(best.submission)**:
`sub_tree_best_1.33563_20260706_115200.csv`——由 `scripts/rebuild_tree_best.py` 重建
8 個成員的 test 預測、以同一組全精度權重混合後產生(2026-07-06),已提交 Kaggle 排行榜
(`best.leaderboard`,結果見第 6 節)。

實驗 2(通用批次管線,原始 OOF 分數在實驗 1、2 中最低者)之 `postprocess` 欄位未記錄任何後
處理步驟,故此欄寫「無後處理紀錄」。其 submission 檔名為
`sub_generic_1.35441_20260703_115739.csv`(來源:`experiments[1].submission`),對應
`id_column = id`、`target_column = Age`,但此筆**並未提交至 Kaggle 排行榜**(`facts.json`
中該實驗無 `leaderboard` 欄位)。

實際提交排行榜者為實驗 1,其 `postprocess` 記錄為 `["round"]`(即對預測值四捨五入,因目標
`Age` 為整數),submission 檔名為 `sub_blend_20260703_091840.csv`(來源:
`experiments[0].submission`),同樣對應 `id_column = id`、`target_column = Age`。此筆之排行榜
結果見節 6。實驗 3、4(Phase B 迭代)之 submission 欄位在 facts.json 中亦無紀錄(見第 3
節,兩輪皆未通過「取整分數需改善」的採納門檻,故未產生正式提交檔)。

## 6. 評估指標

**指標定義**:MAE(Mean Absolute Error)= 預測值與真實值絕對差的平均,單位與目標欄位相同
(此處為年齡)。

| 項目 | 分數 | 語意 |
|------|------|------|
| 實驗 1 Ensemble(2026-07-03 提交#1,線性冠軍) | 1.35589 | 原始 OOF MAE |
| 實驗 2 Ensemble | 1.35441 | 原始 OOF MAE |
| 實驗 3 Ensemble(Phase B round1,未採納) | 1.3385 | 取整後 OOF MAE(`experiments[2].score`) |
| 實驗 4 Ensemble(Phase B round2,未採納) | 1.33893 | 取整後 OOF MAE(`experiments[3].score`) |
| 實驗 5 樹搜尋 Ensemble(facts.best,2026-07-06 提交#2) | **1.33563** | 取整後 OOF MAE(`best.score`) |
| 實驗 5 樹搜尋 Ensemble,同一組權重 | 1.35712 | 原始 OOF MAE(`best.ensemble.score`,劣於實驗 1) |
| Public LB — 提交#1(實驗 1,2026-07-03) | 1.34356 | 取整後(提交檔已四捨五入) |
| Private LB — 提交#1(實驗 1,2026-07-03) | 1.34075 | 取整後(提交檔已四捨五入) |
| Public LB — 提交#2(實驗 5 樹搜尋最佳,2026-07-06) | **1.34315** | 取整後(提交檔已四捨五入) |
| Private LB — 提交#2(實驗 5 樹搜尋最佳,2026-07-06) | **1.33859** | 取整後(提交檔已四捨五入) |

(來源:`experiments[].ensemble.score`、`experiments[].score`、`experiments[0].leaderboard`、
`experiments[4].leaderboard`。本場現有**兩筆**真實 Kaggle 提交:實驗 1(2026-07-03)與
實驗 5(2026-07-06);`facts.leaderboard`(`collect.py` 規則:取最後一筆帶 `leaderboard`
欄位的實驗)現指向較新的實驗 5。)

**CV↔LB gap(提交#1,實驗 1,差值以內嵌算式呈現;實驗 1 的 1.35589 為其原始 OOF,
LB 分數則是對應提交檔取整後的官方結果,兩者本非同一把尺,此處僅依循原報告既有比較方式):**

```
實驗 1 原始 OOF MAE − Public LB  = 1.35589 − 1.34356 = 0.01233
實驗 1 原始 OOF MAE − Private LB = 1.35589 − 1.34075 = 0.01514
```

OOF 分數略高於(即劣於)Public/Private LB,顯示交叉驗證分數並未過度樂觀高估模型表現,LB 分數
反而比 OOF 更好,CV 具參考價值,可安心依 OOF 排序後續實驗。

**樹搜尋(實驗 5)相對已提交線性冠軍(實驗 1)之取整後 MAE 改善,以及同一節點原始 OOF 之
劣化(見第 3a 節「反轉」現象),以內嵌算式呈現:**

```
實驗5取整後 − 實驗1取整後(1.33812,未記入實驗1結構化欄位,取自 experiments[2/3/4].notes
文字紀錄,此處僅供敘述脈絡):
  1.33563 − 1.33812 = −0.00249   (改善)

實驗5原始OOF − 實驗1原始OOF:
  1.35712 − 1.35589 = 0.00123    (劣化,方向與上面相反)
```

**雙 LB 錨點改善(提交#2 相對提交#1,以內嵌算式呈現)**:

```
Public  改善:提交#1(1.34356) − 提交#2(1.34315) = 0.00041
Private 改善:提交#1(1.34075) − 提交#2(1.33859) = 0.00216
```

兩個榜(Public、Private)皆改善(MAE 降低),且改善方向一致——樹搜尋在 OOF 上發現的
取整後增益,此次首度獲得真實 LB 的雙榜獨立驗證,不再只是 OOF-only 的內部比較。

**CV↔LB gap 一致性檢查(提交#2,實驗 5 樹搜尋最佳,取整後 OOF vs 真實 LB,以內嵌算式呈現)**:

```
Public LB  − 實驗5取整後OOF = 1.34315 − 1.33563 = 0.00752
Private LB − 實驗5取整後OOF = 1.33859 − 1.33563 = 0.00296

對照提交#1(實驗1,取整後OOF 1.33812,未記入實驗1結構化欄位,取自
experiments[2/3/4].notes 文字紀錄,此處僅供敘述脈絡):
Public LB  − 實驗1取整後OOF = 1.34356 − 1.33812 = 0.00544
Private LB − 實驗1取整後OOF = 1.34075 − 1.33812 = 0.00263
```

兩筆提交的 Public/Private gap(見上方算式)皆為正值、同一數量級,即真實 LB 分數略劣於
(數字略高於)取整後 OOF——同一方向、同一量級的差距在兩次獨立提交間穩定重現,顯示取整後
OOF MAE 對這個決策指標而言是可信賴的排序依據,未見對 Public LB 過擬合的跡象。
**本場排行榜結果現有兩筆真實提交紀錄**:實驗 1 的 Public 1.34356 / Private 1.34075
(2026-07-03),以及實驗 5 樹搜尋最佳的 Public 1.34315 / Private 1.33859(2026-07-06,
雙榜皆優於提交#1)。

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T09:18:40 | 1.35589 | skill_train |
| 2 | 2026-07-03T11:57:39 | 1.35441 | generic_batch |
| 3 | 2026-07-03T22:56:33 | 1.3385 | v2 |
| 4 | 2026-07-03T22:58:34 | 1.33893 | v2 |
| 5 | 2026-07-04T11:59:43 | 1.33563 | v2 |

（來源：`facts.trajectory`；注意實驗 1、2 之 score 為原始 OOF MAE,實驗 3–5 之 score 為
取整後 OOF MAE,兩者非同一把尺,見第 3 節開頭澄清。）

**突破點 1(exp 1→2)**:第 2 筆實驗(`generic_batch`,通用批次管線)的原始 OOF 分數
(1.35441)低於第 1 筆(`skill_train`,含特徵工程,1.35589)。第 2 筆僅使用原始 8 個欄位、
未套用第 1 筆的 24 個工程特徵,但集成權重不同(LGB 0.3 / XGB 0.2 / CAT 0.5,相較第 1 筆之
LGB 0.6 / XGB 0.3 / CAT 0.1 更偏重 CatBoost),使內部原始 OOF 分數略優於第 1 筆。惟此結果
僅為原始 OOF 內部比較,第 2 筆並未提交 Kaggle 排行榜驗證,無法確認其是否真的更能泛化
(見節 5)。

**Phase B 自我改進迭代(exp 3、4,先前執行但未重產報告,本次補上)**:兩輪皆針對 LightGBM
做 Optuna 調參(exp 3:fold0-proxy,21/50 trials,300s timeout 觸發;exp 4:加碼 seed=2024
的 seed bagging),依驗證過的配方將調參版加入池而非取代原成員。兩輪的**原始** OOF MAE 皆
單調進步(1.35589 → 1.35541 → 1.35533),但**取整後** OOF MAE 皆退步(1.33812 →
1.3385 → 1.33893,退步幅度取自 `experiments[2,3].notes`)——連續 2 輪取整分數未改善,依
協定於 exp 4 後停止線性迭代。此為 `knowledge/experience.md` 記載的「取整整數目標」邊界條件
之直接證據:調參+seed-bagging 帶來的原始增益太小,無法在取整步驟中倖存。

**突破點 2(exp 4→5,本次更新新增,facts.best)**:實驗 5 不是線性迭代的延續回合,而是
Phase E-3(2026-07-04)以 harness v2 對「取整整數 MAE」做的酸性測試——來源
`experiments_tree.json` 的 node #15(27 節點總計,24 已評估,3 個失敗但無害,wall
1409.4s)。此節點的**原始** OOF MAE(1.35712)延續 exp3/4 的退步趨勢、比實驗 1 的
1.35589 更差,但其**取整後** OOF MAE(1.33563)首次真正**優於**實驗 1 的取整後 1.33812
(見第 3a、6 節之反轉現象與內嵌算式)。**更新(2026-07-06,不再是 CV-only)**:
`scripts/rebuild_tree_best.py` 重建出此節點的 test 預測(8 個成員逐一 gate 驗證,重建
OOF 與快取數字逐位一致,詳見 STATUS.md〈Rebuild verification〉節),並實際提交至
Kaggle:`sub_tree_best_1.33563_20260706_115200.csv`(`best.submission`),取得
`best.leaderboard`:**Public 1.34315 / Private 1.33859**(提交:2026-07-06)——雙榜皆
優於實驗 1 的 Public 1.34356 / Private 1.34075(改善算式、CV↔LB gap 一致性檢查見第 6
節)。此為這套樹搜尋配方首次獲得的外部(真實排行榜)驗證,而非僅 OOF-only 推論。完整
節點鏈、8-way blend 組成、與各 lineage 的誠實歸因見
`competitions/playground-series-s3e16/STATUS.md`〈Appendix — Tree-search v2
(Phase E-3, harness_v2, 2026-07-04)〉與〈Rebuild verification〉節。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 實驗 1:手刻管線(EDA → 特徵工程 → 訓練/CV/集成)
uv run python3 competitions/playground-series-s3e16/scripts/eda.py
uv run python3 competitions/playground-series-s3e16/scripts/train.py

# 實驗 2:通用批次管線
uv run python3 competitions/run_competition.py playground-series-s3e16

# Phase B round 1(實驗 3:Optuna fold0-proxy 調參 LGB,未採納)
uv run python3 competitions/playground-series-s3e16/scripts/tune_lgb_optuna.py
uv run python3 competitions/playground-series-s3e16/scripts/round1.py

# Phase B round 2(實驗 4:seed-bagged 調參 LGB,未採納)
uv run python3 competitions/playground-series-s3e16/scripts/round2.py

# Phase E-3 樹搜尋 v2(實驗 5,本場目前 facts.best;可中斷/續跑;樹狀態存
# experiments_tree.json;OOF-only,不產生 test 預測)
uv run python3 tree_search/run_s3e16_v2.py

# 提交至 Kaggle(需先設定有效的 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e16 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。
