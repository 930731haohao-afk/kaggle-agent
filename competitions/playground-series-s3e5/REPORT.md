# 競賽分析報告:playground-series-s3e5

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-03(Phase B 自我改進迭代更新)

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
- 實驗 4–11(Phase B 自我改進迭代)沿用與實驗 2、3 完全相同的 21 個特徵(來源:
  `experiments[3..10].features`,逐筆核對欄位清單一致),本輪迭代未新增特徵工程。

## 3. 模型規格

本場共 11 筆實驗紀錄,`facts.best` 為實驗 8:

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

**實驗 3(`v2`,21 特徵,採「最佳化分割閾值」後處理)— base models:**

同實驗 2 之 base models 分數。Ensemble 權重(來源:`experiments[2].ensemble.weights`):
LGB 0.2 / XGB 0.2 / CAT 0.6,Ensemble 分數(來源:`experiments[2].ensemble.score`):**0.52687**。
此為 Phase B 迭代前的既有最佳分數。

**實驗 4(方法論調整:更細緻的權重搜尋,同 3 個原始成員)— base models:**

| Model | OOF QWK(post-rounder) |
|-------|------------------------|
| LGB | 0.50547 |
| XGB | 0.50847 |
| CAT | 0.52517 |

Ensemble 權重(來源:`experiments[3].ensemble.weights`):LGB 0.028 / XGB 0.023 / CAT 0.949,
Ensemble 分數(來源:`experiments[3].ensemble.score`):**0.52986**。
(註:此處 base model 分數與實驗 2/3 表列數字不同,因改用 post-rounder QWK 而非 naive-round
QWK 呈現各成員分數;來源皆為 `experiments[].base_models[].score`,逐實驗各自的計算基準已於
`experiments[].notes` 註明。)

**實驗 5(4-way:加入 Optuna 全 CV 直接優化 post-rounder QWK 調校之 LGB_tuned)— base models:**

| Model | OOF QWK(post-rounder) |
|-------|------------------------|
| LGB | 0.50547 |
| XGB | 0.50847 |
| CAT | 0.52517 |
| LGB_tuned | 0.56244 |

Ensemble 權重(來源:`experiments[4].ensemble.weights`):LGB 0.02 / XGB 0.0 / CAT 0.0 /
LGB_tuned 0.98,Ensemble 分數:**0.56293**。

**實驗 6(5-way:LGB_tuned 加碼一組 seed=2024 版本)**:權重全數指向 LGB_tuned(來源:
`experiments[5].ensemble.weights`),分數與實驗 5 相同:**0.56293**(來源:
`experiments[5].ensemble.score`)——seed-bagging 未帶來額外增益。

**實驗 7(巢狀切點診斷,非模型變更)**:分數 **0.54649**(來源:`experiments[6].score`),
為 leave-fold-out 巢狀切點擬合下的誠實 QWK 估計,對照實驗 5/6 之全 OOF 切點擬合分數 0.56293。

**實驗 8(6-way:再加入 Optuna 全 CV 直接優化調校之 CAT_tuned,`facts.best`)— base models:**

| Model | OOF QWK(post-rounder) |
|-------|------------------------|
| LGB | 0.50547 |
| XGB | 0.50847 |
| CAT | 0.52517 |
| LGB_tuned | 0.56244 |
| LGB_tuned_seed2024 | 0.55765 |
| CAT_tuned | 0.56466 |

Ensemble 權重(來源:`experiments[7].ensemble.weights`):LGB 0.05 / XGB 0.0 / CAT 0.0 /
LGB_tuned 0.0 / LGB_tuned_seed2024 0.0 / CAT_tuned 0.95,
Ensemble 分數(來源:`experiments[7].ensemble.score`):**0.56769**。此為 `facts.best`。

**實驗 9(7-way:CAT_tuned 加碼一組 seed=2024 版本)**:分數 **0.56716**(來源:
`experiments[8].ensemble.score`),低於實驗 8 的 0.56769,未取代 running best。

**實驗 10(巢狀切點診斷,基於實驗 8 之 6-way blend,非模型變更)**:分數 **0.56393**(來源:
`experiments[9].score`),對照實驗 8 全 OOF 切點擬合分數 0.56769,差距 0.00377。

**實驗 11(7-way:加入 LGB multiclass + class_weight=balanced + expected-value decode 作為
序數/分類頭的多樣性成員)**:分數 **0.56769**(來源:`experiments[10].ensemble.score`),與實驗
8 相同——權重搜尋將新成員權重歸零(來源:`experiments[10].ensemble.weights`),未帶來增益。

**選型理由**:延續既有的三個梯度提升樹迴歸模型(LightGBM、XGBoost、CatBoost)為基礎,Phase B
迭代依序嘗試:(a) 以 Dirichlet 隨機搜尋取代粗網格權重搜尋(實驗 4)、(b) 對 LightGBM 以 Optuna
在完整 5-fold CV 上直接優化「後處理後的 QWK」(而非原始 RMSE)進行 40 組超參數搜尋,並將調校後
版本加入(而非取代)集成池(實驗 5)、(c) 同法調校 CatBoost 並加入集成池(實驗 8,最終最佳)。
每一步的加入/捨棄決策皆僅依「後處理(OptimizedRounder)後的 QWK」判斷,不依原始 OOF 分數
判斷——此為避免序數/離散化指標特有的「原始分數進步但離散化後不進步」陷阱。另嘗試 seed-bagging
(實驗 6、9)與多分類期望值解碼頭(實驗 11)兩種候選改動,皆未通過此判準,故未採用。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2–11 | 5fold | 5 | 42 |

(來源:`experiments[].cv`;實驗 1 之 seed 未記錄,故寫「無紀錄」;實驗 2–11 皆為
`StratifiedKFold on quality`,標籤分層抽樣,folds/seed 全程保持一致以利跨實驗比較。)

各 base model 之 objective/超參數在 `experiments[].base_models[]` 中未記錄對應欄位,故此項為
「無紀錄」;`LGB_tuned`/`CAT_tuned` 之 Optuna 最佳超參數本身未寫入 `experiments.json`(僅分數與
決策記錄於此),故亦標「無紀錄」。

**為何用此 CV**:`quality` 為小基數(3–8 共 6 級)且嚴重不平衡的序數目標(依 STATUS.md 記載,
最稀有等級與最常見等級之比例懸殊),若採用一般隨機 K-fold 切分,稀有等級可能在某些 fold 中
樣本過少甚至掛零,使該 fold 的 QWK 計算不穩定。全部 11 筆實驗皆改用「依 `quality` 標籤分層
抽樣」的 StratifiedKFold,且 folds/seed 固定不變,使各實驗分數可直接比較。

## 5. 推論程序

`facts.best`(實驗 8)之 `postprocess` 欄位記錄:
`OptimizedRounder(cutpoints=[3.574, 4.586, 5.609, 6.174, 7.628])` 與 `clip[3,8]`(來源:
`experiments[7].postprocess`)。即先以最佳化搜尋得到的 5 個分割閾值,將迴歸模型輸出的連續值
轉換為離散等級,再限制於 3 至 8 的合法範圍內。

實驗 7、10 額外記錄了巢狀切點擬合的後處理描述:`OptimizedRounder(nested: fit on 4 folds,
applied to held-out fold)`(來源:`experiments[6].postprocess`、`experiments[9].postprocess`)——
此為診斷用途,用以檢驗全 OOF 切點擬合是否過度貼合本次 OOF 樣本,並非用於正式提交之後處理。

`facts.best`(實驗 8)之 `submission` 欄位在 `facts.json` 中無紀錄(v2 schema 之 submission 為
選填欄位,`log_experiment_v2` 呼叫時未填入,依 Hard Rule 4 標記「無紀錄」)。對應之提交檔由
獨立的 `scripts/iterate.py submit` 步驟產生於 `submissions/` 目錄下(id 欄 `Id`、目標欄
`quality`),其確切檔名/時間戳記未記入 experiments.json,故不在此覆誦。實驗 1 之 submission
有記錄:`sub_generic_0.47871_20260703_120341.csv`(來源:`experiments[0].submission`)。

**本場所有實驗均未提交至 Kaggle 排行榜**(此執行環境未設定 Kaggle 憑證,`facts.json` 中
`leaderboard` 欄位為 null,且列於 `facts.missing`)。

## 6. 評估指標

**指標定義**:quadratic_weighted_kappa(QWK)= 觀察一致性與隨機期望一致性之差,並以「預測與
真實等級距離的平方」作為誤差權重,數值愈接近 1 表示模型排序與真實等級愈一致。

| 項目 | 分數 |
|------|------|
| 實驗 1 Ensemble(generic_batch,11 特徵) | 0.47871 |
| 實驗 2 Ensemble(四捨五入後處理) | 0.47191 |
| 實驗 3 Ensemble(最佳化閾值後處理,Phase B 迭代前基準) | 0.52687 |
| 實驗 4 Ensemble(更細緻權重搜尋) | 0.52986 |
| 實驗 5 Ensemble(+Optuna 調校 LGB) | 0.56293 |
| 實驗 6 Ensemble(+seed-bag LGB_tuned) | 0.56293 |
| 實驗 7(巢狀切點診斷) | 0.54649 |
| 實驗 8 Ensemble(+Optuna 調校 CAT,facts.best) | **0.56769** |
| 實驗 9 Ensemble(+seed-bag CAT_tuned) | 0.56716 |
| 實驗 10(巢狀切點診斷) | 0.56393 |
| 實驗 11 Ensemble(+多分類期望值解碼頭) | 0.56769 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

(來源:`experiments[].ensemble.score` / `experiments[].score`;`facts.leaderboard` 為 null,
`facts.missing` 列出 `leaderboard`,故 Public/Private LB 皆寫「無紀錄」。)

**CV↔LB gap**:因本場所有實驗皆未提交排行榜,`facts.leaderboard` 不存在,故無法計算 CV↔LB
差值,亦不進行任何推測性比較。

`facts.best`(實驗 8)相對於 Phase B 迭代前既有最佳(實驗 3)之增益,以內嵌算式呈現:

```
實驗 8 OOF QWK − 實驗 3 OOF QWK = 0.56769 − 0.52687 = 0.04082
```

實驗 8 相對於實驗 1(最初基準)之增益:

```
實驗 8 OOF QWK − 實驗 1 OOF QWK = 0.56769 − 0.47871 = 0.08898
```

巢狀切點診斷(實驗 10)顯示全 OOF 切點擬合與誠實巢狀估計之差距:

```
實驗 8 全 OOF 切點 QWK − 實驗 10 巢狀切點 QWK = 0.56769 − 0.56393 = 0.00376
```

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:03:41 | 0.47871 | generic_batch |
| 2 | 2026-07-03T18:48:21 | 0.47191 | v2 |
| 3 | 2026-07-03T18:48:21 | 0.52687 | v2 |
| 4 | 2026-07-03T23:08:22 | 0.52986 | v2 |
| 5 | 2026-07-03T23:09:50 | 0.56293 | v2 |
| 6 | 2026-07-03T23:10:42 | 0.56293 | v2 |
| 7 | 2026-07-03T23:11:20 | 0.54649 | v2 |
| 8 | 2026-07-03T23:13:44 | 0.56769 | v2 |
| 9 | 2026-07-03T23:15:20 | 0.56716 | v2 |
| 10 | 2026-07-03T23:15:32 | 0.56393 | v2 |
| 11 | 2026-07-03T23:16:51 | 0.56769 | v2 |

（來源：`facts.trajectory`）

**突破點**:分數最大的單一躍升發生在實驗 3(0.47191 → 0.52687,後處理由四捨五入改為最佳化
切點,Phase B 迭代前既有結論)。本輪 Phase B 迭代中,最大躍升發生在實驗 5(0.52986 → 0.56293,
`experiments[4].notes` 記載:以 Optuna 在完整 5-fold CV 上直接優化 post-rounder QWK 調校
LightGBM,並將調校版本加入而非取代原集成池),其次是實驗 8(0.56293 → 0.56769,同法調校
CatBoost 加入集成池,`experiments[7].notes`)。實驗 4(更細緻權重搜尋,0.52687 → 0.52986)
為方法論層面的小幅穩定增益。實驗 6、9(seed-bagging)與實驗 11(多分類期望值解碼頭)三次嘗試
權重搜尋皆將新成員權重歸零或分數持平/下降(`experiments[5,8,10].notes`),故未被採用為
`facts.best`。實驗 7、10 為診斷性巢狀切點檢驗,非模型改動,用以評估 STATUS.md 中「切點於全 OOF
擬合可能過擬」的風險,結果顯示差距分別為 0.01644(5-way blend)與 0.00377(`facts.best` 之
6-way blend),差距隨集成池成熟而縮小。

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

# 實驗 4–11(Phase B 自我改進迭代,依序執行,每步驟皆會寫入 experiments.json 並快取
# scripts/cache/*.npz 供後續步驟重用):
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py base         # 實驗 4
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_lgb     # Optuna 調校 LGB(不記錄實驗,寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r1_pool      # 實驗 5
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag     # 實驗 6
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut   # 實驗 7
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_cat     # Optuna 調校 CAT(不記錄實驗,寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r2_pool      # 實驗 8(facts.best)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag     # 實驗 9
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut   # 實驗 10
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r3_multiclass # 實驗 11
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py submit       # 產出最終提交檔

# 提交至 Kaggle(本次執行環境未設定憑證,以下指令供後續有憑證時使用)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e5 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保套件
環境一致。
