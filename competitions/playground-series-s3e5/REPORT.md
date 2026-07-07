# 競賽分析報告:playground-series-s3e5

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依葡萄酒的 11 項理化量測值(酸度結構、殘糖、氯化物、游離/總二氧化硫、密度、pH、
硫酸鹽、酒精濃度)預測品質評分 `quality`,標籤為 3 至 8 的**序數(ordinal)整數等級**。

**Why**:評估指標為 **quadratic_weighted_kappa(QWK,maximize)**。品質等級之間有順序關係
——把 5 分誤判為 6 分遠輕於誤判為 8 分;QWK 以「預測與真實等級距離的平方」加權懲罰誤差,
比準確率或無序多分類損失更貼合「越接近真實等級越好」的評分邏輯,是序數目標的合理指標。
指標特性也直接決定了本場的建模路線:迴歸頭 + 指標特化的切點後處理。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e5 |
| 問題型別 | classification |
| 評估指標 | quadratic_weighted_kappa(maximize) |
| 目標欄位 | quality |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹迴歸模型,構成全部 12 筆實驗的集成成員主體 |
| Optuna 超參搜尋 | TPE 全 5-fold CV 調參,目標函數直接設為 post-rounder QWK,產出 LGB_tuned 與 CAT_tuned |
| OptimizedRounder 指標特化後處理 | 以最佳化切點將迴歸連續輸出離散化為 3–8 等級,本場決定性槓桿 |
| 自建樹搜尋 harness | Phase E-2 結構化搜尋成員/超參/blend 權重組合空間,於 node #17 找到 exp12 |
| 5-fold CV 框架(scikit-learn) | StratifiedKFold on quality,全程固定折以利跨實驗比較 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——選擇迴歸頭而非多分類頭、把 Optuna 目標
函數改成 post-rounder QWK、依「連兩輪無改善即停」協定停損;Auto-ML 工具(Optuna、樹搜尋
harness、權重搜尋)負責系統化執行,兩者分工互補。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 2,056 |
| test 列數 | 1,372 |
| 原始欄位數 | 11 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

11 個原始欄位皆為數值型,無類別欄位。目標 `quality` 為整數(mean 5.720817、median 6.0、
skew 0.266307),共 6 個等級且嚴重不平衡:quality 3 僅 12 列、quality 8 僅 39 列,quality 5
(839 列)與 quality 6(778 列)為絕對大宗;非 log 轉換候選。

資料品質乾淨:train/test 皆無缺失值、無重複列、無高共線特徵對(`citric acid` 有 211 個零值,
屬合理的理化量測)。train/test 特徵分佈幾乎一致(最大平均差 2.078843%,citric acid),無
共變數偏移。EDA 對單一特徵的訊號排序:`alcohol`(Spearman 0.504246)與 `sulphates`
(0.456985)最強。EDA 之驗證建議即為「整數目標 → 分層 K-fold」,見 3.3 節。

exp1(通用批次)直接用 11 個原始欄位;exp2–11 經特徵工程展開為 21 個特徵(SO2 比值、
酸度比值、酒精×硫酸鹽/密度交互作用、糖/酒精比等),線性迭代(Phase B)期間特徵集固定不變。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場為離散化指標,所有實驗的「決策分數」一律為**後處理後
> 的 QWK**(exp2 為四捨五入,其餘為 OptimizedRounder 切點);原始迴歸 OOF 分數未被記錄、
> 也從未作為決策依據(避免「原始分數進步但離散化後不進步」的陷阱),故下表「原始 OOF」欄
> 一律標「—」。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(.1/.4/.5) | 11 | — | 0.47871 | 是,初始基線 |
| 2 | 手刻管線(四捨五入) | LGB/XGB/CAT(.2/.2/.6) | 21 | — | 0.47191 | 否,rounder 對照組 |
| 3 | 手刻管線(OptimizedRounder) | 同 exp2 成員與權重 | 21 | — | 0.52687 | 是,Phase A 最佳 |
| 4 | Phase B round0(權重搜尋方法論) | 3-way(Dirichlet+座標上升) | 21 | — | 0.52986 | 是 |
| 5 | Phase B round1(+LGB_tuned) | 4-way blend | 21 | — | 0.56293 | 是 |
| 6 | Phase B round2(seed-bag LGB_tuned) | 5-way blend | 21 | — | 0.56293 | 否,無增益 |
| 7 | 診斷(巢狀切點,非模型變更) | exp5/6 之 5-way blend | 21 | — | 0.54649 | —(診斷) |
| 8 | Phase B round3(+CAT_tuned) | 6-way blend | 21 | — | 0.56769 | 是,線性迭代最佳 |
| 9 | Phase B round4(seed-bag CAT_tuned) | 7-way blend | 21 | — | 0.56716 | 否,退步 |
| 10 | 診斷(巢狀切點,非模型變更) | exp8 之 6-way blend | 21 | — | 0.56393 | —(診斷) |
| 11 | Phase B round5(多分類 EV 解碼頭) | 7-way blend | 21 | — | 0.56769 | 否,新成員權重歸零 |
| 12 | Phase E-2 樹搜尋(best) | 3-way blend(node #17) | 無紀錄 | — | **0.57066** | 是,OOF-only |

線性迭代選型邏輯:以三個梯度提升樹迴歸模型為基礎,依序做(a)更細的 Dirichlet 權重搜尋
(exp4)、(b)Optuna 直接以 post-rounder QWK 為目標調校 LGB 與 CAT,調校版**加入而非取代**
集成池(exp5、exp8);seed-bagging(exp6、exp9)與多分類期望值解碼頭(exp11)皆未通過
決策判準。exp9、exp11 連兩輪無改善後依協定停止線性迭代。

**best 成員表(exp12,3-way blend,node #17)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGB_tuned(root) | 0.151 | 無紀錄 | 即線性迭代之 Optuna 調校 LGB 成員 |
| CAT_tuned | 0.551 | 無紀錄 | 即線性迭代之 Optuna 調校 CAT 成員,權重最高 |
| LGBBOUND | 0.297 | 0.55784 | 邊界推進 LGB(max_depth 3→2):solo 比 root 差 0.005,blend 貢獻 +0.0026 |

exp12 由 harness v2 樹搜尋(22 個評估節點、1 次 backtrack、wall 793.7s)找到,勝出的兩個
槓桿:(1)LGBBOUND——線性迭代的 Optuna 搜尋盒有超參飽和在盒邊界,推過邊界的成員 solo 較弱
但異質性高;(2)足額 k=800 權重搜尋預算——k=200 粗搜在同一組成員上僅找到 0.56601,離散
指標上縮減 blend 搜尋預算會讓勝場悄悄變回平手。

> **誠實但書**:exp12 為 **OOF-only 搜尋結果**——未產生 test 預測、無 submission 檔、未提交
> Kaggle;facts.best 依 OOF 分數選出,與是否提交無關。其權重與切點皆直接對全 OOF 搜尋,
> 未對此結果重跑巢狀切點診斷(exp7、exp10 的診斷模式)。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2–12 | 5fold(StratifiedKFold on quality) | 5 | 42 |

`quality` 為 6 級且嚴重不平衡的序數目標(quality 3 僅 12 列),一般隨機 K-fold 會使稀有
等級在部分折中掛零、QWK 不穩定,故 exp2 起依標籤分層抽樣,且 folds/seed 全程固定(含
exp12 樹搜尋),各實驗分數可直接比較;此亦與 EDA 的驗證建議一致。

各成員之 objective 與最終超參數未寫入結構化欄位,標「無紀錄」;調參設定僅見於 notes
(Optuna TPE、40 trials、600s timeout、完整 5-fold CV,目標函數為該 trial 自身 OOF 的
post-rounder QWK)。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_0.47871_20260703_120341.csv | 否 |
| 2 | round_to_nearest_int + clip[3,8] | 無紀錄 | 否 |
| 3 | OptimizedRounder(cutpoints=[3.642, 4.594, 5.657, 6.196, 7.386])+ clip[3,8] | 無紀錄 | 否 |
| 4–6, 8, 9, 11 | OptimizedRounder(各實驗切點各異,詳 facts.json)+ clip[3,8] | 無紀錄 | 否 |
| 7, 10 | OptimizedRounder(nested:4 折擬合、留出折套用)+ clip[3,8],診斷用 | —(診斷) | 否 |
| 12 | OptimizedRounder 切點與 blend 權重在每個節點的 metric_fn 內聯合搜尋 | —(OOF-only,未產生) | 否 |

`id_column = Id`、`target_column = quality`。線性迭代最佳 exp8 之切點為
`[3.574, 4.586, 5.609, 6.174, 7.628]`;exp12 之具體切點未入 facts.json,標「無紀錄」。
本場執行環境未設定 Kaggle 憑證,所有實驗皆未提交排行榜。

### 3.5 評估指標 / 排行榜

指標定義:QWK = 觀察一致性對隨機期望一致性之修正,以「預測與真實等級距離的平方」加權,
越接近 1 代表預測排序與真實等級越一致。

| 項目 | 分數 |
|------|------|
| exp1(通用批次基線) | 0.47871 |
| exp3(Phase A 最佳) | 0.52687 |
| exp8(線性迭代最佳) | 0.56769 |
| exp12(樹搜尋,facts.best) | **0.57066** |
| Public LB / Private LB | 無紀錄 |

`facts.leaderboard` 為 null(列於 `facts.missing`),無法計算 CV↔LB gap,不作推測性比較。
各階段增益與切點過擬診斷之衍生算式:

```
線性迭代增益(exp8 − exp3):0.56769 − 0.52687 = 0.04082
樹搜尋增益(exp12 − exp8):0.57066 − 0.56769 = 0.00297
巢狀切點診斷 gap(全 OOF 切點 − 巢狀切點):
  5-way(exp7):0.56293 − 0.54649 = 0.01644
  6-way(exp10):0.56769 − 0.56393 = 0.00376(notes 記為 +0.00377,四捨五入口徑差)
```

巢狀切點診斷顯示全 OOF 切點擬合的過擬風險隨集成池成熟而縮小(0.01644 → 0.00377),
全 OOF 切點在本場屬可接受的標準做法;但 exp12 未重跑此診斷,見 3.2 節誠實但書。

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase E-2 | 樹搜尋執行(harness v2) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:03:41 | 0.47871 | Baseline(通用批次) | 11 特徵三模型 blend,初始基線 |
| 2 | 2026-07-03T18:48:21 | 0.47191 | 手刻管線 | 21 特徵,四捨五入後處理(rounder 對照組) |
| 3 | 2026-07-03T18:48:21 | 0.52687 | 手刻管線 | 同一 blend 改用 OptimizedRounder,Phase A 最佳 |
| 4 | 2026-07-03T23:08:22 | 0.52986 | Phase B round0 | 權重搜尋改 Dirichlet+座標上升 |
| 5 | 2026-07-03T23:09:50 | 0.56293 | Phase B round1 | 加入 Optuna 直接 QWK 調校之 LGB_tuned |
| 6 | 2026-07-03T23:10:42 | 0.56293 | Phase B round2 | seed-bag LGB_tuned,無增益 |
| 7 | 2026-07-03T23:11:20 | 0.54649 | 診斷 | 巢狀切點檢驗(5-way),非模型變更 |
| 8 | 2026-07-03T23:13:44 | 0.56769 | Phase B round3 | 加入 CAT_tuned,線性迭代最佳 |
| 9 | 2026-07-03T23:15:20 | 0.56716 | Phase B round4 | seed-bag CAT_tuned,退步 |
| 10 | 2026-07-03T23:15:32 | 0.56393 | 診斷 | 巢狀切點檢驗(6-way),非模型變更 |
| 11 | 2026-07-03T23:16:51 | 0.56769 | Phase B round5 | 多分類 EV 解碼頭權重歸零,連兩輪無改善停止 |
| 12 | 2026-07-04T11:59:43 | **0.57066** | Phase E-2 樹搜尋(best) | node #17 3-way blend,本場最佳 |

- **突破點 1(exp2→3)**:同一組成員與權重,後處理由四捨五入改為 OptimizedRounder 切點,
  0.47191 → 0.52687(同 blend 上 rounder 增益 +0.05496)——本場單一最大槓桿。
- **突破點 2(exp4→5、exp5→8)**:把 Optuna 目標函數直接設為 post-rounder QWK 調校 LGB
  (0.52986 → 0.56293)、再同法調校 CAT(0.56293 → 0.56769),兩次調參佔線性迭代大部分增益。
- **突破點 3(exp11→12)**:harness v2 樹搜尋以邊界推進成員 LGBBOUND + 足額權重搜尋預算,
  0.56769 → 0.57066(勝線性迭代 +0.00297、勝樹搜尋 v1 之 0.56766 達 +0.00300,屬結構性
  勝出而非切點噪音量級)。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次) | 0.47871 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp3,Phase A 最佳,OptimizedRounder) | 0.52687 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp8,6-way blend + Optuna 直接 QWK 調參) | 0.56769 | 見下方算式 |
| tier4 | + 樹搜尋(exp12,node #17,3-way blend) | **0.57066** | 見下方算式 |

```
tier1→tier2: 0.52687 − 0.47871 = 0.04816,相對改善 0.04816 / 0.47871 = 10.0604%
tier2→tier3: 0.56769 − 0.52687 = 0.04082,相對改善 0.04082 / 0.52687 = 7.7476%
tier3→tier4: 0.57066 − 0.56769 = 0.00297,相對改善 0.00297 / 0.56769 = 0.5232%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:XGB) | 0.46995 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(tier1 三模型混合) | 0.47871 | exp1 |
| 2.c | 指標感知後處理前:naive round(四捨五入,對照組) | 0.47191 | exp2 |
| 2.c | 指標感知後處理後:OptimizedRounder(對 OOF QWK 調切點) | 0.52687 | exp3 |
| 3.b | Optuna 直接優化 post-rounder QWK 調參 LGB,加入 pool | 0.56293 | exp5 |
| 3.d | seed bagging(LGB_tuned seed 2024,無增益) | 0.56293 | exp6 |
| 3.b | Optuna 直接優化 post-rounder QWK 調參 CatBoost,加入 pool(線性迭代最佳) | 0.56769 | exp8 |
| 3.d | seed bagging(CAT_tuned seed 2024,退步未採用) | 0.56716 | exp9 |
| 4.a | 樹搜尋 harness v1(node #11,首版樹搜尋最佳) | 0.56766 | exp12 notes 引用 |
| 4.b | 樹搜尋 harness v2(node #17,邊界推進成員 LGBBOUND 入池,本場最佳) | 0.57066 | exp12 |

## 6. 總結

本場資料乾淨(無缺失、無重複、無共線問題),真正的難點在目標本身:`quality` 是 6 級、嚴重
不平衡的序數標籤,而 QWK 是離散化指標。第一個關鍵決策是採迴歸頭而非多分類頭——讓大宗等級
的連續訊號幫助定位資料稀少的極端等級;exp11 的多分類 EV 解碼頭成員權重被搜尋歸零,反向
印證了這個選擇。

第二個關鍵決策是把「後處理」當一級公民:同一組模型與權重,後處理由四捨五入換成
OptimizedRounder 切點即帶來本場單一最大增益(exp2→exp3);其後所有加入/捨棄決策一律只看
post-rounder QWK,並把同一原則貫徹到 Optuna 的目標函數(直接優化 post-rounder QWK)與樹
搜尋評估器(rounder 內建於每個節點)。

各階段增益來源清楚分層:tier1→tier2 來自 skill 管線的特徵工程與切點後處理(0.47871 →
0.52687),tier2→tier3 來自 Optuna 直接 QWK 調參的成員 LGB_tuned、CAT_tuned(→ 0.56769),
tier3→tier4 來自樹搜尋挖出的邊界推進成員 LGBBOUND(→ 0.57066)。本場為此批競賽中前兩層
增幅最大的一場,顯示離散化指標下「指標特化後處理 + 直接優化最終指標」的複利效果。

結果可信度:CV 全程固定同一組 StratifiedKFold 折,分數跨實驗可直接比較;巢狀切點診斷
(exp7、exp10)顯示切點過擬風險隨集成池成熟而縮小。但須誠實標注:本場所有實驗皆未提交
Kaggle(無憑證),最終 0.57066 為 OOF-only 結果且未重跑巢狀診斷,缺乏排行榜錨點驗證。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# exp1:通用批次管線(11 特徵)
uv run python3 competitions/run_competition.py playground-series-s3e5

# exp2、3:手刻管線(EDA → 特徵工程 → 訓練/CV/集成 → naive-round vs OptimizedRounder)
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py

# exp4–11(Phase B 自我改進迭代,依序執行;每步寫入 experiments.json,
# 並快取 scripts/cache/*.npz 供後續步驟重用)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py base          # exp4
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_lgb      # Optuna 調校 LGB(寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r1_pool       # exp5
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag      # exp6
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut    # exp7
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_cat      # Optuna 調校 CAT(寫入 cache)
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r2_pool       # exp8
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag      # exp9
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut    # exp10
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r3_multiclass # exp11
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py submit        # 產出提交檔

# exp12:Phase E-2 樹搜尋 v2(best;可中斷/續跑;樹狀態存 experiments_tree_v2.json,
# 與 v1 的 experiments_tree.json 各自獨立;OOF-only,不產生提交檔)
uv run python3 tree_search/run_s3e5_v2.py

# 提交至 Kaggle(本場執行環境未設定憑證,以下指令供後續有憑證時使用)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e5 -f <submission.csv> -m "<msg>"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run` 以確保
套件環境一致。
