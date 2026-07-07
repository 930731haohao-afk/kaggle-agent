# 競賽分析報告:playground-series-s3e14

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:full | 產生日期:2026-07-06
> 本報告所有數字皆出自 facts.json,經 verify_report.py 驗證。

## 1. 競賽目的

**What**:依野生藍莓田間的 16 個環境與生物特徵(株叢大小 clonesize、四類授粉者密度、
六個高度共線的溫度區間欄位、降雨天數,以及結果率 fruitset、果重 fruitmass、種子數 seeds)
預測該田區的藍莓產量(`yield`),為連續數值輸出的迴歸問題。

**Why**:評估指標為 **MAE(minimize)**。產量為連續實數且分佈大致對稱,MAE 以原始單位
(產量)線性衡量平均絕對誤差、對離群值穩健,不像 RMSE 會被少數大誤差放大主導,適合
此類農業產量估計——關心的是「平均偏多少」,而非極端錯估的平方懲罰。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e14 |
| 問題型別 | regression |
| 評估指標 | mae(minimize) |
| 目標欄位 | yield |
| 素材等級 | full |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基模型,構成 exp1–7 blend 主體與 exp8 樹搜尋 34 成員 solo pool 的核心 |
| Optuna | exp4 對 LGB 做 fold-0 proxy 超參搜尋(36 trials);調參結果於 exp5 起以「加入 pool」方式使用 |
| 自建樹搜尋 harness(v3) | Phase F-2 搜尋模型/超參/集成組合空間,於 node #44 找到 exp8 的 explore-burst mega-blend |
| snap-to-grid 指標特化後處理 | 將 blend 預測貼齊 train 既有 `yield` 值,exp2 起每輪穩定小幅改善 |
| 5-fold CV 框架(scikit-learn) | KFold(shuffle,seed 42)5 折交叉驗證 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

本場工具鏈組合邏輯:Claude Code(LLM)負責決策——由 EDA 推出果實生物學交互特徵、以
重要度探針刪噪音欄位、否決有損多樣性的調參替換與失敗的 isotonic 校準、決定何時停損;
Auto-ML 工具(Optuna、樹搜尋 harness)負責系統化執行超參搜尋與 blend 組合空間探索。

## 3. 流程(how):五大元件

### 3.1 資料規格

| 項目 | 值 |
|------|-----|
| train 列數 | 15,289 |
| test 列數 | 10,194 |
| 原始欄位數 | 16 |
| 特別規則摘要 | 禁外部資料/禁預訓練模型/禁網路存取;每日提交上限 5 次 |

16 個特徵全為數值型、無類別欄位;train/test 皆無缺失值。目標 `yield` 大致對稱
(skew -0.291195,mean 6025.193999、median 6117.4759),非整數值、非 log 轉換候選;
15,289 列中僅 776 個相異值,但此離散性來自低基數環境變數組合,而非粗粒度取整格線——
「貼齊 train 既有 yield 值」因此成為值得實測的後處理假設(見 3.4 節)。

訊號結構明確:`fruitset`/`seeds`/`fruitmass` 為主導預測區塊(pearson 0.885967/
0.868853/0.826481),`AverageRainingDays`/`RainingDays`/`clonesize` 為負相關次要區塊
(pearson -0.48387/-0.477191/-0.382619);六個溫度區間欄位彼此近乎完全共線(高共線
特徵對 pairwise r 最高 0.999974)且單獨與目標幾乎無關。train/test 各欄平均值偏移皆
小於 1%(最大為 RainingDays 之 -0.822601%),無 covariate shift 疑慮。

> **誠實但書**:facts.eda 記錄 train 重複列 14、目標 skew -0.291195;STATUS.md 敘述
> 則為「7 exact duplicate rows」與「skew ≈ -0.16」,兩者口徑不同。本報告以 facts.eda
> 之數字為準。

### 3.2 實驗總表

> **語意澄清(本報告全文適用)**:本場自 exp2 起每輪皆有 snap-to-grid 後處理,故
> 「原始 OOF」為 blend 的 raw 分數、「決策分數」為貼齊 train yield 格線後的分數,
> 以下分別以「(原始)」「(貼格)」標記,不再重複解釋;exp1 無後處理,兩欄同值。
> 本場全部決策皆以本機 OOF 為準(未提交 Kaggle,見 3.4 節)。

| exp | 階段 | 模型/成員 | 特徵數 | 原始 OOF | 決策分數 | 採納 |
|-----|------|-----------|--------|----------|----------|------|
| 1 | Baseline(通用批次) | LGB/XGB/CAT(0.3/0.3/0.4) | 16 | 341.40782 | 341.40782 | 基線參照 |
| 2 | Phase A iter1 | LGB/XGB/CAT(0.5/0.25/0.25) | 27 | 341.02061 | 340.95856 | 是 |
| 3 | Phase A iter2(反思精簡) | LGB/XGB/CAT-native(0.45/0.25/0.3) | 21 | 340.75961 | 340.711795 | 是,Phase A 最佳 |
| 4 | Phase B R1 | LGB(Optuna,替換原 LGB)/XGB/CAT(0.5/0.3/0.2) | 21 | 340.95316 | 340.826945 | 否,多樣性受損 |
| 5 | Phase B R2 | 原 LGB + 調參 LGB 4-way(0.35/0.2/0.2/0.25) | 21 | 340.69924 | 340.627017 | 是 |
| 6 | Phase B R3 | 4-way + isotonic 校準(遭否決,回退 exp5) | 無紀錄 | 340.69924 | 340.627017 | 否,isotonic 346.99717 大幅劣化 |
| 7 | Phase B R4(線性終點) | +LGB(seed 2024)5-way(0.2/0.15/0.2/0.25/0.2) | 21 | 340.65207 | 340.59891 | 是,線性迭代最佳 |
| 8 | Phase F-2 樹搜尋(best) | 34 成員 mega-blend(node #44,28 個權重 >0.005) | 無紀錄 | 340.57853 | **340.35572** | 是,本場最佳 |

**best 成員表(exp8,34 成員 kitchen-sink mega-blend;harness v3 節點未逐一保存
成員權重,僅下列項目有紀錄)**

| 成員 | 權重 | solo 分數 | 備註 |
|------|------|-----------|------|
| LGB_ROOT | 無紀錄 | 342.02154 | 搜尋根節點 = exp3 之 21 特徵 LGB(同模型紀錄) |
| LGBTUNED_S4000 | 無紀錄 | 341.63373 | 調參 LGB 之 seed 變體,pool 中最佳 solo |
| CAT 家族(4 變體) | 0.22(合計) | 343.60196–344.30153 | solo 偏弱但合計權重可觀 |
| EXPL_CATDEEP | 0.051 | 345.1579 | explore-burst 長射成員,solo 最弱之一仍存活 |
| EXPL_REGDEEP | 0.045 | 341.99788 | explore-burst 長射成員 |
| EXPL_DART | 無紀錄 | 6156.32115 | 災難級 solo(DART+MAE 失控),權重搜尋自行處置 |

選型脈絡:exp4 以 Optuna 調參 LGB **替換**原 LGB,單模 342.02154→341.68775 變強,
blend 卻由 340.75961 退至 340.95316——印證 s3e7 的多樣性教訓;exp5 改為「加入而非
替換」立即創新低。exp6 之 isotonic 校準單獨評分 346.99717,自動否決回退;exp7 加
seed-2024 LGB 再小步改善後,依協定(四輪上限、增益遞減)停止線性迭代。

exp8 由 harness v3 的強制 explore-burst 機制對整個 34 成員 solo pool 做 k=800
Dirichlet + coordinate-ascent 權重搜尋(貼格計分內建於 metric_fn)而得;與 s3e7
(29/38 成員歸零)相反,本場 28/34 成員保留權重 >0.005——在較噪的目標上,「廣度
平均」本身即是訊號,再次確認 blend 貢獻與 solo 分數脫鉤。

> **誠實但書**:facts.best(exp8)以 OOF 分數最小選出,為 OOF-only 樹搜尋結果——未
> 產生 test 預測、無 submission 檔、未提交 Kaggle;其 34 維權重直接對全 OOF 擬合
> (無巢狀驗證),第 5 位小數的增益帶有 OOF 權重過擬風險,方向性結論(burst
> mega-blend 優於手工成長 blend)較為穩健。目前可直接提交的最佳檔案仍是 exp7 的
> submission。

### 3.3 訓練規格表

| exp | CV 方案 | folds | seed |
|-----|---------|-------|------|
| 1 | 5fold | 5 | 無紀錄 |
| 2 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 3 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 4 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 5 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 6 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 7 | 5fold_kfold(KFold,shuffle) | 5 | 42 |
| 8 | 5fold_kfold(KFold,shuffle) | 5 | 42 |

目標為連續 i.i.d. 值、無群組/時間結構、train/test 無分佈偏移(見 3.1 節),標準
KFold 即為正解(facts.eda 之 validation_hint 亦同);exp2 起 CV 全程固定同一組折,
8 個實驗分數可直接比較,樹搜尋(exp8)之 solo 種子並經逐位驗證重現線性各輪分數。

Objective 一律為 MAE。有紀錄之關鍵超參僅 exp4 的 Optuna 調參 LGB(fold-0 proxy,
36 trials):

```
LGB(Optuna):learning_rate≈0.0154, num_leaves=63, max_depth=5, min_child_samples=30,
             subsample≈0.790, colsample_bytree≈0.684, reg_alpha≈0.0126, reg_lambda≈0.0035
(來源:exp4 notes 之 best_params;XGB/CAT 沿用 iter2 手設配置,params 無紀錄)
```

exp8(樹搜尋)的 base_models 僅附 solo 分數,未附 params 欄位——無紀錄,不臆測。

### 3.4 推論表

| exp | 後處理 | submission 檔 | 已提交 |
|-----|--------|---------------|--------|
| 1 | 無後處理紀錄 | sub_generic_341.40782_20260703_121146.csv | 否 |
| 2 | snap_to_grid(勝過 none/clip 兩候選) | sub_blend_340.95856_20260703_193646.csv | 否 |
| 3 | snap_to_grid;另 grid_simplex 勝過 ridge_stack(344.04375) | sub_blend_v2_340.71180_20260703_193959.csv | 否 |
| 4 | snap_to_grid | sub_blend_v3_340.82694_20260703_210727.csv | 否 |
| 5 | snap_to_grid | sub_blend_v4_340.62702_20260703_211127.csv | 否 |
| 6 | isotonic=rejected + snap_to_grid | 無紀錄(檔案事後移除:與 exp5 檔 byte-identical) | 否 |
| 7 | snap_to_grid | sub_blend_v6_340.59891_20260703_211503.csv | 否 |
| 8 | snap_to_grid(於權重搜尋 metric_fn 內對每組候選權重計分) | 無紀錄(OOF-only,未產生 test 預測) | 否 |

snap-to-grid(貼格)為本場唯一被採納的後處理:將預測貼齊 train 既有 `yield` 值,
exp2 起每輪穩定貢獻小幅改善(如 exp7 之 340.65207→340.59891)。exp8 的差異在時點:
線性各輪於 blend 完成後才貼格,exp8 於權重搜尋內部即以貼格後分數為目標。欄位格式:
id 欄 `id`、目標欄 `yield`;本場全程未提交 Kaggle(leaderboard 列於 facts.missing)。

### 3.5 評估指標 / 排行榜

指標定義:MAE = 預測值與真實產量之絕對差的平均,單位與 `yield` 相同,越低越好。

| 項目 | OOF MAE |
|------|---------|
| LGB(exp7 成員,原始配置) | 342.02154 |
| LGB_TUNED(exp7 成員,Optuna 調參) | 341.68775 |
| XGB(exp7 成員,手設) | 342.21778 |
| CAT(exp7 成員,native-categorical) | 343.60196 |
| LGB_SEED2024(exp7 成員,seed 變體) | 342.04208 |
| Ensemble(exp7,線性迭代終點,貼格) | 340.59891 |
| Ensemble(exp8,樹搜尋 best,貼格) | **340.35572** |
| Public / Private LB | 無紀錄(未提交) |

本場無排行榜表(`leaderboard` 為 null、missing 含 "leaderboard"),CV↔LB gap 無法
計算。exp8 相對 exp7 之改善:

```
exp8 − exp7:340.59891 − 340.35572 = 0.24319(MAE,越低越好)
```

## 4. 實驗軌跡

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase F-2 | 樹搜尋執行(harness v3) | tier4 |

| exp | 時間 | 決策分數 | 階段 | 一句話摘要 |
|-----|------|----------|------|------------|
| 1 | 2026-07-03T12:11:46 | 341.40782 | Baseline(通用批次) | 16 原始特徵三模型固定權重 blend,設定待超越基線 |
| 2 | 2026-07-03T19:36:46 | 340.958565 | Phase A iter1 | 27 特徵(果實交互等)+ snap-to-grid,全場最大單筆躍升 |
| 3 | 2026-07-03T19:39:59 | 340.711795 | Phase A iter2 | 刪 6 個零重要度特徵 + CAT native-cat,Ridge stack 落敗 |
| 4 | 2026-07-03T21:07:27 | 340.826945 | Phase B R1 | Optuna 調參 LGB 替換原 LGB,多樣性受損反而退步 |
| 5 | 2026-07-03T21:11:27 | 340.627017 | Phase B R2 | 改為調參 LGB 加入 pool 成 4-way,創新低 |
| 6 | 2026-07-03T21:12:45 | 340.627017 | Phase B R3 | isotonic 校準 346.99717 大幅劣化,自動否決回退 |
| 7 | 2026-07-03T21:15:03 | 340.59891 | Phase B R4 | 加 seed-2024 LGB 成 5-way,線性迭代終點 |
| 8 | 2026-07-04T12:17:48 | **340.35572** | Phase F-2 樹搜尋 | harness v3 explore-burst 34 成員 mega-blend,本場最佳 |

- **突破點 1(exp1→exp2)**:果實生物學交互特徵(fruitset×seeds 等)+ OOF 權重搜尋
  + snap-to-grid,341.40782→340.958565,為全場最大單筆增益。
- **突破點 2(exp4→exp5)**:調參 LGB「替換」使 blend 退步、「加入」則創新低
  (340.826945→340.627017)——調參變體應入 pool 而非換人,為本場最具遷移價值的教訓。
- **突破點 3(exp7→exp8)**:harness v3 exploit 階段磨到 340.45150 後,強制
  explore-burst 的 34 成員 mega-blend 一舉收於 340.35572,burst 佔 post-exploit
  增益的 100%。

無法解析之紀錄:無(facts.unparsed 為空陣列)。

## 5. 效能對照:四層消融

| 層級 | 配置 | 分數 | 相對改善 |
|------|------|------|----------|
| tier1 | 基線(Claude Code 直接執行,未引入 skill;exp1 通用批次三模型 blend) | 341.40782 | —(基線) |
| tier2 | + kaggle-agent skill 六階段流程(exp3,Phase A iter2 精簡特徵 blend) | 340.711795 | 見下方算式 |
| tier3 | + self-improvement 線性迭代(exp7,Phase B 四輪之終點) | 340.59891 | 見下方算式 |
| tier4 | + 樹搜尋(exp8,node #44 mega-blend) | **340.35572** | 見下方算式 |

```
MAE 為 minimize 指標,分數下降即改善:
tier1→tier2: 341.40782 − 340.711795 = 0.696025,相對改善 0.696025 / 341.40782 = 0.2039%
tier2→tier3: 340.711795 − 340.59891 = 0.112885,相對改善 0.112885 / 340.711795 = 0.0331%
tier3→tier4: 340.59891 − 340.35572 = 0.24319,相對改善 0.24319 / 340.59891 = 0.0714%
```

本場由導入報告功能後之版本執行,分數自 tier2 起未低於前一層——符合計畫書目標三(效能不退步)。

四層皆為同一 5-fold KFold(seed 42)下的 OOF MAE、口徑一致、無平手;惟 tier4(exp8)
為 OOF-only 樹搜尋結果,未產生 test 預測亦無排行榜對照(本場全程未提交 Kaggle)。

### 子刻度分解(選填)

依 §5 附錄子刻度分類法(僅作分類字典,不強制逐級測量),下表僅列本場 experiments.json
天然存在對應中間紀錄之子刻度;無對應紀錄之子刻度不列,分數逐字引用自 facts.json。

| 子刻度 | 優化辦法 | 分數 | 出處(exp# 或 tree node) |
|--------|----------|------|--------------------------|
| 1.a | 單模預設參數(tier1 三模型中最佳者:XGB) | 344.13027 | exp1 |
| 1.b | 多模+OOF 權重搜尋 blend(tier1 三模型混合) | 341.40782 | exp1 |
| 2.b | iter1:27 特徵(果實生物學交互、pollinator index 等) | 340.958565 | exp2 |
| 2.b | iter2:剪至 21 特徵(移除 6 個近零重要度/高共線特徵) | 340.711795 | exp3 |
| 3.b | Optuna fold-proxy 調參 LGB(替換原 LGB,多樣性受損) | 340.826945 | exp4 |
| 3.c | 入池不替換(調參 LGB 改為加入而非替換,4-way) | 340.627017 | exp5 |
| 3.d | seed bagging(+seed 2024 LGB,5-way,線性迭代終點) | 340.59891 | exp7 |
| 4.a | 樹搜尋 harness v1 原型(較早版本樹檔之最佳節點) | 340.52635 | exp8 notes 引用 |
| 4.e | 樹搜尋 harness v3(node #44,explore-burst 34 成員 mega-blend,本場最佳) | 340.35572 | exp8 |

## 6. 總結

本場資料乾淨且訊號結構清楚:無缺失值、全數值特徵,`fruitset`/`seeds`/`fruitmass`
三欄即主導訊號(pearson 最高 0.885967),六個溫度欄近乎完全共線而個別無用;目標
`yield` 僅 776 個相異值的準格線結構,催生了全場唯一被採納的後處理 snap-to-grid,
自 exp2 起每輪穩定生效。

關鍵決策有三:其一,exp3 以重要度探針刪除 6 個零貢獻特徵並讓 CatBoost 原生處理低基數
環境欄,奠定 21 特徵基準;其二,exp4 揭示「單模變強、blend 變差」後,exp5 改以加入
pool 的方式使用調參結果,一步創新低;其三,exp6 的 isotonic 校準與 exp3 的 Ridge
stack 兩次習得式 meta-model 皆敗給簡單 simplex 權重搜尋,依驗證結果誠實棄用。

各層增益來源分明:tier1→tier2 靠特徵工程 + 貼格後處理(341.40782→340.711795),
tier2→tier3 靠線性迭代四輪紀律化(Phase B;調參入 pool、seed bagging)推至 340.59891,
tier3→tier4 靠 harness v3 強制 explore-burst 的 34 成員 mega-blend 收於 340.35572,
且 28/34 成員保留實質權重——廣度平均本身即是這個噪目標上的訊號。

可信度方面須誠實:全場唯一錨點是同一固定 CV 的 OOF,未提交 Kaggle、無排行榜外部
驗證;exp8 權重直接對全 OOF 擬合、無巢狀驗證,末位小數增益帶有過擬風險,且其為
OOF-only 產物,目前可提交的最佳檔案仍是 exp7 的 submission。方向性結論(burst
mega-blend 優於手工成長 blend)在 s3e7 與本場重複出現,是穩健的部分。

**重現本實驗的最短路徑**:見第 7 節。

## 7. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# Stage 1:EDA
uv run python3 competitions/playground-series-s3e14/scripts/eda.py

# Phase A iter1:特徵工程(27 特徵)+ LGB/XGB/CAT blend + snap-to-grid → exp2
uv run python3 competitions/playground-series-s3e14/scripts/train.py

# Phase A iter2:特徵精簡(21 特徵)+ CAT native-cat + Ridge stack 對照 → exp3
uv run python3 competitions/playground-series-s3e14/scripts/train_v2.py

# Phase B R1:Optuna 調參 LGB(fold-0 proxy)替換原 LGB(未採納)→ exp4
uv run python3 competitions/playground-series-s3e14/scripts/train_v3.py

# Phase B R2:調參 LGB 加入 pool 成 4-way blend(寫出 data/oof_v4.npz)→ exp5
uv run python3 competitions/playground-series-s3e14/scripts/train_v4.py

# Phase B R3:isotonic 校準嘗試(自動否決;需 oof_v4.npz)→ exp6
uv run python3 competitions/playground-series-s3e14/scripts/train_v5.py

# Phase B R4:加 seed-2024 LGB 成 5-way blend(線性迭代終點;需 oof_v4.npz)→ exp7
uv run python3 competitions/playground-series-s3e14/scripts/train_v6.py

# Phase F-2:樹搜尋 harness v3(exp8,本場最佳;可中斷續跑,樹狀態存 experiments_tree_v3.json)
uv run python3 tree_search/run_s3e14_v3.py

# 報告產生(本檔)
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e14
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
    competitions/playground-series-s3e14/REPORT.md competitions/playground-series-s3e14/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e14/REPORT.md \
    competitions/playground-series-s3e14/s3e14_REPORT.pdf

# 提交至 Kaggle(本場未執行;需先設定有效之 KAGGLE_API_TOKEN)
export KAGGLE_API_TOKEN=<your_token>
uv run kaggle competitions submit -c playground-series-s3e14 \
    -f competitions/playground-series-s3e14/submissions/sub_blend_v6_340.59891_20260703_211503.csv \
    -m "5-way blend (LGB + Optuna LGB + XGB + CAT + seed-2024 LGB) + snap-to-grid, OOF MAE 340.59891"
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;所有 Python 執行皆透過 `uv run`。
