# Playground Series S4E1 — Bank Churn Prediction

**Date**: 2026-02-19

## Competition Info
- URL: https://www.kaggle.com/competitions/playground-series-s4e1
- Problem: Binary classification (predict bank churn)
- Metric: ROC AUC (probability predictions)
- Train: 165,034 rows, 14 columns | Test: 110,023 rows
- Target: `Exited` (0/1, 21.2% positive)

## Key EDA Findings
- Clean tabular dataset — no missing values, no train-test shift
- Age is the strongest predictor (r=+0.34), churn peaks in 40s-50s then drops
- NumOfProducts: 3-4 products → ~88% churn, 2 products → 6%, 1 product → 35%
- Geography: Germany 37.9% churn vs France 16.5% vs Spain 17.2%
- Gender: Female 28.0% vs Male 15.9%
- IsActiveMember: inactive 29.7% vs active 12.5%
- Surname is predictive (McGregor 75.9% churn, Davey 0%) — target encoded
- Balance: 54.3% zero balance, non-zero → higher churn (27.1% vs 16.2%)
- No cross-correlations above 0.3 among features
- 54 duplicates in train (minimal)

## Feature Engineering
- 28 features total
- Binary/dummy encodings: is_female, geo_Germany, geo_Spain
- Age features: age_sq, age_decade, age_40_60 (peak churn indicator)
- Balance features: has_balance, balance_salary_ratio
- NumOfProducts features: products_gt2, products_eq1
- Interactions: age_x_products, age_x_balance, age_x_active, geo_gender, active_x_balance, active_x_products, credit_age_ratio
- Surname target encoding (OOF with smoothing=20)

## Experiment Results

| # | Model | CV AUC | Notes |
|---|-------|--------|-------|
| 1 | Majority Class | 0.50000 | Random baseline |
| 2 | LightGBM | 0.89653 | 28 features, 5-fold, surname target enc |

## Submissions
- `lgbm_baseline_20260219_150434.csv` → Public LB: **0.88716**, Private LB: **0.89179**

## Model Details
- **Best model**: LightGBM (binary, AUC metric, is_unbalance=True, lr=0.05, num_leaves=63, ~159 iterations)
- Validation: Stratified 5-Fold CV
- Top features: Age (dominant), NumOfProducts, surname_target_enc, active_x_products, age_sq

## Lessons Learned
- Age and NumOfProducts dominate — together they largely determine churn risk
- Surname target encoding is the 3rd most important feature — surnames carry geographic/cultural signal
- Short training (159 iterations avg) — model converges quickly on clean data
- CV AUC (0.897) closely matches Private LB (0.892) — reliable validation

## Potential Improvements
- XGBoost / CatBoost ensemble
- Hyperparameter tuning with Optuna
- More interaction features (e.g., Geography x Age, Balance x NumOfProducts)
- Feature selection
- Stacking with different model types

---

# 2026-07-07 跨季泛化四層重跑(過夜單元 1/5)

二月紀錄(上文)如實保留,作為前季參照。本次重跑目的:驗證 S3 十場蒸餾出的經驗庫配方
是否跨季(S4)成立,以四層消融(tier1 generic → tier2 skill → tier3 線性迭代 → tier4
樹搜尋 v3)全流程執行,全部四層共用同一 28 特徵集(features.py)與同一 canonical 5-fold
StratifiedKFold(seed=42)。

## 關鍵方法修正(對二月版)
- **surname 目標編碼折修正**:二月版 TE 用獨立 seed=99 折計 OOF,與模型訓練折(seed=42)
  不同——姓氏分組跨兩種折劃分共享,是「看似避免洩漏、實為另一種洩漏路徑」;本次改為
  fold-safe(TE 折 == 模型折)。同配方 solo LGB 由二月的 0.89653 降至 0.8937 量級,
  差距主要來源即此洩漏修正(詳 knowledge/experience.md AUC 節新條目)。

## 四層結果(全部 fold-safe 修正後、同折可比)
| tier | 配置 | OOF AUC |
|------|------|---------|
| tier1 | generic run_competition.py 3-model blend(raw 欄位 label-encode) | 0.8922 |
| tier2 | skill 六階段(28 特徵 + fold-safe surname TE)3-model blend | 0.894302 |
| tier3 | + 線性迭代 3 輪(is_unbalance 消融 / Optuna fold-0 50 trials / seed-bag)5-way blend | 0.894354 |
| tier4 | + 樹搜尋 harness_v3(60 節點滿預算,node #44 全池 37 成員 mega-blend) | **0.894393** |

## tier3 迭代摘要
- r1 is_unbalance 消融:大樣本上結論與 s3e3 同向(移除加權較優,Δ−0.000415),blend 不變 0.894302
- r2 Optuna(fold-0 proxy,50 trials,678s):LGB_tuned solo 0.894171(本場最強 solo),4-way blend 0.894354
- r3 seed-bag(seed=2024):solo 0.894137,5-way blend 0.894354(+0.000000,增益歸零)——依協定停止

## tier4 樹搜尋(v3 預算策略首次跨季實戰 + H-1 resume 契約首次真實復原)
- 前代 agent 被外部 kill 於播種中(5/8 個第一代 lineage 完成);resume 以
  `load_search_state()`(H-1 契約)載回 5 節點狀態,種子分數與 experiments.json 對應項
  逐位元一致(0.894171/0.894137 等),續播 BOUNDARYPUSH/FEATPRUNE/BLEND 後進主迴圈。
- **resume 實戰發現一個 driver 缺口**(非 harness 缺口):BLEND lineage 的 fallback 提案
  空間耗盡後 `propose_child` 永遠回 None,而 `select_next_parent` 只在 MAX_CHILDREN 滿時
  才 plateau——搜尋卡在同一 lineage 空轉直到 ITER_SAFETY_CAP。修正:driver 在提案耗盡時
  自行標 plateau(寫入 backtrack_log)後繼續。修正後跑滿 60 節點預算。
- 相位機:exploit 44 評估 → 全 lineage plateau → 強制 explore burst(mega-blend +
  DEEPLGB + DEEPCAT 長射,sanity gate 全 PASS)→ burst 內 mega-blend(node #44,37 成員,
  0.894393)即為全域最佳 → 60/60 hard cap 停止。0 失敗節點、0 dedup burn。
- 教訓與 S3 一致(07_tree_search.md §3 先驗再驗證):**plateau 後的增益來自強制
  kitchen-sink mega-blend,不來自手寫長射 solo**。

## 提交(Late Submission 關閉,CV-only)
- kaggle CLI 實測:`userHasEntered: False`、ListSubmissions 403、CreateSubmission 403
  (以 tier3 blend CSV 與 tier4 rebuild CSV 各實測一次)——本場 Late Submission 不開放,
  四層全部 CV-only,無 LB 錨點。CV 可信度依據:同資料族的二月版曾實提交
  (CV 0.89653 → Private 0.89179,gap ~0.005,方向與 S3 各場一致)。
- tier4 最佳 blend 已依 s3e16 前例走 OOF 逐位元重現閘門重建 test 預測
  (scripts/06_rebuild_tree_best.py → rebuild_tree_best_result.json)。

## Reproduce(2026-07 版)
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/run_competition.py playground-series-s4e1        # tier1
uv run python3 competitions/playground-series-s4e1/scripts/04_train_blend.py # tier2
uv run python3 competitions/playground-series-s4e1/scripts/05_iterate.py r1_unbalance_ablation
uv run python3 competitions/playground-series-s4e1/scripts/05_iterate.py r2_optuna_lgb
uv run python3 competitions/playground-series-s4e1/scripts/05_iterate.py r3_seed_bag     # tier3
uv run python3 tree_search/run_s4e1_v3.py                                     # tier4(可中斷續跑)
uv run python3 competitions/playground-series-s4e1/scripts/06_rebuild_tree_best.py  # 重建+閘門
```
