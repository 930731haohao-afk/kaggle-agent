# STATUS — playground-series-s3e11 (Media Campaign Cost)

- **Task**: regression, predict `cost`; metric **RMSLE** (minimize); id col `id`
- **Data**: 360,336 train / 240,224 test; 15 raw features (all numeric, many low-cardinality "categorical-like"); no missing, no dups
- **Rules**: no external data, no pretrained, no internet; daily submission limit 5
- **Note**: 週末自主批次 (unattended) — 未提交 Kaggle,無 LB 分數

## Progress
- [x] Stage 0 Setup — config.yaml, data present (pre-existing)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (+ fold-safe target encoding in train.py)
- [x] Stage 3 Modeling + CV — `scripts/train.py {base|engineered}` (LGB/XGB/CatBoost on log1p target)
- [x] Stage 4 Evaluation — trajectory below; engineered features beat baseline
- [x] Stage 5 Submission generated — `submissions/sub_engineered_0.29614_20260703_192819.csv`
- [ ] Submitted to Kaggle leaderboard — 未提交(批次跑無憑證)

## Score trajectory (OOF RMSLE, minimize)

| # | Experiment | OOF RMSLE | Δ vs generic baseline |
|---|-----------|-----------|------------------------|
| 1 | generic batch baseline (15 raw feats) | 0.29723 | — |
| 2 | skill base: log1p target + tuned params + early stop (15 raw feats) | 0.29710 | −0.00013 |
| 3 | **skill engineered: + amenity_count/ratios + store_combo K-fold target encoding (21 feats)** | **0.29614** | **−0.00109** |

Experiment log: `experiments.json` (#1–#3, #2–#3 via log_experiment_v2).

## EDA key findings (scripts/eda.py)
1. Target `cost` ∈ [50.79, 149.75], skew 0.019 — 幾乎對稱,**非**右偏;log1p 後 skew −0.34。
   仍以 log1p(cost)+RMSE objective 訓練,因為 RMSLE = RMSE on log1p — 直接優化競賽指標;
   預測經 expm1 後 clip ≥ 0(安全網)。
2. 低訊號資料集:所有 15 個原始特徵與 cost 的單變量 |r| ≤ 0.11。
3. `store_sqft`(20 個值)+ 5 個 amenity 旗標(coffee_bar/video_store/salad_bar/prepared_food/florist)
   組成 ~111 種「store profile」,每種重複數千列;僅用 profile 組平均即得 in-sample R² ≈ 0.06,
   遠高於任何單一特徵 → 最強特徵候選(需 out-of-fold 編碼防洩漏)。
4. 無缺失、無重複列、train/test id 不重疊;train/test 各欄平均差 < 0.7% → 無 covariate shift。
5. 測試集有 19 個 train 未見過的 combo → target encoding 需 global-mean fallback(已實作)。

## CV scheme
- **5-fold KFold (shuffle, seed=42)** — 無時間/群組結構;store profile 重複數千次,隨機切分安全。
- Fold RMSLE 穩定(0.2947–0.2976)→ CV 可信。

## Feature engineering (21 features)
15 raw + `amenity_count`、`weight_per_case` (gross_weight/units_per_case)、
`sales_ratio` (store_sales/unit_sales)、`children_away`、`cars_per_child`、
**`store_te`**(store_combo 的 K-fold target encoding,fold 內 fit、test 端取 5 fold 平均)。

## Model results (OOF RMSLE, 5-fold, engineered run #3)
| Model | OOF RMSLE | time |
|-------|-----------|------|
| LightGBM | 0.29661 | 70s |
| XGBoost | 0.29713 | 25s |
| CatBoost | 0.29618 | 58s |
| **Blend 0.2/0.0/0.8 (LGB/XGB/CAT)** | **0.29614** | — |

CatBoost 最強且拿 0.8 權重;XGB 被 weight search 淘汰(權重 0)。

## Key observations
- store_te 是主要增益來源(#2→#3 improvement ~0.001,遠大於 #1→#2 的方法論差異)。
- 低訊號 playground 資料集,單次改善幅度小是預期行為;blend 權重兩次 run 都收斂到 LGB 0.2 / CAT 0.8。

## Potential improvements
- Optuna 調參 CatBoost(dominant model)。
- 更多 target encoding 變體:store_sqft 單獨、product 屬性 combo(gross_weight×units_per_case 群)。
- 把低基數 float 欄位轉 CatBoost 原生 categorical 處理。
- Stacking meta-model on OOF;seed averaging。

## Files
```
competitions/playground-series-s3e11/
├── config.yaml
├── STATUS.md
├── experiments.json          (#1 generic, #2 base, #3 engineered)
├── data/{train,test,sample_submission}.csv
├── scripts/{eda.py, features.py, train.py}
└── submissions/
    ├── sub_generic_0.29723_20260703_121020.csv
    ├── sub_base_0.29710_20260703_192507.csv
    └── sub_engineered_0.29614_20260703_192819.csv   <- best
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e11/scripts/eda.py
uv run python3 competitions/playground-series-s3e11/scripts/train.py base        # methodology check
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered  # best run
# submit (use the huangweihaohuang token; kaggle.json 403s):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e11 -f <submission.csv> -m "<msg>"
```
