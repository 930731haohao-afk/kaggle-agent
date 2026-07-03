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
- [x] Stage 5 Submission generated — `submissions/sub_r5_seedbag3_0.29565_20260703_223829.csv`
- [x] Phase B 自我改進迭代(#4–#8)— Optuna CatBoost + seed bagging,0.29614 → 0.29565
- [ ] Submitted to Kaggle leaderboard — 未提交(批次跑無憑證)

## Score trajectory (OOF RMSLE, minimize)

| # | Experiment | OOF RMSLE | Δ vs generic baseline |
|---|-----------|-----------|------------------------|
| 1 | generic batch baseline (15 raw feats) | 0.29723 | — |
| 2 | skill base: log1p target + tuned params + early stop (15 raw feats) | 0.29710 | −0.00013 |
| 3 | skill engineered: + amenity_count/ratios + store_combo K-fold target encoding (21 feats) | 0.29614 | −0.00109 |
| 4 | R1: drop XGB(兩輪權重皆 0),LGB+CAT pool | 0.296143 | −0.00109(與 #3 完全相同) |
| 5 | R2: Optuna fold-0 proxy 調參 CatBoost,加入 pool(不替換) | 0.295781 | −0.00145 |
| 6 | R3: + CAT_tuned seed=2024(seed bagging,4-way) | 0.295715 | −0.00152 |
| 7 | R4: + per-combo mean 特徵(sales_ratio/weight_per_case/gross_weight)| 0.296200 | 退步,棄用 |
| 8 | **R5: + CAT_tuned seed=7(seed bagging 第三 seed,5-way)** | **0.295648** | **−0.00158** |

Experiment log: `experiments.json`(#2–#8 via log_experiment_v2)。

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

## Model results (OOF RMSLE, 5-fold, Phase B final pool, run #8)
| Member | OOF RMSLE | time | 權重 |
|--------|-----------|------|------|
| LightGBM(原參數) | 0.29661 | 75s | 0.0 |
| CatBoost(原參數) | 0.29618 | 58s | 0.0 |
| CatBoost tuned(Optuna, seed=42) | 0.29579 | 31s | 0.4 |
| CatBoost tuned(seed=2024) | 0.29591 | 34s | 0.2 |
| CatBoost tuned(seed=7) | 0.29578 | 32s | 0.4 |
| **Blend(5-way weight search)** | **0.295648** | — | — |

Optuna(TPE, 40 trials, 319s, fold-0 proxy)最佳 CatBoost 參數:
depth=10, lr=0.0824, l2_leaf_reg=5.48, min_data_in_leaf=33, random_strength=0.092
(`scripts/cache/cat_best_params.json`)。調參後單 fold 迭代數 ~300(原參數 ~1300)→ 訓練反而更快。

## Key observations(Phase B)
- store_te 是 Phase A 主要增益來源;Phase B 主要增益來自 Optuna 調參 CatBoost(R2, −0.00036)。
- **Optuna fold-proxy → add-to-pool → seed bagging 配方在 360k 列上完全遷移**(第 4 個競賽驗證)。
- 調參後權重全數流向 tuned CAT 家族(LGB/CAT_orig 權重歸 0)——tuned CAT 單模(0.29579)已勝原 3-way blend(0.29614)。
- R4 反例:對已有 store_te 的 combo 再加 per-combo 特徵均值(非目標)反而全面退步
  (blend 0.295715 → 0.296200),與 s3e1「樹已能用原始欄位時粗粒度群組彙總是冗餘」一致。
- R1 驗證:刪掉兩輪權重 0 的 XGB,blend 分數 byte-identical(0.296143)——尊重 zero-weight 裁決無代價。
- 停止:R4 退步、R5 增益已縮到 −0.000067(噪音級),依 2–3 輪遞減即停原則收手。

## Files
```
competitions/playground-series-s3e11/
├── config.yaml
├── STATUS.md
├── experiments.json          (#1 generic, #2 base, #3 engineered, #4–#8 Phase B R1–R5)
├── facts.json / REPORT.md / REPORT.pdf
├── data/{train,test,sample_submission}.csv
├── scripts/{eda.py, features.py, train.py, iterate2.py}
│   └── cache/                (OOF/pred npz checkpoints + cat_best_params.json,gitignored)
└── submissions/
    ├── sub_generic_0.29723_20260703_121020.csv
    ├── sub_base_0.29710_20260703_192507.csv
    ├── sub_engineered_0.29614_20260703_192819.csv
    ├── sub_r1_noxgb_0.29614_20260703_222353.csv
    ├── sub_r2_cattuned_0.29578_20260703_223101.csv
    ├── sub_r3_seedbag_0.29571_20260703_223204.csv
    ├── sub_r4_deepstore_0.29620_20260703_223633.csv   (棄用)
    └── sub_r5_seedbag3_0.29565_20260703_223829.csv    <- best
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e11/scripts/eda.py
uv run python3 competitions/playground-series-s3e11/scripts/train.py base        # methodology check
uv run python3 competitions/playground-series-s3e11/scripts/train.py engineered  # Phase A best
# Phase B(checkpointed;cache 命中會跳過已訓練成員):
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r1
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py tune
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r2
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r3
uv run python3 competitions/playground-series-s3e11/scripts/iterate2.py r5      # best
# submit (use the huangweihaohuang token; kaggle.json 403s):
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')
uv run kaggle competitions submit -c playground-series-s3e11 -f <submission.csv> -m "<msg>"
```

## 附註:iterate.py(v1)棄用原因
單一長程序版 `iterate.py` 在背景執行時於 R1 CatBoost 階段停滯 33 分鐘後被 timeout 殺掉
(疑與 CatBoost 預設檔案日誌寫入受沙箱限制有關,未能穩定重現)。改寫為 `iterate2.py`:
每輪獨立程序 + npz checkpoint + `allow_writing_files=False` + 明確 thread_count,全部順利完成。
