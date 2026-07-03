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

## Appendix: Phase D-6 tree-search v2 sweep (2026-07-04) — the SCALE case, sweep finale

**Sweep question**: v2 was 4/4 BEAT (s3e3 eval#12, s3e7 eval#13, s3e1 eval#9, s3e19
eval#7). Does the harness hold at 360k rows, where a solo eval costs 30-55s (not the
2-20s of every prior comp) and the node budget must be blend-heavy? **Answer: BEAT —
first beat at evaluation #10, finished 0.295280 by evaluation #21 of 24 (−0.000368 vs
linear's 0.295648, ~5.5× the linear run's entire Phase-B R2→R5 residual gain), total
training+search wall 513.5s (8.6 min), a fraction of the 30-35 min budget.** The two
levers — a 4th tuned-CAT seed the linear run stopped short of, and pushing CatBoost
depth BEYOND Optuna's own search cap — are both direct extensions of s3e11's own
distilled lessons.

Built: `tree_search/eval_s3e11.py` (solo: LGB/CAT on log1p target, identical
KFold(5,shuffle,seed42), OOF cached in log1p space matching iterate2.py's own blend
semantics; blend: harness_v2.eval_blend minimizing rmsle_from_log) +
`tree_search/run_s3e11.py` (harness_v2 driver, two-phase). OOF cache:
`tree_search/cache_s3e11/` (gitignored). Banned per brief: per-combo feature means
after store_te (R4 dead end), XGB (R1 zero-weight dead end), auto_scale postprocess
(s3e19's lever, structurally inapplicable — KFold OOF has no systematic level gap).

### Scale management (the Phase D-6-specific lever)
- **Legacy OOF reuse instead of retraining**: root (CAT_tuned) + 4 pool seeds
  (LGB_ORIG/CAT_ORIG/CAT_S2024/CAT_S7) were loaded directly from the linear run's own
  `scripts/cache/*.npz` checkpoints at wall_s=0.0 each, with digit-for-digit
  verification against experiments.json inside `load_legacy_solo()` (max deviation
  4e-6, pure npz-roundtrip float noise; the reproduction check the brief mandates,
  done as load+recompute instead of a ~10-12 min retrain).
- Eval cost profile confirmed the brief's expectation: new CAT solos 34-54s, deep LGB
  8.5s, blends 17-39s (dirichlet over a 360k×n OOF matrix — blend cost now scales
  with n_rows and is no longer negligible, unlike every smaller comp).
- Node mix ended 12 solo / 12 blend — the blend-heaviest sweep run, as planned.

### Node/backtrack/dedup summary
- **24 evaluated nodes** (12 solo / 12 blend; 5 solos reused at zero cost), 0 failed,
  wall 513.5s total (phase 1: 425.5s / 20 nodes; phase 2: 88.0s / 4 nodes).
- **2 backtracks**, both genuine 3-strike plateaus: BLEND at node #16 (streak on
  0.295492), CAT_S7 at #21 (streak on 0.295280). tie_rate peaked at 0.059 — RMSLE is
  continuous, the adaptive-plateau branch never fired (as designed).
- **Dedup: 0 reactive rejections** — proposer-side pre-checks (find_dup before eval,
  order-insensitive blend-member hashing) silently skipped duplicate member-sets
  inside blend_fallback (e.g. re-adding #2 to #13's pool would recreate #12's set),
  leaving nothing for the safety net. Fifth consecutive comp with a clean dedup ledger.

### Best vs linear, evaluations-to-match/beat
| | OOF RMSLE | evaluations |
|---|---|---|
| Linear-iteration best (exp #8, 5-way 0.1-grid blend) | 0.295648 | 8 experiments |
| Tree v2: node #8 (5-way pool reproduction, dirichlet) | 0.295650 | 9 (match to 2e-6) |
| Tree v2: node #9 (6-way, +CAT_S99 4th seed) | 0.295619 | **10 (first beat)** |
| Tree v2: node #13 (lean 5-way, weakest removed) | 0.295492 | 14 |
| Tree v2: node #17 (SOLO: tuned CAT depth 10→12) | 0.295461 | 18 |
| Tree v2: node #20 (global best: 7-way around depth-12 family) | **0.295280** | **21** |

Winning chain: #8 pool reproduction 0.295650 → #9 +CAT_S99 0.295619 → #10 +DEEPLGB
0.295500 → #12/#13 remove zero-weight LGB/CAT_orig 0.295492 → (BLEND plateaus,
backtrack) → #17 depth-12 solo 0.295461 → (phase 2) #20 re-blend top-7 pool
**0.295280** (weights: depth-12 family .247+.319+.239=.805, DEEPLGB .178, everything
else ≤.01). The depth-12 seed family (seeds 7/3000/3001: 0.295461/0.295462/0.295483)
plus one deliberately-diverse deep LGB carries essentially the whole final blend.

### What actually moved it — and what didn't (honest ledger)
- **Depth 12 > Optuna's cap**: the single biggest discovery. The linear Optuna run
  searched depth 4-10 and chose 10 (its own search boundary); one manual push to
  depth 12 on 360k rows gave solo 0.295779→0.295461, beating every phase-1 blend.
  Extends experience.md's "large data wants MORE capacity" finding with a sharper
  corollary: **when Optuna's optimum sits ON a search-space boundary, the boundary
  itself is the next mutation** — the tuner never saw depth>10.
- **The blend around the new family, not the solo, set the final mark**: phase 1's
  BLEND lineage plateaued BEFORE #17 existed, so no phase-1 blend contained it; the
  phase-2 BLEND2 re-seed (pure weight search, 17.6s, zero retraining) took
  0.295461→0.295280. Sequencing lesson for the harness: a post-plateau solo
  breakthrough should re-open the blend lineage (v3 candidate rule).
- **4th tuned-CAT seed (CAT_S99)**: solo 0.295786 (mid-pool), but as a blend member it
  produced the first beat (#9) — the linear run's "diminishing returns, stop at 3
  seeds" call left real value on the table at this scale; seed-bagging pays as long
  as weight search can arbitrate.
- **DEEPLGB (num_leaves 255, light reg)**: mediocre solo (0.295833) yet held 0.156-0.377
  weight in every blend that contained it and drove #10's −0.000119 — fifth consecutive
  sweep comp confirming blend contribution ≠ solo score, and the capacity direction
  works for LGB here too (beats hand-set LGB_ORIG 0.296608 outright).
- **What lost**: FEATVARIANT feature-subtraction (drop children_away/cars_per_child:
  0.296184→0.297230, the worst solo of the run — at 360k rows even "weak" ratio
  features carry real signal; the s3e7/s3e14 trim-features prior did NOT transfer,
  opposite of s3e19 where it did); adding FEATVARIANT as a blend member (#11,
  0.295520) and re-adding removed members (#14/#15, #22/#23) were all washes or hairs
  worse; depth-12 seed variants beyond the third (#21 seed3002, 0.295562) regressed —
  the seed family saturates at ~3 members, consistent with the linear run's R5
  observation, just one capacity level higher.

### Prior-usage log (idea-injection experiment)
`suggest_priors({"metric":"rmsle","tags":["低訊號","optuna","ensemble"]})` returned 20
bullets (P0-P19); like s3e19, s3e11 got much of its OWN distillate back (P10-P12 are
s3e11 evidence). Among the 14 search-loop mutations (first-gen seeds excluded):
- **3 prior-informed, win rate 3/3 = 100%** (P8 add-4th-seed #9: W; P14 remove-weakest
  #13: W; P6 capacity-push depth-12 #17: W — the run's biggest single gain) — the
  highest informed win rate of the sweep alongside s3e1.
- **11 uninformed, win rate 2/11 = 18%** (#10 DEEPLGB-add and #12 fallback-remove won;
  all fallback seed-variations and re-adds lost).
- Sweep-final pattern across 5 comps (informed vs uninformed): s3e3 —, s3e7 62.5%,
  s3e1 100%, s3e19 33%, s3e11 100% vs consistently-lower uninformed rates. Priors
  reliably nominate WHAT to try; payoff size stays comp-local (here: P6's capacity
  direction paid 4× more than P8's seed direction); and the biggest lever again came
  from comp-local structure the library only partially encodes (Optuna's depth cap
  being the binding constraint). Priors set the floor; comp-local insight sets the
  ceiling — now confirmed on all 5 sweep comps.

### Sweep verdict (Phase D complete)
**5/5 BEAT**: s3e3 (AUC, small data), s3e7 (AUC), s3e1 (RMSE, geo), s3e19 (SMAPE,
TimeSeriesSplit), s3e11 (RMSLE, 360k rows). The v2 harness (ensemble-default node
space + OOF cache + priors + adaptive plateau + dedup) beat the linear-iteration
best on every metric family, CV scheme, and data scale tested, always within
9-21 evaluations. Note: 未提交 Kaggle(批次跑無憑證),tree best 為 OOF 分數。
