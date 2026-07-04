# STATUS — playground-series-s3e5 (Wine Quality, ordinal)

- **Task**: ordinal regression head, predict `quality` (3–8); metric **quadratic_weighted_kappa (QWK)**, maximize; id col `Id`
- **Data**: 2,056 train / 1,372 test; 11 raw physicochemical features, no missing values, no duplicates
- **Rules**: no external data, no pretrained models, no internet; daily submission limit 5 (not used — no credentials on this machine)

## Progress
- [x] Stage 0 Setup — config.yaml already present (generic-baseline entry #1 pre-existing)
- [x] Stage 1 EDA — `scripts/eda.py`
- [x] Stage 2 Feature engineering — `scripts/features.py` (11 → 21 features)
- [x] Stage 3 Modeling + CV — `scripts/train.py` (LGB/XGB/CatBoost regression, 5-fold StratifiedKFold)
- [x] Self-improvement iteration (first pass) — optimized-rounder threshold tuning vs naive rounding
- [x] **Phase B self-improvement iteration (this run)** — `scripts/iterate.py`, 4 real rounds + 2 nested-cutpoint diagnostics
- [x] Stage 5 Submission generated — `submissions/sub_iterate_0.56769_20260703_231704.csv` (new best)
- [ ] Submitted to Kaggle leaderboard — not attempted (no Kaggle credentials configured in this environment; local CV only)

## Current best score
| | OOF QWK (post-rounder) | Public LB | Private LB |
|-|---------|-----------|------------|
| Generic baseline blend (exp #1) | 0.47871 | — | — |
| Regression blend, naive round (exp #2) | 0.47191 | — | — |
| Regression blend, optimized rounder (exp #3, prior best) | 0.52687 | — | — |
| + finer Dirichlet weight search (exp #4) | 0.52986 | — | — |
| + Optuna-tuned LGB added to pool (exp #5) | 0.56293 | — | — |
| + seed-bag LGB_tuned (exp #6, no gain) | 0.56293 | — | — |
| + Optuna-tuned CAT added to pool (exp #8) | **0.56769** | — | — |
| + seed-bag CAT_tuned (exp #9, no gain) | 0.56716 | — | — |
| + LGB multiclass EV-decode member (exp #11, no gain) | 0.56769 | — | — |

**New best: 0.56769** (exp #8). **Delta vs prior best 0.52687: +0.04082** (≈7.7% relative). **Delta vs
original generic baseline 0.47871: +0.08898** (≈18.6% relative). Not submitted — no Kaggle
credentials available in this session, so no Public/Private LB numbers exist yet.

## Phase B iteration (this run) — protocol and outcome
Ran `competitions/playground-series-s3e5/scripts/iterate.py` step by step, deciding keep/reject on
the **final post-rounder QWK only** (per knowledge/experience.md s3e16 lesson — never decide on raw
member OOF). Same CV throughout: 5-fold StratifiedKFold(shuffle=True, random_state=42) on `quality`.

1. **Round 0 (methodology, no new model)**: exp#3 used a coarse 0.1-step grid over the 3-model
   simplex for the blend weight search. Replaced with Dirichlet random search (800 draws) +
   coordinate-ascent refinement, scored on post-rounder QWK. Same 3 members, same folds.
   **0.52687 → 0.52986** (+0.00299). Logged as exp #4.
2. **Round 1**: Optuna (TPE sampler, 40 trials, 600s timeout, full 5-fold CV — data is tiny so the
   full CV is affordable, no fold-0 proxy needed) tuning LightGBM with the objective set to the
   **post-rounder QWK computed directly on that trial's own OOF vector** (not raw RMSE) — this
   dodges the s3e16 "discretization boundary" trap by optimizing the actual final metric. Best
   single-model LGB_tuned post-rounder QWK: 0.56244 (vs orig LGB 0.50547). Added as a 4th pool
   member (kept, not replaced, per s3e14 lesson); weight search moved ~98% of blend weight onto
   LGB_tuned. **0.52986 → 0.56293** (+0.03307). Logged as exp #5.
3. **Round 2 (seed-bag)**: added a second random_state=2024 copy of LGB_tuned to the pool; weight
   search gave it **zero** weight. **No improvement** (0.56293 = 0.56293). Logged as exp #6
   (non-improving round #1).
4. **Diagnostic — nested cutpoints** (exp #7, not a model change): compared the current practice
   (OptimizedRounder cutpoints fit on the full 5-fold OOF vector) against honest nested
   (leave-fold-out) cutpoint fitting on the exp#5/6 5-way blend. Full-OOF QWK 0.56293 vs nested
   QWK 0.54649 (gap +0.01644) — modest, addresses the overfit-risk flag from the prior run's
   STATUS.md.
5. **Round 3**: same Optuna full-CV direct-post-rounder-QWK recipe applied to CatBoost (the
   strongest baseline member). Best single-model CAT_tuned post-rounder QWK: 0.56466. Added as a
   6th pool member (kept, not replaced); weight search moved ~95% weight onto CAT_tuned.
   **0.56293 → 0.56769** (+0.00476). Logged as exp #8 (**new overall best**).
6. **Round 4 (seed-bag)**: second random_state=2024 copy of CAT_tuned. **No improvement**
   (0.56716 < 0.56769). Logged as exp #9 (non-improving round #1 of the final pair).
7. **Diagnostic — nested cutpoints, re-run on the exp#8 6-way blend** (exp #10, not a model
   change): full-OOF QWK 0.56769 vs nested QWK 0.56393 (gap +0.00377) — **even tighter** than the
   5-way blend's gap (0.01644), i.e. the risk shrank as the pool matured. Fold-level cutpoints are
   very stable across folds — full-OOF cutpoint fitting is safe here.
8. **Round 5 (diverse ordinal head)**: LGB multiclass classifier (`class_weight="balanced"`,
   targeting the extreme-class 3/8 data starvation) with expected-value decode
   (`sum(class_i * P(class_i))`) added as a 7th pool member. Single-model post-rounder QWK 0.52040
   (worse than every tuned regression member); weight search gave it **zero** weight. **No
   improvement** (0.56769 = 0.56769). Logged as exp #11 (non-improving round #2 of the final pair).

**STOP reached**: 2 consecutive non-improving rounds (exp #9 seed-bag, exp #11 multiclass-head) per
protocol, on top of an already-substantial gain. Final champion: **exp #8, 6-way blend (LGB/XGB/CAT
originals + LGB_tuned + LGB_tuned_seed2024 + CAT_tuned), weights LGB 0.05 / CAT_tuned 0.95 (rest
≈0), OptimizedRounder(cutpoints=[3.574, 4.586, 5.609, 6.174, 7.628]), OOF QWK = 0.56769**.

## EDA key findings
1. Severe class imbalance: quality 3 = 0.6% (12 rows), quality 8 = 1.9% (39 rows); 5 and 6 dominate (79% combined). Imbalance ratio 69.9x.
2. `alcohol` (Spearman r=0.504) and `sulphates` (r=0.457) are by far the strongest single-feature predictors; `volatile acidity` (r=-0.248) and `total sulfur dioxide` (r=-0.227) next.
3. No missing values, no duplicate rows (feature-only or full), no leakage — `Id` is a plain row index.
4. Train/test feature distributions are near-identical (largest mean diff 2.1%, citric acid) — no covariate shift.
5. Mild collinearity: fixed acidity vs citric acid (r=0.696), vs pH (r=-0.674), vs density (r=0.616); free vs total SO2 (r=0.638). Kept both sides — tree models handle collinearity natively.
6. **Modeling-head decision** (justified in `scripts/eda.py` §7): regression, not multiclass classification. QWK penalizes squared distance between predicted/true class; a regressor's continuous output respects ordering and lets abundant mid-classes (5/6) inform placement of the data-starved extremes (3/8) — reconfirmed this run: the diverse multiclass-EV member (round 5 above) scored 0.52040, well below the tuned regression members, and got zero blend weight.

## CV scheme
- **5-fold StratifiedKFold on the raw `quality` label** (6 classes), shuffle, seed=42. Unchanged
  across all 11 experiments (comparability requirement for this iteration).

## Feature engineering (11 → 21 features, unchanged this run)
Domain-informed ratios/interactions on top of the raw columns: `free_so2_ratio`, `bound_so2` (bound vs free SO2); `fixed_to_volatile_acid`, `citric_to_volatile_acid`, `total_acid`, `acid_ph_ratio` (acidity structure); `alcohol_x_sulphates`, `alcohol_to_density`, `alcohol_x_density` (the two strongest raw predictors combined); `sugar_to_alcohol` (fermentation-completeness proxy).
No feature changes this iteration — all gains came from hyperparameter tuning (Optuna, direct
post-rounder-QWK objective) and finer blend weight search.

## Concerns
- Test-set predictions from the new best (exp #8) contain **no quality-3, 4, or 8 rows** — only
  5/6/7 (589/439/344 of 1372). This is narrower than the prior best's 4/5/6/7 spread. Expected
  given the 12/39-row training support for the true extremes and a global regression signal, but
  worth flagging more strongly than before: the tuned CatBoost member (95% of blend weight) may be
  smoothing predictions harder than the original blend. If the private test set has meaningful
  quality-4 support this could cost QWK there even though OOF says otherwise (OOF also has very
  little quality-4 support to detect this).
- Optimized-rounder cutpoints are fit on the full 5-fold OOF vector (standard practice). This run's
  nested-cutpoint diagnostic (exp #7, #10) shows the overfit risk is modest and shrinking
  (0.01644 → 0.00377 gap as the pool matured) — this addresses and downgrades the concern flagged
  in the prior STATUS.md.
- Not submitted to Kaggle (no credentials in this session) — CV-only result; no CV↔LB gap can be
  assessed yet.

## Potential improvements (not attempted, time-boxed out)
- True ordinal-regression loss (e.g. CORAL/CORN) — the simpler multiclass+EV-decode proxy tried
  this run did not help, but a proper ordinal loss is a different mechanism and untested.
- Class-weight tweaks specifically on the regression heads for extreme classes 3/8 (only tried on
  the multiclass diagnostic member, not on LGB/XGB/CAT regressors directly).
- A small stacking meta-model on OOF predictions (Ridge stacking has failed to beat simplex weight
  search in other competitions per knowledge/experience.md — expected low EV here too).

## Files
```
competitions/playground-series-s3e5/
├── config.yaml
├── STATUS.md
├── experiments.json          (11 experiments)
├── facts.json                (regenerated via kaggle-report collect.py)
├── REPORT.md / REPORT.pdf    (regenerated, facts.best = exp #8)
├── data/ (train.csv, test.csv, sample_submission.csv, train_processed.csv, test_processed.csv)
├── scripts/
│   ├── eda.py
│   ├── features.py
│   ├── train.py               (original 3-model + naive-vs-optimized-rounder pipeline)
│   ├── iterate.py             (Phase B: Optuna tuning, pool growth, seed-bag, nested-cutpoint diagnostics)
│   └── cache/*.npz            (per-member OOF/pred checkpoints, resumable)
└── submissions/
    ├── sub_generic_0.47871_20260703_120341.csv        (pre-existing baseline)
    ├── sub_blend_optround_0.52687_20260703_184829.csv (prior best)
    └── sub_iterate_0.56769_20260703_231704.csv         (NEW BEST, this run)
```

## Reproduce
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 competitions/playground-series-s3e5/scripts/eda.py
uv run python3 competitions/playground-series-s3e5/scripts/features.py
uv run python3 competitions/playground-series-s3e5/scripts/train.py
# Phase B iteration (see REPORT.md §8 for the full ordered command sequence):
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py base
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_lgb
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r1_pool
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py tune_cat
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r2_pool
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py seed_bag
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py nested_cut
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py r3_multiclass
uv run python3 competitions/playground-series-s3e5/scripts/iterate.py submit
```

## Appendix — 樹搜尋泛化驗證(Phase C-2c,2026-07-04)

第三個樹搜尋競賽、也是第一個**離散化指標**(QWK-after-OptimizedRounder)的泛化測試。
Harness 邏輯零改動(`tree_search/harness.py`);評估器 `tree_search/eval_s3e5.py`(solo +
blend 雙節點型,沿用 s3e14 模式),驅動 `tree_search/run_s3e5.py`,樹狀態
`experiments_tree.json`,OOF 快取 `tree_search/cache_s3e5/`(gitignored)。

**離散化指標的關鍵設計**:rounder 內建在評估器裡,不是節點型別——**每個**節點(solo 與
blend)的分數一律 = 該節點自身 OOF 上擬合 OptimizedRounder 後的最終 QWK;任何節點都不可能
以 raw 迴歸分數被比較(experience.md s3e16/s3e5 教訓)。QWK 為 maximize,故餵給 harness
的分數取 `-QWK`(harness 假設 lower-is-better)。CV 與線性迭代完全相同:5-fold
StratifiedKFold(quality, shuffle, seed=42);root 節點精確重現 iterate.py 的 tuned-LGB
0.56244、CAT seed 重現 tuned-CAT 0.56466,分數直接可比。

### 結果
| | 分數 (OOF QWK, post-rounder) |
|-|-|
| 線性迭代最佳(exp #8,6-way blend) | **0.56769** |
| 樹搜尋最佳(node #11,4-way blend) | 0.56766 |

**樹搜尋未擊敗線性迭代**,差距 0.00003(單一樣本的切點歸屬即可翻轉的量級)。最佳節點
#11 = blend(root tuned-LGB #0 + tuned-CAT #2 + XGB #3 + FEAT-17特徵-LGB #5),權重
0.129/0.645/0.018/0.208,cutpoints [3.590, 4.609, 5.614, 6.160, 7.597]——本質上以 4 名成員
重新發現了線性迭代 6-way champion 的同一個最適區(線性 champion 有效成員也只有 2 個:
CAT_tuned 0.95 + LGB 0.05)。

### 搜尋統計
- **40 個評估節點**(30 solo / 10 blend),0 失敗;評估總時 697s(~11.6 分),三段執行
  (26→34→40 節點,resume 機制驗證通過)。
- **11 次 backtrack 事件**:8 條 lineage 全部以「連續 3 子代未破全域最佳」正常 plateau
  (BLEND→CAT→LGBNUDGE→ROOTQ→FEAT→CATORIG→XGB→LGBORIG,嚴格按分數優先序輪替),之後
  **reopen-once 規則首次實戰觸發**——重開後 BLEND lineage 以 fallback 吃進新出爐的 solo
  池成員(#15),得 0.56725,仍未破 #11。
- solo 突變全數未破 root:tuned 參數已是 Optuna 直接 QWK 目標的產物,局部擾動
  (depth/lr/正則化/特徵刪減)最好僅追平(#15 = #2 = 0.56466)。**增益全部來自 blend
  組合**,與 s3e14 的結論一致。

### 離散化指標特有觀察(本次泛化測試的核心產出)
1. **rounder-inside-evaluator 運作正常**:root/seed 分數與線性迭代逐位吻合,決策永遠在
   最終 post-rounder QWK 上,無一節點洩漏 raw 分數。
2. **分數面呈階梯狀,平手極常見**:40 節點中出現 4 組完全同分(至小數 5 位)——
   0.56725×3、0.56664×2、0.56466×2、0.56065×2。連續指標(s3e9 RMSE、s3e14 MAE)幾乎不
   會同分;離散化讓「改進事件」更稀疏,plateau streak 更快觸發,8/8 lineage 全數 plateau
   是三次樹搜尋中最徹底的一次。對 QWK 類指標,PLATEAU_STREAK=3 實質上比連續指標更嚴格。
3. **blend 節點不再近乎免費**:每個候選權重都要跑一次 Nelder-Mead rounder 擬合,單一
   blend 節點 ~45s(s3e14 的 MAE blend 節點 <1s)。離散化指標下 blend/solo 成本比反轉
   (45s vs 1-5s),「blend 節點廉價所以多開」的 s3e14 經驗不能直接遷移。
4. **fallback 的一個小缺陷被離散面放大**:同一 parent 的 blend fallback 因 parent config
   不變而重複產出同一組成員(#37/#38/#39 同為 0.56725),在同分頻繁的離散面下浪費了
   plateau 額度——未來可讓 fallback 檢查兄弟節點已試過的 member 集合。

### 泛化判定
樹搜尋在離散化指標上**機制全部成立**(rounder 內建、sign 翻轉、plateau/backtrack/reopen、
resume),分數與線性迭代**統計上打平**(-0.00003)但未超越:線性迭代的 Optuna-direct-QWK
成員已把單模天花板抬到位,樹的 blend 組合只能重新發現同一最適區。連同 s3e9(未達)與
s3e14(擊敗),三綜資料點的模式:**樹搜尋的價值集中在 blend 組合空間,且在線性迭代已充分
開採該空間時只能追平**。

### 重現
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 tree_search/run_s3e5.py   # 可中斷/續跑;樹狀態存 experiments_tree.json
```

## Appendix — 樹搜尋 v2(harness_v2,Phase E-2,自適應 plateau 首測,2026-07-04)

v1 在本賽以 0.56766 對線性迭代 0.56769「統計上打平」(切點噪音量級差距);harness_v2 的
metric-aware 自適應 plateau(§6.2)正是讀著這個 v1 教訓設計的,本次是它的第一場實戰。
評估器 `tree_search/eval_s3e5_v2.py`、驅動 `tree_search/run_s3e5_v2.py`、樹狀態
`experiments_tree_v2.json`(v1 的 `experiments_tree.json` 未動);v1 的
`cache_s3e5/*.npz` 唯讀重用(root + 6 個 solo seed 逐位驗證後直接載入,|diff|<4e-6,
零重訓),v2 自身快取寫入 `cache_s3e5/v2/` 子目錄。CV/rounder 紀律與 v1 完全相同:
5-fold StratifiedKFold(seed=42),所有節點分數一律為 post-rounder QWK。

### 結果:三方對照
| | 分數 (OOF QWK, post-rounder) |
|-|-|
| 線性迭代最佳(exp #8,6-way blend) | 0.56769 |
| 樹搜尋 v1 最佳(node #11,4-way blend) | 0.56766 |
| **樹搜尋 v2 最佳(node #17,3-way blend)** | **0.57066** |

**v2 首次擊敗線性迭代(+0.00297)與 v1(+0.00300)**——差距約為 v1↔線性 gap
(0.00003)的 100 倍,不再是切點噪音量級。最佳節點 #17 = blend(root tuned-LGB #0 +
tuned-CAT #1 + **LGBBOUND #7**),權重 0.151/0.551/0.297,由 #16(4-way, 0.57027)
remove-weakest(LGBNUDGE 權重僅 0.0117)而來。

### 兩個真正起作用的槓桿(誠實歸因)
1. **邊界推進成員(s3e11 槓桿,PRIOR P18)**:先檢查線性跑的 Optuna 搜索空間
   (scripts/iterate.py tune_lgb),發現 tuned-LGB 有 **3 個超參正好落在搜索盒邊界上**
   (max_depth=3 = 下界 [3,10]、min_child_samples=40 = 上界 [3,40]、reg_lambda≈1e-3 ≈
   下界)。LGBBOUND lineage 推過邊界(max_depth 3→2),solo 僅 0.55784(比 root 低
   0.005),但作為 blend 成員拿到 0.297 權重、貢獻 +0.0026(#13 0.56766 → #16
   0.57027)——「solo 較弱但異質」的教科書級 blend 價值。對照組 CATBOUND(CAT tuned
   參數無任何邊界飽和)如預期沒有可比增益(solo 0.56378 < 0.56466,加入 blend 減分)。
2. **權重搜尋預算實測修正(本次最重要的工程決定)**:任務簡報建議「粗化權重網格省
   45s/blend」;實測 k=200 粗搜在 v1 最佳的同一組 4 成員 OOF 上只找到 0.56601(v1
   0.56766,同資料、純搜尋品質差距;coord-ascent 局部精修救不回,因粗搜根本沒採樣到
   最優盆地附近)。k 掃描(200/500/800/1500 + coord-ascent):0.56601 / 0.56766 /
   **0.56874** / 0.56792。故本 run 用 k=800(≈v1 自身預算)+ coord-ascent,~60s/blend
   節點——**在這個離散指標上,省 blend 搜尋預算會直接把「勝」變回「平」**,而 30 分鐘
   預算根本用不完(全程 793.7s),粗化毫無必要。此教訓與 harness 新機制無關,純屬
   評估器預算配置。

### 搜尋統計
- **22 個評估節點**(11 solo / 11 blend,其中 7 個 solo 從 v1 快取零成本重用),
  1 個 failed(#18,見下),總 wall 793.7s(~13.2 分,預算 28 分)。
- **1 次 backtrack**(vs v1 的 11 次):BLEND lineage 在 #17 之後連續 3 子代未破。
  v1 的 8 條 lineage 全數 plateau + reopen-once;v2 因 blend 一路有進展、且 22 節點
  上限先到,搜尋壓根沒進入「多 lineage 輪替耗竭」階段。
- **1 次 dedup 拒絕**:fold_avg toggle 失敗(timeout)後 fallback 想重提同一 config,
  被 rec #3 的 config-hash dedup 攔下——v1 的 #37/#38/#39 重複子代 bug 在同型情境下
  被正確阻止,並促使 lineage 改試下一個 mutation。
- **Prior 使用率**:informed 9 節點、勝率 33.3%(3 勝:#13/#16/#17,即全部三次
  global-best 刷新都是 prior-tagged 突變,P18/P19);uninformed 2 節點、勝率 0%。
  每個突變的 prior 出處都以 `[PRIOR Pk]` 記錄在 mutation 字串裡(P0-P19 全文存於樹
  JSON 的 `priors` 欄)。

### 自適應 plateau 行為報告(本次首測的核心問題)
**沒有觸發——整場 tie_rate 恆為 0.000**(`tie_rate_log` 11 筆全 0),自適應模式
(ADAPTIVE_PLATEAU_STREAK=5、平手中性計分)從未啟動,全程走 v1 相同的 streak=3 規則。
為什麼為它量身打造的比賽反而沒讓它上場?三個原因,依重要性:
1. **v1 的 4 組同分中最大一組(0.56725×3)是 duplicate-children bug 的產物**——rec #3
   的 dedup 修掉了病因,同分的主要來源隨之消失。機制間有依賴:**dedup 修好後,
   tie-neutrality 的觸發條件在同一情境下反而變得罕見**。
2. k=800 + coord-ascent 的權重搜尋讓每個 blend 節點落點更精細,5 位小數完全同分的
   機率大幅下降(v1 粗搜下不同成員集常收斂到同一個切點格局)。
3. 本 run blend 一路在進步(0.5676→0.56766→0.57027→0.57066),streak 很少累積,
   plateau 邏輯本身就少被觸碰。
結論:**自適應 plateau 在其目標情境的首測中呈「休眠」狀態,既未幫忙也未礙事**;
v2 的勝利歸因於 ensemble-default 節點空間(11/22 節點是 blend、最佳節點是 blend)+
邊界推進 prior + 足額權重搜尋預算,而非 §6.2 機制。它的真正測試要等一場「dedup 修復
後仍高頻同分」的比賽。

### 其他觀察
- **fold_avg rounder(任務簡報的 per-fold-averaged cutpoints 探針)兩種形態都確認
  無益**:solo 形態(#9)0.54766,遠低於同 config 的 full_oof 0.56244(與 STATUS.md
  exp #10 手工診斷方向一致);blend 形態(#18)直接 **timeout>120s**——fold_avg 在
  權重搜尋內意味著每個候選權重要做 5 次 Nelder-Mead 擬合(~5 倍成本),在「rounder
  必須進 metric_fn」的紀律下結構性不可行。此路線可安心關閉。
- 風險註記:0.57066 由更徹底的權重搜尋直接優化 full-OOF post-rounder QWK 而得,
  權重+切點對 OOF 的過擬風險比 v1 略高(搜尋越徹底、對 OOF 噪音的擬合越徹底)。
  本次未重跑 nested-cutpoint 診斷;按 exp #7/#10 的模式,建議在採用此 blend 產生
  submission 前補跑一次 nested 驗證。

### 泛化判定(Phase E-2)
四綜資料點更新:s3e9 v1 未達→v2 逆轉(E-1)、s3e14 勝、s3e5 v1 平→**v2 勝(本次)**。
「樹搜尋的價值集中在 blend 組合空間」的結論加強為:**當 blend 組合空間裡有線性迭代
沒開採過的異質成員來源(這裡是 Optuna 盒邊界外的超參區域)時,樹搜尋能超越而非只是
追平線性迭代——前提是 blend 權重搜尋預算不縮水**。

### 重現
```bash
cd /home/tjyen/ai_agents/kaggle
uv run python3 tree_search/run_s3e5_v2.py   # 可中斷/續跑;樹狀態存 experiments_tree_v2.json
```
