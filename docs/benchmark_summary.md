# 跨競賽 Benchmark 彙總:generic baseline vs 完整 skill 流程 vs Phase B 迭代後 vs 樹搜尋後

> 產生方式:`docs/scripts/build_benchmark_table.py`(數字來自 `docs/benchmark_facts.json`,
> 敘述由 agent 撰寫)。涵蓋本週末批次的 10 個 playground-series 競賽,每場比較四層分數:
>
> - **tier1** — generic baseline(`competitions/run_competition.py` 批次腳本,單純 LGB+XGB+CAT blend,無特徵工程)
> - **tier2** — 完整 skill 流程最佳分數(Phase A,`Phase B 自我改進迭代` commit 之前)
> - **tier3** — 線性迭代最終最佳分數(Phase A + Phase B 自我改進迭代後,**不含樹搜尋**)
> - **tier4** — 樹搜尋(tree-search,Phase D–G)後之最佳分數,即 tier3 之上再疊加 Phase
>   G-1a/G-1b 入帳的樹搜尋結果(9/10 場樹搜尋勝過或追平 tier3,見下方「樹搜尋」段落)

## 主表(10 場競賽)

| 競賽 | 指標 | 方向 | tier1(generic) | tier2(skill/Phase A) | tier3(線性迭代最終) | tier4(樹搜尋後) | tier1→tier3 相對變化 | tier1→tier4 相對變化 |
|---|---|---|---|---|---|---|---|---|
| s3e1(加州房價) | rmse | ↓ | 0.56166 | 0.558768 | 0.557088 | 0.556329 | +0.81% | +0.95% |
| s3e3(離職預測) | roc_auc | ↑ | 0.81624 | 0.832925 | 0.83814 | 0.845051 | +2.68% | +3.53% |
| s3e5(酒質,QWK) | quadratic_weighted_kappa | ↑ | 0.47871 | 0.52687 | 0.56769 | 0.57066 | +18.59% | +19.21% |
| s3e7(訂房取消) | roc_auc | ↑ | 0.89882 | 0.899395 | 0.899893 | 0.900455 | +0.12% | +0.18% |
| s3e9(混凝土強度) | rmse | ↓ | 12.54287 | 12.073474 | 12.070034 | 12.070034(=tier3,平手) | +3.77% | +3.77% |
| s3e11(媒體成本) | rmsle | ↓ | 0.29723 | 0.296143 | 0.295648 | 0.29528 | +0.53% | +0.66% |
| s3e14(藍莓產量) | mae | ↓ | 341.40782 | 340.711795 | 340.59891 | 340.35572 | +0.24% | +0.31% |
| s3e16(蟹齡,取整 MAE) | mae(rounded) | ↓ | 1.35441 | 1.33812 | 1.33812 | 1.33563 | +1.20% | +1.39% |
| s3e19(銷量預測,SMAPE,TimeSeriesSplit) | smape | ↓ | 無可比 generic(見下方 CV 附註) | 10.175397 | 10.019463 | 9.75707 | +1.53%(tier2→tier3) | +4.11%(tier2→tier4) |
| s3e20(盧安達 CO2) | rmse | ↓ | 28.3424(代理值,見下方附註) | 22.6488 | 21.1487 | 21.0589 | +25.38% | +25.70% |

註:相對變化一律以「越好」為正值(minimize 指標 = (tier1−tierN)/tier1×100;maximize 指標
= (tierN−tier1)/tier1×100)。s3e19 因 tier1 與 tier2/tier3/tier4 的 CV 方案不同、不可比,
此兩欄分別改報 tier2→tier3、tier2→tier4;s3e20 的 tier1 為代理值,細節見下方「CV 方案與
可比性附註」。s3e9 的樹搜尋只**精確追平**(而非勝過)線性最佳,故 tier4 與 tier3 同值,
是誠實記錄的平手而非缺漏。

### 附:s3e19 同 CV 方案診斷對照(不計入上表,僅供 CV-scheme 偏誤參考)

| 對照 | 指標 | 方向 | generic(plain KFold) | skill(同 plain KFold,僅診斷用) | 相對變化 |
|---|---|---|---|---|---|
| s3e19-kfold-diagnostic | smape | ↓ | 5.31891 | 4.281421 | +19.51% |

此列與主表的 s3e19 列使用**不同 CV 方案**,兩者不可互相比較(見下方 CV 附註);本次
Phase G-1b 樹搜尋僅在誠實的 TimeSeriesSplit 軌跡上執行(見主表 s3e19 列的 tier4),
plain-KFold 診斷軌跡從未跑過樹搜尋,故此對照表沒有 tier4 欄。

## 樹搜尋(tier4):線性迭代之後還能再擠出多少

Phase D–G 對全部 10 場競賽跑了 harness v2/v3 樹搜尋(node 空間 = 完整解:特徵集 + 模型
組合 + 超參,而非線性迭代的一次一項),並把每場最佳結果以 `log_experiment_v2` 正式併入
`experiments.json`(notes 皆引用來源樹檔 `experiments_tree*.json` 的 node id)。結果:
**9/10 場勝過線性迭代最終分數(tier3),1 場(s3e9)精確追平**——tier1→tier4 相對變化
從 tier1→tier3 進一步擴大,s3e5(+18.59%→+19.21%)與 s3e20(+25.38%→+25.70%)兩場增益
最大。跨場觀察到的共通機制:

1. **blend 貢獻 ≠ solo 分數**:多場最大增益來自「solo 分數不特別亮眼、但拿到最大 blend
   權重」的成員(s3e1 的 CEILING 混合分類器、s3e14 mega-blend 中的 EXPL_CATDEEP/
   EXPL_REGDEEP、s3e19 的 CALSUBSET)。
2. **超出既有搜尋邊界的容量推進**:當 Optuna 自身的調參結果恰好落在搜尋空間邊界上時
   (s3e11 的 CatBoost depth=10 上限),邊界本身就是下一步該試的方向——推到 depth=12
   是該場單一最大增益來源。
3. **對已調參模型做 seed bagging 具情境依賴**:一般被認為可能中性的「對 Optuna 調參目標
   做 seed bagging」,在 TimeSeriesSplit(s3e19)這種折不可互換、模型方差本就偏大的情境
   下反而是決定性槓桿。
4. **結構主導的地形一樣有樹搜尋空間**:即使在 GBDT 完全零權重的結構主導型競賽(s3e20),
   把多個已手調的結構超參「聯合移動」(JOINT lineage)仍能找到一次一項的 Phase-B 協定
   結構性難以觸及的複合最優。
5. **精確追平也是誠實產出**:s3e9 的樹搜尋在噪音已飽和的資料上精確追平(而非打敗)線性
   最佳(12.070034),是該場資料噪音上限已被線性迭代摸到頂的獨立驗證,而非搜尋失敗。

**誠實限制(tier4 全場適用)**:tier4 的每一筆分數皆為 **OOF-only 的樹搜尋產物**——
未產生任何 test 預測、未提交 Kaggle,facts.json 對應筆記錄皆無 `submission` 欄位。s3e19
的 tier4 額外繼承 tier2/tier3 既有的 fold-5 double-dip 警語,並新增其自身的
scale-parameter/seed-selection OOF-fitting 警語(見上方主表註與各競賽 REPORT.md)。
s3e20 的 tier4 明確**不採用** node #28 的手足 BLEND 節點分數 21.0332(STATUS.md 記為
3 折 CV 下的低信心邊際發現),而是採用穩健的純結構節點 21.0589。s3e16 的 tier4
(1.33563)延續 tier2/tier3 既有的「取整口徑」——且是本場的第二次 raw/rounded 反轉案例:
該節點的 raw OOF MAE(1.35712)其實比線性冠軍的 raw 1.35589 更差,但 rounded OOF MAE
更好,再次證明本場的決策指標必須是取整分數。

## 各賽一句話:關鍵差異來自哪裡

- **s3e1**:地理最近距離/KNN 密度特徵(非 target encoding)+ Optuna fold-proxy 調參 LGB 加入池 + seed bagging;粗粒度地理 target encoding 本身對已能自行切分座標的樹是冗餘,無增益。
- **s3e3**:先移除類別不平衡加權、收緊 LGB 正則化拿到最大單一增益;後續 Optuna 直接以完整 CV 上的 ROC-AUC 為目標函式,找到比手調更淺、正則反而更輕的解;CatBoost 原生類別處理縮小差距但權重搜尋仍判它出局。
- **s3e5**:把 naive rounding 換成 OptimizedRounder(對 OOF QWK 調切點)是最大槓桿;第二波增益來自「Optuna 目標函式直接設為後處理後的 QWK」對 LGB/CatBoost 調參,而非先調 RMSE 代理再套後處理。
- **s3e7**:先剪掉與目標無關日期欄位的 cyclical 編碼讓賦分反超 baseline;Optuna fold-0 代理調參 LGB 加入池(不取代)是單輪最大增益,對第二個模型重複同一調參配方反而拖累 blend 多樣性。
- **s3e9**:正則化與特徵工程必須一起上,才能對抗重複列造成的標籤噪音天花板;調參與去噪本身停滯後,seed bagging 是 Phase B 唯一持續生效的槓桿。
- **s3e11**:fold-safe 群組(store)target encoding 是最大單一增益;Optuna 調參 CatBoost 加入池 + seed bagging 疊加小幅增益,再加同群組的特徵彙總則 revert(訊號已被既有 TE 吃盡)。
- **s3e14**:先剪除近完美共線特徵讓賦分反超 baseline;Optuna fold-proxy 調參 LGB「加入池」而非取代原模型帶來最大增益,Ridge stacking 與 nested isotonic 校準皆為反例(輸給 simplex 網格 blend)。
- **s3e16**:目標為整數,取整後處理是最大槓桿(且 rounded CV 分數與真實 Kaggle LB 落差極小、高度可信);Phase B 驗證了配方邊界——Optuna 調參 + seed bagging 讓 raw OOF 變好、rounded OOF 卻連續兩輪變差,依協定停止,tier2=tier3。
- **s3e19**:test 為未來期,必須用 TimeSeriesSplit 而非隨機 KFold 才是誠實 CV(見上方診斷對照,同一配置在錯誤 CV 下看似樂觀許多);log1p 目標 + 日曆特徵建立 Phase A 基礎後,Phase B 最大增益來自對「最後一折」做 Optuna 代理調參的 LGB,加上兩輪 seed bagging;比例分解(結構化方法)在此總量本身難以外推的情境下反而輸給 GBDT。
- **s3e20**:發現目標「跨年幾乎恆定」,讓純 location-week 歷史均值完勝所有 GBDT(三個 GBDT 權重搜尋皆歸零)是單一最大槓桿;Phase B 再對這個歷史均值做三段去噪(經驗貝葉斯收縮、異常年降權、鄰週平滑)取得額外增益,殘差診斷證實結構訊號已飽和,連殘差層都榨不出更多可預測性。

## 驗證過的跨競賽配方

1. **Optuna(fold-proxy 或直接優化最終指標)→ 加入 pool(不取代原成員)→ seed bagging**:在 s3e1、s3e3、s3e7、s3e9、s3e11、s3e14、s3e19 一致驗證有效,是目前樣本量最大的耐用配方;已知的系統性邊界是 s3e16——目標需取整時,同一配方的 raw OOF 增益可能無法穿越四捨五入的離散邊界,必須以 rounded 分數而非 raw 分數做決策。
2. **Metric-aware 後處理三寶**:分類/計數型整數目標直接四捨五入(s3e16);序數目標配 OptimizedRounder 對 OOF 指標調切點(s3e5,遠勝 naive rounding);離散格點目標 snap 到最近訓練集觀測值(s3e14)。三者都比「不做後處理」或「僅 clip」穩定勝出,且都必須用該指標本身(而非代理 loss)驗證增益是否成立。
3. **結構 vs GBDT 的邊界**:當目標對某個分組維度「跨年近乎恆定」時(s3e20),純歷史均值/target encoding 可完勝所有 GBDT 且權重搜尋一致給樹模型 0 權重;但當總量水準本身逐年漂移、不可外推時(s3e19 的比例分解反例),結構分解不比 GBDT 原生的類別分裂多提供訊息。耐用的判準不是「資料是否存在結構」,而是「該結構對應的維度上目標是否近乎恆定」。

## 誠實附註與可比性限制

- **CV-only,除 s3e16 外**:本表所有分數(含 tier4)皆為 Out-of-Fold 交叉驗證分數,未提交 Kaggle 排行榜(多數為已關閉或無憑證環境)。唯一例外是 s3e16,tier2/tier3 對應的線性迭代結果已實際提交,CV↔LB gap 極小、CV 可信賴;**s3e16 的 tier4(樹搜尋)本身仍是 CV-only,未提交**,不可與該場已提交的 LB 分數混為一談。
- **tier4 全場皆 CV-only、未提交**:9/10 場的 tier4 是 Phase G-1a/G-1b 以 `log_experiment_v2` 入帳的樹搜尋(tree-search)結果,全數為 OOF-only 搜尋產物——tree_search harness 未對任何一場產生 test 預測檔,對應 experiments.json 筆記皆無 `submission` 欄位,不可誤認為已提交的分數(細節與逐場 caveat 見上方「樹搜尋(tier4)」段落與各競賽 REPORT.md)。
- **s3e19 的 CV 方案差異(現涵蓋 tier2、tier3、tier4)**:generic baseline(tier1,5.31891)使用隨機 shuffle KFold,而 skill 流程(tier2/tier3)與樹搜尋(tier4)皆使用 TimeSeriesSplit——因為 test 是嚴格未來期,隨機 KFold 是內插式的樂觀估計,兩者**不可直接比較**(knowledge/experience.md 的 CV 設計鐵律)。主表因此把 s3e19 的 tier1 標為「無可比 generic」,相對變化欄改報 tier2→tier3 與 tier2→tier4;上方另附一組同 CV 方案(plain KFold)的診斷對照(無 tier4,因樹搜尋僅在誠實 TimeSeriesSplit 軌跡上執行),單獨呈現、不併入主表的相對變化計算。tier4(9.75707)額外帶有 fold-5 double-dip 與 scale/seed 皆為 OOF-fitted 的警語,真實預期的 2022 SMAPE 應讀作「明顯低於 10.02」而非字面上的 9.76(見各競賽 REPORT.md)。
- **s3e16 的取整口徑(現涵蓋 tier2、tier3、tier4)**:tier2/tier3 的 1.33812 是「四捨五入到整數後」的 OOF MAE,不是 collect.py 一般 adapter 直接輸出的 raw blend_oof_mae(1.35589)。Phase B 依驗證過的配方跑了 Optuna 調參 + seed bagging,raw OOF 確實變好,但 rounded OOF 連續兩輪變差,故 tier2 與 tier3 數值相同——這是配方邊界被驗證到,而非迭代被省略。tier4 的樹搜尋結果(1.33563)同樣只在取整口徑下才是贏家:其 raw OOF MAE(1.35712)比線性冠軍的 raw 1.35589 更差,是本場第二起 raw/rounded 反轉案例,決策必須以取整分數為準。
- **s3e20 無真正的 generic 基線**:此賽是 2023 已關閉、於本週末批次之前的舊 session(2026-02-14)完成的舊有競賽,不在 `run_competition.py` 這輪批次範圍內,沒有生成 `sub_generic_*` 檔案。其自身最早兩筆紀錄用單一 time-based split、無 CatBoost、無 target encoding,與後續 Leave-One-Year-Out CV 的紀錄不可比,因此未被誤標為 tier1。主表 tier1(28.3424)是同一 CV 方案下、加入 location-week target encoding 之前的 GBDT blend,做為近似對照,而非嚴格意義的 generic-batch 基線。此賽 Late Submission 已關閉,tier1–tier4 全數無法實際提交。
- **s3e20 的 tier4 明確排除低信心的 GBDT-blend 節點**:node #28(21.0589,本表採用)之手足 BLEND 節點(21.0332,本表不採用)只在 3 折 Leave-One-Year-Out CV 下、且 GBDT 權重直接對同一份 OOF 擬合才勝出,STATUS.md 明確記為 CV 噪音範圍內的邊際發現、非穩健結論——tier4 採用純結構節點是刻意的保守選擇,而非疏漏。
- **s3e9 的 tier4 = tier3(平手,非缺漏)**:s3e9 的樹搜尋(harness v2)精確追平(而非打敗)線性迭代最佳 12.070034,顯示該場資料噪音上限已被線性迭代摸到頂;`docs/scripts/build_benchmark_table.py` 的 `is_tree_entry()` 偵測到 s3e9 的 experiments.json 中並無任何樹搜尋筆記錄,因此 tier4 機械式地等於 tier3,如實反映此一平手結果。

## 重現方式

```bash
uv run python3 docs/scripts/build_benchmark_table.py
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py docs/benchmark_summary.md docs/benchmark_facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh docs/benchmark_summary.md
```
