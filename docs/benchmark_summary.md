# 跨競賽 Benchmark 彙總:generic baseline vs 完整 skill 流程 vs Phase B 迭代後

> 產生方式:`docs/scripts/build_benchmark_table.py`(數字來自 `docs/benchmark_facts.json`,
> 敘述由 agent 撰寫)。涵蓋本週末批次的 10 個 playground-series 競賽,每場比較三層分數:
>
> - **tier1** — generic baseline(`competitions/run_competition.py` 批次腳本,單純 LGB+XGB+CAT blend,無特徵工程)
> - **tier2** — 完整 skill 流程最佳分數(Phase A,`Phase B 自我改進迭代` commit 之前)
> - **tier3** — 最終最佳分數(Phase A + Phase B 自我改進迭代後)

## 主表(10 場競賽)

| 競賽 | 指標 | 方向 | tier1(generic) | tier2(skill/Phase A) | tier3(最終) | tier1→tier3 相對變化 |
|---|---|---|---|---|---|---|
| s3e1(加州房價) | rmse | ↓ | 0.56166 | 0.558768 | 0.557088 | +0.81% |
| s3e3(離職預測) | roc_auc | ↑ | 0.81624 | 0.832925 | 0.83814 | +2.68% |
| s3e5(酒質,QWK) | quadratic_weighted_kappa | ↑ | 0.47871 | 0.52687 | 0.56769 | +18.59% |
| s3e7(訂房取消) | roc_auc | ↑ | 0.89882 | 0.899395 | 0.899893 | +0.12% |
| s3e9(混凝土強度) | rmse | ↓ | 12.54287 | 12.073474 | 12.070034 | +3.77% |
| s3e11(媒體成本) | rmsle | ↓ | 0.29723 | 0.296143 | 0.295648 | +0.53% |
| s3e14(藍莓產量) | mae | ↓ | 341.40782 | 340.711795 | 340.59891 | +0.24% |
| s3e16(蟹齡,取整 MAE) | mae(rounded) | ↓ | 1.35441 | 1.33812 | 1.33812 | +1.20% |
| s3e19(銷量預測,SMAPE,TimeSeriesSplit) | smape | ↓ | 無可比 generic(見下方 CV 附註) | 10.175397 | 10.019463 | +1.53%(tier2→tier3) |
| s3e20(盧安達 CO2) | rmse | ↓ | 28.3424(代理值,見下方附註) | 22.6488 | 21.1487 | +25.38% |

註:tier1→tier3 相對變化一律以「越好」為正值(minimize 指標 = (tier1−tier3)/tier1×100;
maximize 指標 = (tier3−tier1)/tier1×100)。s3e19 因 tier1 與 tier2/tier3 的 CV 方案不同、不可比,
此欄改報 tier2→tier3;s3e20 的 tier1 為代理值,細節見下方「CV 方案與可比性附註」。

### 附:s3e19 同 CV 方案診斷對照(不計入上表,僅供 CV-scheme 偏誤參考)

| 對照 | 指標 | 方向 | generic(plain KFold) | skill(同 plain KFold,僅診斷用) | 相對變化 |
|---|---|---|---|---|---|
| s3e19-kfold-diagnostic | smape | ↓ | 5.31891 | 4.281421 | +19.51% |

此列與主表的 s3e19 列使用**不同 CV 方案**,兩者不可互相比較(見下方 CV 附註)。

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

- **CV-only,除 s3e16 外**:本表所有分數皆為 Out-of-Fold 交叉驗證分數,未提交 Kaggle 排行榜(多數為已關閉或無憑證環境)。唯一例外是 s3e16,已實際提交,CV↔LB gap 極小、CV 可信賴。
- **s3e19 的 CV 方案差異**:generic baseline(tier1,5.31891)使用隨機 shuffle KFold,而 skill 流程(tier2/tier3)使用 TimeSeriesSplit——因為 test 是嚴格未來期,隨機 KFold 是內插式的樂觀估計,兩者**不可直接比較**(knowledge/experience.md 的 CV 設計鐵律)。主表因此把 s3e19 的 tier1 標為「無可比 generic」,相對變化改報 tier2→tier3;上方另附一組同 CV 方案(plain KFold)的診斷對照,單獨呈現、不併入主表的相對變化計算。
- **s3e16 的取整口徑**:tier2/tier3 的 1.33812 是「四捨五入到整數後」的 OOF MAE,不是 collect.py 一般 adapter 直接輸出的 raw blend_oof_mae(1.35589)。Phase B 依驗證過的配方跑了 Optuna 調參 + seed bagging,raw OOF 確實變好,但 rounded OOF 連續兩輪變差,故 tier2 與 tier3 數值相同——這是配方邊界被驗證到,而非迭代被省略。
- **s3e20 無真正的 generic 基線**:此賽是 2023 已關閉、於本週末批次之前的舊 session(2026-02-14)完成的舊有競賽,不在 `run_competition.py` 這輪批次範圍內,沒有生成 `sub_generic_*` 檔案。其自身最早兩筆紀錄用單一 time-based split、無 CatBoost、無 target encoding,與後續 Leave-One-Year-Out CV 的紀錄不可比,因此未被誤標為 tier1。主表 tier1(28.3424)是同一 CV 方案下、加入 location-week target encoding 之前的 GBDT blend,做為近似對照,而非嚴格意義的 generic-batch 基線。

## 重現方式

```bash
uv run python3 docs/scripts/build_benchmark_table.py
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py docs/benchmark_summary.md docs/benchmark_facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh docs/benchmark_summary.md
```
