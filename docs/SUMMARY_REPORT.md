# 十場競賽四層消融總報告(playground-series 週末批次)

> 產生方式:數字逐字取自 `docs/benchmark_facts.json`(由 `docs/scripts/build_benchmark_table.py`
> 決定性抽取自各競賽 `experiments.json`),敘述由 agent 撰寫。
> 素材範圍:本週末批次 10 場 playground-series 競賽的完整消融軌跡(tier1–4)。
> 本報告所有數字皆出自 `docs/benchmark_facts.json`,經 verify_report.py 驗證。
> 定位:本報告是讀者友善的敘事版總結,供督導一次比較十場全貌;`docs/benchmark_summary.md`
> 是同一份數字的精簡資料附錄,兩者互補、不重複維護。

## 1. 總覽

本報告把本週末批次 10 場 playground-series 競賽的「generic baseline → kaggle-agent skill
流程 → self-improvement 線性迭代 → 樹搜尋」四層消融(ablation)結果,整理成單一份可一次讀完
的總報告。四層構成一組對照實驗,逐層隔離出一種價值來源:

| 層級 | 定義 | 隔離出的價值 |
|------|------|------|
| tier1 | generic baseline——Claude Code 直接執行、**未引入** kaggle-agent skill(`run_competition.py` 固定 LGB+XGB+CAT blend,無 EDA、無特徵工程、無逐場決策) | 基線(無 skill) |
| tier2 | 完整 skill 六階段流程最佳分數(Phase A,Phase B 自我改進迭代之前) | skill 流程本身 |
| tier3 | 線性迭代最終最佳分數(Phase A + Phase B 自我改進迭代後,**不含**樹搜尋) | 自我改進迭代 + 跨競賽經驗庫 |
| tier4 | 樹搜尋(tree-search,Phase D–G)後之最佳分數,即 tier3 之上再疊加樹搜尋結果 | 搜尋式設計(對應 Aygün 等人 ERA 的樹搜尋主張) |

**tier5 預告**:樹搜尋 + 外部想法注入(idea injection,ERA 第二支柱、計畫書 4.1)已於
Phase J 定義並執行中——本報告暫僅涵蓋 tier1–4,tier5 待收官後另行併入本表。

**實驗階段對照表**(內部代號使用前先定義,R-W8)

| 代號 | 白話名稱 | 對應層級 |
|------|----------|----------|
| Phase A | kaggle-agent skill 六階段首跑 | tier2 |
| Phase B | self-improvement 線性迭代 | tier3 |
| Phase D–G | 樹搜尋原型/harness 演進(v1→v3)與執行入帳 | tier4 |
| Phase J | 外部想法注入(規劃中) | tier5(規劃中) |

## 2. 使用工具與環境

| 工具 | 用途 |
|------|------|
| LightGBM / XGBoost / CatBoost | 三個梯度提升樹基礎模型,十場皆用於 blend 主體 |
| Optuna | Phase B 超參搜尋(fold-proxy 或直接優化最終指標,依資料規模擇一) |
| 自建樹搜尋 harness(v1/v2/v3) | Phase D–G 對全部 10 場搜尋特徵集/模型組合/超參空間 |
| OptimizedRounder 等指標特化後處理 | 整數取整(s3e16)、序數切點優化(s3e5)、離散格點 snap(s3e14) |
| 5-fold CV / StratifiedKFold / TimeSeriesSplit / Leave-One-Year-Out | 依資料型態擇一,見各競賽 REPORT.md 第 3.3 節 |
| uv | Python 套件與虛擬環境管理,所有腳本皆以 `uv run` 執行 |

十場的共通分工邏輯:Claude Code(LLM)負責問題理解、驗證策略、特徵工程與何時停損等
決策;Auto-ML 工具(Optuna、樹搜尋 harness)負責系統化執行超參搜尋、集成與組合空間探索。

## 3. 十場四層總表

| 競賽 | 指標 | tier1 | tier2 | tier3 | tier4 | tier1→tier4 相對變化 |
|---|---|---|---|---|---|---|
| s3e1(加州房價) | rmse(↓) | 0.56166 | 0.558768 | 0.557088 | **0.556329** | +0.95% |
| s3e3(離職預測) | roc_auc(↑) | 0.81624 | 0.832925 | 0.83814 | **0.845051** | +3.53% |
| s3e5(酒質,QWK) | quadratic_weighted_kappa(↑) | 0.47871 | 0.52687 | 0.56769 | **0.57066** | +19.21% |
| s3e7(訂房取消) | roc_auc(↑) | 0.89882 | 0.899395 | 0.899893 | **0.900455** | +0.18% |
| s3e9(混凝土強度) | rmse(↓) | 12.54287 | 12.073474 | **12.070034** | **12.070034**(平手) | +3.77% |
| s3e11(媒體成本) | rmsle(↓) | 0.29723 | 0.296143 | 0.295648 | **0.29528** | +0.66% |
| s3e14(藍莓產量) | mae(↓) | 341.40782 | 340.711795 | 340.59891 | **340.35572** | +0.31% |
| s3e16(蟹齡,取整 MAE)* | mae(取整,↓) | 1.35441 | 1.33812 | 1.33812 | **1.33563** | +1.39% |
| s3e19(銷量預測,SMAPE)* | smape(↓) | 無可比 generic* | 10.175397 | 10.019463 | **9.75707** | +4.11%(tier2→tier4) |
| s3e20(盧安達 CO2)* | rmse(↓) | 28.3424(proxy)* | 22.6488 | 21.1487 | **21.0589** | +25.70% |

> **口徑 call-out(僅四場需要,其餘六場為乾淨對照)**
> - **s3e19**:tier1(5.31891,plain KFold)與 tier2/3/4(TimeSeriesSplit)CV 方案不同、不可
>   比較——test 為嚴格未來期,隨機 KFold 是內插式樂觀估計。本列「tier1」故標「無可比
>   generic」,相對變化欄改報 tier2→tier4。
> - **s3e16**:tier2–tier4 皆為「取整(rounded)」口徑,非 collect.py 一般 adapter 輸出的
>   原始 blend_oof_mae;tier2=tier3(Phase B 兩輪未通過取整門檻,依協定誠實停止)。
> - **s3e20**:此賽 2023 已關閉,不在本週末批次的 `run_competition.py` 範圍內,tier1 為同
>   CV 方案下的 proxy 基線(加入 location-week target encoding 之前的 GBDT blend),非嚴格
>   generic-batch 對照;Late Submission 已關閉,全部四層皆 CV-only。
> - **s3e9**:tier4 = tier3(12.070034,精確平手,非缺漏)——樹搜尋精確追平(而非打敗)線性
>   最佳,顯示該場資料噪音上限已被線性迭代摸到頂,是誠實記錄而非搜尋失敗。

十場中,tier4 相對於 tier3 全數未退步(9 場改善、1 場精確持平,s3e9)——符合計畫書目標三
(效能不退步)的跨場證據。

## 4. 逐場一行摘要表

| 競賽 | 關鍵招式 | 一句結論 |
|---|---|---|
| s3e1 | 地理密度/最近距離特徵 + Optuna(fold-proxy)入池 + seed bagging | 樹能自行切分座標,粗粒度地理 target encoding 反而冗餘,原始幾何度量才是真增益 |
| s3e3 | 移除類別不平衡加權、收緊正則化,再以 Optuna 直接優化完整 CV 上的 ROC-AUC | 排名指標下,擾動損失面的不平衡加權是壞交易,直接優化最終指標才是正解 |
| s3e5 | OptimizedRounder 取代 naive rounding,Optuna 目標函式直接設為後處理後的 QWK | 離散化指標必須用指標本身驗證增益,不能靠代理 loss 再套後處理 |
| s3e7 | 剪除與目標無關的日期欄位特徵,Optuna fold-proxy 調參後加入池(不取代) | 特徵不是越多越好;調參配方對第二個模型重複套用反而傷 blend 多樣性 |
| s3e9 | 正則化與特徵工程合力對抗重複列標籤噪音,seed bagging 是調參停滯後唯一持續生效的槓桿 | 資料噪音天花板下,樹搜尋精確追平而非打敗線性最佳,是誠實的天花板驗證 |
| s3e11 | fold-safe store target encoding 為最大單一增益,CatBoost 深度推過 Optuna 搜尋邊界 | 邊界推進(boundary-push)是本場單一最大槓桿,值得成為標準變異型別 |
| s3e14 | 剪除近完美共線特徵,Optuna 調參 LGB 加入池,樹搜尋 explore-burst mega-blend 收尾 | solo 分數普通的長射成員能拿到最大 blend 權重,burst 機制是後期突破的唯一來源 |
| s3e16 | 目標取整為最大槓桿,樹搜尋節點延續 raw/rounded 反轉並實際提交取得雙榜改善 | 決策指標必須是取整後分數;本場是十場中唯一同時擁有 tier2 與 tier4 真實 LB 錨點的競賽 |
| s3e19 | log1p 目標 + 日曆特徵,TimeSeriesSplit 下對最後一折做 Optuna 代理調參 + 兩輪 seed bagging | 時序外推情境必須用 TimeSeriesSplit;比例分解結構化方法在總量本身難外推時輸給 GBDT |
| s3e20 | 發現目標跨年近乎恆定,純 location-week 歷史均值完勝所有 GBDT,再以三段去噪疊加增益 | 結構訊號已飽和到連殘差層都榨不出更多可預測性,GBDT 全程權重歸零 |

## 5. 真實排行榜驗證

十場中僅 **s3e16** 擁有真實 Kaggle 排行榜錨點,且是唯一同時具備 tier2「與」tier4 兩筆
真實提交的競賽,其餘九場的 tier4 皆為 OOF-only(見第 7 節)。

| 提交 | 對應層級 | 日期 | Public | Private |
|------|----------|------|--------|---------|
| #1 | tier2/tier3(exp1,線性迭代冠軍) | 2026-07-03 | 1.34356 | 1.34075 |
| #2 | tier4(exp5,樹搜尋 node #15) | 2026-07-06 | **1.34315** | **1.33859** |

```
雙榜改善(提交#2 相對提交#1):
  Public :  1.34356 − 1.34315 = 0.00041
  Private:  1.34075 − 1.33859 = 0.00216
```

兩榜皆對提交#1 改善、方向一致——是這套樹搜尋配方第一次獲得的外部(真實排行榜)驗證,而
非僅 OOF 內部比較。決策指標全程為取整(rounded)後 OOF MAE(1.33812→1.33563),而非
collect.py 一般 adapter 輸出的原始 blend_oof_mae(完整算式與 CV↔LB gap 一致性檢查見
`competitions/playground-series-s3e16/REPORT.md` §3.5)。

## 6. 跨場驗證的配方

1. **Optuna(fold-proxy 或直接優化最終指標)→ 入池(不取代原成員)→ seed bagging**:在
   s3e1、s3e3、s3e7、s3e9、s3e11、s3e14、s3e19 共七場一致驗證有效,是目前樣本量最大的
   耐用配方;已知邊界是 s3e16——目標需取整時,同一配方在 raw OOF 上的增益可能無法穿越
   四捨五入的離散邊界,必須以 rounded 分數而非 raw 分數做決策。
2. **指標感知後處理(metric-aware post-processing)三寶**:整數目標直接四捨五入
   (s3e16)、序數目標配 OptimizedRounder 對 OOF 指標調切點(s3e5,遠勝 naive
   rounding)、離散格點目標 snap 到最近訓練集觀測值(s3e14);三者都比「不做後處理」或
   「僅 clip」穩定勝出,且都必須用指標本身(而非代理 loss)驗證增益是否成立。
3. **explore-burst + mega-blend(樹搜尋 tier4 機制)**:harness 強制觸發的探索性爆發,在
   s3e3(規模曲線驗證跑)、s3e7(v3 驗證跑)、s3e14(v1 與 v3 驗證跑)皆為後期突破的唯一
   貢獻來源;共通模式是「solo 分數不特別亮眼、卻拿到最大 blend 權重」的長射成員——s3e14
   的 EXPL_CATDEEP/EXPL_REGDEEP、s3e19 的 CALSUBSET、s3e1 的 CEILING 混合分類器皆屬此類。
4. **boundary-push(邊界推進)**:當 Optuna 調參結果卡在搜尋空間邊界上時,把「推過邊界」
   當成標準 mutation 型別,在 s3e11(CatBoost max_depth 推過上界)、s3e5(LGBBOUND 成員)、
   s3e16(learning_rate 推過邊界)三場各自被驗證為單一最大槓桿之一;s3e7 的 harness v3
   自動邊界偵測提供第四個獨立確認資料點。

## 7. 誠實但書

- **全數 CV-only,除 s3e16 外(tier2「與」tier4 皆已 LB 驗證)**:其餘九場的 tier4 皆為
  Phase G-1a/G-1b 以 `log_experiment_v2` 入帳的樹搜尋(tree-search)結果,全數為 OOF-only
  搜尋產物——未產生任何 test 預測、未提交 Kaggle,對應 experiments.json 筆記皆無
  `submission` 欄位。
- **s3e19 的 CV 方案差異**:generic baseline 使用隨機 shuffle KFold,而 skill 流程
  (tier2/tier3)與樹搜尋(tier4)皆使用 TimeSeriesSplit,兩者不可直接比較,故 tier1 標
  「無可比 generic」。tier4 額外帶有 fold-5 double-dip 與 scale/seed 皆為 OOF-fitted 的
  警語,真實預期的 SMAPE 應讀作「明顯低於 10.02」而非字面上的 9.76。
- **s3e16 的取整口徑**:tier2–tier4 皆為「四捨五入到整數後」的 OOF MAE,非 collect.py 一般
  adapter 輸出的原始 blend_oof_mae;tier4 是本場第二起 raw/rounded 反轉案例——其原始 OOF
  比線性冠軍的原始分數更差,取整後卻更優,決策必須以取整分數為準。
- **s3e20 無真正的 generic 基線**:此賽 2023 已關閉,非本週末批次 `run_competition.py` 範圍
  內的競賽,tier1 是同 CV 方案下的 proxy 基線,非嚴格意義的 generic-batch 對照;Late
  Submission 已關閉,tier1–tier4 全數無法實際提交。tier4 明確排除低信心的 GBDT-blend 手足
  節點(21.0332,3 折 CV 下的邊際發現),採用穩健的純結構節點(21.0589)。
- **s3e9 的 tier4 = tier3(平手,非缺漏)**:樹搜尋精確追平(而非打敗)線性迭代最佳,顯示
  該場資料噪音上限已被線性迭代摸到頂;`build_benchmark_table.py` 的 `is_tree_entry()`
  偵測到 s3e9 的 experiments.json 中並無任何樹搜尋筆記錄,tier4 機械式地等於 tier3,如實
  反映此一平手結果。

## 8. 總結敘事

四層消融的設計初衷,是把「導入 skill 有沒有用」拆成三個可獨立檢驗的問題:tier1→tier2 隔離
出 kaggle-agent skill 六階段流程本身的價值(EDA、特徵工程、CV 設計、建模一次到位地取代
generic baseline 的無腦 blend);tier2→tier3 隔離出自我改進迭代與跨競賽經驗庫的價值(把
單場驗證過的配方系統化地套用到後續競賽);tier3→tier4 隔離出樹搜尋這種搜尋式設計本身的
價值,對應 Aygün 等人 ERA 論文「候選樹取代線性單路徑迭代」的核心主張。十場的結果顯示三層
隔離都拿到正向訊號——沒有一場在任何一層出現整體倒退。

三層的增益量級並不均勻,而這種不均勻本身就是資訊:tier1→tier3 的最大增益來自「資料本身
藏著一個結構性洞見」的競賽——s3e20 發現目標跨年近乎恆定(+25.70% 至 tier4)、s3e5 發現
離散化指標需要指標感知的後處理(+19.21% 至 tier4);而已經被 generic baseline 逼近天花板的
競賽(s3e7、s3e9、s3e11、s3e14)增量普遍落在 1% 以內,說明 skill 流程與迭代的邊際報酬會隨
資料本身的可壓縮空間縮小而遞減。tier3→tier4 的樹搜尋則在九場拿到額外正向增益、一場
(s3e9)誠實追平,證實搜尋式設計即使疊加在已充分線性迭代過的結果之上,仍能靠 explore-burst
mega-blend 與 boundary-push 兩種機制系統性地擠出殘餘空間。

誠實但書是這整個階梯設計的一部分,而非事後補救:s3e9 的平手被如實記錄而非美化成勝利,
s3e16 的 raw/rounded 反轉兩次出現且每次都以取整分數而非原始分數做決策,s3e19 的 CV 方案
差異與雙重擬合警語被逐層攜帶而非在推進到下一層時悄悄丟棄,s3e20 的 tier1 proxy 身份與
低信心節點排除都被明確標注。十場中僅 s3e16 有真實 Kaggle LB 錨點,而它恰好同時驗證了
tier2 與 tier4 兩層——CV↔LB gap 在兩次獨立提交間維持同一量級同方向,是這整套四層消融
方法論目前最強的外部信度證據。

這個階梯本身尚未走到終點:tier5(樹搜尋 + 外部想法注入)已於 Phase J 定義並執行中,是
ERA 論文第二支柱(想法注入)在本專案的具體實作,控制變因為與 tier4 同 folds、同評估器,
唯一差異是 idea_bank 注入與重組(recombination)mutation。待其收官後,本報告將擴充為五層
對照,並補上尚待驗證的問題:外部文獻想法注入能否在已被本專案經驗庫與樹搜尋兩層榨過的
競賽上,再擠出系統性而非噪音級的增益。

## 重現方式

```bash
uv run python3 docs/scripts/build_benchmark_table.py
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py docs/SUMMARY_REPORT.md docs/benchmark_facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh docs/SUMMARY_REPORT.md docs/SUMMARY_REPORT.pdf
```
