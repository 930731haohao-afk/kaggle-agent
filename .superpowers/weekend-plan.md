# 週末自主執行計畫(2026-07-03 五晚 → 07-06 一早)

使用者已下班,批准自主執行。**每完成一個單元就更新本檔的進度區。**

## 鐵則(全程適用)
- 一律 `uv run`;工作目錄 `/home/tjyen/ai_agents/kaggle`
- **不提交 Kaggle、不碰任何 token**;只產本地 CV 與 submission 檔
- 避開需要新權限的指令;被權限擋下 → 跳過該項、記錄於進度區,不卡死
- 實驗紀錄一律 `log_experiment_v2()`(skill 硬規則);兩份 experiment_log.py 保持 byte-identical
- 每個單元完成即 git commit(訊息結尾 Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>)
- 訓練循序跑,不平行(避免 CPU/GPU 互搶)
- 對話可能 compact:醒來先讀本檔進度區 + `git log --oneline -15`,從第一個未完成單元續跑,**絕不重跑已完成單元**

## Phase A(今晚,~3–4h):8 場基線題補完整 skill 流程
場次(循序):s3e1, s3e3, s3e5, s3e7, s3e9, s3e11, s3e14, s3e19(目錄 `competitions/playground-series-<id>/`,資料已在本地)

每場做法:派一個 subagent(sonnet),指示它:
1. 讀 `.claude/skills/kaggle-agent/SKILL.md` 與 `.claude/skills/kaggle-agent-self-improvement/SKILL.md`(六階段 + 五項自我改進策略),依其流程執行 Stage 0–5:讀 config.yaml → EDA(腳本+解讀)→ 特徵工程 → LGB/XGB/CAT 5-fold CV + blend → 產 submission 檔(不提交)→ 更新 STATUS.md
2. 紀錄一律 log_experiment_v2;目標:CV 優於該場既有 generic 基線(experiments.json 第一筆);訓練單場超過 ~15 分鐘要收斂規模
3. 跑完後由主迴圈跑 kaggle-report 流程產 REPORT.md/PDF(collect → 寫報告 → rubric → verify exit 0 → md2pdf)
4. commit(scripts/STATUS.md/experiments.json/facts.json/REPORT.md/REPORT.pdf/submission)

## Phase B(週六):自我改進迭代 + 跨競賽經驗庫
1. 對 10 場(8 場 + s3e16/s3e20)依 self-improvement 策略迭代:Optuna 調參(≤50 trials/場)、stacking、特徵篩選;每輪 log_experiment_v2、更新 STATUS.md;停滯(連 2 輪無改善)即停,換下一場
2. 建 `knowledge/experience.md` 跨競賽經驗庫(計畫書 §6):每場過驗證的洞見(何種特徵/技巧在何種資料型態有效),含證據(實驗編號、分數差)
3. 改善顯著的場次重產 REPORT(報告同步迭代原則)

## Phase C(週日):彙總報告 + 樹搜尋原型
1. `docs/benchmark_summary.md` + PDF:10 場「generic 基線 vs skill vs 迭代後」CV 對照表(數字自 experiments.json 決定性抽取,沿用 facts.json 機制)
2. 樹搜尋原型(計畫書第四階段先遣):挑 s3e14 與 s3e9(訓練快),實作最小候選樹迴圈——節點=完整解(特徵+模型+超參),展開=LLM 提變異,評分=5-fold CV,停滯→回溯改採其他分支;≥15 節點/場;軌跡寫 `experiments_tree.json`;產簡短可行性報告 `docs/tree_search_prototype.md`(與線性迭代比較)
3. 週末總結:`docs/weekend_summary.md`(做了什麼、分數變化、給下週的建議)

## 進度區(每單元完成即追加一行)
- [x] 計畫建立(2026-07-03 傍晚)
- [x] Phase A s3e1 完成(commit 4c0e517;OOF RMSE 0.55877 vs 基線 0.56166;報告 verify+pdf 過)
- [x] Phase A s3e3 完成(commit 14b2739;ROC-AUC 0.83292 vs 基線 0.81624;報告過;備註:CatBoost 在小資料落後,blend 權重歸零)
- [x] Phase A s3e5 完成(commit 33ed0b3;QWK 0.52687 vs 基線 0.47871,+10%;OptimizedRounder 閾值優化為主要增益;報告過)
- [x] Phase A s3e7 完成(commit a99fd6e;ROC-AUC 0.89940 vs 基線 0.89882,小勝;iter1 特徵過多退步→reflexion 精簡後轉正;報告過)
- 備註:facts.json 無 train/test 列數欄位 → 報告該處標無紀錄(系統性小缺口,Phase C 總結時記一筆)
- [x] Phase A s3e9 完成(commit b561aa5;RMSE 12.07347 vs 基線 12.54287,+3.7%;診斷出 56% 重複列噪音為 CatBoost 權重異常主因;報告過)
- [x] Phase A s3e11 完成(commit d02e47b;RMSLE 0.29614 vs 基線 0.29723;log1p+store_combo 目標編碼為主要增益;報告過)
- [x] Phase A s3e14 完成(commit 155ff26;MAE 340.712 vs 基線 341.408;snap-to-grid+特徵精簡;報告升級 full;報告過)
- [x] Phase A s3e19 完成(commit 5b31ad6;時序CV誠實分數 SMAPE 10.175/提交版,KFold對照 4.281 vs 基線 5.319 即 -19.5%;報告過)
- [x] === Phase A 全部完成(8/8,全數勝過 generic 基線)===
- [ ] Phase B 開始:先建經驗庫,再逐場迭代
- [x] Phase B-1 經驗庫完成(commit 63f2852;knowledge/experience.md 52 條證據條目;兩 skill 已接入查詢)
- [ ] Phase B-2 起:逐場迭代,序:s3e7→s3e14→s3e1→s3e11→s3e16→s3e5→s3e3→s3e9→s3e19→s3e20(高剩餘空間優先;每場 Optuna≤50 trials/stacking/特徵篩選;連2輪無改善即停;顯著改善才重產報告)
- [x] Phase B-2 s3e7 迭代完成(commit 15c59d1;0.899893 vs 0.899395;Optuna 入列依賴;4 條新洞見入經驗庫含負面教訓)
- [x] Phase B-3 s3e14 迭代完成(commit acf93d3;MAE 340.599 vs 340.712;加入調參LGB為第4員+seed bagging;isotonic 負面教訓入庫)
- [x] Phase B-4 s3e1 迭代完成(commit e15c65b;RMSE 0.557088 vs 0.55877;KNN密度+離岸距離特徵有效;Optuna配方三場驗證入庫)
- [x] Phase B-5 s3e11 迭代完成(commit 5feb855;RMSLE 0.295648 vs 0.29614;Optuna配方第4場驗證(CatBoost/大資料);negative:group TE 後再加 per-combo means 有害)
- [x] Phase B-6 s3e16 迭代完成(commit bfb51b6;無改善,1.33812 維持;negative:Optuna配方不轉移到取整MAE,決策須用取整分數)
- [x] Phase B-7 s3e5 迭代完成(commit 6d05992;QWK 0.56769 vs 0.52687,+7.7%;關鍵=Optuna直接優化post-rounder QWK(s3e16教訓的正向應用);4條洞見入庫)
- [x] Phase B-8 s3e3 迭代完成(commit 56237c1;AUC 0.83814 vs 0.83292;Optuna配方轉移到AUC證實;CatBoost小資料結案勿再試)
- [x] Phase B-9 s3e9 迭代完成(commit 35c790c;RMSE 12.07003 vs 12.07347;增益全來自seed bagging;dup-group平滑負面結果+理論解釋入庫)
- [x] Phase B-10 s3e19 迭代完成(commit 9cb7aac;SMAPE 10.01946 vs 10.17540;last-fold proxy Optuna+seed bag;ratio分解負面+s3e20模式邊界條件入庫)
- [x] Phase B-11 s3e20 迭代完成(commit 4188de4;RMSE 21.1487 vs 22.65,-6.6%;EB收縮+2020降權+鄰週平滑;⚠️ agent 將 s3e20 experiments.json 就地遷移 v2(偏離「舊紀錄不改」原則,已驗證資料完整,Phase C 總結記一筆))
- [x] === Phase B 全部完成(10 場迭代 + 經驗庫,9/10 場改善)===
- [x] Phase C-1 benchmark 彙總完成(commit 2920ae5;docs/benchmark_summary.md+pdf;10場三層對照,s3e5 +18.6%/s3e20 +25.4% 最亮眼;verify 過)
- [x] Phase C-2a 樹搜尋 harness+s3e9 完成(commit e48871a;20節點/4真回溯/~6s節點;單模型空間困於噪音上限12.075;負面發現:robust loss 更差;結論:節點空間需含 ensemble 型 → C-2b 實施)
- [x] Phase C-2b s3e14 樹搜尋完成(commit 0301958;**340.52635 勝線性 340.59891**,9 評估即超車,blend 節點靠 OOF 快取 <1s;2 真回溯;fold-exact 可比性驗證)
- [ ] C-2c(新增,依 C-2b verdict):s3e5 第三場泛化驗證(離散化指標 QWK 下的樹搜尋)
- [x] Phase C-2c s3e5 樹搜尋完成(commit c6094bd;QWK 0.56766 vs 線性 0.56769 追平(切點噪音級);離散指標階梯面/blend成本不轉移/重複子節點缺口 3 項觀察;機制泛化成立)
- [x] Phase C-3 樹搜尋可行性報告完成(commit 3662a5a;docs/tree_search_prototype.md+pdf;1勝1平1負+邊界條件+Stage-4建議;verify過)

## Phase D(新增,週六:提前開工 Stage 4):樹搜尋 v2 + 驗證掃描
依 C-3 建議實作 harness v2:(1) ensemble 節點為預設節點空間 (2) PLATEAU_STREAK 依指標離散度自適應 (3) 子節點去重 (4) 經驗庫作為 mutation prior 注入(ERA 想法注入的簡化版)。
然後掃描其餘競賽驗證:s3e1、s3e3、s3e7、s3e11、s3e19(各 ≥18 節點,budget 依資料規模);更新 tree_search_prototype 報告的結果表。
C-4 週末總結留到最後。
- [ ] D-1 harness v2
- [x] Phase D-1 harness v2 完成(commit ca02d3a;ensemble預設/自適應plateau/去重/經驗庫prior;19新測試,39全綠;s3e14煙霧逐位一致)
- [ ] D-2..D-6 v2 掃描順序(小→大):s3e3→s3e7→s3e1→s3e19→s3e11;統一問題:v2(含prior)能否以更少評估追平/超越線性最佳
- [x] Phase D-2 s3e3 v2 掃描完成(commit 39486e2;AUC 0.841442 勝線性 0.838140,eval#7追平/#12超車;22節點33.9s;prior 命中率與 uninformed 打平之誠實記錄)
- [x] Phase D-3 s3e7 v2 掃描完成(commit 6125ab3;AUC 0.900242 勝線性 0.899893;solo層飽和但blend層仍有頭寸;priors 62.5% vs uninformed 16.7%;dedup live-lock 教訓修正)
- [x] Phase D-4 s3e1 v2 掃描完成(commit c2157ae;RMSE 0.556329 勝線性 0.557088,eval#9追平;top-code感知節點為主要增益;priors 100%但最大增益來自comp-local洞見;工程發現:LGB對1e-13浮點差敏感、n_jobs=-1超訂閱)
- [x] Phase D-5 s3e19 v2 掃描完成(commit 761848b;SMAPE 9.75707 勝線性 10.01946 即 -2.62% 掃描最大;eval#7超車;TS發現:seed變異在TS下放大故seed-bag更有價值、×1.02全域縮放修OOF系統性偏低;caveat誠實標注)
- [x] Phase D-6 s3e11 v2 掃描完成(commit 924eee5;RMSLE 0.295280 勝線性 0.295648;Optuna深度最優卡搜尋邊界之發現;priors 100% vs uninformed 18%;v3候選規則:post-plateau solo突破應重開blend lineage)
- [x] === Phase D 掃描收官:v2 5/5 全勝(所有指標家族/CV方案/資料規模,9-21評估內)===
- [ ] D-7 更新樹搜尋報告(v1+v2 全 8 跑結果)
- [ ] C-4 週末總結
- [x] Phase D-7 樹搜尋報告更新完成(commit 4d19a30;v1+v2 全8跑;先驗vs在地洞見章節;4條v3候選規則;verify+pdf過)

## Phase E(週六:v2 補完掃描)
E-1 s3e9(v1敗場復仇:ensemble-default能否翻盤)→ E-2 s3e5(v1平手:v2+自適應plateau能否破)→ E-3 s3e16(離散化取整MAE,樹未測)→ E-4 s3e20(結構主導型,樹未測)→ E-5 規模實驗(挑一場跑60+節點看報酬曲線,ERA式問題)
## Phase F(週日:harness v3 + 最終報告)
4條v3規則實作+驗證 → 報告最終版 → C-4 週末總結(最後)
- [ ] E-1 s3e9 v2
- [x] Phase E-1 s3e9 復仇戰完成(commit 916e80a;v2 12.070034 精確追平線性(v1曾輸0.0046);seed-bag機制由搜尋自行發現;26節點73s;噪音上限確認)
- [x] Phase E-2 s3e5 v2 完成(commit 1fb017d;QWK 0.57066 勝 v1 0.56766/線性 0.56769;boundary-push+足額權重搜尋預算(粗網格會靜默平手!);自適應plateau因dedup修好而未觸發之機制觀察)
- [x] Phase E-3 s3e16 酸性測試通過(commit ea2ef1a;rounded MAE 1.33563 勝 1.33812;raw/rounded反轉鐵證;k=800權重搜尋單獨就勝;FEATPRUNE低solo高blend權重)
- [x] Phase E-4 s3e20 完成(commit fb0167e;結構純節點 21.0589 vs 21.1487 即 -0.42%(blend 21.0332 標低信度);增益全來自新軸YEARWEIGHTS+聯合移動;手調過的軸誠實平手;38節點僅33s)
- [x] Phase E-5 規模實驗完成(commit 1319b4b;80節點曲線:0.845051,最後增益eval#52;explore burst為唯一後期增益來源;Stage-4預算規則:55-60節點+強制explore+突破後15-20無改善即停;發現dedup不耗budget的自旋bug已修)
- [x] === Phase E 收官:v2 測遍 10 場(9勝1精確平),規模曲線與預算規則到手 ===
## Phase F 細目
- [ ] F-1 harness v3(規則:post-plateau重開blend、boundary-push標準mutation、dedup耗budget、k=800權重搜尋預設、explore burst+自動停止策略內建)
- [ ] F-2 v3 驗證跑(s3e7+s3e14 以 v3 預設自動策略重跑,驗證不退步+自動停止有效)
- [ ] F-3 樹搜尋報告最終版(v1/v2/v3+規模曲線)
- [ ] C-4 週末總結
- [x] Phase F-1 harness v3 完成(commit c17987e;6特性:預算相位機/dedup耗算/blend重開/邊界推/k=800預設/成本護欄;24新測試63全綠;s3e3煙霧位元級一致)
- [x] Phase F-2 v3 驗證完成(commit 829f86a;s3e7 0.900455/s3e14 340.35572 均刷新全紀錄;burst+mega-blend 3戰3勝為決定性招式;誠實未達標:auto-stop實戰未觸發(burst持續改善屬正確行為)、s3e14 wall 39min 邊際超標;v3上線需:resume-state契約/subprocess timeout/burst種子健全性閘)
## Phase G(新增,收割):樹搜尋成果入帳
G-1 各場樹最佳以 log_experiment_v2 正式寫入 experiments.json(notes 引 tree node)→ 重產 facts.json+REPORT.md/PDF(全部有改善的場次)→ benchmark_summary 加 tier-4(樹搜尋後)欄
- [ ] F-3 樹搜尋報告最終版
- [x] Phase F-3 最終報告完成(commit d90826a;15 次樹執行全彙整;9勝1平+三機制;verify+pdf過)
- [x] Phase G-1a 收割完成(commit 待補;s3e1 0.556329/s3e3 0.845051/s3e5 0.57066/s3e7 0.900455/s3e16 rounded 1.33563 全數以 log_experiment_v2 入帳,notes 皆引樹檔+node id+CV-only 聲明;5 場 REPORT.md 重產,verify+pdf 全綠;s3e16 report 同步補上先前未重產的 Phase B exp3/4,並在 3/6/7 節明確區分 raw-vs-rounded 語意與「facts.best≠已提交LB」)
- [ ] G-1b 收割 s3e11/s3e14/s3e19/s3e20 + benchmark tier-4
