# 跨競賽經驗庫(自動蒸餾,每則附證據)

最後更新:2026-07-03。查詢方式:先按「資料型態/指標」節找,再看「證據」欄確認遷移性。
來源:playground-series s3e1/s3e3/s3e5/s3e7/s3e9/s3e11/s3e14/s3e16/s3e19/s3e20 的 STATUS.md 與 experiments.json。
凡未附分數差的傳聞一律不收錄;「證據」欄格式為 `競賽, exp #N, 分數A→分數B`。

---

## 依指標的技巧

### MAE/整數目標
- 目標為整數時,把連續預測「四捨五入到整數」直接降低 MAE,且 CV 增益可反映到 LB。 | 證據:s3e16, exp #1, LGB 未取整 OOF MAE 1.35651 → 取整 1.33885;blend 取整 1.33812,Kaggle Private LB 1.34075(gap 僅 ~0.006)。
- 目標值域是離散組合(非整數格點)時,「snap 到最近的訓練集觀測值」小幅但穩定優於不處理/僅 clip。 | 證據:s3e14, exp #2/#3,blend 341.02061 → snap 340.95856;exp #3 blend 340.75961 → snap 340.71180(兩次迭代皆勝出)。

### QWK/序數目標
- 序數目標用「迴歸 + OptimizedRounder(對 OOF QWK 調切點)」遠勝 naive rounding——這是小資料序數題的單一最高槓桿改動。 | 證據:s3e5, exp #2→#3,同一 blend、naive round QWK 0.47191 → 優化切點 0.52687(+0.055,約 +10% 相對)。
- naive rounding 在嚴重類別不平衡(69.9x)下甚至比不做特徵工程的 baseline 更差:稀有類預測被主流類質量拉偏,對稱取整假設崩潰。 | 證據:s3e5, exp #1 baseline 0.47871 vs exp #2 naive round 0.47191。
- 序數題選迴歸頭而非多分類頭:QWK 罰平方距離,迴歸的連續輸出能借用中間類的訊號安置樣本極少的極端類(12–39 例)。 | 證據:s3e5, STATUS.md 建模決策;最終 0.52687 由迴歸頭達成。
- 風險:切點若直接對整份 OOF 向量擬合(非 nested CV),有輕度過擬 OOF 的風險;且極端類可能完全不被預測(s3e5 測試集無 quality 3/8 預測)。 | 證據:s3e5, STATUS.md 明示風險註記。

### SMAPE/時序
- test 為未來期間(與 train 零日期重疊)時,必須用 TimeSeriesSplit(以唯一日期切)驗證;隨機 KFold 是「內插式」樂觀估計,會嚴重高估。 | 證據:s3e19, exp #2(TimeSeriesSplit)SMAPE 10.175 vs exp #3(同特徵同模型、隨機 KFold)4.281——差 2.4 倍,純粹是 CV 方案差異。
- 外推情境下模型排序會改變:內插時三模型幾乎等強(blend 0.4/0.2/0.4),外推時 LGB 明顯領先、XGB 被權重搜尋歸零——實際提交應採「與 test 情境相同的 CV」所得權重。 | 證據:s3e19, exp #2 權重 LGB0.9/XGB0.0/CAT0.1 vs exp #3 權重 0.4/0.2/0.4。
- 右偏目標(skew 1.75)以 log1p 訓練、expm1 還原,使分布近對稱(-0.22)。 | 證據:s3e19, exp #2 採 log1p,配合上述最佳 blend。

### ROC-AUC(排名指標)
- AUC 是排名指標:對不平衡二分類加 scale_pos_weight/class_weights 主要是擾動損失面而非改善排名,在小樣本上是壞交易;移除不平衡加權 + 加強正則化才是解方。 | 證據:s3e3, exp #2(48 特徵+不平衡加權)0.81901 → exp #3(同特徵、移除加權、LGB num_leaves 15→7、L1 0.5→1.0、L2 1.0→2.0)0.83292(+0.017)。

### RMSLE
- RMSLE = 對 log1p 目標的 RMSE:即使目標不偏態(skew 0.019)也應以 log1p+RMSE objective 直接優化該指標,預測 expm1 還原並 clip ≥ 0。 | 證據:s3e11, exp #2/#3 均採此法;方法論本身增益小(exp #1→#2 僅 -0.00013),但為正確優化目標的前提。

### RMSE/極偏態目標
- 極右偏目標(skew 10.2)必 log1p;線性模型(Ridge)在此設定下可災難性爆炸,GBDT 穩健。 | 證據:s3e20, exp #1,Ridge RMSE 23,917.63 vs LGB 32.75(均值 baseline 155.54)。

---

## 依資料型態

### 小樣本(<10k 列)
- 小樣本上 CatBoost 可能持續劣於 LGB/XGB 且被權重搜尋歸零——不要假設三模型 blend 永遠有益,先看各模型 OOF。 | 證據:s3e3(1,677 列), exp #2/#3,CAT 0.77663/0.76268 遠低於 LGB 0.81901/0.83292,兩次 blend 權重皆 LGB=1.0。
- 小樣本 + 相關性高的工程特徵 → 優先加正則化而非加容量。 | 證據:s3e3, exp #3 靠緊縮 LGB 正則拿到 +0.017;s3e9, exp #2 同理(見下)。
- 小樣本 per-fold 分數波動大(AUC 0.79–0.89)是常態,不是 bug;拿到明確增益後應「見好就收」,避免在噪音上過度搜尋。 | 證據:s3e3, STATUS.md 明示 plateau 停止準則,一輪 reflexion 後停。

### 重複列/標籤噪音
- 特徵完全重複但目標不同(~56% 列)= 標籤噪音上限:獎勵正則化、懲罰容量;預設超參的 LGB/XGB 會嚴重過擬,CatBoost 的 ordered boosting 天然抗性較強。 | 證據:s3e9, exp #1,預設參數下權重搜尋給 CAT 100%(12.54287);exp #2 正則化 LGB(leaves15/depth5/L1=2/L2=4)後 LGB 13.21→12.11。
- 正則化與特徵工程要「一起上」才有效——同樣特徵但不正則,LGB/XGB 仍過擬。 | 證據:s3e9, exp #2,blend 12.54287 → 12.07347(-3.74%),STATUS.md 明示為兩者合力。

### 高共線性特徵
- 近完美共線欄位(r≥0.999 的 6 個 TRange 欄)可安全精簡,配合重要度探針剪枝反而更好。 | 證據:s3e14, exp #2(27 特徵)blend 341.02 → exp #3(剪至 21 特徵)340.76。
- 共線性可轉化為特徵:已知恆等式(Weight≈Shucked+Viscera+Shell, r=0.993)的殘差 `weight_resid` 是有效工程特徵。 | 證據:s3e16, exp #1,含 weight_resid 的 24 特徵組取整 blend 1.33812 vs 通用 baseline 1.35441。

### 低訊號資料(全部原始特徵 |r|≤0.11)
- 原始特徵全弱時,找「群組結構」做 fold-safe 目標編碼是最大槓桿(店面輪廓群組均值 R²≈0.06 已勝任何單一原始特徵)。 | 證據:s3e11, exp #2(方法論)0.29710 → exp #3(+store_te)0.29614,增益主要來自 store_te,遠大於方法論改動。

### 地理座標資料
- 原始經緯度線性相關弱(r≈-0.12/-0.06)但為 GBDT 重要度前二——地理訊號是非線性的,樹自己會切,直接餵原始座標 + 距離特徵即可。 | 證據:s3e1, EDA + exp #2(含 dist_nearest_city、KMeans geo_cluster)0.56166 → 0.55877。
- 在樹已能利用原始座標的前提下,再加粗粒度空間目標編碼無增益。 | 證據:s3e1, exp #3,+geo_te 後 LGB 0.56109 → 0.56102(-0.00007,噪音級,未採用)。

### 具跨年穩定結構的時空資料
- 先驗證「歷史均值」基準:若目標對 (位置, 週次) 跨年近乎恆定(跨年 std 中位數 ≈3.1),純 location-week 歷史均值可完勝所有 GBDT。 | 證據:s3e20, exp #4,4 路權重搜尋給 TE-mean 100%、三個 GBDT 全 0;TE-mean 22.6488 vs 最佳 GBDT blend 28.3424。理解資料勝過模型複雜度。

### 資料品質例行檢查(多賽驗證)
- 常見瑕疵及處置:常數欄直接刪(s3e3 三欄零變異)、物理不可能值當缺失中位數補(s3e16 Height==0 共 24 列)、>90% 缺失欄刪除 + 10–90% 缺失做指示旗標(s3e20 刪 7 欄、建 21 旗標)、目標 top-coding 封頂認知(s3e1 MedHouseVal 4.92% 封在 5.00001,封住 RMSE 上限)。 | 證據:各競賽 STATUS.md EDA 節。

---

## CV 設計

- **絕不跨 CV 方案比較分數;每個實驗必須連同 CV 方案一起記錄。** 換方案後分數「看似退步」可能純屬方案更誠實。 | 證據:s3e19, exp #2 (10.18) 看似輸 baseline (5.32),診斷性 exp #3 證明同配置在舊方案下其實是 4.28(更好)。
- test 為純未來期 → TimeSeriesSplit(對唯一日期切);並以該方案下的權重做實際提交。 | 證據:s3e19, exp #2。
- 多年度時空資料 → Leave-One-Year-Out CV,可誠實評估跨年外推並支撐無洩漏目標編碼。 | 證據:s3e20, exp #3/#4。
- 迴歸目標的分箱分層:小樣本噪音迴歸用目標十分位 StratifiedKFold(s3e9, qcut 10);整數長尾目標用分箱 + 尾端合併(s3e16, Age≥20 併一箱,fold MAE 穩定於 1.34–1.37)。 | 證據:各該賽 STATUS.md。
- 嚴重類別不平衡的序數/分類目標 → 對 label 直接 StratifiedKFold,避免 fold 內極端類為零。 | 證據:s3e5(69.9x 不平衡), 5-fold Stratified on quality,per-fold std 0.033。
- 無時間/群組結構、目標連續、train/test 分布近同 → 普通 shuffle KFold 即正解,不必過度設計。 | 證據:s3e1、s3e11、s3e14(s3e14 各欄 train/test 均值差 <1%)。
- CV↔LB gap 實測紀錄:s3e16 OOF 1.33812 → Public 1.34356 / Private 1.34075,gap ~0.006(CV 輕微樂觀、可信賴)。 | 證據:s3e16, exp #1 Kaggle 提交。
- 固定 seed + 固定 fold 才能跨實驗比較;revert 後重跑應可 byte-identical 重現。 | 證據:s3e9, exp #4 重現 exp #2 的 12.07347 分毫不差。

---

## 超參調校(Optuna)

- 只調「最強單模」的超參是 blend 的高性價比槓桿:LGB 50 trials(TPE)即 +0.0004 單模 / +0.0005 blend;最佳解為「淺而強正則」(depth 3、lr 0.068、reg_alpha 2.14),再次印證小/中型資料獎勵正則化而非容量。 | 證據:s3e7, exp #3→#4,LGB 0.898824→0.899215,blend 0.899395→0.899891。
- 調參預算控制:每 trial 跑完整 5-fold 會爆時間(50 trials × 5 folds ≈ 25 分鐘,一次逾時全歸零);改用「單一 fold(fold-0)代理目標」調參、只對優勝配置重跑完整 5-fold 驗證,50 trials 僅 300s,且代理排序成功遷移(fold-0 最佳配置在全 OOF 也最佳)。 | 證據:s3e7, exp #4,第一次全 5-fold 嘗試 25 分鐘逾時無結果;fold-0 代理 300.2s 完成並拿到上述增益。
- **對第二個基模型重複同一調參配方會傷 blend**:XGB 同法調參後單模微升(0.898765→0.898860)但 blend 反而退步(0.899891→0.899722)——兩個模型都收斂到相似的淺樹最優解,ensemble 多樣性下降。單模分數與 blend 貢獻不是同一回事;調完最強單模後,其餘基模型的「次優但異質」參數是資產不是負債。 | 證據:s3e7, exp #5(棄用)。
- **調參後的單模應「加入」pool 而非「替換」原成員**:Optuna 調參版 LGB 單模由 342.02154 進步到 341.68775,但若直接替換掉原 LGB,blend 反退步(340.75961→340.95316,多樣性流失,權重被迫壓到 XGB/CAT);改為「保留原 LGB 並把調參版當額外成員」的 4-way blend 則降至 340.62702。是 s3e7「別替換多樣成員」教訓對最強模型自身的推廣。 | 證據:s3e14, exp #4(替換)340.95316 vs exp #5(增列)340.62702。
- **seed bagging(同超參、換 random_state)是調參之後最便宜的殘餘增益**:再加一個 seed=2024 的原參數 LGB 作第 5 成員,blend 340.62702→340.59891;成本僅一次 5-fold 訓練(~52s)。 | 證據:s3e14, exp #7,5-way blend 權重 0.2/0.15/0.2/0.25/0.2。
- 混合層微調(0.05→0.01 權重網格、rank-average)在 AUC 上只有噪音級差異(+0.000002 / −0.000007),不是可靠增益來源。 | 證據:s3e7, exp #6,prob 0.01grid 0.899893 vs 0.05grid 0.899891 vs rank 0.899884。

---

## Ensemble/後處理

- OOF 權重搜尋常直接淘汰弱模型(權重=0)——這是特徵而非 bug,尊重它:s3e3 CAT=0(兩次)、s3e11 XGB=0(兩次)、s3e19 外推下 XGB=0(兩次)、s3e20 三個 GBDT 全=0。 | 證據:各該賽 experiments.json blend_weights。
- 三模型時,單純 simplex 網格權重搜尋難被學習式 stacking 打敗:Ridge(非負)stacking 反而明顯更差。 | 證據:s3e14, exp #3,Ridge stack 344.04 vs simplex blend 340.76。
- 未優化的 50/50 等權 ensemble 可能比最佳單模差——權重必須搜尋。 | 證據:s3e20, exp #2,LGB 32.75 + XGB 34.44 等權 = 33.21 > LGB 單模。
- 模型實力相近時 blend 才穩定勝過所有單模。 | 證據:s3e1, exp #2,blend 0.55877 < LGB 0.56109/XGB 0.56297/CAT 0.56188。
- 後處理三寶皆有實證:整數取整(s3e16, 1.3565→1.3389)、OptimizedRounder(s3e5, +0.055)、snap-to-grid(s3e14, ~+0.05~0.06);另 clip 到訓練目標值域作安全網(s3e1、s3e11)。 | 證據:見「依指標」節各條。

---

## 特徵工程模式

- fold-safe(K-fold out-of-fold)目標編碼是群組結構資料的最大單一增益來源;未見組合用全域均值 fallback。 | 證據:s3e11, exp #3 store_te(~111 組合、19 個測試集未見組合)貢獻主要增益 0.29710→0.29614;s3e20, te_locweek(以其他年度計算,無洩漏)單獨即達 22.65。
- 比值/物理領域特徵優於原始欄:s3e9 water_binder_ratio(修正為總膠結料,Pearson -0.227 vs 傳統水灰比 -0.151)、log1p(AgeInDays)(0.558 vs 原始 0.334);s3e16 部位重量比、密度、殼肉比;s3e5 alcohol_x_sulphates(|Spearman| 0.550 vs 單特徵 0.504/0.457,top-10 重要度中 7 個是工程特徵);s3e1 rooms_per_person(r=0.452,最強新特徵)。 | 證據:各該賽 exp 分數已列前節。
- **特徵過多反而退步 → 精簡是常規武器**:s3e7 exp #2(+14 特徵)0.89788 低於 baseline 0.89882,剪掉噪音特徵後 exp #3 0.89939 反超;s3e14 剪 6 特徵後 341.02→340.76;s3e9 加 3 個交互項 12.07347→12.09483 後 revert。 | 證據:各該賽 experiments.json。
- 樹模型會自己學乘法交互:顯式 product 交互欄(即使有領域道理)在小噪音資料上只添共線與維度,無新資訊。 | 證據:s3e9, exp #3,+binder×log_age 等 3 項 12.07347→12.09483,revert。
- cyclical(sin/cos)編碼看情境:目標真由日曆驅動時有效(s3e19 時序,20 特徵組是最終最佳);底層日期欄與目標近零相關(r 0.003/0.008)時是純噪音,刪之反升。 | 證據:s3e19 exp #2 vs s3e7 exp #2→#3。
- 迭代停止準則:2–3 輪後增益遞減即停(s3e7 兩輪、s3e14 兩輪、s3e3 一輪 reflexion),把預算留給下一賽。 | 證據:各該賽 STATUS.md 明示停止理由。

---

## 反面教訓(試過沒用的)

- 對 AUC 目標加類別不平衡加權(scale_pos_weight/class_weights)→ 反而拖累所有模型;XGB/CAT 甚至跌破自己的無特徵 baseline。 | 證據:s3e3, exp #2,XGB 0.80511→0.79196、CAT 0.80874→0.77663;exp #3 移除後回升。
- 樹已用原始經緯度時再加粗粒度地理目標編碼 → 噪音級增益,不採用。 | 證據:s3e1, exp #3,-0.00007。
- 顯式交互項(有物理依據亦然)加給 GBDT → 退步,revert。 | 證據:s3e9, exp #3,12.07347→12.09483。
- 對「與目標無關的日期欄」做 cyclical 編碼與交互 → 整組特徵工程輸給 baseline;剪掉才反超。 | 證據:s3e7, exp #2 0.89788 < baseline 0.89882;exp #3 剪後 0.89939。
- Ridge stacking meta-model(僅 3 個 base OOF)→ 輸給 simplex 網格搜尋 3.3 個 MAE。 | 證據:s3e14, exp #3,344.04 vs 340.76。
- 對已收斂的 GBDT blend 輸出做 nested OOF isotonic 校準(MAE 目標)→ 強烈負向:isotonic 本身 346.99717,較未校準 blend 340.69924 惡化 +6.3 MAE,自動否決。blend 在中位數意義上已良好校準,單調重校只增變異。是繼 Ridge 之後,s3e14 上第二個敗給 simplex 網格混合的 meta/後處理法。 | 證據:s3e14, exp #6,iso 346.99717 vs blend 340.69924。
- Ridge 線性模型直上 log 轉換之極偏態目標 → RMSE 爆炸至 23,917(均值 baseline 才 155.5)。 | 證據:s3e20, exp #1。
- 序數目標 naive rounding → 比不做特徵工程的 baseline 還差。 | 證據:s3e5, exp #2,0.47191 < baseline 0.47871。
- 等權(未搜尋)ensemble → 比最佳單模差。 | 證據:s3e20, exp #2,33.21 > LGB 32.75。
- 在「目標跨年近恆定」的資料上堆 GBDT 與 63 個感測器特徵 → 全被純歷史均值淘汰(權重 0)。感測器特徵對此目標本質上是噪音。 | 證據:s3e20, exp #4,GBDT blend 28.34 vs loc-week mean 22.65。

---

## 待驗證想法(有 EDA 依據但未實測,勿當作已驗證)

- s3e19:store/product 年度佔比極穩(Kagglazon ≈0.691 每年)但 country 佔比逐年漂移 → 比例分解預測 + country-share 趨勢外推,未試。
- s3e3:CatBoost 在小樣本落後的原因(原生類別處理 vs 正則化需求不同)未診斷。
