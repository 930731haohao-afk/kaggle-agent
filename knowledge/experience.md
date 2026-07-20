# 跨競賽經驗庫(自動蒸餾,每則附證據)

最後更新:2026-07-04(s3e20 Phase B 迭代新增:歷史均值的三段去噪配方(經驗貝葉斯收縮/異常年降權/鄰週平滑,22.6488→21.1487)與「結構飽和延伸到殘差層」的殘差診斷反例)。查詢方式:先按「資料型態/指標」節找,再看「證據」欄確認遷移性。
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
- 序數題選迴歸頭而非多分類頭:QWK 罰平方距離,迴歸的連續輸出能借用中間類的訊號安置樣本極少的極端類(12–39 例)。 | 證據:s3e5, STATUS.md 建模決策;最終 0.52687 由迴歸頭達成;Phase B 迭代再驗證:多分類(class_weight=balanced)+期望值解碼作為多樣性成員加入 blend,單模 post-rounder QWK 僅 0.52040(顯著低於已調校的迴歸成員 0.56244/0.56466),權重搜尋歸零,對 blend 無貢獻。 | 證據:s3e5, exp #11(棄用),0.56769 = 0.56769(無變化)。
- **小資料(~2k列)序數 + QWK 的最高槓桿不是加特徵或加模型,而是把 Optuna 目標函式直接設為「後處理(OptimizedRounder)後的 QWK」,在完整 5-fold CV 上跑(資料小到不需要 fold-0 代理),而非傳統的「先調 RMSE,再套用 rounder」——這是避開離散化指標陷阱最直接的做法,增益遠超過單純優化切點或權重搜尋本身。** | 證據:s3e5, exp #4→#5,LGB 以此法調校(TPE,40 trials,600s timeout,實際僅 26s)後加入池,0.52986→0.56293(+0.03307);exp #5→#8,同法調校 CatBoost 再加入,0.56293→0.56769(+0.00476);累計較「切點優化但未調參」的既有最佳 0.52687 提升 +0.04082(+7.7% 相對)。
- 巢狀(leave-fold-out)切點擬合可驗證「切點對全 OOF 過擬」的風險大小:風險並非固定值,會隨集成池成熟而縮小。 | 證據:s3e5, exp #7(5-way blend,單一調校模型主導)全 OOF 0.56293 vs 巢狀 0.54649,gap +0.01644;exp #10(6-way blend,兩個調校模型)全 OOF 0.56769 vs 巢狀 0.56393,gap 縮小至 +0.00377。
- 風險(原始記錄,已由上述巢狀診斷部分緩解):切點若直接對整份 OOF 向量擬合(非 nested CV),有輕度過擬 OOF 的風險;且極端類可能完全不被預測(s3e5 測試集無 quality 3/8 預測,Phase B 迭代後的新最佳甚至連 quality 4 也不再預測,僅剩 5/6/7)。 | 證據:s3e5, STATUS.md 明示風險註記。

### SMAPE/時序
- test 為未來期間(與 train 零日期重疊)時,必須用 TimeSeriesSplit(以唯一日期切)驗證;隨機 KFold 是「內插式」樂觀估計,會嚴重高估。 | 證據:s3e19, exp #2(TimeSeriesSplit)SMAPE 10.175 vs exp #3(同特徵同模型、隨機 KFold)4.281——差 2.4 倍,純粹是 CV 方案差異。
- 外推情境下模型排序會改變:內插時三模型幾乎等強(blend 0.4/0.2/0.4),外推時 LGB 明顯領先、XGB 被權重搜尋歸零——實際提交應採「與 test 情境相同的 CV」所得權重。 | 證據:s3e19, exp #2 權重 LGB0.9/XGB0.0/CAT0.1 vs exp #3 權重 0.4/0.2/0.4。
- 右偏目標(skew 1.75)以 log1p 訓練、expm1 還原,使分布近對稱(-0.22)。 | 證據:s3e19, exp #2 採 log1p,配合上述最佳 blend。
- **反例(界定 s3e20「結構勝 GBDT」模式的邊界條件):比例分解(total(date) × country_share 趨勢外推 × 店/產品固定佔比)在「總量序列本身難外推」的面板時序上輸給 GBDT,權重搜尋歸零**。即使 additive OLS 驗證 log1p(target) 僅用三個類別欄(完全不用日期)就能解釋 R²=0.974 的變異——結構確實存在——分解模型 solo OOF SMAPE 14.75 仍遠劣於 LGB 10.18。oracle 診斷(餵入真實每日總量、隔離佔比模型誤差)顯示佔比分解部分 SMAPE 10.0–10.8,與 GBDT 相當而非更好:GBDT 的原生類別分裂已把佔比結構吃掉了,真正瓶頸是「總量一年外推」(5 年 COVID 干擾序列:4.45M→4.72M→4.52M→4.09M→4.88M,僅 4-5 個年度點無可靠趨勢),而這瓶頸對任何方法一視同仁。**與 s3e20 對照得出的耐用二分法:結構信號要贏 GBDT,條件是「目標跨年近恆定」(s3e20 跨年 std 中位數 ≈3.1 → 歷史均值完勝);若總量水平本身逐年漂移且不可外推(s3e19),結構分解不添加 GBDT 沒有的資訊。** 附帶發現:漂移中的 country share 用線性趨勢外推比直接用 pooled 平值更差(oracle SMAPE 10.84 vs 10.04)——年度點太少太噪,趨勢擬合是負資產。 | 證據:s3e19, exp #4,blend 10.17540→10.17489(增益來自更細權重網格而非 RD;RD 權重 0.0),棄用。
- **TimeSeriesSplit 下的 Optuna fold-proxy 應選「最後一折」而非 fold-0**:時序折不可互換——最後一折訓練窗最大、驗證區位最接近真實 test 外推情境。以 fold-5 SMAPE 為目標(40 trials 僅 42s),調出的 LGB 更淺更簡(num_leaves 63→20、min_child_samples 20→38),再次印證外推情境獎勵正則化;調參版全 5-fold solo OOF 10.14833(勝過所有 seed-bag 成員),依「加入池不替換」配方進 6-way 權重搜尋後拿 0.5 權重,blend 10.15721→10.01946,為該場單輪最大增益。**誠實註記:fold-5 既是調參目標又佔 OOF 的 1/5,10.019 有部分樂觀成分(tuned 的 fold-5 分數 8.13 是被選擇過的);但方向性增益真實(folds 2-3 未被調參也小勝原 LGB)。更嚴謹做法是權重搜尋只用 folds 1-4。** | 證據:s3e19, exp #7,10.15721→10.01946(-0.137)。
- seed bagging 配方在 SMAPE/時序外推情境同樣有效,兩輪皆正增益(與 s3e9/s3e14 一致):+LGB seed2024 得 10.17489→10.16605,+LGB seed7 與 +CAT seed2024 再得 10.16605→10.15721。 | 證據:s3e19, exp #5/#6。

### ROC-AUC(排名指標)
- AUC 是排名指標:對不平衡二分類加 scale_pos_weight/class_weights 主要是擾動損失面而非改善排名,在小樣本上是壞交易;移除不平衡加權 + 加強正則化才是解方。 | 證據:s3e3, exp #2(48 特徵+不平衡加權)0.81901 → exp #3(同特徵、移除加權、LGB num_leaves 15→7、L1 0.5→1.0、L2 1.0→2.0)0.83292(+0.017)。
- **s3e5 的「Optuna 目標函式直接設為完整 CV 上的最終指標」配方(原本只在 QWK/離散化後處理場景驗證過)遷移到 AUC 同樣有效,且 AUC 是連續排名指標,完全沒有 QWK 那種離散化陷阱**:小資料(1677 列)上 50 trials 完整 5-fold CV 僅需 111.3 秒,直接以 OOF ROC-AUC(而非某代理 loss)為 Optuna 目標,找到的最優解甚至比既有「淺而強正則」的 exp #3 LGB 更淺(num_leaves 7→3)但正則反而更輕(reg_lambda 2.0→0.0012,靠 min_child_samples=60 控制過擬合)。 | 證據:s3e3, exp #4,LGB 單模 0.832925 → 0.837305(+0.00438)。
- 調參後「加入池而非取代 + seed bagging」配方在 AUC 上同樣可疊加小幅增益,惟量級隨迭代次數遞減(+0.0044→+0.0005),與 s3e14/s3e11 觀察一致。 | 證據:s3e3, exp #5,5-way 權重搜尋 blend 0.837305 → 0.837776,權重集中在 LGB_orig(0.3)+LGB_tuned(0.7),seed-bag 版本權重 0。
- **「移除不平衡加權對 AUC 有利」的結論原僅在小樣本(1,677 列)驗證過;跨季泛化首測(s4e1,165,034 列,銀行流失預測)首次在大樣本上做逐位元可比的消融,結論同向成立**:同一 28 特徵集、同一 5-fold StratifiedKFold(seed=42),LGB 加 `is_unbalance=True` OOF AUC 0.893235 vs 不加 0.893650,delta −0.000415——方向與 s3e3 一致但量級小得多(大樣本下加權對排序的擾動本就較弱)。附帶發現:2026-02 該賽的舊基線(is_unbalance=True,CV 0.89653)之所以看起來大幅優於本次不加權版本,主因並非 is_unbalance 本身,而是舊版 surname 目標編碼使用與模型訓練折「不同的」獨立 5-fold(seed=99)計算 OOF——因編碼分組(姓氏)在兩種折劃分間共享,可能造成模型訓練折的特徵值部分帶有該折驗證樣本目標的資訊,是比 is_unbalance 更值得懷疑的分數膨脹來源;改為與模型訓練折完全相同的 fold-safe 編碼後,同配方 solo LGB 分數降至 0.8937 量級。**教訓:目標編碼的 OOF 折必須與模型評估折完全相同,用不同 seed 的獨立折「看似避免洩漏」实为另一種洩漏路徑。** | 證據:s4e1, tier3 r1(is_unbalance 消融),0.893650→0.893235(Δ−0.000415);對照 STATUS.md(Feb 2026)exp#2 舊基線 0.89653。
- **s6e2(心臟病,630k 列)第三度在大樣本 AUC 上證實「移除不平衡加權」方向**:預設 vs `is_unbalance=True` 同折對照,加權後 OOF AUC 0.954993 < 不加 0.955009(Δ−0.000016)——三場(s3e3 小樣本、s4e1 165k、s6e2 630k)一致確認此先驗跨樣本規模穩健,惟量級隨樣本增大而縮小(排序指標下加權的擾動本就隨 n 變弱)。 | 證據:s6e2, 階段3.1 is_unbalance 消融,0.955009→0.954993。

### RMSLE
- RMSLE = 對 log1p 目標的 RMSE:即使目標不偏態(skew 0.019)也應以 log1p+RMSE objective 直接優化該指標,預測 expm1 還原並 clip ≥ 0。 | 證據:s3e11, exp #2/#3 均採此法;方法論本身增益小(exp #1→#2 僅 -0.00013),但為正確優化目標的前提。

### RMSE/極偏態目標
- 極右偏目標(skew 10.2)必 log1p;線性模型(Ridge)在此設定下可災難性爆炸,GBDT 穩健。 | 證據:s3e20, exp #1,Ridge RMSE 23,917.63 vs LGB 32.75(均值 baseline 155.54)。
- **目標落在離散格點 ≠ 應把預測貼回格點(對 RMSE 有害)**:s5e10 目標 accident_risk 落在 0.01 格點(僅 98 相異值),誘使套用 s3e14 的「貼齊最近訓練觀測」後處理;但同折對照顯示貼格點後 RMSE 0.056095 > 只 clip 的 0.056027——平方誤差要的是條件期望(連續均值),把連續預測硬拉到格點只增誤差。與 s3e16 取整 MAE 恰互補:**是指標(而非目標的離散外觀)決定該不該做離散化後處理**。 | 證據:s5e10, 階段3先驗檢驗B,snap 0.056095 vs clip 0.056027。

---

## 依資料型態

### 小樣本(<10k 列)
- 小樣本上 CatBoost 可能持續劣於 LGB/XGB 且被權重搜尋歸零——不要假設三模型 blend 永遠有益,先看各模型 OOF。 | 證據:s3e3(1,677 列), exp #2/#3,CAT 0.77663/0.76268 遠低於 LGB 0.81901/0.83292,兩次 blend 權重皆 LGB=1.0。
- **「CatBoost 在小資料上偏弱」的診斷結論已釐清:原生類別處理(cat_features 傳原始字串,而非 label/freq 預編碼)可大幅縮小差距,但無法逆轉排序——是「編碼方式」與「模型/資料規模不匹配」兩個獨立問題疊加,且後者才是主因。** 對 s3e3(1,677 列)重試 CatBoost,改用原生 cat_features(7 個類別欄位保留原始字串)+ `allow_writing_files=False` + 明確 thread_count,OOF 由標籤/頻率編碼版的 0.762684 大幅回升至 0.814259(+0.0516,深度 3、l2_leaf_reg 16、lr 0.05 之淺樹強正則配置最佳),證明原本落後的一部分原因確實是編碼方式不當;但即使原生處理後仍比同資料上的 LGB(0.837305)低 2.3 個百分點,加入 6-way blend 池後權重搜尋依然給予 0 權重(與未原生處理時的判定一致)。**結論:「CatBoost 在極小表格資料上結構性偏弱」才是耐用的教訓,原生類別處理是必要但不充分的緩解,不必再對同數量級資料重試此模型。** | 證據:s3e3, exp #6(CAT_native 診斷,solo 0.814259)、exp #7(6-way blend 含 CAT_native,權重搜尋仍給 0,blend 分數與未含 CAT_native 的 exp #5 在 prob-space 相同 0.837776)。
- 小樣本 + 相關性高的工程特徵 → 優先加正則化而非加容量。 | 證據:s3e3, exp #3 靠緊縮 LGB 正則拿到 +0.017;s3e9, exp #2 同理(見下)。
- 小樣本 per-fold 分數波動大(AUC 0.79–0.89)是常態,不是 bug;拿到明確增益後應「見好就收」,避免在噪音上過度搜尋。 | 證據:s3e3, STATUS.md 明示 plateau 停止準則,一輪 reflexion 後停。

### 重複列/標籤噪音
- 特徵完全重複但目標不同(~56% 列)= 標籤噪音上限:獎勵正則化、懲罰容量;預設超參的 LGB/XGB 會嚴重過擬,CatBoost 的 ordered boosting 天然抗性較強。 | 證據:s3e9, exp #1,預設參數下權重搜尋給 CAT 100%(12.54287);exp #2 正則化 LGB(leaves15/depth5/L1=2/L2=4)後 LGB 13.21→12.11。
- 正則化與特徵工程要「一起上」才有效——同樣特徵但不正則,LGB/XGB 仍過擬。 | 證據:s3e9, exp #2,blend 12.54287 → 12.07347(-3.74%),STATUS.md 明示為兩者合力。
- **反例(勿再試):「重複列群組目標平滑」(fold-safe 地把訓練目標換成訓練 fold 內同特徵群組的平均目標)對 GBDT+平方損失無效且有害**——平方損失的最小化解本來就是同輸入列的目標均值,GBDT 已隱含做了這件事;顯式平滑不添資訊,反而(a)改變群組的有效樣本權重、(b)對跨 fold 分裂的群組引入 fold-local 均值的估計噪音。三模型中兩個變差(LGB 12.11061→12.12277、CAT 12.07459→12.08207;XGB 微升 12.12086→12.11109 但不足以救 blend)。診斷「標籤噪音上限」正確,但「補救方法」已被損失函數自動內建。 | 證據:s3e9, exp #5,blend 12.07347→12.08122(+0.00775),棄用。
- **標籤噪音天花板下,seed bagging 是僅存的可靠增益方向(調參與去噪都失效後仍有效)**:Optuna 全 5-fold CV 調參 LGB(60 trials, 371.6s)單模 12.12074 反劣於手調 12.11061、權重搜尋給 0——但同一組調參超參換 seed 重訓的第 2、3 個成員卻各拿到 12.7%/12.9% 權重;再 seed-bag 權重最大的 CatBoost(77%→47.8/26.5 分拆)得到單輪最大增益。兩輪 seed bagging 貢獻了 Phase B 全部改善(12.07347→12.07143→12.07003),而 Round 1(去噪)與 Round 2(調參本身)貢獻為 0。機制解讀:貼近噪音上限時,任何「找更好的單模」都在噪音裡打轉,唯有對獨立隨機性取平均(變異數縮減)仍是免費午餐。與 s3e5 的「對直接優化最終指標的 Optuna 模型 seed-bag 無效」反例並列:seed bagging 的效力取決於成員變異數是否仍高,噪音資料上的 GBDT 變異數高,故有效。 | 證據:s3e9, exp #6→#7(+seed2 tuned LGB)12.07347→12.07143;exp #7→#8(+CAT_seed2、+seed3)12.07143→12.07003。

### 高共線性特徵
- 近完美共線欄位(r≥0.999 的 6 個 TRange 欄)可安全精簡,配合重要度探針剪枝反而更好。 | 證據:s3e14, exp #2(27 特徵)blend 341.02 → exp #3(剪至 21 特徵)340.76。
- 共線性可轉化為特徵:已知恆等式(Weight≈Shucked+Viscera+Shell, r=0.993)的殘差 `weight_resid` 是有效工程特徵。 | 證據:s3e16, exp #1,含 weight_resid 的 24 特徵組取整 blend 1.33812 vs 通用 baseline 1.35441。

### 低訊號資料(全部原始特徵 |r|≤0.11)
- 原始特徵全弱時,找「群組結構」做 fold-safe 目標編碼是最大槓桿(店面輪廓群組均值 R²≈0.06 已勝任何單一原始特徵)。 | 證據:s3e11, exp #2(方法論)0.29710 → exp #3(+store_te)0.29614,增益主要來自 store_te,遠大於方法論改動。

### 地理座標資料
- 原始經緯度線性相關弱(r≈-0.12/-0.06)但為 GBDT 重要度前二——地理訊號是非線性的,樹自己會切,直接餵原始座標 + 距離特徵即可。 | 證據:s3e1, EDA + exp #2(含 dist_nearest_city、KMeans geo_cluster)0.56166 → 0.55877。
- 在樹已能利用原始座標的前提下,再加粗粒度空間目標編碼無增益。 | 證據:s3e1, exp #3,+geo_te 後 LGB 0.56109 → 0.56102(-0.00007,噪音級,未採用)。
- **地理 target encoding 無效不代表地理特徵工程已到頂**:同一資料集上,非 target-encoding 的原始幾何特徵(KNN(k=10) 對 train+test 合併座標的平均鄰距、至海岸線錨點的最近距離)仍帶來真實增益,且增益量級明顯高於 exp #3 的噪音級結果——關鍵區別在於「target encoding 一個粗粒度分組」與「餵入更多樣的原始幾何度量」是兩種不同槓桿,前者對已能自行切分座標的樹模型是冗餘,後者提供樹難以僅從 Lat/Long 切分推導出的密度/海岸線資訊。 | 證據:s3e1, exp #6,LGB 單模 0.56109 → 0.560444(-0.00065,判定真實增益,採用);exp #7 全 pool 重訓後 blend 0.557859 → 0.557088。

### 具跨年穩定結構的時空資料
- 先驗證「歷史均值」基準:若目標對 (位置, 週次) 跨年近乎恆定(跨年 std 中位數 ≈3.1),純 location-week 歷史均值可完勝所有 GBDT。 | 證據:s3e20, exp #4,4 路權重搜尋給 TE-mean 100%、三個 GBDT 全 0;TE-mean 22.6488 vs 最佳 GBDT blend 28.3424。理解資料勝過模型複雜度。
- **歷史均值贏了之後,下一層槓桿是「對均值本身去噪」,三招依序疊加共 −6.6%(22.6488→21.1487),GBDT 每輪權重仍全 0**(s3e20 Phase B,LOYO CV 全程同折可比):
  1. **經驗貝葉斯收縮(count-sparse group means 通用)**:cell 均值觀測數極少(平衡面板下每折恰 n=2)時,向高觀測數的上層群組均值(loc 均值,~106 obs)收縮 `α·cell + (1−α)·loc`,α 網格掃出 0.935。 | 證據:s3e20, exp #5,22.6488→22.4897(−0.159)。
  2. **異常年降權(部分信任,勿全排除)**:2020 COVID 年觀測降權至 W2020≈0.29 計算歷史均值——單輪最大增益;關鍵是 U 形曲線:W2020=1.0(不處理)22.49、0.25 附近最優 21.65、0.0(全排除)22.24 反而變差,異常年仍含部分有效訊號。 | 證據:s3e20, exp #6,22.4897→21.6317(−0.858)。
  3. **鄰週(week±1)平滑**:cell 均值與同位置相鄰週均值加權平均(WNB≈0.28);窗寬 ±1 即最優,±2/±3 全變差(21.23/21.51)。**且鄰週平滑會「吸收」第 1 招:α 聯合重調後收斂到 1.0——兩招攻擊同一噪音源(cell 均值的少樣本方差),疊加冗餘,擇一到位即可(鄰週平滑更強)。** | 證據:s3e20, exp #7,21.6317→21.1487(−0.483);exp #8(4a)寬窗反例。
- **殘差診斷確認「結構飽和」邊界條件延伸到殘差層**:LGB 訓練於 (y − 去噪TE) 殘差、以 1.0/0.5/0.25/0.1 任何比例加回都變差(22.07/21.39/21.21/21.16,單調趨近 TE-only)——感測特徵連殘差都解釋不了。與 s3e19 反例合併:目標跨年近恆定時,結構不只贏 GBDT,而是吃光全部可預測訊號;不必再對此型資料嘗試「結構+GBDT 殘差」混合。 | 證據:s3e20, exp #8(4b),21.1487 → 全部 ≥21.1592,棄用。

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
- **樹搜尋(tree-search)發現的 OOF 增益可以真的轉移到真實 LB,不只是 OOF 內部比較的產物**:s3e16 樹搜尋 v2(node #15)在 OOF 上把取整後 MAE 從 1.33812 推進到 1.33563,當時只是 CV-only 的推論;重建 test 預測並實際提交後,Private LB 從 1.34075 進步到 1.33859(Public 1.34356→1.34315),雙榜皆改善,是這套樹搜尋配方第一次獲得的外部(真實排行榜)驗證。同時 CV↔LB gap 在兩次獨立提交間維持同一數量級且同方向(約 0.003–0.0075,LB 略劣於 OOF)——不僅增益本身可信,gap 的穩定性也隨提交次數增加而更受信賴。 | 證據:s3e16, exp #5(`experiments.json` leaderboard 欄位),Public 1.34315/Private 1.33859,提交 2026-07-06,對照 exp #1 的 Public 1.34356/Private 1.34075。
- 固定 seed + 固定 fold 才能跨實驗比較;revert 後重跑應可 byte-identical 重現。 | 證據:s3e9, exp #4 重現 exp #2 的 12.07347 分毫不差。

---

## 超參調校(Optuna)

- 只調「最強單模」的超參是 blend 的高性價比槓桿:LGB 50 trials(TPE)即 +0.0004 單模 / +0.0005 blend;最佳解為「淺而強正則」(depth 3、lr 0.068、reg_alpha 2.14),再次印證小/中型資料獎勵正則化而非容量。 | 證據:s3e7, exp #3→#4,LGB 0.898824→0.899215,blend 0.899395→0.899891。
- 調參預算控制:每 trial 跑完整 5-fold 會爆時間(50 trials × 5 folds ≈ 25 分鐘,一次逾時全歸零);改用「單一 fold(fold-0)代理目標」調參、只對優勝配置重跑完整 5-fold 驗證,50 trials 僅 300s,且代理排序成功遷移(fold-0 最佳配置在全 OOF 也最佳)。 | 證據:s3e7, exp #4,第一次全 5-fold 嘗試 25 分鐘逾時無結果;fold-0 代理 300.2s 完成並拿到上述增益。
- **對第二個基模型重複同一調參配方會傷 blend**:XGB 同法調參後單模微升(0.898765→0.898860)但 blend 反而退步(0.899891→0.899722)——兩個模型都收斂到相似的淺樹最優解,ensemble 多樣性下降。單模分數與 blend 貢獻不是同一回事;調完最強單模後,其餘基模型的「次優但異質」參數是資產不是負債。 | 證據:s3e7, exp #5(棄用)。
- **調參後的單模應「加入」pool 而非「替換」原成員**:Optuna 調參版 LGB 單模由 342.02154 進步到 341.68775,但若直接替換掉原 LGB,blend 反退步(340.75961→340.95316,多樣性流失,權重被迫壓到 XGB/CAT);改為「保留原 LGB 並把調參版當額外成員」的 4-way blend 則降至 340.62702。是 s3e7「別替換多樣成員」教訓對最強模型自身的推廣。 | 證據:s3e14, exp #4(替換)340.95316 vs exp #5(增列)340.62702。
- **seed bagging(同超參、換 random_state)是調參之後最便宜的殘餘增益**:再加一個 seed=2024 的原參數 LGB 作第 5 成員,blend 340.62702→340.59891;成本僅一次 5-fold 訓練(~52s)。 | 證據:s3e14, exp #7,5-way blend 權重 0.2/0.15/0.2/0.25/0.2。
- **反例:seed bagging 並非普適增益,對「Optuna 已直接優化最終指標」調校出的模型可能完全無效**——同一招對 Optuna(直接優化 post-rounder QWK)調校出的 LGB 與 CatBoost 各做一次 seed=2024 bagging,兩次權重搜尋皆給新成員 0 權重、blend 分數不變或微降。可能原因:直接優化最終指標的 Optuna 已收斂到「淺而穩定」的解(與 s3e7 fold-0 代理法找到的「淺而強正則」解型態一致),模型本身變異已低,第二個 seed 提供的多樣性有限。 | 證據:s3e5, exp #6(LGB_tuned seed-bag)0.56293→0.56293(無變化);exp #9(CAT_tuned seed-bag)0.56769→0.56716(退步,未採用)。
- **資料極小(~2k 列)時,Optuna 全 5-fold CV 本身就很便宜,不需要 fold-0 代理**:與 s3e7/s3e11 需要代理目標控制時間預算的情境相反,40 trials 的完整 5-fold CV(每 trial 皆重新做 5 折訓練)僅需 26–46 秒,直接對「後處理後的最終指標」做目標函式優化,而非先對 RMSE 代理指標調參再套用後處理。此模式僅適用於資料量與模型訓練時間都足夠小的情境;資料規模較大或訓練較慢時仍應遵循 fold-0 代理的既有配方。 | 證據:s3e5, exp #5(LGB 全 CV 40 trials,26s)、exp #8(CAT 全 CV 40 trials,46s)。
- **三個競賽(s3e7/s3e14/s3e1)一致驗證同一套「Optuna fold-proxy 調參 → 加入 pool(不替換)→ seed bagging」流程**,即使 RMSE 尺度、資料規模、特徵集完全不同:s3e1 fold-0 代理 50 trials 僅 74.3s(vs 完整 5-fold 逾時風險),調參 LGB 加為 4-way 第 4 成員後 blend 0.558768→0.557977,再加 seed=2024 版為第 5 成員後 0.557977→0.557859——流程本身可視為此類「中型表格迴歸/GBDT blend」場景的預設起手式,不必每場重新論證。 | 證據:s3e1, exp #4/#5。
- **第 4 個競賽驗證:配方遷移到「CatBoost 為最強成員」且資料放大到 360k 列仍成立**:fold-0 代理 40 trials 319s 調 CatBoost(depth/lr/l2/min_data_in_leaf/random_strength),調參版加入 pool 後 blend 0.296143→0.295781(該輪迭代最大單項);seed bagging 第 2、3 個 seed 再各得 0.295781→0.295715→0.295648。兩點新觀察:(1) 大資料上最優解是「更深+更高 lr」(depth 10、lr 0.082、min_data_in_leaf 33)而非 s3e7 小資料的「淺+強正則」——容量/正則需求隨資料量反轉;(2) 調參後 early-stopping 收斂迭代數 ~300(原參數 ~1300),單 fold 訓練反而快近一倍,調參同時買到分數與速度。 | 證據:s3e11, exp #4→#5→#6→#8。
- **調參成功後權重可能全數流向調參版 seed 家族**:s3e11 權重搜尋把 LGB 與原參數 CAT 都歸零(tuned CAT 單模 0.29579 已勝原 3-way blend 0.296143)。與 s3e7「異質成員是資產」不矛盾——把原成員留在 pool、讓權重搜尋自行裁決即可(留著零成本,勝出與否由 OOF 決定)。 | 證據:s3e11, exp #8 權重 {LGB 0, CAT_orig 0, tuned×3 = 0.4/0.2/0.4}。
- **刪除連續兩輪權重 0 的成員零代價**:s3e11 刪 XGB 後 2-way blend 分數與原 3-way 完全相同(0.296143 = 0.296143),每輪訓練省 ~25s。零權重裁決可放心執行,不必「以防萬一」保留。 | 證據:s3e11, exp #3 vs #4。
- **⚠️ 邊界條件(第一個反例):目標需要「四捨五入到整數」時,配方在 raw OOF 上仍成立但可能無法轉化為 rounded OOF 的實際增益,必須以 rounded 分數而非 raw 分數做決策**。s3e16(已有真實 LB 錨點,CV↔LB gap 僅 0.006、CV 可信)重跑同一套「fold-0 代理 Optuna 調參 LGB → 加入 pool → seed bagging」:raw OOF 確實單調變好(1.35589→1.35541→1.35533),但 rounded OOF 兩輪都變差(1.33812→1.33850→1.33893)。原因:調參版只改了 depth/lr/reg,目標函式、特徵、CV 折都相同,新成員與原 LGB 在 OOF 空間高度相關(多樣性不足),raw 分數的小幅增益不足以跨過四捨五入的離散邊界。連續兩輪未改善 rounded 分數 → 依協定停止,未提交新版。與「MAE/整數目標」一節的「取整降低 MAE」原則合併看:取整本身是穩定增益的來源,但**取整之後的模型選擇/調參決策必須重新以取整分數驗證**,不能沿用 RMSE 尺度競賽驗證過的「raw OOF 越低就選」捷徑。 | 證據:s3e16, exp #3(raw 1.35541 rounded 1.33850,劣於 1.33812)、exp #4(raw 1.35533 rounded 1.33893,更劣)。
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
- **[跨季修正,重要] 上一條「樹自己學交互、顯式交互欄無用」的先驗不是普適的**:同一先驗做同折 on/off 對照,在 s5e10(12 基礎特徵 0.056061 勝 27 特徵 0.056069)、s6e2(base13 勝 33 特徵)被**確認**,卻在 s6e1(22 特徵集 R2 勝 11 基礎,Δ+0.000094)被**否決**——該場交互欄確實帶新訊號。**教訓:這條先驗必須逐場以同折 on/off 對照檢驗後才採用,不能當普適規則直接套用;同一先驗跨場可得相反判決,是「先驗要連同適用條件一起搬、檢驗後才用」最乾淨的實例。** | 證據:s5e10 階段3檢驗A、s6e2 同檢驗、s6e1 反例(Δ+0.000094)。
- cyclical(sin/cos)編碼看情境:目標真由日曆驅動時有效(s3e19 時序,20 特徵組是最終最佳);底層日期欄與目標近零相關(r 0.003/0.008)時是純噪音,刪之反升。 | 證據:s3e19 exp #2 vs s3e7 exp #2→#3。
- 迭代停止準則:2–3 輪後增益遞減即停(s3e7 兩輪、s3e14 兩輪、s3e3 一輪 reflexion),把預算留給下一賽。 | 證據:各該賽 STATUS.md 明示停止理由。

---

## 反面教訓(試過沒用的)

- 對 AUC 目標加類別不平衡加權(scale_pos_weight/class_weights)→ 反而拖累所有模型;XGB/CAT 甚至跌破自己的無特徵 baseline。 | 證據:s3e3, exp #2,XGB 0.80511→0.79196、CAT 0.80874→0.77663;exp #3 移除後回升。
- 樹已用原始經緯度時再加粗粒度地理目標編碼 → 噪音級增益,不採用。 | 證據:s3e1, exp #3,-0.00007。
- 已有群組 target encoding(store_te)後,再加同一群組的「非目標特徵均值」(per-combo mean of sales_ratio/weight_per_case/gross_weight)→ 所有成員與 blend 全面退步(0.295715→0.296200),revert。群組鍵的目標訊號已被 TE 吃盡,同鍵的特徵彙總對樹只是冗餘+噪音;與 s3e1「粗粒度群組彙總對已能自行切分的樹是冗餘」同構。 | 證據:s3e11, exp #7(棄用)。
- 工程面:單一長程序跑多輪迭代,CatBoost 在沙箱背景執行時因預設檔案日誌(catboost_info)疑似寫檔受阻而無輸出停滯 33 分鐘,整程序被 timeout 殺掉、所有已完成訓練付諸流水。解方:每輪獨立程序 + OOF/pred npz checkpoint + `allow_writing_files=False` + 明確 thread_count,重跑全部順利。 | 證據:s3e11, Phase B iterate.py(棄用)vs iterate2.py。
- 工程面:LightGBM 預設的 timing-dependent 直方圖選擇(auto row/col-wise)會讓同參數跨進程訓練結果有微小差異,害 OOF 逐位重現閘門對不上(s6e1 初次 root 驗證差 max 0.87/樣本)。解方:所有 GBDT 訓練固定 `deterministic=True, force_row_wise=True, num_threads=<固定值>`(XGB/CAT 對應的決定性設定 + 固定 nthread),即達 max|dOOF|=0 的位元級跨進程重現;s6e1 補救(重置紀錄重生階段2-3)、s6e2 前置即用一次到位。 | 證據:s6e1/s6e2, eval_s6e1.py/eval_s6e2.py。
- 顯式交互項(有物理依據亦然)加給 GBDT → 退步,revert。 | 證據:s3e9, exp #3,12.07347→12.09483。
- 重複列群組目標平滑(fold-safe group-mean denoising)給 GBDT+平方損失 → 退步,棄用;平方損失已隱含以群組均值擬合重複列,顯式平滑只加噪音(詳見「重複列/標籤噪音」節)。 | 證據:s3e9, exp #5,12.07347→12.08122。
- 對「與目標無關的日期欄」做 cyclical 編碼與交互 → 整組特徵工程輸給 baseline;剪掉才反超。 | 證據:s3e7, exp #2 0.89788 < baseline 0.89882;exp #3 剪後 0.89939。
- Ridge stacking meta-model(僅 3 個 base OOF)→ 輸給 simplex 網格搜尋 3.3 個 MAE。 | 證據:s3e14, exp #3,344.04 vs 340.76。
- 對已收斂的 GBDT blend 輸出做 nested OOF isotonic 校準(MAE 目標)→ 強烈負向:isotonic 本身 346.99717,較未校準 blend 340.69924 惡化 +6.3 MAE,自動否決。blend 在中位數意義上已良好校準,單調重校只增變異。是繼 Ridge 之後,s3e14 上第二個敗給 simplex 網格混合的 meta/後處理法。 | 證據:s3e14, exp #6,iso 346.99717 vs blend 340.69924。
- Ridge 線性模型直上 log 轉換之極偏態目標 → RMSE 爆炸至 23,917(均值 baseline 才 155.5)。 | 證據:s3e20, exp #1。
- 序數目標 naive rounding → 比不做特徵工程的 baseline 還差。 | 證據:s3e5, exp #2,0.47191 < baseline 0.47871。
- 等權(未搜尋)ensemble → 比最佳單模差。 | 證據:s3e20, exp #2,33.21 > LGB 32.75。
- 在「目標跨年近恆定」的資料上堆 GBDT 與 63 個感測器特徵 → 全被純歷史均值淘汰(權重 0)。感測器特徵對此目標本質上是噪音。 | 證據:s3e20, exp #4,GBDT blend 28.34 vs loc-week mean 22.65。

---

## 待驗證想法(有 EDA 依據但未實測,勿當作已驗證)

- (2026-07-04 已驗證並移入「SMAPE/時序」節:s3e19 比例分解已實測,結論為反例——佔比結構被 GBDT 原生吃掉、總量外推是共同瓶頸,勿再以「佔比穩定」為由單獨重試。)

### balanced accuracy (imbalanced multiclass)
- **balanced accuracy INVERTS the AUC imbalance-weighting rule: here class balancing is MANDATORY.** balanced accuracy averages per-class recall, so the majority class is capped at weight 1/K; class_weight='balanced' (LGB) / auto_class_weights='Balanced' (CAT) / sample_weight (XGB) is essential (predict-all-majority scores only 1/K). Do NOT transfer the "remove imbalance weighting helps AUC" prior to this metric. | 證據:s6e7 (health_condition, 86/8/6% split), LGB/XGB/CAT all class-balanced -> CV balacc 0.949; blend 0.94994 -> Public LB 0.94939.
- **Per-class posterior multiplier tuned on OOF beats plain argmax for balanced accuracy.** After balanced training, grid-search a minority-class probability scale (e.g. x1.5) on OOF balanced accuracy; shifts predictions toward minorities (pred dist 0.81/0.07/0.12 vs true prior 0.86/0.06/0.08). | 證據:s6e7, blend raw->adjusted lifted balacc, adj=[1,1.5,1.5].
- **CatBoost cat_features reject NaN** — fill categorical NaN with a "missing" string first (LGB/XGB handle NaN natively; only CatBoost errors). | 證據:s6e7 CatBoostError object_idx NaN -> fixed with fillna("missing").
