# 外部想法庫(External Idea Bank,[EXT])

> **本庫定位**:此為 **[EXT]** 條目——來源是**外部文獻**(公認論文 / DOI、Kaggle 冠軍或金牌 write-up、標準教科書 / 函式庫官方文件)。
> 與 `knowledge/experience.md` 的 **[INT]** 條目(本專案跨 s3/s4/s5 競賽**實測蒸餾、每則附分數差證據**)嚴格區隔:
> - [INT] = 「我們在這些場**驗證過**什麼」(有 `競賽, exp #N, 分數A→分數B`)。
> - [EXT] = 「**文獻/社群普遍認為**有效、機制清楚、可作為樹搜尋候選節點**先驗**的技法」(未必在本專案實測過)。
>
> **用途**:供 **Phase J harness v4 的 `suggest_priors` 合併注入**。現行 `suggest_priors`(harness_v2/v3)僅讀 `experience.md`,以 `##`/`###` 標題對 `comp_meta`(metric/tags/data_type/keywords)做小寫子字串比對、回傳標題下的條列行;本庫刻意沿用「標題含可比對關鍵字(mae/auc/qwk/smape/rmse/小樣本/時序/類別…)+ 條列式內文」的結構,讓 v4 能以**同一機制**把 [EXT] 先驗與 [INT] 經驗**併池**後餵給變異提議器。注入時務必保留 `[EXT-NN]` 標記,使代理能分辨「文獻先驗」與「本場實證」。
>
> **引文誠信聲明**:本庫**每一條的出處都經 WebSearch / WebFetch 實際查證存在**(2026-07-07)。學術條目經 arXiv/DOI/期刊頁多重比對;Kaggle 頁面為前端渲染(WebFetch 僅能取回標題),但網址均解析成功且標題吻合、並有多筆搜尋結果佐證。**無任何「來源待查」或杜撰條目**。收錄準則:寧缺勿假——機制清楚且出處可點開查證者才收。
>
> 最後更新:2026-07-07。收錄 24 條。查詢方式:先按類別標題(對應指標/資料型態)找,再看「與經驗庫關係」欄判斷是**補強既有 [INT]**、**新方向(經驗庫未涵蓋)**、還是**已知(對照 [INT] 反例更要小心)**。

---

## A. 目標編碼家族(target/mean/count encoding、fold-safe、EB 收縮)

### [EXT-01] 目標編碼 + 經驗貝葉斯平滑(high-cardinality target/mean encoding with empirical-Bayes smoothing)
- **出處**:Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32. DOI: 10.1145/507533.507538. <https://dl.acm.org/doi/10.1145/507533.507538>
- **機制**:把類別欄換成該類別的目標均值,並以經驗貝葉斯 `λ·cell_mean + (1−λ)·prior` 向上層/全域先驗收縮,`λ` 隨組內樣本數上升(觀測愈少收縮愈重),抑制高基數稀有類的方差。
- **適用條件**:高基數類別欄(categorical、high-cardinality);任何指標,尤以樹模型難以自行切分的字串/ID 類欄位增益最大;小樣本更需收縮。
- **預期成本**:低(純特徵工程,一次群組聚合)。
- **與經驗庫關係**:**補強理論依據**。[INT] 已把「fold-safe 目標編碼」列為群組結構資料**最大單一增益**(s3e11 store_te、s3e20 te_locweek),且 s3e20 exp #5 實測了 `α·cell + (1−α)·loc` 的經驗貝葉斯收縮(α≈0.935,−0.159)——本條即該收縮公式的原始文獻出處。

### [EXT-02] Fold-safe / Leave-one-out 目標編碼 + 加噪(out-of-fold TE, LOO with noise)
- **出處**:Owen Zhang, "Tips for Data Science Competitions"(SlideShare,#1 Kaggler 簡報,leave-one-out 平均 + 加噪 + GBM out-of-fold 處理高基數欄)。<https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions> ;實作對照 `category_encoders.LeaveOneOutEncoder`(含 `sigma` 加噪參數)官方文件 <https://contrib.scikit-learn.org/category_encoders/leaveoneout.html>
- **機制**:計算目標均值時**排除當前列/當前折**(leave-one-out 或 K-fold out-of-fold),並對訓練特徵加入小量高斯雜訊,阻斷「目標值洩漏回自身特徵」的過擬合路徑。
- **適用條件**:高基數類別 + 目標編碼;任何指標。**折劃分必須與模型評估折完全一致**。
- **預期成本**:低。
- **與經驗庫關係**:**補強 +（加噪為)新方向**。[INT] 重度使用 fold-safe OOF 編碼,並有慘痛教訓:**目標編碼的 OOF 折必須與模型訓練/評估折完全相同**,用不同 seed 的獨立折「看似避免洩漏」實為另一種洩漏(s4e1,分數膨脹來源)。Owen Zhang 的**加噪**變體本專案尚未實測,可作候選節點。

### [EXT-03] CatBoost 有序目標統計(ordered target statistics / ordered boosting)
- **出處**:Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS 2018*, 6639–6649. arXiv:1706.09516. <https://arxiv.org/abs/1706.09516> ;NeurIPS 論文頁 <https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html>
- **機制**:以隨機排列造「歷史前綴」計算類別的目標統計(只用該樣本之前的列),並用同構的 ordered boosting 消除梯度估計中的目標洩漏(prediction shift),不需手動做 OOF 編碼。
- **適用條件**:含類別欄的表格資料;把原始字串直接傳 `cat_features`(勿先 label/freq 預編碼);對標籤噪音天然較抗。
- **預期成本**:中(訓練較 LGB 慢,需 `allow_writing_files=False` 避免沙箱寫檔停滯)。
- **與經驗庫關係**:**已知 + 補強機制**。[INT] 已驗證「CatBoost 傳原生 `cat_features` 大幅優於 label/freq 預編碼」(s3e3,0.7627→0.8143)且「ordered boosting 對重複列標籤噪音抗性較強」(s3e9,預設參數下權重給 CAT 100%);但也界定了邊界——**極小資料(~1.7k 列)上 CatBoost 結構性偏弱、權重搜尋仍給 0**。本條是其原生機制的原始文獻。

### [EXT-04] 頻率 / 計數編碼(frequency / count encoding)
- **出處**:`category_encoders.CountEncoder` 官方文件(scikit-learn-contrib)。<https://contrib.scikit-learn.org/category_encoders/count.html>(WebFetch 已確認:以組內出現次數取代類別名,支援 normalize、min-group 合併)
- **機制**:用類別的出現頻率(或計數)取代類別本身;頻率本身常帶訊號(稀有 vs 常見),且不洩漏目標、無需 fold-safe 處理。
- **適用條件**:中高基數類別欄;與目標編碼**互補**(一個編頻率、一個編目標);任何指標。
- **預期成本**:低。
- **與經驗庫關係**:**新方向 / 補充**。[INT] 僅在 s3e3 把 freq 編碼當 CatBoost 的**次佳**預處理提過(且輸給原生 cat handling),未把「頻率編碼」當獨立特徵工程槓桿系統性驗證過——可作低成本候選節點,尤其與 [EXT-01] 目標編碼並用。

---

## B. GBDT 演算法與超參調校(Optuna)

### [EXT-05] XGBoost 可擴展梯度提升樹
- **出處**:Chen, T., Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*, 785–794. arXiv:1603.02754. <https://arxiv.org/abs/1603.02754>
- **機制**:二階泰勒近似的正則化提升樹 + 稀疏感知分裂 + 加權分位數草圖;內建 L1/L2 與樹複雜度正則。
- **適用條件**:通用表格資料;所有指標(可自訂 objective)。
- **預期成本**:低—中(成熟、快速)。
- **與經驗庫關係**:**已知(三模型之一)**。[INT] 全程以 LGB/XGB/CAT 三模型 blend 為基座;本條為 XGB 原始文獻,補全先驗出處。

### [EXT-06] LightGBM(GOSS + EFB + leaf-wise 生長)
- **出處**:Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.-Y. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS 2017*. <https://www.semanticscholar.org/paper/LightGBM:-A-Highly-Efficient-Gradient-Boosting-Tree-Ke-Meng/497e4b08279d69513e4d2313a7fd9a55dfb73273>
- **機制**:Gradient-based One-Side Sampling(保留大梯度樣本)+ Exclusive Feature Bundling(綁定互斥稀疏特徵)+ leaf-wise(best-first)生長,大幅加速且常較 level-wise 準,但 leaf-wise 易過擬需以 `num_leaves`/`min_child_samples` 正則。
- **適用條件**:通用表格資料;所有指標;中大型資料的預設主力。
- **預期成本**:低(最快)。
- **與經驗庫關係**:**已知(主力模型)**。[INT] 幾乎每場以 LGB 為最強單模,並反覆驗證「小/中資料獎勵**淺而強正則**的 LGB」(s3e3 num_leaves 7、s3e7 depth 3);本條為原始文獻。

### [EXT-07] Optuna 超參最佳化框架
- **出處**:Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD 2019*, 2623–2631. arXiv:1907.10902. <https://arxiv.org/abs/1907.10902>
- **機制**:define-by-run 動態搜尋空間 + 高效取樣/剪枝(預設 TPE 取樣器 + median pruner),支援分散式與提早剪枝。
- **適用條件**:任何需調超參的模型;搭配「以最終指標(或其代理)為目標函式」。
- **預期成本**:中(視 trials × CV 折數;可用 fold-0 代理控預算)。
- **與經驗庫關係**:**已知(核心配方工具)**。[INT] 的耐用配方「**Optuna(fold-0 代理或全 CV)→ 加入 pool(不取代)→ seed bagging**」已在 s3e1/s3e3/s3e7/s3e9/s3e11/s3e14/s3e19 七場驗證;本條為 Optuna 原始文獻。

### [EXT-08] TPE(Tree-structured Parzen Estimator)取樣器
- **出處**:Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). "Algorithms for Hyper-Parameter Optimization." *NeurIPS 2011*, 2546–2554. <https://proceedings.neurips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>
- **機制**:以 `l(x)`(好結果)與 `g(x)`(壞結果)兩個密度建模,依 Expected Improvement ∝ `l(x)/g(x)` 選下一組超參——非 Gaussian-process 的序貫貝葉斯最佳化,適合高維/條件式空間。
- **適用條件**:同 [EXT-07];Optuna 的預設取樣器即 TPE。
- **預期成本**:低(取樣器本身開銷小)。
- **與經驗庫關係**:**補強理論**。[INT] 所有 Optuna 調參實際跑的就是 TPE(如 s3e3「50 trials TPE 111s」);本條說明其背後演算法,並提示「小資料全 CV 即可、不必 fold-0 代理」的成本判斷有理論依據。

---

## C. 集成:stacking / blending / rank-average / seed bagging

### [EXT-09] Stacked generalization(堆疊泛化 / stacking)
- **出處**:Wolpert, D.H. (1992). "Stacked Generalization." *Neural Networks*, 5(2), 241–259. DOI: 10.1016/S0893-6080(05)80023-1. <https://dl.acm.org/doi/10.1016/S0893-6080%2805%2980023-1>
- **機制**:用基模型的 out-of-fold 預測當**新特徵**訓練一個 meta 模型(第二層),讓 meta 學習如何組合基模型、修正各自偏差。
- **適用條件**:base 模型夠多且異質時最有效;分類/迴歸皆可。
- **預期成本**:中—高(需嚴謹 OOF、易在少 base 上過擬)。
- **與經驗庫關係**:**已知 + 重要 [INT] 反例**。[INT] 實測「**僅 3 個 base 時,Ridge/非負 stacking 明顯輸給 simplex 網格權重搜尋**」(s3e14,Ridge 344.04 vs blend 340.76);故 stacking 的適用**前提是大 base library**(見 [EXT-10]),少 base 時本專案傾向 [EXT-10] 的權重搜尋。

### [EXT-10] Ensemble selection / hill-climbing 權重搜尋(greedy forward blend)
- **出處**:Caruana, R., Niculescu-Mizil, A., Crew, G., Ksikes, A. (2004). "Ensemble Selection from Libraries of Models." *ICML 2004*. DOI: 10.1145/1015330.1015432. <https://dl.acm.org/doi/10.1145/1015330.1015432> ;作者 PDF <https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf>
- **機制**:從模型庫**貪婪前向選擇**(可重複挑同一模型 = 等效整數權重),每步加入能最大化 OOF 指標的模型;可直接對任意指標(AUC、logloss、MAE…)最佳化,並用「有放回」抑制過擬。
- **適用條件**:有多個基模型 OOF、想避免 meta 過擬時;任何指標。
- **預期成本**:低(只在 OOF 向量上搜尋)。
- **與經驗庫關係**:**補強——這是本專案主力集成法的原始文獻**。[INT] 全程用「simplex/網格 OOF 權重搜尋」且它常直接淘汰弱模型(權重=0:s3e3 CAT、s3e11 XGB、s3e20 三 GBDT 全 0),又打敗 Ridge/isotonic meta——正是 Caruana ensemble selection 的精神。本條為其學術根基。

### [EXT-11] Out-of-fold stacking + 加權 blending(Kaggle 實務配方)
- **出處**:van Veen, H.J. 等(MLWave),"Kaggle Ensembling Guide"(2015)。程式碼 <https://github.com/MLWave/Kaggle-Ensemble-Guide>(WebFetch 已確認含 voting/averaging/rankavg/geomean);Kaggle 鏡像 <https://www.kaggle.com/discussions/getting-started/106655>
- **機制**:社群整理的實務集成手冊——簡單平均、加權平均、多層 stacked generalization + OOF、幾何平均等;強調「用更多折算 OOF 比單一 holdout 穩」。
- **適用條件**:各種指標與資料型態;是「先簡單平均、再視情況上 stacking」的決策地圖。
- **預期成本**:低—中。
- **與經驗庫關係**:**已知**。[INT] 的「模型實力相近時 blend 才穩勝所有單模」(s3e1)、「未搜尋的 50/50 等權可能輸最佳單模」(s3e20 33.21>32.75)都與本指南一致;本條提供社群出處與更完整的方法選單。

### [EXT-12] Rank averaging(排名平均集成)
- **出處**:同 "Kaggle Ensembling Guide"(`kaggle_rankavg.py`)。<https://github.com/MLWave/Kaggle-Ensemble-Guide>
- **機制**:先把各模型預測轉成**名次**再平均——當模型未校準或輸出尺度/分布不同時,rank 空間平均比機率空間平均更穩健。
- **適用條件**:**排名型指標(AUC)**、或成員校準差異大時;對嚴格看機率值的指標(logloss)不宜。
- **預期成本**:低。
- **與經驗庫關係**:**已知,且 [INT] 實測為噪音級(在該場)**。[INT] 在 s3e7(AUC)比較過 prob-space vs rank-average,差異僅 ±0.000002~0.000007(噪音級)——說明 rank-average **非普適增益**,其價值取決於成員是否真的尺度不一;本條界定其適用條件。

### [EXT-13] Bagging / seed bagging(自助聚合、種子袋裝的變異數縮減)
- **出處**:Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), 123–140. DOI: 10.1023/A:1018054314350. <https://doi.org/10.1023/A:1018054314350>
- **機制**:對多個(bootstrap 抽樣 / 不同隨機種子)的高變異模型取平均,降低方差而幾乎不動偏差;對不穩定學習器(深樹)增益最大。
- **適用條件**:模型仍有可觀隨機變異時;任何指標。**成員變異數愈高愈有效**。
- **預期成本**:低—中(線性於成員數)。
- **與經驗庫關係**:**補強——seed bagging 是 [INT] 的關鍵殘餘增益**。[INT] 反覆驗證「同超參換 `random_state` 的 seed bagging 是調參後最便宜的增益」(s3e14/s3e9/s3e19 皆正),並發現其**邊界**:對「Optuna 已直接優化最終指標、收斂到淺穩解」的模型 seed-bag **可能無效**(s3e5)——恰印證本條「成員變異數低則無利可圖」的方差縮減機制。

---

## D. 驗證設計:adversarial validation / nested CV / 時序切分

### [EXT-14] Adversarial validation(對抗驗證,偵測 train/test 分布位移)
- **出處**:Zając, Z. "Adversarial validation, part one / part two"(FastML, 2016-05)。<http://fastml.com/adversarial-validation-part-one/>(WebFetch 已確認:作者 Zygmunt Zając、以 Santander 賽示範);概覽 KDnuggets <https://www.kdnuggets.com/2020/02/adversarial-validation-overview.html>;亦見《The Kaggle Book》。
- **機制**:合併 train/test、標 0/1,訓一個二分類器去分辨兩者。AUC≈0.5 → 分布一致;AUC 顯著>0.5 → 有位移,且可用該分類器對訓練列的「像 test 程度」評分,選最像 test 的子集當驗證集,或找出並處理造成位移的欄位。
- **適用條件**:懷疑 train/test 分布不同、或 CV↔LB 脫節時;任何指標。
- **預期成本**:低(多訓一個分類器)。
- **與經驗庫關係**:**新方向(經驗庫未涵蓋)**。[INT] 反覆強調「CV 方案要與 test 情境一致」(s3e19 TimeSeriesSplit、s3e20 LOYO),但**從未做過對抗驗證**來量化位移——這是一條乾淨的新候選節點,尤其適合下一階段 harness 在 CV↔LB gap 異常時觸發。

### [EXT-15] Nested cross-validation(巢狀交叉驗證,避免選擇偏誤)
- **出處**:Cawley, G.C., Talbot, N.L.C. (2010). "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." *JMLR*, 11, 2079–2107. <https://www.jmlr.org/papers/v11/cawley10a.html> ;Varma, S., Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics*, 7:91. DOI: 10.1186/1471-2105-7-91. <https://doi.org/10.1186/1471-2105-7-91>
- **機制**:外圈估效能、內圈選模型/調超參——把「模型選擇」也放進交叉驗證,避免「用同一份 CV 既調參又報分」造成的樂觀偏誤(該偏誤可能不小)。
- **適用條件**:要**誠實估計**調參後泛化、或後處理切點對 OOF 過擬風險時;小樣本尤需。
- **預期成本**:中—高(外×內折數相乘)。
- **與經驗庫關係**:**補強——[INT] 已在用其精神**。[INT] 用「巢狀(leave-fold-out)切點擬合」診斷 OptimizedRounder 對全 OOF 的過擬,並發現風險隨集成池成熟而縮小(s3e5 gap +0.0164→+0.0038);也誠實標注「fold-5 既是調參目標又佔 OOF 1/5,分數有部分樂觀」(s3e19)——本條是這類偏誤的原始理論文獻。

### [EXT-16] 淨化 + 禁運時序切分(purged & embargoed K-fold CV)
- **出處**:López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley,第 7 章(Cross-Validation in Finance)。ISBN 978-1-119-48208-6。免費摘要:<https://en.wikipedia.org/wiki/Purged_cross-validation>
- **機制**:當樣本的標籤在時間上重疊時,(a)**purge**:從訓練折剔除與驗證折標籤時間重疊的列;(b)**embargo**:在驗證折之後再留一小段緩衝不用於訓練——雙管齊下阻斷時序洩漏,避免走勢外推被高估。
- **適用條件**:時序 / 面板資料、標籤具時間跨度、test 為未來期(SMAPE/時序);比單純 TimeSeriesSplit 更嚴格。
- **預期成本**:中(需標籤時間區間資訊)。
- **與經驗庫關係**:**補強 + 新變體**。[INT] 已重度使用時序感知 CV:「test 為純未來期 → TimeSeriesSplit(對唯一日期切)」(s3e19,隨機 KFold 高估 2.4 倍)、「多年度時空 → Leave-One-Year-Out」(s3e20)。**purge/embargo 這個更嚴格的變體本專案未試**,可作時序場的候選升級。

---

## E. 特徵工程:交互 / 聚合 / 高基數表示

### [EXT-17] Deep Feature Synthesis / 自動聚合特徵(aggregation across relations)
- **出處**:Kanter, J.M., Veeramachaneni, K. (2015). "Deep Feature Synthesis: Towards automating data science endeavors." *IEEE DSAA 2015*, 1–10.(演進為開源 Featuretools)。函式庫 <https://pypi.org/project/featuretools/>
- **機制**:沿實體關係自動堆疊聚合原語(count/mean/std/max/min/mode…)與轉換原語,系統化生成群組統計/跨表聚合特徵,取代人工枚舉。
- **適用條件**:有群組鍵、多表 / 交易型資料;數值 + 類別混合。
- **預期成本**:中(特徵爆炸需配特徵選擇)。
- **與經驗庫關係**:**新方向,但務必對照 [INT] 反例**。[INT] 驗證「群組**目標**編碼(TE)是最大增益」(s3e11 store_te),卻也發現「**已有 TE 後,再加同群組鍵的非目標特徵均值 → 全面退步、revert**」(s3e11 exp #7,群組目標訊號已被 TE 吃盡)與「粗粒度群組彙總對已能自行切分的樹是冗餘」(s3e1)。故自動聚合須避免與既有 TE 撞鍵、並嚴格用重要度/OOF 裁決。

### [EXT-18] Entity embeddings(高基數類別的神經嵌入)
- **出處**:Guo, C., Berkhahn, F. (2016). "Entity Embeddings of Categorical Variables." arXiv:1604.06737(Rossmann Store Sales 第 3 名方案)。<https://arxiv.org/abs/1604.06737>
- **機制**:用神經網路把每個類別值學成低維連續向量,語意相近的類別在嵌入空間相鄰;可餵回 GBDT 或直接在 NN 內用,對高基數特別有效、抗過擬。
- **適用條件**:**高基數類別**、且資料量足以訓 NN;數值嵌入亦可回饋給樹模型。
- **預期成本**:高(需訓 NN、調嵌入維度)。
- **與經驗庫關係**:**新方向(NN-based,[INT] 為純 GBDT 基座)**。[INT] 目前完全以 GBDT + 目標/頻率編碼處理類別,未用學習式嵌入;本條是高基數類別的一條「新方向」候選,成本較高,適合已被 TE/GBDT 榨過仍想突破的場。

---

## F. 後處理:機率校準 / 切點優化

### [EXT-19] 機率校準之 Platt scaling(sigmoid 校準)
- **出處**:Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods." *Advances in Large Margin Classifiers*, MIT Press。實作與 CV 校準見 scikit-learn `CalibratedClassifierCV` 官方文件 <https://scikit-learn.org/stable/modules/calibration.html>(WebFetch 已確認:sigmoid/isotonic + 交叉驗證避免偏誤);概念頁 <https://en.wikipedia.org/wiki/Platt_scaling>
- **機制**:把模型分數 `f` 過一個以 logistic 回歸擬合的 sigmoid `1/(1+exp(Af+B))`,得到校準機率;參數少、抗過擬,適合校準資料量小時。
- **適用條件**:**分類 + 看機率值的指標**(logloss、Brier);boosted trees 常呈 sigmoid 形失真尤其受益。**校準器須在獨立折/CV 上擬合**。
- **預期成本**:低。
- **與經驗庫關係**:**新方向(分類機率校準)+ 注意 [INT] 迴歸反例**。[INT] 目前的後處理集中在迴歸/序數(取整、snap、OptimizedRounder),**未做過分類機率校準**;但有相關警訊:對已收斂 blend 做 isotonic(見 [EXT-20])在**迴歸 MAE** 上強烈負向(s3e14)。Platt 校準是分類 + logloss 場的候選,勿套到迴歸中位數已良好校準的情形。

### [EXT-20] 機率校準之 Isotonic regression(保序回歸校準)
- **出處**:Zadrozny, B., Elkan, C. (2002). "Transforming classifier scores into accurate multiclass probability estimates." *KDD 2002*, 694–699. DOI: 10.1145/775047.775151. <https://dl.acm.org/doi/10.1145/775047.775151> ;比較研究 Niculescu-Mizil, A., Caruana, R. (2005). "Predicting good probabilities with supervised learning." *ICML 2005*. <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
- **機制**:擬合一條**單調非遞減**的分段常數函數把分數映到機率,非參數、比 Platt 更彈性,但需較多校準資料、少資料易過擬。
- **適用條件**:分類 + 機率指標、校準資料充足;Niculescu-Mizil & Caruana(2005)實證 boosted trees/SVM 校準後顯著改善。
- **預期成本**:低—中。
- **與經驗庫關係**:**已知反例 + 澄清適用條件**。[INT] 對已收斂的 **GBDT 迴歸 blend** 做 nested isotonic(MAE 目標)得到**強烈負向**(s3e14 exp #6,346.997 vs 340.699,自動否決)——因 blend 在中位數意義上已良好校準。本條澄清:isotonic 的正確戰場是**分類機率**(logloss/Brier),而非迴歸點估計。

### [EXT-21] OptimizedRounder — QWK 切點優化(Nelder-Mead threshold search)
- **出處**:Kaggle "PetFinder.my Adoption Prediction"(2019)競賽普及之後處理法(迴歸輸出 + 以 `scipy.optimize`(Nelder-Mead)搜切點最大化 QWK)。競賽頁 <https://www.kaggle.com/c/petfinder-adoption-prediction> ;QWK 指標與切點說明 kernel <https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps>
- **機制**:序數目標用**迴歸頭**得到連續分數,再以無梯度最佳化(Nelder-Mead)搜尋 K−1 個切點,把連續分數切成序數類、直接最大化訓練/OOF 上的 QWK,取代對稱四捨五入。
- **適用條件**:**序數目標 + QWK**;類別不平衡時尤其關鍵(對稱取整假設崩潰)。
- **預期成本**:低(切點搜尋很快)。
- **與經驗庫關係**:**已知——[INT] 已重度驗證,本條補外部原始出處**。[INT] 把「迴歸 + OptimizedRounder(對 OOF QWK 調切點)」列為**小資料序數題單一最高槓桿**(s3e5,naive round 0.4719→優化切點 0.5269,+0.055),並進一步發現「把 Optuna 目標函式直接設為後處理後的 QWK」更強。本條給出該技法在社群的起源(PetFinder)與 Nelder-Mead 實作。

---

## G. 特徵重要度 / 冠軍組合拳 / 綜合參考

### [EXT-22] Permutation feature importance(置換特徵重要度,穩健剪枝依據)
- **出處**:Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. DOI: 10.1023/A:1010933404324(置換重要度出處)。<https://link.springer.com/article/10.1023/A:1010933404324> ;方法整理 <https://christophm.github.io/interpretable-ml-book/feature-importance.html>
- **機制**:訓練後**隨機打亂單一特徵**、量測驗證指標的退化幅度作為該特徵重要度;不依賴模型內部分裂計數,較 GBDT 內建 gain/split importance 不易被高基數/共線欄誤導。
- **適用條件**:特徵選擇 / 剪枝;高共線性或高基數欄尤需(內建重要度會偏)。
- **預期成本**:低—中(每特徵重評一次驗證)。
- **與經驗庫關係**:**補強——[INT] 已用重要度剪枝但用內建版**。[INT]「特徵過多反而退步 → 精簡是常規武器」以重要度探針剪枝(s3e14 剪至 21 特徵 340.76、s3e7 剪噪音特徵反超)——置換重要度是更抗偏誤的剪枝依據,可作既有剪枝步驟的升級變體,尤其在 s3e14 那種近完美共線(r≥0.999)場景。

### [EXT-23] 去噪自編碼器表示學習(denoising autoencoder,冠軍級表格方案)
- **出處**:Jahrer, M. (2017). "1st place solution — Porto Seguro's Safe Driver Prediction (Representation Learning)." Kaggle write-up。<https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/writeups/michael-jahrer-1st-place-with-representation-learn>(WebFetch 確認頁面標題「1st place with representation learning」)
- **機制**:對數值特徵加噪、訓 denoising autoencoder 學出更好的**表示**(隱藏層激活),再把該表示餵給下游 NN(冠軍方案為 1×LGB + 5×NN 的 blend);在 XGBoost 未奪冠的罕見表格賽中勝出。
- **適用條件**:純數值 / 已編碼表格、資料量大、願投入 NN;通常需與 GBDT blend。
- **預期成本**:高(訓 DAE + 多個 NN,調參重)。
- **與經驗庫關係**:**新方向(表示學習,[INT] 純 GBDT)**。[INT] 尚無任何自編碼器/表示學習條目;這是最著名的「GBDT 不夠、表示學習翻盤」表格賽先例,列為高成本、高風險的突破型候選,適合經驗庫與樹搜尋兩層都榨過仍想找系統性增益的場。

### [EXT-24] The Kaggle Book — 表格競賽綜合手冊(交叉索引)
- **出處**:Banachewicz, K., Massaron, L. (2022). *The Kaggle Book: Data analysis and machine learning for competitive data science.* Packt。ISBN 9781801812214。<https://www.packtpub.com/en-us/product/the-kaggle-book-9781801812214>
- **機制**:兩位 Kaggle Grandmaster 系統整理的競賽方法論——涵蓋驗證設計、對抗驗證、特徵工程、集成/stacking、超參調參、各資料型態(表格/影像/文字)配方,並附各賽 Master 訪談。
- **適用條件**:作為 [EXT-09~14]、[EXT-01~04] 等條目的二手交叉索引與情境判斷指南;不替代原始論文。
- **預期成本**:—(參考資料)。
- **與經驗庫關係**:**綜合補強**。本書把上述多條 [EXT] 技法(對抗驗證、目標編碼、ensembling、denoising autoencoder 案例)整合成可操作流程,可在 harness 需要「該用哪一招」的情境判斷時作為人類可讀的對照;與 [INT] 的差別在於 [INT] 是**本專案實測分數差**、本書是**社群共識配方**。

---

## 附:與 [INT] 的重疊 / 互補一覽(供 v4 併池去重參考)

- **已被 [INT] 實證的先驗(注入時應與 [INT] 證據並列,避免重複觸發)**:EXT-01(EB 收縮=s3e20)、EXT-02(fold-safe=多場)、EXT-03(CatBoost 原生=s3e3/s3e9)、EXT-05/06(LGB/XGB=基座)、EXT-07/08(Optuna/TPE=核心配方)、EXT-10(權重搜尋=主力)、EXT-13(seed bagging=關鍵殘餘增益)、EXT-15(nested=切點診斷)、EXT-16(時序切分=s3e19/s3e20)、EXT-21(OptimizedRounder=s3e5)、EXT-22(重要度剪枝=s3e14)。
- **[INT] 有反例、[EXT] 界定其正確適用條件(注入時務必連同反例)**:EXT-09(stacking:少 base 輸權重搜尋 s3e14)、EXT-11/12(等權/rank-average 非普適 s3e20/s3e7)、EXT-17(群組非目標彙總在已有 TE 後冗餘 s3e11)、EXT-20(isotonic 在迴歸 MAE 反效果 s3e14)。
- **經驗庫尚未涵蓋的乾淨新方向(最有價值的候選節點)**:**EXT-14 對抗驗證**、**EXT-04 頻率編碼(獨立槓桿)**、**EXT-16 purge/embargo(時序升級)**、**EXT-18 entity embeddings**、**EXT-19 分類機率校準(Platt)**、**EXT-23 denoising autoencoder**。
