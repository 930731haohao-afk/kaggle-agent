# 與 Aygün 等人 ERA 系統的方法對照

本文把本專案的 Kaggle 混合式 agent,與 **Eser Aygün 等人(2026,Nature;預印本 arXiv:2509.06503v3,78 頁)** 的 **ERA(Empirical Research Assistant,Google DeepMind)** 系統做逐項對照:方法、benchmark、結果、嚴謹度,並**誠實標出本專案「不是」什麼、以及哪裡反而更嚴謹**。

> 引用標記:ERA 側標 `[ERA p.N]` 為預印本頁碼;本專案側標 `檔案:行` 為 repo 內位置。所有數字均可回溯。

---

## 1. 一句話定位

- **ERA** 是一套**自動化系統**:給定可機器評分的任務,**LLM 直接寫 Python 程式碼**,在沙盒執行評分,再由**樹搜尋**在「程式碼空間」爬坡;外部研究想法透過**注入 prompt** 引導 LLM 寫出特定做法。[ERA p.2, p.13]
- **本專案** 是一套 **Claude Code Skill + 人在迴圈**的競賽流程,外加一支**自建樹搜尋**;搜尋是在「**固定 GBDT 家族(LGB/XGB/CAT)的組態空間**」做手寫組合搜尋,**LLM 不在變異迴圈內**。`docs/TECH_REPORT.md:28-34`、`docs/PREFACE.md:94-101`

兩者共用同一組「兩支柱」語言(**LLM + 樹搜尋**;第二支柱=外部想法整合),本專案的階段 4/5 正是照這兩支柱設計來做**受控消融**。但如下所述,兩者在「LLM 扮演什麼角色」上有本質差異,對照時必須講清楚,不能暗示等價。

---

## 2. 方法對照

### 2.1 搜尋引擎

| 面向 | ERA | 本專案 |
|---|---|---|
| 搜尋型態 | PUCT 樹搜尋(作者自稱更接近 **Flat-UCB** 而非 AlphaZero;無 rollout,從全樹節點取樣展開)[ERA p.13] | 候選樹 + 回溯的組合搜尋(harness_v2/v3/v4);breadth-then-depth 預算相位機 + explore burst + auto-stop `docs/PREFACE.md:131-137` |
| 一個節點是 | **一段被執行並評分的 LLM 產生程式碼** [ERA p.13] | **一個組態**:solo(單模型+單超參,完整 K-fold)或 blend `docs/PREFACE.md:94-96` |
| 變異算子 | **LLM 依父節點 code+score 生成一個新的子程式碼**;隨機性來自 LLM 取樣 [ERA p.13-14] | `propose_child` / `select_next_parent` 走 `SOLO_SEEDS` 種子集 + mutation 佇列;boundary-push 把調參範圍邊界外推一格 `docs/phase_j_j3_findings.md:37-39` |
| 集成 | 由 LLM 在程式碼內自行決定 | **OOF 權重搜尋**(blend = solo 的 OOF 加權平均,dirichlet 權重)、mega-blend(全 solo 集成)`docs/PREFACE.md:86,101` |
| 用的 LLM | 全實驗 **Gemini 2.5 Flash**(換 Pro 提升有限)[ERA p.13] | Claude Code(但**僅**在流程外做 EDA/特徵/決策與**手寫種子**,不在搜尋迴圈內產生程式碼) |
| 調參常數 | `c_puct = 1`,在 Kaggle benchmark 上調出 [ERA p.13] | 無對應超參;預算以節點/相位控制 |

**核心差異(必須明說)**:ERA 的樹搜尋是在**由 LLM 生成的任意程式碼**上爬坡——變異算子就是「LLM 重寫程式」。本專案的樹搜尋是在**人預先定義好的模型家族與調參空間**上做組合/權重搜尋——變異算子是手寫的 `propose_child`。因此本專案階段 4 對齊的是 ERA 的「**以樹搜尋取代線性單路徑**」這個**結構性想法**,**不是**「LLM 自動寫程式」的能力。這點在報告與對外說明時不可含糊。

### 2.2 外部想法注入(第二支柱)——最關鍵的對照

| 面向 | ERA | 本專案 |
|---|---|---|
| 注入點 | **prepend 到「LLM 產生程式碼」的 prompt**;LLM 隨即寫出遵循該想法的程式碼 [ERA p.14] | 併入 `suggest_priors` 回傳的 `tree['priors']`(harness_v4 的 `suggest_priors_v4`)`docs/phase_j_j3_findings.md:15-22` |
| 想法來源 | 專家建議、**高被引論文/教科書的 LLM 摘要**、Deep Research、AI co-scientist [ERA p.3,5,14] | 外部想法庫 `knowledge/idea_bank.md`(`[EXT]`);內部經驗庫 `knowledge/experience.md`(`[INT]`)`README.md:48-51` |
| 是否真的被消費 | **是**——注入點就是變異算子的輸入,直接左右每次生成的程式碼 [ERA p.12,14] | **否**——`tree['priors']` 是 **write-only**,只供列印/記錄;`SOLO_SEEDS`/mutation 佇列/`propose_child`/`select_next_parent`/評估器全部不讀它 `docs/phase_j_j3_findings.md:5-42` |
| 實證 | 文獻引導的 ERA 產出 BBKNN(TS),較最佳已發表法 **+14%**;44%(24/55)重組解勝過雙親 [ERA p.5] | 單變數 A/B(s3e3):`off`([INT]) vs `ext`([INT]+2 條淨新 [EXT])→ 全域最佳同為節點 #11、AUC **0.841442 完全相同**、22 節點**逐位元相同**(Δ=+0.00000000)`docs/phase_j_j3_findings.md:5-42` |
| 想法重組 | LLM 把兩個父方法的原理口語化後生成新 prompt(概念空間重組)[ERA p.15] | harness_v4 有重組 mutation(合成兩高分節點核心概念),但因上述消費端未接線,對搜尋同樣無效 |

**結論**:ERA 的第二支柱**是負重的**——注入的文字直接是變異算子(LLM prompt)的輸入。本專案的第二支柱在**現行架構下是「構造性 no-op」**:注入機制(池合併/去重/閘門)本身正確,但**消費端沒接線**。經驗庫知識**確實**引導了搜尋,但**是透過作者手寫種子**(作者讀 experience.md、把先驗手寫進 seed 組態;`[PRIOR Pk]` 是手寫字串常數,`prior_usage_summary` 只 regex 解析、不回溯 `tree['priors']`)。這是本專案與 ERA 最大的方法落差,已誠實記錄於 `docs/phase_j_j3_findings.md` 與 `docs/TECH_REPORT.md:90-96`。**根因**:本專案搜尋迴圈裡沒有「LLM 產生程式碼」這個算子,自然沒有可供注入的 prompt;要讓注入負重,得先把 `tree['priors']` 接進 `propose_child`/種子邏輯,或引入 LLM-in-loop 的程式碼變異。

---

## 3. Benchmark 對照

| 面向 | ERA | 本專案 |
|---|---|---|
| Kaggle 場次 | **2023 賽季(S3)16 場** Playground(迴歸+分類)[ERA p.4, Table S2 p.57] | **15 場**:10 場同季 S3 + 5 場跨季 S4–S6 `docs/benchmark_facts.json:4-7` |
| 評分方式 | **直接提交 Kaggle 真實排行榜**,指標=**公開榜人類百分位** [ERA p.4] | 幾乎全為**本地 OOF(5-fold)**;**僅 s3e16 有真實 LB**,其餘 14 場已關榜(403)只有 CV `docs/TECH_REPORT.md:52-55`、`docs/SUMMARY_REPORT.md:68-71` |
| 對照基準 | Single(1 次呼叫)、Best-of-1000、AIDE、TS、TS+Expert、TS+BDT [ERA p.3] | 階段 1(無 skill 基線)→2→3→4→5 的自身消融階梯 `docs/PREFACE.md:104-143` |
| Kaggle 以外 | scRNA-seq、COVID 預測、GIFT-Eval、geospatial、ZAPBench、數值積分 6 個科學領域 [ERA p.5-9] | 無(僅表格競賽) |

**強對齊點**:本專案的 10 場 S3(s3e1/3/5/7/9/11/14/16/19/20)**全部落在 ERA 的 16 場名單內**;本專案拿不到的 6 場(s3e2/4/8/10/12/17,帳號無法接受已關閉競賽規則、下載 403)**正好是 ERA 有、本專案缺**的那幾場。因此本專案的 Kaggle benchmark 可視為 **ERA 完全相同 benchmark 的 10/16 子集**,只是評分改為本地 CV。

**關鍵落差**:ERA 用**真實榜的人類百分位**衡量「達到專家水準」;本專案除 s3e16 外**沒有**真實榜、也**沒有**人類百分位校準。因此本專案能誠實宣稱的是「**每加一項能力,受控 CV 分數階梯式改善**」,**不能**宣稱「達到人類第 N 百分位」。

---

## 4. 結果對照

| | ERA | 本專案 |
|---|---|---|
| Kaggle 主結果 | 16 場平均公開榜百分位:Single≈34% → Best-of-1000≈49% → AIDE≈56% → **TS≈64% → TS+Expert≈68% → TS+BDT≈71%**(讀自 Fig.1b,無數值表)[ERA p.3-4] | 15 場**階段 1→4 階梯場場成立**(9 升 1 平,無任一階整體倒退)`docs/TECH_REPORT.md:59-60` |
| 增幅型態 | TS 明顯優於單次與 best-of-1000,亦勝 AIDE [ERA p.4] | 增幅集中在**結構性洞見**場(s3e20 **+25.7%**、s3e5 **+19.2%**),接近天花板的場 <1% `docs/SUMMARY_REPORT.md:131` |
| 真實榜證據 | 全 benchmark 真榜;scRNA **40 個**方法勝過人類榜首、COVID **14 個**模型勝過 CDC 集成 [ERA p.1,5,7] | **僅 s3e16**:樹搜尋版兩榜同向勝過線性冠軍(Public 1.34356→1.34315、Private 1.34075→1.33859)`docs/SUMMARY_REPORT.md:68-85` |
| 泛化 | 跨 6 科學領域皆達專家水準 [ERA p.5-9] | 跨季 5 場(5 種指標)階梯仍成立,增幅收斂 `docs/TECH_REPORT.md:68` |

**量級誠實話**:ERA 的成果是「在真實榜上勝過人類/官方集成」的**跨領域、大規模**結果;本專案的成果是「**在 10/16 相同 benchmark 的子集上,以受控消融證明每項自主能力的邊際貢獻**」,增幅多為 0.0x%(僅結構性場顯著),且第五支柱貢獻**恰為零(構造性)**。兩者是**不同量級、不同宣稱**的工作——本專案的價值在**方法論的乾淨與可回溯**,不在絕對戰績。

---

## 5. 消融與統計嚴謹度對照

| 面向 | ERA | 本專案 |
|---|---|---|
| 核心消融 | TS vs Best-of-N(N=1000/128):5 個 LLM 中 4 個、兩問題上 TS 勝(唯 GPT-5 在 batch integration 已飽和)[ERA p.9,15] | 5 階段階梯 + 每階子階段;跨場 tier1→tier4 對照 15 場 `docs/SUMMARY_REPORT.md` |
| 注入消融 | 44%(24/55)重組勝雙親;無建議只到 ComBat 水準、文獻引導達 BBKNN 榜首 [ERA p.5,29] | 第五階段單變數 A/B = **逐位元相同**(構造性 no-op)`docs/phase_j_j3_findings.md` |
| 顯著性檢定 | **無**;主結果以長條圖與名次呈現,未報信賴區間 | **paired bootstrap(B=5000)**:整條 2→4 階梯 5 場全部 CI 排除 0;樹搜尋單步僅在大樣本(n≈52–63 萬)顯著,小樣本落在 OOF 噪音內 `docs/statistical_rigor.md:66-82` |
| 重現閘門 | 有沙盒重跑;採遵性人工稽核(幾乎全「遵循」注入算法)[ERA p.34] | **OOF 逐位重現閘門**:最佳解成員從頭重訓、逐位比對 OOF 才產提交;部分達 max\|ΔOOF\|=0 位元級 `docs/TECH_REPORT.md:45-48` |

**本專案反而更嚴謹之處(可對老師說的亮點)**:ERA 的 Kaggle 主結果只給長條圖、**沒有信賴區間**;本專案對每一階轉換做了 **paired bootstrap 顯著性**,而且**誠實把「階梯場場成立」校準成「信累積增益、別過度宣稱單一 0.0x% 步」**(`docs/statistical_rigor.md`)。這種「小步是否只是噪音」的自問自答,是本專案方法論上的獨立貢獻。**但範圍限制**:bootstrap 目前只涵蓋 5 場跨季(有 OOF 快取),10 場 S3 缺快取尚未納入;且 CI 只量 OOF 抽樣變異,不含重跑搜尋(seed/fold/Optuna)變異,是**下限**。

---

## 6. 誠實差距與定位總結

**本專案「不是」什麼**(避免對外暗示等價):
1. **不是**「LLM 自動寫程式」系統——搜尋在固定 GBDT 組態空間,LLM 不在變異迴圈內。
2. 第二支柱(外部注入)**在現行架構下不負重**——ERA 的注入是核心機制,本專案的是 no-op(靠手寫種子引導)。
3. **不是**真實榜/人類百分位驗證——除 s3e16 外全為本地 CV,無「達到專家水準」的外部尺標。
4. 規模與領域遠小於 ERA(單一表格領域、單一 driver、手寫先驗種子)。

**本專案的真實價值**:
1. 在 **ERA 完全相同 benchmark 的 10/16 子集**上,以**受控消融**逐段隔離「skill 化流程 / 線性自我迭代 / 樹搜尋」各自的邊際貢獻——這是 ERA 論文**沒做到的細粒度歸因**。
2. **統計嚴謹度**(paired bootstrap + 位元級 OOF 重現閘門)在方法論上**比 ERA 主結果更保守可信**。
3. 對第二支柱做出並誠實記錄了一個**構造性負面發現**(注入 no-op),把「知識有引導」與「自動 plumbing 被消費」兩種宣稱清楚切開——這正是科學誠信的展現。

**一句話**:本專案不是 ERA 的復刻或縮小版戰績挑戰,而是**借用 ERA 的兩支柱框架,在其相同 benchmark 的子集上,做一次乾淨、可回溯、統計誠實的能力歸因研究**;並在過程中誠實界定了自建系統與 ERA 的本質差距。

---

## 附錄:待接線後可真正對齊 ERA 第二支柱的路徑

若要把「外部想法注入」從 no-op 變成負重(真正對齊 ERA pillar 2),最小改動有二選一:
1. **接線既有先驗**:把 `tree['priors']` 實際接進 `propose_child`/`SOLO_SEEDS`,讓注入的 `[EXT]` 想法改變候選生成(仍在 GBDT 組態空間內,量得到邊際值但不等於 ERA 的 code-mutation)。
2. **引入 LLM-in-loop 程式碼變異**:在搜尋迴圈內讓 LLM 依 prompt(含注入想法)生成/改寫特徵或建模程式碼——這才是與 ERA 同構的做法,但工程量與算力顯著較大,且 arm64 環境有套件風險。

方向題(接線 vs 接受誠實負面結果)留待決定,見 `docs/phase_j_j3_findings.md` 結語。
