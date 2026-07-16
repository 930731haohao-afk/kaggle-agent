# 筆記:樹搜尋 harness v1 → v4 的演進(詳版)

> **一句話總綱**:**每一版都不動前一版**(以 `import` 繼承前一版、只覆寫/新增),因此每版都是前一版的**嚴格超集**。
> 好處有二:①舊實驗永遠**逐位可重現**;②新功能能被**單獨歸因**(把新開關關掉 = 回到前一版的行為)。四版對應四個不同的問題。
>
> **配色慣例**(對應四張流程圖):灰 = 繼承自前版、不變;綠 = v2 新增;橘 = v3 新增/改良;紫 = v4 新增。
>
> **流程圖**(drawio 匯出):
> [v1](tree-search-v1-flowchart.drawio.png) ·
> [v2](tree-search-v2-flowchart.drawio.png) ·
> [v3](tree-search-v3-flowchart.drawio.png) ·
> [v4](tree-search-v4-flowchart.drawio.png)。
> 架構總圖:[tree_search_architecture.png](tree_search_architecture.png)。

---

## 一眼版

| 版 | 檔 | 行數 | 核心 = 新增了什麼 | 回答的問題 | 用在 |
|---|---|---|---|---|---|
| **v1** | `harness.py` | 252 | 最小候選樹骨架(節點/血脈/選擇/回溯/崩潰安全) | 「線性單路徑 → 搜尋樹?」 | s3e9 / s3e14 / s3e5 |
| **v2** | `harness_v2.py` | 376 | 集成一等公民 + 4 建議(blend、離散指標 plateau、去重、經驗先驗) | 「blend 怎麼放進樹?離散指標怎麼辦?」 | s3e3 / s3e7 / s3e1 / s3e19 / s3e11 |
| **v3** | `harness_v3.py` | 856 | 預算相位機 + 9 件生產化(收尾/防空轉/防卡死/resume/邊界外推) | 「跑久了會空轉、卡死、resume 壞掉」 | 跨季 5 場 + 部分 S3 回填 |
| **v4** | `harness_v4.py` | 540 | 外部想法注入 `[EXT]` + 雙親重組(ERA 第二支柱) | 「內部經驗用盡,怎麼引外部想法?」 | 階段 5 注入 / 重組 |

> 三個共通不變的骨幹(v1 定義、v2–v4 全繼承):**節點 schema、lineage/選擇/plateau 回溯規則、崩潰安全 save/load**。

---

## v1 — 最小骨架(`harness.py`)

**是什麼**:一棵「候選解的搜尋樹」。**節點(Node)= 一個完整、已評估(或失敗)的解**;**血脈(lineage)= 從根長出的一條分支**(根的每個直接子節點各自開啟一條血脈 = 一個起始方向)。

**資料結構**
- 節點欄位:`{id, parent_id, mutation, config, score, status, wall_s}`
  - `mutation` = 一行人類可讀的「相對父節點改了什麼」;`config` = 該解的完整設定;`status` ∈ `evaluated|failed`。
- 樹 = JSON,**每次 `add_node` 之後立刻存檔**(原子寫:temp file + `os.replace`)→ 崩潰後可從磁碟 `load` 精確續跑。

**主迴圈(見流程圖 v1)**
1. **driver 播種第一代血脈**:直接對 root 加數個子節點 = 數個起始方向(`select_next_parent` 只負責「已有血脈之後」要續拉哪一條)。
2. **driver 預算用盡?** — v1 **沒有內建停止規則**,節點數由 driver 控制;是 → 輸出 `global_best`(分數最低的已評估節點)。
3. **`select_next_parent`**:取「**分數最佳(最低)的未 plateau 血脈**」為 active lineage;在其中選「**分數最佳、且自身子節點 < `MAX_CHILDREN_PER_NODE`(=3)**」的節點當父節點。
4. **driver/agent 提出突變**(一行描述 + 新 config)。
5. **`evaluate(config)`**:K-fold CV → `score`(全 harness 採「**越低越好**」慣例;maximize 指標請先翻符號)或 `status=failed`。**每場競賽只需提供這一個函式。**
6. **`add_node` + save**:插入前先比對 global best — **有改善 → streak 歸 0;沒改善 → streak+1**。
7. **`streak ≥ 3`?** 是 → **把該血脈標為 plateaued(回溯)**,記進 `backtrack_log`,下次 `select` 改挑次佳血脈;否 → 回到步驟 2。

**兩個 dead-end 護欄(文件化的延伸)**
- ① 一條血脈的擴充預算(全節點都滿 3 子)用盡 → **強制 plateau(強制回溯)**。
- ② **所有血脈都 plateaued** → 把 plateau 旗標**一次清空(reopen)**,朝節點預算繼續跑,避免提早死掉。

**限制**:`kind` 埋在 `config` 裡,**沒有一等的 blend 支援**;沒有停止規則、去重、離散指標處理——這些正是 v2/v3 要補的。

---

## v2 — 集成一等公民 + 四項 Stage-4 建議(`harness_v2.py`)

> 來源:prototype 報告的「已知缺口 → 建議」四條。**選擇 / plateau 回溯規則原樣繼承 v1**;v2 只加下列(流程圖綠色節點)。

**① Ensemble-default node space —— blend 升為一等公民**
- 節點多一個一等欄位 `kind ∈ {solo, blend}`(v1 埋在 config 內)。
- **`eval_solo(config) → (score, OOF)`**;`cache_oof()` 把 OOF/test 預測**原子快取**到 `solo_<id>.npz`。
- **`eval_blend(members)`**:`load_oof()` 載入成員 OOF 矩陣 → **Dirichlet 權重搜尋(k=1500,coarse+fine 兩輪)** 或 grid-simplex 搜尋。
- **鐵則**:任何後處理(取整 / OptimizedRounder / snap / clip)**必須寫在 `metric_fn` 內**,才會對「每一組候選權重」都套用,而非只套在贏家上——否則離散化指標下權重搜尋會選錯。

**② Metric-aware plateau —— 離散指標不再誤判卡關**
- `tie_rate(tree)` 量測所有已評估分數中的**完全重複比例**。
- `tie_rate > 0.15`(分數面被離散化,如 QWK-after-rounder)時:(a) plateau 門檻 **3 → 5**;(b) 與 global best **完全平手**的子節點視為**中性**(既不重置也不累加 streak,因為離散面上平手是常態、非卡關訊號)。
- 門檻以下(連續指標幾乎不會平手)→ 行為與 v1 完全相同。

**③ Child dedup —— 擋掉重複配方**
- `config_hash`(排序後、與 key 順序無關的穩定雜湊)。`add_node` 入口若撞到既有節點 → **拒絕、回傳 `(None, dup_id)`**,明確告訴提議器「這個配方已存在,換一個」。
- 修的是 s3e5 曾出現 node #37/#38/#39 三個 byte-identical 子節點白白燒掉 plateau 預算的問題。

**④ 經驗庫先驗 `suggest_priors(comp_meta)`(不呼叫 LLM)**
- 用競賽的 metric/tags 關鍵字,對 `knowledge/experience.md` 的 `##`/`###` 標題做**子字串比對**;命中的段落把其「`… | 證據:…`」條列行**原文回傳**給提議器當先驗。

---

## v3 — 預算相位機 + 九件生產化(`harness_v3.py`,Phase F-1/H-1)

> 把 11 場實戰中 driver 各自的臨時解法,收斂成**引擎預設**(流程圖橘色節點)。節點 schema / kind / 快取 / 去重 / plateau / 先驗**全繼承 v2**。

1. **預算相位機** `update_phase`(**①**):`exploit → explore_burst → stopped`,每次 `add_node` 後**單向自動推進**。
2. **停止規則** `should_stop`(**①**):**硬上限 = 已評估節點 ≥ 60**;或在 burst window 之後**連續 20 次評估無改善**即停;`stop_reason` 記錄為何停。(v1/v2 都沒有。)
3. **dedup 消耗預算**(**②**,改良自 v2 純擋):**同一父節點連撞 2 次** → 燒一個 `status="failed"` 佔位子節點(算進 `MAX_CHILDREN`、不算進 `n_evaluated`)→ 讓該父節點自然「用滿」、搜尋往前走,**防止無限 re-propose**。
4. **solo 突破自動重開 blend 血脈**(**③**):某個 **solo** 節點成為新 global best 時,自動**解凍**已 plateau 的 blend 血脈(streak 重置),讓它下一輪就能把新成員吸收進去。(修 s3e11 需要人工「phase 2 重播種」的痛。)
5. **邊界外推突變** `boundary_candidates`(**④**):任何數值超參落在其宣告搜尋範圍**任一端 ±5%** 內 → 自動提議「**再往邊界外推**」(log 型參數在 log space 判定/外推)。s3e11 CatBoost depth 10→12、s3e16 lr 跌破 Optuna 下限 0.01 都是這樣挖到的最大槓桿。
6. **權重搜尋強化** `eval_blend`(**⑤⑥**):預設 **k=800 + 座標上升(coordinate-ascent)細修(只進不退)**;**成本守衛**:單次 eval > 45s → **降到 k=200 重跑**並**明確寫警告到 `cost_guard_log`(絕不靜默降精度)**。
7. **resume 契約**(**⑦**):`save/load_search_state` + `validate_state` 自檢;**driver 自己的簿記一律放 `tree["search_state"]["driver_state"]`**(不放模組全域)→ 進程重啟後存活。(修 F-2 s3e7 三次重啟、兩次是 resume bug。)
8. **子行程層級評估逾時** `eval_solo_subprocess`(**⑧**):`evaluate(config)` 在**子進程**跑,逾時對**整個 process group 送 SIGKILL** → 記 `failed`。因為 `signal.alarm`(SIGALRM)**殺不掉原生 `fit()` 的卡死**——根因是 F-2 s3e7 一次 CatBoost 卡 28 分鐘、313% CPU。
9. **burst 種子健全性閘** `burst_seed_sanity_gate`(**⑨**):plateau 飽和(所有血脈都 plateau)時,driver 注入 **5–8 條長射程種子**;若某種子分數**爛過 `global_best + max(3×gap, 5%×|best|)`** → 該種子血脈**立刻 plateau(只燒一格)**,不讓爛種子衍生子節點。

---

## v4 — 外部想法注入 `[EXT]` + 雙親重組(`harness_v4.py`,Phase J / 階段 5)

> 在 v3 搜尋核心上**只加下列**(流程圖紫色節點);**整個 v3 核心原樣 re-export**,`mode='off'` 時逐位元等於 v3。

1. **`idea_bank.md` 解析**(**①②**):
   - `parse_idea_bank()` 解析 `### [EXT-NN]` 條目(用 `suggest_priors` 同一個 markdown 切段器;把所屬 `##` 類別標題**折進比對 haystack**,讓類別關鍵字也能命中)。
   - `parse_dedup_suppress_ids()`:**已被 `[INT]` 實證的 `[EXT]` 不重複觸發**;輸出文字蓋章「literature prior, no in-project evidence」以資區別。
2. **`suggest_ext_priors()`**(**③**):`[EXT]` 版的 `suggest_priors`,對每條 `[EXT]` 的 haystack 做子字串比對,回傳帶 provenance(`provenance='EXT', source='EXT-NN'`)的先驗,並套用去重抑制集。
3. **`suggest_priors_v4(comp_meta, mode=)`**(**④**,併池入口):
   - `mode='off'`(stage-4 baseline)= **只回 `[INT]`,與 `harness_v3.suggest_priors` 逐位元相同**;
   - `mode='ext'`(stage-5)= `[INT]` + **淨新增(去重後)的 `[EXT]`**;每條先驗都帶 `INT/EXT` provenance,供 J-4 歸因。
   - 設計保證:`prior_texts(suggest_priors_v4(cm, 'off')) == harness_v3.suggest_priors(cm)`,所以**關掉注入絕不擾動 stage-4 基線**。
4. **雙親重組**(**⑤**,J-2b,機械式、無 LLM):
   - `should_offer_recombination`:僅在 `phase == explore_burst` 時提供;
   - `propose_recombinations`:把 **top-4 強節點**配對(≤6 對)→ **機械融合**:blend 成員**取聯集**、特徵族**取聯集**(drop 欄取交集 = 保留聯集)、model/超參**繼承較強的一方**;丟掉退化/重複配方。
   - 重組出的 blend 子節點,其**權重在 `eval_blend` 內重新搜尋**(recombine **從不固定權重**)。
   - `record_recombination`:重組子節點的**雙親 provenance 寫進 `search_state.driver_state`**(供 J-3/J-4 把每個突變歸因到 INT / EXT / recombine;**config 不被污染,`config_hash` 仍代表真實模型身分**)。

---

## 直向看:哪個機制屬於哪一版

| 機制 | v1 | v2 | v3 | v4 |
|---|:--:|:--:|:--:|:--:|
| 節點 / 血脈 / 選擇 / 回溯 / 崩潰安全 save | ● | | | |
| plateau streak = 3 | ● | → adaptive 3/5(tie_rate>0.15) | | |
| dead-end 護欄(血脈用盡→強制 plateau;全 plateau→reopen) | ● | | | |
| **blend 一等 / `eval_solo` / `eval_blend` / `metric_fn` 鐵則** | | ● | | |
| Dirichlet 權重搜尋 | | ● k=1500 兩輪 | → k=800+座標上升;>45s 降 k=200 | |
| **config 去重 `config_hash`** | | ● 純擋 | → 連撞 2 次燒 failed 預算 | |
| 經驗庫先驗 `suggest_priors`([INT]) | | ● | | |
| **預算相位機 / `should_stop`(60 / 20)** | | | ● | |
| solo 突破自動重開 blend 血脈 | | | ● | |
| 邊界外推突變 `boundary_candidates`(±5%) | | | ● | |
| resume 契約 / 子行程逾時 / burst 種子閘 | | | ● | |
| **外部注入 `[EXT]` / `suggest_priors_v4(mode)`** | | | | ● |
| **雙親重組 `recombine` / provenance 記錄** | | | | ● |

(●=該版新增;箭頭=在該版被改;空白=繼承前版不變。)

---

## 一個重要後話:v4 之後沒有 v5

v4 建了「**想法進得來的門**」(`[EXT]` 注入 hook 正確),但當時**消費端沒接**——`suggest_priors_v4` 回傳存進 `tree['priors']`、**搜尋從沒讀**。這就是 **write-only bug**(先驗從未真正注入搜尋)。

修復**不是 v5**,而是獨立的 **`prior_wiring.py`**(2026-07-13)把消費端接上,做成三臂對照:
- **OFF** = 不注入(基線);
- **arm A(機械版)**:規則模板把先驗翻成 config,蓋 `[PRIOR-*]` 標記 + 交付帳本(決定性可重現);
- **arm B(LLM in the loop)**:LLM 讀先驗+即時戰況提案,frozen-replay + 動態白名單驗證。

結論:**同池重加權受凸最優上限約束**(機械版 5/5 無增益);**LLM 擴池注入去相關新成員**在 **2/5 場統計顯著**(s6e1 / s4e1),s5e10 NNLS 診斷增益為真。詳見 `docs/prior_wiring_findings.md`、`docs/bug_prior_injection_noop.md`。

> 記法:**v4 = 門;`prior_wiring` = 門後的路。**

---

## 三個要記住的重點

1. **演進哲學**:v1 骨架 → v2「集成 + 離散指標正確性」→ v3「跑得久、穩、生產化」→ v4「想法來源擴到外部」。每版只加「上一批實戰**證據要求**的東西」,前版不動 → 舊結果永遠可重現、新功能可單獨歸因。
2. **去重 v2 vs v3**:偵測法一樣(`config_hash`、同一入口),差別在**撞到之後**——v2 純擋、v3 **連撞 2 次燒一格 `failed` 預算**(避免無限 re-propose 空轉)。
3. **v4 的 bug 教訓**:「有寫入、有日誌」≠「有被消費」——注入 hook 對,但消費端沒接就是空操作;驗收要**驗到消費端**(最後靠「開/關注入輸出**逐位相同**」抓到 write-only bug)。
