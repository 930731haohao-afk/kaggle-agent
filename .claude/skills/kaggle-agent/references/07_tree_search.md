# 樹搜尋(Tree Search)—— Stage 4 的進階迴圈

## Contents
1. [何時切換到樹搜尋](#1-何時切換到樹搜尋)
2. [Node-space 設計(依競賽型態)](#2-node-space-設計依競賽型態)
3. [預算與停止規則](#3-預算與停止規則)
4. [驅動腳本(driver)檢查清單](#4-驅動腳本driver檢查清單)
5. [先驗(priors)](#5-先驗priors)
6. [誠實回報規則](#6-誠實回報規則)
7. [快速上手](#7-快速上手)

本文件是 [05_evaluation.md](05_evaluation.md)(線性迭代協定)之後的**進階版 Stage 4 迴圈**——不是取代它,而是在條件成立時接手。證據來源:`docs/tree_search_prototype.md`(15 次執行、10 場競賽的完整報告)與 `docs/scaling_experiment.md`(Phase E-5 規模曲線)。

## 1. 何時切換到樹搜尋

**觸發條件**:Stage 3 已有一個 baseline solo 模型,且至少已手動或自動產出 **1 個 blend**(即 05_evaluation.md 的線性迭代協定已跑過第一輪)。在此之前不要跳過線性迭代——樹搜尋的 node-space 需要至少一個 solo 池與已知的 blend 概念才有東西可以變異。

**為什麼(證據)**:`docs/tree_search_prototype.md` 第 10 節——15 次執行、10 場競賽,樹搜尋 **9 勝、1 精確平(s3e9)、0 負**,對照線性迭代最佳分數。v1 三場的唯一敗場(s3e9)根因是節點空間結構性缺 ensemble,v2 加入 ensemble-default 節點空間後翻盤為精確平手。換言之樹搜尋不是「換一種訓練方式」,而是「系統性地把線性迭代原本靠人工試出來的變異(邊界推進、seed-bagging、kitchen-sink blend)自動化並窮舉」。

**何時仍用線性迭代(fallback)**:
- 第一輪迭代(還沒有 baseline + blend)——見 [05_evaluation.md](05_evaluation.md)。
- 極小資料/評估成本極高、負擔不起 ~60 節點預算的場合。
- 探索性 EDA 階段(樹搜尋是「已知要優化什麼」之後的收斂工具,不是探索工具)。

## 2. Node-space 設計(依競賽型態)

| 競賽型態 | Node-space 設計 | 依據 |
|---|---|---|
| 一般表格(GBDT 為主) | **solo + blend 雙軌(預設)**:`kind="solo"` 節點是單一模型組態(model/params/features),`kind="blend"` 節點是既有 solo 節點的加權混合。這是 v2/v3 的預設 schema,10 場中 8 場直接適用。 | `harness_v2.py` node schema;第 6 節「ensemble-default 節點空間」 |
| 結構主宰型(如 s3e20:分組歷史均值完勝 GBDT) | **結構化組態(structural configs)**,`kind="solo"` 但 `params` 是結構旋鈕(如 window 大小、異常年降權、joint 移動量),而非 GBDT 超參。先確認 Phase-B 殘差診斷是否已顯示「特徵對目標無結構外可預測訊號」——若是,不要浪費預算硬塞 GBDT node-space。 | `eval_s3e20_v2.py`;第 9 節 E-4 |
| 離散化 / 取整後決策指標(如 QWK-after-rounder、rounded MAE) | metric_fn **必須內含後處理**(rounding/OptimizedRounder),決策永遠對**後處理後**的分數做,不能對 raw OOF 分數做——否則會出現「raw 改善、rounded 反而變差」的鏡像失敗模式。 | 第 9 節 E-3(s3e16):勝出組合 raw MAE 比線性冠軍更差,但 rounded MAE 更好 |

無論哪種型態,**每個節點的 `config` 欄位必須是 canonical/hashable 的「儲存形式」**(dedup 依賴的雜湊才有意義)——參考 `run_s3e7_v3.py` 的 `core(cfg)` 慣例:去掉 `result`/`want_importance` 等輸出欄位,blend 的 `members` 排序後再存。

## 3. 預算與停止規則

`harness_v3.py` 已把這些規則編碼為預設常數,驅動腳本不需要重新發明:

- **總預算 60 個已評估節點**(`DEFAULT_TOTAL_BUDGET`)= exploit 階段 ~35–40 + 強制 explore burst 5–8 條長射程 lineage。60 這個數字來自 15 次執行的 best/total 比值均值 0.63(範圍 0.10–1.00),留有充分餘裕。
- **Explore burst 不可省略**,即使 exploit 階段「看起來已收斂」。三份獨立證據(E-5 的 s3e3-scale、F-2 的 s3e7、F-2 的 s3e14)顯示 post-exploit 階段的**全部**增益都來自強制注入的 kitchen-sink mega-blend——沒有一次是手寫長射程 solo lineage 單獨貢獻的。`update_phase`/`should_stop` 會自動把相位機切到 `"explore_burst"`;driver 收到這個相位後必須注入至少一個「把當前全部 solo 池」的 kitchen-sink blend,外加 1–2 個對照方向的長射程 solo。
- **停止規則**:burst 開始後連續 **15–20 次**評估(v3 預設 20)未刷新全域最佳即停;不論相位機是否觸發,**60 節點數值上限保底**。誠實但書:auto-stop 的耐心計數器路徑在兩場 F-2 實戰中都**未曾真正觸發**(burst 持續改善,計數器一直被重置,最終是數值上限結束搜尋)——只在一次小預算煙霧測試中端到端驗證過,尚無「burst 是 dud」的實戰資料點。
- **邊界推進(boundary-push)是一等公民變異型別**:`boundary_candidates()` 自動標記卡在宣告搜尋空間邊緣(`edge_frac=0.05`)的超參並提出往外推的候選。四個獨立場次(s3e11/s3e5/s3e16/s3e7)證實這往往是單一最大槓桿之一——呼叫它,不要只靠人工重讀 Optuna trial 表。

## 4. 驅動腳本(driver)檢查清單

以 `tree_search/run_s3e7_v3.py` 為模板(已接上 Phase H-1 三項生產化功能),新競賽的 driver **必須**做到:

1. **Root 逐位元驗證(digit-for-digit)**:root 節點的組態必須與該場已知最佳 solo(通常來自線性迭代或先前 harness 版本)位元相同,評估完立刻 `assert round(auc_or_score, 6) == <known_value>`,失敗就終止——不要讓 root 本身的資料/特徵漂移悄悄污染整棵樹。
2. **OOF cache 重用**(可選但建議):若舊樹(v2 或線性迭代)已有相同組態的 cached OOF,重新載入並重算指標,只在對到小數點後 6 位時才接受重用(digit-verified),否則走真實訓練——省下重複訓練的算力,同時不犧牲正確性。
3. **Resume-state 契約**(H-1 功能 7,`save_search_state`/`load_search_state`):driver 自身的執行期狀態(lineage 名稱映射、burst 已注入旗標、per-node 結果暫存、dedup offset)一律放進 `tree["search_state"]["driver_state"]`,**絕不**留在模組層級 Python 全域變數。永遠用 `hv3.save_search_state(tree, path)` 存、`hv3.load_search_state(path)` 讀——這兩個入口點會先跑 `validate_state` 自我檢查,resume 壞掉時給出明確診斷而不是三次迭代後才冒出的 KeyError。根因:s3e7 F-2 驗證跑 3 次重啟中有 1 次直接源自模組層級狀態沒有隨 resume 存活。
4. **子行程層級評估逾時**(H-1 功能 8,`eval_solo_subprocess`):solo 節點的訓練呼叫一律透過 `hv3.eval_solo_subprocess(eval_module_path, config, timeout_s, node_id=...)` 而非直接呼叫 `ev.evaluate(...)`——`signal.alarm` 無法中斷原生 `fit()`(CatBoost/LightGBM/XGBoost 不會把控制權交還 Python bytecode),只有 OS 層級的子行程邊界能強制 kill。根因:s3e7 F-2 一次 CatBoost 訓練(depth 9, bagging_temperature 2.0)在原生呼叫中卡死 28 分鐘、313% CPU。
5. **Burst 種子健全性閘**(H-1 功能 9,`apply_burst_seed_sanity_gate`):每個 explore-burst 長射程種子評估完後立刻呼叫,若分數超出「root↔全域最佳差距的 3 倍」健全性帶(附絕對值下限,避免差距為 0 時全部失敗),立刻把該 lineage 標記 plateau,不讓它繼續衍生子節點。根因:s3e14 F-2 一個 DART 長射程種子 OOF MAE 高達基準 18 倍,driver 又花了 ~12 分鐘(6 次評估)在它身上繼續訓練子節點才放棄。

## 5. 先驗(priors)

呼叫 `harness_v2.suggest_priors(comp_meta)`(v3 原樣繼承,未修改)對 `knowledge/experience.md` 做關鍵字比對——`comp_meta = {"metric": ..., "tags": [...], "data_type": ...}`,回傳逐字附證據引用的 bullet,不呼叫 LLM。在提出每條 lineage 的第一個變異前先查一次。

**先驗定下限,在地洞見定上限**(9 場橫跨 D+E 掃描的重複發現):先驗持續正確地避免浪費算力在已知死路上,但每場真正拉開分數差距的最大單一槓桿,幾乎都來自該場自己的 EDA/comp-local 洞見或搜尋機制本身的結構性改變(tenure-prune、top-code clip、auto_scale、depth-boundary-push、LGBBOUND、YEARWEIGHTS/JOINT),而非經驗庫命中本身。所以:**用 priors 篩掉死路,但不要因為 priors 沒命中就停止尋找 comp-local 的新變異方向**。

每次搜尋後,若發現新的、驗證過的洞見,補回 `knowledge/experience.md`(與線性迭代協定的既有慣例一致)。

## 6. 誠實回報規則

每次樹搜尋跑完,回報時必須包含:

- **evals-to-beat**:第幾次評估追平/超過線性迭代最佳分數(driver 應像 `run_s3e7_v3.py` 一樣機械算出 `evals_to_match_linear_best`,不要用印象估計)。
- **backtracks / plateau 記錄**:`tree["search_state"]["backtrack_log"]` 與 `plateaued` lineage 清單,原樣附上,不摘要掉細節。
- **dedup 拒絕次數**:`DEDUP_REJECTIONS`/`dedup_rejections`——過多代表 node-space 設計太窄,值得檢討變異佇列。
- **cost-guard / sanity-gate 觸發紀錄**:兩者都預設「觸發就必須留下顯式警告」,絕不靜默——回報時逐條列出,即使是「本次兩者都未觸發」也要明說。
- **OOF-only 但書**:除非這次搜尋的最終權重/後處理參數已在真實 Kaggle Public/Private LB 驗證過,否則必須明說「本結論僅在 OOF 分數上成立,尚未經 LB 驗證」——15 次歷史執行中只有 1 場(s3e16/E-3)有真實 LB 錨點,其餘全部是 OOF-only。
- **小樣本/擬合但書**:若最終分數的關鍵決定(權重、後處理超參)是直接對評估用的同一份 OOF/CV 折擬合出來的,且折數少(如 3 折 LOYO)或差距量級極小(<1% 相對改善),要明講這可能是 CV 噪音而非真訊號(參考 s3e20 的 2.17% blend 權重、s3e19 的 fold-5 double-dip)。

## 7. 快速上手

```python
import sys; sys.path.insert(0, "tree_search")
import harness_v3 as hv3

tree = hv3.load_search_state(TREE_PATH) if os.path.exists(TREE_PATH) else hv3.new_tree(COMP)
hv3.init_budget(tree)  # 60/EXPLORE_BURST 5-8/PATIENCE 20,除非有特殊理由覆寫

# root + 第一代 solo 種子 + 至少一個 blend 種子,root 必須 assert 逐位元驗證
# 主迴圈:
while not hv3.should_stop(tree):
    phase = tree["search_state"]["budget"]["phase"]
    if phase == "explore_burst" and not burst_injected:
        inject_explore_burst(tree, ...)          # 每個種子後接 apply_burst_seed_sanity_gate
        continue
    parent_id, lineage_id = hv3.select_next_parent(tree)
    if parent_id is None: break
    child_cfg, mutation = propose_child(tree, parent_id, lineage_id)  # 含 suggest_priors 查詢
    r = hv3.eval_solo_subprocess("tree_search/eval_<comp>.py", child_cfg, timeout_s=200, node_id=nid)
    ...
    hv3.save_search_state(tree, TREE_PATH)
```

完整可運行範例:`tree_search/run_s3e7_v3.py`。API 全清單見 `tree_search/harness_v3.py` 模組docstring。
