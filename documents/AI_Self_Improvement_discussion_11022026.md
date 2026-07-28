# AI Self-Improvement 技術全覽

> 本文整理了 AI 如何進行 self-improvement 的主要技術，包括 LLM-as-a-Judge、Verifiable Rewards，以及在不改動權重的情況下訓練 agent 的方法。

---

## 一、AI Self-Improvement 的主要方法

### 訓練階段的 Self-Improvement

#### RLHF / RLAIF

最經典的路線。RLHF 用人類偏好來訓練 reward model，再用 RL 優化模型。RLAIF（AI feedback）則是用另一個 LLM 取代人類標註者來產生偏好資料，降低成本。Anthropic 的 Constitutional AI 就是這條路線的代表——讓模型根據一組原則自我批評、自我修正，再用這些資料做訓練。

#### Self-Play / Self-Instruct

讓模型自己生成訓練資料。例如 Self-Instruct 讓 LLM 產生 instruction-following 的資料對，再拿來 fine-tune 自己（或下一代模型）。類似的還有 Stanford 的 Alpaca 做法。

#### STaR (Self-Taught Reasoner)

模型嘗試解題，把答對的 chain-of-thought 拿回來當訓練資料，反覆迭代。這在數學和推理任務上效果不錯。

### 推理階段的 Self-Improvement（不改權重）

#### LLM-as-a-Judge / Self-Evaluation

生成多個候選回答，再讓同一個（或另一個）LLM 來評分、排序、選最佳答案。這是目前非常流行的做法，廣泛用在 benchmark 評估和 best-of-N sampling 上。

#### Self-Refine / Iterative Refinement

模型先產生初稿，然後自己給 feedback，再根據 feedback 修改。這個循環可以跑多輪。像 "Reflexion" 這類方法也屬於這個範疇——agent 執行任務失敗後，把失敗經驗寫成文字記憶，下次再試時參考。

#### Tree-of-Thought / Monte Carlo Tree Search

把推理過程展開成樹狀結構，在多條路徑中搜索最佳解。DeepMind 在數學證明和 coding 上大量使用這類方法。

#### Verification / Process Reward Model (PRM)

不只看最終答案對不對，而是逐步驗證推理過程中每一步是否正確。OpenAI 的 "Let's Verify Step by Step" 就是這個方向，對數學推理特別有效。

### Agent 框架中的 Self-Improvement

在 agentic 系統裡，self-improvement 通常是組合技：

- **記憶與經驗累積**：把過去的成功/失敗經驗存起來，下次檢索參考
- **工具使用與環境反饋**：執行 code 後拿到 error message，據此修正（本質上是用環境當 judge）
- **多 agent 辯論**：讓多個 LLM instance 互相挑戰、辯論，最後收斂到更好的答案

### 小結

LLM-as-a-judge 是一個重要且流行的組件，但它比較偏向「評估」端。完整的 self-improvement pipeline 通常需要結合：生成多樣候選 → 評估/驗證 → 篩選或修正 → 可能回饋到訓練。目前最前沿的趨勢是把 RL + verifiable rewards（如 code execution、數學驗證）結合起來，因為這類 signal 比 LLM 自己判斷更可靠，不容易出現「自己騙自己」的問題。

---

## 二、什麼是 Verifiable Rewards？

核心概念：**reward signal 來自可以客觀、自動驗證的來源，而不是靠另一個 LLM 的主觀判斷。**

這很重要，因為如果用 LLM 當 judge，它可能會被「聽起來很有道理但其實是錯的」回答騙過去（reward hacking）。Verifiable rewards 則提供了一個 ground truth 的錨點。

### 具體例子

#### 1. 數學題：比對最終答案

模型解一道數學題，產生完整推理過程，最後給出答案 `42`。系統直接比對標準答案——對就是對、錯就是錯。DeepSeek-R1 和 OpenAI 的 o1/o3 系列在訓練時就大量使用這種方式：用 RL 讓模型嘗試解題，答對了就給正向 reward，不需要任何人類標註。

#### 2. 程式碼：跑 Unit Test

模型寫一段 code，系統直接執行並跑預先準備好的 test cases。全部通過 → reward = 1，有 test 失敗 → reward = 0（或按通過比例給分）。這就是 AlphaCode 和許多 coding agent 的做法。對 Kaggle 來說也很直覺——submit 後 leaderboard 分數本身就是一個 verifiable reward。

#### 3. 形式化證明：Proof Checker

模型產生一個數學證明，丟給 Lean 4 或 Isabelle 這類 proof assistant 做形式化驗證。通過驗證 = 正確，沒通過 = 有邏輯錯誤。AlphaProof 就是這個路線。

#### 4. 遊戲 / 模擬環境：勝負或分數

下棋贏了就是贏了，Atari 遊戲的分數就是分數，不需要任何主觀判斷。這是最早期的 verifiable reward，AlphaGo 就是靠這個訓練的。

#### 5. 資訊檢索：Factual Verification

問模型一個有明確答案的事實問題（例如「法國首都是哪裡？」），直接 string match 或用簡單規則比對。雖然比較粗糙，但在訓練 factuality 時是有效的 signal。

### 跟 LLM-as-a-Judge 的對比

| | Verifiable Rewards | LLM-as-a-Judge |
|---|---|---|
| **信號來源** | 外部客觀標準（test case、答案、proof checker） | 另一個 LLM 的判斷 |
| **可靠性** | 高，不會被「聽起來對」騙過 | 可能被 fluent but wrong 的回答誤導 |
| **適用範圍** | 有明確對錯的任務 | 開放式任務（寫作、對話品質等） |
| **成本** | 低（自動化執行） | 中等（需要 LLM inference） |

### 實際趨勢

目前最強的 reasoning model（o1/o3、DeepSeek-R1、Gemini 2.5）幾乎都是用 **RL + verifiable rewards** 訓練出來的。核心迴路就是：

> 模型嘗試解題 → 自動驗證答案 → 對了強化這條推理路徑 → 反覆迭代

這比依賴人類標註或 LLM judge 的方式更容易 scale up，因為你可以無限量自動生成題目和驗證結果。

---

## 三、不改權重的情況下，如何用 Verifiable Rewards 訓練 Agent？

嚴格來說「不改權重」不算「訓練」，更準確的說法是 **inference-time adaptation** 或 **in-context learning-based improvement**。但效果上確實能讓 agent 越來越好。

### 方法 1：Best-of-N Sampling（Rejection Sampling）

最直接的方法：

> 讓模型對同一個問題生成 N 個候選方案 → 每個都跑 verifiable reward（例如跑 test case）→ 選分數最高的

具體流程：
- 給模型一個 Kaggle 任務，讓它生成 10 個不同的 feature engineering + model pipeline
- 每個都實際執行、算出 CV score
- 選最好的那個繼續往下做

OpenAI 早期研究發現，Best-of-N 在很多場景下的效果接近甚至超過用 RL fine-tune 的模型，只要 N 夠大。缺點是 inference cost 線性成長。

### 方法 2：Iterative Refinement with Environmental Feedback

目前 coding agent 最常用的模式：

```
loop:
    1. Agent 產生/修改 code
    2. 執行 code，收集 feedback（error message、score、test 結果）
    3. 把 feedback 放進 context，讓 agent 修正
    4. 重複直到滿意或達到上限
```

關鍵在於 verifiable reward 變成 context 的一部分，模型透過 in-context learning 來「理解」自己哪裡做錯了。這就是 Claude Code、Devin、OpenHands 這類工具的核心迴路。

以 Kaggle 為例：
- Agent 寫了一個 XGBoost pipeline，CV score = 0.78
- 把 score 和 error log 餵回 context
- Agent 分析後決定加 feature interaction，改完再跑
- 新 CV score = 0.81
- 繼續迭代...

模型權重完全沒變，但 agent 的「表現」透過環境反饋不斷改善。

### 方法 3：Reflexion — 結構化的經驗記憶

Reflexion（Shinn et al., 2023）把 iterative refinement 更系統化：

```
Episode 1:
  - Agent 嘗試解題 → 失敗
  - Agent 寫一段「反思」：「我犯的錯是 X，下次應該 Y」
  - 這段反思存入 memory

Episode 2:
  - Agent 帶著之前的反思記憶重新嘗試
  - 表現通常會改善
```

Verifiable reward 在這裡的角色是觸發反思的信號——score 沒提升或 test 沒通過，就觸發反思流程。Memory 是純文字的，塞在 prompt 裡，完全不動權重。

### 方法 4：Experience Library / Retrieval-Augmented Agent

把成功經驗系統化存儲：

- Agent 解決了很多 Kaggle 任務，每次成功的方案都存進一個 knowledge base
- 格式可以是：`{任務描述, 嘗試過的方法, 最終成功方案, score}`
- 遇到新任務時，用 RAG 檢索最相關的歷史經驗塞進 context

這類似於人類的「經驗累積」，但機制是檢索而非學習。Verifiable reward 在這裡決定哪些經驗值得保留（只存 score 超過某閾值的方案）。

### 方法 5：Tree Search + Verifiable Reward 剪枝

把解題過程建模成一棵搜索樹：

```
root: 原始任務
├── 方案 A: 用 LightGBM
│   ├── A1: + target encoding → score 0.82 ✓ 繼續探索
│   └── A2: + PCA → score 0.75 ✗ 剪掉
├── 方案 B: 用 Neural Net
│   ├── B1: + embedding → score 0.80 ✓ 繼續探索
│   └── B2: + raw features → score 0.71 ✗ 剪掉
```

每個節點都用 verifiable reward（CV score）來決定要不要繼續往下展開。這本質上是 MCTS 的變體，AlphaCode 在 competitive programming 上就用了類似的策略——生成大量程式、用 test case 過濾、再對通過的方案做 clustering。

### 方法 6：Prompt / Strategy Evolution

受遺傳演算法啟發的做法：

- 維護一個 prompt 或 strategy 的 population
- 每個 strategy 都跑一遍拿 verifiable reward
- 高分的 strategy 保留、交叉、變異，低分的淘汰
- 重複多代

Google DeepMind 的 **FunSearch** 就是這個思路——用 LLM 生成數學函數的候選解，用 verifiable evaluator 篩選，再把好的解餵回 LLM 作為 few-shot example 引導下一輪生成。整個過程不改權重，但解的品質持續提升。

### 統一框架

這些方法有個共通結構：

```
while not satisfied:
    candidates = LLM.generate(context + past_experience)
    scores = [verify(c) for c in candidates]      # verifiable reward
    best = select(candidates, scores)
    past_experience.update(best, scores)           # 某種形式的 memory
```

差異只在於：

- **搜索策略**：random sampling vs. tree search vs. evolutionary
- **記憶機制**：in-context vs. 外部 database vs. 結構化 reflection
- **驗證方式**：unit test vs. score vs. proof checker

---

## 四、結合外部搜尋的 Self-Improvement 策略

Self-improvement 機制很依賴 LLM 內部既有的知識。但就像學生解題卡關時會上網搜尋，agent 也可以在推理過程中 **主動決定何時需要外部資訊**，而不是只靠自己腦中的知識硬撐。

### 核心概念：從傳統 RAG 到 Adaptive Retrieval

傳統 RAG 是「先搜再答」，但更進階的做法是 **adaptive retrieval**——模型在推理途中判斷「我卡住了」或「我不確定」，才觸發搜尋。這比每次都搜要聰明得多。

### 策略 1：Self-Ask / Chain-of-Search

模型在推理過程中，主動把問題拆解成子問題，對自己不確定的子問題發起搜尋：

```
原始問題：「這個 Kaggle 時序資料集適合用什麼方法？」

Agent 內部推理：
  → 我需要知道資料的特性（已知，從 EDA 得到）
  → 我需要知道最新的時序模型有哪些（不確定）
  → [觸發搜尋] "state-of-the-art time series forecasting 2025"
  → 搜到 TimesFM、Chronos、Moirai 等 foundation model
  → 結合搜尋結果繼續推理
```

Google 的 **Self-Ask**（Press et al., 2022）就是這個思路的早期版本——模型自問 "Do I need to look this up?"，如果答案是 yes，就呼叫搜尋工具。

### 策略 2：ReAct（Reasoning + Acting）

目前最知名的框架。模型交替進行 **思考（Thought）** 和 **行動（Action）**：

```
Thought: 我需要了解這個比賽的評估指標是什麼
Action: search("Kaggle competition X evaluation metric")
Observation: 比賽使用 RMSLE 作為評估指標
Thought: RMSLE 對大值的懲罰較小，我應該對 target 做 log transform
Action: write_code("y_train = np.log1p(y_train)")
...
```

關鍵是模型自己決定什麼時候需要搜尋、搜什麼。搜尋結果變成 context 的一部分，影響後續推理。

### 策略 3：Adaptive Retrieval — 知道自己不知道

更精緻的做法是讓模型評估自己的不確定性，只在需要時才搜尋：

**Self-RAG**（Asai et al., 2023）訓練模型產生特殊 token 來表示：
- `[Retrieve]` — 我需要搜尋外部資訊
- `[No Retrieve]` — 我自己就能回答
- `[IsRelevant]` — 搜到的東西是否相關
- `[IsSupportive]` — 搜到的東西是否支持我的回答

這樣模型不會盲目搜尋（浪費時間），也不會該搜的時候不搜（導致幻覺）。

另一個方向是用 **confidence calibration**：如果模型對自己的回答信心低於某個閾值，就自動觸發搜尋。

### 策略 4：Tool-Augmented Reasoning

把搜尋當作眾多「工具」之一，模型自己選擇何時用什麼工具：

- 搜尋引擎（查最新資訊、查 API 文件）
- Code interpreter（驗證想法）
- 計算器（精確計算）
- 資料庫查詢（查特定數據）

**Toolformer**（Schick et al., 2023）的做法是讓模型在生成過程中自動插入 API call。比方模型在寫一段分析時，發現需要某個統計數字，就中途插入一個搜尋 call，拿到結果後繼續寫。

### 策略 5：多輪搜尋 + 驗證迴圈

結合 verifiable rewards 和外部搜尋的完整迴圈：

```
loop:
    1. Agent 嘗試解題
    2. 驗證結果（跑 test / 看 score）
    3. 如果失敗：
       a. 分析錯誤原因
       b. 判斷是「知識不足」還是「邏輯錯誤」
       c. 如果知識不足 → 搜尋相關技術文件、論文、Kaggle discussion
       d. 如果邏輯錯誤 → 用 self-refine 修正
    4. 帶著新資訊重新嘗試
```

這就是目前最先進的 coding agent（如 Devin、SWE-Agent）的做法——遇到不熟悉的 library 就去讀文件，遇到 bug 就搜 Stack Overflow，跟真正的工程師行為非常相似。

### 策略 6：知識蒸餾式搜尋（Search → Internalize → Apply）

不只是搜到就用，而是把搜到的資訊「消化」後再應用：

```
1. 搜尋：「LightGBM categorical feature handling best practices」
2. 讀取多篇文章，提取關鍵資訊
3. 整合成一段結構化知識：
   「LightGBM 原生支援 categorical feature，不需要 one-hot encoding，
    但需要設定 categorical_feature 參數，且 cardinality > 數百時效果下降」
4. 把這段知識存入 working memory
5. 在後續的 code 生成中應用這個知識
```

這比直接把搜尋結果塞進 prompt 更有效，因為模型先做了一次「理解」和「壓縮」。

### 跟純 Self-Improvement 的對比

| | 純 Self-Improvement | + 外部搜尋 |
|---|---|---|
| **知識邊界** | 受限於模型訓練資料 | 可以獲取最新資訊 |
| **失敗模式** | 可能在錯誤方向上反覆嘗試 | 卡住時能找到新方向 |
| **效率** | 靠 trial-and-error | 搜尋可以大幅減少嘗試次數 |
| **風險** | 幻覺（自信地說錯） | 搜尋品質影響結果 |

---

## 五、Kaggle Agent 的實用建議

對 Kaggle agent 來說，最實用的組合大概是 **Iterative Refinement + Experience Library + Best-of-N + 外部搜尋**，因為：

- CV score 天然就是 verifiable reward
- 每次嘗試的成本（跑一次 training pipeline）雖然不低但完全可自動化
- Leaderboard 分數提供了最終的客觀驗證
- 主動搜尋 Kaggle discussion、相關論文、類似比賽的 winning solution 可以大幅加速迭代

一個理想的 Kaggle agent 應該像一個有經驗的 Kaggler：

1. 先用自己的知識做 baseline
2. 跑出 CV score（verifiable reward）
3. 分析哪裡可以改進
4. **主動搜尋** Kaggle discussion、相關論文、類似比賽的 winning solution
5. 把學到的技巧整合進下一版方案
6. 重複迭代

這個「推理 + 搜尋 + 驗證」的三角結構，可能是目前最接近人類學習方式的 agent 架構。

---

## 參考資源

- **Constitutional AI** — Anthropic（RLAIF 路線的代表）
- **STaR: Self-Taught Reasoner** — Zelikman et al.
- **Reflexion** — Shinn et al., 2023
- **Let's Verify Step by Step** — OpenAI（Process Reward Model）
- **AlphaCode / AlphaProof** — DeepMind
- **FunSearch** — DeepMind（Prompt/Strategy Evolution）
- **DeepSeek-R1** — DeepSeek（RL + Verifiable Rewards）
- **Self-Ask** — Press et al., 2022（Chain-of-Search）
- **ReAct** — Yao et al., 2022（Reasoning + Acting）
- **Self-RAG** — Asai et al., 2023（Adaptive Retrieval）
- **Toolformer** — Schick et al., 2023（Tool-Augmented Reasoning）
- **SWE-Agent** — Princeton NLP（多輪搜尋 + 驗證的 Coding Agent）
