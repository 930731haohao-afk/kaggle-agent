# 以 Claude Code 建立自動 Kaggle Agent 的討論整理

## 背景：Moltbot vs Claude Code

Moltbot（現已更名為 OpenClaw）是一個本地運行的開源 AI 個人助理，由 Peter Steinberger 開發。它能 24/7 持續運行，連接 Telegram、WhatsApp、行事曆、Email 等，隨時待命，且擁有持久記憶。

相比之下，Claude Code 主要定位在**程式開發工作流**，並非通用的生活助理。但它可以透過以下方式達成自動化：

- **Headless 模式（`claude -p`）**：無需互動介面即可執行任務
- **GitHub Actions 整合**：透過觸發條件自動執行
- **Tasks 系統**：跨 session 的任務追蹤和依賴關係管理

---

## 核心概念

### 1. Cron Job（Linux 定時排程）

Cron job 是 Linux 內建的定時排程工具，概念等同於 Windows Task Scheduler。用五個欄位定義排程：

```
分  時  日  月  星期幾    要執行的指令
*   *   *   *   *         command
```

範例：
```bash
# 每天凌晨 3 點執行備份
0 3 * * * /home/user/backup.sh

# 每天午夜讓 Claude Code 分析 log
0 0 * * * claude -p "Analyze today's log files" > /home/user/report.txt
```

常用操作：
```bash
crontab -l    # 查看排程
crontab -e    # 編輯排程
crontab -r    # 刪除所有排程
```

### 2. Headless 模式（`-p` flag）

將 Claude Code 從「聊天對象」變成「可程式化的工具」，用腳本直接呼叫：

```bash
# 直接執行任務
claude -p "Review src/model.py for potential bugs"

# 用 pipe 餵資料
cat error.log | claude -p "Summarize the errors in this log"

# 輸出成 JSON
claude -p "List all TODO comments" --output-format json

# 指定允許的工具
claude -p "Fix the failing tests" --allowedTools "Read,Write,Bash(npm test)"

# 接續上一次對話
claude -p "Now focus on database queries" --continue
```

### 3. dangerously-skip-permissions（YOLO 模式）

跳過所有權限檢查，讓 Claude Code 不需要人工批准即可執行任何操作。

**Headless 與 YOLO 的差異：**

| | Headless (`-p`) | YOLO (`--dangerously-skip-permissions`) |
|---|---|---|
| 解決什麼問題 | 不需要人互動操作 | 不需要人批准權限 |
| 權限檢查 | **還是有**，需設定 `--allowedTools` | **完全跳過** |
| 風險程度 | 較低（受權限限制） | 較高（無限制） |

**四種組合：**
```bash
# 1. 互動 + 有權限（預設，最安全）
claude

# 2. 互動 + 跳過權限
claude --dangerously-skip-permissions

# 3. Headless + 有權限（推薦用於自動化）
claude -p "task" --allowedTools "Read,Write,Bash(npm test)"

# 4. Headless + 跳過權限（風險最高）
claude -p --dangerously-skip-permissions "task"
```

> **建議**：自動化場景優先使用第 3 種組合。若需使用 YOLO 模式，建議在 Docker 容器中執行。

---

## 現有 Kaggle 自動化 Agent 框架

### AIDE（最成熟）
- GitHub: WecoAI/aideml
- 核心：將 ML 工程當成程式碼空間中的樹搜索問題
- 在 MLE-Bench 上贏得的獎牌數是第二名的 4 倍
- 支援 OpenAI、Anthropic、Gemini 等多種 LLM

```bash
pip install -U aideml
export OPENAI_API_KEY=<your-key>
aide data_dir="data/" goal="Predict house prices" eval="RMSE"
```

### AutoKaggle（學術研究導向）
- 論文：arXiv 2410.20424
- 五個專門角色：Reader、Planner、Developer、Reviewer、Summarizer
- 六階段工作流，有效提交率 83%

### MLE-STAR（Google Research）
- 63% 的 MLE-Bench-Lite 比賽能拿到獎牌（36% 金牌）
- 特色：先用網路搜尋找 SOTA 模型再優化

### Qgentic-AI（社群專案）
- GitHub: bogoconic1/Qgentic-AI
- 兩個 LLM agent 協作：Researcher + Developer

### Impulse AI（商業平台）
- 在 featured Kaggle 比賽中達前 2.5%（31,791 人中排 782）

### 評估基準：MLE-Bench
- OpenAI 開源，75 個 Kaggle 比賽
- GitHub: openai/mle-bench

---

## 以 Claude Code 建立自動 Kaggle Agent

### 前置準備：Kaggle API

不需要另外建帳號，用自己的 Kaggle 帳號設定 API：

```bash
# kaggle.com → Account → Create New API Token
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions list  # 測試
```

### 工作流架構

```
1. 下載比賽資料（kaggle API）
2. 理解題目（Claude 讀取比賽描述）
3. EDA + 資料清理
4. 特徵工程 + 建模
5. 訓練 + 評估（迭代改進）
6. 產生 submission.csv
7. 提交結果（kaggle API）
```

### 起始腳本範例

```bash
#!/bin/bash
COMPETITION="house-prices-advanced-regression-techniques"
WORKDIR="/home/user/kaggle/$COMPETITION"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

# 下載資料
kaggle competitions download -c "$COMPETITION" -p ./data
unzip -o ./data/*.zip -d ./data

# 讓 Claude Code 開始工作
claude -p "
You are participating in the Kaggle competition: $COMPETITION.
Data is in ./data/.
Steps:
1. Read the competition description and data
2. Do EDA and understand the data
3. Build a strong ML pipeline
4. Generate submission.csv
Focus on achieving the best possible score.
" --allowedTools "Read,Write,Bash(python*),Bash(pip*),Bash(kaggle*)"
```

### 迭代策略

```bash
# 第一輪：baseline
claude -p "Build a simple baseline model..." --resume kaggle-round1

# 第二輪：改進
claude -p "Current score is 0.85 RMSE. Try feature engineering..." --resume kaggle-round2

# 第三輪：ensemble
claude -p "Try ensembling the top 3 models..." --resume kaggle-round3
```

### 重要注意事項

**安全與權限：**
- 用 `--allowedTools` 精確控制 Claude 能做什麼
- 確保 `kaggle.json` 的檔案權限設定正確（600）
- 考慮在 Docker 容器裡跑，隔離風險

**Kaggle 規則：**
- 通常沒有禁止使用 AI agent，但要注意個別比賽的 rules
- 每天有提交次數限制（通常 5 次）
- 有些比賽禁止使用外部資料

**技術面：**
- Linux server 需要足夠的 GPU/記憶體來跑模型訓練
- 注意 Claude Code 的 API 用量，可用 `--max-budget-usd` 設定花費上限
- 讓 Claude 存檔中間結果（checkpoint），避免中斷後從頭來

**建議的起步方式：**
1. 先挑已結束的入門比賽（Titanic、House Prices）練手
2. 用互動模式跟 Claude Code 一起做一次
3. 慢慢把流程腳本化，逐步切換到 headless 模式
4. 確認穩定後再嘗試進行中的比賽
