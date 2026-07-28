<!--
═══════════════════════════════════════════════════════════════════════════
 SKILL.md — 一個 Skill 的核心檔案 / The core file of a Skill
───────────────────────────────────────────────────────────────────────────
 教程對照 (Tutorial mapping):
   技能 (Skills) 放在 .claude/skills/ 中,由「指令 + 腳本 + 資源」組成的資料夾。
   A Skill lives in .claude/skills/ and bundles instructions + scripts + resources.

   漸進式揭露 (Progressive disclosure) 有三層:
     1. metadata(下方 frontmatter 的 name + description)—— 對話一開始就載入
     2. SKILL.md 本文(body)—— 技能被「呼叫」時才載入
     3. scripts/ 與 references/ —— 需要時才讀取 (on demand)
   Three loading levels: metadata always loaded; body loaded on invoke;
   bundled resources read only when needed.
═══════════════════════════════════════════════════════════════════════════
-->
---
# ── YAML frontmatter ──────────────────────────────────────────────────────
# 註解: name 是技能的識別碼,必須與資料夾名稱一致,用 kebab-case。
# Note: `name` is the skill identifier; keep it identical to the folder name.
name: kaggle-safe-submit

# 註解: description 是「觸發機制」——Claude 靠它決定何時呼叫這個 skill。
#        教程建議寫得「主動一點 (pushy)」,因為 Claude 常常「該觸發卻不觸發」。
#        所以這裡同時寫清楚 (1) 這個 skill 做什麼 + (2) 什麼情境該用它。
# Note: `description` IS the trigger. It must say WHAT the skill does AND
#        WHEN to use it. Kept deliberately assertive to fight under-triggering.
description: >-
  Validate a Kaggle submission CSV against the competition's sample_submission
  BEFORE uploading, check the remaining daily submission quota, log the run to
  experiments.json, then submit via the Kaggle CLI. ALWAYS use this skill
  whenever the user wants to submit to a Kaggle competition, upload predictions,
  push a submission file, or asks "is my submission file valid / correctly
  formatted?" — even if they don't say the word "submit". This prevents wasted
  daily submissions from malformed files (wrong columns, wrong row count,
  misaligned IDs, NaNs) and keeps the experiment log in sync.
---

<!--
 註解: 從這裡開始是 SKILL.md 的「本文 (body)」。
       它只有在此技能被呼叫時才進入脈絡,所以要用祈使句、精簡、聚焦流程。
 Note: Everything below is the body. It loads only when the skill triggers,
       so write it as an imperative, procedural playbook — lean and focused.
-->

# Kaggle Safe Submit

在把任何 submission 上傳到 Kaggle 之前,先驗證、再提交。目標是**永不浪費每日提交額度**在格式錯誤的檔案上。
Validate before you upload. The goal is to never waste a daily submission slot on a malformed file.

<!--
 註解: 「先決條件」區塊。教程說 procedural 指令(前置檢查、環境需求)最適合放 skill。
 Note: Preconditions block — procedural setup belongs in the skill, not CLAUDE.md.
-->
## Preconditions / 先決條件

1. **競賽工作區存在 / Competition workspace exists** — `competitions/<name>/` 底下要有 `config.yaml`(內含 `evaluation_metric`、`id_column`、`sample_submission_file`、`special_rules.daily_submission_limit`)。
2. **憑證 / Credentials** — Kaggle 新式 token 需要環境變數 `KAGGLE_API_TOKEN`。它**不會跨 Bash 呼叫保存**,所以每個 kaggle 指令都要在同一行 `export` 它(見下方)。The `KAGGLE_API_TOKEN` env var does not persist across Bash calls — chain the `export` on every kaggle command.
3. **提交檔存在 / Submission file exists** — 使用者指定的 `.csv`,或 `competitions/<name>/submissions/` 中最新的一個。

<!--
 註解: 主流程。用「編號步驟」讓 Claude 一步步照做,並在關鍵步驟停下讓人確認(human-in-the-loop)。
 Note: The main workflow. Numbered steps = a deterministic sequence Claude can
       follow; stop for confirmation at the irreversible step (the upload).
-->
## Workflow / 主流程

### 1. 讀取競賽設定 / Read the competition config
讀 `competitions/<name>/config.yaml`,取得:`id_column`、`evaluation_metric`、`optimization_direction`、`sample_submission_file`、`special_rules.daily_submission_limit`。若使用者沒指明競賽名稱,從路徑或最近的 experiments.json 推斷,並向使用者確認。

### 2. 驗證格式 / Validate the format  ⭐ 核心
執行綁定的驗證腳本(這是**確定性、可重複**的工作,所以固化成 script 而非每次重寫):
Run the bundled validator (this is deterministic + repetitive work → a script, per the tutorial):

```bash
uv run python .claude/skills/kaggle-safe-submit/scripts/validate_submission.py \
  --submission <path/to/submission.csv> \
  --sample competitions/<name>/data/<sample_submission_file> \
  --id-col <id_column>
```

腳本會檢查:欄位名稱與順序、列數、ID 對齊、NaN/Inf、機率欄位是否落在 [0,1]。**任何一項失敗就停下**,把問題回報給使用者,**先不要提交**。
The script checks columns/order, row count, ID alignment, NaN/Inf, and probability ranges. **On any failure, STOP and report — do not submit.**

深入的檢查清單與各競賽型別的規則,見 → `references/submission_checklist.md`(需要時才讀)。
For the deep checklist and per-problem-type rules, read → `references/submission_checklist.md` (load on demand).

### 3. 檢查剩餘額度 / Check remaining quota
在提交前,先看今天還剩幾次(對照 config 的 `daily_submission_limit`):
```bash
export KAGGLE_API_TOKEN=<token> && uv run kaggle competitions submissions -c <name> | head
```
數今天(UTC 日期)已用的次數。若已達上限,**告訴使用者並停止**,別讓提交失敗白白浪費。

### 4. 記錄實驗 / Log the experiment
在提交前把這次提交寫進 `competitions/<name>/experiments.json`(用 `utils/experiment_log.py` 的慣例):提交檔路徑、模型/描述、本地 CV 分數、時間戳、備註。這樣公開 LB 分數回來後可以對回是哪個實驗。

### 5. 提交 / Submit  ⚠️ 不可逆,先確認
這一步會消耗一次每日額度,屬於**外向、難以撤回**的動作——先向使用者複述「競賽、檔案、訊息」並取得同意再執行:
This consumes a daily slot and is outward-facing/hard-to-undo — confirm competition + file + message with the user first:
```bash
export KAGGLE_API_TOKEN=<token> && uv run kaggle competitions submit \
  -c <name> -f <submission.csv> -m "<concise message: model + CV score>"
```

### 6. 確認並回填 / Confirm and back-fill
```bash
export KAGGLE_API_TOKEN=<token> && uv run kaggle competitions submissions -c <name> | head
```
把回傳的 public LB 分數填回 experiments.json 對應的那筆,並向使用者回報 CV vs LB(注意別過擬合公開 LB)。

<!--
 註解: 「安全準則」區塊。集中列出不可違反的界線,呼應教程「hooks/rules 管確定性,skill 管流程」。
 Note: Guardrails — the hard boundaries. Keep them explicit and few.
-->
## Guardrails / 安全準則
- **絕不硬編碼 token** / Never hardcode the token — 只用 `KAGGLE_API_TOKEN` 環境變數;別把它寫進任何檔案或訊息。
- **驗證未過就不提交** / No submit before validation passes。
- **提交前一定停下確認** / Always confirm before the irreversible upload。
- **尊重競賽規則** / Respect `special_rules`(external data / internet / 每日上限)。
- **別為了追公開 LB 過擬合** / Track public vs private LB; don't overfit to public.

<!--
 註解: 綁定資源清單。讓 Claude 知道「還有什麼可以按需讀取」,是漸進式揭露的指路牌。
 Note: Bundled resources — signposts for the on-demand third layer.
-->
## Bundled resources / 綁定資源
- `scripts/validate_submission.py` — 確定性的格式驗證器 / deterministic format validator。
- `references/submission_checklist.md` — 完整檢查清單 + 各問題型別的提交規則 / full checklist + per-problem-type rules。
