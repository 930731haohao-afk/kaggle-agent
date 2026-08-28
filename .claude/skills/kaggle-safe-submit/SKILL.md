---
# ── YAML frontmatter ──────────────────────────────────────────────────────
# Note: name is the skill's identifier; it must match the folder name, in kebab-case.
# Note: `name` is the skill identifier; keep it identical to the folder name.
name: kaggle-safe-submit

# Note: the description is the "trigger mechanism" — Claude relies on it to decide when to invoke this skill.
#        The tutorial suggests writing it a bit "pushy", because Claude often fails to trigger when it should.
#        So this spells out both (1) what this skill does + (2) in which situations to use it.
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
═══════════════════════════════════════════════════════════════════════════
 SKILL.md — The core file of a Skill
───────────────────────────────────────────────────────────────────────────
 Tutorial mapping:
   Skills live in .claude/skills/, as folders bundling "instructions + scripts + resources".
   A Skill lives in .claude/skills/ and bundles instructions + scripts + resources.

   Progressive disclosure has three levels:
     1. metadata (the name + description in the frontmatter below) — loaded at conversation start
     2. the SKILL.md body — loaded only when the skill is "invoked"
     3. scripts/ and references/ — read only when needed (on demand)
   Three loading levels: metadata always loaded; body loaded on invoke;
   bundled resources read only when needed.
═══════════════════════════════════════════════════════════════════════════
-->


<!--
 Note: The SKILL.md "body" starts here.
       It enters context only when this skill is invoked, so use imperative, lean, process-focused wording.
 Note: Everything below is the body. It loads only when the skill triggers,
       so write it as an imperative, procedural playbook — lean and focused.
-->

# Kaggle Safe Submit

Before uploading any submission to Kaggle, validate first, then submit. The goal is to **never waste a daily submission slot** on a malformed file.
Validate before you upload. The goal is to never waste a daily submission slot on a malformed file.

<!--
 Note: The "Preconditions" block. The tutorial says procedural instructions (pre-checks, environment requirements) fit best in a skill.
 Note: Preconditions block — procedural setup belongs in the skill, not CLAUDE.md.
-->
## Preconditions

1. **Competition workspace exists** — `competitions/<name>/` must contain `config.yaml` (with `evaluation_metric`, `id_column`, `sample_submission_file`, `special_rules.daily_submission_limit`).
2. **Credentials** — new-style Kaggle tokens (`KGAT_` prefix) **only work through the env var `KAGGLE_API_TOKEN`**; the old `~/.kaggle/kaggle.json` path does not work for `KGAT_` tokens (see "Credential Setup" in the project CLAUDE.md). This project's token lives in `~/.kaggle/kaggle_api_token.txt` and is read by `utils/kaggle_auth.sh`:
   ```bash
   source utils/kaggle_auth.sh && uv run kaggle <command>
   ```
   The script **exits non-0 and prints the reason** when the token file is missing/empty, so `&&` blocks the kaggle command that follows; on success it prints which file it used, so if you later get a 401 you know which credential expired and needs a fresh one pasted in from https://www.kaggle.com/settings. Equivalent one-liner: `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/kaggle_api_token.txt | tr -d '[:space:]')`.
   The env var **does not persist across Bash calls**, so re-`source` it on the same line for every kaggle command.
   The `KGAT_` tokens only work through the `KAGGLE_API_TOKEN` env var — the old
   `kaggle.json` path does not work with them. The var does not persist across Bash
   calls, so chain the `source` on every kaggle command. (2026-08-03 audit)
3. **Submission file exists** — the `.csv` the user specified, or the most recent one under `competitions/<name>/submissions/`.

<!--
 Note: The main workflow. Numbered steps let Claude follow along one by one, stopping at the critical step for human confirmation (human-in-the-loop).
 Note: The main workflow. Numbered steps = a deterministic sequence Claude can
       follow; stop for confirmation at the irreversible step (the upload).
-->
## Workflow

### 1. Read the competition config
Read `competitions/<name>/config.yaml` to get: `id_column`, `evaluation_metric`, `optimization_direction`, `sample_submission_file`, `special_rules.daily_submission_limit`. If the user did not name the competition, infer it from the path or the most recent experiments.json, and confirm with the user.

### 2. Validate the format  ⭐ core
Run the bundled validation script (this is **deterministic, repetitive** work, so it is frozen into a script instead of rewritten every time):
Run the bundled validator (this is deterministic + repetitive work → a script, per the tutorial):

```bash
uv run python .claude/skills/kaggle-safe-submit/scripts/validate_submission.py \
  --submission <path/to/submission.csv> \
  --sample competitions/<name>/data/<sample_submission_file> \
  --id-col <id_column> \
  --metric <evaluation_metric> \
  --train competitions/<name>/data/train.csv --target <target_column>
```

`--metric` (the `evaluation_metric` from config.yaml) and `--train/--target` are **optional but strongly recommended**: without them, the range check and the probability/hard-label mix-up check print SKIP instead of guessing.
`--metric` and `--train/--target` are optional but strongly recommended — without a
reference the range and label-kind checks report SKIP instead of guessing.

The script gives a two-tier verdict, distinguished by exit code (before the 2026-08-03 audit this script did not exist, so this step had effectively never been run):
The script reports two tiers, distinguished by exit code:

| Exit | Tier | Action |
|---|---|---|
| 0 | PASS | proceed |
| 1 | STRUCTURAL — row count/columns/ID alignment/duplicate IDs/NaN/±inf | **STOP**, fix the file, never waivable |
| 2 | SUSPICIOUS — constant predictions, all 0s, out of range, probability↔hard-label mix-up | **STOP and report to the user**; only after the user explicitly confirms may you rerun with `--allow-suspicious "<reason>"` |

SUSPICIOUS means "the CSV is legal but almost certainly a bug". Constant predictions are legitimate in a few competitions, so the waiver gate exists — but **never self-waive**: always explain the problem to the user and get agreement first.
The SUSPICIOUS tier is legal-but-almost-certainly-wrong. A constant column is a valid
submission on rare competitions, hence the waiver exists — but never self-waive: report
to the user and get explicit agreement first.

For the in-depth checklist and per-competition-type rules, see → `references/submission_checklist.md` (read only when needed).
For the deep checklist and per-problem-type rules, read → `references/submission_checklist.md` (load on demand).

### 3. Check remaining quota
Before submitting, check how many attempts remain today (against `daily_submission_limit` in the config):
```bash
source utils/kaggle_auth.sh && uv run kaggle competitions submissions -c <name> | head
```
Count the attempts already used today (UTC date). If the limit is reached, **tell the user and stop** — don't waste one on a submission that will just fail.

### 4. Log the experiment
Before submitting, record this submission in `competitions/<name>/experiments.json` (following the `utils/experiment_log.py` conventions): submission file path, model/description, local CV score, timestamp, notes. This lets the public LB score be traced back to the right experiment once it comes in.

### 5. Submit  ⚠️ irreversible, confirm first
This step consumes one daily slot and is an **outward-facing, hard-to-undo** action — restate the "competition, file, message" to the user and get agreement before running:
This consumes a daily slot and is outward-facing/hard-to-undo — confirm competition + file + message with the user first:
```bash
source utils/kaggle_auth.sh && uv run kaggle competitions submit \
  -c <name> -f <submission.csv> -m "<concise message: model + CV score>"
```

### 6. Confirm and back-fill
```bash
source utils/kaggle_auth.sh && uv run kaggle competitions submissions -c <name> | head
```
Fill the returned public LB score back into the matching experiments.json entry, and report CV vs LB to the user (beware of overfitting the public LB).

<!--
 Note: The "Guardrails" block. Lists the inviolable boundaries in one place, echoing the tutorial's "hooks/rules for determinism, skills for process".
 Note: Guardrails — the hard boundaries. Keep them explicit and few.
-->
## Guardrails
- **Never hardcode the token** — use only the `KAGGLE_API_TOKEN` env var; never write it into any file or message.
- **No submit before validation passes**.
- **Never self-waive the SUSPICIOUS tier** — `--allow-suspicious` may be used only after explaining to the user and getting agreement (2026-08-03 audit).
- **Always confirm before the irreversible upload**.
- **Respect competition rules** — respect `special_rules` (external data / internet / daily limit).
- **Don't overfit chasing the public LB** — track public vs private LB; don't overfit to public.

<!--
 Note: The bundled-resource list. Lets Claude know "what else can be read on demand" — the signpost for progressive disclosure.
 Note: Bundled resources — signposts for the on-demand third layer.
-->
## Bundled resources
- `scripts/validate_submission.py` — deterministic format validator.
- `references/submission_checklist.md` — full checklist + per-problem-type submission rules.
