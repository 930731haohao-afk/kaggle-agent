#!/bin/bash
# conway retry — the first attempt ended early.
#
# It wrote eda.py and train_cnn.py, passed a smoke test, launched fold-0 training in the
# background, and then had no further tool call to make. A headless session ends the moment
# the agent stops calling tools, so the training died unfinished and the competition produced
# no submission. The prompt below forbids backgrounding for exactly that reason.
set -uo pipefail

source /home/tjyen/ai_agents/lane_lock.sh
BASE=/home/tjyen/ai_agents/kaggle
STATUS=$BASE/MYAGENT_LANES_STATUS.md
c=conway-s-reverse-game-of-life

log(){ echo "- \`$(date '+%m-%d %H:%M')\` $*" | tee -a "$STATUS"; }

lane_acquire "my-agent/$c-retry" || log "WARNING: proceeding without the lane lock — contended"
log "RETRY START $c"
start=$(date +%s)

PROMPT="You are running ONE benchmark competition with the kaggle-agent skill: $c.

Work in competitions/$c/ (config.yaml present; data/ symlinks to the official files). Follow the skill's full pipeline — setup, EDA, features, modeling, evaluation, submission — including consulting knowledge/experience.md before EDA and modeling, appending to competitions/$c/experiments.json, and writing validated new insights back to knowledge/experience.md.

CRITICAL — the previous attempt failed this exact way, do not repeat it: run every training job in the FOREGROUND and wait for it to finish. Do not use '&', nohup, setsid, or any background launch, and never end a turn while a job is still running. This session ends the moment you stop making tool calls, so a backgrounded job dies unfinished and the competition produces nothing. If a full-scale model would not finish inside the budget, shrink it — fewer folds, fewer epochs, a smaller network — until it completes in the foreground.

Constraints:
- STRICT lane isolation: never read or reference anything under ~/ai_agents/aideml*, ~/ai_agents/nvidia-kaggle*, or other agents' outputs. Ignore .earlystop-run/ and .feb-archive/ — quarantined stale runs.
- Threads <= 10. Fix seeds; deterministic=true, force_row_wise=true for LightGBM.
- Budget: about 3 hours of real work; the pipeline decides when it is done.
- Finish by writing competitions/$c/submission.csv (columns/id order per sample_submission.csv) and a 3-line summary at the top of competitions/$c/STATUS.md with the final CV score.

Work autonomously; never ask questions; take documented fallbacks when blocked."

( cd "$BASE" && timeout 21600 /home/tjyen/.local/bin/claude -p "$PROMPT" \
    --dangerously-skip-permissions > "competitions/$c/headless_run.log" 2>&1 )
rc=$?

lane_release
mins=$(( ($(date +%s) - start) / 60 ))
if [ -f "$BASE/competitions/$c/submission.csv" ]; then
  log "RETRY DONE $c: submission written (${mins} min, rc=$rc)"
else
  log "RETRY DONE $c: STILL NO SUBMISSION (${mins} min, rc=$rc)"
fi
