#!/usr/bin/env bash
# Archive the 20 benchmark workspaces (and everything else a fresh run could warm-start
# from) ahead of the 20-competition re-run.
#
# WHY EACH PIECE EXISTS:
#   1. run_myagent_headless.sh SKIPs any competition whose submission.csv exists; all 20
#      have one from the recorded run, so an un-archived relaunch skips everything and
#      reports success in minutes (2026-08-04 finding).
#   2. Stale caches warm-start a "fresh" run off its own previous results -- the exact
#      contamination that voided AIDE's cells. This includes the NON-obvious location:
#      tree_search/cache_*/ holds OOF vectors keyed by NODE ID, node ids restart at 0 in
#      every new tree, and the identity stamp that would catch the collision has no
#      callers (bucket-A #25/#40/#52) -- so a fresh tree would silently score the OLD
#      tree's vectors.
#   3. Committed experiments_tree*.json files make every driver "resume" a tree whose
#      search_state says stopped: zero nodes evaluated, instant DONE (bucket-A #51).
#
# The archive is a MOVE, not a delete: everything lands under
#   archive/rerun-baseline-<stamp>/ preserving relative paths, and git history holds the
# committed files regardless. Still destructive to the working tree -- run only with the
# user's explicit go-ahead.
#
# Usage:
#   bash benchmark_infra/archive_workspaces_for_rerun.sh            # DRY RUN (default)
#   bash benchmark_infra/archive_workspaces_for_rerun.sh --execute  # actually move
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

MODE="${1:---dry-run}"
STAMP="$(date +%Y%m%d-%H%M)"
DEST="archive/rerun-baseline-${STAMP}"

COMPS=(
  playground-series-s3e1 playground-series-s3e3 playground-series-s3e5
  playground-series-s3e7 playground-series-s3e9 playground-series-s3e11
  playground-series-s3e14 playground-series-s3e16 playground-series-s3e19
  playground-series-s4e11 playground-series-s5e10 playground-series-s6e1
  playground-series-s6e2 afsis-soil-properties cat-in-the-dat
  conway-s-reverse-game-of-life tabular-playground-series-aug-2022
  tabular-playground-series-jan-2022 playground-series-s5e1
  tabular-playground-series-sep-2022
)

# Inside each workspace, KEEP (the run's inputs) vs ARCHIVE (the previous run's outputs).
# config.yaml is kept. rules_verdict.json is kept IF present -- a Stage 0.5 input, not an
# output. data/ is kept but PRUNED, not trusted wholesale: only 5 of the 20 workspaces
# symlink to the manifest-verified clean root; the other 15 are real directories, and six of
# them held the recorded run's own OOF/test prediction matrices, its Optuna champion
# parameters and its engineered feature tables (2026-08-10 round-8). A file inside data/
# survives only if the same name exists in ~/ai_agents/bench-comps/<comp>/data -- that root
# IS the definition of "the competition's own inputs".
# data_official/ is kept for a different reason: every entry of the SHARED clean root
# ~/ai_agents/bench-comps/<comp>/data is a symlink into it, so archiving it would leave
# all three agents' data root a farm of dangling links (2026-08-10 round-9).
KEEP_RE='^(data|data_official|config\.yaml|rules_verdict\.json)$'
CLEAN_ROOT="$HOME/ai_agents/bench-comps"

move() {  # move $1 under DEST preserving its relative path
  local src="$1" dst="${DEST}/$1"
  if [ "$MODE" = "--execute" ]; then
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
  else
    echo "DRY  mv $src -> $dst"
  fi
}

total=0
echo "== 1/5 workspace artifacts =="
for c in "${COMPS[@]}"; do
  ws="competitions/$c"
  [ -d "$ws" ] || { echo "MISSING $ws"; continue; }
  while IFS= read -r entry; do
    base="$(basename "$entry")"
    [[ "$base" =~ $KEEP_RE ]] && continue
    move "$entry"; total=$((total+1))
  done < <(find "$ws" -mindepth 1 -maxdepth 1)
  # data/ pruning: keep only what the clean root also has. Real directories only -- a
  # symlinked data/ IS the clean root and must never be written through.
  if [ -d "$ws/data" ] && [ ! -L "$ws/data" ]; then
    if [ -d "$CLEAN_ROOT/$c/data" ]; then
      while IFS= read -r f; do
        base="$(basename "$f")"
        [ -e "$CLEAN_ROOT/$c/data/$base" ] && continue
        move "$f"; total=$((total+1))
      done < <(find "$ws/data" -mindepth 1 -maxdepth 1)
    else
      echo "WARN  no clean root for $c -- data/ left untouched, inspect by hand"
    fi
  fi
  # derived workspaces of this competition (repeat runs, v5 arms, leftovers) go whole
  for d in competitions/"$c".* competitions/"$c"-v5-*; do
    [ -e "$d" ] || continue
    move "$d"; total=$((total+1))
  done
done

echo "== 2/5 sibling workspaces and loose result files =="
# Non-benchmark workspaces are not inputs to anything, and four of them hold verbatim
# snapshots of the pre-redesign experience.md naming benchmark competitions with their own
# scores -- including one Kaggle PRIVATE LB. The self-improvement skill prescribes scanning
# `competitions/*/experiments.json`, and after this archive that glob resolves to exactly
# those survivors (2026-08-10 round-8). They go whole.
while IFS= read -r d; do
  base="$(basename "$d")"
  [ "$base" = "__pycache__" ] && { move "$d"; total=$((total+1)); continue; }
  keep=0
  for c in "${COMPS[@]}"; do
    case "$base" in "$c"|"$c".*|"$c"-v5-*) keep=1; break;; esac
  done
  [ "$keep" = 1 ] && continue
  move "$d"; total=$((total+1))
done < <(find competitions -mindepth 1 -maxdepth 1 -type d)
# Loose files at the root of competitions/ match none of the per-workspace patterns:
# _batch_results.json and _batch.log are the recorded batch run's per-competition metric,
# baseline score and winning blend weights for 8 lanes -- 7 of them byte-identical to the
# frozen tier-1 reference the re-run is scored against.
for f in competitions/_batch_results.json competitions/_batch.log; do
  [ -e "$f" ] || continue
  move "$f"; total=$((total+1))
done

echo "== 3/5 tree_search caches and generated arm evaluators =="
for d in tree_search/cache_*; do
  [ -e "$d" ] || continue
  move "$d"; total=$((total+1))
done
# arm evaluators are generated per run by make_v5_arm; a stale one shadows a fresh build
for f in tree_search/eval_ratio_* tree_search/eval_featurejoin_* tree_search/eval_*_v5_* \
         tree_search/eval_ratiogated_* tree_search/eval_ratioconst_*; do
  [ -e "$f" ] || continue
  move "$f"; total=$((total+1))
done

echo "== 4/5 frozen per-competition tree-search artifacts =="
# Round 6 demoted the dedicated run_<comp>_v3.py drivers because they embed their lane's
# recorded champion. Their DATA siblings were never touched: llm_proposer_input_<slug>.json
# carries that lane's champion metric AND champion config, 24-63 evaluated node scores, and
# a pre-redesign snapshot of experience.md whose bullets cite the lane itself (one of them
# quotes a Kaggle private LB). wire_<slug>_*.json exposes champion_metric as a top-level
# key. Every lane enters tree_search/ to copy run_template_v3.py, so a file named for its
# own competition sits in the listing (2026-08-10 round-8).
for f in tree_search/llm_proposer_input_*.json tree_search/llm_proposals_*.json \
         tree_search/wire_*.json; do
  [ -e "$f" ] || continue
  move "$f"; total=$((total+1))
done

echo "== 5/6 per-competition reproduction drivers and loose tree files =="
# Round 6 un-pinned the dedicated run_<comp>_v3.py drivers from the manifest but left them in
# the directory every lane enters to copy run_template_v3.py. They embed their competition's
# recorded root config, champion parameters and digit-verify targets. Keep only the
# competition-agnostic tools; the pinned evaluators (manifest 'eval') also stay, redacted.
KEEP_TREE_TOOLS='^(run_template_v3|run_v3_generic|run_arm_search|run_faithful_v3|run_wire_v3)\.py$'
mapfile -t PINNED_EVAL < <(python3 -c "
import json
print('\n'.join(sorted(s['eval'] for s in json.load(open('docs/rerun_manifest.json'))['competitions'].values())))")
while IFS= read -r f; do
  base="$(basename "$f")"
  [[ "$base" =~ $KEEP_TREE_TOOLS ]] && continue
  skip=0
  for e in "${PINNED_EVAL[@]}"; do [ "$base" = "$e" ] && { skip=1; break; }; done
  [ "$skip" = 1 ] && continue
  move "$f"; total=$((total+1))
done < <(find tree_search -mindepth 1 -maxdepth 1 -type f \
           \( -name 'run_*.py' -o -name 'tree_*.json' -o -name 'smoke_*.json' \
              -o -name 'report_*.json' -o -name '*_probe*.json' \
              -o -name 'injection_bootstrap*.json' -o -name 'structural_probe.py' \
              -o -name 'eval_*.py' -o -name '*.log' \) | sort)

echo "== 6/6 the research record =="
# docs/, benchmark_results/ and the plan files hold every competition's scores for all three
# lanes -- the frozen references the re-run is measured against, the three-way tables, the
# per-competition reports. Nothing in the kaggle-agent skill reads them during a run, and no
# rule kept a lane out of them: they simply sat in the working tree (2026-08-10 round-8,
# found by benchmark_infra/verify_clean_slate.py, which round 8's attackers did not reach).
# Root STATUS.md got this treatment in round 7 for the same reason; instruction-only
# protection is what kept failing.
DOCS_KEEP='^(rerun_manifest\.json|reproducibility\.md)$'
while IFS= read -r f; do
  base="$(basename "$f")"
  [[ "$base" =~ $DOCS_KEEP ]] && continue
  move "$f"; total=$((total+1))
done < <(find docs -mindepth 1 -maxdepth 1 | sort)
# documents/ is a near-homograph of docs/ that no earlier glob mentioned: it holds a
# CV / Public LB / Private LB / winning-technique table for 9 of the 20 lanes. mlflow.db
# mirrors 16 lanes' metric, score history and hyper-parameters (2026-08-10 round-9).
# *.drawio figures at the root plot per-competition score lineages; the sibling
# self-improvement skill quotes 17 per-lane values (2026-08-10 round-10).
for d in benchmark_results .superpowers documents mlflow.db search-tree-states.drawio \
         cat-lineage-growth.drawio .claude/skills/kaggle-agent-self-improvement; do
  [ -e "$d" ] || continue
  move "$d"; total=$((total+1))
done

echo "== launcher state =="
for f in MYAGENT_LANES_STATUS.md myagent_lanes.log; do
  [ -e "$f" ] || continue
  move "$f"; total=$((total+1))
done

echo
echo "$total item(s) $( [ "$MODE" = "--execute" ] && echo moved || echo would move ) -> $DEST"
if [ "$MODE" != "--execute" ]; then
  echo "(dry run; pass --execute to perform the archive)"
else
  {
    echo "archived: $(date -Iseconds)"
    echo "reason  : clean-slate for the 20-competition re-run (docs/rerun_manifest.json)"
    echo "restore : mv the contents back to the repo root, preserving relative paths"
  } > "$DEST/ARCHIVE_MANIFEST.txt"
  echo "manifest -> $DEST/ARCHIVE_MANIFEST.txt"
  echo
  echo "== verifying the clean slate =="
  # The glob lists above are the mechanism; this is the check. Five audit rounds running,
  # a hand-maintained glob list missed a file the next round found.
  python3 benchmark_infra/verify_clean_slate.py || {
    echo "ARCHIVE INCOMPLETE — the re-run must not start until this passes." >&2
    exit 3
  }
fi
