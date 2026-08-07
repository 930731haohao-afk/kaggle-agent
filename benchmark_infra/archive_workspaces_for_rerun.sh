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
# data/ is kept: it symlinks to the manifest-verified clean root. config.yaml is kept.
# rules_verdict.json is kept IF present -- it is a Stage 0.5 input, not an output.
KEEP_RE='^(data|config\.yaml|rules_verdict\.json)$'

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
echo "== 1/3 workspace artifacts =="
for c in "${COMPS[@]}"; do
  ws="competitions/$c"
  [ -d "$ws" ] || { echo "MISSING $ws"; continue; }
  while IFS= read -r entry; do
    base="$(basename "$entry")"
    [[ "$base" =~ $KEEP_RE ]] && continue
    move "$entry"; total=$((total+1))
  done < <(find "$ws" -mindepth 1 -maxdepth 1)
  # derived workspaces of this competition (repeat runs, v5 arms, leftovers) go whole
  for d in competitions/"$c".* competitions/"$c"-v5-*; do
    [ -e "$d" ] || continue
    move "$d"; total=$((total+1))
  done
done

echo "== 2/3 tree_search caches and generated arm evaluators =="
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

echo "== 3/3 launcher state =="
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
fi
