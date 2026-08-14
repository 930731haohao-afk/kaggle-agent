#!/usr/bin/env python3
"""Set the rerun's fresh Kaggle scores beside the frozen AIDE/NVIDIA facts — a table that copies and never computes.

WHY THIS EXISTS. The isolated my-agent rerun produces exactly one new fact per competition:
what the fresh lane scored on Kaggle. The question the rerun exists to answer — did the final
architecture's numbers survive a clean slate — is a comparison against numbers that already
exist: the frozen AIDE and NVIDIA lanes, and my-agent's own original run. Every time this
project has assembled such a table by hand, a number moved in transit; the standing exhibit
is that three_way_scores.csv exists in two diverged copies. So the assembly is owned by this
script, and its one rule is that it copies cells. It never averages, never rescales, never
recomputes a metric, and never fills a gap with anything but the empty string.

THE FROZEN-FACT MERGE RULE. There are two copies of the frozen facts and they are not
identical. Base is the repo copy (benchmark_results/three_way_scores.csv, 24 rows, which also
carries playground-series-s5e1 under its full name and tabular-playground-series-sep-2022).
Overlay is the aideml-runs copy, which fills the s6e7 private columns the base copy left
empty. For any cell empty in base and non-empty in overlay, the overlay value is taken. A
cell non-empty in BOTH with different values is a CONFLICT: every conflicting cell is printed
as comp/column/base/overlay and the build exits 4, because silently preferring either copy is
exactly how divergent copies rot into divergent reports.

WHAT IT DOES NOT DO. It does not score anything, fetch anything, or estimate anything. The
single piece of arithmetic in the file is the winner_priv column: when — and only when — all
three private scores (aide_priv, nvidia_priv, mine_rerun_priv) are present, they are compared
under the LEADERBOARD direction — the frozen row's metric direction (min or max), except
where LB_DIRECTION_OVERRIDES records a known local/LB divergence (s6e1), in which case the
note says so. Any missing operand leaves the winner
blank with the note "incomplete". Blanks stay blank: a competition the scoring step has no
usable row for renders as empty cells, never as 0 or None, and a scores row whose status is
not "scored"/"already-submitted" contributes blanks with its status in the note column.

LIVENESS IS PRINTED, NOT ASSUMED. The summary line is always printed, including zeros:
"comps: N  rerun scored: N  rerun blank: N  conflicts: N". A build that could examine
nothing — missing manifest, missing frozen copy, missing scores without --allow-unscored —
exits 2 rather than writing an empty table that reads as a finished one.

Usage:
    build_rerun_three_way.py                       # scoring-step CSV + both frozen copies
    build_rerun_three_way.py --allow-unscored      # build the frame before the run is scored
    build_rerun_three_way.py --scores S --frozen-base B --frozen-overlay O \
                             --out-csv C --out-md M
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOME = os.path.expanduser("~")

# The manifest is THE definition of which competitions the rerun covers; the table has one
# row per manifest competition and nothing else, so a frozen row for a competition outside
# the manifest can never leak into the deliverable.
MANIFEST = os.path.join(REPO, "docs", "rerun_manifest.json")

DEFAULT_SCORES = os.path.join(HOME, "benchruns", "myagent-rerun.scores.csv")
DEFAULT_BASE = os.path.join(REPO, "benchmark_results", "three_way_scores.csv")
DEFAULT_OVERLAY = os.path.join(os.path.dirname(REPO), "aideml-runs", "three_way_scores.csv")
DEFAULT_OUT = os.path.join(REPO, "benchmark_results", "rerun_three_way.csv")

# Output columns, in the order the table is read: the two frozen reference lanes first, then
# my-agent's OLD numbers as the drift reference, then the rerun's fresh ones, then the one
# computed cell and the reason column.
OUT_COLS = ["comp", "metric", "direction",
            "aide_pub", "aide_priv", "aide_pr", "aide_rank",
            "nvidia_pub", "nvidia_priv", "nvidia_pr", "nvidia_rank",
            "mine_prev_pub", "mine_prev_priv",
            "mine_rerun_pub", "mine_rerun_priv",
            "winner_priv", "note"]

# Scoring-step statuses that mean the public/private cells are real Kaggle numbers. Anything
# else ("missing", "quota", "error", ...) contributes blanks, and the status itself lands in
# the note column so a hole in the table always says why it is a hole.
SCORED_STATUSES = {"scored", "already-submitted"}

# The frozen CSV names playground-series competitions by their tail ("s3e1") and everything
# else by its full slug; the manifest always uses the full slug — and the base copy carries
# playground-series-s5e1 under its FULL name. The naming is reconciled by canonicalizing
# every comp key to the short name AT READ TIME, in read_table: aliased rows for the same
# competition then collide in the merge, where the cell-conflict gate can see them, instead
# of coexisting under two spellings with one silently shadowing the other.
PLAYGROUND_PREFIX = "playground-series-"

# The per-competition node budget of the original benchmark. The conway caveat's ACHIEVED
# count is pulled from the frozen CSV's aide_nodes column, never hardcoded; only the budget
# it is measured against is a constant here.
NODE_BUDGET = "20"

CONWAY_SLUG = "conway-s-reverse-game-of-life"

# The frozen direction column is the LOCAL metric's direction, and for one competition the
# leaderboard disagrees with it: s6e1 is scored locally as R2 (max) but its Kaggle LB privs
# are error-like, lower is better. The precedent is already in this project's record —
# build_three_way_report.py hardcoded lb_minimize=True for s6e1, and the frozen CSV's own
# lb_winner for s6e1 is 'mine' at the LOWEST priv — so crowning winner_priv under the
# frozen direction would award the WORST lane. The override applies ONLY to the winner
# computation; the direction column itself is still copied through untouched, and the row's
# note says which direction the winner used.
LB_DIRECTION_OVERRIDES = {"s6e1": "min"}


def short_name(slug: str) -> str:
    """Manifest slug -> the frozen CSV's short name (strip the playground-series- prefix)."""
    if slug.startswith(PLAYGROUND_PREFIX):
        return slug[len(PLAYGROUND_PREFIX):]
    return slug


def clean_cell(text: str | None) -> str:
    """Strip a raw CSV cell; the literal tokens nan/none (any case) read as EMPTY.

    They are absent values wearing a value's clothes: float('nan') parses, compares false
    against everything, and poisons min/max — so left alone the token would both defeat the
    winner gates and get copied into the deliverable as if it were a number.
    """
    v = (text or "").strip()
    if v.lower() in ("nan", "none"):
        return ""
    return v


def read_table(path: str) -> tuple[dict[str, dict[str, str]], list[tuple[str, str]]]:
    """Read a CSV keyed by the CANONICAL (short) comp name, every cell stripped.

    Canonicalization lives here, at read time, because it is a property of the inputs, not
    of any one lookup: the same competition appears as "playground-series-s5e1" in one copy
    and "s5e1" in another, and a merge keyed on raw spellings would hold both rows without
    ever comparing them.

    A second row for the same canonical key in the SAME file is the merge conflict's
    within-file twin — keeping either row is choosing a side — so it is returned as a
    (comp, path) duplicate for the caller's conflict gate, never resolved by last-row-wins.
    """
    rows: dict[str, dict[str, str]] = {}
    dupes: list[tuple[str, str]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            comp = short_name(clean_cell(rec.get("comp")))
            if not comp:
                continue
            if comp in rows:
                dupes.append((comp, path))
                continue
            rows[comp] = {k: clean_cell(v) for k, v in rec.items() if k}
    return rows, dupes


def merge_frozen(base: dict[str, dict[str, str]],
                 overlay: dict[str, dict[str, str]],
                 ) -> tuple[dict[str, dict[str, str]], list[tuple[str, str, str, str]]]:
    """Overlay fills base's empty cells; a disagreement is returned, never resolved.

    The rule is asymmetric on purpose: base is the repo's copy of record, overlay exists only
    because it holds the s6e7 private scores the base copy never received. Choosing a side on
    a real disagreement is the operator's job — the last time a script chose silently is why
    there are two copies to merge.
    """
    merged = {comp: dict(row) for comp, row in base.items()}
    conflicts: list[tuple[str, str, str, str]] = []
    for comp, orow in overlay.items():
        if comp not in merged:
            # A row only the overlay has: every base cell is (vacuously) empty, so the whole
            # row is an overlay fill.
            merged[comp] = dict(orow)
            continue
        brow = merged[comp]
        for col in sorted(set(brow) | set(orow)):
            if col == "comp":
                continue
            b, o = brow.get(col, ""), orow.get(col, "")
            if b and o and b != o:
                conflicts.append((comp, col, b, o))
            elif not b and o:
                brow[col] = o
    conflicts.sort()
    return merged, conflicts


def md_cell(text: str) -> str:
    """Escape '|' for a Markdown table cell — to the renderer an unescaped pipe is a
    column delimiter, and one stray pipe in a note shifts every cell after it one column
    right. The CSV path never uses this: CSV has its own quoting rules and the cell must
    land there verbatim.
    """
    return text.replace("|", "\\|")


def winner_of(direction: str, privs: list[tuple[str, str]]) -> str:
    """Name the best private score under min|max — the file's only arithmetic.

    Called only once all three operands are non-empty. Ties are reported joined with '=',
    not broken: inventing a tiebreak would be computing a fact the inputs do not contain.
    """
    try:
        vals = [(label, float(text)) for label, text in privs]
    except ValueError:
        return ""
    if any(v != v for _, v in vals):
        # clean_cell already reads the bare nan token as empty, but float() also accepts
        # spellings like '-nan'; a NaN compares false against everything, so min/max over
        # it is garbage — treat it as unparsable, never as a comparable score
        return ""
    best = min(v for _, v in vals) if direction == "min" else max(v for _, v in vals)
    return "=".join(label for label, v in vals if v == best)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", default=DEFAULT_SCORES,
                    help="the scoring step's CSV (comp,file,sha256,message,submitted_at,"
                         "status,public,private,note)")
    ap.add_argument("--frozen-base", default=DEFAULT_BASE,
                    help="the repo copy of the frozen three-way facts")
    ap.add_argument("--frozen-overlay", default=DEFAULT_OVERLAY,
                    help="the aideml-runs copy; fills cells the base copy left empty")
    ap.add_argument("--out-csv", default=DEFAULT_OUT)
    ap.add_argument("--out-md", default=None,
                    help="default: --out-csv with its extension replaced by .md")
    ap.add_argument("--allow-unscored", action="store_true",
                    help="a missing --scores file builds the frame with blank rerun columns "
                         "instead of failing; the frozen columns do not depend on the run")
    a = ap.parse_args(argv)
    out_md = a.out_md or os.path.splitext(a.out_csv)[0] + ".md"

    def bail(msg: str) -> int:
        # The summary line is printed on EVERY exit path, zeros included: an aborted build
        # that prints nothing is indistinguishable from a build nobody started.
        print("comps: 0  rerun scored: 0  rerun blank: 0  conflicts: 0")
        print(msg, file=sys.stderr)
        return 2

    try:
        slugs = sorted(json.load(open(MANIFEST, encoding="utf-8"))["competitions"])
    except (OSError, ValueError, KeyError) as e:
        return bail(f"cannot read manifest {MANIFEST}: {e}")
    if not slugs:
        return bail(f"manifest {MANIFEST} lists no competitions; a table over nothing is "
                    f"not a table")
    for flag, path in (("--frozen-base", a.frozen_base),
                       ("--frozen-overlay", a.frozen_overlay)):
        if not os.path.isfile(path):
            return bail(f"{flag} not found: {path}")

    scores: dict[str, dict[str, str]] = {}
    score_dupes: list[tuple[str, str]] = []
    scores_present = os.path.isfile(a.scores)
    if scores_present:
        scores, score_dupes = read_table(a.scores)
    elif not a.allow_unscored:
        return bail(f"--scores not found: {a.scores} — pass --allow-unscored to build the "
                    f"frame with the rerun columns blank")

    base_rows, base_dupes = read_table(a.frozen_base)
    overlay_rows, overlay_dupes = read_table(a.frozen_overlay)
    merged, conflicts = merge_frozen(base_rows, overlay_rows)
    dupes = base_dupes + overlay_dupes + score_dupes
    if conflicts or dupes:
        for comp, path in dupes:
            print(f"CONFLICT {comp}: more than one row for this comp in {path}")
        for comp, col, b, o in conflicts:
            print(f"CONFLICT {comp}/{col}: base={b}  overlay={o}")
        print(f"comps: {len(slugs)}  rerun scored: 0  rerun blank: 0  "
              f"conflicts: {len(conflicts) + len(dupes)}")
        print("the frozen copies disagree — across the two copies or within one file; "
              "refusing to pick a side — reconcile them first, because a silent "
              "preference is how divergent copies rot into divergent reports",
              file=sys.stderr)
        return 4

    out_rows: list[dict[str, str]] = []
    scored_n = 0
    for slug in slugs:
        frozen = merged.get(short_name(slug), {})
        notes: list[str] = []
        if not frozen:
            notes.append("no frozen row")

        rerun_pub = rerun_priv = ""
        srow = scores.get(short_name(slug))
        if srow is not None:
            status = srow.get("status", "")
            if status in SCORED_STATUSES:
                scored_n += 1
                rerun_pub = srow.get("public", "")
                rerun_priv = srow.get("private", "")
            else:
                notes.append(status or "no status")

        direction = frozen.get("direction", "")
        lb_direction = LB_DIRECTION_OVERRIDES.get(short_name(slug), direction)
        privs = [("aide", frozen.get("aide_priv", "")),
                 ("nvidia", frozen.get("nvidia_priv", "")),
                 ("mine", rerun_priv)]
        if direction in ("min", "max") and all(v for _, v in privs):
            winner = winner_of(lb_direction, privs)
            if not winner:
                notes.append("unparsable priv")
            elif lb_direction != direction:
                notes.append(f"winner uses LB direction {lb_direction} "
                             f"(local metric is {direction})")
        else:
            winner = ""
            notes.append("incomplete")

        out_rows.append({
            "comp": slug,
            "metric": frozen.get("metric", ""),
            "direction": direction,
            "aide_pub": frozen.get("aide_pub", ""),
            "aide_priv": frozen.get("aide_priv", ""),
            "aide_pr": frozen.get("aide_pr", ""),
            "aide_rank": frozen.get("aide_rank", ""),
            "nvidia_pub": frozen.get("nvidia_pub", ""),
            "nvidia_priv": frozen.get("nvidia_priv", ""),
            "nvidia_pr": frozen.get("nvidia_pr", ""),
            "nvidia_rank": frozen.get("nvidia_rank", ""),
            "mine_prev_pub": frozen.get("mine_pub", ""),
            "mine_prev_priv": frozen.get("mine_priv", ""),
            "mine_rerun_pub": rerun_pub,
            "mine_rerun_priv": rerun_priv,
            "winner_priv": winner,
            "note": "; ".join(notes),
        })

    out_dir = os.path.dirname(os.path.abspath(a.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(out_rows)

    # The conway caveat's achieved node count comes from the frozen facts, not from memory:
    # a caveat that restates a number from the author's head is the fabrication class this
    # script exists to prevent.
    nodes = merged.get(CONWAY_SLUG, {}).get("aide_nodes", "")
    if nodes:
        conway_tail = f"at {nodes}/{NODE_BUDGET} nodes (aide_nodes column of the frozen CSV)"
    else:
        conway_tail = ("before its node budget (aide_nodes is empty in the frozen input "
                       "given to this build, so the count is not restated here)")

    scores_note = "" if scores_present else \
        " (absent; built with --allow-unscored, rerun columns blank)"
    md = [
        "# my-agent isolated rerun — three-lane comparison",
        "",
        "Generated from:",
        "",
        f"- rerun scores: `{a.scores}`{scores_note}",
        f"- frozen base: `{a.frozen_base}`",
        f"- frozen overlay: `{a.frozen_overlay}`",
        "",
        "AIDE/NVIDIA columns and mine_prev_* are copied from the frozen facts; "
        "mine_rerun_* are copied from the scoring step; winner_priv is the only computed "
        "cell and blanks mean the input was blank.",
        "",
        "| " + " | ".join(OUT_COLS) + " |",
        "|" + "---|" * len(OUT_COLS),
    ]
    md.extend("| " + " | ".join(md_cell(r[c]) for c in OUT_COLS) + " |" for r in out_rows)
    md += [
        "",
        "## Caveats",
        "",
        "- AIDE and NVIDIA numbers are frozen from the original benchmark runs and were "
        "not re-run alongside this rerun.",
        "- The experience library serves my-agent's own recorded percentile ranks for "
        "OTHER competitions — a designed channel, self-filtered for the lane's own "
        "competition — and the reference lanes have no equivalent.",
        f"- The {CONWAY_SLUG} AIDE result hit the step timeout {conway_tail}.",
        "- s6e1's leaderboard metric direction differs from the local metric (local R2 "
        "max, LB RMSE-like min).",
        "",
    ]
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    blank_n = len(slugs) - scored_n
    print(f"comps: {len(slugs)}  rerun scored: {scored_n}  rerun blank: {blank_n}  "
          f"conflicts: 0")
    print(f"wrote {a.out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
