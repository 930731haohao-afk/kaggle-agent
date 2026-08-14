#!/usr/bin/env python3
"""End-to-end pins for build_rerun_three_way.py — the copy-only three-lane comparison table.

The builder's whole contract is negative: it copies cells and refuses to invent. Each test
here drives the CLI the way the operator does (subprocess, the REAL manifest, handwritten
frozen fixtures in tmp_path) and pins one way that contract has historically been broken in
this project: a silently chosen side of a diverged frozen copy, a blank rendered as a zero,
a winner computed from two lanes instead of three, a caveat dropped from the deliverable.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "benchmark_infra", "build_rerun_three_way.py")

# The frozen copies' real header — fixtures use it verbatim so a drift between the builder's
# expectations and the actual files cannot hide behind a simplified fixture schema.
FROZEN_COLS = ("comp,metric,direction,mine_local,nvidia_local,aide_local,aide_nodes,"
               "mine_pub,mine_priv,mine_pr,mine_rank,nvidia_pub,nvidia_priv,nvidia_pr,"
               "nvidia_rank,aide_pub,aide_priv,aide_pr,aide_rank,local_winner,"
               "lb_winner").split(",")
# The scoring step's header, likewise verbatim.
SCORES_COLS = "comp,file,sha256,message,submitted_at,status,public,private,note".split(",")


def _csv_text(cols: list[str], rows: list[dict[str, str]]) -> str:
    lines = [",".join(cols)]
    lines.extend(",".join(r.get(c, "") for c in cols) for r in rows)
    return "\n".join(lines) + "\n"


def _manifest_size() -> int:
    with open(os.path.join(REPO, "docs", "rerun_manifest.json"), encoding="utf-8") as fh:
        return len(json.load(fh)["competitions"])


class TestRerunThreeWay:
    def _build(self, tmp_path, base_rows, overlay_rows, scores_rows=None, extra=()):
        """Run the builder end-to-end; scores_rows=None means the scores file does not exist.

        --scores is ALWAYS passed explicitly: the default points at the real scoring-step
        output, and a test that silently reads it would pass or fail with the state of the
        machine rather than the state of the code.
        """
        base = tmp_path / "base.csv"
        overlay = tmp_path / "overlay.csv"
        base.write_text(_csv_text(FROZEN_COLS, base_rows), encoding="utf-8")
        overlay.write_text(_csv_text(FROZEN_COLS, overlay_rows), encoding="utf-8")
        out_csv = tmp_path / "out.csv"
        args = [sys.executable, SCRIPT,
                "--frozen-base", str(base), "--frozen-overlay", str(overlay),
                "--out-csv", str(out_csv)]
        if scores_rows is None:
            args += ["--scores", str(tmp_path / "no-such-scores.csv")]
        else:
            sp = tmp_path / "scores.csv"
            sp.write_text(_csv_text(SCORES_COLS, scores_rows), encoding="utf-8")
            args += ["--scores", str(sp)]
        args += list(extra)
        r = subprocess.run(args, capture_output=True, text=True, cwd=REPO, check=False)
        return r.returncode, r.stdout + r.stderr, out_csv, tmp_path / "out.md"

    @staticmethod
    def _rows(out_csv) -> dict[str, dict[str, str]]:
        """Read the output table — and screen EVERY cell of EVERY test's output: the
        literal tokens nan/None (any case) are absent values wearing a value's clothes,
        and no build under any fixture may ever emit one."""
        with open(out_csv, encoding="utf-8", newline="") as fh:
            rows = {r["comp"]: r for r in csv.DictReader(fh)}
        for comp, r in rows.items():
            for col, val in r.items():
                assert val.lower() not in ("nan", "none"), (comp, col, val)
        return rows

    def test_overlay_fills_a_cell_the_base_copy_left_empty(self, tmp_path):
        """The s6e7 shape: the repo copy's private columns are empty and only the
        aideml-runs copy holds the values. A merge that ignores the overlay silently
        reports the private scores as never having existed."""
        base = [{"comp": "s3e3", "metric": "AUC", "direction": "max",
                 "aide_pub": "0.88484"}]
        overlay = [{"comp": "s3e3", "metric": "AUC", "direction": "max",
                    "aide_pub": "0.88484", "aide_priv": "0.86863"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, overlay,
                                            extra=("--allow-unscored",))
        assert rc == 0, out[-800:]
        row = self._rows(out_csv)["playground-series-s3e3"]
        assert row["aide_priv"] == "0.86863", row
        assert row["aide_pub"] == "0.88484", row

    def test_divergent_frozen_copies_are_a_conflict_not_a_choice(self, tmp_path):
        """Two copies of the same fact with different values is how numbers rot: whichever
        copy a script silently prefers becomes 'the' number in the next report. The builder
        must print BOTH values with their address and exit 4, and build nothing."""
        base = [{"comp": "s3e3", "metric": "AUC", "direction": "max",
                 "aide_priv": "0.111"}]
        overlay = [{"comp": "s3e3", "metric": "AUC", "direction": "max",
                    "aide_priv": "0.222"}]
        rc, out, out_csv, md = self._build(tmp_path, base, overlay,
                                           extra=("--allow-unscored",))
        assert rc == 4, out[-800:]
        assert "s3e3" in out and "aide_priv" in out, out[-800:]
        assert "0.111" in out and "0.222" in out, out[-800:]
        assert "conflicts: 1" in out, out[-800:]
        assert not out_csv.exists() and not md.exists(), \
            "a conflicted build must not leave a table behind for someone to trust"

    def test_aliased_comp_names_collide_into_the_conflict_gate(self, tmp_path):
        """The base copy carries s5e1 under its full slug and the overlay under the short
        name. If the merge keys on the raw comp strings, the two rows never meet, the
        conflict gate never fires, and the overlay's value silently wins under the alias —
        the exact silent preference the gate exists to forbid. Canonicalized at read time,
        the aliased rows must collide: rc 4 with both values printed."""
        base = [{"comp": "playground-series-s5e1", "metric": "MAPE", "direction": "min",
                 "aide_pub": "0.12573"}]
        overlay = [{"comp": "s5e1", "metric": "MAPE", "direction": "min",
                    "aide_pub": "0.99999"}]
        rc, out, out_csv, md = self._build(tmp_path, base, overlay,
                                           extra=("--allow-unscored",))
        assert rc == 4, out[-800:]
        assert "aide_pub" in out, out[-800:]
        assert "0.12573" in out and "0.99999" in out, out[-800:]
        assert not out_csv.exists() and not md.exists(), \
            "a conflicted build must not leave a table behind for someone to trust"

    def test_a_duplicate_comp_row_within_one_file_is_a_conflict_naming_the_file(self, tmp_path):
        """Two rows for the same competition inside ONE frozen copy — under two spellings
        that canonicalize to the same key — is the merge conflict's within-file twin, and
        csv.DictReader-into-dict resolves it by silent last-row-wins. The builder must
        refuse exactly like the cross-copy case: rc 4, and the message must name the file
        so the operator knows which copy to repair."""
        base = [{"comp": "playground-series-s5e1", "metric": "MAPE", "direction": "min",
                 "aide_pub": "0.12573"},
                {"comp": "s5e1", "metric": "MAPE", "direction": "min",
                 "aide_pub": "0.99999"}]
        rc, out, out_csv, md = self._build(tmp_path, base, [],
                                           extra=("--allow-unscored",))
        assert rc == 4, out[-800:]
        assert "s5e1" in out and "base.csv" in out, out[-800:]
        assert not out_csv.exists() and not md.exists(), \
            "a conflicted build must not leave a table behind for someone to trust"

    def test_manifest_slug_finds_the_frozen_row_under_its_short_name(self, tmp_path):
        """The frozen CSV says 's3e3' where the manifest says 'playground-series-s3e3'. A
        lookup that misses the mapping renders 20 rows of 'no frozen row' — a table that
        looks built and contains nothing."""
        base = [{"comp": "s3e3", "metric": "AUC", "direction": "max",
                 "mine_pub": "0.89262", "mine_priv": "0.87303"},
                {"comp": "cat-in-the-dat", "metric": "AUC", "direction": "max",
                 "aide_pub": "0.80790"},
                # ...except s5e1, which the base copy carries under its FULL name;
                # read-time canonicalization must fold it onto the short key so the
                # lookup finds it rather than declaring absence.
                {"comp": "playground-series-s5e1", "metric": "MAPE", "direction": "min",
                 "aide_pub": "0.12573"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [],
                                            extra=("--allow-unscored",))
        assert rc == 0, out[-800:]
        rows = self._rows(out_csv)
        got = rows["playground-series-s3e3"]
        assert got["metric"] == "AUC" and got["mine_prev_pub"] == "0.89262", got
        assert got["mine_prev_priv"] == "0.87303", got
        assert rows["cat-in-the-dat"]["aide_pub"] == "0.80790", rows["cat-in-the-dat"]
        assert rows["playground-series-s5e1"]["aide_pub"] == "0.12573", \
            rows["playground-series-s5e1"]

    def test_blanks_stay_blank_never_zero_never_none(self, tmp_path):
        """A blank rendered as 0 is a fabricated score, and str(None) in a cell is the same
        defect wearing Python's clothes. With no scores file and a frozen row with empty LB
        cells, every unknown must land in the table as the empty string."""
        base = [{"comp": "s3e5", "metric": "QWK", "direction": "max"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [],
                                            extra=("--allow-unscored",))
        assert rc == 0, out[-800:]
        rows = self._rows(out_csv)
        row = rows["playground-series-s3e5"]
        for col in ("mine_rerun_pub", "mine_rerun_priv", "aide_priv", "nvidia_priv",
                    "mine_prev_pub", "winner_priv"):
            assert row[col] == "", (col, row)
        # no cell anywhere in the table renders an absent value as a numeral or a Python
        # literal; the comp column is excluded because slugs legitimately contain digits
        for comp, r in rows.items():
            for col, val in r.items():
                if col != "comp":
                    assert val not in ("0", "None", "nan", "NaN"), (comp, col, val)

    def test_a_literal_nan_is_an_absent_value_not_a_score(self, tmp_path):
        """float('nan') parses, compares false against everything, and min/max silently
        return garbage around it — so a 'NaN' cell that reaches the winner computation
        defeats both the all-three-lanes gate and the direction comparison, and the token
        itself gets copied into the deliverable as if it were a number. Any-case nan/none
        must be read as EMPTY: the frozen priv lands blank in the table, the winner stays
        blank with 'incomplete', and the token never appears in the output."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min",
                 "aide_priv": "NaN", "nvidia_priv": "0.4"},
                {"comp": "s3e3", "metric": "AUC", "direction": "max",
                 "aide_priv": "0.5", "nvidia_priv": "0.4"}]
        scores = [{"comp": "playground-series-s3e1", "status": "scored",
                   "public": "0.35", "private": "0.3"},
                  # the scores input is normalized too: a scored row whose private reads
                  # 'nan' contributes a blank cell, not a token
                  {"comp": "playground-series-s3e3", "status": "scored",
                   "public": "0.3", "private": "nan"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [], scores)
        assert rc == 0, out[-800:]
        rows = self._rows(out_csv)
        frozen_nan = rows["playground-series-s3e1"]
        assert frozen_nan["aide_priv"] == "", frozen_nan
        assert frozen_nan["winner_priv"] == "", frozen_nan
        assert "incomplete" in frozen_nan["note"], frozen_nan
        scored_nan = rows["playground-series-s3e3"]
        assert scored_nan["mine_rerun_priv"] == "", scored_nan
        assert scored_nan["winner_priv"] == "", scored_nan
        assert "incomplete" in scored_nan["note"], scored_nan

    def test_winner_priv_needs_all_three_lanes_and_respects_direction(self, tmp_path):
        """A winner computed from two lanes crowns whoever showed up — the comparison is
        three-way or it is nothing. And min/max must come from the frozen direction column:
        a hardcoded max silently inverts every error-metric competition."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min",
                 "aide_priv": "0.5", "nvidia_priv": "0.4"},
                {"comp": "s3e3", "metric": "AUC", "direction": "max",
                 "aide_priv": "0.5", "nvidia_priv": "0.4"},
                {"comp": "s3e5", "metric": "QWK", "direction": "max",
                 "aide_priv": "0.5"}]  # nvidia_priv deliberately absent
        scores = [{"comp": "playground-series-s3e1", "status": "scored",
                   "public": "0.35", "private": "0.3"},
                  {"comp": "playground-series-s3e3", "status": "already-submitted",
                   "public": "0.3", "private": "0.3"},
                  {"comp": "playground-series-s3e5", "status": "scored",
                   "public": "0.6", "private": "0.6"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [], scores)
        assert rc == 0, out[-800:]
        rows = self._rows(out_csv)
        assert rows["playground-series-s3e1"]["winner_priv"] == "mine", \
            rows["playground-series-s3e1"]
        assert rows["playground-series-s3e3"]["winner_priv"] == "aide", \
            rows["playground-series-s3e3"]
        incomplete = rows["playground-series-s3e5"]
        assert incomplete["winner_priv"] == "", incomplete
        assert "incomplete" in incomplete["note"], incomplete
        # an already-submitted row is a scored row: the numbers were copied
        assert rows["playground-series-s3e3"]["mine_rerun_pub"] == "0.3", \
            rows["playground-series-s3e3"]
        n = _manifest_size()
        assert f"comps: {n}  rerun scored: 3  rerun blank: {n - 3}  conflicts: 0" in out, \
            out[-800:]

    def test_s6e1_winner_is_crowned_under_the_lb_direction_not_the_local_one(self, tmp_path):
        """s6e1's frozen direction column says max because the LOCAL metric is R2 — but its
        Kaggle LB privs are error-like, lower is better (build_three_way_report.py hardcoded
        lb_minimize=True for exactly this comp, and the frozen CSV's own lb_winner for s6e1
        is 'mine' at the LOWEST priv). A winner_priv computed under the frozen direction
        crowns the WORST lane. With the real frozen privs — aide 8.72191, nvidia 8.73239 —
        and a rerun priv of 8.70999, the winner must be mine, and the note must say the
        winner used the LB direction."""
        base = [{"comp": "s6e1", "metric": "R2", "direction": "max",
                 "aide_priv": "8.72191", "nvidia_priv": "8.73239"}]
        scores = [{"comp": "playground-series-s6e1", "status": "scored",
                   "public": "8.69931", "private": "8.70999"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [], scores)
        assert rc == 0, out[-800:]
        row = self._rows(out_csv)["playground-series-s6e1"]
        assert row["winner_priv"] == "mine", row
        # the direction column itself is copied through untouched — the override bends
        # only the winner computation
        assert row["direction"] == "max", row
        assert "winner uses LB direction min (local metric is max)" in row["note"], row

    def test_an_unscored_status_contributes_blanks_and_its_status_as_the_note(self, tmp_path):
        """A scores row whose status is 'missing' may still carry numbers in its cells
        (leftovers from a retry, a hand edit). Copying them would launder an unscored
        submission into a scored one; the row must contribute blanks and say why."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min",
                 "aide_priv": "0.5", "nvidia_priv": "0.4"}]
        scores = [{"comp": "playground-series-s3e1", "status": "missing",
                   "public": "9.9", "private": "9.9"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [], scores)
        assert rc == 0, out[-800:]
        row = self._rows(out_csv)["playground-series-s3e1"]
        assert row["mine_rerun_pub"] == "" and row["mine_rerun_priv"] == "", row
        assert "missing" in row["note"], row
        assert row["winner_priv"] == "" and "incomplete" in row["note"], row
        assert "9.9" not in out_csv.read_text(encoding="utf-8")
        n = _manifest_size()
        assert f"comps: {n}  rerun scored: 0  rerun blank: {n}  conflicts: 0" in out, \
            out[-800:]

    def test_the_output_schema_is_pinned_to_the_exact_seventeen_columns(self, tmp_path):
        """Downstream readers address this table by column name and position; a renamed,
        reordered, or added column silently re-wires every one of them. The raw header
        line — not a parsed, order-forgiving view of it — is the contract."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min"}]
        rc, out, out_csv, _md = self._build(tmp_path, base, [],
                                            extra=("--allow-unscored",))
        assert rc == 0, out[-800:]
        header = out_csv.read_text(encoding="utf-8").splitlines()[0]
        assert header == ("comp,metric,direction,"
                          "aide_pub,aide_priv,aide_pr,aide_rank,"
                          "nvidia_pub,nvidia_priv,nvidia_pr,nvidia_rank,"
                          "mine_prev_pub,mine_prev_priv,"
                          "mine_rerun_pub,mine_rerun_priv,"
                          "winner_priv,note"), header
        with open(out_csv, encoding="utf-8", newline="") as fh:
            assert len(csv.DictReader(fh).fieldnames) == 17

    def test_the_md_carries_the_caveats_including_the_library_asymmetry(self, tmp_path):
        """The caveats ARE the comparison's validity conditions: frozen reference lanes, the
        experience-library percentile channel the reference lanes lack, conway's truncated
        node budget, s6e1's flipped LB direction. A table that ships without them reads as a
        fair fight it was not. The conway node count must come from the frozen CSV."""
        base = [{"comp": "conway-s-reverse-game-of-life", "metric": "MAE",
                 "direction": "min", "aide_nodes": "16"}]
        rc, out, _out_csv, md = self._build(tmp_path, base, [],
                                            extra=("--allow-unscored",))
        assert rc == 0, out[-800:]
        text = md.read_text(encoding="utf-8")
        assert "## Caveats" in text, text[-800:]
        assert "frozen from the original benchmark runs" in text
        assert "not re-run alongside this rerun" in text
        assert "experience library" in text and "percentile ranks" in text, text[-1200:]
        assert "no equivalent" in text, text[-1200:]
        assert "16/20 nodes" in text, text[-1200:]
        assert "s6e1" in text and "R2" in text, text[-1200:]
        # the header must say where the numbers came from
        assert str(tmp_path / "base.csv") in text and str(tmp_path / "overlay.csv") in text

    def test_a_pipe_in_a_cell_is_escaped_in_the_md_and_untouched_in_the_csv(self, tmp_path):
        """A '|' inside a Markdown table cell is a column delimiter to the renderer: one
        stray pipe in a status or note shifts every cell after it one column right, and the
        row silently reads as different facts. The MD rendering must escape it as '\\|';
        the CSV, which has its own quoting rules, must carry the cell verbatim."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min"}]
        scores = [{"comp": "playground-series-s3e1", "status": "quota|exceeded"}]
        rc, out, out_csv, md = self._build(tmp_path, base, [], scores)
        assert rc == 0, out[-800:]
        assert "quota|exceeded" in self._rows(out_csv)["playground-series-s3e1"]["note"]
        text = md.read_text(encoding="utf-8")
        assert "quota\\|exceeded" in text, text[-1200:]
        line = next(ln for ln in text.splitlines()
                    if ln.startswith("| playground-series-s3e1 "))
        # 17 cells means exactly 18 unescaped structural pipes — one more and the
        # renderer has invented an 18th column out of the note's content
        assert len(re.findall(r"(?<!\\)\|", line)) == 18, line

    def test_a_missing_scores_file_without_the_flag_is_rc2(self, tmp_path):
        """Building the table with silently-blank rerun columns is a legitimate ASK
        (--allow-unscored, for framing the table before the run is scored) but a terrible
        DEFAULT: a typo'd --scores path would produce a complete-looking table that says
        the rerun scored nothing."""
        base = [{"comp": "s3e1", "metric": "RMSE", "direction": "min"}]
        rc, out, out_csv, md = self._build(tmp_path, base, [])
        assert rc == 2, out[-800:]
        assert "--allow-unscored" in out, out[-800:]
        assert not out_csv.exists() and not md.exists()
        # the summary line is printed on every exit path, zeros included
        assert "rerun scored: 0" in out and "conflicts: 0" in out, out[-800:]

    def test_a_missing_frozen_copy_is_rc2(self, tmp_path):
        """Half the frozen facts missing is not a smaller table, it is no table: the merge
        rule is defined over both copies, and building from one alone silently answers the
        conflict question with 'there were none'."""
        base = tmp_path / "base.csv"
        base.write_text(_csv_text(FROZEN_COLS, []), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, SCRIPT, "--frozen-base", str(base),
             "--frozen-overlay", str(tmp_path / "no-overlay.csv"),
             "--scores", str(tmp_path / "no-scores.csv"),
             "--out-csv", str(tmp_path / "out.csv"), "--allow-unscored"],
            capture_output=True, text=True, cwd=REPO, check=False)
        out = r.stdout + r.stderr
        assert r.returncode == 2, out[-800:]
        assert "no-overlay.csv" in out, out[-800:]
        assert not (tmp_path / "out.csv").exists()
