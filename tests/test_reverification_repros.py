"""The 2026-08-07 re-verification's surviving holes, as permanent regression tests.

Every test here was written FAILING, from the auditor's exact repro, BEFORE the fix it
verifies. That ordering is the point: the previous two fix rounds were verified against the
inputs that exposed the bugs, and re-verification then found 1 of 15 and 1 of 14 fixes
generalized. A fix that cannot make a pre-written failing test pass has not reproduced the
defect it claims to fix.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tree_search"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# champion selection (utils/experiment_log.py)
# ---------------------------------------------------------------------------
class TestChampionSelection:
    def _f(self):
        from utils.experiment_log import _entry_is_diagnostic
        return _entry_is_diagnostic

    def test_explicit_int_zero_flag_is_not_diagnostic(self):
        # isinstance(0, bool) is False, so tier 1 was skipped and tier 2 matched "probe"
        assert self._f()({"model": "probe blend", "score": 0.9, "diagnostic": 0}) is False

    def test_explicit_int_one_flag_is_diagnostic(self):
        assert self._f()({"model": "anything", "diagnostic": 1}) is True

    def test_string_true_flag_is_diagnostic(self):
        assert self._f()({"model": "anything", "diagnostic": "true"}) is True

    def test_probe_inside_a_blend_description_is_not_diagnostic(self):
        # real s3e19 champion id=8: model text is a 10-way blend description that happens to
        # contain a member name with "probe"; substring on model text demoted the champion
        assert self._f()({"model": "tree-search v2 blend (10-way: lgb + calendar-probe + ...)",
                          "score": 9.757}) is False

    def test_leak_probe_marked_the_documented_way_is_diagnostic(self):
        # notes BEGINNING with a diagnostic label assert what the entry is
        assert self._f()({"model": "LGB + row_id feature", "score": 0.9987,
                          "notes": "leakage check: target leaks via row ordering"}) is True

    def test_prose_mentioning_a_passed_check_is_not_diagnostic(self):
        assert self._f()({"model": "lgb", "notes": "blend of 5; passed the leakage check, clean"}) is False

    def test_real_s3e19_champion_is_id8(self):
        from utils.experiment_log import get_best_experiment
        path = os.path.join(REPO, "competitions/playground-series-s3e19/experiments.json")
        if not os.path.exists(path):
            pytest.skip("s3e19 experiments.json not present")
        best = get_best_experiment(os.path.dirname(path))
        assert best is not None and best.get("score") == pytest.approx(9.75707, abs=1e-4), (
            f"champion regressed to {(best or {}).get('score')} — the 10-way blend must win")


# ---------------------------------------------------------------------------
# features / params in BOTH copies of experiment_log
# ---------------------------------------------------------------------------
LOG_COPIES = ["utils/experiment_log.py",
              ".claude/skills/kaggle-agent/assets/utils/experiment_log.py",
              ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py"]


class TestExperimentLogBothCopies:
    @pytest.mark.parametrize("rel", LOG_COPIES)
    def test_features_string_not_exploded(self, rel, tmp_path):
        mod = _load(rel, f"explog_{abs(hash(rel))}")
        mod.log_experiment_v2(str(tmp_path), model="m", metric="rmse", direction="min",
                              score=1.0, features="feat_a,feat_b")
        e = json.loads((tmp_path / "experiments.json").read_text())
        rec = e[0] if isinstance(e, list) else e["experiments"][0]
        assert rec["features"] == ["feat_a", "feat_b"], rec["features"]
        assert rec["n_features"] == 2

    @pytest.mark.parametrize("rel", LOG_COPIES)
    def test_single_feature_string(self, rel, tmp_path):
        mod = _load(rel, f"explog1_{abs(hash(rel))}")
        mod.log_experiment_v2(str(tmp_path), model="m", metric="rmse", direction="min",
                              score=1.0, features="raw")
        e = json.loads((tmp_path / "experiments.json").read_text())
        rec = e[0] if isinstance(e, list) else e["experiments"][0]
        assert rec["features"] == ["raw"] and rec["n_features"] == 1

    @pytest.mark.parametrize("rel", LOG_COPIES)
    def test_numpy_params_do_not_discard_the_experiment(self, rel, tmp_path):
        # np.int64 values and keys raised TypeError AFTER training, discarding the experiment
        mod = _load(rel, f"explognp_{abs(hash(rel))}")
        mod.log_experiment_v2(str(tmp_path), model="m", metric="auc", direction="max",
                              score=0.87,
                              params={"num_leaves": np.int64(63),
                                      "weights": {np.int64(0): np.float64(1.5)},
                                      "grid": np.array([1, 2])})
        e = json.loads((tmp_path / "experiments.json").read_text())
        rec = e[0] if isinstance(e, list) else e["experiments"][0]
        assert rec["params"]["num_leaves"] == 63
        assert rec["params"]["weights"] == {"0": 1.5}
        # and no orphan tmp file left behind
        assert not [p for p in os.listdir(tmp_path) if ".tmp" in p]


# ---------------------------------------------------------------------------
# rules-verdict plumbing (make_v5_arm + apply_operators precedence)
# ---------------------------------------------------------------------------
class TestRulesVerdictPlumbing:
    def test_forbidden_verdict_beats_allow_unchecked(self):
        import pandas as pd
        from external_data.apply import apply_operators
        tr = pd.DataFrame({"country": ["Sweden"] * 3, "y": [1.0] * 3,
                           "date": ["2019-01-01", "2020-01-01", "2021-01-01"]})
        idea = [{"operator": "join_feature",
                 "params": {"source": "worldbank:gdp_per_capita",
                            "join": {"keys": ["country", "year"], "lag": 0}},
                 "rationale": "x"}]
        with pytest.raises(ValueError):
            apply_operators(tr, tr.drop(columns=["y"]).head(1), idea, target_col="y",
                            rules_verdict={"verdict": "forbidden"},
                            allow_unchecked_rules=True)

    def test_truthy_nonbool_does_not_open_the_gate(self):
        import pandas as pd
        from external_data.apply import apply_operators
        tr = pd.DataFrame({"country": ["Sweden"] * 3, "y": [1.0] * 3,
                           "date": ["2019-01-01", "2020-01-01", "2021-01-01"]})
        idea = [{"operator": "join_feature",
                 "params": {"source": "worldbank:gdp_per_capita",
                            "join": {"keys": ["country", "year"], "lag": 0}},
                 "rationale": "x"}]
        with pytest.raises((ValueError, TypeError)):
            apply_operators(tr, tr.drop(columns=["y"]).head(1), idea, target_col="y",
                            allow_unchecked_rules="no")

    def test_build_reads_the_workspace_verdict(self, tmp_path, monkeypatch):
        # build() had no way to pass a verdict, so every external-op arm was unbuildable
        import inspect
        import make_v5_arm
        src = inspect.getsource(make_v5_arm)
        assert "rules_verdict" in src, (
            "make_v5_arm never mentions rules_verdict: the rules gate added to "
            "apply_operators makes every external-op arm raise at construction")


# ---------------------------------------------------------------------------
# the arm-name hole (make_v5_arm books an arm whose operator was refused)
# ---------------------------------------------------------------------------
class TestArmNameIntegrity:
    def test_build_source_asserts_realization(self):
        import inspect
        import make_v5_arm
        src = inspect.getsource(make_v5_arm.build)
        assert ("realized" in src and ("raise" in src or "assert" in src)), (
            "build() never checks plan['realized'] against what it was asked to build: an arm "
            "named 'ratio' whose transform was refused is a baseline clone with a false name")


# ---------------------------------------------------------------------------
# the emitted-config ledger path (seed_from_ledger reads a different file)
# ---------------------------------------------------------------------------
class TestLedgerPathAgreement:
    def test_load_emitted_reads_the_documented_path(self, tmp_path):
        import seed_from_ledger
        ws = tmp_path / "competitions" / "comp-x"
        ws.mkdir(parents=True)
        # the path 03_features.md documents:
        (ws / "injection_ledger.json").write_text(json.dumps(
            {"emitted_config": [{"operator": "encoding",
                                 "config": {"encoding": {"scheme": "target", "columns": ["c"]}}}]}))
        got = seed_from_ledger.load_emitted(str(ws))
        assert got, ("load_emitted returns [] for the exact path 03_features.md instructs "
                     "Stage 2 to write — the operator is emitted into a ledger nothing reads")


# ---------------------------------------------------------------------------
# the library doors: '| 證據:' on its own continuation line
# ---------------------------------------------------------------------------
class TestLibraryContinuationEvidence:
    BULLET = ("### MAE/integer targets postprocess\n"
              "- **snap-to-nearest-observed beats plain clip**: round the blend to the\n"
              "  nearest value that occurs in y_train rather than clipping.\n"
              "  | 證據:s3e16, exp #14/#19, MAE 341.02 → 340.96\n")

    def test_suggest_priors_sees_pipe_prefixed_evidence(self, tmp_path):
        import harness_v2 as hv2
        lib = tmp_path / "experience.md"
        lib.write_text("## MAE section\n" + self.BULLET)
        hits = hv2.suggest_priors({"comp": "playground-series-s3e16", "metric": "mae"},
                                  experience_path=str(lib), max_items=10)
        assert not any("341.02" in h for h in hits), (
            "a bullet whose 證據 sits on a '| '-prefixed continuation line is served verbatim "
            "to the competition it cites — the filter never saw its evidence")

    def test_query_library_sees_pipe_prefixed_evidence(self, tmp_path, monkeypatch):
        from knowledge import query_library as ql
        lib = tmp_path / "experience.md"
        lib.write_text("## MAE section\n" + self.BULLET)
        entries = ql.parse_library(lib)
        hits = ql.query(["MAE", "snap"], entries, comp="playground-series-s3e16")
        assert not any("341.02" in h["text"] for h in hits), (
            "query_library serves the s3e16 bullet to an s3e16 re-run when its 證據 is on a "
            "'| '-prefixed continuation line")


# ---------------------------------------------------------------------------
# champion direction and metric handling (bucket-A #13/#17/#18)
# ---------------------------------------------------------------------------
class TestChampionDirection:
    def _write(self, tmp_path, entries):
        import json as _json
        (tmp_path / "experiments.json").write_text(_json.dumps(entries))
        return str(tmp_path)

    def test_v1_schema_minimize_metric_not_inverted(self, tmp_path):
        # v1 entries carry no direction; the old default silently MAXIMIZED rmse
        from utils.experiment_log import get_best_experiment
        d = self._write(tmp_path, [
            {"experiment_id": 1, "model_name": "a", "eval_metric": "rmse", "cv_mean": 10.0},
            {"experiment_id": 2, "model_name": "b", "eval_metric": "rmse", "cv_mean": 5.0}])
        best = get_best_experiment(d)
        assert best["experiment_id"] == 2, (
            f"rmse champion is the HIGHER score (id={best['experiment_id']}): with no "
            f"direction recorded, the maximize default inverted a minimize metric")

    def test_freeform_direction_spellings(self, tmp_path):
        from utils.experiment_log import get_best_experiment
        for spelling in ("min", "minimise", "MINIMIZE", "lower_is_better"):
            d = self._write(tmp_path, [
                {"experiment_id": 1, "model": "a", "metric": "rmse", "direction": spelling,
                 "score": 10.0},
                {"experiment_id": 2, "model": "b", "metric": "rmse", "direction": spelling,
                 "score": 5.0}])
            best = get_best_experiment(d)
            assert best["experiment_id"] == 2, (
                f"direction={spelling!r} was not recognized as minimize; the champion is the "
                f"worse model")

    def test_mixed_metrics_refuse_rather_than_rank_apples_against_oranges(self, tmp_path):
        from utils.experiment_log import get_best_experiment
        d = self._write(tmp_path, [
            {"experiment_id": 1, "model": "a", "metric": "rmse", "direction": "minimize",
             "score": 0.2},
            {"experiment_id": 2, "model": "b", "metric": "auc", "direction": "maximize",
             "score": 0.9}])
        with pytest.raises(ValueError):
            get_best_experiment(d)

    def test_explicit_metric_filter_still_works_on_mixed(self, tmp_path):
        from utils.experiment_log import get_best_experiment
        d = self._write(tmp_path, [
            {"experiment_id": 1, "model": "a", "metric": "rmse", "direction": "minimize",
             "score": 0.2},
            {"experiment_id": 2, "model": "b", "metric": "auc", "direction": "maximize",
             "score": 0.9}])
        best = get_best_experiment(d, metric="auc")
        assert best["experiment_id"] == 2

    def test_unknown_direction_and_metric_refuses(self, tmp_path):
        from utils.experiment_log import get_best_experiment
        d = self._write(tmp_path, [
            {"experiment_id": 1, "model": "a", "metric": "mystery_metric", "score": 1.0},
            {"experiment_id": 2, "model": "b", "metric": "mystery_metric", "score": 2.0}])
        with pytest.raises(ValueError):
            get_best_experiment(d)


# ---------------------------------------------------------------------------
# derived workspace names must not open the library door (bucket-A #49)
# ---------------------------------------------------------------------------
class TestDerivedWorkspaceExclusion:
    LIB = ("## MAE section\n- **the s3e16 recipe**: snap to observed values. "
           "| 證據:s3e16, exp #14, MAE 341.02 → 340.96\n")

    @pytest.mark.parametrize("comp", [
        "playground-series-s3e16",
        "playground-series-s3e16.repeat-r1-20260731",   # the launcher's repeat naming
        "playground-series-s3e16.leftover_pre_run",
        "playground-series-s3e16-v5-ratio",             # make_v5_arm's arm naming
    ])
    def test_all_workspace_derivations_exclude(self, comp, tmp_path):
        import harness_v2 as hv2
        lib = tmp_path / "experience.md"
        lib.write_text(self.LIB)
        hits = hv2.suggest_priors({"comp": comp, "metric": "mae"},
                                  experience_path=str(lib), max_items=5)
        assert not any("341.02" in h for h in hits), (
            f"{comp}: the workspace-derived name failed open — the base competition's own "
            f"answer was served")

    def test_unrelated_comp_still_gets_the_bullet(self, tmp_path):
        import harness_v2 as hv2
        lib = tmp_path / "experience.md"
        lib.write_text(self.LIB)
        hits = hv2.suggest_priors({"comp": "playground-series-s4e1", "metric": "mae"},
                                  experience_path=str(lib), max_items=5)
        assert any("341.02" in h for h in hits), "over-exclusion: an unrelated comp lost it"


# ---------------------------------------------------------------------------
# blend routes must agree THROUGH the cost guard, not only beside it
# ---------------------------------------------------------------------------
class TestBlendRouteAgreementThroughGuard:
    def test_guard_and_direct_routes_agree_on_a_coarsening_size(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        rng = np.random.default_rng(5)
        y = rng.normal(size=3000)
        for nid in range(8):
            hv2.cache_oof(str(tmp_path), nid, y + rng.normal(scale=0.3 + 0.05 * nid, size=3000))
        m = lambda v: float(np.sqrt(((v - y) ** 2).mean()))          # noqa: E731
        members = list(range(8))
        # force the deterministic coarsening threshold BELOW this problem size on both routes
        w_g, s_g, _o, _warn = hv3.eval_blend_with_cost_guard(
            str(tmp_path), members, m, cost_budget_units=1_000)
        w_d, s_d, _o = hv2.eval_blend(str(tmp_path), members, m, cost_budget_units=1_000)
        assert s_g == s_d, (
            f"guard route {s_g} != direct route {s_d}: coarsening applied on one route only "
            f"reintroduces a score per route")
        np.testing.assert_array_equal(w_g, w_d)

    def test_budget_boundary_matches_the_k800_calibration(self):
        import harness_v3 as hv3
        # the k=800 guard coarsened when members*rows > 1.875e6; raising k to 1500 must not
        # silently lower that boundary (s4e11's 8x140,700=1.13e6 blend started coarsening)
        boundary = hv3.DEFAULT_COST_BUDGET_UNITS / hv3.DEFAULT_BLEND_K
        assert boundary == pytest.approx(1_875_000, rel=0.01), (
            f"members*rows boundary is {boundary:,.0f}; the k=800 calibration was 1,875,000 — "
            f"s4e11's champion blend (1,125,600) must not coarsen")

    def test_oof_rows_skips_a_corrupt_member(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        (tmp_path / "solo_0.npz").write_bytes(b"not an npz")
        hv2.cache_oof(str(tmp_path), 1, np.zeros(1234))
        assert hv3._oof_rows(str(tmp_path), [0, 1]) == 1234, (
            "_oof_rows returns 0 on the FIRST unreadable member instead of trying the next — "
            "0 disables coarsening, so one corrupt file silently unguards the blend")


# ---------------------------------------------------------------------------
# covariate volatility: NaN must not decide, in either direction
# ---------------------------------------------------------------------------
class TestVolatilityNaN:
    def _cov(self, drop_nor_years=()):
        import pandas as pd
        rows = []
        for iso, base, growth in (("SWE", 50000, 1.02), ("NOR", 70000, 1.3)):
            v = base
            for yr in range(2017, 2022):
                if not (iso == "NOR" and yr in drop_nor_years):
                    rows.append({"iso3": iso, "year": yr, "value": v})
                v *= growth
        import pandas as pd
        return pd.DataFrame(rows)

    def test_missing_years_cannot_certify_stable(self):
        from external_data.apply import covariate_volatility_check
        full = covariate_volatility_check(self._cov(), {"Sweden": "SWE", "Norway": "NOR"},
                                          list(range(2017, 2022)))
        holey = covariate_volatility_check(self._cov(drop_nor_years=(2019, 2020)),
                                           {"Sweden": "SWE", "Norway": "NOR"},
                                           list(range(2017, 2022)))
        assert full["verdict"] == "volatile"
        assert holey["verdict"] != "stable", (
            f"dropping the volatile years turned the verdict into {holey['verdict']!r}: "
            f"NaN > threshold is False, so missing data certified stability")

    def test_single_group_is_not_volatile(self):
        from external_data.apply import covariate_volatility_check
        import pandas as pd
        cov = pd.DataFrame([{"iso3": "NOR", "year": y, "value": 70000 * 1.02 ** (y - 2017)}
                            for y in range(2017, 2022)])
        r = covariate_volatility_check(cov, {"Norway": "NOR"}, list(range(2017, 2022)))
        assert r["verdict"] != "volatile", (
            f"a single-country panel has no cross-group spread to measure; verdict "
            f"{r['verdict']!r} refuses ratio_target with 'moves up to nanpp'")


# ---------------------------------------------------------------------------
# Stage 5 transform table completeness
# ---------------------------------------------------------------------------
class TestInverseTransformTable:
    def test_every_kind_apply_can_write_has_a_row(self):
        import re
        apply_src = open(os.path.join(REPO, "external_data/apply.py")).read()
        # ONLY kinds assigned into plan["target_transform"] — node configs also carry a
        # "kind" key ("solo", "blend") that has nothing to invert
        resolved = set()
        for m in re.finditer(r'target_transform"?\]?\s*=\s*\{[^}]*"kind":\s*("[^"]+"|[^,}]+)',
                             apply_src):
            expr = m.group(1).strip()
            if expr.startswith('"'):
                resolved.add(expr.strip('"'))
            else:
                # a conditional expression: both branches are kinds
                resolved |= set(re.findall(r'"([a-z_]+)"', expr))
        doc = open(os.path.join(
            REPO, ".claude/skills/kaggle-agent/references/06_submission.md")).read()
        missing = [k for k in resolved if f"`{k}`" not in doc]
        assert not missing, (
            f"target_transform kind(s) {missing} can be written by apply.py but have no row in "
            f"06_submission.md's inversion table — a submission in y/covariate units passes "
            f"through the fall-through row and ships in the wrong units")


# ---------------------------------------------------------------------------
# submission validator: per-column value checks, token metric resolution, id position
# ---------------------------------------------------------------------------
class TestSubmissionValidator:
    def _vs(self):
        return _load(".claude/skills/kaggle-safe-submit/scripts/validate_submission.py", "vs_repro")

    def test_constant_nonlast_column_is_flagged(self):
        import pandas as pd
        vs = self._vs()
        sample = pd.DataFrame({"PIDN": ["a", "b", "c"], "Ca": [0.0] * 3, "P": [0.0] * 3,
                               "pH": [0.0] * 3})
        sub = pd.DataFrame({"PIDN": ["a", "b", "c"], "Ca": [1.1, 2.2, 3.3], "P": [0.0] * 3,
                            "pH": [4.4, 5.5, 6.6]})
        rep = vs.validate(sub, sample, id_col="PIDN", metric="mcrmse")
        assert rep.suspicious, (
            "the non-last column P is entirely 0.0 (a placeholder never filled) and no "
            "suspicious check fired — value checks still read only sample.columns[-1]")

    def test_verbose_metric_name_resolves(self):
        vs = self._vs()
        assert vs._resolve_expectation("area under the ROC curve (AUC)", "auto", None) == "probability"
        assert vs._resolve_expectation("categorization_accuracy", "auto", None) == "label"

    def test_every_repo_config_metric_resolves(self):
        import glob
        import re
        vs = self._vs()
        unresolved = []
        for cfg in glob.glob(os.path.join(REPO, "competitions*/*/config.yaml")):
            m = re.search(r"evaluation_metric:\s*(.+)", open(cfg).read())
            if not m:
                continue
            # strip YAML inline comments the way a real parser would
            name = m.group(1).split("#")[0].strip().strip("'\"")
            if vs._resolve_expectation(name, "auto", None) == "unknown":
                unresolved.append(name)
        assert not unresolved, (
            f"metric spellings used by this repo's own configs resolve to 'unknown', which "
            f"silently downgrades the value gate: {sorted(set(unresolved))}")

    def test_id_column_not_first_is_not_value_checked_as_predictions(self):
        import pandas as pd
        vs = self._vs()
        sample = pd.DataFrame({"target": [0.0] * 3, "image_name": ["a", "b", "c"]})
        sub = pd.DataFrame({"target": [5.0] * 3, "image_name": ["a", "b", "c"]})
        rep = vs.validate(sub, sample, id_col="image_name", metric="auc")
        assert rep.suspicious or rep.structural, (
            "with the id column last, pred_col resolved TO the id column: constant 5.0 on an "
            "AUC competition was never examined and the gate passed")


# ===========================================================================
# ROUND 3 (2026-08-07): fresh-input attacks that survived verification
# ===========================================================================
class TestFourthDoorPreregistrations:
    def test_prereg_read_is_filtered_for_the_comp_it_names(self):
        # Stage 0.5 step 8 mandates reading docs/preregistrations/ on firing-class comps;
        # the file names s3e19/sep-2022/s5e1 with their private scores and winning forms.
        import subprocess
        out = subprocess.run(
            [sys.executable, os.path.join(REPO, "knowledge/task_priors_for.py"),
             "playground-series-s5e1", "--prereg"],
            capture_output=True, text=True, cwd=REPO, check=False)
        assert out.returncode == 0, out.stderr[-300:]
        assert "0.12417" not in out.stdout and "0.15626" not in out.stdout, (
            "the mandated pre-registration read serves s5e1 its own private MAPE")
        # the registered hypothesis itself must survive for OTHER competitions
        out2 = subprocess.run(
            [sys.executable, os.path.join(REPO, "knowledge/task_priors_for.py"),
             "playground-series-s9e9", "--prereg"],
            capture_output=True, text=True, cwd=REPO, check=False)
        assert "H-HORIZON" in out2.stdout or "horizon" in out2.stdout.lower(), (
            "the filter destroyed the registered hypothesis for an unrelated competition")

    def test_step8_routes_through_the_filter(self):
        doc = open(os.path.join(
            REPO, ".claude/skills/kaggle-agent/references/00_problem_dossier.md")).read()
        import re as _re
        step8 = doc[doc.index("Open pre-registrations are binding"):]
        step8 = step8[:step8.index("\n9.")]
        assert "--prereg" in step8, (
            "step 8 still orders a raw read of docs/preregistrations/ -- the fourth door")


class TestNewSlugSelfExclusion:
    MD = ("## TASK-NEW — some new family\n"
          "- **Trigger**: something.\n"
          "- **Action**: the recipe.\n"
          "- **Evidence**: playground-series-s6e5 — ratio_target won, private SMAPE 3.1415 "
          "vs 3.9999; carried by an independent fact about s3e7.\n")

    def test_filter_priors_excludes_a_slug_not_in_the_token_list(self):
        from knowledge.task_priors_for import filter_priors
        out, _ = filter_priors(self.MD, "playground-series-s6e5")
        assert "3.1415" not in out, (
            "a post-benchmark slug absent from _ALL_COMP_TOKENS is served its own evidence")

    def test_filter_ops_excludes_it_too(self):
        from knowledge.task_priors_for import filter_ops
        out, _ = filter_ops("Prose: playground-series-s6e5 scored private SMAPE 3.1415 "
                            "with ratio_target.\n", "playground-series-s6e5")
        assert "3.1415" not in out


class TestNonTaskWriteBackScrubbed:
    MD = ("## TASK-A — x\n- **Trigger**: t.\n- **Action**: a.\n- **Evidence**: s3e7, 0.5.\n\n"
          "## Run notes 2026-08 (appended mid-run)\n"
          "s3e11 re-run: native categorical handling, private AUC 0.80311 beat NVIDIA 0.77084.\n"
          "AIDE's winning s3e11 node enabled native cats accidentally.\n"
          "\n"
          "An unrelated generic note that names nothing.\n")

    def test_non_task_section_is_scrubbed_not_verbatim(self):
        from knowledge.task_priors_for import filter_priors
        out, _ = filter_priors(self.MD, "playground-series-s3e11")
        assert "0.80311" not in out, "self result leaked through a non-TASK section"
        assert "NVIDIA" not in out and "AIDE" not in out, "cross-agent leaked via write-back"
        # A generic note in its OWN paragraph survives; one in the SAME paragraph as
        # self-naming prose inherits the subject and is withheld -- failing SAFE. The cost of
        # over-withholding is a lost prior; the cost of under-withholding is the benchmark.
        assert "unrelated generic note" in out, "over-scrubbing: a separate paragraph was eaten"

    def test_whitelist_table_still_survives(self):
        from knowledge.task_priors_for import filter_priors, PRIORS
        md = PRIORS.read_text(encoding="utf-8")
        out, _ = filter_priors(md, "playground-series-s3e19")
        rows = sum(1 for ln in out.splitlines() if ln.startswith("| "))
        raw = sum(1 for ln in md.splitlines() if ln.startswith("| "))
        assert rows == raw, "the scrub ate the external-data whitelist again"


class TestUnicodeDashAliases:
    def test_unicode_dash_spelling_is_excluded(self):
        from knowledge.task_priors_for import filter_priors
        md = ("## TASK-B — x\n- **Trigger**: t.\n- **Action**: a.\n"
              "- **Evidence**: tps‑sep‑2022 scored 11.348 with ratio_target; "
              "also s3e7, 0.4.\n")           # non-breaking hyphens
        out, _ = filter_priors(md, "tabular-playground-series-sep-2022")
        assert "11.348" not in out, "a unicode-dash spelling evaded the alias match"


class TestExperimentLogRound3:
    def test_nan_score_is_refused_at_log_time(self, tmp_path):
        from utils.experiment_log import log_experiment_v2
        with pytest.raises(ValueError):
            log_experiment_v2(str(tmp_path), model="m", metric="rmse",
                              direction="minimize", score=float("nan"))
        assert not list(tmp_path.glob("*.tmp*")), "orphan tmp left behind"

    def test_nan_score_in_an_existing_log_cannot_become_champion(self, tmp_path):
        import json as _json
        from utils.experiment_log import get_best_experiment
        (tmp_path / "experiments.json").write_text(_json.dumps([
            {"experiment_id": 1, "model": "broken-cv", "metric": "rmse",
             "direction": "minimize", "score": float("nan")},
            {"experiment_id": 2, "model": "good", "metric": "rmse",
             "direction": "minimize", "score": 0.45},
            {"experiment_id": 3, "model": "bad", "metric": "rmse",
             "direction": "minimize", "score": 0.90}]), )
        best = get_best_experiment(str(tmp_path))
        assert best["experiment_id"] == 2, f"nan poisoned selection: {best}"

    def test_numpy_anywhere_in_the_entry_survives(self, tmp_path):
        import json as _json
        from utils.experiment_log import log_experiment_v2
        log_experiment_v2(str(tmp_path), model="m", metric="rmse", direction="minimize",
                          score=0.5,
                          cv={"mean": np.float64(0.5), "folds": np.array([0.4, 0.6])},
                          ensemble={"weights": np.array([0.3, 0.7])},
                          leaderboard={"public": np.float32(0.51)},
                          cv_scores=np.array([1, 2]))
        rec = _json.loads((tmp_path / "experiments.json").read_text())[0]
        assert rec["cv"]["folds"] == [0.4, 0.6]
        assert rec["ensemble"]["weights"] == [0.3, 0.7]
        assert not list(tmp_path.glob("*.tmp*"))


class TestBlendRound3:
    def _mk(self, tmp_path, n_members, rows):
        import harness_v2 as hv2
        rng = np.random.default_rng(3)
        y = rng.normal(size=rows)
        for nid in range(n_members):
            hv2.cache_oof(str(tmp_path), nid, y + rng.normal(scale=0.3, size=rows))
        return y

    def test_hv3_eval_blend_applies_the_same_coarsening(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        # 4 members x 600k rows > 1.875e6 boundary -> both routes must coarsen identically
        y = self._mk(tmp_path, 4, 600_000)
        m = lambda v: float(np.sqrt(((v - y) ** 2).mean()))          # noqa: E731
        w2, s2, _ = hv2.eval_blend(str(tmp_path), [0, 1, 2, 3], m)
        w3, s3, _ = hv3.eval_blend(str(tmp_path), [0, 1, 2, 3], m)
        assert s2 == s3, f"hv3.eval_blend bypasses the unified coarsening: {s2} vs {s3}"
        np.testing.assert_array_equal(w2, w3)

    def test_explicit_k_is_honoured_identically_on_both_routes(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        y = self._mk(tmp_path, 3, 2000)
        m = lambda v: float(np.sqrt(((v - y) ** 2).mean()))          # noqa: E731
        _w2, s2, _ = hv2.eval_blend(str(tmp_path), [0, 1, 2], m, k=800)
        _w3, s3, _ = hv3.eval_blend(str(tmp_path), [0, 1, 2], m, k=800)
        assert s2 == s3, f"explicit k=800 diverges: {s2} vs {s3}"


class TestRatioArmLinearFamily:
    def test_run_linear_fold_aware_inversion(self):
        import pandas as pd
        from tree_search import eval_support as esup
        rng = np.random.default_rng(0)
        n, nte = 100, 37
        cov_tr = rng.uniform(10, 20, n)
        cov_te = rng.uniform(10, 20, nte)
        y = cov_tr * rng.uniform(0.9, 1.1, n)        # y ~ covariate level
        Xdf = pd.DataFrame({"f": rng.normal(size=n)})
        Xte = pd.DataFrame({"f": rng.normal(size=nte)})
        folds = []
        idx = np.arange(n)
        for k in range(3):
            va = (idx % 3) == k
            folds.append((~va, va))
        y_ratio = np.log(y / cov_tr)
        oof, pred = esup.run_linear({}, Xdf, Xte, [], folds, y_ratio, n,
                                    invert=lambda p: np.exp(p) * cov_te,
                                    invert_va=lambda p, mask: np.exp(p) * cov_tr[mask])
        assert oof.shape == (n,) and pred.shape == (nte,)
        # inverted OOF must be in TARGET units (~cov level), not ratio units (~1)
        assert 5 < np.median(oof) < 40, f"OOF median {np.median(oof)} is not in target units"
        assert 5 < np.median(pred) < 40

    def test_codegen_emits_fold_aware_linear_inversion(self):
        from tree_search.make_v5_arm import _patch_target_transform
        base = open(os.path.join(REPO, "tree_search/eval_s5e1.py")).read()
        for kind in ("ratio_log", "ratio_linear"):
            out = _patch_target_transform(base, "cov_level", kind)
            assert "invert_va=_invert_va" in out, (
                f"{kind}: the linear family's OOF inversion is not fold-aware -- fold-sized "
                f"predictions broadcast against the test-sized covariate (or stay in ratio "
                f"space) and the lin member ships wrong units")
            assert "invert=lambda p: p)" not in out, f"{kind}: identity inversion survives"
            assert "_invert_test(p) / _COV_TE" not in out, f"{kind}: ratio-space algebra survives"


class TestArmDefiningOpLookup:
    def test_gdp_hol_resolves_to_flag_feature(self):
        from tree_search.make_v5_arm import ARM_DEFINING_OP, _defining_op_for
        assert _defining_op_for("gdp_hol") == "flag_feature"
        assert _defining_op_for("ratio") == "ratio_target"
        assert _defining_op_for("featurejoin") == "join_feature"
        assert _defining_op_for("ratio-v2") == "ratio_target"


class TestRulesVerdictRound3:
    def test_non_dict_verdict_is_a_stated_refusal(self, tmp_path):
        import pandas as pd
        from external_data.apply import apply_operators
        vp = tmp_path / "rules_verdict.json"
        vp.write_text('["permitted"]')
        tr = pd.DataFrame({"country": ["Sweden"] * 3, "y": [1.0] * 3,
                           "date": ["2019-01-01", "2020-01-01", "2021-01-01"]})
        idea = [{"operator": "join_feature",
                 "params": {"source": "worldbank:gdp_per_capita",
                            "join": {"keys": ["country", "year"], "lag": 0}},
                 "rationale": "x"}]
        with pytest.raises(ValueError):
            apply_operators(tr, tr.drop(columns=["y"]).head(1), idea, target_col="y",
                            rules_verdict=str(vp))

    def test_verdict_bound_to_its_competition(self, tmp_path):
        import json as _json
        import pandas as pd
        from external_data.apply import apply_operators
        vp = tmp_path / "rules_verdict.json"
        vp.write_text(_json.dumps({"verdict": "permitted",
                                   "competition": "some-other-competition"}))
        tr = pd.DataFrame({"country": ["Sweden"] * 3, "y": [1.0] * 3,
                           "date": ["2019-01-01", "2020-01-01", "2021-01-01"]})
        idea = [{"operator": "join_feature",
                 "params": {"source": "worldbank:gdp_per_capita",
                            "join": {"keys": ["country", "year"], "lag": 0}},
                 "rationale": "x"}]
        with pytest.raises(ValueError):
            apply_operators(tr, tr.drop(columns=["y"]).head(1), idea, target_col="y",
                            rules_verdict=str(vp), expected_competition="playground-series-x")


class TestValidatorRound3:
    def _vs(self):
        return _load(".claude/skills/kaggle-safe-submit/scripts/validate_submission.py", "vs_r3")

    def test_object_dtype_predictions_fail_for_numeric_kinds(self):
        import pandas as pd
        vs = self._vs()
        sample = pd.DataFrame({"id": [1, 2, 3], "target": [0.0] * 3})
        sub = pd.DataFrame({"id": [1, 2, 3], "target": ["0.5", "0.7", "0.9"]})
        rep = vs.validate(sub, sample, id_col="id", metric="auc")
        assert rep.structural or rep.suspicious, (
            "numeric-as-strings predictions on a probability metric passed clean")

    def test_nonascii_metric_still_resolves(self):
        vs = self._vs()
        assert vs._resolve_expectation("ＲＯＣ－ＡＵＣ", "auto", None) == "probability"


# ===========================================================================
# ROUND 4 (2026-08-07): mechanical findings (isolation redesign tracked separately)
# ===========================================================================
class TestArmIntegrityRound4:
    def test_unknown_arm_name_with_multiple_ideas_still_guarded(self):
        # 'ratioconst' resolves to no defining op and ideas>1 skipped the check entirely
        from tree_search.make_v5_arm import _defining_op_for
        assert _defining_op_for("ratioconst") == "ratio_target"
        assert _defining_op_for("ratiogated") == "ratio_target"

    def test_unresolvable_arm_name_is_refused_not_unguarded(self):
        from tree_search.make_v5_arm import _defining_op_for
        with pytest.raises(ValueError):
            _defining_op_for("mystery-arm", strict=True)


class TestStaticSourceTimeAxisAliases:
    def test_year_and_date_columns_are_a_time_axis_too(self):
        import pandas as pd
        from external_data.admit_source import check_leakage, _good_spec
        from external_data.source_registry import SourceRejected
        look = _good_spec(key="cpc:titles",
                          url="https://www.cooperativepatentclassification.org/x",
                          publisher="CPC", join_key_class="lookup",
                          join_columns=["code"], value_columns=["title"],
                          leakage_rule="static")
        for col in ("year", "date", "month", "timestamp", "asof"):
            frame = pd.DataFrame({"code": ["A", "A", "B"], col: [2020, 2021, 2020],
                                  "title": ["x", "y", "z"]})
            with pytest.raises(SourceRejected):
                check_leakage(look, frame)


class TestVerdictBindingNotInert:
    def test_make_v5_arm_requires_the_competition_field(self, tmp_path, monkeypatch):
        # the documented recording command omitted --competition, so the binding never fired
        import json as _json
        import inspect
        from tree_search import make_v5_arm
        src = inspect.getsource(make_v5_arm.build)
        assert "competition" in src and ("require" in src.lower() or "raise" in src), (
            "build() accepts an unbound verdict; the transfer guard is inert in the "
            "documented flow")

    def test_step5_instruction_passes_competition(self):
        doc = open(os.path.join(
            REPO, ".claude/skills/kaggle-agent/references/00_problem_dossier.md")).read()
        assert "--competition" in doc, (
            "the documented rules_gate recording command still omits --competition, so "
            "every verdict it produces is unbound")


class TestBlendRound4:
    def test_guard_boundary_matches_unified_rule_for_any_k(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        rng = np.random.default_rng(4)
        rows = 1_000_000                     # 2 members x 1e6 = 2e6 > 1.875e6 boundary
        y = rng.normal(size=rows)
        for nid in range(2):
            hv2.cache_oof(str(tmp_path), nid, y + rng.normal(scale=0.4, size=rows))
        m = lambda v: float(np.sqrt(((v - y) ** 2).mean()))          # noqa: E731
        for k in (6000, 800, 300):
            _w, s_g, _o, _warn = hv3.eval_blend_with_cost_guard(str(tmp_path), [0, 1], m, k=k)
            _w, s_d, _o = hv2.eval_blend(str(tmp_path), [0, 1], m, k=k)
            assert s_g == s_d, f"k={k}: guard {s_g} != direct {s_d}"

    def test_nnls_routes_agree(self, tmp_path):
        import harness_v2 as hv2
        import harness_v3 as hv3
        rng = np.random.default_rng(5)
        y = rng.normal(size=500)
        for nid in range(3):
            hv2.cache_oof(str(tmp_path), nid, y + rng.normal(scale=0.3, size=500))
        m = lambda v: float(np.sqrt(((v - y) ** 2).mean()))          # noqa: E731
        w2, s2, _ = hv2.eval_blend(str(tmp_path), [0, 1, 2], m, weight_search="nnls", target=y)
        w3, s3, _ = hv3.eval_blend(str(tmp_path), [0, 1, 2], m, weight_search="nnls", target=y)
        assert s2 == s3, f"nnls diverges across routes: {s2} vs {s3}"
        np.testing.assert_array_equal(w2, w3)


class TestExperimentLogRound4:
    def test_datetime64_survives_jsonable(self, tmp_path):
        import json as _json
        from utils.experiment_log import log_experiment_v2
        log_experiment_v2(str(tmp_path), model="m", metric="rmse", direction="minimize",
                          score=0.4,
                          params={"cutoff": np.datetime64("2026-01-01"),
                                  "delta": np.timedelta64(3, "D")})
        rec = _json.loads((tmp_path / "experiments.json").read_text())[0]
        assert isinstance(rec["params"]["cutoff"], str)
        assert not list(tmp_path.glob("*.tmp*"))

    def test_dict_schema_experiments_json_is_read(self, tmp_path):
        import json as _json
        from utils.experiment_log import get_best_experiment
        (tmp_path / "experiments.json").write_text(_json.dumps(
            {"competition": "x", "experiments": [
                {"experiment_id": 1, "model": "a", "metric": "rmse",
                 "direction": "minimize", "score": 0.5},
                {"experiment_id": 2, "model": "b", "metric": "rmse",
                 "direction": "minimize", "score": 0.3}]}))
        best = get_best_experiment(str(tmp_path))
        assert best is not None and best["experiment_id"] == 2

    def test_print_leaderboard_direction_agrees_with_champion(self, tmp_path, capsys):
        import json as _json
        from utils.experiment_log import print_leaderboard, get_best_experiment
        (tmp_path / "experiments.json").write_text(_json.dumps([
            {"experiment_id": 1, "model": "worse", "metric": "wrmsse", "score": 0.70},
            {"experiment_id": 2, "model": "better", "metric": "wrmsse", "score": 0.55}]))
        best = get_best_experiment(str(tmp_path))
        assert best["experiment_id"] == 2
        print_leaderboard(str(tmp_path))
        outlines = capsys.readouterr().out.strip().splitlines()
        rank1 = next(ln for ln in outlines if ln.strip().startswith("1"))
        assert "better" in rank1, (
            f"print_leaderboard ranks the worse wrmsse entry first: {rank1!r} — its inline "
            f"direction set disagrees with get_best_experiment")


class TestValidatorRound4:
    def _vs(self):
        return _load(".claude/skills/kaggle-safe-submit/scripts/validate_submission.py", "vs_r4")

    def test_range_checked_on_every_target_column(self):
        import pandas as pd
        vs = self._vs()
        rng = np.random.default_rng(0)
        sample = pd.DataFrame({"PIDN": list("abc"), "Ca": [0.0] * 3, "P": [0.0] * 3,
                               "Sand": [0.0] * 3})
        train = pd.DataFrame({"Ca": rng.normal(0, 1, 50), "P": rng.normal(0, 1, 50),
                              "Sand": rng.normal(0, 1, 50)})
        sub = pd.DataFrame({"PIDN": list("abc"), "Ca": [0.1, -0.2, 0.3],
                            "P": [1000.0, 2000.0, 1500.0],       # 1000x scale error, NON-last
                            "Sand": [0.2, 0.1, -0.1]})
        rep = vs.validate(sub, sample, id_col="PIDN", metric="mcrmse",
                          y_train_frame=train)
        assert rep.suspicious, "a 1000x scale error in a non-last target column passed clean"

    def test_train_without_target_is_loud(self, tmp_path):
        import subprocess
        import pandas as pd
        vs_path = os.path.join(REPO, ".claude/skills/kaggle-safe-submit/scripts/validate_submission.py")
        pd.DataFrame({"id": [1, 2], "t": [0.5, 0.6]}).to_csv(tmp_path / "sub.csv", index=False)
        pd.DataFrame({"id": [1, 2], "t": [0.0, 0.0]}).to_csv(tmp_path / "sample.csv", index=False)
        pd.DataFrame({"t": [0.4, 0.7]}).to_csv(tmp_path / "train.csv", index=False)
        r = subprocess.run([sys.executable, vs_path, str(tmp_path / "sub.csv"),
                            str(tmp_path / "sample.csv"), "--metric", "rmse",
                            "--train", str(tmp_path / "train.csv")],
                           capture_output=True, text=True, check=False)
        out = r.stdout + r.stderr
        # LOUD, not silent: either the tool matched training columns by name and SAYS so
        # (and the Range gate actually ran), or it refuses telling the user to pass --target.
        assert ("matched" in out and "Range" in out and "SKIP" not in
                [ln for ln in out.splitlines() if "Range" in ln][0]) or \
               ("--target" in out and r.returncode != 0), (
            f"--train without --target was silent: rc={r.returncode}\n{out[-400:]}")

        # and when NOTHING matches by name, it must refuse rather than silently skip
        import pandas as pd
        pd.DataFrame({"unrelated": [0.4, 0.7]}).to_csv(tmp_path / "train2.csv", index=False)
        r2 = subprocess.run([sys.executable, vs_path, str(tmp_path / "sub.csv"),
                             str(tmp_path / "sample.csv"), "--metric", "rmse",
                             "--train", str(tmp_path / "train2.csv")],
                            capture_output=True, text=True, check=False)
        assert "--target" in (r2.stdout + r2.stderr) and r2.returncode != 0, (
            "no name match and no --target: the Range gate silently skipped")
