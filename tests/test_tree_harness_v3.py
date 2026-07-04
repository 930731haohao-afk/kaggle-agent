"""tests/test_tree_harness_v3.py — unit-style coverage for tree_search/harness_v3.py
(Phase F-1: folding the 11-tree-run lessons into the harness as defaults). Loaded via
the `load_module` fixture (conftest.py) since tree_search/ is not a package, same
pattern as test_tree_harness_v2.py.
"""
import os

import numpy as np
import pytest

HARNESS_V3_PATH = "tree_search/harness_v3.py"
HARNESS_V2_PATH = "tree_search/harness_v2.py"


@pytest.fixture()
def hv3(load_module):
    return load_module(HARNESS_V3_PATH, "harness_v3_under_test")


@pytest.fixture()
def hv2(load_module):
    return load_module(HARNESS_V2_PATH, "harness_v2_under_test_for_v3")


# ---------------------------------------------------------------------------
# Feature 1: budget & stopping policy (phase machine exploit -> explore_burst -> stopped)
# ---------------------------------------------------------------------------
def test_init_budget_defaults(hv3):
    tree = hv3.new_tree("c")
    budget = hv3.init_budget(tree)
    assert budget["total_budget"] == hv3.DEFAULT_TOTAL_BUDGET == 60
    assert budget["phase"] == "exploit"
    assert budget["explore_burst_size"] == hv3.DEFAULT_EXPLORE_BURST_SIZE
    assert budget["post_burst_patience"] == hv3.DEFAULT_POST_BURST_PATIENCE


def test_init_budget_idempotent(hv3):
    tree = hv3.new_tree("c")
    b1 = hv3.init_budget(tree, total_budget=10)
    b2 = hv3.init_budget(tree, total_budget=999)  # second call must NOT overwrite
    assert b1 is b2
    assert tree["search_state"]["budget"]["total_budget"] == 10


def _plateau_a_lineage(hv3, tree, root_id, seed_score, streak_scores, active=True):
    """Seed one first-gen lineage and drive it to plateau (PLATEAU_STREAK non-improving
    children); returns the lineage id."""
    lid, _ = hv3.add_node(tree, root_id, "seed", {"kind": "solo", "v": "seed"},
                           seed_score, "evaluated", 1.0)
    if active:
        tree["search_state"]["active_lineage"] = lid
    parent = lid
    for i, s in enumerate(streak_scores):
        nid, dup = hv3.add_node(tree, parent, f"child{i}", {"kind": "solo", "v": f"c{i}"},
                                 s, "evaluated", 1.0)
        assert dup is None
        parent = nid
    return lid


def test_phase_machine_exploit_to_burst_to_stop(hv3):
    tree = hv3.new_tree("c")
    hv3.init_budget(tree, total_budget=60, explore_burst_size=3, post_burst_patience=4)
    root_id, _ = 0, None
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 100.0, "evaluated", 1.0)

    # Drive the single lineage to plateau (3 non-improving children against global best).
    _plateau_a_lineage(hv3, tree, tree["root_id"], seed_score=50.0,
                        streak_scores=[60.0, 61.0, 62.0])
    assert hv3.plateau_saturated(tree)
    assert tree["search_state"]["budget"]["phase"] == "explore_burst"  # add_node already advanced it

    # Explore-burst window: burst_size=3 evals with no improvement, then patience=4 more
    # non-improving evals should trigger "stopped".
    lid2, _ = hv3.add_node(tree, tree["root_id"], "burst-seed", {"kind": "solo", "v": "b0"},
                            90.0, "evaluated", 1.0)  # 1 eval into the burst, no improvement
    for i in range(6):  # plenty of further non-improving evals
        nid, dup = hv3.add_node(tree, lid2, f"burst{i}", {"kind": "solo", "v": f"b{i+1}"},
                                 90.0 + i, "evaluated", 1.0)
        assert dup is None
        if hv3.should_stop(tree):
            break
    assert hv3.should_stop(tree)
    assert tree["search_state"]["budget"]["phase"] == "stopped"
    assert "post-burst" in tree["search_state"]["budget"]["stop_reason"]


def test_phase_machine_burst_resets_patience_on_improvement(hv3):
    tree = hv3.new_tree("c")
    hv3.init_budget(tree, total_budget=60, explore_burst_size=2, post_burst_patience=3)
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 100.0, "evaluated", 1.0)
    _plateau_a_lineage(hv3, tree, tree["root_id"], seed_score=50.0,
                        streak_scores=[60.0, 61.0, 62.0])
    assert tree["search_state"]["budget"]["phase"] == "explore_burst"

    lid2, _ = hv3.add_node(tree, tree["root_id"], "burst-seed", {"kind": "solo", "v": "b0"},
                            90.0, "evaluated", 1.0)
    hv3.add_node(tree, lid2, "burst1", {"kind": "solo", "v": "b1"}, 91.0, "evaluated", 1.0)
    # 2 evals in (== explore_burst_size), no improvement yet -- not stopped
    assert not hv3.should_stop(tree)
    # a genuine improvement (breaks the 50.0-vs-90/91 losing streak) resets the patience clock
    nid, dup = hv3.add_node(tree, lid2, "burst-win", {"kind": "solo", "v": "bwin"},
                             10.0, "evaluated", 1.0)
    assert tree["search_state"]["budget"]["evals_since_burst_improve"] == 0
    assert not hv3.should_stop(tree)


def test_phase_machine_hard_stop_at_budget(hv3):
    tree = hv3.new_tree("c")
    hv3.init_budget(tree, total_budget=3)
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 100.0, "evaluated", 1.0)
    lid, _ = hv3.add_node(tree, tree["root_id"], "seed", {"kind": "solo", "v": 1},
                           90.0, "evaluated", 1.0)
    assert not hv3.should_stop(tree)  # 2 evaluated so far (root + seed)
    hv3.add_node(tree, lid, "child", {"kind": "solo", "v": 2}, 80.0, "evaluated", 1.0)
    assert hv3.should_stop(tree)  # 3rd evaluated node hits the hard cap
    assert "hard budget cap" in tree["search_state"]["budget"]["stop_reason"]


def test_plateau_saturated_false_with_no_evaluated_lineages(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 100.0, "evaluated", 1.0)
    assert hv3.plateau_saturated(tree) is False


# ---------------------------------------------------------------------------
# Feature 2: dedup consumes budget
# ---------------------------------------------------------------------------
def test_dedup_single_rejection_does_not_burn_placeholder(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    cfg = {"kind": "blend", "members": [0, 2, 5]}
    nid1, dup1 = hv3.add_node(tree, 0, "first", cfg, 0.5, "evaluated", 1.0)
    assert nid1 is not None and dup1 is None

    nid2, dup2 = hv3.add_node(tree, 0, "dup attempt 1", dict(cfg), 0.5, "evaluated", 1.0)
    assert nid2 is None and dup2 == nid1
    # only 2 real nodes so far (root + accepted child) -- first rejection alone doesn't burn
    assert len(tree["nodes"]) == 2
    assert not any(n["config"].get("kind") == "placeholder" for n in tree["nodes"])


def test_dedup_two_consecutive_rejections_burn_placeholder(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    cfg = {"kind": "blend", "members": [0, 2, 5]}
    nid1, _ = hv3.add_node(tree, 0, "first", cfg, 0.5, "evaluated", 1.0)

    nid2, dup2 = hv3.add_node(tree, 0, "dup attempt 1", dict(cfg), 0.5, "evaluated", 1.0)
    assert nid2 is None and dup2 == nid1

    nid3, dup3 = hv3.add_node(tree, 0, "dup attempt 2", dict(cfg), 0.5, "evaluated", 1.0)
    assert nid3 is None and dup3 == nid1  # rejection contract unchanged
    # but a failed placeholder child should now have been burned under parent 0
    placeholders = [n for n in tree["nodes"] if n["config"].get("kind") == "placeholder"]
    assert len(placeholders) == 1
    ph = placeholders[0]
    assert ph["status"] == "failed"
    assert ph["parent_id"] == 0
    assert ph["config"]["dead_end_parent"] == 0
    # placeholder doesn't count as "evaluated" (budget is only spent on real evals)
    assert hv3.n_evaluated(tree) == 2  # root + nid1 only -- the placeholder is status="failed"


def test_dedup_streak_resets_after_burn(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    cfg = {"kind": "blend", "members": [0, 2, 5]}
    hv3.add_node(tree, 0, "first", cfg, 0.5, "evaluated", 1.0)
    hv3.add_node(tree, 0, "dup1", dict(cfg), 0.5, "evaluated", 1.0)
    hv3.add_node(tree, 0, "dup2", dict(cfg), 0.5, "evaluated", 1.0)  # burns placeholder #1
    assert tree["search_state"]["dedup_streak"]["0"] == 0
    hv3.add_node(tree, 0, "dup3", dict(cfg), 0.5, "evaluated", 1.0)  # streak restarts at 1
    placeholders = [n for n in tree["nodes"] if n["config"].get("kind") == "placeholder"]
    assert len(placeholders) == 1  # not burned again yet -- needs a 2nd consecutive rejection


# ---------------------------------------------------------------------------
# Feature 3: post-plateau solo breakthrough re-opens blend lineage
# ---------------------------------------------------------------------------
def test_solo_breakthrough_reopens_plateaued_blend_lineage(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 10.0, "evaluated", 1.0)

    # BLEND lineage that plateaus (3 non-improving children against global best 10.0... use root as best)
    blend_lid, _ = hv3.add_node(tree, tree["root_id"], "blend-seed",
                                 {"kind": "blend", "members": [0, 1]}, 8.0, "evaluated", 1.0)
    tree["search_state"]["active_lineage"] = blend_lid
    parent = blend_lid
    for i in range(3):
        nid, dup = hv3.add_node(tree, parent, f"blend-child{i}",
                                 {"kind": "blend", "members": [0, 1, 10 + i]},
                                 8.1 + i * 0.1, "evaluated", 1.0)  # never beats 8.0
        parent = nid
    assert blend_lid in tree["search_state"]["plateaued"]

    # A fresh solo lineage later finds a new global best -> should auto-reopen the blend lineage
    solo_lid, _ = hv3.add_node(tree, tree["root_id"], "solo-seed",
                                {"kind": "solo", "v": "s0"}, 5.0, "evaluated", 1.0)
    assert blend_lid not in tree["search_state"]["plateaued"]
    assert tree["search_state"]["streak"].get(str(blend_lid)) == 0


def test_non_breakthrough_solo_does_not_reopen(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 10.0, "evaluated", 1.0)
    blend_lid, _ = hv3.add_node(tree, tree["root_id"], "blend-seed",
                                 {"kind": "blend", "members": [0, 1]}, 8.0, "evaluated", 1.0)
    tree["search_state"]["active_lineage"] = blend_lid
    parent = blend_lid
    for i in range(3):
        nid, _ = hv3.add_node(tree, parent, f"blend-child{i}",
                               {"kind": "blend", "members": [0, 1, 10 + i]},
                               8.1 + i * 0.1, "evaluated", 1.0)
        parent = nid
    assert blend_lid in tree["search_state"]["plateaued"]

    # A solo node that does NOT beat the global best must not reopen anything
    hv3.add_node(tree, tree["root_id"], "solo-seed-weak",
                 {"kind": "solo", "v": "weak"}, 9.0, "evaluated", 1.0)
    assert blend_lid in tree["search_state"]["plateaued"]


def test_reopen_helper_returns_empty_for_non_solo_or_non_breakthrough(hv3):
    tree = hv3.new_tree("c")
    hv3.add_root(tree, "root", {"kind": "solo", "v": 0}, 10.0, "evaluated", 1.0)
    nid, _ = hv3.add_node(tree, tree["root_id"], "blend", {"kind": "blend", "members": [0, 1]},
                           20.0, "evaluated", 1.0)  # worse than root, and blend (not solo)
    assert hv3.reopen_blend_lineage_on_solo_breakthrough(tree, nid) == []
    assert hv3.reopen_blend_lineage_on_solo_breakthrough(tree, 0) == []  # root isn't the breakthrough trigger path either (no plateaued lineages yet)


# ---------------------------------------------------------------------------
# Feature 4: boundary_candidates
# ---------------------------------------------------------------------------
def test_boundary_candidates_detects_low_edge_linear(hv3):
    config = {"params": {"num_leaves": 5, "reg_lambda": 4.0}}
    search_space = {"num_leaves": (4, 10), "reg_lambda": (0.0, 10.0)}
    out = hv3.boundary_candidates(config, search_space, edge_frac=0.2)
    names = {c["param"]: c for c in out}
    assert "num_leaves" in names  # (5-4)/(10-4) = 0.167 <= 0.2 -> low edge
    assert names["num_leaves"]["edge"] == "low"
    assert "reg_lambda" not in names  # 4.0/10.0 = 0.4, interior


def test_boundary_candidates_detects_high_edge_and_log_scale(hv3):
    config = {"params": {"depth": 10, "learning_rate": 0.0102}}
    search_space = {"depth": (4, 10), "learning_rate": {"low": 0.01, "high": 0.06, "log": True}}
    out = hv3.boundary_candidates(config, search_space, edge_frac=0.05)
    names = {c["param"]: c for c in out}
    assert names["depth"]["edge"] == "high"
    assert names["depth"]["new_value"] > 10
    assert names["learning_rate"]["edge"] == "low"
    assert names["learning_rate"]["new_value"] < 0.0102


def test_boundary_candidates_ignores_interior_and_unknown_params(hv3):
    config = {"params": {"max_depth": 5, "not_in_space": 3}}
    search_space = {"max_depth": (1, 10)}
    out = hv3.boundary_candidates(config, search_space)
    assert out == []


def test_boundary_candidates_empty_when_no_params(hv3):
    assert hv3.boundary_candidates({}, {"x": (0, 1)}) == []


# ---------------------------------------------------------------------------
# Feature 5: weight-search default k=800 + coordinate-ascent
# ---------------------------------------------------------------------------
def test_eval_blend_default_k_is_800(hv3):
    assert hv3.DEFAULT_BLEND_K == 800


def test_eval_blend_default_uses_k800(hv3, tmp_path, monkeypatch):
    calls = {}
    real_v2_eval_blend = hv3.v2.eval_blend

    def spy(cache_dir, members, metric_fn, weight_search="dirichlet", k=1500, seed=42, grid_step=0.05):
        calls["k"] = k
        return real_v2_eval_blend(cache_dir, members, metric_fn, weight_search=weight_search,
                                   k=k, seed=seed, grid_step=grid_step)

    monkeypatch.setattr(hv3.v2, "eval_blend", spy)
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    hv3.cache_oof(str(tmp_path), 0, y)
    hv3.cache_oof(str(tmp_path), 1, rng.normal(size=50))
    hv3.eval_blend(str(tmp_path), [0, 1], lambda v: float(np.mean(np.abs(v - y))))
    assert calls["k"] == 800


def test_eval_blend_coordinate_ascent_no_worse_than_raw_search(hv3, tmp_path):
    rng = np.random.default_rng(3)
    y = rng.normal(size=300)
    hv3.cache_oof(str(tmp_path), 0, y + rng.normal(scale=0.3, size=300))
    hv3.cache_oof(str(tmp_path), 1, y + rng.normal(scale=0.9, size=300))

    def mae(v):
        return float(np.mean(np.abs(v - y)))

    w_no_ascent, s_no_ascent, _ = hv3.eval_blend(str(tmp_path), [0, 1], mae, k=100,
                                                  coordinate_ascent=False)
    w_ascent, s_ascent, _ = hv3.eval_blend(str(tmp_path), [0, 1], mae, k=100,
                                            coordinate_ascent=True)
    assert s_ascent <= s_no_ascent + 1e-9  # ascent only ever refines, never regresses


def test_eval_blend_no_coordinate_ascent_matches_v2_exactly(hv3, hv2, tmp_path):
    rng = np.random.default_rng(7)
    y = rng.normal(size=150)
    hv3.cache_oof(str(tmp_path), 0, y + rng.normal(scale=0.2, size=150))
    hv3.cache_oof(str(tmp_path), 1, y + rng.normal(scale=0.6, size=150))

    def mae(v):
        return float(np.mean(np.abs(v - y)))

    w_v2, s_v2, _ = hv2.eval_blend(str(tmp_path), [0, 1], mae, weight_search="dirichlet", k=250, seed=42)
    w_v3, s_v3, _ = hv3.eval_blend(str(tmp_path), [0, 1], mae, weight_search="dirichlet", k=250,
                                    seed=42, coordinate_ascent=False)
    assert s_v3 == s_v2
    np.testing.assert_array_equal(w_v3, w_v2)


# ---------------------------------------------------------------------------
# Feature 6: metric-aware blend-cost guard
# ---------------------------------------------------------------------------
def test_cost_guard_no_warning_when_fast(hv3, tmp_path):
    rng = np.random.default_rng(9)
    y = rng.normal(size=50)
    hv3.cache_oof(str(tmp_path), 0, y)
    hv3.cache_oof(str(tmp_path), 1, rng.normal(size=50))

    w, s, oofs, warning = hv3.eval_blend_with_cost_guard(
        str(tmp_path), [0, 1], lambda v: float(np.mean(np.abs(v - y))),
        k=50, wall_time_threshold_s=45.0)
    assert warning is None


def test_cost_guard_warns_and_coarsens_when_slow(hv3, tmp_path):
    import time as _time
    rng = np.random.default_rng(11)
    y = rng.normal(size=30)
    hv3.cache_oof(str(tmp_path), 0, y)
    hv3.cache_oof(str(tmp_path), 1, rng.normal(size=30))

    call_count = {"n": 0}

    def slow_metric(v):
        call_count["n"] += 1
        _time.sleep(0.01)  # artificial per-candidate cost to blow the (tiny) threshold
        return float(np.mean(np.abs(v - y)))

    tree = hv3.new_tree("c")
    w, s, oofs, warning = hv3.eval_blend_with_cost_guard(
        str(tmp_path), [0, 1], slow_metric, tree=tree, k=10,
        wall_time_threshold_s=0.05, coarsen_k=3, coordinate_ascent=False)
    assert warning is not None
    assert "COARSENING" in warning
    assert "cost_guard_log" in tree["search_state"]
    log = tree["search_state"]["cost_guard_log"][0]
    assert log["original_k"] == 10
    assert log["coarsened_k"] == 3


def test_cost_guard_logs_visibility_only_when_already_coarse(hv3, tmp_path):
    import time as _time
    rng = np.random.default_rng(13)
    y = rng.normal(size=20)
    hv3.cache_oof(str(tmp_path), 0, y)
    hv3.cache_oof(str(tmp_path), 1, rng.normal(size=20))

    def slow_metric(v):
        _time.sleep(0.01)
        return float(np.mean(np.abs(v - y)))

    tree = hv3.new_tree("c")
    w, s, oofs, warning = hv3.eval_blend_with_cost_guard(
        str(tmp_path), [0, 1], slow_metric, tree=tree, k=3,
        wall_time_threshold_s=0.001, coarsen_k=3, coordinate_ascent=False)  # k already == coarsen_k -> nothing to coarsen
    assert warning is not None
    assert "no further coarsening" in warning
    assert tree["search_state"]["cost_guard_log"][0]["coarsened_k"] is None


# ---------------------------------------------------------------------------
# Smoke test: replay s3e3's cached OOFs under v3's eval_blend and digit-match v2
# ---------------------------------------------------------------------------
_S3E3_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "tree_search", "cache_s3e3")
_EVAL_S3E3_PATH = "tree_search/eval_s3e3.py"


@pytest.mark.skipif(not os.path.isdir(_S3E3_CACHE_DIR),
                     reason="tree_search/cache_s3e3/ not present in this checkout")
def test_smoke_s3e3_blend_comparability_v2_vs_v3(hv2, hv3, load_module):
    """Phase D-2's node #8 (BLEND, members=[0,2,1,3], dirichlet, v2 defaults) stored AUC
    0.839448 in competitions/playground-series-s3e3/experiments_tree.json. Reproduce it
    fresh from the cached OOFs via harness_v2.eval_blend, then verify harness_v3.eval_blend
    with coordinate_ascent=False and the same k/seed reproduces it EXACTLY (digit-for-
    digit) -- confirming v3 didn't silently change v2's underlying search behavior, only
    its defaults."""
    import numpy as _np

    members = [0, 2, 1, 3]
    for m in members:
        path = os.path.join(_S3E3_CACHE_DIR, f"solo_{m}.npz")
        if not os.path.exists(path):
            pytest.skip(f"cache_s3e3/solo_{m}.npz missing in this checkout")

    ev = load_module(_EVAL_S3E3_PATH, "eval_s3e3_under_test_for_v3_smoke")

    def neg_auc(vec):
        return -ev.auc(ev._y, vec)

    w_v2, s_v2, _ = hv2.eval_blend(_S3E3_CACHE_DIR, members, neg_auc, weight_search="dirichlet")
    w_v3, s_v3, _ = hv3.eval_blend(_S3E3_CACHE_DIR, members, neg_auc, weight_search="dirichlet",
                                    k=1500, seed=42, coordinate_ascent=False)
    assert s_v3 == s_v2
    _np.testing.assert_array_equal(w_v3, w_v2)
    assert round(-s_v2, 6) == pytest.approx(0.839448, abs=1e-5)
