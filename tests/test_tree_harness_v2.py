"""tests/test_tree_harness_v2.py — unit-style coverage for tree_search/harness_v2.py
(Phase D-1: the four Stage-4 recommendations from docs/tree_search_prototype.md §5-6).
Loaded via the `load_module` fixture (conftest.py) since tree_search/ is not a package.
"""
import numpy as np
import pytest

HARNESS_V2_PATH = "tree_search/harness_v2.py"


@pytest.fixture()
def hv2(load_module):
    return load_module(HARNESS_V2_PATH, "harness_v2_under_test")


def _seed_root_and_lineage(hv2, score_seq, comp="testcomp"):
    """Build a tree with one root + one first-gen lineage node + len(score_seq) further
    children of that lineage, each with score_seq[i] (all `evaluated`, distinct configs
    unless the caller deliberately repeats one)."""
    tree = hv2.new_tree(comp)
    hv2.add_root(tree, "root", {"kind": "solo", "v": 0}, 10.0, "evaluated", 1.0)
    # first-gen lineage seed (its own score is score_seq[0])
    lineage_id, _ = hv2.add_node(tree, tree["root_id"], "seed", {"kind": "solo", "v": 1},
                                  score_seq[0], "evaluated", 1.0)
    tree["search_state"]["active_lineage"] = lineage_id
    parent_id = lineage_id
    for i, s in enumerate(score_seq[1:], start=2):
        nid, dup = hv2.add_node(tree, parent_id, f"child{i}", {"kind": "solo", "v": i},
                                 s, "evaluated", 1.0)
        assert dup is None
        parent_id = nid
    return tree, lineage_id


# ---------------------------------------------------------------------------
# Recommendation 3: child dedup
# ---------------------------------------------------------------------------
def test_dedup_rejects_identical_config(hv2):
    tree = hv2.new_tree("c")
    hv2.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    cfg = {"kind": "blend", "members": [0, 2, 5, 15], "weight_search": "dirichlet"}
    nid1, dup1 = hv2.add_node(tree, 0, "first", cfg, 0.5, "evaluated", 1.0)
    assert nid1 is not None and dup1 is None

    # byte-identical config proposed again (s3e5's #37/#38/#39 bug) -> rejected
    nid2, dup2 = hv2.add_node(tree, 0, "duplicate attempt", dict(cfg), 0.5, "evaluated", 1.0)
    assert nid2 is None
    assert dup2 == nid1
    assert len(tree["nodes"]) == 2  # root + the one accepted child; duplicate never added


def test_dedup_ignores_key_order_and_allows_override(hv2):
    tree = hv2.new_tree("c")
    hv2.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    cfg_a = {"members": [1, 2], "kind": "blend"}
    cfg_b = {"kind": "blend", "members": [1, 2]}  # same content, different key order
    nid1, _ = hv2.add_node(tree, 0, "a", cfg_a, 0.5, "evaluated", 1.0)
    nid2, dup = hv2.add_node(tree, 0, "b", cfg_b, 0.5, "evaluated", 1.0)
    assert nid2 is None and dup == nid1

    nid3, dup3 = hv2.add_node(tree, 0, "explicit override", cfg_b, 0.5, "evaluated", 1.0,
                               allow_duplicate=True)
    assert nid3 is not None and dup3 is None


def test_find_duplicate_config_helper(hv2):
    tree = hv2.new_tree("c")
    hv2.add_root(tree, "root", {"a": 1}, 1.0, "evaluated", 1.0)
    assert hv2.find_duplicate_config(tree, {"a": 1}) == 0
    assert hv2.find_duplicate_config(tree, {"a": 2}) is None


# ---------------------------------------------------------------------------
# Recommendation 2: metric-aware plateau (tie-rate detection + neutral ties)
# ---------------------------------------------------------------------------
def test_tie_rate_zero_when_all_distinct(hv2):
    tree, _ = _seed_root_and_lineage(hv2, [1.0, 0.9, 0.8, 0.7])
    assert hv2.tie_rate(tree) == 0.0


def test_tie_rate_reflects_duplicate_scores(hv2):
    tree, _ = _seed_root_and_lineage(hv2, [1.0, 1.0, 1.0])
    # scores: root(10.0) + 1.0,1.0,1.0 -> 4 evaluated, 2 distinct -> tie_rate = 1 - 2/4
    assert hv2.tie_rate(tree) == pytest.approx(0.5)


def test_continuous_metric_tie_counts_as_non_improving(hv2):
    # Dilute tie_rate well below TIE_RATE_THRESHOLD with lots of distinct-score noise
    # nodes (a continuous-metric tree essentially never has a high tie_rate), then check
    # that 3 exact ties against the global best on the ACTIVE lineage still plateau it —
    # v1's original "ties count as non-improving" behavior must be preserved when the
    # surface isn't discretized.
    tree = hv2.new_tree("continuous")
    hv2.add_root(tree, "root", {"kind": "solo", "v": 0}, 100.0, "evaluated", 1.0)
    noise_parent, _ = hv2.add_node(tree, 0, "noise-seed", {"kind": "solo", "v": "n0"},
                                    50.0, "evaluated", 1.0)
    parent = noise_parent
    for i in range(29):  # 29 further distinct-score noise nodes -> plenty of dilution
        nid, _ = hv2.add_node(tree, parent, f"noise{i}", {"kind": "solo", "v": f"n{i+1}"},
                              49.0 - i, "evaluated", 1.0)
        parent = nid

    active_id, _ = hv2.add_node(tree, 0, "active-seed", {"kind": "solo", "v": "a0"},
                                 10.0, "evaluated", 1.0)  # new global best
    tree["search_state"]["active_lineage"] = active_id
    assert hv2.tie_rate(tree) <= hv2.TIE_RATE_THRESHOLD

    parent = active_id
    for i in range(hv2.PLATEAU_STREAK):  # 3 exact ties against the global best (10.0)
        nid, dup = hv2.add_node(tree, parent, f"tie{i}", {"kind": "solo", "v": f"a{i+1}"},
                                10.0, "evaluated", 1.0)
        assert dup is None
        parent = nid
    assert active_id in tree["search_state"]["plateaued"]


def test_discretized_metric_tie_is_neutral_and_streak_extended(hv2):
    # Manufacture a tie_rate above threshold by padding the tree with lots of duplicate
    # scores from an unrelated (non-active) lineage, then verify the ACTIVE lineage's
    # exact ties against global best do NOT plateau it within PLATEAU_STREAK(3) children.
    tree = hv2.new_tree("discrete")
    hv2.add_root(tree, "root", {"kind": "solo", "v": 0}, 10.0, "evaluated", 1.0)
    # noise lineage: 6 duplicate scores (never touched again -> stays inactive)
    noise_parent, _ = hv2.add_node(tree, 0, "noise-seed", {"kind": "solo", "v": "n0"},
                                    9.0, "evaluated", 1.0)
    for i in range(6):
        hv2.add_node(tree, noise_parent, f"noise{i}", {"kind": "solo", "v": f"n{i}"},
                     9.0, "evaluated", 1.0)  # all tie at 9.0

    # active lineage: seed beats global best (9.0 -> new best), then 4 exact ties in a row
    active_id, _ = hv2.add_node(tree, 0, "active-seed", {"kind": "solo", "v": "a0"},
                                 5.0, "evaluated", 1.0)
    tree["search_state"]["active_lineage"] = active_id
    assert hv2.tie_rate(tree) > hv2.TIE_RATE_THRESHOLD

    parent = active_id
    for i in range(hv2.ADAPTIVE_PLATEAU_STREAK - 1):  # one fewer than the adaptive limit
        nid, dup = hv2.add_node(tree, parent, f"tie{i}", {"kind": "solo", "v": f"a{i+1}"},
                                 5.0, "evaluated", 1.0)  # exact tie with global best
        assert dup is None
        parent = nid
    # under v1's fixed PLATEAU_STREAK=3 this lineage would already be plateaued (3 ties);
    # under the adaptive rule (discretized + neutral ties) it must NOT be plateaued yet
    assert active_id not in tree["search_state"]["plateaued"]
    assert tree["search_state"]["streak"].get(str(active_id), 0) == 0


# ---------------------------------------------------------------------------
# Recommendation 1: ensemble-default node space (cache_oof/load_oof + eval_blend)
# ---------------------------------------------------------------------------
def test_cache_and_load_oof_roundtrip(hv2, tmp_path):
    oof = np.array([1.0, 2.0, 3.5])
    hv2.cache_oof(str(tmp_path), 7, oof, pred=np.array([9.0]))
    loaded = hv2.load_oof(str(tmp_path), 7)
    np.testing.assert_allclose(loaded, oof)


def test_load_oof_missing_raises_clear_error(hv2, tmp_path):
    with pytest.raises(ValueError, match="no cached OOF"):
        hv2.load_oof(str(tmp_path), 999)


def test_eval_blend_on_synthetic_oofs_recovers_perfect_member(hv2, tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    # member 0 == y exactly (perfect); member 1 is pure noise
    hv2.cache_oof(str(tmp_path), 0, y)
    hv2.cache_oof(str(tmp_path), 1, rng.normal(size=200))

    def mae_metric(blended_oof):
        return float(np.mean(np.abs(blended_oof - y)))

    best_w, best_s, oofs = hv2.eval_blend(str(tmp_path), [0, 1], mae_metric,
                                           weight_search="dirichlet", k=300)
    assert oofs.shape == (200, 2)
    assert best_s < 0.05  # weight search should land close to all-weight-on-member-0
    assert best_w[0] > 0.9


def test_eval_blend_grid_simplex_matches_dirichlet_ballpark(hv2, tmp_path):
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    hv2.cache_oof(str(tmp_path), 0, y + rng.normal(scale=0.1, size=100))
    hv2.cache_oof(str(tmp_path), 1, y + rng.normal(scale=0.5, size=100))

    def mae_metric(blended_oof):
        return float(np.mean(np.abs(blended_oof - y)))

    w_grid, s_grid, _ = hv2.eval_blend(str(tmp_path), [0, 1], mae_metric,
                                        weight_search="grid_simplex", grid_step=0.1)
    w_dir, s_dir, _ = hv2.eval_blend(str(tmp_path), [0, 1], mae_metric,
                                      weight_search="dirichlet", k=500)
    assert s_grid < 0.3 and s_dir < 0.3
    # both should favor member 0 (the less noisy one)
    assert w_grid[0] > 0.5 and w_dir[0] > 0.5


def test_eval_blend_requires_two_members(hv2, tmp_path):
    hv2.cache_oof(str(tmp_path), 0, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match=">=2 members"):
        hv2.eval_blend(str(tmp_path), [0], lambda o: 0.0)


def test_node_kind_is_first_class_field(hv2):
    tree = hv2.new_tree("c")
    hv2.add_root(tree, "root", {"kind": "solo", "model": "lgb"}, 1.0, "evaluated", 1.0)
    assert tree["nodes"][0]["kind"] == "solo"
    nid, _ = hv2.add_node(tree, 0, "blend child",
                          {"kind": "blend", "members": [0, 1]}, 0.5, "evaluated", 1.0)
    node = next(n for n in tree["nodes"] if n["id"] == nid)
    assert node["kind"] == "blend"


# ---------------------------------------------------------------------------
# Recommendation 4: experience-library mutation prior
# ---------------------------------------------------------------------------
# suggest_priors requires a 'comp' key since the 2026-07-13 self-exclusion wiring;
# a fixture comp that cites no evidence in the library keeps the filter inert.
FIXTURE_COMP = {"comp": "zz-test-fixture-comp"}


def test_suggest_priors_nonempty_for_mae(hv2):
    priors = hv2.suggest_priors({"metric": "mae", **FIXTURE_COMP})
    assert isinstance(priors, list)
    assert len(priors) > 0
    assert all(isinstance(p, str) and p for p in priors)


def test_suggest_priors_nonempty_for_qwk(hv2):
    priors = hv2.suggest_priors({"metric": "qwk", **FIXTURE_COMP})
    assert len(priors) > 0


def test_suggest_priors_empty_for_no_keywords(hv2):
    assert hv2.suggest_priors({}) == []


def test_suggest_priors_empty_for_unmatched_keyword(hv2):
    assert hv2.suggest_priors({"metric": "zzz_no_such_metric_zzz", **FIXTURE_COMP}) == []


def test_suggest_priors_missing_comp_raises(hv2):
    with pytest.raises(ValueError, match="comp"):
        hv2.suggest_priors({"metric": "mae"})


def test_suggest_priors_respects_max_items(hv2):
    priors = hv2.suggest_priors({"metric": "mae", **FIXTURE_COMP}, max_items=1)
    assert len(priors) == 1


def test_suggest_priors_tags_match_data_type_sections(hv2):
    priors = hv2.suggest_priors({"tags": ["duplicate_rows", "小樣本"], **FIXTURE_COMP})
    assert len(priors) > 0
