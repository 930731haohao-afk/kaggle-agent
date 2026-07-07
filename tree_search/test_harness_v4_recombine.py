"""tree_search/test_harness_v4_recombine.py — Phase J J-2b unit tests for the v4 recombination
mutation (harness_v4.recombine and its glue). DELIBERATELY tiny / config-level: it only builds
synthetic node dicts and exercises the pure recombination operator — it trains NO model, reads
NO large file, spawns NO subprocess, and finishes well under a second, so it is safe to run
while the s5e10 v3 search holds the CPU.

Run:  uv run python3 tree_search/test_harness_v4_recombine.py
(also importable by pytest — every check is a `test_*` function; the __main__ block prints
PASS/FAIL and exits nonzero on any failure.)

Assertions (mapping to the J-2b spec):
  (a) recombine of two parents unions their feature families AND their blend members.
  (b) hyperparameters + model are inherited from the STRONGER parent (lower score = better),
      independent of argument order.
  (c) the child record is provenance='recombine' and records BOTH parent node ids;
      record_recombination persists that provenance into search_state['driver_state'].
  (d) boundary: two config-identical parents produce a child honestly flagged degenerate
      (child duplicates a parent); recombining a node with itself raises.
plus: the real drop-list feature schema unions correctly (drop-intersection == union-of-used);
propose_recombinations skips degenerate/existing dups; should_offer_recombination reads the
v3 explore_burst phase; V4_MUTATION_TYPES registers recombine.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import harness_v4 as hv4   # noqa: E402


def _node(nid, score, config):
    return dict(id=nid, parent_id=0, config=config, score=score, status="evaluated",
                kind=config.get("kind", "solo"))


# hybrid synthetic parents carrying model+params+features(include-list)+members at once, so a
# single recombine exercises BOTH the feature-family union AND the member union.
def _hybrid_parents():
    a = _node(11, 0.30, {"kind": "blend", "model": "lgb", "params": {"num_leaves": 31, "lr": 0.05},
                         "features": {"use": ["fam_x", "fam_y"]}, "members": [2, 3],
                         "weight_search": "dirichlet"})
    b = _node(22, 0.42, {"kind": "blend", "model": "cat", "params": {"depth": 8, "lr": 0.03},
                         "features": {"use": ["fam_y", "fam_z"]}, "members": [3, 4],
                         "weight_search": "dirichlet"})
    return a, b


def test_a_unions_features_and_members():
    """(a) feature families and blend members are BOTH set-unioned into the child."""
    a, b = _hybrid_parents()
    rec = hv4.recombine(a, b)
    child = rec["config"]
    assert child["features"]["use"] == ["fam_x", "fam_y", "fam_z"], \
        f"feature families not unioned: {child['features']}"
    assert child["members"] == [2, 3, 4], f"members not unioned: {child['members']}"
    assert child["kind"] == "blend", "member-carrying child should be kind='blend'"
    # unioning did not mutate the parents
    assert a["config"]["members"] == [2, 3] and b["config"]["members"] == [3, 4], \
        "recombine mutated a parent config"
    return f"features {child['features']['use']} and members {child['members']} are unions"


def test_b_inherits_stronger_parent_hyperparams():
    """(b) model+hyperparams come from the stronger (lower-score) parent, either arg order."""
    a, b = _hybrid_parents()                         # a stronger (0.30 < 0.42)
    for order, (x, y) in enumerate([(a, b), (b, a)]):
        rec = hv4.recombine(x, y)
        assert rec["base_parent"] == 11, f"order {order}: base should be the stronger #11, got {rec['base_parent']}"
        assert rec["config"]["model"] == "lgb", f"order {order}: model not inherited from stronger"
        assert rec["config"]["params"] == {"num_leaves": 31, "lr": 0.05}, \
            f"order {order}: params not inherited from stronger: {rec['config']['params']}"
    return "model=lgb + params={num_leaves:31,lr:0.05} inherited from stronger #11 in both arg orders"


def test_c_provenance_and_parent_ids():
    """(c) child is provenance='recombine' with both parent ids; record_recombination persists it."""
    a, b = _hybrid_parents()
    rec = hv4.recombine(a, b)
    assert rec["provenance"] == "recombine", f"provenance={rec['provenance']!r}"
    assert rec["parents"] == [11, 22], f"parents={rec['parents']}"
    assert rec["base_parent"] == 11 and rec["donor_parent"] == 22, "base/donor ids wrong"
    assert "parents=[11, 22]" in rec["mutation"], "mutation string omits parent ids"
    # persistence into the v3-blessed driver_state
    tree = {"root_id": 0, "nodes": [], "search_state": {}}
    hv4.record_recombination(tree, 99, rec)
    prov = tree["search_state"]["driver_state"]["recombine_provenance"]["99"]
    assert prov["provenance"] == "recombine" and prov["parents"] == [11, 22] \
        and prov["base_parent"] == 11 and prov["donor_parent"] == 22, f"persisted provenance wrong: {prov}"
    return "provenance='recombine', parents=[11,22] on the record and persisted to driver_state"


def test_d_boundary_identical_and_self():
    """(d) identical-config parents -> child honestly flagged degenerate (duplicates a parent);
    recombining a node with itself raises."""
    cfg = {"kind": "solo", "model": "lgb", "params": {"num_leaves": 15}, "features": {"drop": ["a"]}}
    c = _node(31, 0.30, dict(cfg))
    d = _node(32, 0.35, dict(cfg))                   # distinct node, identical config
    rec = hv4.recombine(c, d)
    assert rec["degenerate"] is True, "identical-config parents not flagged degenerate"
    assert hv4.config_hash(rec["config"]) == hv4.config_hash(cfg), \
        "degenerate child should duplicate the shared parent config"
    raised = False
    try:
        hv4.recombine(c, c)                          # same node id
    except ValueError:
        raised = True
    assert raised, "recombining a node with itself did not raise ValueError"
    return "identical parents -> degenerate=True (child duplicates parent); self-recombine raises"


def test_e_drop_list_schema_union():
    """Real s5e10-style drop-list schema: two solos union to a solo whose drop = intersection
    (== the union of the feature families each parent kept)."""
    e = _node(41, 0.30, {"kind": "solo", "model": "lgb", "params": {"num_leaves": 31},
                         "features": {"drop": ["a", "b"]}})
    f = _node(42, 0.40, {"kind": "solo", "model": "xgb", "params": {"max_depth": 6},
                         "features": {"drop": ["b", "c"]}})
    child = hv4.recombine(e, f)["config"]
    assert child["features"]["drop"] == ["b"], f"drop should be intersection ['b'], got {child['features']['drop']}"
    assert child["kind"] == "solo" and "members" not in child, "solo pair should stay solo (no members)"
    assert child["model"] == "lgb" and child["params"] == {"num_leaves": 31}, "did not inherit stronger solo"
    return "solo⊕solo drop-list: child.drop=['b'] (intersection == union-of-used), stays solo/lgb"


def test_f_propose_and_phase_hook():
    """propose_recombinations picks top pairs, skips degenerate + existing dups; the phase hook
    reads v3's explore_burst; V4_MUTATION_TYPES registers recombine. Drops are chosen so the
    three pairwise intersections ([b],[a],[c]) are distinct -> three distinct child configs."""
    tree = {"root_id": 0, "search_state": {}, "nodes": [
        _node(0, 0.50, {"kind": "solo", "model": "lgb", "params": {"n": 1}, "features": {"drop": ["z"]}}),
        _node(1, 0.30, {"kind": "solo", "model": "lgb", "params": {"n": 2}, "features": {"drop": ["a", "b"]}}),
        _node(2, 0.35, {"kind": "solo", "model": "lgb", "params": {"n": 2}, "features": {"drop": ["b", "c"]}}),
        _node(3, 0.40, {"kind": "solo", "model": "lgb", "params": {"n": 2}, "features": {"drop": ["a", "c"]}}),
    ]}
    tree["nodes"][0]["parent_id"] = None
    recs = hv4.propose_recombinations(tree, top_k=3, max_pairs=6)
    assert len(recs) == 3, f"expected 3 unique pairs from top-3 nodes, got {len(recs)}"
    assert all(r["provenance"] == "recombine" and not r["degenerate"] for r in recs), "bad recombination records"
    assert all(set(r["parents"]) <= {1, 2, 3} for r in recs), "pairs drew from outside the top-3 set"
    # skip_existing: insert the first pair's child as a real node (worst score so it can't perturb
    # the top-3) -> that one pair drops out, the other two survive.
    tree["nodes"].append(_node(4, 0.99, dict(recs[0]["config"])))
    recs2 = hv4.propose_recombinations(tree, top_k=3, max_pairs=6)
    assert len(recs2) == 2, f"skip_existing should drop the now-duplicate pair, got {len(recs2)}"
    # phase hook (reads v3 state, never writes it)
    assert hv4.should_offer_recombination(tree) is False, "no burst phase yet -> should be False"
    tree["search_state"]["budget"] = {"phase": "explore_burst"}
    assert hv4.should_offer_recombination(tree) is True, "explore_burst phase -> should offer recombination"
    assert hv4.V4_MUTATION_TYPES.get("recombine") is hv4.propose_recombinations, "recombine not registered"
    return "proposed 3 distinct pairs; skip_existing dropped 1 dup (->2); phase hook + V4_MUTATION_TYPES wired"


_TESTS = [
    ("a", "unions feature families AND blend members", test_a_unions_features_and_members),
    ("b", "inherits stronger parent's model+hyperparams", test_b_inherits_stronger_parent_hyperparams),
    ("c", "provenance='recombine' + both parent ids (record + driver_state)", test_c_provenance_and_parent_ids),
    ("d", "boundary: identical parents degenerate; self raises", test_d_boundary_identical_and_self),
    ("e", "real drop-list schema unions correctly", test_e_drop_list_schema_union),
    ("f", "propose_recombinations + phase hook + registry", test_f_propose_and_phase_hook),
]


if __name__ == "__main__":
    t0 = time.time()
    failures = 0
    for tag, name, fn in _TESTS:
        try:
            detail = fn()
            print(f"PASS ({tag}) {name}\n        -> {detail}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL ({tag}) {name}\n        -> {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR ({tag}) {name}\n        -> {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed in {dt*1000:.0f} ms")
    sys.exit(1 if failures else 0)
