"""tree_search/run_s3e20_v2.py -- harness_v2 (Phase E-4) candidate-tree search for
playground-series-s3e20 (Rwanda CO2, RMSE). This is the REGIME-CONTRAST run the task
brief asks for: every prior harness_v2 build (s3e5/s3e9/s3e14/s3e16/s3e19) searched a
MODEL/BLEND landscape (hyperparams of GBDTs, ensemble composition). s3e20's own Phase-B
linear iteration (scripts/train_v3/v4/v5.py) already proved the opposite regime here --
GBDTs get 0 weight in every blend search (experiments.json #4-#8), the entire predictable
signal lives in a hand-tuned STRUCTURAL pipeline (denoised location-week historical means:
alpha=EB shrinkage, w2020=COVID year down-weight, wnb=neighbor-week-smoothing weight,
window=neighbor half-width). This run treats those 4 scalars (plus a few genuinely new
structural ideas Phase B never tried) as the tree-search node space instead, and asks: on
a landscape this specialized, does automated tree search beat hand-tuned one-at-a-time
iteration?

--- What's genuinely new vs Phase B here ---
Phase B tuned alpha/w2020/wnb one axis at a time (STATUS.md Round 1/2/3 headers literally
say "一次一項" -- one change per round), occasionally jointly RE-tuning 2-3 already-known
axes via an ad hoc, never-saved scratch pre-sweep when adding a new lever (e.g. Round 3's
docstring: "jointly re-tuned ALPHA, W2020, WNB"). It never (a) tried a 4th/5th structural
axis together with the original 3 in the same move, (b) weighted years OTHER than 2020,
(c) gave locations their OWN shrinkage strength, (d) tried a decayed (vs flat) neighbor
window, or (e) tried wrapping the week index across the year boundary. This run's mutation
menu is built to cover exactly that gap:

  ALPHA / W2020 / WNB lineages -- fine local re-sweeps of Phase B's own 3 tuned scalars.
    Pre-flight probes (done before writing this module, see eval_s3e20_v2.py) show W2020
    and WNB are ALREADY at their individual-axis optimum (0.24/0.28 -- moving either alone
    in any direction makes OOF RMSE worse); ALPHA has a tiny, real single-axis improvement
    available (1.0 -> 0.98, 21.1487 -> 21.1460). These 3 lineages exist mainly as an HONEST
    check: does automated re-sweeping just reproduce Phase B's already-correct optimum
    (the expected, non-embarrassing answer for 2 of the 3), or find something Phase B missed?

  WINDOW lineage -- decayed-weight neighbor smoothing (offset d gets wnb**d instead of
    flat wnb for every offset): Phase B's own window=2/3 regression (21.2341/21.3163,
    experiments.json #8) used FLAT neighbor weights out to +-2/+-3; a decayed scheme lets
    far neighbors contribute less, which might recover some of a wider window's extra
    information without its extra noise.

  YEARWEIGHTS lineage -- the single highest-conviction NEW idea: Phase B invented "down-
    weight an anomalous year" for 2020 (COVID) but never asked whether OTHER years also
    deserve non-1.0 weight. 2019 is the panel's ramp-up year (first year of a new sensor
    program) and 2021 is the year closest in time to the 2022 test year. Pre-flight probe:
    down-weighting 2019 alone (year_weights={"2019":0.7}) already beats the full Phase-B
    root, 21.1487 -> 21.0951 (real, -0.054), and up-weighting 2021 alone is also a small
    win (1.2x -> 21.1125) [PRIOR: "具跨年穩定結構的時空資料" section -- structural signal
    lives entirely in the historical mean here, so any lever that denoises that mean
    further is squarely on-prior].

  PERLOCALPHA lineage -- per-location EB alpha (tau2_loc/(tau2_loc+sigma2_loc/(k*n_year)))
    instead of one global scalar: locations with flatter/noisier weekly profiles shrink
    harder toward their own loc mean. Pre-flight: k=4.0 gets close to (21.1026) but does
    NOT beat the plain scalar alpha=1.0 root alone -- a genuine, moderately-negative new
    idea, kept in the tree as an honest result rather than dropped silently.

  CIRCULAR lineage -- wrap week 52's neighbor to week 0 (the annual cycle has no true
    edge). Pre-flight: worse alone (21.3295) -- probably because week 0/52 (Rwanda's
    seasons) are not actually alike enough for wraparound to be a valid prior; kept as a
    quick, honest negative result.

  MONTHFALLBACK lineage -- add a location-month mean as an extra fallback tier for cells
    with no data. Pre-flight: EXACT tie with root (21.1487, not even in the 4th decimal)
    -- this dataset's panel is perfectly balanced (497 locs x 53 weeks x 3 years, verified
    in eval_s3e20_v2.py), so no cell is EVER missing in any LOYO fold and this fallback
    tier can never fire. Kept in the tree as a documented, expected no-op (a regime
    observation in its own right: structural fallback tiers are worthless on a complete
    panel, useful only on a genuinely sparse one).

  JOINT lineage -- the run's real thesis test: each move changes 2+ structural knobs AT
  ONCE as a single tree-search mutation (not a separate hand-run pre-sweep script). Seed:
  alpha AND w2020 bumped together straight from the root. Subsequent moves fold in wnb,
  then the new year_weights axis, then a further year_weights refinement -- i.e. this
  lineage is where the OTHER lineages' single-axis discoveries get composed together,
  something Phase B's actual per-round PROPOSAL step (always one axis) never did as its
  primary move (only as an occasional manual scratch-pad correction).

  GBDT lineage -- EXACTLY ONE lightweight diagnostic solo node (single LGB, log1p, reduced
  n_estimators=400, LOYO CV) -- not a full 3-model ensemble, since Phase B already proved
  (5 separate blend-weight searches, experiments.json #4-#8) that GBDTs get 0 weight
  against this structural signal; re-running the whole ensemble here would just re-answer
  an already-answered question. One cheap model is enough to re-confirm the finding
  transfers to harness_v2's own OOF-cache + blend-search path.

  BLEND -- exactly ONE blend node: best-structural-TE-so-far + the GBDT diagnostic,
  weight-searched. This is the only "materially different" pair in this run (a structural
  predictor vs a model-based one) -- every OTHER pair of structural variants correlates
  >0.9999 in OOF-space (measured: root vs YEARWEIGHTS-seed corr = 0.99997, and blending
  them gives the weight search 100%/0% -- no gain, exactly as harness_v2's own
  near-duplicate-member behavior elsewhere in this repo predicts) so blending
  structural-vs-structural nodes is deliberately NOT done -- would burn nodes to
  demonstrate something already obvious from the correlation check.

--- Idea-injection (recommendation #4) ---
harness_v2.suggest_priors(COMP_META) is called once at startup; COMP_META's keywords hit
"### 具跨年穩定結構的時空資料" (tag matches "跨年穩定結構"), "## Ensemble/後處理" (tag
"ensemble"), and "### RMSE/極偏態目標" (metric="rmse"). Every mutation description tags
itself `[PRIOR Pk]` when directly shaped by one of those bullets, `[PRIOR none]`
otherwise, exactly as run_s3e16_v2.py/run_s3e5_v2.py do, so prior-usage win-rate is
computable post-hoc.

Writes competitions/playground-series-s3e20/experiments_tree.json (does NOT touch
experiments.json or scripts/train_v5.py per the task brief's constraints).
"""
import copy
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness_v2 as hv2  # noqa: E402
import eval_s3e20_v2 as ev  # noqa: E402

COMP = "playground-series-s3e20"
TREE_PATH = os.path.join(os.path.dirname(_HERE), "competitions", COMP, "experiments_tree.json")
TARGET_NODES = 38        # task brief: >=25 nodes; cheap evals (~0.1s each) leave huge margin
MAX_WALL_S = 20 * 60      # hard stop per task brief
EVAL_TIMEOUT_S = 120
LINEAR_BEST = 21.1487     # scripts/train_v5.py / experiments.json #7

COMP_META = {"metric": "rmse", "tags": ["跨年穩定結構", "ensemble"]}
# matches "### 具跨年穩定結構的時空資料", "### RMSE/極偏態目標" (metric), "## Ensemble/後處理"

ROOT_CONFIG = {"kind": "solo", "method": "te", "alpha": 1.0, "w2020": 0.24, "wnb": 0.28,
               "window": 1}
ROOT_MUTATION = ("root: scripts/train_v5.py's exact Phase-B Round-3 config (EB shrinkage "
                  "alpha=1.0 [neighbor smoothing subsumed Round-1's shrinkage], COVID "
                  "year down-weight w2020=0.24, neighbor-week-smoothing weight wnb=0.28, "
                  "window=+-1), OOF RMSE 21.1487 (experiments.json #7), verified digit-"
                  "for-digit before searching begins")

DEDUP_REJECTIONS = []
_dedup_offset = {}


def dc(cfg):
    return copy.deepcopy(cfg)


def find_dup(tree, cfg):
    target = hv2.config_hash(cfg)
    for n in tree["nodes"]:
        if hv2.config_hash(n["config"]) == target:
            return n["id"]
    return None


# ---------------------------------------------------------------------------
# first-generation seeds (children of root)
# ---------------------------------------------------------------------------
def seed_alpha():
    cfg = dict(ROOT_CONFIG, alpha=0.98)
    desc = ("ALPHA fine re-sweep: 1.0->0.98 -- pre-flight probe found this the local "
            "single-axis optimum near Phase B's own alpha=1.0 (21.1487->21.1460, tiny "
            "but real; Phase B's Round-3 pre-sweep never checked alpha values strictly "
            "between 0.95 and 1.0) [PRIOR none -- pure numeric re-sweep of an existing knob]")
    return cfg, desc


def seed_w2020():
    cfg = dict(ROOT_CONFIG, w2020=0.22)
    desc = ("W2020 fine re-sweep: 0.24->0.22 -- HONEST check whether Phase B's Round-3 "
            "joint pre-sweep already found this axis's true optimum; pre-flight grid "
            "(0.15..0.32) shows 0.24 IS the single-axis minimum (0.22 alone scores "
            "21.1508, worse) [PRIOR none]")
    return cfg, desc


def seed_wnb():
    cfg = dict(ROOT_CONFIG, wnb=0.30)
    desc = ("WNB fine re-sweep: 0.28->0.30 -- same honest-check purpose as W2020; "
            "pre-flight grid (0.20..0.40) shows 0.28 IS the single-axis minimum "
            "[PRIOR none]")
    return cfg, desc


def seed_window():
    cfg = dict(ROOT_CONFIG, window=2, neighbor_weight_mode="decay")
    desc = ("WINDOW: widen to +-2 weeks but with DECAYED weight (wnb**d per offset "
            "distance d) instead of Phase B's flat wnb for every offset -- Phase B's "
            "flat-weight +-2 was worse (21.2341, experiments.json #8); decay lets far "
            "neighbors contribute less, an untested combination [PRIOR: 具跨年穩定結構 "
            "section -- denoising the historical mean is the entire lever on this comp]")
    return cfg, desc


def seed_yearweights():
    cfg = dict(ROOT_CONFIG, year_weights={"2019": 0.7, "2020": 0.24})
    desc = ("YEARWEIGHTS (NEW axis): down-weight 2019 (ramp-up year of the sensor "
            "program, potentially less reliable) in addition to Phase B's own 2020 "
            "COVID down-weight -- Phase B invented 'down-weight an anomalous year' but "
            "never asked whether it applies to OTHER years too. Pre-flight: real "
            "improvement, 21.1487->21.0951 [PRIOR: 具跨年穩定結構 section -- same "
            "denoise-the-historical-mean lever generalized to a 2nd year]")
    return cfg, desc


def seed_perlocalpha():
    cfg = dict(ROOT_CONFIG, per_loc_alpha=True, per_loc_k=4.0, alpha=1.0)
    desc = ("PERLOCALPHA (NEW axis): replace the single global alpha with a per-location "
            "empirical-Bayes value (tau2_loc/(tau2_loc+sigma2_loc/(k*n_year)) -- "
            "locations with flatter/noisier weekly profiles shrink harder toward their "
            "own loc mean, strength constant k=4.0 chosen from a pre-flight k-grid "
            "(0.25..8.0). Phase B's alpha was always a single global scalar. Pre-flight: "
            "21.1026, close to but NOT beating the root's plain scalar alpha=1.0 -- "
            "kept as an honest (moderately negative) new-idea result [PRIOR: 具跨年穩定"
            "結構 section]")
    return cfg, desc


def seed_circular():
    cfg = dict(ROOT_CONFIG, circular=True)
    desc = ("CIRCULAR (NEW axis): wrap the neighbor-week lookup across the year boundary "
            "(week 52's +1 neighbor becomes week 0) instead of leaving edge weeks with "
            "only one real neighbor -- the annual cycle has no true edge. Pre-flight: "
            "worse (21.3295) -- Rwanda's week-0/week-52 seasons are evidently not alike "
            "enough for this prior to hold; kept as a quick honest negative result "
            "[PRIOR: 具跨年穩定結構 section, negative outcome]")
    return cfg, desc


def seed_monthfallback():
    cfg = dict(ROOT_CONFIG, month_fallback=True)
    desc = ("MONTHFALLBACK (NEW axis): add a location-month mean as an extra fallback "
            "tier for any (loc,week) cell with no data in the source years. Pre-flight: "
            "EXACT tie with root (21.1487) -- this dataset's panel is PERFECTLY BALANCED "
            "(497 locs x 53 weeks x 3 years, every combination present every year), so "
            "the fallback tier can mathematically never fire in any LOYO fold. Kept as a "
            "documented, expected no-op -- itself a regime observation (fallback tiers "
            "are worthless on a complete panel) [PRIOR: 具跨年穩定結構 section]")
    return cfg, desc


def seed_gbdt():
    cfg = {"kind": "solo", "method": "gbdt_lgb_quick",
           "params": {"n_estimators": 400, "learning_rate": 0.05}}
    desc = ("GBDT diagnostic (the ONE allowed per task brief): single lightweight LGB "
            "(log1p target, 63 sensor cols + lat/lon/week, n_estimators=400, LOYO CV) -- "
            "NOT a full 3-model ensemble, since Phase B already proved GBDTs get 0 blend "
            "weight against this structural signal 5 separate times (experiments.json "
            "#4-#8); this re-confirms the finding transfers to harness_v2's own OOF-cache "
            "+ blend-search path without re-answering an already-answered question "
            "[PRIOR: 具跨年穩定結構 section -- GBDT-vs-structure is this section's own "
            "central finding]")
    return cfg, desc


NEW_SOLO_SEEDS = [
    ("ALPHA", seed_alpha), ("W2020", seed_w2020), ("WNB", seed_wnb),
    ("WINDOW", seed_window), ("YEARWEIGHTS", seed_yearweights),
    ("PERLOCALPHA", seed_perlocalpha), ("CIRCULAR", seed_circular),
    ("MONTHFALLBACK", seed_monthfallback), ("GBDT", seed_gbdt),
]


# ---------------------------------------------------------------------------
# per-lineage authored mutation queues
# ---------------------------------------------------------------------------
def _set(cfg, **kw):
    c = dc(cfg)
    c.update(kw)
    return c


ALPHA_QUEUE = [
    lambda c: (_set(c, alpha=0.99), "ALPHA: 0.98->0.99, bracket other direction [PRIOR none]"),
    lambda c: (_set(c, alpha=0.965), "ALPHA: 0.98->0.965, refine toward finer optimum [PRIOR none]"),
    lambda c: (_set(c, alpha=0.95), "ALPHA: push further to 0.95 [PRIOR none]"),
]
W2020_QUEUE = [
    lambda c: (_set(c, w2020=0.26), "W2020: 0.22->0.26, bracket other direction around root's 0.24 [PRIOR none]"),
    lambda c: (_set(c, w2020=0.25), "W2020: fine step 0.25 [PRIOR none]"),
]
WNB_QUEUE = [
    lambda c: (_set(c, wnb=0.26), "WNB: 0.30->0.26, bracket other direction around root's 0.28 [PRIOR none]"),
    lambda c: (_set(c, wnb=0.29), "WNB: fine step 0.29 [PRIOR none]"),
]
WINDOW_QUEUE = [
    lambda c: (_set(c, window=3, neighbor_weight_mode="decay"),
               "WINDOW: widen decayed window +-2->+-3 -- does decay keep improving with "
               "more neighbors or already peaked at +-2? [PRIOR none]"),
    lambda c: (_set(c, window=2, neighbor_weight_mode="decay", wnb=0.45),
               "WINDOW: same +-2 decayed window, larger base wnb (0.28->0.45) since decay "
               "damps far neighbors anyway -- tests whether decay mode wants a different "
               "base weight than flat mode's optimum [PRIOR none]"),
]
YEARWEIGHTS_QUEUE = [
    lambda c: (_set(c, year_weights={"2019": 0.6, "2020": 0.24}),
               "YEARWEIGHTS: 2019 weight 0.7->0.6, pre-flight grid showed 0.6-0.7 is the "
               "flat minimum region [PRIOR: 具跨年穩定結構 section]"),
    lambda c: (_set(c, year_weights={"2019": 0.65, "2020": 0.24}),
               "YEARWEIGHTS: fine step 2019 weight=0.65 [PRIOR: 具跨年穩定結構 section]"),
    lambda c: (_set(c, year_weights={"2019": 0.65, "2020": 0.24, "2021": 1.15}),
               "YEARWEIGHTS: ALSO up-weight 2021 (closest year to the 2022 test year) "
               "within the same year-weights axis -- pre-flight found w21=1.2 alone is "
               "also a small win (21.1125); combine with the already-better 2019 weight "
               "[PRIOR: 具跨年穩定結構 section]"),
]
PERLOCALPHA_QUEUE = [
    lambda c: (_set(c, per_loc_k=2.0), "PERLOCALPHA: k 4.0->2.0, pre-flight grid direction toward more shrinkage [PRIOR none]"),
    lambda c: (_set(c, per_loc_k=8.0), "PERLOCALPHA: k 4.0->8.0, other bracket direction (less shrinkage, closer to plain scalar) [PRIOR none]"),
]
CIRCULAR_QUEUE = [
    lambda c: (_set(c, circular=True, wnb=0.20),
               "CIRCULAR: lower wnb (0.28->0.20) to test whether circular wraparound "
               "only hurt because it was injecting a full-weight but poorly-matched "
               "neighbor at the season boundary [PRIOR none]"),
]
MONTHFALLBACK_QUEUE = []  # proven exact no-op at the seed; no further exploration warranted
GBDT_QUEUE = []           # exactly one diagnostic per task brief; no further GBDT nodes

SOLO_QUEUES = {
    "ALPHA": ALPHA_QUEUE, "W2020": W2020_QUEUE, "WNB": WNB_QUEUE, "WINDOW": WINDOW_QUEUE,
    "YEARWEIGHTS": YEARWEIGHTS_QUEUE, "PERLOCALPHA": PERLOCALPHA_QUEUE,
    "CIRCULAR": CIRCULAR_QUEUE, "MONTHFALLBACK": MONTHFALLBACK_QUEUE, "GBDT": GBDT_QUEUE,
}


def solo_fallback(lineage_name, parent_cfg, attempt):
    """Small deterministic perturbation of the lineage's primary numeric knob, used only
    once each lineage's authored queue is exhausted."""
    c = dc(parent_cfg)
    rng = np.random.default_rng(9000 + attempt)
    primary = {"ALPHA": "alpha", "W2020": "w2020", "WNB": "wnb",
               "PERLOCALPHA": "per_loc_k"}.get(lineage_name)
    if primary is None or primary not in c:
        return None
    step = rng.normal(0, 0.02 if primary != "per_loc_k" else 0.5)
    c[primary] = max(0.01, c[primary] + step)
    desc = (f"fallback perturbation (queue exhausted): {primary} += {step:+.4f} "
            f"[PRIOR none -- last-resort filler]")
    return c, desc


# ---------------------------------------------------------------------------
# JOINT lineage: each move changes >=2 structural knobs at once, straight from root
# ---------------------------------------------------------------------------
def seed_joint():
    cfg = dict(ROOT_CONFIG, alpha=0.98, w2020=0.20)
    desc = ("JOINT (2-param move in ONE step, straight from root): alpha 1.0->0.98 AND "
            "w2020 0.24->0.20 TOGETHER -- this is the run's central thesis test: Phase "
            "B's per-round PROPOSAL step always changed exactly one axis (STATUS.md's "
            "own Round headers: 一次一項); joint re-tuning only ever happened as a "
            "separate, ad hoc manual scratch pre-sweep, never as the tree's own single "
            "mutation. A tree-search node can just BE a 2-param move [PRIOR: 具跨年穩定"
            "結構 section]")
    return cfg, desc


JOINT_QUEUE = [
    lambda c: (_set(c, wnb=0.30, year_weights={"2019": 0.7, "2020": c.get("w2020", 0.20)}),
               "JOINT (3rd+4th axis in one move): fold in wnb 0.28->0.30 AND the NEW "
               "year_weights axis (2019=0.7) on top of the alpha+w2020 move already made "
               "-- 4 structural knobs moved across 2 tree nodes total, something no "
               "single Phase-B round attempted [PRIOR: 具跨年穩定結構 section]"),
    lambda c: (_set(c, year_weights={"2019": 0.6, "2020": c.get("w2020", 0.20), "2021": 1.2}),
               "JOINT: refine year_weights further (2019 0.7->0.6, add 2021=1.2 up-weight) "
               "while keeping alpha/w2020/wnb fixed at this lineage's current point -- "
               "tests whether the single-axis YEARWEIGHTS-lineage's own best point still "
               "helps once composed with the other 3 axes [PRIOR: 具跨年穩定結構 section]"),
    lambda c: (_set(c, alpha=0.99), "JOINT: fine nudge alpha 0.98->0.99 at this joint point -- local refinement of the composed optimum [PRIOR none]"),
    lambda c: (_set(c, w2020=0.22), "JOINT: fine nudge w2020 0.20->0.22 at this joint point [PRIOR none]"),
]


# ---------------------------------------------------------------------------
# BLEND lineage: exactly one node, best-structural-TE + the GBDT diagnostic
# ---------------------------------------------------------------------------
def best_te_node(tree):
    best = None
    for n in tree["nodes"]:
        if (n["status"] == "evaluated" and n["config"].get("kind") == "solo"
                and n["config"].get("method") == "te"):
            if best is None or n["score"] < best["score"]:
                best = n
    return best


def gbdt_node(tree):
    for n in tree["nodes"]:
        if n["config"].get("method") == "gbdt_lgb_quick" and n["status"] == "evaluated":
            return n
    return None


# ---------------------------------------------------------------------------
def propose_child(tree, parent_id, lineage_id, lineage_names):
    parent_node = next(n for n in tree["nodes"] if n["id"] == parent_id)
    name = lineage_names[lineage_id]
    idx = hv2.lineage_size(tree, lineage_id) - 1 + _dedup_offset.get(lineage_id, 0)
    if name == "JOINT":
        queue = JOINT_QUEUE
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            return None, None
    else:
        queue = SOLO_QUEUES.get(name, [])
        if idx < len(queue):
            child_cfg, desc = queue[idx](parent_node["config"])
        else:
            result = solo_fallback(name, parent_node["config"], idx - len(queue))
            if result is None:
                return None, None
            child_cfg, desc = result
    return child_cfg, f"[{name}] {desc} (parent=#{parent_id})"


LINEAGE_NAMES = {}


def eval_and_add(tree, parent_id, mutation, child_cfg, is_root=False, lineage_id=None):
    dup_id = None if is_root else find_dup(tree, child_cfg)
    if dup_id is not None:
        DEDUP_REJECTIONS.append(dict(mutation=mutation, dup_id=dup_id))
        if lineage_id is not None:
            _dedup_offset[lineage_id] = _dedup_offset.get(lineage_id, 0) + 1
        return None, dup_id, None

    nid = hv2.next_id(tree)
    r = ev.evaluate(child_cfg, node_id=nid, timeout_s=EVAL_TIMEOUT_S)
    stored_cfg = child_cfg
    if r["status"] == "evaluated" and r.get("result") is not None:
        stored_cfg = {**child_cfg, "result": r["result"]}
    if is_root:
        real_nid = hv2.add_root(tree, mutation, stored_cfg, r["score"], r["status"], r["wall_s"])
    else:
        real_nid, add_dup = hv2.add_node(tree, parent_id, mutation, stored_cfg, r["score"],
                                          r["status"], r["wall_s"], allow_duplicate=True)
        assert add_dup is None
    assert real_nid == nid, f"id-prediction mismatch: predicted {nid}, got {real_nid}"
    hv2.save(tree, TREE_PATH)
    return real_nid, None, r


def score_of(r):
    return None if r is None else r.get("score")


_PRIOR_TAG_RE = re.compile(r"\[PRIOR([^\]]*)\]")


def prior_usage_summary(tree):
    by_id = {n["id"]: n for n in tree["nodes"]}
    informed, uninformed = [], []
    for n in tree["nodes"]:
        if n["status"] != "evaluated" or n["parent_id"] is None or n["parent_id"] == tree["root_id"]:
            continue
        m = _PRIOR_TAG_RE.search(n["mutation"])
        if not m:
            continue
        tag = m.group(1).strip()
        parent = by_id.get(n["parent_id"])
        if parent is None or parent["score"] is None or n["score"] is None:
            continue
        won = n["score"] < parent["score"]  # lower RMSE is better
        (informed if tag != "none" else uninformed).append(dict(node_id=n["id"], tag=tag, won=won))

    def rate(lst):
        return (sum(1 for x in lst if x["won"]) / len(lst)) if lst else None

    return dict(informed=informed, uninformed=uninformed,
                informed_win_rate=rate(informed), uninformed_win_rate=rate(uninformed))


def verify_root_digit_for_digit():
    print("=== root digit-for-digit verification (BEFORE searching) ===")
    score, _ = ev.score_te(ROOT_CONFIG)
    print(f"  ROOT_CONFIG {ROOT_CONFIG} -> OOF RMSE {score:.6f} vs scripts/train_v5.py's "
          f"21.1487 (experiments.json #7)")
    assert abs(score - LINEAR_BEST) < 1e-3, (
        f"ROOT DOES NOT REPRODUCE linear-best digit-for-digit: {score} vs {LINEAR_BEST}")
    print("=== verification passed ===\n")


def main():
    t_start = time.time()
    verify_root_digit_for_digit()

    if os.path.exists(TREE_PATH):
        tree = hv2.load(TREE_PATH)
        print(f"Resuming existing tree at {TREE_PATH} ({len(tree['nodes'])} nodes)")
    else:
        tree = hv2.new_tree(COMP)
        priors = hv2.suggest_priors(COMP_META)
        tree["priors"] = priors
        print(f"suggest_priors({COMP_META}) -> {len(priors)} bullets:")
        for i, p in enumerate(priors):
            print(f"  P{i}: {p[:200]}")

    def n_evaluated():
        return sum(1 for n in tree["nodes"] if n["status"] == "evaluated")

    # --- root ---
    if not tree["nodes"]:
        nid, dup, r = eval_and_add(tree, None, ROOT_MUTATION, ROOT_CONFIG, is_root=True)
        print(f"[root] #{nid} rmse={score_of(r)} wall_s={r['wall_s']}")

    # --- first-generation lineage seeds ---
    all_names = [n for n, _ in NEW_SOLO_SEEDS] + ["JOINT"]
    for name, seed_fn in NEW_SOLO_SEEDS:
        already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
        if any(n["mutation"].startswith(f"[{name}]") for n in already):
            continue
        cfg, desc = seed_fn()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[{name}] {desc}", cfg)
        print(f"[{name} seed] #{nid} rmse={score_of(r)} status={r['status']} wall_s={r['wall_s']}")

    already = [n for n in tree["nodes"] if n["parent_id"] == tree["root_id"]]
    if not any(n["mutation"].startswith("[JOINT]") for n in already):
        cfg, desc = seed_joint()
        nid, dup, r = eval_and_add(tree, tree["root_id"], f"[JOINT] {desc}", cfg)
        print(f"[JOINT seed] #{nid} rmse={score_of(r)} status={r['status']} wall_s={r['wall_s']}")

    # rebuild LINEAGE_NAMES (resume-safety)
    for n in tree["nodes"]:
        if n["parent_id"] == tree["root_id"] and n["id"] not in LINEAGE_NAMES:
            for name in all_names:
                if n["mutation"].startswith(f"[{name}]"):
                    LINEAGE_NAMES[n["id"]] = name

    # --- adaptive tree-search loop over the SOLO/JOINT lineages ---
    iterations = 0
    while n_evaluated() < TARGET_NODES - 1 and (time.time() - t_start) < MAX_WALL_S:  # -1 leaves room for the BLEND node
        iterations += 1
        if iterations > 200:
            print("Safety cap on iterations reached, stopping.")
            break
        parent_id, lineage_id = hv2.select_next_parent(tree)
        if parent_id is None:
            print("select_next_parent exhausted (no more eligible expansions). Stopping search loop.")
            break
        child_cfg, mutation = propose_child(tree, parent_id, lineage_id, LINEAGE_NAMES)
        if child_cfg is None:
            st = tree["search_state"]
            if lineage_id not in st["plateaued"]:
                st["plateaued"].append(lineage_id)
                st["backtrack_log"].append(dict(
                    at_node_id=None, plateaued_lineage=lineage_id,
                    reason=(f"lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)}'s "
                            f"mutation proposals exhausted -> forced backtrack")))
            hv2.save(tree, TREE_PATH)
            print(f"FORCED BACKTRACK: lineage {LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"exhausted; plateaued={tree['search_state']['plateaued']}")
            continue
        nid, dup, r = eval_and_add(tree, parent_id, mutation, child_cfg, lineage_id=lineage_id)
        if nid is None:
            print(f"DEDUP REJECTED <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
                  f"(identical to existing node #{dup}); retrying with next mutation-queue slot")
            continue
        gb = hv2.global_best(tree)
        print(f"#{nid} <- parent#{parent_id} lineage={LINEAGE_NAMES.get(lineage_id, lineage_id)} "
              f"rmse={score_of(r)} status={r['status']} wall_s={r['wall_s']} "
              f"| global_best={gb['score'] if gb else None} (#{gb['id'] if gb else '-'}) "
              f"| plateaued={tree['search_state']['plateaued']} "
              f"| elapsed={time.time() - t_start:.1f}s")

    # --- BLEND: best structural TE so far + the one GBDT diagnostic ---
    already_blend = any(n["config"].get("kind") == "blend" for n in tree["nodes"])
    if not already_blend:
        best_te = best_te_node(tree)
        gbdt = gbdt_node(tree)
        if best_te is not None and gbdt is not None:
            cfg = {"kind": "blend", "members": [best_te["id"], gbdt["id"]], "weight_search": "dirichlet"}
            desc = (f"BLEND (the one materially-different pair this run tests): best "
                    f"structural TE node so far (#{best_te['id']}, rmse={best_te['score']}) "
                    f"+ the GBDT diagnostic (#{gbdt['id']}, rmse={gbdt['score']}) -- "
                    f"weight-searched. Structural-vs-structural blends were deliberately "
                    f"NOT tried (root vs YEARWEIGHTS-seed OOF correlation = 0.99997, "
                    f"blend weight search already confirmed to give 100%/0% -- see module "
                    f"docstring) [PRIOR: Ensemble/後處理 section -- weight search "
                    f"arbitrates, no cost to trying once]")
            nid, dup, r = eval_and_add(tree, tree["root_id"], f"[BLEND] {desc}", cfg)
            print(f"[BLEND] #{nid} rmse={score_of(r)} status={r['status']} wall_s={r['wall_s']}")
        else:
            print("BLEND skipped: missing best_te or gbdt node")

    total_wall = time.time() - t_start
    gb = hv2.global_best(tree)
    n_solo = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "solo")
    n_blend = sum(1 for n in tree["nodes"] if n["status"] == "evaluated" and n["config"].get("kind") == "blend")
    prior_stats = prior_usage_summary(tree)
    tree["prior_usage_summary"] = {
        "n_informed": len(prior_stats["informed"]), "n_uninformed": len(prior_stats["uninformed"]),
        "informed_win_rate": prior_stats["informed_win_rate"],
        "uninformed_win_rate": prior_stats["uninformed_win_rate"],
    }
    tree["dedup_rejections"] = DEDUP_REJECTIONS
    tree["final_tie_rate"] = round(hv2.tie_rate(tree), 4)
    tree["linear_best_rmse"] = LINEAR_BEST
    hv2.save(tree, TREE_PATH)

    print(f"\nDone. {n_evaluated()} evaluated nodes ({n_solo} solo / {n_blend} blend), "
          f"{len(tree['nodes'])} total, wall={total_wall:.1f}s")
    print(f"Global best: #{gb['id']} rmse={gb['score']} mutation={gb['mutation'][:150]}")
    print(f"Linear-iteration best: {LINEAR_BEST} (v2 tree "
          f"{'BEAT' if gb['score'] < LINEAR_BEST else ('MATCHED' if gb['score'] == LINEAR_BEST else 'did NOT beat')} it)")
    print(f"Final tie_rate: {hv2.tie_rate(tree):.4f}")
    print(f"Backtrack log ({len(tree['search_state']['backtrack_log'])} events):")
    for e in tree["search_state"]["backtrack_log"]:
        print(" ", e)
    print(f"Dedup rejections ({len(DEDUP_REJECTIONS)}):")
    for e in DEDUP_REJECTIONS:
        print(" ", e)
    print(f"Prior usage: informed={len(prior_stats['informed'])} "
          f"(win rate={prior_stats['informed_win_rate']}), "
          f"uninformed={len(prior_stats['uninformed'])} "
          f"(win rate={prior_stats['uninformed_win_rate']})")


if __name__ == "__main__":
    main()
