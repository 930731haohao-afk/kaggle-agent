"""tree_search/harness_v4.py — Phase J J-2: the external-idea-injection hook (the second
ERA pillar / 計畫書 4.1「階段 5:外部想法注入」). harness_v3.py (the Phase F-1/H-1 search
core) is left COMPLETELY untouched for reproducibility and because a live s5e10 v3 run
imports it; this module `import harness_v3 as v3`, re-exports every v3 search-core piece
unchanged (so a driver can `import harness_v4 as hv4` and get a strict superset of v3), and
adds ONLY the `[EXT]` idea-injection path on top.

--- Why this module exists (the ERA consumption model) -------------------------------------
Stage 4 (tree search) already consumes the INTERNAL experience library via
`harness_v3.suggest_priors(comp_meta)` — keyword-to-header substring matching against
`knowledge/experience.md`, whose bullets are `[INT]` (this project's own cross-competition,
score-diff-backed lessons). Stage 5 is "stage 4 + external-literature idea injection": once
the internal priors are exhausted, we ALSO inject `[EXT]` candidates — techniques the
literature/community consider effective (real DOIs / champion write-ups / library docs),
distilled into `knowledge/idea_bank.md`. idea_bank.md was authored to mirror suggest_priors'
matchable structure (`##`/`###` headers carrying comparable tokens + bulleted body), so the
SAME substring mechanism pools `[EXT]` priors alongside `[INT]` ones.

--- What v4 adds (each documented at its definition below) --------------------------------
1. `parse_idea_bank()` — parse idea_bank.md's `### [EXT-NN]` entries (reusing
   `harness_v2._split_sections`, the exact splitter suggest_priors uses), attaching each
   entry to its parent `## ` category so category tokens propagate to the entry's haystack.
2. `parse_dedup_suppress_ids()` — read idea_bank.md's tail "附:...供 v4 併池去重參考" table's
   "已被 [INT] 實證的先驗 ... 避免重複觸發" bullet → the set of `[EXT]` ids that internal
   experience already proved, so they are NOT re-fired as if they were new.
3. `suggest_ext_priors()` — the `[EXT]` analogue of `suggest_priors`: substring-match
   comp_meta keywords against each entry's haystack, returning provenance-tagged prior
   dicts (`provenance='EXT'`, `source='EXT-NN'`), with the dedup-suppress set applied.
4. `suggest_priors_v4(comp_meta, mode=...)` — the pooled entry point. `mode='off'` (stage-4
   baseline) returns ONLY `[INT]` (byte-for-byte the same lines as `harness_v3.suggest_priors`,
   just wrapped with `provenance='INT'`); `mode='ext'` (stage-5) returns `[INT]` + the
   net-new (dedup-suppressed) `[EXT]` priors. The `mode` flag is exactly the switch the J-4
   attribution run flips to isolate the marginal contribution of external-idea injection.

Every prior is a dict `{"text", "provenance", "source", ...}` so J-4 can attribute each
mutation to `[INT]` vs `[EXT]` (vs no-prior). `prior_texts(priors)` flattens back to the
bare `list[str]` that `suggest_priors` returns — and by construction
`prior_texts(suggest_priors_v4(cm, mode='off')) == harness_v3.suggest_priors(cm)` exactly, so
turning injection OFF cannot perturb the stage-4 baseline.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # so `import harness_v3` resolves regardless of the caller's cwd
    sys.path.insert(0, _HERE)
import harness_v3 as v3  # noqa: E402 -- tree_search/harness_v3.py, unchanged (search core)
import harness_v2 as v2  # noqa: E402 -- for _split_sections / _load_experience_text / DEFAULT_EXPERIENCE_PATH

# --- re-export the ENTIRE v3 search core unchanged -----------------------------------------
# A driver may `import harness_v4 as hv4` and use it as a strict superset of harness_v3: the
# node schema, add_node/add_root, select_next_parent, the budget/phase machine, eval_blend,
# the resume contract, subprocess eval, and the burst-seed sanity gate are all v3's, verbatim.
new_tree = v3.new_tree
load = v3.load
save = v3.save
next_id = v3.next_id
lineage_of = v3.lineage_of
global_best = v3.global_best
lineage_size = v3.lineage_size
select_next_parent = v3.select_next_parent
config_hash = v3.config_hash
find_duplicate_config = v3.find_duplicate_config
cache_oof = v3.cache_oof
load_oof = v3.load_oof
tie_rate = v3.tie_rate
add_root = v3.add_root
add_node = v3.add_node
suggest_priors = v3.suggest_priors  # the [INT]-only hook, unchanged (stage-4 behavior)

init_budget = v3.init_budget
n_evaluated = v3.n_evaluated
plateau_saturated = v3.plateau_saturated
update_phase = v3.update_phase
should_stop = v3.should_stop
reopen_blend_lineage_on_solo_breakthrough = v3.reopen_blend_lineage_on_solo_breakthrough
boundary_candidates = v3.boundary_candidates
eval_blend = v3.eval_blend
eval_blend_with_cost_guard = v3.eval_blend_with_cost_guard
validate_state = v3.validate_state
save_search_state = v3.save_search_state
load_search_state = v3.load_search_state
eval_solo_subprocess = v3.eval_solo_subprocess
burst_seed_sanity_bound = v3.burst_seed_sanity_bound
burst_seed_sanity_gate = v3.burst_seed_sanity_gate
apply_burst_seed_sanity_gate = v3.apply_burst_seed_sanity_gate

MAX_CHILDREN_PER_NODE = v3.MAX_CHILDREN_PER_NODE
PLATEAU_STREAK = v3.PLATEAU_STREAK
ADAPTIVE_PLATEAU_STREAK = v3.ADAPTIVE_PLATEAU_STREAK
TIE_RATE_THRESHOLD = v3.TIE_RATE_THRESHOLD
DEFAULT_TOTAL_BUDGET = v3.DEFAULT_TOTAL_BUDGET

# ---------------------------------------------------------------------------
# v4 paths & constants
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(_HERE)
DEFAULT_EXPERIENCE_PATH = v2.DEFAULT_EXPERIENCE_PATH                       # knowledge/experience.md ([INT])
DEFAULT_IDEA_BANK_PATH = os.path.join(_REPO_ROOT, "knowledge", "idea_bank.md")  # [EXT]
DEFAULT_MAX_ITEMS = 20            # mirror suggest_priors' own default cap
VALID_MODES = ("off", "ext")     # 'off' = stage-4 ([INT] only); 'ext' = stage-5 ([INT]+[EXT])

_EXT_ID_RE = re.compile(r"EXT-(\d+)")        # entry header "[EXT-14] ..." -> 14
_EXT_TOKEN_RE = re.compile(r"EXT-[\d/]+")    # dedup table token, incl. compact "EXT-05/06"
_NUM_RE = re.compile(r"\d+")
_ID_PREFIX_RE = re.compile(r"^\[EXT-\d+\]\s*")


def _norm_ext_id(num) -> str:
    """Canonicalize an EXT id number to zero-padded 'EXT-NN' (so '5' and '05' agree)."""
    return f"EXT-{int(num):02d}"


def _as_comp_meta(comp_meta):
    """Accept either a full comp_meta dict (like suggest_priors) OR a bare metric string
    (so `suggest_priors_v4('auc', 'ext')` works too). Returns a dict."""
    if isinstance(comp_meta, str):
        return {"metric": comp_meta}
    return comp_meta or {}


def _extract_keywords(comp_meta) -> list:
    """The keyword list suggest_priors matches on, extracted identically: the `metric`
    value plus every value under `tags`/`data_type`/`keywords`, lowercased. Kept in lockstep
    with harness_v2.suggest_priors so `[EXT]` matching uses the exact same comp_meta contract
    as `[INT]` matching."""
    comp_meta = _as_comp_meta(comp_meta)
    keywords = []
    metric = comp_meta.get("metric")
    if metric:
        keywords.append(str(metric))
    for k in ("tags", "data_type", "keywords"):
        v = comp_meta.get(k)
        if v:
            keywords.extend([str(v)] if isinstance(v, str) else [str(x) for x in v])
    return [kw.lower() for kw in keywords if kw]


def _field(lines: list, name: str) -> str:
    """Pull the value of a labelled bullet (e.g. name='機制' from '**機制**:...') out of an
    entry's bullet lines; '' if absent. Tolerant of the fullwidth/halfwidth colon variants
    used across the doc."""
    for ln in lines:
        core = ln.replace("**", "").strip()
        if core.startswith(name):
            return core[len(name):].lstrip("：: 　\t").strip()
    return ""


# ---------------------------------------------------------------------------
# v4 feature 1: parse idea_bank.md's [EXT] entries (same splitter as suggest_priors)
# ---------------------------------------------------------------------------
def parse_idea_bank(idea_bank_path: str = None) -> list:
    """Parse `knowledge/idea_bank.md` into a flat list of `[EXT]` entry dicts, reusing
    `harness_v2._split_sections` — the very splitter `suggest_priors` uses on experience.md —
    so `[EXT]` and `[INT]` are read by one identical markdown mechanism.

    Each returned entry is::

        {"id":       "EXT-14",                 # canonical, zero-padded
         "category": "D. 驗證設計:...",         # nearest preceding `## ` header
         "title":    "[EXT-14] Adversarial validation(...)",
         "title_clean": "Adversarial validation(...)",   # id prefix stripped
         "lines":    ["**出處**:...", "**機制**:...", ...],  # its `- ` bullets, verbatim
         "haystack": "<lowercased category + title + all bullet lines>"}

    The `haystack` deliberately folds in the parent CATEGORY header so a category token (e.g.
    "驗證設計"/"時序" on `## D`) propagates to its entries — unlike experience.md, idea_bank.md
    organizes headers by TECHNIQUE and states metric/data-type applicability inside the
    `**適用條件**` bullet, so matching on the whole entry (header + 條列式內文) is what makes the
    comparable tokens actually reachable (idea_bank.md's own intro describes this "標題含可比對
    關鍵字 + 條列式內文" structure)."""
    idea_bank_path = idea_bank_path or DEFAULT_IDEA_BANK_PATH
    text = v2._load_experience_text(idea_bank_path)
    sections = v2._split_sections(text)

    entries, category = [], None
    for sec in sections:
        if sec["level"] == 2:                 # `## ` category header
            category = sec["title"]
            continue
        m = _EXT_ID_RE.search(sec["title"])   # `### [EXT-NN] ...` entry header
        if not m:
            continue                          # a non-[EXT] `###` (none today), skip defensively
        ext_id = _norm_ext_id(m.group(1))
        title_clean = _ID_PREFIX_RE.sub("", sec["title"]).strip()
        haystack = " ".join([category or "", sec["title"]] + sec["lines"]).lower()
        entries.append(dict(id=ext_id, category=category, title=sec["title"],
                            title_clean=title_clean, lines=list(sec["lines"]),
                            haystack=haystack))
    return entries


# ---------------------------------------------------------------------------
# v4 feature 2: parse the tail dedup table ("v4 併池去重參考")
# ---------------------------------------------------------------------------
def parse_dedup_suppress_ids(idea_bank_path: str = None) -> set:
    """Read idea_bank.md's tail section "附:與 [INT] 的重疊 / 互補一覽(供 v4 併池去重參考)",
    specifically its "**已被 [INT] 實證的先驗(... 避免重複觸發)**" bullet, and return the set of
    `[EXT]` ids that this project's INTERNAL experience already validated — the ideas that
    must NOT double-fire as if they were fresh external candidates (that would double-count
    an idea already represented by an `[INT]` prior). Compact forms like "EXT-05/06" and
    "EXT-07/08" are expanded to {EXT-05, EXT-06} / {EXT-07, EXT-08}.

    Returns `set()` if the section/bullet is absent (fail-open: no suppression rather than a
    crash), so a driver never loses the whole injection path over a doc edit."""
    idea_bank_path = idea_bank_path or DEFAULT_IDEA_BANK_PATH
    text = v2._load_experience_text(idea_bank_path)
    sections = v2._split_sections(text)

    for sec in sections:
        if "併池去重" not in sec["title"] and "重疊" not in sec["title"]:
            continue
        for line in sec["lines"]:
            if "已被" in line and "實證" in line:   # the suppression bullet (not the "反例"/"新方向" ones)
                ids = set()
                for tok in _EXT_TOKEN_RE.findall(line):
                    for num in _NUM_RE.findall(tok):
                        ids.add(_norm_ext_id(num))
                return ids
    return set()


# ---------------------------------------------------------------------------
# v4 feature 3: the [EXT] analogue of suggest_priors
# ---------------------------------------------------------------------------
def _ext_prior_text(entry: dict) -> str:
    """Render one `[EXT]` entry into a single prior line in the same spirit as an
    experience.md bullet (technique + mechanism + applicability), explicitly stamped as an
    external/literature prior with no in-project score evidence — so a reader can never
    mistake an `[EXT]` prior for an `[INT]` proven lesson."""
    mech = _field(entry["lines"], "機制")
    applic = _field(entry["lines"], "適用條件")
    parts = [entry["title_clean"]]
    if mech:
        parts.append(f"機制: {mech}")
    if applic:
        parts.append(f"適用: {applic}")
    parts.append(f"[EXT 文獻先驗 {entry['id']},出處見 idea_bank.md;無本場 [INT] 實證]")
    return " | ".join(parts)


def suggest_ext_priors(comp_meta, idea_bank_path: str = None, max_items: int = DEFAULT_MAX_ITEMS,
                       suppress_ids=None, respect_dedup: bool = True) -> list:
    """The `[EXT]` counterpart of `harness_v3.suggest_priors`: lowercase every comp_meta
    keyword and substring-match it against each `[EXT]` entry's `haystack`; every matching
    entry becomes one provenance-tagged prior dict, in idea_bank.md file order, capped at
    `max_items`::

        {"text": <rendered prior line>, "provenance": "EXT", "source": "EXT-14",
         "category": <category>, "lines": [<verbatim bullets>]}

    Dedup (`respect_dedup=True`, the default): any entry whose id is in the suppress set
    (`suppress_ids`, or `parse_dedup_suppress_ids()` when None) is skipped — those ideas are
    already covered by an `[INT]` prior and would double-fire. Pass `respect_dedup=False` to
    get the raw, un-deduped match set (used by the unit test to prove an already-`[INT]`-proven
    entry WOULD have matched, and is excluded specifically by the dedup step, not by a
    no-match).

    Returns `[]` if comp_meta has no usable keywords or nothing matches."""
    keywords = _extract_keywords(comp_meta)
    if not keywords:
        return []
    if respect_dedup:
        suppress_ids = suppress_ids if suppress_ids is not None else parse_dedup_suppress_ids(idea_bank_path)
    else:
        suppress_ids = set()

    out = []
    for e in parse_idea_bank(idea_bank_path):
        if e["id"] in suppress_ids:
            continue
        if any(kw in e["haystack"] for kw in keywords):
            out.append(dict(text=_ext_prior_text(e), provenance="EXT", source=e["id"],
                            category=e["category"], lines=list(e["lines"])))
            if len(out) >= max_items:
                break
    return out


# ---------------------------------------------------------------------------
# v4 feature 4: the pooled, mode-switched entry point
# ---------------------------------------------------------------------------
def suggest_priors_v4(comp_meta, mode: str = "off", experience_path: str = None,
                      idea_bank_path: str = None, max_items: int = DEFAULT_MAX_ITEMS) -> list:
    """Pooled `[INT]`+`[EXT]` prior suggestion with provenance tags and dedup — the single
    hook the J-4 attribution run toggles between stage 4 and stage 5.

    `comp_meta` is the usual dict (`{"metric": "auc", "tags": [...]}`) OR a bare metric string.

    `mode`:
      - `'off'` (stage-4 baseline): return ONLY `[INT]` priors. Each is
        `{"text": <experience.md bullet, verbatim>, "provenance": "INT", "source": "experience.md"}`,
        built directly from `harness_v3.suggest_priors` output IN ORDER — so
        `prior_texts(result) == harness_v3.suggest_priors(comp_meta)` byte-for-byte. Zero
        `[EXT]` injected: switching injection off cannot perturb the stage-4 baseline.
      - `'ext'` (stage-5): `[INT]` priors (as above) followed by the net-new `[EXT]` priors
        from `suggest_ext_priors` (dedup-suppress set applied, so `[EXT]` ideas already proven
        by `[INT]` never double-fire). A final exact-text dedup guards against any cross-channel
        duplicate line (formats differ, so normally a no-op).

    Returns a list of prior dicts (see `suggest_ext_priors` / above for the schema). Use
    `prior_texts()` to flatten to bare strings and `priors_by_provenance()` to split channels."""
    if mode not in VALID_MODES:
        raise ValueError(f"suggest_priors_v4: mode={mode!r} invalid; expected one of {VALID_MODES} "
                         f"('off' = stage-4 [INT]-only baseline, 'ext' = stage-5 [INT]+[EXT])")
    comp_meta = _as_comp_meta(comp_meta)

    int_lines = v3.suggest_priors(comp_meta, experience_path=experience_path, max_items=max_items)
    priors = [dict(text=s, provenance="INT", source="experience.md") for s in int_lines]

    if mode == "ext":
        seen = {p["text"] for p in priors}
        for e in suggest_ext_priors(comp_meta, idea_bank_path=idea_bank_path,
                                    max_items=max_items, respect_dedup=True):
            if e["text"] not in seen:
                seen.add(e["text"])
                priors.append(e)
    return priors


# ---------------------------------------------------------------------------
# small helpers for drivers / the J-4 attribution run
# ---------------------------------------------------------------------------
def prior_texts(priors: list) -> list:
    """Flatten a list of prior dicts to the bare `list[str]` of `text` fields (the shape
    `suggest_priors` returns)."""
    return [p["text"] for p in priors]


def priors_by_provenance(priors: list, provenance: str) -> list:
    """Subset of `priors` whose `provenance` equals `provenance` ('INT' or 'EXT')."""
    return [p for p in priors if p["provenance"] == provenance]
