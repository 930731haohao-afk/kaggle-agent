"""Injection-operator dispatcher — the execution half of the shared contract.

The judgment layer (Stage 0.5) emits typed operators from
`knowledge/injection_operators.md`; this module realizes the ones it implements and
records the rest in a coverage ledger. An idea is never dropped in silence: that
silence is precisely how a correct "GDP as a level covariate" judgment turned into a
plain feature join on s3e19 and cost the run its point.

Realized here: join_feature, ratio_target, log_offset, trend_term, flag_feature,
sample_weight, and the target-free encoding schemes (count / ordinal / crosses).
Emitted as node configs for the search driver to seed: objective, blend_member
(consumed by tree_search/seed_from_ledger.py, which writes back what it seeded).
Recorded as advisory, NOT as done: split_policy and postprocess. This module runs on
dataframes and has no handle on the evaluator, so it cannot check whether the evaluator
implements what was asked -- and for months this branch nevertheless read as "honored by
the evaluator", which is how a pure vocabulary mismatch on s3e19's postprocess survived
unnoticed. `external_data/verify_advisory.py` is the checker that closes that hole; the
ledger entry now points at it instead of asserting a verification that never happened.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .join import merge_holiday_flags, merge_year_safe
from .sources import fetch_holidays, fetch_worldbank, resolve_iso3

# operators this module can realize; everything else lands in the ledger
REALIZED = {"join_feature", "ratio_target", "log_offset", "trend_term",
            "flag_feature", "sample_weight", "encoding"}
# encoding schemes this module can actually compute (target-free, so no leakage path);
# the rest are reported unrealized with the reason, never booked as done
ENCODING_DATA_SCHEMES = {"count", "ordinal", "crosses"}
# Config-only operators: they change what the search proposes, not the data, so they are
# realized as node configs by the search driver rather than as columns here. Recorded as
# realized with the config they imply, so the ledger stays a complete account.
CONFIG_ONLY = {"objective", "blend_member", "encoding"}

# encoding schemes and the evidence behind each
ENCODING_SCHEMES = {
    "native":     "GBDT native categorical handling — the strong default (s3e11: AIDE's only "
                  "outright win came from a node that accidentally enabled it)",
    "ordinal":    "ordered categories keep their order; only for genuinely ordinal levels",
    "count":      "frequency/count encoding — cheap, no target involved, no leakage path",
    "onehot_sparse": "sparse one-hot with a linear member: on all-categorical data this beat "
                     "GBDT outright (cat-in-the-dat), so it belongs in the pool as a member "
                     "rather than as a preprocessing step",
    "target":     "target encoding — REQUIRES fold-aligned computation (see below)",
    "crosses":    "pairwise feature crosses — WARNING: on an already-saturated sparse linear "
                  "model these HURT (cat-in-the-dat); useful mainly for tree members",
}
# Advisory: requested here, executed (or not) by the evaluator. NOT verified by this module --
# see verify_advisory.py. Measured 2026-07-30: of 5 advisory requests across the typed dossiers,
# 4 check out and 1 (s5e10's postprocess) has no mechanism in its evaluator at all.
EVALUATOR_OWNED = {"split_policy", "postprocess"}

# metric family -> (lgb objective, cat loss_function) and the evidence for the choice
OBJECTIVE_MAP = {
    "mae":      ("l1", "MAE", "MAE-family metric: L1 matches the loss the metric charges"),
    "smape":    ("l1", "MAE", "SMAPE is a relative L1 metric; L1 on log-space targets is the "
                              "closest tractable surrogate"),
    "rmse":     ("rmse", "RMSE", "squared-error metric"),
    "rmsle":    ("rmse", "RMSE", "RMSLE = RMSE on log1p targets; transform the target, keep L2"),
    "auc":      ("binary", "Logloss", "AUC is a ranking metric: keep the plain likelihood loss "
                                      "and drop imbalance weighting (s3e3 exp #3: 0.81901 -> "
                                      "0.83292 after removing it; s4e1 confirms the sign at "
                                      "scale, 0.893235 -> 0.893650)"),
    "accuracy": ("binary", "Logloss", "threshold metric: tune the threshold in postprocess, "
                                      "not the loss"),
    "qwk":      ("rmse", "RMSE", "ordinal target: regression head plus an OptimizedRounder beats "
                                 "a multiclass head (s3e5 exp #2->#3: QWK 0.47191 -> 0.52687)"),
}
_WB_PREFIX = "worldbank:"


def proportionality_check(df: pd.DataFrame, cov: pd.DataFrame, mapping: dict,
                          *, country_col: str, date_col: str, target_col: str,
                          threshold: float = 0.05) -> dict:
    """The experience library's precondition for a ratio target (tpsjan22 exp #2/#3):
    total(country, year) / covariate must be near-equal ACROSS countries, leaving only a
    common year drift. Returns per-year cross-country dispersion and the years that break it.

    Evidence for why this gate exists: s3e19 satisfies it in every year (dispersion
    0.006-0.017) and the ratio target is worth -2.36 SMAPE there; sep-2022 satisfies it in
    2017-2019 (~0.01) but 2020 breaks it 40x (0.466), and applying the ratio target blindly
    there COST +0.46 SMAPE. The operator is conditional, not universal.
    """
    d = df.copy()
    d["_year"] = pd.to_datetime(d[date_col]).dt.year
    g = cov.set_index(["iso3", "year"])["value"]
    tot = d.groupby([country_col, "_year"])[target_col].sum().reset_index()
    tot["_cov"] = [g.get((mapping.get(c), y), float("nan"))
                   for c, y in zip(tot[country_col], tot["_year"])]
    tot["_ratio"] = tot[target_col] / tot["_cov"]
    piv = tot.pivot(index="_year", columns=country_col, values="_ratio")
    disp = (piv.std(axis=1) / piv.mean(axis=1)).round(4)
    bad = {int(y): float(v) for y, v in disp.items() if v > threshold}
    # A year whose covariate is missing yields NaN dispersion, and `NaN > threshold` is False:
    # unhandled, the check silently PASSES exactly the years it cannot see. Report them as
    # their own verdict class rather than as evidence of proportionality (2026-08-03 audit).
    unmeasured = sorted(int(y) for y, v in disp.items() if pd.isna(v))
    verdict = "proportional" if not bad else "broken_in_" + ",".join(map(str, bad))
    if unmeasured:
        suffix = "unmeasured_" + ",".join(map(str, unmeasured))
        verdict = suffix if not bad else verdict + ";" + suffix
    return {"dispersion_by_year": {int(y): (None if pd.isna(v) else float(v))
                                   for y, v in disp.items()},
            "threshold": threshold, "violating_years": bad,
            "unmeasured_years": unmeasured, "verdict": verdict}


CURRENT_PRICE_SUFFIX = ("NY.GDP.PCAP.CD", "NY.GDP.MKTP.CD")   # nominal, FX-exposed series


def covariate_volatility_check(cov: pd.DataFrame, mapping: dict, years: list[int],
                               *, threshold_pp: float = 8.0) -> dict:
    """Is the covariate itself a stable proxy for the level, or does it carry FX/inflation shocks?

    A ratio target multiplies the covariate straight into the prediction, so a nominal series
    propagates currency moves as if they were demand moves. Measured on s3e19: current-USD GDP
    per capita moved 2021->2022 by -14.5% (Japan, yen collapse) to +30% (Argentina, inflation),
    a 14.6pp cross-country spread, while the constant-price series moved within 3.0pp. The
    ratio arm lost on the leaderboard to the plain feature join for exactly this reason.
    """
    iso = sorted(set(mapping.values()))
    c = cov[cov["iso3"].isin(iso)].pivot_table(index="year", columns="iso3", values="value")
    # Keep the fetched base year (min(years)-1) so the transition INTO the first panel year
    # is measured; the old filter (`y in years or y - 1 in years`) dropped it, hiding exactly
    # the shock the base year is fetched for (2026-08-03 audit).
    c = c.loc[[y for y in c.index if y in years or y + 1 in years]].sort_index()
    if len(c) < 2:
        # Not a pass: with one year there is no transition to measure, and the caller must
        # treat that as "cannot certify stable", not as "stable".
        return {"verdict": "insufficient_history", "spread_pp_by_year": {},
                "worst_spread_pp": None, "threshold_pp": threshold_pp}
    pct = (c.pct_change() * 100).dropna(how="all")
    spread = pct.std(axis=1).round(2)
    worst = float(spread.max()) if len(spread) else 0.0
    return {"spread_pp_by_year": {int(y): float(v) for y, v in spread.items()},
            "worst_spread_pp": worst, "threshold_pp": threshold_pp,
            "verdict": "stable" if worst <= threshold_pp else "volatile"}


def _fetch_covariate(source: str, years: list[int], countries: list[str]):
    """Return (frame, mapping, meta) for a whitelisted yearly country covariate."""
    if not source.startswith(_WB_PREFIX):
        raise ValueError(f"source '{source}' is not a whitelisted yearly covariate")
    key = source[len(_WB_PREFIX):]
    mapping, unmatched = resolve_iso3(countries)
    if unmatched:
        raise ValueError(f"countries not resolvable to ISO3, refusing to guess: {unmatched}")
    frame, meta = fetch_worldbank(key, min(years) - 1, max(years))
    return frame, mapping, meta


def _key_class_of(source_key: str) -> str:
    """Join key class of a source, from the admitted registry; panel sources default to
    country_year so behaviour is unchanged for everything admitted before the registry."""
    try:
        from .source_registry import JOIN_KEY_CLASSES  # noqa: F401
        rec = _admitted_sources().get(source_key)
        if rec and rec.get("join_key_class"):
            return rec["join_key_class"]
    except Exception:  # noqa: BLE001
        pass
    return "country_year"


def _admitted_sources() -> dict:
    path = Path(__file__).parent / "admitted_sources.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _apply_lookup_join(tr, te, params: dict, plan: dict):
    """Realize a `lookup`-class join_feature: code column -> static attribute columns."""
    from .join import merge_lookup_safe
    src = params["source"]
    rec = _admitted_sources().get(src) or {}
    join = params.get("join") or {}
    df_key = join.get("on") or (rec.get("join_columns") or [None])[0]
    if not df_key:
        raise ValueError(f"lookup join for {src!r} needs params.join.on (the competition's "
                         f"key column) or a join_columns entry in the admitted record")
    fetch = _lookup_fetchers().get(src)
    if fetch is None:
        raise ValueError(f"no fetcher registered for lookup source {src!r} "
                         f"(known: {sorted(_lookup_fetchers())})")
    keys = sorted(set(tr[df_key].astype(str)) | set(te[df_key].astype(str)))
    frame, meta = fetch(keys)
    ext_key = rec.get("ext_key", "code")
    value_cols = rec.get("value_columns") or [c for c in frame.columns if c != ext_key]
    prefix = params.get("as_prefix") or f"{src.split(':')[0]}_"
    outs = []
    for name in ("train", "test"):
        df = tr if name == "train" else te
        joined, report = merge_lookup_safe(df, frame, df_key=df_key, ext_key=ext_key,
                                           value_cols=value_cols, out_prefix=prefix,
                                           meta=meta)
        outs.append(joined)
        if report["rows_unmatched"]:
            plan["unrealized"].append({
                "operator": "join_feature", "params_subset": f"{name}: unresolved keys",
                "reason": f"{report['unresolved_key_count']} key(s) had no entry in {src!r} "
                          f"(e.g. {report['unresolved_keys'][:5]}); those rows carry NaN and "
                          f"are reported rather than filled"})
        # A column that arrived entirely empty is not a feature. It used to be booked
        # realized alongside the columns that did arrive, because the realized/unrealized
        # decision read only rows_unmatched -- which is driven by the FIRST value column
        # (2026-08-04 adversarial verification).
        for empty in report.get("all_null_columns", []):
            plan["unrealized"].append({
                "operator": "join_feature", "params_subset": f"{name}: {empty}",
                "reason": f"column {empty!r} is null for every row: {src!r} supplied the key "
                          f"but no value. Recorded as unrealized rather than counted as a "
                          f"joined feature."})
    added = [f"{prefix}{c}" for c in value_cols]
    plan["added_columns"].extend(added)
    plan["sources"][src] = meta.get("snapshot")
    plan["realized"].append({"operator": "join_feature", "columns": added,
                             "join_key_class": "lookup"})
    return outs[0], outs[1]


def _lookup_fetchers() -> dict:
    """Registered lookup-class fetchers, keyed by the vocabulary name a dossier emits."""
    from .sources import fetch_cpc_titles
    return {"cpc:titles": lambda keys: fetch_cpc_titles(codes=keys)}


def dispatch_route(source_key: str, join_key_class: str) -> str:
    """Which code path would realize this source, or raise if there is none.

    Exists so admission and dispatch cannot disagree. Admitting a source the dispatcher has
    no route to fetch produces the worst kind of failure: the registry says the source is
    available, a dossier emits it in good faith, and the run reports it as an unrealized
    idea with an obscure reason. The admission gate calls this so the disagreement is caught
    at admission time, where the fix is obvious, instead of mid-competition.
    """
    if join_key_class == "lookup":
        if source_key in _lookup_fetchers():
            return f"_apply_lookup_join via _lookup_fetchers()[{source_key!r}]"
        raise ValueError(
            f"no lookup fetcher registered for {source_key!r}. Add it to "
            f"external_data/apply.py:_lookup_fetchers() (known: {sorted(_lookup_fetchers())}); "
            f"admitting a source the dispatcher cannot reach makes the registry lie.")
    if join_key_class == "country_year":
        if source_key.startswith(_WB_PREFIX):
            return f"_fetch_covariate via fetch_worldbank({source_key[len(_WB_PREFIX):]!r})"
        raise ValueError(
            f"the country_year path fetches only {_WB_PREFIX}* sources; {source_key!r} has no "
            f"route. Extending _fetch_covariate to another publisher is a code change, not an "
            f"admission -- until it exists, this source would be admitted but unreachable.")
    if join_key_class == "date":
        # Only the holiday calendar is implemented on this path. Blessing any date-class key
        # here would let a source pass the reachability gate and then be silently realized as
        # holiday flags under its own column name (2026-08-04 adversarial verification).
        if source_key in ("holidays", "calendar"):
            return "flag_feature / calendar path (merge_holiday_flags)"
        raise ValueError(
            f"the date-class path realizes only the holiday calendar; {source_key!r} has no "
            f"route. Add a fetcher and a branch in the flag_feature dispatch first -- "
            f"otherwise this source would be admitted and then silently realized as "
            f"something else.")
    raise ValueError(
        f"join key class {join_key_class!r} has no dispatcher route yet. The class is a legal "
        f"vocabulary entry but nothing in apply.py realizes it, so a dossier emitting "
        f"{source_key!r} would record an unrealized idea rather than a joined column.")


def _config_only(op: str, params: dict) -> dict:
    """Turn a config-only operator into the node config the search driver should seed.

    These operators change what the search *proposes*, not the data, so they land in
    plan["node_configs"] and the driver seeds them as lineages. Keeping them in the ledger
    matters: an objective mismatch is invisible in a score table but decides the metric.
    """
    if op == "objective":
        fam = str(params.get("metric_family", "")).lower()
        if fam not in OBJECTIVE_MAP:
            raise ValueError(f"metric_family {fam!r} unknown (known: {sorted(OBJECTIVE_MAP)})")
        lgb_obj, cat_loss, why = OBJECTIVE_MAP[fam]
        cfg = {"kind": "solo", "seed_role": "objective",
               "params_by_model": {"lgb": {"objective": lgb_obj},
                                   "cat": {"loss_function": cat_loss}},
               "metric_family": fam, "rationale": why}
        if fam in ("auc", "accuracy") and params.get("drop_imbalance_weighting", True):
            cfg["forbid_params"] = ["is_unbalance", "scale_pos_weight", "class_weight"]
        return cfg
    if op == "encoding":
        scheme = str(params.get("scheme", "native"))
        if scheme not in ENCODING_SCHEMES:
            raise ValueError(f"scheme {scheme!r} unknown (known: {sorted(ENCODING_SCHEMES)})")
        cols = params.get("columns")
        cfg = {"kind": "solo", "seed_role": "encoding", "scheme": scheme,
               "columns": cols, "rationale": ENCODING_SCHEMES[scheme]}
        if scheme == "target":
            # The one non-negotiable: s4e1 measured that computing the encoding on folds that
            # differ from the model's evaluation folds is itself a leakage path, even though it
            # looks safer than in-fold encoding. An apparent 0.89653 collapsed to ~0.8937 once
            # the folds were aligned. So the operator carries the requirement, and an evaluator
            # that cannot honour it must refuse rather than approximate.
            cfg["fold_aligned"] = True
            cfg["requires_evaluator_support"] = "per_fold_target_encoding"
            cfg["smoothing"] = float(params.get("smoothing", 20.0))
            cfg["rationale"] += (" — computed inside each fold from that fold's TRAINING rows "
                                 "only, using the model's own folds (s4e1: independent folds "
                                 "with a different seed inflated AUC to 0.89653 vs ~0.8937 "
                                 "fold-aligned; 'looks safer' was another leakage path)")
        if scheme == "onehot_sparse":
            # Needs a linear model family: one-hot into a GBDT is strictly worse than its
            # native categorical handling, so the scheme only pays inside a sparse linear
            # member. Evaluators advertise the capability via their SUPPORTS attribute.
            cfg["requires_evaluator_support"] = "linear_family"
        if scheme == "crosses":
            cfg["max_pairs"] = int(params.get("max_pairs", 10))
            cfg["restrict_to"] = params.get("restrict_to", "tree_members")
        return cfg
    if op == "blend_member":
        archetype = str(params.get("archetype", "shallow_regularized"))
        if archetype != "shallow_regularized":
            raise ValueError(f"archetype {archetype!r} not implemented "
                             "(known: shallow_regularized)")
        # the transferable prototype: a bias-dominated member decorrelates a variance-heavy
        # pool (prior-wiring: largest LLM-member weight in s6e1 +3e-5 R2 and s4e1 +7e-5 AUC,
        # and s5e10's NNLS diagnostic reused the same shape)
        depth = int(params.get("depth", 3))
        return {"kind": "solo", "seed_role": "blend_member",
                "params_by_model": {
                    "lgb": {"num_leaves": 2 ** depth - 1, "min_child_samples": 120,
                            "reg_alpha": 3.0, "reg_lambda": 10.0, "learning_rate": 0.03},
                    "cat": {"depth": depth, "l2_leaf_reg": 12.0, "learning_rate": 0.03}},
                "add_to_pool_not_replace": True,
                "weight_search": params.get("weight_search", "nnls_then_dirichlet"),
                "rationale": ("bias-dominated decorrelator added to the pool rather than "
                              "replacing a member; for RMSE-family metrics verify weights with "
                              "the NNLS closed form (Dirichlet approximation error ~1e-5 can eat "
                              "the real gain)")}
    raise ValueError(f"{op} is not a config-only operator")


def _encode_columns(tr: pd.DataFrame, te: pd.DataFrame, scheme: str,
                    params: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Target-free categorical encodings, computed on train+test jointly.

    Joint computation is safe here precisely because no target is involved: a category's
    frequency or its ordinal position carries no label information, so there is nothing for a
    fold boundary to protect. The one scheme that DOES involve the target (`target`) is refused
    by the caller and routed to `unrealized` with the reason, because computing it anywhere but
    inside the model's own folds is the s4e1 leakage path (AUC inflated to 0.89653, fold-aligned
    ~0.8937).

    Columns are ADDED, never replaced, so a config that drops them reproduces the baseline.
    """
    cols = params.get("columns")
    if not cols:
        # pandas 3 gives string columns a dedicated 'str' dtype, so an `== object` test
        # matches nothing and the operator silently encodes zero columns (2026-08-03 audit).
        cols = [c for c in tr.columns
                if (tr[c].dtype == object or str(tr[c].dtype) == "category"
                    or pd.api.types.is_string_dtype(tr[c]))
                and c in te.columns]
    cols = [c for c in cols if c in tr.columns and c in te.columns]
    if not cols:
        raise ValueError("no categorical columns available to encode "
                         f"(requested {params.get('columns')!r})")
    tr, te = tr.copy(), te.copy()
    added: list[str] = []
    if scheme == "count":
        for c in cols:
            counts = pd.concat([tr[c], te[c]]).value_counts()
            name = f"{c}_count"
            tr[name] = tr[c].map(counts).astype("float64")
            te[name] = te[c].map(counts).astype("float64")
            added.append(name)
    elif scheme == "ordinal":
        # Lexical order is NOT ordinal order (S/M/L/XL sorts to L,M,S,XL; "10" sorts before
        # "2"). The contract promises "ordered categories keep their order", so the order must
        # be declared, or be genuinely numeric. Refusing beats fabricating (2026-08-03 audit).
        declared = params.get("order") or {}
        for c in cols:
            observed = set(tr[c].dropna().unique()) | set(te[c].dropna().unique())
            order = declared.get(c) if isinstance(declared, dict) else declared
            if order:
                missing = [v for v in observed if v not in order]
                if missing:
                    raise ValueError(f"ordinal encoding of {c!r}: declared order omits "
                                     f"{sorted(map(str, missing))[:5]}")
                levels = list(order)
            elif pd.api.types.is_numeric_dtype(pd.Series(sorted(observed))):
                levels = sorted(observed)
            else:
                raise ValueError(
                    f"ordinal encoding of {c!r} needs an explicit params['order'] "
                    f"(observed {sorted(map(str, observed))[:6]}): a lexical sort would "
                    f"fabricate an order this operator's contract promises is real")
            code = {v: i for i, v in enumerate(levels)}
            name = f"{c}_ord"
            tr[name] = tr[c].map(code).astype("float64")
            te[name] = te[c].map(code).astype("float64")
            added.append(name)
    elif scheme == "crosses":
        max_pairs = int(params.get("max_pairs", 6))
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]][:max_pairs]
        if not pairs:
            raise ValueError("crosses needs at least two categorical columns")
        for a, b in pairs:
            name = f"{a}_x_{b}"
            joint = (tr[a].astype(str) + "|" + tr[b].astype(str))
            joint_te = (te[a].astype(str) + "|" + te[b].astype(str))
            levels = sorted(set(joint.unique()) | set(joint_te.unique()))
            code = {v: i for i, v in enumerate(levels)}
            tr[name] = joint.map(code).astype("float64")
            te[name] = joint_te.map(code).astype("float64")
            added.append(name)
    else:                                        # unreachable: caller filters the scheme
        raise ValueError(f"scheme {scheme!r} is not a data-layer encoding")
    return tr, te, added


def apply_operators(train: pd.DataFrame, test: pd.DataFrame, ideas: list[dict], *,
                    country_col: str = "country", date_col: str = "date",
                    target_col: str = "num_sold",
                    ledger_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Realize typed injection operators on (train, test).

    Returns (train2, test2, plan) where `plan` carries the coverage ledger plus the
    target transform the evaluator must apply:
        plan["target_transform"] = {"kind": "ratio_log", "covariate_col": "gdp_pc"}
    means fit on log(target / covariate) and invert with exp(pred) * covariate.
    """
    tr, te = train.copy(), test.copy()

    # Panel keys are resolved LAZILY (07-30). They used to be computed unconditionally here,
    # which raised KeyError('date') on any competition without a date and a country column --
    # including for operators that need neither (encoding, objective, split_policy). That made
    # the dispatcher country-panel-only by accident rather than by declaration.
    _panel: dict = {}

    def panel_keys():
        if not _panel:
            for col, what in ((date_col, "date"), (country_col, "country")):
                if col not in tr.columns or col not in te.columns:
                    raise KeyError(f"operator needs a {what} column ({col!r}); this competition "
                                   f"has {sorted(tr.columns)[:8]}...")
            _panel["years"] = sorted(set(pd.to_datetime(tr[date_col]).dt.year) |
                                     set(pd.to_datetime(te[date_col]).dt.year))
            _panel["countries"] = sorted(set(tr[country_col]) | set(te[country_col]))
        return _panel["years"], _panel["countries"]
    # Ledger buckets, kept separate on purpose (07-30). "realized" used to mix three very
    # different things: columns this module wrote, configs emitted for a driver to seed, and
    # decisions the evaluator owns. Only the first is applied here, so realized_count now
    # counts only that; the rest are reported under their own names and the consumer that
    # picks them up writes its own confirmation (tree_search/seed_from_ledger.py).
    plan: dict = {"proposed": len(ideas), "realized": [], "unrealized": [],
                  "emitted_config": [], "advisory": [],
                  "target_transform": None, "added_columns": [], "sources": {}}

    for idea in ideas:
        op = idea.get("operator")
        params = idea.get("params") or {}
        if op in EVALUATOR_OWNED:
            # NOT a claim that anything executed this. The dispatcher runs on dataframes and
            # has no handle on the evaluator, so it cannot check -- and for months this branch
            # nevertheless read as "honored by the evaluator", which is how a pure vocabulary
            # mismatch on s3e19's postprocess survived unnoticed until 2026-07-30. The entry
            # now says what it actually knows (a request was made, unverified) and names the
            # tool that can check it: external_data/verify_advisory.py.
            plan["advisory"].append({"operator": op, "params": params,
                                     "verified": False,
                                     "note": "REQUEST RECORDED, NOT VERIFIED. This module cannot "
                                             "see the evaluator. Run external_data/"
                                             "verify_advisory.py <comp> to check that the "
                                             "evaluator implements these exact parameters; do "
                                             "not treat this entry as evidence it ran."})
            continue
        if op == "encoding":
            scheme = str(params.get("scheme", "native"))
            if scheme == "native":
                plan["advisory"].append({"operator": op, "scheme": scheme, "note":
                    "native categorical handling is the evaluators' default; nothing to apply"})
                continue
            if scheme not in ENCODING_DATA_SCHEMES:
                # target / onehot_sparse cannot run on dataframes (per-fold computation and a
                # linear family live inside the evaluator), so they are emitted as node
                # configs; the seed driver checks the evaluator's SUPPORTS declaration and
                # books an honest unconsumable entry when the capability is missing.
                # (Until 2026-07-31 this branch refused both schemes outright.)
                try:
                    plan.setdefault("node_configs", []).append(_config_only(op, params))
                    plan["emitted_config"].append({"operator": op, "scheme": scheme, "note":
                        "emitted as a node config; NOT applied here — requires evaluator "
                        "support, which tree_search/seed_from_ledger.py verifies against the "
                        "evaluator's SUPPORTS declaration and records in "
                        "injection_consumed.json"})
                except Exception as e:  # noqa: BLE001
                    plan["unrealized"].append({"operator": op, "scheme": scheme,
                                               "reason": f"{type(e).__name__}: {e}"})
                continue
            try:
                tr, te, cols = _encode_columns(tr, te, scheme, params)
                plan["added_columns"].extend(cols)
                plan["realized"].append({"operator": op, "scheme": scheme, "columns": cols})
            except Exception as e:  # noqa: BLE001
                plan["unrealized"].append({"operator": op, "scheme": scheme,
                                           "reason": f"{type(e).__name__}: {e}"})
            continue
        if op in CONFIG_ONLY:
            try:
                plan.setdefault("node_configs", []).append(_config_only(op, params))
                plan["emitted_config"].append({"operator": op, "note":
                    "emitted as a node config; NOT applied here — the search driver confirms "
                    "consumption in injection_consumed.json"})
            except Exception as e:  # noqa: BLE001
                plan["unrealized"].append({"operator": op, "reason": f"{type(e).__name__}: {e}"})
            continue
        if op not in REALIZED:
            plan["unrealized"].append({"operator": op,
                                       "reason": f"not implemented in apply.py (known set: {sorted(REALIZED)})"})
            continue
        try:
            if op == "join_feature" and _key_class_of(params.get("source", "")) == "lookup":
                # Non-panel join: a static attribute table resolved by a code column. Routing
                # by the source's DECLARED key class is what lets the source layer grow past
                # country x time -- the dispatcher used to assume every join was a country-year
                # as-of merge, which is why the layer only ever fired on panel competitions
                # (2026-08-03 source-layer extension).
                tr, te = _apply_lookup_join(tr, te, params, plan)
                continue
            if op in ("join_feature", "ratio_target", "log_offset"):
                src = params["source"]
                out_col = (params.get("as") or ["cov"])[0] if op == "join_feature" else "cov_level"
                frame, mapping, meta = _fetch_covariate(src, *panel_keys())
                lag = (params.get("join") or {}).get("lag", 0)
                for name, df in (("train", tr), ("test", te)):
                    df["_year"] = pd.to_datetime(df[date_col]).dt.year
                joined = {}
                for name, df in (("train", tr), ("test", te)):
                    out, rep = merge_year_safe(df, frame, df_country=country_col,
                                               df_year="_year", out_col=out_col,
                                               mapping=mapping, lag=lag, meta=meta)
                    if rep["rows_matched"] < rep["rows"]:
                        # carry forward is required when the test period post-dates coverage
                        if not params.get("carry_forward"):
                            raise ValueError(
                                f"{name}: {rep['rows_unmatched']} rows unmatched and "
                                f"carry_forward is not set")
                        # 07-30 (readiness-audit finding): this used to fill every unmatched row
                        # with the panel's single globally-last year, applied to train and test
                        # alike. merge_year_safe's own backward asof already leakage-checks every
                        # MATCHED row (used year <= row's effective year); a row is unmatched only
                        # when NO ext year <= its own effective year exists for that country, which
                        # means the covariate's coverage starts after that row's need. Filling such
                        # a row with the panel's latest year is therefore filling an EARLIER row
                        # with LATER data — a training-set leak, not a benign forward-carry. A row
                        # whose effective year is AT OR AFTER the country's last covered year (the
                        # legitimate "test post-dates coverage" case) fills safely because the
                        # source year used is still <= the row's own year.
                        eff_year = out["_year"].astype("int64") - lag
                        iso = out[country_col].map(mapping)
                        last_year_by_iso = frame.groupby("iso3")["year"].max()
                        last_val_by_iso = frame.sort_values("year").groupby("iso3")["value"].last()
                        fill_source_year = iso.map(last_year_by_iso)
                        needs_fill = out[out_col].isna()
                        safe = needs_fill & fill_source_year.notna() & (fill_source_year <= eff_year)
                        unsafe = needs_fill & ~safe
                        out.loc[safe, out_col] = iso[safe].map(last_val_by_iso)
                        plan.setdefault("carry_forward_applied", []).append(
                            {"name": name, "rows_filled": int(safe.sum()),
                             "rows_refused_future_data": int(unsafe.sum()),
                             "note": "refused rows would have been filled with a source year "
                                     "later than the row's own effective year; left as NaN "
                                     "instead of leaking future information into that row"})
                        if int(unsafe.sum()):
                            plan["unrealized"].append({
                                "operator": op, "params_subset": "carry_forward",
                                "reason": f"{name}: {int(unsafe.sum())} row(s) would need a "
                                         f"source year later than their own effective year to "
                                         f"fill — refused rather than leaking future data; "
                                         f"left as NaN"})
                    joined[name] = out.drop(columns=["_year"], errors="ignore")
                tr, te = joined["train"], joined["test"]
                plan["sources"][src] = meta.get("snapshot")
                if op in ("ratio_target", "log_offset"):
                    diag = proportionality_check(tr, frame, mapping, country_col=country_col,
                                                 date_col=date_col, target_col=target_col)
                    vol = covariate_volatility_check(frame, mapping, panel_keys()[0])
                    diag["covariate_volatility"] = vol
                    plan.setdefault("diagnostics", {})[op] = diag
                    gate_failures = []
                    if meta.get("indicator") in CURRENT_PRICE_SUFFIX:
                        gate_failures.append("nominal covariate series")
                        plan["unrealized"].append({
                            "operator": op, "params_subset": "nominal covariate series",
                            "reason": (f"{meta['indicator']} is a current-price series; a ratio "
                                       "target multiplies FX/inflation shocks into predictions. "
                                       "Use gdp_per_capita_const or gdp_per_capita_ppp instead "
                                       "(measured: 14.6pp vs 3.0pp cross-country spread)")})
                    if vol.get("verdict") != "stable":
                        gate_failures.append(f"covariate stability ({vol.get('verdict')})")
                        plan["unrealized"].append({
                            "operator": op, "params_subset": "covariate stability",
                            "reason": (f"covariate moves up to {vol['worst_spread_pp']}pp "
                                       f"differently across groups (threshold "
                                       f"{vol['threshold_pp']}pp) — the level it carries is "
                                       "contaminated")})
                    if diag["violating_years"] or diag.get("unmeasured_years"):
                        gate_failures.append("proportionality")
                        plan["unrealized"].append({
                            "operator": op, "params_subset": "unconditional application",
                            "reason": ("precondition failed: target is not covariate-proportional in "
                                       f"{sorted(diag['violating_years'])} (dispersion "
                                       f"{diag['violating_years']}); the ratio target inherits the "
                                       "broken relationship unless those periods are down-weighted "
                                       "or excluded (see sample_weight)")})
                    if gate_failures and not params.get("force"):
                        # Until 2026-08-03 a failed precondition only added a NOTE: the target
                        # transform was installed anyway and the operator was simultaneously
                        # booked realized, so a ledger reader saw both "unrealized: precondition
                        # failed" and a trained ratio arm. Refuse now; an author who has read the
                        # diagnostics can still pass params["force"]=true, which is recorded.
                        plan["unrealized"].append({
                            "operator": op, "params_subset": "REFUSED",
                            "reason": f"preconditions failed ({'; '.join(gate_failures)}); "
                                      "operator not applied. Pass params.force=true to apply "
                                      "anyway, which records the override in this ledger."})
                        for df in (tr, te):
                            df.drop(columns=[out_col], inplace=True, errors="ignore")
                        continue
                    if gate_failures:
                        plan.setdefault("forced_overrides", []).append(
                            {"operator": op, "failed_gates": gate_failures})
                if op == "join_feature":
                    plan["added_columns"].append(out_col)
                else:
                    plan["target_transform"] = {
                        "kind": "ratio_log" if params.get("space", "log") == "log" else "ratio_linear",
                        "covariate_col": out_col}
                plan["realized"].append({"operator": op, "column": out_col})

            elif op == "trend_term":
                unit, deg = params.get("unit", "year"), int(params.get("degree", 1))
                if unit != "year":
                    raise ValueError("only unit='year' is implemented")
                base = pd.to_datetime(tr[date_col]).dt.year
                centre = float(base.mean()) if params.get("centered", True) else 0.0
                for df in (tr, te):
                    y = pd.to_datetime(df[date_col]).dt.year - centre
                    for d in range(1, deg + 1):
                        col = "year_c" if d == 1 else f"year_c{d}"
                        df[col] = y ** d
                cols = ["year_c"] + [f"year_c{d}" for d in range(2, deg + 1)]
                plan["added_columns"] += cols
                plan["realized"].append({"operator": op, "columns": cols, "centre": centre})

            elif op == "sample_weight":
                # predicate is restricted to "year in [...]" so it stays declarative and safe
                pred = str(params.get("predicate", ""))
                m = re.match(r"\s*year\s+in\s*\[([0-9,\s]+)\]\s*$", pred)
                if not m:
                    raise ValueError(f"only 'year in [..]' predicates are supported, got {pred!r}")
                years_sel = {int(x) for x in m.group(1).split(",") if x.strip()}
                w = float(params.get("weight", 0.5))
                col = "row_weight"
                for name in ("train", "test"):
                    df = tr if name == "train" else te
                    yr = pd.to_datetime(df[date_col]).dt.year
                    if col in df.columns:
                        df[col] = df[col] * yr.isin(years_sel).map({True: w, False: 1.0})
                    else:
                        df[col] = yr.isin(years_sel).map({True: w, False: 1.0}).astype(float)
                    if params.get("also_flag"):
                        df["regime_flag"] = yr.isin(years_sel).astype(int)
                if params.get("also_flag") and "regime_flag" not in plan["added_columns"]:
                    plan["added_columns"].append("regime_flag")
                plan["weight_column"] = col
                plan["realized"].append({"operator": op, "weight_column": col,
                                         "years": sorted(years_sel), "weight": w})

            elif op == "flag_feature":
                as_cols = params.get("as") or ["is_holiday"]
                src = params.get("source", "holidays")
                if src not in ("holidays", "calendar"):
                    # This branch realizes exactly one thing: a same-day national holiday
                    # flag. It used to ignore params["source"] entirely, so a dossier naming
                    # any other date-class source got holiday flags back under the requested
                    # column name, with the ledger citing 'holidays' as the origin -- a
                    # feature that is not what was asked for and does not say so
                    # (2026-08-04 adversarial verification).
                    raise ValueError(
                        f"flag_feature realizes only the holiday calendar; source {src!r} has "
                        f"no implementation. Name it in the dossier's not_recorded field, or "
                        f"add a fetcher and a branch here -- do not let it silently become "
                        f"holiday flags.")
                _yrs, _ctys = panel_keys()
                hol, meta = fetch_holidays(_ctys, _yrs)
                # Countries the calendar could not supply. Without this the operator books
                # itself realized while those rows carry a systematically-zero flag, which is
                # the exact defect admit_source.check_coverage's docstring cites -- and it was
                # still live in this path until 2026-08-04.
                uncovered = sorted(set(meta.get("unmatched") or []))
                reports = {}
                for name in ("train", "test"):
                    df = tr if name == "train" else te
                    out, rep = merge_holiday_flags(df, hol, df_country=country_col,
                                                   df_date=date_col, out_col=as_cols[0],
                                                   meta=meta)
                    reports[name] = rep
                    uncovered = sorted(set(uncovered) | set(rep.get("countries_without_calendar") or []))
                    if name == "train":
                        tr = out
                    else:
                        te = out
                if uncovered:
                    plan["unrealized"].append({
                        "operator": op, "params_subset": f"{as_cols[0]}: uncovered entities",
                        "reason": f"no calendar for {uncovered}; those rows carry a "
                                  f"systematically ZERO flag, which is not the same as 'no "
                                  f"holidays'. Reported rather than filled."})
                if not any(r.get("holiday_rows") for r in reports.values()):
                    # Zero flagged rows across the whole panel is not a plausible calendar
                    # result; it is the signature of a key mismatch (dtype, timezone, name
                    # spelling) that would otherwise ship as a constant column.
                    raise ValueError(
                        f"the holiday calendar flagged 0 of {len(tr) + len(te)} rows across "
                        f"every entity. A constant-zero flag is a join failure, not a "
                        f"feature; check the date dtypes and the country name spellings "
                        f"(calendar covers {sorted(set(hol['country'])) if 'country' in hol else 'n/a'}).")
                plan["added_columns"].append(as_cols[0])
                plan["sources"]["holidays"] = meta.get("snapshot")
                extra = [k for k in ("window", "per_name") if params.get(k)]
                if extra:
                    plan["unrealized"].append({"operator": op, "params_subset": extra,
                                               "reason": "only a boolean same-day flag is implemented"})
                plan["realized"].append({"operator": op, "column": as_cols[0],
                                         "holiday_rows": {k: r.get("holiday_rows")
                                                          for k, r in reports.items()},
                                         "entities_without_calendar": uncovered})
        except Exception as e:  # noqa: BLE001 — a failed operator is recorded, never hidden
            plan["unrealized"].append({"operator": op, "reason": f"{type(e).__name__}: {e}"})

    plan["realized_count"] = len(plan["realized"])          # data-layer only, by construction
    plan["emitted_config_count"] = len(plan["emitted_config"])
    plan["advisory_count"] = len(plan["advisory"])
    if ledger_path:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ledger_path).write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return tr, te, plan
