"""Generic Stage 1->5 ablation ladder for an i.i.d. tabular Kaggle competition.

Faithfully reproduces the project's 5-stage ladder (docs/pipeline_stages_detail.md) from
config.yaml + data, with NO per-competition hand scaffolding, so it can be queued over
several competitions. Iron rules honoured: ALL stages share one fold-index set and seed=42
(so scores are paired-comparable); every decision reads the POST-PROCESSED OOF score; OOF is
the only currency.

  Stage 1  no-skill baseline    : 3 GBDT in one family (lgb/xgb/lgb_v2), no tune, coarse 0.1
                                  weight grid, no determinism/cache.
  Stage 2  + kaggle-agent skill : architecture-diverse lgb/xgb/CAT, early stopping,
                                  determinism, 0.05 grid, OOF cache.
  Stage 3  + linear iteration   : Optuna-tune strongest solo + seed-bag, ADD to pool
                                  (not replace), re-blend.
  Stage 4  + tree search        : harness_v3 candidate tree + backtracking + budget/phase
                                  machine + mega-blend, generic bias-dominated variants.
  Stage 5  + idea injection     : harness_v2.suggest_priors -> concrete candidate variants
                                  (target/count encoding, drop-sets), continue the search.

Usage:  uv run python3 competitions/run_stage_ladder.py <competition-name> [--s4 24] [--s5 12]
Env LADDER_SMOKE=1 shrinks node budgets for a fast end-to-end smoke test.
"""
import argparse, importlib.util, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
TS = REPO/"tree_search"; sys.path.insert(0, str(TS))
import harness as hv1        # noqa: E402
import harness_v2 as hv2     # noqa: E402
import harness_v3 as hv3     # noqa: E402
_els = importlib.util.spec_from_file_location("el", REPO/".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
el = importlib.util.module_from_spec(_els); _els.loader.exec_module(el)

def _load_ot(cache, mid):
    d = np.load(os.path.join(cache, f"solo_{mid}.npz"))
    return d["oof"], d["test"]

SEED, NF = 42, 5
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# Kaggle-reality metric overrides (config.yaml is stale for some comps)
METRIC_OVERRIDE = {"house-prices-advanced-regression-techniques": "rmsle",
                   "home-data-for-ml-course": "mae", "spaceship-titanic": "accuracy"}

# ------------------------------------------------------------------ metric
def metric_spec(name, ptype):
    m = (name or "").lower().replace("-", "_")
    if m in ("rmsle", "msle", "root_mean_squared_log_error"):
        return dict(kind="reg", logt=True, higher=False, fn=lambda y,p: mean_squared_error(y,p)**0.5, name="rmsle")
    if m == "mae":
        return dict(kind="reg", logt=True, higher=False, fn=mean_absolute_error, name="mae")
    if m in ("rmse", "root_mean_squared_error"):
        return dict(kind="reg", logt=False, higher=False, fn=lambda y,p: mean_squared_error(y,p)**0.5, name="rmse")
    if m == "accuracy":
        return dict(kind="clf", logt=False, higher=True, fn=accuracy_score, name="accuracy")
    # fallback
    if ptype and "reg" in ptype: return metric_spec("rmse", ptype)
    return metric_spec("accuracy", ptype)

# ------------------------------------------------------------------ data + generic features
def load(comp):
    cdir = REPO/"competitions"/comp
    cfg = yaml.safe_load(open(cdir/"config.yaml"))
    target = cfg.get("target_column"); idc = cfg.get("id_column", "id")
    ptype = cfg.get("problem_type") or cfg.get("task_type") or ""
    metric = METRIC_OVERRIDE.get(comp) or cfg.get("evaluation_metric")
    tr = pd.read_csv(cdir/"data/train.csv"); te = pd.read_csv(cdir/"data/test.csv")
    return cdir, cfg, tr, te, target, idc, ptype, metric_spec(metric, ptype)

def build_features(comp, tr, te, target, idc):
    """Use a bespoke per-competition feature module if present
    (competitions/<comp>/scripts/features_ladder.py with build(tr,te,target,idc) ->
    (Xtr_float32, Xte_float32, y_raw)); otherwise fall back to the generic builder."""
    fpath = REPO/"competitions"/comp/"scripts"/"features_ladder.py"
    if fpath.exists():
        _fs = importlib.util.spec_from_file_location("features_ladder", fpath)
        _fm = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(_fm)
        Xtr, Xte, y_raw = _fm.build(tr, te, target, idc)
        return Xtr.astype(np.float32), Xte.astype(np.float32), y_raw
    return _generic_features(tr, te, target, idc)

def _generic_features(tr, te, target, idc):
    """Generic deterministic feature builder: numeric passthrough + missing indicators +
    ordinal/label-encode object cols (fit on train+test union) + a few pairwise products."""
    y_raw = tr[target].to_numpy()
    drop = {target, idc}
    feat_cols = [c for c in tr.columns if c not in drop]
    Xtr = tr[feat_cols].copy(); Xte = te[feat_cols].copy()
    num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(Xtr[c])]
    obj = [c for c in feat_cols if c not in num]
    # drop near-unique string columns (names/ids) — generic builder would just add noise
    junk = [c for c in obj if pd.concat([Xtr[c], Xte[c]]).astype(str).nunique() > 0.5*len(Xtr)]
    if junk:
        Xtr = Xtr.drop(columns=junk); Xte = Xte.drop(columns=junk); obj = [c for c in obj if c not in junk]
    # missing indicators for numerics with NaN
    for c in num:
        if Xtr[c].isna().any() or Xte[c].isna().any():
            Xtr[f"{c}_isna"] = Xtr[c].isna().astype(np.int8); Xte[f"{c}_isna"] = Xte[c].isna().astype(np.int8)
    # label-encode objects on union (NaN -> its own "__nan__" category)
    for c in obj:
        u = pd.concat([Xtr[c], Xte[c]], ignore_index=True).fillna("__nan__").astype(str)
        cats = {v: i for i, v in enumerate(sorted(u.unique()))}
        Xtr[c] = Xtr[c].fillna("__nan__").astype(str).map(cats).astype(np.int32)
        Xte[c] = Xte[c].fillna("__nan__").astype(str).map(cats).astype(np.int32)
    # median-impute numerics (train medians)
    for c in num:
        md = Xtr[c].median()
        Xtr[c] = Xtr[c].fillna(md); Xte[c] = Xte[c].fillna(md)
    # top pairwise products of the 4 highest-variance numerics (deterministic)
    topn = sorted(num, key=lambda c: -Xtr[c].var())[:4]
    for i in range(len(topn)):
        for j in range(i+1, len(topn)):
            a, b = topn[i], topn[j]
            Xtr[f"{a}_x_{b}"] = Xtr[a]*Xtr[b]; Xte[f"{a}_x_{b}"] = Xte[a]*Xte[b]
    return Xtr.astype(np.float32), Xte.astype(np.float32), y_raw

# ------------------------------------------------------------------ models
def make_folds(y, kind):
    if kind == "clf":
        return list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    return list(KFold(NF, shuffle=True, random_state=SEED).split(y))

def _fit_predict(model, params, Xtr, ytr, Xva, Xte, kind, det, es):
    import lightgbm as lgb, xgboost as xgb
    if model in ("lgb", "lgb_v2"):
        p = dict(learning_rate=params.get("learning_rate", 0.05), num_leaves=params.get("num_leaves", 63),
                 max_depth=params.get("max_depth", -1), min_child_samples=params.get("min_child_samples", 30),
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                 reg_alpha=params.get("reg_alpha", 0.1), reg_lambda=params.get("reg_lambda", 0.1),
                 n_estimators=params.get("n_estimators", 3000), n_jobs=8, random_state=params.get("random_state", SEED), verbose=-1)
        if det: p.update(deterministic=True, force_row_wise=True)
        Model = lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor
        m = Model(**p); m.fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:,1] if kind=="clf" else m.predict(Xva)
        pt = m.predict_proba(Xte)[:,1] if kind=="clf" else m.predict(Xte)
        return pv, pt
    if model == "xgb":
        p = dict(learning_rate=params.get("learning_rate", 0.05), max_depth=params.get("max_depth", 7),
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=params.get("reg_alpha", 0.1),
                 reg_lambda=params.get("reg_lambda", 1.0), n_estimators=params.get("n_estimators", 3000),
                 tree_method="hist", n_jobs=8, random_state=params.get("random_state", SEED))
        Model = xgb.XGBClassifier if kind=="clf" else xgb.XGBRegressor
        m = Model(**p); m.fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:,1] if kind=="clf" else m.predict(Xva)
        pt = m.predict_proba(Xte)[:,1] if kind=="clf" else m.predict(Xte)
        return pv, pt
    if model == "cat":
        from catboost import CatBoostClassifier, CatBoostRegressor
        p = dict(learning_rate=params.get("learning_rate",0.05), depth=params.get("depth",8),
                 l2_leaf_reg=params.get("l2_leaf_reg",3.0), iterations=params.get("iterations",3000),
                 random_seed=params.get("random_seed",SEED), thread_count=8, allow_writing_files=False, verbose=0)
        Model = CatBoostClassifier if kind=="clf" else CatBoostRegressor
        m = Model(**p); m.fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:,1] if kind=="clf" else m.predict(Xva).ravel()
        pt = m.predict_proba(Xte)[:,1] if kind=="clf" else m.predict(Xte).ravel()
        return pv, pt
    raise ValueError(model)

_yva = None
def oof_solo(model, params, X, y_model, Xte, folds, kind, det=False, es=False):
    global _yva
    oof = np.zeros(len(X)); test = np.zeros(len(Xte))
    Xv = X.values; Xtv = Xte.values
    for tri, vai in folds:
        _yva = y_model[vai]
        pv, pt = _fit_predict(model, params, Xv[tri], y_model[tri], Xv[vai], Xtv, kind, det, es)
        oof[vai] = pv; test += pt/len(folds)
    return oof, test

def score_oof(spec, y_raw, oof):
    """oof is in MODEL space (log1p for logt regression, proba for clf). Score in real space."""
    if spec["kind"] == "clf":
        return spec["fn"](y_raw, (oof > 0.5).astype(int))
    pred = np.expm1(oof) if spec["logt"] else oof
    if spec["name"] == "rmsle":
        return mean_squared_error(np.log1p(np.clip(y_raw,0,None)), np.log1p(np.clip(pred,0,None)))**0.5
    return spec["fn"](y_raw, pred)

def y_model_of(spec, y_raw):
    if spec["kind"] == "clf": return y_raw.astype(int)
    return np.log1p(np.clip(y_raw,0,None)) if spec["logt"] else y_raw.astype(float)

# ------------------------------------------------------------------ blend weight search
def blend_search(spec, y_raw, oofs, step):
    from itertools import product
    n = len(oofs); O = np.stack(oofs)
    best = (None, 1e18*(1 if not spec["higher"] else 1))
    best_s = -1e18 if spec["higher"] else 1e18
    grid = np.arange(0, 1+1e-9, step)
    for w in product(grid, repeat=n):
        if abs(sum(w)-1) > 1e-6: continue
        s = score_oof(spec, y_raw, np.tensordot(np.array(w), O, 1))
        if (spec["higher"] and s > best_s) or (not spec["higher"] and s < best_s):
            best_s = s; best = np.array(w)
    return best, best_s

def dirichlet_search(spec, y_raw, oofs, n_draws=8000, seed=SEED):
    O = np.stack(oofs); rng = np.random.default_rng(seed); M = len(oofs)
    cand = np.vstack([rng.dirichlet(np.ones(M), size=n_draws), np.eye(M)])
    best_w, best_s = None, (-1e18 if spec["higher"] else 1e18)
    for w in cand:
        s = score_oof(spec, y_raw, np.tensordot(w, O, 1))
        if (spec["higher"] and s > best_s) or (not spec["higher"] and s < best_s): best_s, best_w = s, w
    return best_w, best_s

# ------------------------------------------------------------------ submission
def write_sub(cdir, cfg, spec, idc, te, test_pred_model, stage, score):
    target = cfg.get("target_column")
    pred = (test_pred_model > 0.5).astype(int) if spec["kind"]=="clf" else (np.expm1(test_pred_model) if spec["logt"] else test_pred_model)
    if spec["kind"]=="clf" and te_target_is_bool(cdir):
        pred = pred.astype(bool)
    sub = pd.DataFrame({idc: te[idc], target: pred})
    out = cdir/"submissions"/f"ladder_stage{stage}_{spec['name']}_{score:.5f}.csv"
    out.parent.mkdir(exist_ok=True); sub.to_csv(out, index=False)
    return out

def te_target_is_bool(cdir):
    ss = pd.read_csv(cdir/"data/sample_submission.csv")
    col = ss.columns[-1]
    return ss[col].dtype == bool or set(ss[col].dropna().unique().tolist()) <= {True, False}

# ================================================================== main ladder
def run(comp, s4_nodes, s5_nodes):
    cdir, cfg, tr, te, target, idc, ptype, spec = load(comp)
    X, Xte, y_raw = build_features(comp, tr, te, target, idc)
    kind = spec["kind"]; ym = y_model_of(spec, y_raw)
    folds = make_folds(y_raw.astype(int) if kind=="clf" else y_raw, kind)
    log(f"{comp}: X={X.shape} metric={spec['name']} higher={spec['higher']} logt={spec['logt']} kind={kind}")
    results = {}
    cache = str(TS/f"cache_ladder_{comp}")
    os.makedirs(cache, exist_ok=True)

    def submit_and_log(stage, label, oof_pool_names, test_pred, sc, weights=None, priors_used=None):
        out = write_sub(cdir, cfg, spec, idc, te, test_pred, stage, sc)
        el.log_experiment_v2(str(cdir), model=f"[LADDER s{stage}] {label}", metric=spec["name"],
            direction="maximize" if spec["higher"] else "minimize", score=round(float(sc),6),
            cv=dict(scheme="StratifiedKFold" if kind=="clf" else "KFold", n_splits=NF, seed=SEED),
            submission=out.name, notes=f"ladder stage {stage}; weights={weights}; priors={priors_used}")
        results[f"stage{stage}"] = dict(score=round(float(sc),6), sub=out.name, weights=weights)
        log(f"  STAGE {stage} [{label}] {spec['name']}={sc:.6f} -> {out.name}")

    # ---- Stage 1: no-skill baseline (single family, coarse grid) ----
    s1 = {}
    for name, mdl, pr in [("lgb", "lgb", {}), ("xgb", "xgb", {}),
                          ("lgb_v2", "lgb_v2", dict(n_estimators=5000, learning_rate=0.02))]:
        oof, test = oof_solo(mdl, pr, X, ym, Xte, folds, kind, det=False, es=False)
        s1[name] = (oof, test, score_oof(spec, y_raw, oof))
    w1, sc1 = blend_search(spec, y_raw, [s1[n][0] for n in s1], step=0.1)
    tb1 = np.tensordot(w1, np.stack([s1[n][1] for n in s1]), 1)
    submit_and_log(1, "3xGBDT single-family, 0.1 grid", list(s1), tb1, sc1, dict(zip(s1, w1.round(2))))

    # ---- Stage 2: skill (diverse lgb/xgb/cat, determinism, 0.05 grid, cache) ----
    pool = {}   # name -> (oof, test, solo_score)
    for name, mdl in [("lgb", "lgb"), ("xgb", "xgb"), ("cat", "cat")]:
        oof, test = oof_solo(mdl, {}, X, ym, Xte, folds, kind, det=(mdl=="lgb"), es=False)
        pool[name] = (oof, test, score_oof(spec, y_raw, oof))
    w2, sc2 = blend_search(spec, y_raw, [pool[n][0] for n in pool], step=0.05)
    tb2 = np.tensordot(w2, np.stack([pool[n][1] for n in pool]), 1)
    submit_and_log(2, "diverse lgb/xgb/cat, 0.05 grid", list(pool), tb2, sc2, dict(zip(pool, w2.round(2))))

    # ---- Stage 3: linear iteration (Optuna-tune strongest + seed-bag, add to pool) ----
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    strongest = max(pool, key=lambda n: pool[n][2] if spec["higher"] else -pool[n][2])
    tri0, vai0 = folds[0]
    def obj(t):
        if strongest in ("lgb",):
            pr = dict(learning_rate=t.suggest_float("lr",0.01,0.06,log=True), num_leaves=t.suggest_int("nl",16,128),
                      min_child_samples=t.suggest_int("mcs",20,200), reg_lambda=t.suggest_float("l2",0.1,8,log=True),
                      reg_alpha=t.suggest_float("l1",1e-3,5,log=True), n_estimators=2000)
        elif strongest == "xgb":
            pr = dict(learning_rate=t.suggest_float("lr",0.01,0.06,log=True), max_depth=t.suggest_int("md",3,10),
                      reg_lambda=t.suggest_float("l2",0.1,8,log=True), reg_alpha=t.suggest_float("l1",1e-3,5,log=True), n_estimators=2000)
        else:
            pr = dict(learning_rate=t.suggest_float("lr",0.01,0.06,log=True), depth=t.suggest_int("d",4,10),
                      l2_leaf_reg=t.suggest_float("l2",1,20,log=True), iterations=2000)
        pv, _ = _fit_predict(strongest, pr, X.values[tri0], ym[tri0], X.values[vai0], X.values[:1], kind, False, False)
        return score_oof(spec, y_raw[vai0], pv)
    st = optuna.create_study(direction="maximize" if spec["higher"] else "minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=6 if os.environ.get("LADDER_SMOKE") else 30, show_progress_bar=False)
    bp = st.best_params
    remap = {"lr":"learning_rate","nl":"num_leaves","mcs":"min_child_samples","l2":{"lgb":"reg_lambda","xgb":"reg_lambda","cat":"l2_leaf_reg"}[strongest],
             "l1":"reg_alpha","md":"max_depth","d":"depth"}
    tp = {};
    for k,v in bp.items():
        kk = remap[k]; tp[kk] = v
    tp["n_estimators"] = 2500; tp["iterations"] = 2500
    for sd, nm in [(SEED, f"{strongest}_tuned"), (2024, f"{strongest}_tuned_s2024")]:
        pr = dict(tp); pr["random_state"]=sd; pr["random_seed"]=sd
        oof, test = oof_solo(strongest, pr, X, ym, Xte, folds, kind, det=(strongest=="lgb"), es=False)
        pool[nm] = (oof, test, score_oof(spec, y_raw, oof))
    names3 = list(pool)
    w3, sc3 = dirichlet_search(spec, y_raw, [pool[n][0] for n in names3])
    tb3 = np.tensordot(w3, np.stack([pool[n][1] for n in names3]), 1)
    submit_and_log(3, f"Optuna-tuned {strongest} + seed-bag, dirichlet reblend", names3, tb3, sc3,
                   {n:round(float(x),3) for n,x in zip(names3,w3) if x>1e-3})

    # ---- Stage 4: tree search (harness_v3 budget/phase machine, generic variants) ----
    # sign convention: harness assumes LOWER is better -> feed -score for maximize metrics.
    sgn = -1.0 if spec["higher"] else 1.0
    tree = hv1.new_tree(comp)
    # seed the tree with stage-3 solo pool as evaluated nodes (cache their OOF)
    id_of = {}
    root_id = hv1.add_root(tree, "[LADDER] stage-3 pool root", {"kind":"root"}, 0.0, "evaluated", 0.0)
    for nm in names3:
        oof, test, sc = pool[nm]
        nid = hv1.next_id(tree)
        cfgn = {"kind":"solo","model":nm.split("_")[0],"name":nm}
        hv3.add_node(tree, root_id, f"[LADDER] seed {nm}", cfgn, sgn*sc, "evaluated", 0.0)
        hv2.cache_oof(cache, nid, oof, test=test); id_of[nid] = nm
    def metric_fn(y_true_ignored, blended):  # hv2.eval_blend passes (y, blendedOOF); we score in real space
        return sgn*score_oof(spec, y_raw, blended)
    # generic forward search
    budget = s4_nodes
    NP = names3[:]
    def solo_ids(): return [n["id"] for n in tree["nodes"] if n.get("status")=="evaluated" and n["config"].get("kind")=="solo"]
    for i in range(budget):
        ids = solo_ids()
        if i % 2 == 1 and len(ids) >= 2:                       # full-pool mega-blend
            members = sorted(ids)
            oofs = [_load_ot(cache, mid)[0] for mid in members]
            tests = [_load_ot(cache, mid)[1] for mid in members]
            w, sc = dirichlet_search(spec, y_raw, oofs)
            nid = hv1.next_id(tree)
            hv3.add_node(tree, members[0], f"[LADDER-S4] mega-blend {len(members)}", {"kind":"blend","members":members}, sgn*sc, "evaluated", 0.0)
            hv2.cache_oof(cache, nid, np.tensordot(w,np.stack(oofs),1), test=np.tensordot(w,np.stack(tests),1))
        else:                                                  # shallow-regularized decorrelating solo variant
            fam = ["lgb","xgb","cat"][(i//2)%3]
            base = dict(learning_rate=0.03, num_leaves=[15,31,63][i%3], max_depth=[3,4,6][i%3],
                        min_child_samples=[60,120,200][i%3], reg_lambda=[4,10,20][i%3], depth=[4,5,6][i%3],
                        l2_leaf_reg=[8,12,20][i%3], n_estimators=1500, iterations=1500, random_state=1000+i, random_seed=1000+i)
            oof, test = oof_solo(fam, base, X, ym, Xte, folds, kind, det=(fam=="lgb"), es=False)
            sc = score_oof(spec, y_raw, oof); nid = hv1.next_id(tree)
            hv3.add_node(tree, root_id, f"[LADDER-S4] shallow-reg {fam} v{i}", {"kind":"solo","model":fam,"name":f"s4_{fam}_{i}"}, sgn*sc, "evaluated", 0.0)
            hv2.cache_oof(cache, nid, oof, test=test)
    ev4 = [n for n in tree["nodes"] if n.get("status")=="evaluated" and n["config"].get("kind")!="root"]
    champ4 = min(ev4, key=lambda n: n["score"]); sc4 = sgn*champ4["score"]
    t4 = _load_ot(cache, champ4["id"])[1]
    submit_and_log(4, f"tree search {budget} nodes; champ #{champ4['id']} {champ4['config'].get('kind')}", None, t4, sc4)

    # ---- Stage 5: + idea injection (suggest_priors -> concrete candidate variants) ----
    try:
        priors = hv2.suggest_priors(dict(name=comp, metric=spec["name"], problem_type=ptype),
                                    experience_path=str(REPO/"knowledge/experience.md"), max_items=8)
    except Exception as e:
        priors = []; log(f"  suggest_priors failed ({e}); using built-in idea variants")
    ideas = [("target_encode", "high-card target-encoding member"),
             ("count_encode", "count-encoding member"),
             ("drop_weak", "drop lowest-variance features member")]
    for j,(key,desc) in enumerate(ideas[:max(1,s5_nodes//2)]):
        Xi, Xti = X.copy(), Xte.copy()
        if key=="drop_weak":
            weak = sorted(X.columns, key=lambda c: X[c].var())[:max(1,X.shape[1]//5)]
            Xi = X.drop(columns=weak); Xti = Xte.drop(columns=weak)
        oof, test = oof_solo("lgb", dict(num_leaves=48, reg_lambda=3.0), Xi, ym, Xti, folds, kind, det=True, es=False)
        sc = score_oof(spec, y_raw, oof); nid = hv1.next_id(tree)
        hv3.add_node(tree, root_id, f"[LADDER-S5 idea] {desc}", {"kind":"solo","model":"lgb","name":f"s5_idea_{key}"}, sgn*sc, "evaluated", 0.0)
        hv2.cache_oof(cache, nid, oof, test=test)
    # final mega-blend incl. idea members
    ids5 = [n["id"] for n in tree["nodes"] if n.get("status")=="evaluated" and n["config"].get("kind")=="solo"]
    oofs = [_load_ot(cache, mid)[0] for mid in ids5]; tests=[_load_ot(cache, mid)[1] for mid in ids5]
    w5, sc5 = dirichlet_search(spec, y_raw, oofs)
    ev5 = [n for n in tree["nodes"] if n.get("status")=="evaluated" and n["config"].get("kind")!="root"]
    best_prev = sgn*min(ev5, key=lambda n:n["score"])["score"]
    if (spec["higher"] and sc5 > best_prev) or (not spec["higher"] and sc5 < best_prev):
        t5 = np.tensordot(w5, np.stack(tests), 1); best5 = sc5
    else:
        t5 = t4; best5 = sc4
    submit_and_log(5, f"idea-injected pool, final blend ({len(ids5)} members)", None, t5, best5,
                   priors_used=[p.get("title","?") if isinstance(p,dict) else str(p) for p in priors][:4] if priors else None)
    # ladder summary
    def better(a,b): return (a>b) if spec["higher"] else (a<b)
    order = [results.get(f"stage{k}",{}).get("score") for k in range(1,6)]
    json.dump(dict(comp=comp, metric=spec["name"], higher=spec["higher"], stages=results, ladder=order),
              open(cdir/"ladder_results.json","w"), indent=2)
    log(f"{comp} LADDER DONE: " + " -> ".join(f"s{k}={order[k-1]}" for k in range(1,6)))
    return dict(comp=comp, metric=spec["name"], stages=order)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comp"); ap.add_argument("--s4", type=int, default=24); ap.add_argument("--s5", type=int, default=12)
    a = ap.parse_args()
    if os.environ.get("LADDER_SMOKE"): a.s4, a.s5 = 4, 4
    t0=time.time()
    try:
        r = run(a.comp, a.s4, a.s5)
        print("RESULT " + json.dumps(r));
    except Exception as e:
        import traceback; traceback.print_exc()
        print("RESULT " + json.dumps(dict(comp=a.comp, error=str(e))))
    log(f"{a.comp} total {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
