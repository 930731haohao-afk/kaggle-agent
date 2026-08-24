"""
Deterministic fact extractor for SPECIAL / COMPLEX (non-GBDT) competition lanes.

Any lane whose pipeline is not the GBDT tree-search (deep-learning fallback lanes of any
modality: vision, NLP, audio, video, multimodal, sequence models, ...) has records the
mlspec skill's collect.py does not understand. This extractor discovers records
GENERICALLY in any competition workspace — nothing is hardcoded to specific competition
ids or artifact filenames — and produces the same kind of facts.json contract:

    facts.json is the ONLY source of numbers for the ML-spec report — the LLM never
    reads the raw records directly when writing, and NOTHING here is guessed:
    every value is copied verbatim from a record file; absent records are listed
    under "missing" so the report writes "not recorded".

Usage (from anywhere; paths may be absolute or relative to the repo root):
    VIRTUAL_ENV= uv run python benchmark_infra/collect_dl_facts.py <competition-dir> \
        [--grade-record PATH] [--out PATH] [--force]

Decision rule vs collect.py: if the workspace contains GBDT tree-search records
(experiments_tree*.json), it is a GBDT-pipeline lane and the canonical extractor is the
skill's collect.py — this tool then exits with a clear message instead of overwriting
that pipeline's facts.json (override with --force, ideally with --out).

Discovery (all glob-based, applied to whatever exists):
  config.yaml|yml            competition metadata (metric name / direction / rules)
  experiments*.json          v2 entries passed through verbatim; primary file is
                             experiments.json, other matches listed by name
  STATUS.md                  parsed into {heading: text} sections (narrative context)
  headless*.log              carried verbatim (prose record)
  cv_result*.json, res_*.json, *_results.json, discovery*.json, dossier.json,
  eda_summary*.json, scripts/*_cv.json, scripts/*decision*.json, scripts/*_results.json
                             structured records, copied verbatim
  scripts/*.py               deterministic greps only: SEED constant, loss / optimizer /
                             LR-scheduler identifiers actually referenced in the code
  submission*.csv, submissions/*.csv   row counts (deterministic)
  benchmark_results/run2/<comp>__*.json, else benchmark_results/**/<comp>__*.json
                             offline grade records (my-agent + any comparator agents);
                             none found -> grading recorded as "not recorded"

Metric name/direction come only from records (config, dossier, experiment entries, grade
record is_lower_better) — never from a lookup table of known competitions.
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_ASSETS = os.path.join(ROOT, ".claude", "skills", "kaggle-mlspec-report", "assets")
sys.path.insert(0, SKILL_ASSETS)
from collect import _is_diagnostic, parse_status  # noqa: E402  (reuse, don't fork)

# Structured JSON records a complex lane may have produced (any modality). Glob-based —
# no filename is tied to a specific competition.
JSON_RECORD_GLOBS = (
    "cv_result*.json",
    "res_*.json",
    "*_results.json",
    "discovery*.json",
    "dossier.json",
    "eda_summary*.json",
    "scripts/*_cv.json",
    "scripts/*decision*.json",
    "scripts/*_results.json",
)
# Prose records carried verbatim so any number quoted from them stays traceable.
TEXT_RECORD_GLOBS = ("headless*.log",)
# GBDT tree-search pipeline signature -> that lane belongs to the skill's collect.py.
GBDT_SIGNATURE_GLOB = "experiments_tree*.json"

_SEED_RE = re.compile(r"^SEED\s*=\s*(\d+)\s*$", re.M)
_LOSS_RES = (
    re.compile(r"\b([A-Za-z_]*Loss[A-Za-z_]*)\s*\("),
    re.compile(r"\bF\.(binary_cross_entropy_with_logits|binary_cross_entropy|"
               r"cross_entropy|l1_loss|mse_loss|smooth_l1_loss|huber_loss|nll_loss|"
               r"kl_div)\b"),
)
_OPT_RES = (
    re.compile(r"torch\.optim\.([A-Za-z]\w*)\s*\("),
    re.compile(r"\b(AdamW|Adam|SGD|RMSprop|Adagrad|LAMB|Lion)\s*\("),
)
_SCHED_RES = (
    re.compile(r"lr_scheduler\.([A-Za-z]\w*)\s*\("),
    re.compile(r"\b(OneCycleLR|CosineAnnealingLR|CosineAnnealingWarmRestarts|LambdaLR|"
               r"StepLR|MultiStepLR|ExponentialLR|ReduceLROnPlateau)\b"),
    re.compile(r"\b(get_(?:cosine|linear|polynomial)_schedule\w*)\b"),
)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)   # bad JSON must raise loudly, never be silently skipped


def _globs(comp_dir, patterns):
    seen, out = set(), []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(comp_dir, pattern))):
            rel = os.path.relpath(path, comp_dir)
            if rel not in seen and os.path.isfile(path):
                seen.add(rel)
                out.append((rel, path))
    return out


def collect_records(comp_dir: str):
    """Copy every discovered structured/text record verbatim."""
    records = {rel: _read_json(path) for rel, path in _globs(comp_dir, JSON_RECORD_GLOBS)
               if rel != "facts.json"}    # never re-ingest our own output
    texts = {rel: open(path, encoding="utf-8").read()
             for rel, path in _globs(comp_dir, TEXT_RECORD_GLOBS)}
    return records, texts


def grep_scripts(comp_dir: str):
    """Deterministic verbatim greps over scripts/*.py: SEED constant and the loss /
    optimizer / LR-scheduler identifiers the code actually references. No inference."""
    per_file = {}
    for rel, path in _globs(comp_dir, ("scripts/*.py",)):
        src = open(path, encoding="utf-8").read()
        entry = {}
        m = _SEED_RE.search(src)
        if m:
            entry["seed"] = int(m.group(1))
        for key, regexes in (("loss", _LOSS_RES), ("optimizer", _OPT_RES),
                             ("lr_scheduler", _SCHED_RES)):
            hits = sorted({m for rx in regexes for m in rx.findall(src)})
            if hits:
                entry[key] = hits
        if entry:
            per_file[rel] = entry
    seeds = {e["seed"] for e in per_file.values() if "seed" in e}
    unique_seed = seeds.pop() if len(seeds) == 1 else None
    return per_file, unique_seed


def _csv_rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1  # minus header


def submission_facts(comp_dir: str):
    """Deterministic row counts (minus header) for every submission CSV present."""
    files = {rel: _csv_rows(path)
             for rel, path in _globs(comp_dir, ("submission*.csv", "submissions/*.csv"))}
    return files.get("submission.csv"), files


def grade_records(comp: str, grade_record_arg, missing: list):
    """Generic offline grade-record lookup: benchmark_results/run2/<comp>__*.json first,
    then any <comp>__*.json under benchmark_results/. --grade-record overrides my-agent."""
    agents, source_dirs = {}, []
    candidates = sorted(glob.glob(
        os.path.join(ROOT, "benchmark_results", "run2", f"{comp}__*.json")))
    if not candidates:
        candidates = sorted(glob.glob(
            os.path.join(ROOT, "benchmark_results", "**", f"{comp}__*.json"),
            recursive=True))
    for path in candidates:
        agent = os.path.basename(path).split("__", 1)[1].rsplit(".json", 1)[0]
        if "." in agent:
            continue   # quarantined records (e.g. *.STALLED-attempt) are not grades
        if agent not in agents:
            agents[agent] = _read_json(path)
            source_dirs.append(os.path.relpath(os.path.dirname(path), ROOT))
    if grade_record_arg:
        agents["my-agent"] = _read_json(grade_record_arg)
        source_dirs.append(os.path.dirname(grade_record_arg) or ".")
    if not agents:
        missing.append("grading (no grade record found under benchmark_results/)")
        return "not recorded"
    return {
        "note": ("offline grade record(s) — graded against a held-out split, not a live "
                 "leaderboard; see competition config notes for grading context"),
        "source_dirs": sorted(set(source_dirs)),
        "agents": agents,
    }


def resolve_metric(config, experiments, grading, missing):
    """Metric name/direction strictly from records — no competition lookup table."""
    name = source = None
    if config.get("evaluation_metric"):
        name, source = config["evaluation_metric"], "config:evaluation_metric"
    # else: the caller falls back to a discovered dossier.json record (still a record)
    direction = dsource = None
    if config.get("optimization_direction"):
        direction, dsource = config["optimization_direction"], "config:optimization_direction"
    else:
        dirs = {e.get("direction") for e in experiments if e.get("direction")}
        if len(dirs) == 1:
            direction, dsource = dirs.pop(), "experiments entries"
        elif isinstance(grading, dict):
            rec = grading["agents"].get("my-agent") or next(iter(grading["agents"].values()))
            if "is_lower_better" in rec:
                direction = "minimize" if rec["is_lower_better"] else "maximize"
                dsource = "grade record:is_lower_better"
    if name is None:
        missing.append("metric_name (not in config or records)")
    if direction is None:
        missing.append("metric_direction (not in config, experiments, or grade record)")
    return {"name": name, "name_source": source,
            "direction": direction, "direction_source": dsource}


def _resolve_dirs_and_paths(comp_dir_arg):
    comp_dir = comp_dir_arg
    if not os.path.isdir(comp_dir):
        alt = os.path.join(ROOT, "competitions", comp_dir_arg)
        if os.path.isdir(alt):
            comp_dir = alt
        else:
            sys.exit(f"ERROR: competition directory not found: {comp_dir_arg}")
    return os.path.abspath(comp_dir)


def build_facts(comp_dir: str, grade_record_arg=None) -> dict:
    comp = os.path.basename(comp_dir.rstrip("/"))
    cfg_path = next((p for p in (os.path.join(comp_dir, "config.yaml"),
                                 os.path.join(comp_dir, "config.yml"))
                     if os.path.exists(p)), None)
    if cfg_path is None:
        sys.exit(f"ERROR: config.yaml/yml not found in {comp_dir}")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    exp_files = [rel for rel, _ in _globs(comp_dir, ("experiments*.json",))
                 if not re.match(r"experiments_tree", rel)]
    if not exp_files:
        sys.exit(f"ERROR: no experiments*.json found in {comp_dir} — nothing to report on")
    primary = "experiments.json" if "experiments.json" in exp_files else exp_files[0]
    raw = _read_json(os.path.join(comp_dir, primary))

    # Complex lanes log schema_version 2 entries — pass them through verbatim (they are
    # already structured records); anything else goes to "unparsed", never guessed at.
    experiments, unparsed = [], []
    for i, entry in enumerate(raw):
        if isinstance(entry, dict) and entry.get("schema_version") == 2:
            e = dict(entry)
            e.setdefault("experiment_id", i + 1)
            e.setdefault("direction", config.get("optimization_direction"))
            experiments.append(e)
        else:
            unparsed.append(entry)

    missing = []
    grading = grade_records(comp, grade_record_arg, missing)
    metric = resolve_metric(config, experiments, grading, missing)

    scored = [e for e in experiments if e.get("score") is not None]
    eligible = [e for e in scored if not _is_diagnostic(e)]
    dirs = {e["direction"] for e in scored if e.get("direction")}
    if len(dirs) > 1:
        raise ValueError(f"{comp_dir}: experiments disagree on optimization direction "
                         f"{sorted(dirs)}; fix the log rather than letting entry 0 decide")
    direction = dirs.pop() if dirs else metric["direction"]
    minimize = direction == "minimize"
    best = (min if minimize else max)(eligible, key=lambda e: e["score"]) if eligible else None
    trajectory = [{"experiment_id": e["experiment_id"], "timestamp": e.get("timestamp"),
                   "model": e.get("model"), "score": e.get("score")} for e in experiments]

    records, texts = collect_records(comp_dir)
    script_greps, script_seed = grep_scripts(comp_dir)
    sub_rows, sub_files = submission_facts(comp_dir)

    # metric name fallback from a discovered dossier record (still a record, not a table)
    if metric["name"] is None:
        for rel, rec in records.items():
            if rel.startswith("dossier") and isinstance(rec, dict):
                name = (rec.get("metric") or {}).get("name") if isinstance(
                    rec.get("metric"), dict) else None
                if name:
                    metric["name"], metric["name_source"] = name, f"{rel}:metric.name"
                    missing[:] = [m for m in missing if not m.startswith("metric_name")]
                    break

    if sub_rows is None:
        missing.append("submission.csv")
    if not any(rel.startswith("eda_summary") for rel in records):
        missing.append("eda_summary (no eda_summary*.json — data volume/profiling "
                       "not recorded)")
    if not texts:
        missing.append("headless log (no headless*.log)")

    return {
        "competition": config,
        "lane_type": "special/complex (non-GBDT records; collected by collect_dl_facts.py)",
        "metric": metric,
        "grading": grading,
        "status": parse_status(comp_dir),
        "texts": texts,
        "experiments": experiments,
        "experiments_file": primary,
        "other_experiment_files": [f for f in exp_files if f != primary],
        "best": best,
        "excluded_diagnostics": [{"experiment_id": e["experiment_id"], "score": e.get("score")}
                                 for e in scored if _is_diagnostic(e)],
        "trajectory": trajectory,
        "records": records,
        "script_greps": script_greps,
        "script_seed": script_seed,
        "submission_rows": sub_rows,
        "submission_files": sub_files,
        "missing": missing,
        "unparsed": unparsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("competition_dir",
                    help="path to the competition workspace (or a name under competitions/)")
    ap.add_argument("--grade-record", default=None,
                    help="explicit path to this lane's offline grade record JSON "
                         "(overrides the benchmark_results/ my-agent lookup)")
    ap.add_argument("--out", default=None,
                    help="output path (default: <competition-dir>/facts.json)")
    ap.add_argument("--force", action="store_true",
                    help="extract even if the workspace looks like a GBDT tree-search lane")
    args = ap.parse_args()

    comp_dir = _resolve_dirs_and_paths(args.competition_dir)
    tree_records = sorted(os.path.basename(p) for p in
                          glob.glob(os.path.join(comp_dir, GBDT_SIGNATURE_GLOB)))
    if tree_records and not args.force:
        sys.exit(f"NOT A DL/COMPLEX LANE: {comp_dir} contains GBDT tree-search records "
                 f"({', '.join(tree_records)}) — use the mlspec skill's collect.py for "
                 f"this lane. (--force to extract generic facts anyway; add --out to "
                 f"avoid overwriting collect.py's facts.json)")

    facts = build_facts(comp_dir, args.grade_record)
    out = args.out or os.path.join(comp_dir, "facts.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    grading = facts["grading"]
    n_agents = len(grading["agents"]) if isinstance(grading, dict) else 0
    print(f"wrote {out}: {len(facts['experiments'])} experiments "
          f"(from {facts['experiments_file']}), {len(facts['records'])} record files, "
          f"{len(facts['texts'])} text records, {n_agents} grade records, "
          f"{len(facts['unparsed'])} unparsed, missing={facts['missing']}")


if __name__ == "__main__":
    main()
