"""
Deterministic fact extractor for kaggle-report.

Usage (from project root):
    uv run python3 .claude/skills/kaggle-report/assets/collect.py <competition-name>

Reads  competitions/<name>/config.yaml + experiments.json (+ STATUS.md presence)
Writes competitions/<name>/facts.json

facts.json is the ONLY source of numbers for REPORT.md — the LLM never reads
experiments.json directly. Legacy entries are normalized by tolerant adapters;
unrecognizable entries go to "unparsed" verbatim (never dropped, never guessed).
"""
import json
import os
import re
import sys

import yaml

MINIMIZE_METRICS = {"mae", "rmse", "rmsle", "msle", "logloss", "log_loss", "smape"}


def detect_format(e: dict) -> str:
    if e.get("schema_version") == 2:
        return "v2"
    if "base_models" in e and "blend_oof_mae" in e:
        return "skill_train"       # s3e16 scripts/train.py 手刻型
    if "per_model" in e and "blend_score" in e:
        return "generic_batch"     # competitions/run_competition.py 型
    if "cv_mean" in e and "cv_scores" in e:
        return "log_v1"            # experiment_log.py v1 型
    return "unknown"


def _parse_cv_scheme(s) -> dict:
    cv = {"scheme": s}
    m = re.match(r"(\d+)fold", str(s))
    if m:
        cv["n_splits"] = int(m.group(1))
    return cv


V2_PASSTHROUGH = ("model", "score", "cv", "features", "n_features", "base_models",
                  "ensemble", "postprocess", "submission", "leaderboard", "notes")


def normalize(e: dict, idx: int, default_metric, default_direction):
    """Map one raw entry onto v2 fields; return None if unrecognizable."""
    fmt = detect_format(e)
    if fmt == "unknown":
        return None
    n = {
        "experiment_id": e.get("experiment_id", idx + 1),
        "timestamp": e.get("timestamp"),
        "source_format": fmt,
        "metric": e.get("metric") or e.get("eval_metric") or default_metric,
    }
    n["direction"] = e.get("direction") or default_direction or (
        "minimize" if str(n["metric"]).lower() in MINIMIZE_METRICS else "maximize")

    if fmt == "v2":
        n.update({k: e[k] for k in V2_PASSTHROUGH if k in e})

    elif fmt == "skill_train":
        n["base_models"] = [
            {"name": b["model"],
             "score": next(v for k, v in b.items() if k.startswith("oof_")),
             "time_s": b.get("time_s")}
            for b in e["base_models"]
        ]
        n["model"] = "+".join(b["name"] for b in n["base_models"]) + " blend"
        n["ensemble"] = {"weights": e.get("blend_weights"), "score": e["blend_oof_mae"]}
        n["score"] = e["blend_oof_mae"]
        if e.get("use_round"):
            n["postprocess"] = ["round"]
        if "features" in e:
            n["features"] = e["features"]
            n["n_features"] = e.get("n_features", len(e["features"]))
        if "cv" in e:
            n["cv"] = _parse_cv_scheme(e["cv"])
        for k in ("submission", "leaderboard"):
            if k in e:
                n[k] = e[k]

    elif fmt == "generic_batch":
        n["model"] = e.get("model", "generic blend")
        n["base_models"] = [{"name": k, "score": v} for k, v in e["per_model"].items()]
        n["ensemble"] = {"weights": e.get("blend_weights"), "score": e["blend_score"]}
        n["score"] = e["blend_score"]
        if "n_features" in e:
            n["n_features"] = e["n_features"]
        if "cv" in e:
            n["cv"] = _parse_cv_scheme(e["cv"])
        if "submission" in e:
            n["submission"] = e["submission"]

    elif fmt == "log_v1":
        n["model"] = e.get("model")
        n["score"] = e["cv_mean"]
        n["cv"] = {"scheme": e.get("cv_strategy"),
                   "fold_scores": e.get("cv_scores"), "std": e.get("cv_std")}
        if "n_features" in e:
            n["n_features"] = e["n_features"]
        if e.get("features"):
            n["feature_file"] = e["features"]   # v1 的 features 是檔名字串
        if e.get("notes"):
            n["notes"] = e["notes"]

    return n


def build_facts(comp_dir: str) -> dict:
    cfg_path = os.path.join(comp_dir, "config.yaml")
    exp_path = os.path.join(comp_dir, "experiments.json")
    for path, what in ((cfg_path, "config.yaml"), (exp_path, "experiments.json")):
        if not os.path.exists(path):
            sys.exit(f"ERROR: {what} not found in {comp_dir} — "
                     f"run the kaggle-agent pipeline first (Stage 0+).")
    config = yaml.safe_load(open(cfg_path))
    raw = json.load(open(exp_path))  # 壞 JSON → 直接讓例外炸出,不靜默跳過

    experiments, unparsed = [], []
    for i, entry in enumerate(raw):
        n = normalize(entry, i, config.get("evaluation_metric"),
                      config.get("optimization_direction"))
        if n is None:
            unparsed.append(entry)
        else:
            experiments.append(n)

    scored = [e for e in experiments if e.get("score") is not None]
    minimize = bool(scored) and scored[0]["direction"] == "minimize"
    best = (min if minimize else max)(scored, key=lambda e: e["score"]) if scored else None
    trajectory = [{"experiment_id": e["experiment_id"], "timestamp": e.get("timestamp"),
                   "score": e.get("score"), "source_format": e["source_format"]}
                  for e in experiments]
    material_level = ("full" if any(e["source_format"] != "generic_batch"
                                    for e in experiments) else "baseline-only")
    leaderboard = next((e["leaderboard"] for e in reversed(experiments)
                        if e.get("leaderboard")), None)
    missing = []
    if leaderboard is None:
        missing.append("leaderboard")
    if not any(e.get("features") for e in experiments):
        missing.append("feature_list")
    if not any(isinstance(e.get("cv"), dict) and e["cv"].get("seed") is not None
               for e in experiments):
        missing.append("cv_seed")

    return {
        "competition": config,
        "material_level": material_level,
        "status_md_present": os.path.exists(os.path.join(comp_dir, "STATUS.md")),
        "experiments": experiments,
        "best": best,
        "trajectory": trajectory,
        "leaderboard": leaderboard,
        "missing": missing,
        "unparsed": unparsed,
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: collect.py <competition-name>")
    comp_dir = os.path.join("competitions", sys.argv[1])
    facts = build_facts(comp_dir)
    out = os.path.join(comp_dir, "facts.json")
    with open(out, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}: {len(facts['experiments'])} experiments "
          f"({facts['material_level']}), {len(facts['unparsed'])} unparsed, "
          f"missing={facts['missing']}")


if __name__ == "__main__":
    main()
