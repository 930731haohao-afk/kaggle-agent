# kaggle-report Skill 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 `kaggle-report` skill:競賽跑完後自動產出結構化分析報告(REPORT.md + PDF),數字由 Python 決定性抽取、敘述由 LLM 撰寫;並順手統一 experiments.json 的 v2 schema。

**Architecture:** 混合式三層——`collect.py` 把 config.yaml + experiments.json 正規化成 `facts.json`(唯一數字來源);agent 依 `report_structure.md` 填模板寫敘述;`verify_report.py` 核對報告數字皆可回溯 facts.json;`md2pdf.sh` 以 chromium headless 轉 PDF。同時升級 `experiment_log.py` 為 v2 正規 schema 並在既有 skill 強制使用。

**Tech Stack:** Python 3.13(uv 管理)、pytest(dev)、python-markdown(dev)、PyYAML(既有)、chromium headless(系統已裝,149)。

**Spec:** `docs/superpowers/specs/2026-07-03-kaggle-report-skill-design.md`

## Global Constraints

- 專案根目錄:`/home/tjyen/ai_agents/kaggle`,所有指令從這裡執行
- 一律用 `uv`:跑腳本 `uv run python3 …`、跑測試 `uv run pytest …`、加依賴 `uv add --dev …`;禁止裸 pip/python
- 兩份 `experiment_log.py`(kaggle-agent 與 kaggle-agent-self-improvement 的 assets/utils/)必須保持 byte-identical:改 kaggle-agent 那份後 `cp` 過去,測試會驗證
- LLM 禁止改寫數字:報告中每個數字必須逐字來自 facts.json(spec §3 單一資料流原則)
- 舊 experiments.json 一律不改動(容錯讀取,spec §2)
- commit 訊息結尾加:`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: 測試基礎建設 + experiment_log.py v2

**Files:**
- Modify: `pyproject.toml`(經 `uv add --dev`)
- Create: `tests/conftest.py`
- Create: `tests/test_experiment_log_v2.py`
- Modify: `.claude/skills/kaggle-agent/assets/utils/experiment_log.py`(追加函式,不動既有函式)
- Modify: `.claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py`(cp 同步)

**Interfaces:**
- Produces: `log_experiment_v2(competition_dir, model, metric, direction, score, cv=None, features=None, base_models=None, ensemble=None, postprocess=None, submission=None, leaderboard=None, notes="") -> int`(回傳 experiment_id);v2 entry 含 `"schema_version": 2` 欄位(Task 2 的 adapter 靠它辨識)
- Produces: `tests/conftest.py` 的 `load_module` fixture 與 `ROOT`(後續所有測試共用)

- [ ] **Step 1: 安裝 dev 依賴**

```bash
uv add --dev pytest markdown
```

預期:pyproject.toml 出現 `[dependency-groups] dev = ["markdown>=…", "pytest>=…"]`。

- [ ] **Step 2: 建 conftest**

`tests/conftest.py`:

```python
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def load_module():
    """Load a module from a repo-relative path (skills/ 不是 package,只能用路徑載入)."""
    def _load(rel_path: str, name: str):
        spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return _load
```

- [ ] **Step 3: 寫失敗測試**

`tests/test_experiment_log_v2.py`:

```python
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_PATH = ".claude/skills/kaggle-agent/assets/utils/experiment_log.py"


def test_v2_writes_full_schema(tmp_path, load_module):
    log = load_module(LOG_PATH, "explog")
    exp_id = log.log_experiment_v2(
        str(tmp_path),
        model="LGB+XGB+CAT blend", metric="mae", direction="minimize", score=1.33812,
        cv={"scheme": "stratified_kfold_agebin", "n_splits": 5, "seed": 42},
        features=["Length", "Diameter"],
        base_models=[{"name": "LGB", "score": 1.35651, "time_s": 30.8, "params": {}}],
        ensemble={"weights": {"LGB": 0.6, "XGB": 0.3, "CAT": 0.1}, "score": 1.35589},
        postprocess=["round", "clip(1, 29)"],
        submission="sub.csv",
        leaderboard={"public": 1.34356, "private": 1.34075, "submitted": "2026-07-03"},
        notes="test note",
    )
    data = json.load(open(tmp_path / "experiments.json"))
    assert exp_id == 1 and len(data) == 1
    e = data[0]
    assert e["schema_version"] == 2
    assert e["experiment_id"] == 1
    assert e["score"] == 1.33812
    assert e["n_features"] == 2
    assert e["ensemble"]["weights"]["LGB"] == 0.6
    assert e["leaderboard"]["public"] == 1.34356
    assert e["postprocess"] == ["round", "clip(1, 29)"]


def test_v2_appends_and_omits_optional_fields(tmp_path, load_module):
    log = load_module(LOG_PATH, "explog2")
    log.log_experiment_v2(str(tmp_path), model="LGB", metric="rmse",
                          direction="minimize", score=0.5)
    exp_id = log.log_experiment_v2(str(tmp_path), model="XGB", metric="rmse",
                                   direction="minimize", score=0.4)
    data = json.load(open(tmp_path / "experiments.json"))
    assert exp_id == 2 and len(data) == 2
    e = data[1]
    # 未提供的選填欄位不寫入(JSON 保持乾淨)
    for absent in ("cv", "features", "base_models", "ensemble",
                   "postprocess", "submission", "leaderboard", "notes"):
        assert absent not in e


def test_both_skill_copies_identical():
    a = (ROOT / ".claude/skills/kaggle-agent/assets/utils/experiment_log.py").read_bytes()
    b = (ROOT / ".claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py").read_bytes()
    assert a == b, "兩份 experiment_log.py 必須 byte-identical(改完要 cp 同步)"
```

- [ ] **Step 4: 跑測試確認失敗**

```bash
uv run pytest tests/test_experiment_log_v2.py -v
```

預期:前兩個 FAIL(`AttributeError: … has no attribute 'log_experiment_v2'`),第三個 PASS(目前兩份相同)。

- [ ] **Step 5: 實作 log_experiment_v2**

在 `.claude/skills/kaggle-agent/assets/utils/experiment_log.py` 的 `log_experiment`(v1,保留不動)之後追加:

```python
def log_experiment_v2(
    competition_dir: str,
    model: str,
    metric: str,
    direction: str,
    score: float,
    cv: Optional[dict] = None,
    features: Optional[List[str]] = None,
    base_models: Optional[List[dict]] = None,
    ensemble: Optional[dict] = None,
    postprocess: Optional[List[str]] = None,
    submission: Optional[str] = None,
    leaderboard: Optional[dict] = None,
    notes: str = "",
) -> int:
    """
    Log an experiment in the canonical v2 schema (spec:
    docs/superpowers/specs/2026-07-03-kaggle-report-skill-design.md §6).

    Required: model, metric, direction ("minimize"/"maximize"),
    score (final representative score, post-processed).
    Optional fields are omitted from the JSON entry when not given.
    Returns the experiment_id.
    """
    experiments = load_experiments(competition_dir)
    entry = {
        "schema_version": 2,
        "experiment_id": len(experiments) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "metric": metric,
        "direction": direction,
        "score": round(float(score), 6),
    }
    if features is not None:
        entry["features"] = list(features)
        entry["n_features"] = len(features)
    for key, val in (("cv", cv), ("base_models", base_models), ("ensemble", ensemble),
                     ("postprocess", postprocess), ("submission", submission),
                     ("leaderboard", leaderboard)):
        if val is not None:
            entry[key] = val
    if notes:
        entry["notes"] = notes
    experiments.append(entry)
    save_experiments(competition_dir, experiments)
    return entry["experiment_id"]
```

然後同步副本:

```bash
cp .claude/skills/kaggle-agent/assets/utils/experiment_log.py \
   .claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py
```

- [ ] **Step 6: 跑測試確認全過**

```bash
uv run pytest tests/test_experiment_log_v2.py -v
```

預期:3 PASS。

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tests/ \
  .claude/skills/kaggle-agent/assets/utils/experiment_log.py \
  .claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py
git commit -m "feat: experiment_log v2 正規 schema + 測試基礎建設

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: collect.py 事實抽取器

**Files:**
- Create: `.claude/skills/kaggle-report/assets/collect.py`
- Create: `tests/test_collect.py`

**Interfaces:**
- Consumes: v2 schema 的 `schema_version == 2` 標記(Task 1)
- Produces: CLI `uv run python3 .claude/skills/kaggle-report/assets/collect.py <competition-name>` → 寫 `competitions/<name>/facts.json`
- Produces: `build_facts(comp_dir) -> dict`、`normalize(entry, idx, default_metric, default_direction) -> dict | None`、`detect_format(entry) -> str`(測試與 Task 3 驗證用)
- Produces: facts.json 頂層鍵:`competition`(config.yaml 原文)、`material_level`("full"/"baseline-only")、`status_md_present`(bool)、`experiments`(正規化列表,各含 `source_format`)、`best`、`trajectory`、`leaderboard`、`missing`、`unparsed`

- [ ] **Step 1: 寫失敗測試(fixtures 取自真實資料)**

`tests/test_collect.py`:

```python
import json

import pytest

COLLECT = ".claude/skills/kaggle-report/assets/collect.py"

# ---- fixtures:三種舊格式 + v2 + 未知,欄位取自專案真實紀錄 ----

SKILL_TRAIN_ENTRY = {  # s3e16 scripts/train.py 手刻型
    "timestamp": "2026-07-03T09:18:40",
    "n_features": 24,
    "features": ["Length", "Diameter", "volume"],
    "cv": "5fold_stratified_agebin",
    "base_models": [
        {"model": "LGB", "oof_mae": 1.35651, "time_s": 30.8},
        {"model": "XGB", "oof_mae": 1.35763, "time_s": 36.7},
    ],
    "blend_weights": {"LGB": 0.6, "XGB": 0.4},
    "blend_oof_mae": 1.35589,
    "use_round": True,
    "submission": "sub_blend_20260703_091840.csv",
    "leaderboard": {"public": 1.34356, "private": 1.34075, "submitted": "2026-07-03"},
}

GENERIC_BATCH_ENTRY = {  # run_competition.py 型
    "experiment_id": 2,
    "timestamp": "2026-07-03T11:57:39",
    "model": "generic LGB+XGB+CAT blend",
    "metric": "mae",
    "n_features": 8,
    "cv": "5fold",
    "per_model": {"LGB": 1.3623, "XGB": 1.35886, "CAT": 1.35714},
    "blend_weights": {"LGB": 0.3, "XGB": 0.2, "CAT": 0.5},
    "blend_score": 1.35441,
    "submission": "sub_generic_1.35441_20260703_115739.csv",
}

LOG_V1_ENTRY = {  # experiment_log.py v1 型
    "experiment_id": 3,
    "timestamp": "2026-07-01T10:00:00",
    "model": "lgbm_baseline",
    "features": "features_v1.py",
    "n_features": 10,
    "params": {"n_estimators": "500"},
    "cv_strategy": "5-fold",
    "cv_scores": [1.36, 1.35, 1.37, 1.36, 1.35],
    "cv_mean": 1.358,
    "cv_std": 0.0075,
    "eval_metric": "mae",
    "notes": "baseline",
}

V2_ENTRY = {
    "schema_version": 2, "experiment_id": 4, "timestamp": "2026-07-04T08:00:00",
    "model": "LGB tuned", "metric": "mae", "direction": "minimize", "score": 1.33,
    "ensemble": {"weights": {"LGB": 1.0}, "score": 1.33},
}

UNKNOWN_ENTRY = {"foo": "bar", "score?": 1}

CONFIG = """\
name: playground-series-s3e16
problem_type: regression
evaluation_metric: mae
optimization_direction: minimize
target_column: Age
id_column: id
"""


@pytest.fixture
def comp_dir(tmp_path):
    d = tmp_path / "playground-series-s3e16"
    d.mkdir()
    (d / "config.yaml").write_text(CONFIG)
    return d


def _write_exps(d, entries):
    (d / "experiments.json").write_text(json.dumps(entries))


def test_detect_format(load_module):
    c = load_module(COLLECT, "collect")
    assert c.detect_format(SKILL_TRAIN_ENTRY) == "skill_train"
    assert c.detect_format(GENERIC_BATCH_ENTRY) == "generic_batch"
    assert c.detect_format(LOG_V1_ENTRY) == "log_v1"
    assert c.detect_format(V2_ENTRY) == "v2"
    assert c.detect_format(UNKNOWN_ENTRY) == "unknown"


def test_skill_train_mapping(load_module, comp_dir):
    c = load_module(COLLECT, "collect")
    _write_exps(comp_dir, [SKILL_TRAIN_ENTRY])
    facts = c.build_facts(str(comp_dir))
    e = facts["experiments"][0]
    assert e["source_format"] == "skill_train"
    assert e["score"] == 1.35589                      # blend_oof_mae → score
    assert e["ensemble"]["score"] == 1.35589
    assert e["ensemble"]["weights"]["LGB"] == 0.6
    assert e["base_models"][0] == {"name": "LGB", "score": 1.35651, "time_s": 30.8}
    assert e["postprocess"] == ["round"]              # use_round → postprocess
    assert e["cv"] == {"scheme": "5fold_stratified_agebin", "n_splits": 5}
    assert e["metric"] == "mae" and e["direction"] == "minimize"  # 取自 config
    assert e["leaderboard"]["private"] == 1.34075
    assert e["experiment_id"] == 1                    # 無 id → 以序補上


def test_generic_batch_mapping_and_material_level(load_module, comp_dir):
    c = load_module(COLLECT, "collect")
    _write_exps(comp_dir, [GENERIC_BATCH_ENTRY])
    facts = c.build_facts(str(comp_dir))
    e = facts["experiments"][0]
    assert e["source_format"] == "generic_batch"
    assert e["score"] == 1.35441
    assert e["base_models"] == [{"name": "LGB", "score": 1.3623},
                                {"name": "XGB", "score": 1.35886},
                                {"name": "CAT", "score": 1.35714}]
    assert facts["material_level"] == "baseline-only"  # 只有 generic 紀錄
    assert facts["leaderboard"] is None
    assert "leaderboard" in facts["missing"]


def test_log_v1_mapping(load_module, comp_dir):
    c = load_module(COLLECT, "collect")
    _write_exps(comp_dir, [LOG_V1_ENTRY])
    e = c.build_facts(str(comp_dir))["experiments"][0]
    assert e["source_format"] == "log_v1"
    assert e["score"] == 1.358                        # cv_mean → score
    assert e["cv"]["fold_scores"] == [1.36, 1.35, 1.37, 1.36, 1.35]
    assert e["cv"]["std"] == 0.0075


def test_mixed_best_trajectory_unparsed(load_module, comp_dir):
    c = load_module(COLLECT, "collect")
    _write_exps(comp_dir, [SKILL_TRAIN_ENTRY, GENERIC_BATCH_ENTRY, V2_ENTRY, UNKNOWN_ENTRY])
    facts = c.build_facts(str(comp_dir))
    assert len(facts["experiments"]) == 3
    assert facts["material_level"] == "full"          # 有非 generic 紀錄
    assert facts["best"]["score"] == 1.33             # minimize → 最小者(v2 entry)
    assert [t["score"] for t in facts["trajectory"]] == [1.35589, 1.35441, 1.33]
    assert facts["unparsed"] == [UNKNOWN_ENTRY]       # 保留原文,不丟棄
    assert facts["leaderboard"]["public"] == 1.34356  # 取自有 leaderboard 的紀錄


def test_missing_inputs_exit(load_module, tmp_path):
    c = load_module(COLLECT, "collect")
    with pytest.raises(SystemExit):
        c.build_facts(str(tmp_path / "nonexistent"))
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run pytest tests/test_collect.py -v
```

預期:全部 FAIL(`FileNotFoundError`,collect.py 不存在)。

- [ ] **Step 3: 實作 collect.py**

`.claude/skills/kaggle-report/assets/collect.py`:

```python
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
```

- [ ] **Step 4: 跑測試確認全過**

```bash
uv run pytest tests/test_collect.py -v
```

預期:7 PASS。

- [ ] **Step 5: 對真實資料煙霧測試**

```bash
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e16 && \
uv run python3 -c "
import json
f = json.load(open('competitions/playground-series-s3e16/facts.json'))
assert f['material_level'] == 'full'
assert f['leaderboard']['public'] == 1.34356
assert len(f['experiments']) == 2 and not f['unparsed']
print('real-data smoke OK')"
```

預期:`wrote … 2 experiments (full) …` + `real-data smoke OK`。

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/kaggle-report/assets/collect.py tests/test_collect.py \
        competitions/playground-series-s3e16/facts.json
git commit -m "feat: collect.py 事實抽取器(3 種舊格式容錯 + facts.json)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: verify_report.py 數字一致性檢查器

**Files:**
- Create: `.claude/skills/kaggle-report/assets/verify_report.py`
- Create: `tests/test_verify_report.py`

**Interfaces:**
- Consumes: facts.json(Task 2 的結構,但只依賴「JSON 裡有數值」這件事)
- Produces: CLI `uv run python3 .claude/skills/kaggle-report/assets/verify_report.py <REPORT.md> <facts.json>`,全數可回溯 → exit 0;有可疑數字 → 列出並 exit 1
- Produces: `find_suspects(report_text, facts) -> list[str]`(測試用)

檢查規則(啟發式,對應 spec §9.1):報告中每個「帶小數點的數」與「≥3 位數整數」必須出現在 facts.json 的任一數值或字串中(容許 1–6 位小數的四捨五入變體);年份(19xx/20xx)與 code block 內文字豁免(重現指令含路徑/時間戳)。

- [ ] **Step 1: 寫失敗測試**

`tests/test_verify_report.py`:

```python
FACTS = {
    "competition": {"name": "x", "evaluation_metric": "mae"},
    "experiments": [{"score": 1.35589, "n_features": 24,
                     "submission": "sub_blend_20260703_091840.csv",
                     "leaderboard": {"public": 1.34356}}],
    "trajectory": [],
}

VERIFY = ".claude/skills/kaggle-report/assets/verify_report.py"


def test_traceable_numbers_pass(load_module):
    v = load_module(VERIFY, "verify")
    text = ("OOF MAE 為 1.35589,特徵共 24 個,Public LB **1.34356**。"
            "1.3559 亦可(四捨五入變體)。競賽年份 2023。")
    assert v.find_suspects(text, FACTS) == []


def test_invented_number_flagged(load_module):
    v = load_module(VERIFY, "verify")
    assert v.find_suspects("改善了 85.4%", FACTS) == ["85.4"]


def test_code_blocks_exempt(load_module):
    v = load_module(VERIFY, "verify")
    text = "指令:\n```bash\nkaggle submit -f sub_99999.csv\n```\n"
    assert v.find_suspects(text, FACTS) == []


def test_numbers_inside_facts_strings_count(load_module):
    v = load_module(VERIFY, "verify")
    # 20260703 出現在 facts 的 submission 檔名字串中 → 可回溯
    assert v.find_suspects("提交檔時間戳 20260703", FACTS) == []
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run pytest tests/test_verify_report.py -v
```

預期:全部 FAIL(檔案不存在)。

- [ ] **Step 3: 實作 verify_report.py**

```python
"""
Heuristic number-traceability check: every significant number in REPORT.md
must exist in facts.json (spec §9.1). Exit 0 = clean; exit 1 = suspects found.

Usage:
    uv run python3 …/verify_report.py <REPORT.md> <facts.json>
"""
import json
import pathlib
import re
import sys

_NUM = re.compile(r"\d+\.\d+|\d{3,}")


def _collect_values(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_values(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, str):
        for tok in _NUM.findall(obj):
            out.add(float(tok))


def find_suspects(report_text: str, facts: dict) -> list:
    known = set()
    _collect_values(facts, known)
    variants = set()
    for v in known:
        variants.add(v)
        for nd in range(1, 7):          # 容許四捨五入到 1–6 位小數的變體
            variants.add(round(v, nd))
    text = re.sub(r"```.*?```", "", report_text, flags=re.S)   # code block 豁免
    suspects = []
    for tok in _NUM.findall(text):
        if re.fullmatch(r"(19|20)\d{2}", tok):                  # 年份豁免
            continue
        if float(tok) in variants:
            continue
        if tok not in suspects:
            suspects.append(tok)
    return suspects


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: verify_report.py <REPORT.md> <facts.json>")
    report = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    facts = json.load(open(sys.argv[2]))
    suspects = find_suspects(report, facts)
    if suspects:
        print("FAIL — numbers not traceable to facts.json:")
        for s in suspects:
            print(f"  {s}")
        sys.exit(1)
    print("OK — all numbers traceable to facts.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑測試確認全過**

```bash
uv run pytest tests/test_verify_report.py -v
```

預期:4 PASS。

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/kaggle-report/assets/verify_report.py tests/test_verify_report.py
git commit -m "feat: verify_report.py 報告數字可回溯性檢查

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: run_competition.py 改用 v2 schema

**Files:**
- Modify: `competitions/run_competition.py:233-241`(log 區塊)
- Test: `tests/test_collect.py`(既有 v2 passthrough 測試已覆蓋讀取端;本 task 加一個寫入端形狀測試)

**Interfaces:**
- Consumes: v2 schema 欄位名(Task 1);`detect_format`(Task 2,驗證用)

- [ ] **Step 1: 加寫入端形狀測試**

在 `tests/test_collect.py` 末尾追加:

```python
def test_run_competition_logs_v2_shape(load_module, comp_dir):
    """run_competition.py 改版後寫出的 entry 應被辨識為 v2(不再是 generic_batch)。"""
    c = load_module(COLLECT, "collect")
    new_entry = {  # 與 run_competition.py 改版後 hist.append 的形狀一致
        "schema_version": 2, "experiment_id": 1, "timestamp": "2026-07-03T12:00:00",
        "model": "generic LGB+XGB+CAT blend", "metric": "mae", "direction": "minimize",
        "score": 1.35441, "n_features": 8, "cv": {"scheme": "5fold", "n_splits": 5, "seed": 42},
        "base_models": [{"name": "LGB", "score": 1.3623}],
        "ensemble": {"weights": {"LGB": 1.0}, "score": 1.35441},
        "submission": "sub_generic.csv",
    }
    assert c.detect_format(new_entry) == "v2"
    _write_exps(comp_dir, [new_entry])
    facts = c.build_facts(str(comp_dir))
    assert facts["experiments"][0]["score"] == 1.35441
    # generic 基線即使是 v2 格式,素材等級仍應是 baseline-only —— 見 Step 3 的 material 判定調整
    assert facts["material_level"] == "baseline-only"
```

- [ ] **Step 2: 跑測試確認失敗**

```bash
uv run pytest tests/test_collect.py::test_run_competition_logs_v2_shape -v
```

預期:FAIL 在最後一個 assert(v2 格式會被當成 "full")。

- [ ] **Step 3: 調整 material_level 判定 + 改 run_competition.py**

(a) `collect.py` 的 `material_level` 行改為(generic 基線用模型名辨識,與格式脫鉤):

```python
    def _is_generic(e):
        return (e["source_format"] == "generic_batch"
                or str(e.get("model", "")).startswith("generic "))
    material_level = ("full" if any(not _is_generic(e) for e in experiments)
                      else "baseline-only")
```

(b) `competitions/run_competition.py` 的 log 區塊(原 233–241 行)整段改為:

```python
    # log (canonical v2 schema — see docs/superpowers/specs/2026-07-03-kaggle-report-skill-design.md §6)
    exp = os.path.join(cdir, "experiments.json")
    hist = json.load(open(exp)) if os.path.exists(exp) else []
    hist.append(dict(
        schema_version=2,
        experiment_id=len(hist) + 1,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model="generic LGB+XGB+CAT blend",
        metric=mname,
        direction=("maximize" if higher_better else "minimize"),
        score=round(float(best_s), 5),
        n_features=len(feats),
        cv=dict(scheme=f"{N_SPLITS}fold", n_splits=N_SPLITS, seed=SEED),
        base_models=[dict(name=n, score=round(float(per[n]), 5)) for n in NAMES],
        ensemble=dict(weights=dict(zip(NAMES, [round(float(x), 2) for x in best_w])),
                      score=round(float(best_s), 5)),
        submission=os.path.basename(path),
    ))
    json.dump(hist, open(exp, "w"), indent=2)
```

注意:`higher_better` 是 run_competition.py 既有變數(resolve_metric 回傳的第 4 個值);實作時先確認該作用域內的實際變數名,以檔案現況為準。

- [ ] **Step 4: 跑全部測試確認過**

```bash
uv run pytest tests/ -v
```

預期:全 PASS(既有 material_level 測試也不能壞:s3e16 skill_train 紀錄的 model 名不以 "generic " 開頭)。

- [ ] **Step 5: Commit**

```bash
git add competitions/run_competition.py .claude/skills/kaggle-report/assets/collect.py tests/test_collect.py
git commit -m "feat: run_competition.py 改寫 v2 schema;material_level 判定與格式脫鉤

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: 既有兩個 kaggle-agent skill 加「強制使用 experiment_log」硬規則

**Files:**
- Modify: `.claude/skills/kaggle-agent/SKILL.md`(Behavioral Guidelines 的 Iteration Protocol 小節)
- Modify: `.claude/skills/kaggle-agent-self-improvement/SKILL.md`(同位置)

**Interfaces:** 無程式介面;規則文字如下。

- [ ] **Step 1: 兩份 SKILL.md 的 Iteration Protocol 小節各加一條**

在 `- **Track everything**: Log all experiments with parameters, features, and scores` 之後插入:

```markdown
- **Experiment logging is MANDATORY via `experiment_log.log_experiment_v2()`** (assets/utils/experiment_log.py).
  Never hand-roll experiment dicts in training scripts — past hand-rolled entries created three
  incompatible schemas. Import it by path in generated scripts:

  ```python
  import importlib.util
  _spec = importlib.util.spec_from_file_location(
      "experiment_log",
      ".claude/skills/kaggle-agent/assets/utils/experiment_log.py")
  experiment_log = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(experiment_log)
  experiment_log.log_experiment_v2(comp_dir, model=..., metric=..., direction=..., score=..., ...)
  ```
```

(self-improvement 那份的路徑字串改為 `.claude/skills/kaggle-agent-self-improvement/assets/utils/experiment_log.py`。)

- [ ] **Step 2: 驗證**

```bash
grep -c "log_experiment_v2" .claude/skills/kaggle-agent/SKILL.md \
                            .claude/skills/kaggle-agent-self-improvement/SKILL.md
```

預期:兩檔各 ≥1。

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/kaggle-agent/SKILL.md .claude/skills/kaggle-agent-self-improvement/SKILL.md
git commit -m "docs: skill 硬規則——實驗紀錄一律經 log_experiment_v2

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: kaggle-report skill 文件(SKILL.md + references + 模板)

**Files:**
- Create: `.claude/skills/kaggle-report/SKILL.md`
- Create: `.claude/skills/kaggle-report/references/report_structure.md`
- Create: `.claude/skills/kaggle-report/references/rubric.md`
- Create: `.claude/skills/kaggle-report/assets/report_template.md`

**Interfaces:**
- Consumes: collect.py CLI(Task 2)、verify_report.py CLI(Task 3)、md2pdf.sh(Task 7,SKILL.md 先引用,Task 7 補實作)

- [ ] **Step 1: 寫 SKILL.md**

`.claude/skills/kaggle-report/SKILL.md`:

````markdown
---
name: kaggle-report
description: |
  Generate a structured, reproducible data-analysis report (REPORT.md + PDF) for a completed
  Kaggle competition run. Numbers are extracted deterministically into facts.json by collect.py;
  the agent writes only the what/why narrative and never rewrites a number.

  Use when the user asks to: generate a competition report, 產報告, 產出分析報告, summarize a
  finished competition run, or after completing the kaggle-agent pipeline (Stage 5).

  Trigger phrases: "report", "報告", "REPORT.md", "分析報告", "kaggle report".
---

# Kaggle Report

Turns a finished competition folder into a report a data scientist can use to
reproduce the pipeline **without reading the code**.

## Hard Rules (non-negotiable)

1. **Every number in REPORT.md must be copied verbatim from facts.json.**
   Never recompute, re-round, or invent a number. Derived stats (improvement %,
   gaps) may only appear if the underlying numbers are in facts.json and the
   arithmetic is shown inline (e.g. "1.34356 − 1.33812 = 0.00544").
2. **STATUS.md is narrative context only** — you may read it to understand *why*
   decisions were made, but numbers still come from facts.json.
3. **material_level == "baseline-only"** → say so explicitly: the run used only
   the generic batch baseline; mark EDA/feature sections as 未執行. Never
   fabricate analysis that did not happen.
4. **facts.missing / facts.unparsed** → write 無紀錄 for missing items and list
   unparsed entries verbatim in section 7. Never guess.

## Pipeline

Run from project root (`/home/tjyen/ai_agents/kaggle`), in order:

1. **Collect facts (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-report/assets/collect.py <competition-name>
   ```
   Writes `competitions/<name>/facts.json`. On error: report the message to the
   user and stop — do not improvise around missing inputs.

2. **Write the report (you)**
   - Read `competitions/<name>/facts.json` and [references/report_structure.md](references/report_structure.md).
   - Copy `assets/report_template.md` to `competitions/<name>/REPORT.md`, fill every
     section per the structure spec, embedding numbers/tables from facts.json verbatim.

3. **Rubric self-check** — walk [references/rubric.md](references/rubric.md) item by
   item; fix any gap before proceeding.

4. **Verify number traceability (deterministic)**
   ```bash
   uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
       competitions/<name>/REPORT.md competitions/<name>/facts.json
   ```
   Must exit 0. If it flags numbers: fix the report. Do not add numbers to
   facts.json by hand — facts.json is generated only by collect.py.

5. **PDF (deterministic)**
   ```bash
   bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/<name>/REPORT.md
   ```
   If chromium is unavailable, keep REPORT.md as the deliverable and tell the
   user the PDF step was skipped (MD is the primary artifact).

6. Show the user where REPORT.md / REPORT.pdf landed and summarize rubric result.
````

- [ ] **Step 2: 寫 report_structure.md**

`.claude/skills/kaggle-report/references/report_structure.md`:

```markdown
# 報告結構規格(8 節)

每節列出:必填內容 + facts.json 對應來源。「無紀錄」處理見 SKILL.md Hard Rule 4。

## 1. 競賽目的
- What:競賽要解決的問題(一段)。來源:competition.description、competition.name
- Why:為何重要/評估指標為何合理(一段)。由 competition.evaluation_metric 與
  problem_type 反推(如 MAE → 對離群值穩健的絕對誤差)
- 基本資訊表:名稱、URL、問題型別、指標、優化方向。來源:competition.*

## 2. 資料規格
- 列數/欄位數/型別概述。來源:competition.notes(若記載)、experiments[].n_features
- 特別規則(外部資料、每日提交上限)。來源:competition.special_rules
- material_level == "baseline-only" → 註明「本場未執行 EDA,以下僅資料基本形狀」

## 3. 模型規格
- 使用的模型清單與各自分數表。來源:best.base_models(name/score/time_s)
- Ensemble 權重與分數。來源:best.ensemble.weights、best.ensemble.score
- 選型理由(敘述;可參考 STATUS.md 脈絡)

## 4. 訓練規格
- CV 方案表:scheme、n_splits、seed。來源:best.cv(seed 無紀錄則寫「無紀錄」)
- Objective 與關鍵超參(有紀錄才寫)。來源:best.base_models[].params
- 為何用此 CV(敘述)

## 5. 推論程序
- 後處理步驟。來源:best.postprocess(無則寫「無後處理紀錄」)
- Submission 檔名與格式。來源:best.submission、competition.id_column/target_column

## 6. 評估指標
- 指標定義(一句)+ 分數總表:各 base model、ensemble、(若有)Public/Private LB。
  來源:best.*、leaderboard
- CV↔LB gap:僅當 leaderboard 存在;差值以內嵌算式呈現(Hard Rule 1)

## 7. 實驗軌跡
- 逐實驗分數表(id、timestamp、score、source_format)。來源:trajectory
- 突破點敘述:分數躍升發生在哪筆、當時改了什麼(參考 experiments[].notes、STATUS.md)
- unparsed 非空 → 原文列出並註明「無法解析之紀錄」

## 8. 重現指令
- 從資料下載到 submission 的完整命令序列(bash code block)
- 來源:STATUS.md 的 Reproduce 節(有則沿用)或依 competitions/<name>/scripts/ 實際檔名組出
- 註明執行目錄與 uv 需求
```

- [ ] **Step 3: 寫 rubric.md**

`.claude/skills/kaggle-report/references/rubric.md`:

```markdown
# 報告驗收檢核表(rubric)

逐項核對;任一項 ✗ 即回頭補完再往下走。

| # | 檢核項 | 通過標準 |
|---|--------|----------|
| R1 | 節次完整 | 8 節齊備且順序正確 |
| R2 | What/Why | 第 1 節同時回答「解什麼問題」與「為何重要」,非罐頭句 |
| R3 | 五大元件 | 資料規格/模型規格/訓練規格/推論程序/評估指標(節 2–6)各有實質內容或明確「無紀錄/未執行」標注 |
| R4 | 數字可回溯 | verify_report.py exit 0 |
| R5 | 不看碼可重現 | 節 8 指令完整:環境(uv)、執行目錄、腳本順序、提交指令 |
| R6 | 軌跡與突破點 | 節 7 有逐實驗表;多於一筆實驗時指出最佳分數出現處 |
| R7 | 誠實性 | baseline-only 場次明確聲明;missing 項寫「無紀錄」;unparsed 原文列出;無編造分析 |
| R8 | 表格自 facts | 報告中每個表格的數據列皆可對應 facts.json 欄位 |
```

- [ ] **Step 4: 寫 report_template.md**

`.claude/skills/kaggle-report/assets/report_template.md`:

```markdown
# 競賽分析報告:<competition.name>

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:<material_level> | 產生日期:<日期>

## 1. 競賽目的
<!-- What:解什麼問題;Why:為何重要、指標為何合理 -->

| 項目 | 值 |
|------|-----|
| 競賽 | <competition.name> |
| 問題型別 | <competition.problem_type> |
| 評估指標 | <competition.evaluation_metric>(<optimization_direction>) |
| 目標欄位 | <competition.target_column> |

## 2. 資料規格
<!-- 列數/欄位/型別/特別規則;baseline-only 場次註明未執行 EDA -->

## 3. 模型規格
<!-- base models 分數表 + ensemble 權重 + 選型理由 -->

## 4. 訓練規格
<!-- CV 方案表 + objective/超參 + 為何此 CV -->

## 5. 推論程序
<!-- 後處理步驟 + submission 格式 -->

## 6. 評估指標
<!-- 分數總表 + (若有) LB 與 CV↔LB gap(內嵌算式) -->

## 7. 實驗軌跡
<!-- trajectory 表 + 突破點敘述 + unparsed 列出 -->

## 8. 重現指令
```bash
# 填:完整重現命令序列
```
```

- [ ] **Step 5: 驗證(文件型 task,無單元測試)**

```bash
ls .claude/skills/kaggle-report/SKILL.md \
   .claude/skills/kaggle-report/references/report_structure.md \
   .claude/skills/kaggle-report/references/rubric.md \
   .claude/skills/kaggle-report/assets/report_template.md
```

預期:4 檔皆存在。

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/kaggle-report/
git commit -m "feat: kaggle-report skill 文件(SKILL.md/結構規格/rubric/模板)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: report_style.css + md2pdf.sh

**Files:**
- Create: `.claude/skills/kaggle-report/assets/report_style.css`
- Create: `.claude/skills/kaggle-report/assets/md2pdf.sh`

**Interfaces:**
- Consumes: dev 依賴 `markdown`(Task 1 已裝)
- Produces: `bash md2pdf.sh <file.md> [out.pdf]` → 同目錄產出 PDF

- [ ] **Step 1: 寫 CSS**

`.claude/skills/kaggle-report/assets/report_style.css`:

```css
@page { margin: 20mm; }
body {
  font-family: "Noto Sans CJK TC", "Noto Sans TC", sans-serif;
  font-size: 11pt; line-height: 1.6; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #bbb; padding-bottom: 3px; }
table { border-collapse: collapse; margin: 0.8em 0; width: auto; }
th, td { border: 1px solid #999; padding: 4px 10px; font-size: 10pt; }
th { background: #f0f0f0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9.5pt; background: #f5f5f5; padding: 1px 4px; }
pre { background: #f5f5f5; padding: 10px; overflow-x: auto; border: 1px solid #ddd; }
pre code { background: none; padding: 0; }
blockquote { color: #555; border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; }
```

- [ ] **Step 2: 寫 md2pdf.sh**

`.claude/skills/kaggle-report/assets/md2pdf.sh`:

```bash
#!/usr/bin/env bash
# Convert a markdown report to PDF via python-markdown + chromium headless.
# Usage: bash md2pdf.sh <REPORT.md> [out.pdf]
set -euo pipefail

MD="$1"
PDF="${2:-${MD%.md}.pdf}"
CSS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/report_style.css"
HTML="${MD%.md}.tmp.html"   # 不用隱藏檔名:snap chromium 讀不到部分隱藏路徑

uv run python3 - "$MD" "$CSS" "$HTML" <<'PY'
import pathlib, sys
import markdown
md_path, css_path, html_path = sys.argv[1:4]
body = markdown.markdown(
    pathlib.Path(md_path).read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"])
css = pathlib.Path(css_path).read_text(encoding="utf-8")
pathlib.Path(html_path).write_text(
    f"<!doctype html><html><head><meta charset='utf-8'>"
    f"<style>{css}</style></head><body>{body}</body></html>",
    encoding="utf-8")
PY

CHROME="$(command -v chromium || command -v chromium-browser)"
if [ -z "$CHROME" ]; then
  rm -f "$HTML"
  echo "WARN: chromium not found — skipping PDF, REPORT.md is the deliverable" >&2
  exit 2
fi
"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$PDF" "file://$(cd "$(dirname "$HTML")" && pwd)/$(basename "$HTML")" 2>/dev/null
rm -f "$HTML"
echo "wrote $PDF"
```

```bash
chmod +x .claude/skills/kaggle-report/assets/md2pdf.sh
```

- [ ] **Step 3: 煙霧測試(含中文與表格)**

```bash
mkdir -p /tmp/claude-1000/-home-tjyen/e55b7d5b-3a0b-433a-aa8a-f3b1475f1c17/scratchpad/md2pdf_test
cat > /tmp/claude-1000/-home-tjyen/e55b7d5b-3a0b-433a-aa8a-f3b1475f1c17/scratchpad/md2pdf_test/t.md <<'EOF'
# 測試報告
中文段落與表格:

| 模型 | OOF MAE |
|------|---------|
| LGB  | 1.35651 |
EOF
bash .claude/skills/kaggle-report/assets/md2pdf.sh \
  /tmp/claude-1000/-home-tjyen/e55b7d5b-3a0b-433a-aa8a-f3b1475f1c17/scratchpad/md2pdf_test/t.md
pdfinfo /tmp/claude-1000/-home-tjyen/e55b7d5b-3a0b-433a-aa8a-f3b1475f1c17/scratchpad/md2pdf_test/t.pdf | head -5
```

預期:`wrote …/t.pdf`;pdfinfo 顯示 Pages: 1。若 snap chromium 因沙箱讀不到 /tmp,改在專案內建暫存目錄重試並記錄於 md2pdf.sh 註解(snap 對 /tmp 的存取受限是已知情況;此時煙霧測試移到 `competitions/` 下的臨時資料夾,測完刪除)。

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/kaggle-report/assets/report_style.css .claude/skills/kaggle-report/assets/md2pdf.sh
git commit -m "feat: md2pdf 轉換(python-markdown + chromium headless,中文 OK)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: 端對端驗收(s3e16 full + s3e14 baseline-only)

本 task 是「實際執行 skill」:執行者(agent)依 `.claude/skills/kaggle-report/SKILL.md` 的 Pipeline 完整走一遍,產出兩份真實報告。這裡的步驟即 SKILL.md 的步驟。

**Files:**
- Create: `competitions/playground-series-s3e16/REPORT.md`、`REPORT.pdf`、`facts.json`(Task 2 已產)
- Create: `competitions/playground-series-s3e14/REPORT.md`、`REPORT.pdf`、`facts.json`

**Interfaces:**
- Consumes: Task 2–7 全部產物

- [ ] **Step 1: s3e16 — collect**

```bash
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e16
```

預期:`2 experiments (full)`(若 Task 4 後有新增紀錄,數量照實)。

- [ ] **Step 2: s3e16 — 依 SKILL.md 撰寫 REPORT.md**

執行者閱讀 `competitions/playground-series-s3e16/facts.json`、`STATUS.md`(敘述脈絡)、`references/report_structure.md`,複製 `assets/report_template.md` 為 `competitions/playground-series-s3e16/REPORT.md` 並填滿 8 節。Hard Rules 全程適用(數字逐字取自 facts.json)。

- [ ] **Step 3: s3e16 — rubric 自檢 + verify + PDF**

依 `references/rubric.md` R1–R8 逐項檢查並修正,然後:

```bash
uv run python3 .claude/skills/kaggle-report/assets/verify_report.py \
  competitions/playground-series-s3e16/REPORT.md competitions/playground-series-s3e16/facts.json
bash .claude/skills/kaggle-report/assets/md2pdf.sh competitions/playground-series-s3e16/REPORT.md
```

預期:`OK — all numbers traceable` + `wrote …REPORT.pdf`。verify FAIL → 修報告重跑,不得改 facts.json。

- [ ] **Step 4: s3e14 — 全流程(baseline-only 誠實性驗證)**

```bash
uv run python3 .claude/skills/kaggle-report/assets/collect.py playground-series-s3e14
```

預期:`1 experiments (baseline-only)`。(若 s3e14 無 experiments.json,改用批次八場中任一有紀錄者:s3e1/s3e3/s3e5/s3e7/s3e9/s3e11/s3e19。)

接著同 Step 2–3 產出 REPORT.md:必須明確聲明「本場僅執行通用基線,未進行 EDA 與特徵工程」,節 2 標注未執行、節 7 僅一筆紀錄;再跑 verify + md2pdf,皆須通過。

- [ ] **Step 5: 全測試套件迴歸**

```bash
uv run pytest tests/ -v
```

預期:全 PASS。

- [ ] **Step 6: Commit(驗收產物入版控)**

```bash
git add competitions/playground-series-s3e16/facts.json competitions/playground-series-s3e16/REPORT.md \
        competitions/playground-series-s3e14/facts.json competitions/playground-series-s3e14/REPORT.md
git add -f competitions/playground-series-s3e16/REPORT.pdf competitions/playground-series-s3e14/REPORT.pdf 2>/dev/null || true
git commit -m "feat: kaggle-report 端對端驗收——s3e16(full)與 s3e14(baseline-only)報告

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(若 .gitignore 排除 pdf/facts.json 且 `-f` 仍失敗,MD 入版控即可,PDF 留在資料夾。)

---

## Self-Review 紀錄

- **Spec coverage**:§2 四決定 → Task 1/2/6/7;§3 資料流 → Task 2/3/6/7;§4 佈局 → Task 1–7;§5 八節 → Task 6;§6 schema+adapter 對照表 → Task 1/2;§7 facts.json → Task 2;§8 素材等級 → Task 2/4/8;§9 驗證四項 → verify(Task 3)、rubric(Task 6/8)、兩測試案例(Task 8)、回歸測試(Task 1–4);§10 錯誤處理 → Task 2 Step 3(exit+訊息、壞 JSON 直炸)、Task 7 Step 2(chromium 缺席 exit 2);§11 範圍外未混入。
- **Placeholder scan**:模板內 `<!-- 填:… -->` 為交付物本身的設計(執行期由 agent 填),非計畫佔位符;計畫步驟皆含完整程式碼與指令。
- **Type consistency**:`log_experiment_v2` 簽名(Task 1)與 SKILL.md 引用(Task 5)一致;`detect_format`/`build_facts`/`find_suspects` 名稱在 Task 2/3/4 測試間一致;facts.json 鍵名在 Task 2/3/6 間一致(`material_level`、`trajectory`、`unparsed`…)。
