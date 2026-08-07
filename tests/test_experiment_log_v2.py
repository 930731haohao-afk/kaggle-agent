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


def test_all_duplicated_utils_identical():
    """EVERY duplicated util must be byte-identical across all three locations.

    This test used to cover experiment_log only, across the two skill copies only — so the
    2026-08-03 dtype-consistency fix in utils/data_loader.py never reached the copies the
    agent actually runs, and two fix rounds' worth of experiment_log repairs sat in utils/
    while the live copies stayed broken (2026-08-07 re-verification). One test, every file,
    every location, including the repo utils/ as the source of truth.
    """
    duplicated = ["experiment_log.py", "data_loader.py", "evaluation.py"]
    roots = [ROOT / "utils",
             ROOT / ".claude/skills/kaggle-agent/assets/utils",
             ROOT / ".claude/skills/kaggle-agent-self-improvement/assets/utils"]
    for fname in duplicated:
        variants = {}
        for r in roots:
            fp = r / fname
            if fp.exists():
                variants.setdefault(fp.read_bytes(), []).append(str(fp.relative_to(ROOT)))
        assert len(variants) == 1, (
            f"{fname} has {len(variants)} distinct versions: "
            f"{[v for v in variants.values()]} — 改完要 cp 同步到三份")
