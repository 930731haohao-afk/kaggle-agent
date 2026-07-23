import json

import pytest

COLLECT = ".claude/skills/kaggle-mlspec-report/assets/collect.py"

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


def test_skill_train_base_model_without_oof_key_yields_none_score(load_module, comp_dir):
    """base_models 中若某筆缺 oof_* 鍵,不應讓 build_facts 炸出 StopIteration,score 應為 None。"""
    c = load_module(COLLECT, "collect")
    entry = dict(SKILL_TRAIN_ENTRY)
    entry["base_models"] = [
        {"model": "LGB", "oof_mae": 1.35651, "time_s": 30.8},
        {"model": "MYSTERY", "time_s": 12.0},  # 缺 oof_* 鍵
    ]
    _write_exps(comp_dir, [entry])
    facts = c.build_facts(str(comp_dir))
    e = facts["experiments"][0]
    scores_by_name = {b["name"]: b["score"] for b in e["base_models"]}
    assert scores_by_name["LGB"] == 1.35651
    assert scores_by_name["MYSTERY"] is None
