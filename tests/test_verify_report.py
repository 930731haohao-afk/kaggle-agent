import pytest

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
