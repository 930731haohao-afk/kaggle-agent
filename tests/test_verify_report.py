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


def test_thousands_separator_normalized(load_module):
    v = load_module(VERIFY, "verify")
    # 74,051 是千分位寫法,facts 存 74051 → 應可回溯,不得被切成 051 誤報
    assert v.find_suspects("資料共 74,051 筆", {"rows": 74051}) == []


def test_thousands_separator_in_fact_strings(load_module):
    v = load_module(VERIFY, "verify")
    # facts 字串內的千分位數字也要正規化後才收集
    assert v.find_suspects("樣本數 74051", {"note": "count: 74,051 rows"}) == []


def test_negative_number_matches(load_module):
    v = load_module(VERIFY, "verify")
    # 報告中的 -0.234 應與 facts 的 -0.234 對上,負號不可被丟棄
    assert v.find_suspects("分數變化 -0.234(改善)", {"delta": -0.234}) == []


def test_date_dash_is_not_negative_sign(load_module):
    v = load_module(VERIFY, "verify")
    # 2026-07-03 的 dash 是日期分隔,不是負號;2026 年份豁免、07/03 不足三位
    assert v.find_suspects("執行日期 2026-07-03", {"x": 1}) == []


def test_hyphen_after_word_is_not_negative_sign(load_module):
    v = load_module(VERIFY, "verify")
    # sub-20260703 的 hyphen 接在字母後是分隔符;20260703 仍應以正數比對成功
    assert v.find_suspects("檔名 sub-20260703", FACTS) == []
