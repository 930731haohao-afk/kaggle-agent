"""tests/fixtures/dummy_eval_module.py — a minimal stand-in for a per-comp eval_*.py
module, used ONLY by test_tree_harness_v3.py's `eval_solo_subprocess` tests (feature 8).
Implements the same `evaluate(config, node_id=None, timeout_s=...) -> dict` contract
every real eval_*.py in tree_search/ implements, without any of the heavy ML
dependencies (lgb/xgb/cat) -- deliberately cheap and fast so the subprocess-timeout test
suite runs in milliseconds instead of minutes.

`config["sleep_s"]` (default 0) controls how long `evaluate()` sleeps before returning,
so a test can simulate a "hung native fit" by passing a sleep longer than the timeout
passed to `eval_solo_subprocess` -- the parent's hard SIGKILL is what must end it, not
this module's own cooperation (this module does NOT self-interrupt on `timeout_s`,
mirroring exactly the SIGALRM-can't-interrupt-native-code failure mode being tested).
`config["raise"]` (default False), if true, raises an exception instead of returning.
"""
import time


def evaluate(config: dict, node_id: int = None, timeout_s: float = 200) -> dict:
    sleep_s = config.get("sleep_s", 0)
    if config.get("raise"):
        raise RuntimeError("dummy_eval_module: deliberate failure for testing")
    t0 = time.time()
    time.sleep(sleep_s)
    score = config.get("score", 1.0)
    return dict(status="evaluated", score=score, wall_s=round(time.time() - t0, 2),
                result={"node_id": node_id, "sleep_s": sleep_s}, error=None)
