"""把各競賽現有的 experiments.json 匯進本地 MLflow store。

達成計畫書 §4「以 MLflow 記錄實驗」,而**不重寫**既有自建 logger:權威紀錄仍是
experiments.json / facts / 報告管線;本腳本把它們**鏡射**進 MLflow(供 MLflow UI/查詢)。

相容兩種 schema:
  - 跨季(log_experiment_v2):cv_mean / cv_std / params / eval_metric
  - S3(早批):blend_score / blend_weights / per_model / metric

用法:uv run python3 docs/scripts/export_to_mlflow.py
本地 sqlite store 於 ./mlflow.db(gitignored、可重生;MLflow 3.x 建議的後端)。
唯讀 experiments.json,不改任何原檔。查看:uv run --with mlflow mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import glob
import json
import os

import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")


def score_of(e):
    for k in ("cv_mean", "blend_score", "cv_score", "score"):
        v = e.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def metric_name(e):
    return e.get("eval_metric") or e.get("metric") or "cv_score"


def main():
    n_comp = n_run = 0
    for exp_path in sorted(glob.glob(os.path.join(ROOT, "competitions", "*", "experiments.json"))):
        comp = os.path.basename(os.path.dirname(exp_path))
        try:
            d = json.load(open(exp_path))
        except Exception:
            continue
        entries = d if isinstance(d, list) else d.get("experiments", [])
        if not entries:
            continue
        mlflow.set_experiment(comp)
        n_comp += 1
        for e in entries:
            eid = str(e.get("experiment_id", "?"))
            with mlflow.start_run(run_name=f"{comp}-exp{eid}"):
                for k in ("model", "n_features", "cv", "cv_strategy", "metric",
                          "eval_metric", "submission"):
                    if e.get(k) is not None:
                        mlflow.log_param(k, str(e[k])[:250])
                if e.get("params"):
                    mlflow.log_param("params", str(e["params"])[:250])
                if e.get("blend_weights"):
                    mlflow.log_param("blend_weights", str(e["blend_weights"])[:250])
                if e.get("per_model"):
                    mlflow.log_param("per_model", str(list(e["per_model"].keys()))[:250])
                s = score_of(e)
                if s is not None:
                    mlflow.log_metric(metric_name(e), s)
                if e.get("cv_std") is not None:
                    try:
                        mlflow.log_metric("cv_std", float(e["cv_std"]))
                    except (ValueError, TypeError):
                        pass
                mlflow.set_tag("comp", comp)
                mlflow.set_tag("experiment_id", eid)
                if e.get("timestamp"):
                    mlflow.set_tag("timestamp", str(e["timestamp"]))
                if e.get("notes"):
                    mlflow.set_tag("notes", str(e["notes"])[:500])
                mlflow.set_tag("source", "experiments.json")
                n_run += 1
    print(f"MLflow: logged {n_run} runs across {n_comp} competitions to ./mlflow.db (sqlite)")
    return n_comp, n_run


if __name__ == "__main__":
    main()
