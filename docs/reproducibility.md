# 可重現性(對照計畫書 §4「模組化與可重現」+ 目標五)

> 對照《暑期實習計畫》§4 與目標五,說明本專案的可重現機制、與計畫書的對映,以及
> 「可重現標準於專案中途提高」的誠實分層。日期:2026-07-08。

## 計畫書要求 ↔ 現況

| 計畫書 §4「模組化與可重現」 | 現況 |
|---|---|
| 固定 torch/numpy/scikit-learn 隨機種子 | ✅ 全場 **seed=42**(模型 `random_state`、`KFold(shuffle, seed=42)`、numpy `RandomState(42)`)。本專案表格題以 GBDT(LGB/XGB/CAT)為主,亦固定;torch 少用但同原則 |
| 以 uv 管理套件 | ✅ `pyproject.toml` + `uv.lock`;`uv sync` 重建環境 |
| 以 MLflow 記錄實驗 | ✅ `docs/scripts/export_to_mlflow.py` 把各場 `experiments.json` 鏡射進本地 **sqlite MLflow store**(`mlflow.db`;284 runs/38 場)。權威紀錄仍為 experiments.json / facts / 報告管線,MLflow 為對應的標準化查詢層 |
| 沿用 data/models/trainers/utils 結構 | 等效、命名不同(對映見下) |

## 結構對映(data/models/trainers/utils ↔ 實際佈局)

| 計畫書模組 | 本專案實際位置 |
|---|---|
| **data** | `competitions/<comp>/data`(gitignored 原始資料)+ `config.yaml`(規格) |
| **models / trainers** | `competitions/<comp>/scripts`(`features.py` + `train*.py` / `04_train_blend.py` / `05_iterate.py`);`tree_search/`(`harness_v2/v3` + `run_*_v3.py` 樹搜尋 driver) |
| **utils** | `utils/`(data_loader / evaluation / experiment_log)、`templates/`、`.claude/skills/kaggle-report/assets`(collect / verify / md2pdf 報告管線) |

## 可重現分層(誠實記錄:標準於專案中途提高)

有兩個嚴格度,都是真的、都誠實記錄:

**層一(計畫書標準)——全 15 場**:固定種子 + 固定折 + 數字可追溯(verify 閘門)+ OOF 快取
+ uv 鎖環境 + MLflow 記錄。S3(早批 10 場)做在此層,全 CV-only(競賽關閉、不提交)。

**層二(額外的逐位決定性)——跨季 5 場**:層一 +「重訓能逐位吐出同一份 OOF」的閘門
(pin LightGBM `deterministic`/`force_row_wise`/`num_threads`)。
- 為**認證提交檔**而引入(重訓的 test 預測要對得上驗證過的 OOF);過程中發現並修掉 LGBM
  跨進程不決定性坑(見記憶 lgbm-determinism-oof-gate)。
- 此層**超出計畫書要求**——計畫書要「固定種子」,而**固定種子 ≠ 逐位決定性**:LGBM 那個坑
  來自臨時計時選 row/col-wise + 執行緒累加順序,固定種子也躲不掉。

**S3 為何不回頭補到層二**:S3 全 CV-only、不提交,層二的目的(認證提交)對它不適用;S3 已
滿足計畫書標準(層一)。保留為專案演進的合理進步、誠實記錄,不重訓。(定案 2026-07-08,
見 `docs/plan_compliance_audit.md`。)

## 如何重現

```bash
uv sync                        # 依 uv.lock 重建環境
# 各場逐步重現指令見 competitions/<comp>/REPORT.md 第 7 節
# 交付層一鍵建置+自檢見 setup.sh / REPRODUCE.md
uv run python3 docs/scripts/export_to_mlflow.py                       # 重生 MLflow 紀錄
uv run --with mlflow mlflow ui --backend-store-uri sqlite:///mlflow.db  # 開 MLflow UI 查看
```
