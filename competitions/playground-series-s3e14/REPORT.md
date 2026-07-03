# 競賽分析報告:playground-series-s3e14

> 產生方式:kaggle-report skill(數字來自 facts.json,敘述由 agent 撰寫)
> 素材等級:baseline-only | 產生日期:2026-07-03

## 1. 競賽目的

**What**:本場競賽要求依野生藍莓田區的環境與授粉相關量測值,預測藍莓產量(`yield`)。這是一個
以連續數值輸出產量估計的迴歸問題。

**Why**:評估指標為 **MAE(平均絕對誤差,minimize)**。以絕對誤差的平均衡量預測誤差,單位與
產量本身一致,直觀反映「平均預測誤差是多少產量單位」,不會像平方誤差類指標般被少數極端樣本
過度放大,適合作為此類連續產量預測任務的評估基準。

| 項目 | 值 |
|------|-----|
| 競賽 | playground-series-s3e14 |
| 問題型別 | regression |
| 評估指標 | mae(minimize) |
| 目標欄位 | yield |

## 2. 資料規格

**本場僅執行通用基線,未執行 EDA,以下僅資料基本形狀**:

- 依 `facts.json`,本場僅有 1 筆實驗紀錄,來源格式為 `generic_batch`(通用批次管線),使用
  16 個特徵(來源:`experiments[0].n_features`)。
- 列數/欄位型別概述、缺漏值檢查等 EDA 產出:**無紀錄**(`facts.status_md_present = false`,
  無 `STATUS.md`,亦無 EDA 腳本產出可供引用)。
- 特別規則:不可使用外部資料、不可使用預訓練模型、不可存取網路;每日提交上限 5 次(來源:
  `competition.special_rules`)。

## 3. 模型規格

本場僅有 1 筆實驗紀錄(`generic_batch` 通用批次管線),未執行任何額外的模型選型比較或調參:

| Model | OOF MAE |
|-------|---------|
| LGB | 345.32118 |
| XGB | 344.13027 |
| CAT | 344.24276 |

Ensemble 權重(來源:`experiments[0].ensemble.weights`):LGB 0.3 / XGB 0.3 / CAT 0.4,
Ensemble 分數(來源:`experiments[0].ensemble.score`):**341.40782**。

**選型理由**:此為通用批次管線(`run_competition.py`)對三個梯度提升樹模型(LightGBM、
XGBoost、CatBoost)的預設加權集成,並未針對本場資料進行特徵工程或超參數調整,選型理由僅為
管線預設策略,非本場專屬分析。

## 4. 訓練規格

| 實驗 | CV scheme | n_splits | seed |
|------|-----------|----------|------|
| 1 | 5fold | 5 | 無紀錄 |

(來源:`experiments[0].cv`;`facts.missing` 列出 `cv_seed`,故 seed 寫「無紀錄」。)

各 base model 之 objective/超參數在 `experiments[0].base_models[]` 中未記錄對應欄位,故此項為
「無紀錄」。

**為何用此 CV**:5-fold 為通用批次管線之預設交叉驗證切法,並非針對本場資料特性(如目標分布、
分層需求)特別設計或調整,僅為系統性套用之基準設定。

## 5. 推論程序

依 `facts.best`(即本場唯一實驗)之 `postprocess` 欄位未記錄任何後處理步驟,故此欄寫
「無後處理紀錄」。其 submission 檔名為 `sub_generic_341.40782_20260703_121146.csv`(來源:
`experiments[0].submission`),對應 `id_column = id`、`target_column = yield`。

## 6. 評估指標

**指標定義**:MAE(Mean Absolute Error)= 預測值與真實值絕對差的平均,單位與目標欄位相同
(此處為產量)。

| 項目 | 分數 |
|------|------|
| Ensemble(OOF MAE) | 341.40782 |
| Public LB | 無紀錄 |
| Private LB | 無紀錄 |

(來源:`experiments[0].ensemble.score`;`facts.leaderboard` 為 `null`,`facts.missing` 列出
`leaderboard`,顯示本場未提交至 Kaggle 排行榜,故無法計算 CV↔LB gap。)

## 7. 實驗軌跡

| experiment_id | timestamp | score | source_format |
|---------------|-----------|-------|----------------|
| 1 | 2026-07-03T12:11:46 | 341.40782 | generic_batch |

（來源：`facts.trajectory`）

**突破點**:本場僅有 1 筆實驗紀錄,為通用批次管線之單次基準跑,無多筆實驗可比較,故無分數躍升
可供描述。

`facts.unparsed` 為空陣列,無法解析之紀錄:無。

## 8. 重現指令

```bash
cd /home/tjyen/ai_agents/kaggle

# 通用批次管線(唯一執行過的流程;本場未執行 EDA / 特徵工程腳本)
uv run python3 competitions/run_competition.py playground-series-s3e14
```

執行目錄為專案根目錄 `/home/tjyen/ai_agents/kaggle`;執行透過 `uv run` 以確保套件環境一致。
本場未提交至 Kaggle 排行榜,故無對應的 `kaggle competitions submit` 指令可重現。
