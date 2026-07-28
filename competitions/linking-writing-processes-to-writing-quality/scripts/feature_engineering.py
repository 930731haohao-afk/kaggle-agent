"""Feature engineering for Linking Writing Processes to Writing Quality.

Aggregates raw keystroke logs (~3400 events per essay) into essay-level features.
"""
import numpy as np
import pandas as pd

data_dir = "competitions/linking-writing-processes-to-writing-quality/data"


def engineer_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Engineer essay-level features from keystroke logs."""
    logs = logs.sort_values(["id", "event_id"]).copy()

    # Pre-compute useful columns
    logs["iki"] = logs.groupby("id")["down_time"].diff()
    logs["is_input"] = logs["activity"] == "Input"
    logs["is_remove"] = logs["activity"].str.startswith("Remove")
    logs["is_nonproduction"] = logs["activity"] == "Nonproduction"
    logs["is_backspace"] = logs["down_event"].isin(["Backspace", "Delete"])
    logs["is_space"] = logs["down_event"] == "Space"
    logs["is_enter"] = logs["down_event"] == "Enter"
    logs["is_shift"] = logs["down_event"] == "Shift"
    logs["is_arrow"] = logs["down_event"].isin(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"])
    logs["is_click"] = logs["down_event"].isin(["Leftclick", "Rightclick"])
    logs["is_punctuation"] = logs["down_event"].isin([".", ",", "?", "!", ";", ":", "'", '"'])
    logs["is_burst"] = logs["iki"] < 500
    logs["is_long_pause"] = logs["iki"] > 2000
    logs["is_short_pause"] = (logs["iki"] > 1000) & (logs["iki"] <= 2000)

    features = []

    for essay_id, g in logs.groupby("id"):
        f = {"id": essay_id}

        # === LENGTH FEATURES ===
        f["n_events"] = len(g)
        f["final_word_count"] = g["word_count"].max()
        f["max_cursor_pos"] = g["cursor_position"].max()

        # === TIMING FEATURES ===
        total_time_ms = g["down_time"].max() - g["down_time"].min()
        f["total_time_s"] = total_time_ms / 1000
        total_time_min = total_time_ms / 60000
        f["wpm"] = f["final_word_count"] / max(total_time_min, 0.01)
        f["events_per_min"] = f["n_events"] / max(total_time_min, 0.01)

        # === IKI (INTER-KEY INTERVAL) FEATURES ===
        iki = g["iki"].dropna()
        if len(iki) > 0:
            f["iki_mean"] = iki.mean()
            f["iki_median"] = iki.median()
            f["iki_std"] = iki.std()
            f["iki_p10"] = iki.quantile(0.1)
            f["iki_p25"] = iki.quantile(0.25)
            f["iki_p75"] = iki.quantile(0.75)
            f["iki_p90"] = iki.quantile(0.9)
            f["iki_iqr"] = f["iki_p75"] - f["iki_p25"]
            f["iki_skew"] = iki.skew()
        else:
            for k in ["iki_mean", "iki_median", "iki_std", "iki_p10", "iki_p25",
                       "iki_p75", "iki_p90", "iki_iqr", "iki_skew"]:
                f[k] = 0

        # === ACTION TIME (key hold duration) ===
        at = g["action_time"]
        f["action_time_mean"] = at.mean()
        f["action_time_median"] = at.median()
        f["action_time_std"] = at.std()

        # === ACTIVITY COUNTS ===
        f["n_input"] = g["is_input"].sum()
        f["n_remove"] = g["is_remove"].sum()
        f["n_nonproduction"] = g["is_nonproduction"].sum()
        f["input_rate"] = f["n_input"] / f["n_events"]
        f["remove_rate"] = f["n_remove"] / f["n_events"]
        f["nonproduction_rate"] = f["n_nonproduction"] / f["n_events"]

        # === REVISION FEATURES ===
        f["n_backspace"] = g["is_backspace"].sum()
        f["backspace_rate"] = f["n_backspace"] / f["n_events"]
        # Net efficiency: how much of the input was kept
        f["net_efficiency"] = f["max_cursor_pos"] / max(f["n_input"], 1)

        # === NAVIGATION FEATURES ===
        f["n_arrows"] = g["is_arrow"].sum()
        f["n_clicks"] = g["is_click"].sum()
        f["arrow_rate"] = f["n_arrows"] / f["n_events"]
        f["click_rate"] = f["n_clicks"] / f["n_events"]

        # === SPECIAL KEY FEATURES ===
        f["n_spaces"] = g["is_space"].sum()
        f["n_enters"] = g["is_enter"].sum()
        f["n_shifts"] = g["is_shift"].sum()
        f["n_punctuation"] = g["is_punctuation"].sum()
        f["shift_rate"] = f["n_shifts"] / f["n_events"]
        f["punctuation_rate"] = f["n_punctuation"] / f["n_events"]

        # === PAUSE FEATURES ===
        f["n_long_pauses"] = g["is_long_pause"].sum()
        f["n_short_pauses"] = g["is_short_pause"].sum()
        f["long_pause_rate"] = f["n_long_pauses"] / f["n_events"]
        # Total time in pauses > 2s
        long_pauses = iki[iki > 2000]
        f["long_pause_total_s"] = long_pauses.sum() / 1000 if len(long_pauses) > 0 else 0
        f["long_pause_mean_s"] = long_pauses.mean() / 1000 if len(long_pauses) > 0 else 0
        # Pause to active ratio
        active_time = iki[iki <= 2000].sum()
        f["pause_to_active_ratio"] = long_pauses.sum() / max(active_time, 1)

        # === BURST FEATURES ===
        burst_mask = g["is_burst"].values
        bursts = []
        current_len = 0
        for b in burst_mask:
            if b:
                current_len += 1
            else:
                if current_len > 0:
                    bursts.append(current_len)
                current_len = 0
        if current_len > 0:
            bursts.append(current_len)

        if bursts:
            f["n_bursts"] = len(bursts)
            f["mean_burst_len"] = np.mean(bursts)
            f["max_burst_len"] = max(bursts)
            f["std_burst_len"] = np.std(bursts)
            f["median_burst_len"] = np.median(bursts)
        else:
            f["n_bursts"] = 0
            f["mean_burst_len"] = 0
            f["max_burst_len"] = 0
            f["std_burst_len"] = 0
            f["median_burst_len"] = 0

        # Words per burst
        f["words_per_burst"] = f["final_word_count"] / max(f["n_bursts"], 1)

        # === WRITING PHASE FEATURES (split into time quartiles) ===
        time_range = g["down_time"].max() - g["down_time"].min()
        if time_range > 0:
            g_copy = g.copy()
            g_copy["time_frac"] = (g_copy["down_time"] - g_copy["down_time"].min()) / time_range
            for qi, (lo, hi) in enumerate([(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]):
                phase = g_copy[(g_copy["time_frac"] >= lo) & (g_copy["time_frac"] < hi + 0.001)]
                f[f"phase_{qi}_events"] = len(phase)
                f[f"phase_{qi}_input_rate"] = phase["is_input"].mean() if len(phase) > 0 else 0
                phase_iki = phase["iki"].dropna()
                f[f"phase_{qi}_iki_mean"] = phase_iki.mean() if len(phase_iki) > 0 else 0
                # Word count change in this phase
                f[f"phase_{qi}_wc_delta"] = phase["word_count"].max() - phase["word_count"].min() if len(phase) > 0 else 0
        else:
            for qi in range(4):
                f[f"phase_{qi}_events"] = 0
                f[f"phase_{qi}_input_rate"] = 0
                f[f"phase_{qi}_iki_mean"] = 0
                f[f"phase_{qi}_wc_delta"] = 0

        # === SENTENCE/PARAGRAPH INDICATORS ===
        # Periods followed by space = approximate sentence count
        text_changes = g["text_change"].values
        f["n_paragraphs"] = f["n_enters"]
        # Approximate sentence count from punctuation
        f["approx_sentences"] = f["n_punctuation"]
        f["words_per_sentence"] = f["final_word_count"] / max(f["approx_sentences"], 1)

        features.append(f)

    return pd.DataFrame(features)


# ============================================================
# PROCESS TRAIN
# ============================================================
print("Loading train logs...")
train_logs = pd.read_csv(f"{data_dir}/train_logs.csv")
scores = pd.read_csv(f"{data_dir}/train_scores.csv")
print(f"Train logs: {train_logs.shape}")

print("Engineering train features...")
X_train = engineer_features(train_logs)
X_train = X_train.merge(scores, on="id")
print(f"Train features: {X_train.shape}")

# ============================================================
# PROCESS TEST
# ============================================================
print("\nLoading test logs...")
test_logs = pd.read_csv(f"{data_dir}/test_logs.csv")
print(f"Test logs: {test_logs.shape}")

print("Engineering test features...")
X_test = engineer_features(test_logs)
print(f"Test features: {X_test.shape}")

# ============================================================
# VALIDATION
# ============================================================
feature_cols = [c for c in X_train.columns if c not in ["id", "score"]]
print(f"\n=== VALIDATION ===")
print(f"Feature count: {len(feature_cols)}")

# Check columns match (excluding target)
test_feature_cols = [c for c in X_test.columns if c != "id"]
assert set(feature_cols) == set(test_feature_cols), f"Column mismatch: {set(feature_cols) - set(test_feature_cols)}"
print("Column match: OK")

# NaN check
train_nans = X_train[feature_cols].isnull().sum()
if train_nans.sum() > 0:
    print(f"NaN in train:\n{train_nans[train_nans > 0]}")
else:
    print("No NaN in train: OK")

test_nans = X_test[test_feature_cols].isnull().sum()
if test_nans.sum() > 0:
    print(f"NaN in test:\n{test_nans[test_nans > 0]}")
else:
    print("No NaN in test: OK")

# Save
X_train.to_parquet(f"{data_dir}/train_processed.parquet", index=False)
X_test.to_parquet(f"{data_dir}/test_processed.parquet", index=False)
print(f"\nSaved train_processed.parquet ({X_train.shape})")
print(f"Saved test_processed.parquet ({X_test.shape})")

# ============================================================
# QUICK FEATURE IMPORTANCE
# ============================================================
print("\n=== QUICK FEATURE IMPORTANCE (LightGBM) ===")
import lightgbm as lgb

target = X_train["score"]
dtrain = lgb.Dataset(X_train[feature_cols], label=target)

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": 42,
}
model = lgb.train(params, dtrain, num_boost_round=300)

importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
importance = importance.sort_values(ascending=False)

print(f"\nTop 30 features by importance (gain):")
for i, (feat, imp) in enumerate(importance.head(30).items()):
    bar = "#" * int(imp / importance.max() * 40)
    print(f"  {i+1:2d}. {feat:30s}: {imp:10.1f} {bar}")

zero_imp = importance[importance == 0]
if len(zero_imp) > 0:
    print(f"\nZero importance features ({len(zero_imp)}): {list(zero_imp.index)}")
else:
    print("\nNo zero-importance features.")

print(f"\nTotal features: {len(feature_cols)}")
