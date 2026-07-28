"""EDA for Linking Writing Processes to Writing Quality."""
import pandas as pd
import numpy as np

data_dir = "competitions/linking-writing-processes-to-writing-quality/data"

scores = pd.read_csv(f"{data_dir}/train_scores.csv")
# Read full train logs
logs = pd.read_csv(f"{data_dir}/train_logs.csv")

print(f"Logs shape: {logs.shape}")
print(f"Scores shape: {scores.shape}")

# ============================================================
# 1. TARGET ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("1. TARGET ANALYSIS")
print("=" * 60)
print(f"\nScore distribution:")
print(scores["score"].describe())
print(f"\nSkewness: {scores['score'].skew():.3f}")
print(f"\nValue counts:")
for s in sorted(scores["score"].unique()):
    n = (scores["score"] == s).sum()
    bar = "#" * (n // 5)
    print(f"  {s:4.1f}: {n:4d} ({n/len(scores)*100:5.1f}%) {bar}")

# ============================================================
# 2. PER-ESSAY AGGREGATION (basic stats)
# ============================================================
print("\n" + "=" * 60)
print("2. PER-ESSAY BASIC STATS")
print("=" * 60)

essay_stats = logs.groupby("id").agg(
    n_events=("event_id", "count"),
    total_time_ms=("down_time", lambda x: x.max() - x.min()),
    final_word_count=("word_count", "max"),
    max_cursor_pos=("cursor_position", "max"),
).reset_index()
essay_stats = essay_stats.merge(scores, on="id")

print(f"\nEvents per essay:")
print(essay_stats["n_events"].describe())
print(f"\nTotal writing time (seconds):")
print((essay_stats["total_time_ms"] / 1000).describe())
print(f"\nFinal word count:")
print(essay_stats["final_word_count"].describe())

# Correlation with score
print(f"\nCorrelation with score:")
for col in ["n_events", "total_time_ms", "final_word_count", "max_cursor_pos"]:
    corr = essay_stats[col].corr(essay_stats["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 3. ACTIVITY TYPE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. ACTIVITY TYPE ANALYSIS")
print("=" * 60)

# Classify activities
logs["activity_type"] = logs["activity"].apply(
    lambda x: "Input" if x == "Input" else
              "Remove" if x.startswith("Remove") else
              "Nonproduction" if x == "Nonproduction" else
              "Replace" if x == "Replace" else
              "Paste" if x == "Paste" else
              "Move" if x.startswith("Move") else "Other"
)

activity_per_essay = logs.groupby(["id", "activity_type"]).size().unstack(fill_value=0)
activity_per_essay = activity_per_essay.div(activity_per_essay.sum(axis=1), axis=0)  # proportions
activity_per_essay = activity_per_essay.merge(scores, left_index=True, right_on="id")

print("\nActivity proportion correlation with score:")
for col in activity_per_essay.columns:
    if col not in ["id", "score"]:
        corr = activity_per_essay[col].corr(activity_per_essay["score"])
        print(f"  {col}: {corr:.3f}")

# ============================================================
# 4. TYPING SPEED & PAUSE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. TYPING SPEED & PAUSES")
print("=" * 60)

# Inter-key interval (IKI) - time between consecutive keystrokes
logs_sorted = logs.sort_values(["id", "event_id"])
logs_sorted["iki"] = logs_sorted.groupby("id")["down_time"].diff()

# Per-essay typing speed stats
iki_stats = logs_sorted.groupby("id")["iki"].agg(
    iki_mean="mean",
    iki_median="median",
    iki_std="std",
    iki_p10=lambda x: x.quantile(0.1),
    iki_p90=lambda x: x.quantile(0.9),
).reset_index()
iki_stats = iki_stats.merge(scores, on="id")

print("\nInter-key interval (ms):")
print(f"  Mean IKI across essays: {iki_stats['iki_mean'].mean():.0f} ms")
print(f"  Median IKI across essays: {iki_stats['iki_median'].mean():.0f} ms")

print(f"\nIKI correlation with score:")
for col in ["iki_mean", "iki_median", "iki_std", "iki_p10", "iki_p90"]:
    corr = iki_stats[col].corr(iki_stats["score"])
    print(f"  {col}: {corr:.3f}")

# Pause analysis (IKI > 2000ms = thinking pause)
pause_thresholds = [1000, 2000, 5000, 10000]
for thresh in pause_thresholds:
    pause_count = logs_sorted.groupby("id")["iki"].apply(lambda x: (x > thresh).sum())
    pause_count = pause_count.reset_index()
    pause_count.columns = ["id", "n_pauses"]
    pause_count = pause_count.merge(scores, on="id")
    corr = pause_count["n_pauses"].corr(pause_count["score"])
    print(f"  Pauses > {thresh}ms: mean={pause_count['n_pauses'].mean():.1f}, corr={corr:.3f}")

# ============================================================
# 5. REVISION BEHAVIOR
# ============================================================
print("\n" + "=" * 60)
print("5. REVISION BEHAVIOR")
print("=" * 60)

# Count backspaces and deletes per essay
logs["is_backspace"] = logs["down_event"].isin(["Backspace", "Delete"])
logs["is_remove"] = logs["activity_type"] == "Remove"

revision_stats = logs.groupby("id").agg(
    n_backspace=("is_backspace", "sum"),
    n_remove_events=("is_remove", "sum"),
    n_total=("event_id", "count"),
).reset_index()
revision_stats["backspace_rate"] = revision_stats["n_backspace"] / revision_stats["n_total"]
revision_stats["remove_rate"] = revision_stats["n_remove_events"] / revision_stats["n_total"]
revision_stats = revision_stats.merge(scores, on="id")

print("\nRevision stats:")
print(f"  Backspace rate: mean={revision_stats['backspace_rate'].mean():.3f}")
print(f"  Remove rate: mean={revision_stats['remove_rate'].mean():.3f}")

print(f"\nCorrelation with score:")
for col in ["n_backspace", "backspace_rate", "n_remove_events", "remove_rate"]:
    corr = revision_stats[col].corr(revision_stats["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 6. NAVIGATION BEHAVIOR
# ============================================================
print("\n" + "=" * 60)
print("6. NAVIGATION BEHAVIOR")
print("=" * 60)

logs["is_arrow"] = logs["down_event"].isin(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"])
logs["is_click"] = logs["down_event"].isin(["Leftclick", "Rightclick"])

nav_stats = logs.groupby("id").agg(
    n_arrows=("is_arrow", "sum"),
    n_clicks=("is_click", "sum"),
    n_total=("event_id", "count"),
).reset_index()
nav_stats["arrow_rate"] = nav_stats["n_arrows"] / nav_stats["n_total"]
nav_stats["click_rate"] = nav_stats["n_clicks"] / nav_stats["n_total"]
nav_stats = nav_stats.merge(scores, on="id")

print("\nNavigation stats:")
print(f"  Arrow key rate: mean={nav_stats['arrow_rate'].mean():.4f}")
print(f"  Click rate: mean={nav_stats['click_rate'].mean():.4f}")

print(f"\nCorrelation with score:")
for col in ["n_arrows", "arrow_rate", "n_clicks", "click_rate"]:
    corr = nav_stats[col].corr(nav_stats["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 7. WRITING PRODUCTIVITY
# ============================================================
print("\n" + "=" * 60)
print("7. WRITING PRODUCTIVITY")
print("=" * 60)

# Words per minute, characters per minute
prod = essay_stats[["id", "total_time_ms", "final_word_count", "n_events", "score"]].copy()
prod["total_time_min"] = prod["total_time_ms"] / 60000
prod["wpm"] = prod["final_word_count"] / prod["total_time_min"]
prod["events_per_min"] = prod["n_events"] / prod["total_time_min"]

print(f"\nWords per minute:")
print(prod["wpm"].describe())
print(f"\nEvents per minute:")
print(prod["events_per_min"].describe())

print(f"\nCorrelation with score:")
for col in ["wpm", "events_per_min", "total_time_min"]:
    corr = prod[col].corr(prod["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 8. BURST ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("8. BURST ANALYSIS (typing bursts)")
print("=" * 60)

# A burst = consecutive keystrokes with IKI < 500ms
BURST_THRESHOLD = 500  # ms
logs_sorted["is_burst"] = logs_sorted["iki"] < BURST_THRESHOLD

# Count bursts and their lengths per essay
def burst_stats(group):
    is_burst = group["is_burst"].values
    bursts = []
    current_len = 0
    for b in is_burst:
        if b:
            current_len += 1
        else:
            if current_len > 0:
                bursts.append(current_len)
            current_len = 0
    if current_len > 0:
        bursts.append(current_len)
    if not bursts:
        return pd.Series({"n_bursts": 0, "mean_burst_len": 0, "max_burst_len": 0})
    return pd.Series({
        "n_bursts": len(bursts),
        "mean_burst_len": np.mean(bursts),
        "max_burst_len": max(bursts),
    })

burst_df = logs_sorted.groupby("id").apply(burst_stats, include_groups=False).reset_index()
burst_df = burst_df.merge(scores, on="id")

print(f"\nBurst stats:")
print(f"  N bursts per essay: mean={burst_df['n_bursts'].mean():.0f}")
print(f"  Mean burst length: {burst_df['mean_burst_len'].mean():.1f} events")
print(f"  Max burst length: mean={burst_df['max_burst_len'].mean():.0f}")

print(f"\nCorrelation with score:")
for col in ["n_bursts", "mean_burst_len", "max_burst_len"]:
    corr = burst_df[col].corr(burst_df["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 9. ACTION TIME ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("9. ACTION TIME (key hold duration)")
print("=" * 60)

action_stats = logs.groupby("id")["action_time"].agg(
    action_mean="mean",
    action_median="median",
    action_std="std",
).reset_index()
action_stats = action_stats.merge(scores, on="id")

print(f"\nAction time (ms):")
print(f"  Mean: {action_stats['action_mean'].mean():.1f}")
print(f"  Median: {action_stats['action_median'].mean():.1f}")

print(f"\nCorrelation with score:")
for col in ["action_mean", "action_median", "action_std"]:
    corr = action_stats[col].corr(action_stats["score"])
    print(f"  {col}: {corr:.3f}")

# ============================================================
# 10. SCORE BY ESSAY LENGTH BUCKETS
# ============================================================
print("\n" + "=" * 60)
print("10. SCORE BY WORD COUNT BUCKETS")
print("=" * 60)

essay_stats["wc_bucket"] = pd.cut(essay_stats["final_word_count"],
                                   bins=[0, 200, 300, 400, 500, 800],
                                   labels=["<200", "200-300", "300-400", "400-500", "500+"])
bucket_stats = essay_stats.groupby("wc_bucket", observed=True)["score"].agg(["mean", "std", "count"])
print(bucket_stats)

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
