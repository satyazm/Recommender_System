"""Phase 1 profiling: RetailRocket events.csv

Answers the questions that should drive every downstream design choice:
- sparsity (drives CF method choice: dense ALS vs. approximate methods)
- interaction distribution / long-tail (drives popularity-prior weight, cold-start severity)
- time range (drives feasibility + boundaries of a time-based train/test split)
- funnel conversion (view -> cart -> purchase) since this dataset's whole value is having all three signals
"""
import numpy as np
import pandas as pd

DATA_DIR = "data/raw/retailrocket"
FIG_DIR = "reports/figures"

events = pd.read_csv(
    f"{DATA_DIR}/events.csv",
    dtype={"visitorid": "int32", "event": "category", "itemid": "int32"},
)
events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")

print("=" * 70)
print("RETAILROCKET — events.csv")
print("=" * 70)

n_events = len(events)
n_users = events["visitorid"].nunique()
n_items = events["itemid"].nunique()
print(f"\nTotal events:        {n_events:,}")
print(f"Unique visitors:     {n_users:,}")
print(f"Unique items:        {n_items:,}")
print(f"Date range:          {events['timestamp'].min()} -> {events['timestamp'].max()}")
print(f"Span:                {(events['timestamp'].max() - events['timestamp'].min()).days} days")

print("\n--- Event type breakdown (this is the funnel) ---")
counts = events["event"].value_counts()
for ev, c in counts.items():
    print(f"  {ev:12s} {c:>10,}  ({c / n_events:.2%})")

view_users = set(events.loc[events["event"] == "view", "visitorid"])
cart_users = set(events.loc[events["event"] == "addtocart", "visitorid"])
txn_users = set(events.loc[events["event"] == "transaction", "visitorid"])
print("\n--- Funnel conversion (user-level, not event-level) ---")
print(f"  view -> cart:      {len(cart_users & view_users) / len(view_users):.2%}")
print(f"  cart -> purchase:  {len(txn_users & cart_users) / max(len(cart_users), 1):.2%}")
print(f"  view -> purchase:  {len(txn_users & view_users) / len(view_users):.2%}")

# Sparsity based on UNIQUE (user, item) interaction pairs, not raw event count —
# raw event count double-counts repeat views of the same item by the same user.
unique_pairs = events.drop_duplicates(subset=["visitorid", "itemid"]).shape[0]
possible_pairs = n_users * n_items
sparsity = 1 - unique_pairs / possible_pairs
print(f"\n--- Sparsity ---")
print(f"  Unique (user,item) pairs: {unique_pairs:,}")
print(f"  Possible pairs:           {possible_pairs:,}")
print(f"  Sparsity:                 {sparsity:.6%}  (fraction of the user-item matrix that's empty)")

print("\n--- Per-user interaction count distribution ---")
user_counts = events.groupby("visitorid").size()
pct = user_counts.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(pct.to_string())
print(f"  mean: {user_counts.mean():.2f}  max: {user_counts.max()}")
print(f"  % users with exactly 1 event (cold-start-like): {(user_counts == 1).mean():.2%}")

print("\n--- Per-item interaction count distribution ---")
item_counts = events.groupby("itemid").size()
pct = item_counts.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(pct.to_string())
print(f"  mean: {item_counts.mean():.2f}  max: {item_counts.max()}")
print(f"  % items with < 5 events (cold-start items): {(item_counts < 5).mean():.2%}")

# Long-tail plot: item popularity rank (log) vs. interaction count (log)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sorted_counts = item_counts.sort_values(ascending=False).values
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts)
ax.set_xlabel("Item popularity rank (log)")
ax.set_ylabel("Interaction count (log)")
ax.set_title("RetailRocket: item popularity long tail")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/retailrocket_item_longtail.png", dpi=120)
print(f"\nSaved long-tail plot -> {FIG_DIR}/retailrocket_item_longtail.png")

# Events per day, to sanity-check time range is continuous (no big gaps) before we
# rely on it for a time-based split later.
daily = events.set_index("timestamp").resample("D").size()
fig, ax = plt.subplots(figsize=(8, 3))
daily.plot(ax=ax)
ax.set_title("RetailRocket: events per day")
ax.set_ylabel("events")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/retailrocket_events_per_day.png", dpi=120)
print(f"Saved events/day plot -> {FIG_DIR}/retailrocket_events_per_day.png")
