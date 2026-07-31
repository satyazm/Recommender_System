"""Phase 1 profiling: H&M transactions_train.csv

Same questions as RetailRocket (sparsity, distribution, time range), plus two
things specific to H&M's shape:
- it's purchases only (no view/cart funnel) — quantify repeat-purchase rate
  since that's the only "engagement intensity" signal available here
- customers.csv / articles.csv contain users/items with ZERO transactions —
  that's a naturally occurring cold-start population worth sizing
"""
import numpy as np
import pandas as pd

DATA_DIR = "data/raw/hm"
FIG_DIR = "reports/figures"

# customer_id is a 64-char hex string repeated 31M times — read as category to avoid
# materializing millions of duplicate Python str objects. article_id fits in int32.
txns = pd.read_csv(
    f"{DATA_DIR}/transactions_train.csv",
    dtype={"customer_id": "category", "article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    parse_dates=["t_dat"],
)

print("=" * 70)
print("H&M — transactions_train.csv")
print("=" * 70)

n_txns = len(txns)
n_users = txns["customer_id"].nunique()
n_items = txns["article_id"].nunique()
print(f"\nTotal transactions:  {n_txns:,}")
print(f"Unique customers:    {n_users:,}")
print(f"Unique articles:     {n_items:,}")
print(f"Date range:          {txns['t_dat'].min().date()} -> {txns['t_dat'].max().date()}")
print(f"Span:                {(txns['t_dat'].max() - txns['t_dat'].min()).days} days")

# No view/cart events here — purchases are the only signal. Repeat-purchase rate is
# the closest thing to an "engagement intensity" signal we get.
pair_counts = txns.groupby(["customer_id", "article_id"], observed=True).size()
repeat_rate = (pair_counts > 1).mean()
print(f"\n--- Repeat purchases (only engagement signal available, no views/carts) ---")
print(f"  % of (customer,article) pairs bought more than once: {repeat_rate:.2%}")

unique_pairs = len(pair_counts)
possible_pairs = n_users * n_items
sparsity = 1 - unique_pairs / possible_pairs
print(f"\n--- Sparsity ---")
print(f"  Unique (customer,article) pairs: {unique_pairs:,}")
print(f"  Possible pairs:                  {possible_pairs:,}")
print(f"  Sparsity:                        {sparsity:.6%}")

print("\n--- Per-customer purchase count distribution ---")
user_counts = txns.groupby("customer_id", observed=True).size()
pct = user_counts.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(pct.to_string())
print(f"  mean: {user_counts.mean():.2f}  max: {user_counts.max()}")
print(f"  % customers with exactly 1 purchase: {(user_counts == 1).mean():.2%}")

print("\n--- Per-article purchase count distribution ---")
item_counts = txns.groupby("article_id", observed=True).size()
pct = item_counts.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
print(pct.to_string())
print(f"  mean: {item_counts.mean():.2f}  max: {item_counts.max()}")
print(f"  % articles with < 5 purchases: {(item_counts < 5).mean():.2%}")

# Naturally-occurring cold start: customers/articles in the master files that never
# transacted. This is the real cold-start population your fallback logic has to handle.
customers = pd.read_csv(f"{DATA_DIR}/customers.csv", usecols=["customer_id"], dtype={"customer_id": "category"})
articles = pd.read_csv(f"{DATA_DIR}/articles.csv", usecols=["article_id"], dtype={"article_id": "int32"})
cold_customers = customers["customer_id"].nunique() - n_users
cold_articles = articles["article_id"].nunique() - n_items
print(f"\n--- Naturally occurring cold start ---")
print(f"  Customers in customers.csv with 0 transactions: {cold_customers:,} "
      f"({cold_customers / customers['customer_id'].nunique():.2%} of all customers)")
print(f"  Articles in articles.csv with 0 transactions:   {cold_articles:,} "
      f"({cold_articles / articles['article_id'].nunique():.2%} of all articles)")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sorted_counts = item_counts.sort_values(ascending=False).values
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(np.arange(1, len(sorted_counts) + 1), sorted_counts)
ax.set_xlabel("Article popularity rank (log)")
ax.set_ylabel("Purchase count (log)")
ax.set_title("H&M: article popularity long tail")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/hm_item_longtail.png", dpi=120)
print(f"\nSaved long-tail plot -> {FIG_DIR}/hm_item_longtail.png")

daily = txns.set_index("t_dat").resample("D").size()
fig, ax = plt.subplots(figsize=(8, 3))
daily.plot(ax=ax)
ax.set_title("H&M: transactions per day")
ax.set_ylabel("transactions")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/hm_txns_per_day.png", dpi=120)
print(f"Saved txns/day plot -> {FIG_DIR}/hm_txns_per_day.png")
