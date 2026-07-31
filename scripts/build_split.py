"""Phase 2, step 1: time-based train/val/test split.

Cutoff matches the actual H&M Kaggle competition protocol: last 7 days = test,
prior 7 days = val, everything else = train. Using the same protocol as the
original competition is a deliberate, citable choice (not an arbitrary one).

Splitting by TIME (not randomly) matters because a random split leaks future
purchase patterns into training — e.g. a random split could put a customer's
Sept 21 purchase in train and their Sept 18 purchase in test, letting the model
implicitly "see the future" relative to what it's being asked to predict.

User/item integer id mappings are built from TRAIN ONLY. Any customer/article
that appears in val/test but not in train is, by construction, cold-start for
that split — this is the correct and only way to measure cold-start rate,
since building the mapping from the full dataset would silently leak
"knowledge that this item/user exists" from the future into what train sees.
"""
import pandas as pd

DATA_DIR = "data/raw/hm"
OUT_DIR = "data/processed"

txns = pd.read_csv(
    f"{DATA_DIR}/transactions_train.csv",
    dtype={"customer_id": "category", "article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
    parse_dates=["t_dat"],
)

max_date = txns["t_dat"].max()
test_start = max_date - pd.Timedelta(days=6)      # 2020-09-16
val_start = test_start - pd.Timedelta(days=7)     # 2020-09-09

txns["split"] = "train"
txns.loc[txns["t_dat"] >= test_start, "split"] = "test"
txns.loc[(txns["t_dat"] >= val_start) & (txns["t_dat"] < test_start), "split"] = "val"

print("Split boundaries:")
print(f"  train: <  {val_start.date()}")
print(f"  val:   {val_start.date()} .. {(test_start - pd.Timedelta(days=1)).date()}")
print(f"  test:  {test_start.date()} .. {max_date.date()}")
print()
print(txns["split"].value_counts())

train = txns[txns["split"] == "train"]
train_users = train["customer_id"].unique()
train_items = train["article_id"].unique()

user_map = pd.DataFrame({"customer_id": train_users, "user_idx": range(len(train_users))})
item_map = pd.DataFrame({"article_id": train_items, "item_idx": range(len(train_items))})

print(f"\nTrain users: {len(user_map):,}  Train items: {len(item_map):,}")

for split_name in ["val", "test"]:
    split_df = txns[txns["split"] == split_name]
    warm_users = split_df["customer_id"].isin(user_map["customer_id"]).mean()
    warm_items = split_df["article_id"].isin(item_map["article_id"]).mean()
    print(f"{split_name}: {warm_users:.2%} of rows have a customer seen in train, "
          f"{warm_items:.2%} have an article seen in train")

txns.to_parquet(f"{OUT_DIR}/transactions.parquet", index=False)
user_map.to_parquet(f"{OUT_DIR}/user_id_map.parquet", index=False)
item_map.to_parquet(f"{OUT_DIR}/item_id_map.parquet", index=False)
print(f"\nSaved -> {OUT_DIR}/transactions.parquet, user_id_map.parquet, item_id_map.parquet")
