"""Feature engineering for the ranking stage.

Every function here takes a `feature_df` slice that must already be truncated
to strictly before the target window being predicted — that boundary is
enforced by the caller (scripts/train_ranker.py), not here, but every feature
below assumes it. Mixing in even one row from the target window would leak
the label into the input.

Static profile joins (customers.csv, articles.csv) are a smaller, harder-to-avoid
leakage risk worth flagging explicitly: both are single point-in-time snapshots,
not time-versioned history. A customer's `age` or `club_member_status` at the time
of the snapshot may differ from its true value at the historical feature cutoff.
This is unavoidable with this dataset (no historical profile versions exist) and
is the same assumption the original competition itself made — but it should be
named, not silently assumed.
"""
import numpy as np
import pandas as pd


def item_popularity_features(feature_df: pd.DataFrame, target_date, item_map: pd.DataFrame) -> pd.DataFrame:
    """Popularity counted over multiple recency windows ending at target_date.

    Multiple windows (not just all-time) matter because fashion has real trend/
    seasonal dynamics — an item popular in the last 7 days is a different signal
    than one popular over the entire 2-year history.
    """
    windows = {"pop_7d": 7, "pop_30d": 30, "pop_90d": 90, "pop_all": None}
    out = item_map[["item_idx"]].copy()
    for name, days in windows.items():
        if days is None:
            window_df = feature_df
        else:
            window_df = feature_df[feature_df["t_dat"] >= target_date - pd.Timedelta(days=days)]
        counts = window_df.groupby("item_idx").size().rename(name)
        out = out.merge(counts, on="item_idx", how="left")
    out[list(windows.keys())] = out[list(windows.keys())].fillna(0)
    return out


def item_price_features(feature_df: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    price = feature_df.groupby("item_idx")["price"].mean().rename("item_avg_price")
    return item_map[["item_idx"]].merge(price, on="item_idx", how="left")


def user_features(feature_df: pd.DataFrame, target_date, user_map: pd.DataFrame) -> pd.DataFrame:
    grp = feature_df.groupby("user_idx")
    stats = pd.DataFrame({
        "user_purchase_count": grp.size(),
        "user_avg_price": grp["price"].mean(),
        "user_distinct_items": grp["item_idx"].nunique(),
        "user_last_purchase_date": grp["t_dat"].max(),
    })
    stats["days_since_last_purchase"] = (target_date - stats["user_last_purchase_date"]).dt.days
    stats["user_repeat_rate"] = 1 - stats["user_distinct_items"] / stats["user_purchase_count"]
    stats = stats.drop(columns=["user_last_purchase_date"])
    return user_map[["user_idx"]].merge(stats, on="user_idx", how="left")


def user_item_history_features(feature_df: pd.DataFrame, target_date, pairs: pd.DataFrame) -> pd.DataFrame:
    """For each (user_idx, item_idx) candidate pair: has this user bought this
    exact item before, how many times, and how recently. This is the strongest
    single feature family per the research — repurchase behavior dominates.
    """
    ui = feature_df.groupby(["user_idx", "item_idx"]).agg(
        ui_purchase_count=("t_dat", "size"),
        ui_last_purchase_date=("t_dat", "max"),
    ).reset_index()
    ui["days_since_last_purchase_of_item"] = (target_date - ui["ui_last_purchase_date"]).dt.days
    ui = ui.drop(columns=["ui_last_purchase_date"])
    merged = pairs.merge(ui, on=["user_idx", "item_idx"], how="left")
    merged["ui_purchase_count"] = merged["ui_purchase_count"].fillna(0)
    return merged


def category_affinity_features(feature_df: pd.DataFrame, articles: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Does this candidate's product_type/department match what the user usually buys?"""
    art = articles[["item_idx", "product_type_name", "department_name"]]
    user_hist = feature_df.merge(art, on="item_idx", how="left")
    type_counts = user_hist.groupby(["user_idx", "product_type_name"], observed=True).size().rename("n").reset_index()
    user_type_totals = user_hist.groupby("user_idx").size().rename("total").reset_index()
    type_affinity = type_counts.merge(user_type_totals, on="user_idx")
    type_affinity["type_affinity"] = type_affinity["n"] / type_affinity["total"]
    type_affinity = type_affinity[["user_idx", "product_type_name", "type_affinity"]]

    pairs = pairs.merge(art, on="item_idx", how="left")
    pairs = pairs.merge(type_affinity, on=["user_idx", "product_type_name"], how="left")
    pairs["type_affinity"] = pairs["type_affinity"].fillna(0)
    return pairs.drop(columns=["product_type_name", "department_name"])
