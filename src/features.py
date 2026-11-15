from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SHOP_KEYS = ["distCode", "shopNo"]
RICE_COLS = ["riceAfsc", "riceFsc", "riceAap"]

# Features that go into scaling / PCA / clustering
MODEL_FEATURES = [
    "log_avg_trans",   # shop size
    "log_rice",        # rice volume (log fixes skew)
    "rice_per_card",   # commodity intensity per entitled card
    "local_util",      # local use of the shop's own cards
    "port_share",      # share of transactions from other shops' cards
    "wheat_flag",      # shop distributes wheat (rare)
    "trans_cv",        # volatility: std / mean of monthly transactions
    "util_std",        # volatility of utilization across months
    "port_growth",     # change in portability share, first to last month
]


def add_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level (shop-month) features."""
    df = df.copy()
    cards = df["totalRcs"].where(df["totalRcs"] > 0)   # avoid divide-by-zero

    df["rice_total"] = df[RICE_COLS].sum(axis=1)
    df["log_rice"] = np.log1p(df["rice_total"])
    df["util"] = df["noOfTrans"] / cards                      # can exceed 1 (portability)
    df["local_trans"] = (df["noOfTrans"] - df["otherShopTransCnt"]).clip(lower=0)
    df["local_util"] = df["local_trans"] / cards
    df["port_share"] = df["otherShopTransCnt"] / df["noOfTrans"]
    df["rice_per_card"] = df["rice_total"] / cards
    df["wheat_flag"] = (df["wheat"] > 0).astype(int)
    return df


def build_shop_features(df: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    """Collapse shop-month rows into one row per shop."""
    df = add_monthly_features(df).sort_values("date")
    g = df.groupby(SHOP_KEYS)

    feats = g.agg(
        distName=("distName", "first"),
        officeName=("officeName", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        fpsType=("fpsType", "first"),
        fpsStatus=("fpsStatus", "first"),
        n_months=("date", "nunique"),
        avg_trans=("noOfTrans", "mean"),
        avg_cards=("totalRcs", "mean"),
        util=("util", "mean"),
        local_util=("local_util", "mean"),
        port_share=("port_share", "mean"),
        rice_per_card=("rice_per_card", "mean"),
        log_rice=("log_rice", "mean"),
        wheat_flag=("wheat_flag", "max"),
        trans_std=("noOfTrans", "std"),
        util_std=("util", "std"),
    ).reset_index()

    # Volatility and growth (data is sorted by date, so first/last are chronological)
    feats["trans_cv"] = feats["trans_std"] / feats["avg_trans"]
    fl = g["port_share"].agg(["first", "last"]).reset_index()
    fl["port_growth"] = fl["last"] - fl["first"]
    feats = feats.merge(fl[SHOP_KEYS + ["port_growth"]], on=SHOP_KEYS, how="left")

    feats["log_avg_trans"] = np.log1p(feats["avg_trans"])

    # Only shops with enough months and complete features are modelled
    complete = feats[MODEL_FEATURES].notna().all(axis=1)
    feats["use_for_model"] = complete & (feats["n_months"] >= min_months)
    return feats


if __name__ == "__main__":
    master = pd.read_csv(PROCESSED / "master_monthly.csv", parse_dates=["date"])
    shops = build_shop_features(master)
    out = PROCESSED / "shop_features.csv"
    shops.to_csv(out, index=False)

    print("Shops:", len(shops), "| usable for modelling:", int(shops.use_for_model.sum()))
    print("Shops by months available:\n", shops.n_months.value_counts().sort_index())
    print("\nShare of shops with wheat:", round(shops.wheat_flag.mean(), 3))
    print("\nFeature summary:\n", shops[MODEL_FEATURES].describe().T.round(3))
    print("\nCorrelations:\n", shops[MODEL_FEATURES].corr().round(2))
    print("\nSaved to", out)