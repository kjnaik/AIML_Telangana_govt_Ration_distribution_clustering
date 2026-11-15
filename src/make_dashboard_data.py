from pathlib import Path
import pandas as pd
from features import add_monthly_features

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

m = pd.read_csv(PROCESSED / "master_monthly.csv", parse_dates=["date"])
m = add_monthly_features(m)
y = (m.groupby(["distCode", "shopNo", "year"])
       .agg(avg_trans=("noOfTrans", "mean"), port_share=("port_share", "mean"),
            util=("util", "mean"), rice_per_card=("rice_per_card", "mean"),
            months=("date", "nunique"))
       .reset_index())
y.to_csv(PROCESSED / "shop_year.csv", index=False)
print("shop_year.csv:", y.shape, "| years:", sorted(y.year.unique()))