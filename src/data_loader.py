from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

MONTHLY_KEYS = ["distCode", "shopNo", "month", "year"]
SHOP_KEYS = ["distCode", "shopNo"]

CARD_COLS = [
    "rcNfsaAay", "unitsNfsaAay", "rcNfsaPhh", "unitsNfsaPhh",
    "totalRcNfsa", "totalUnitsNfsa", "rcStateAay", "unitsStateAay",
    "rcStatePhh", "unitsStatePhh", "rcStateAap", "unitsStateAap",
    "totalRcState", "totalUnitsState", "totalRcs", "totalUnits",
]
LOC_COLS = ["address", "latitude", "longitude", "fpsStatus", "fpsType"]


def load_folder(folder: str) -> pd.DataFrame:
    """Loop over every CSV in data/raw/<folder> and stack them into one DataFrame."""
    files = sorted((RAW / folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW / folder}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["source_file"] = f.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(f"{folder}: {len(files)} files -> {out.shape[0]:,} rows")
    return out


def build_master() -> pd.DataFrame:
    trans = load_folder("transactions")
    card = load_folder("card_status")
    loc = load_folder("locations")

    # Remove accidental duplicates
    trans = trans.drop_duplicates(subset=MONTHLY_KEYS)
    card = card.drop_duplicates(subset=MONTHLY_KEYS)
    loc = loc.sort_values("dateTime").drop_duplicates(subset=SHOP_KEYS, keep="last")

    # Keep only useful columns (avoids duplicate distName/officeName columns on merge)
    card = card[MONTHLY_KEYS + CARD_COLS]
    loc = loc[SHOP_KEYS + LOC_COLS]

    # Join 1: transactions + card status (same shop, same month)
    master = trans.merge(
        card, on=MONTHLY_KEYS, how="left", validate="one_to_one", indicator="card_match"
    )
    # Join 2: + shop locations (one row per shop)
    master = master.merge(
        loc, on=SHOP_KEYS, how="left", validate="many_to_one", indicator="loc_match"
    )

    # Match report
    print("\nTransactions rows with no card-status match:",
          (master["card_match"] == "left_only").sum())
    print("Transactions rows with no location match:",
          (master["loc_match"] == "left_only").sum())
    master = master.drop(columns=["card_match", "loc_match"])

    # Proper date column for time-series work
    master["date"] = pd.to_datetime(
        dict(year=master["year"], month=master["month"], day=1)
    )
    return master.sort_values(["distCode", "shopNo", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    master = build_master()
    out_path = PROCESSED / "master_monthly.csv"
    master.to_csv(out_path, index=False)

    print("\nMaster shape:", master.shape)
    print("Months loaded:", sorted(master["date"].dt.strftime("%Y-%m").unique()))
    print("Unique shops:", master[["distCode", "shopNo"]].drop_duplicates().shape[0])
    print("Missing lat/long:", master["latitude"].isna().sum())
    print("\nNull % (top 10):")
    print((master.isna().mean() * 100).round(1).sort_values(ascending=False).head(10))
    print("\nSaved to", out_path)