import glob
import pandas as pd

for folder in ["transactions", "card_status", "locations"]:
    print("\n" + "=" * 60)
    print(folder.upper())
    for f in sorted(glob.glob(f"data/raw/{folder}/*.csv")):
        df = pd.read_csv(f, low_memory=False)
        print("\nFILE:", f)
        print("SHAPE:", df.shape)
        print("COLUMNS:", list(df.columns))
        print(df.head(2).to_string())