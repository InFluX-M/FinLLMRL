import json
import pandas as pd

STOCKS = ['AAPL', 'GS', 'BA', 'JPM']

for stock in STOCKS:
    # File names
    file1 = f"new_{stock}_fin.json"
    file2 = f"{stock}_final.json"

    # Load JSON files
    with open(file1, "r") as f:
        data1 = json.load(f)
    with open(file2, "r") as f:
        data2 = json.load(f)

    # Merge lists
    merged = data1 + data2

    # Convert to DataFrame for sorting
    df = pd.DataFrame(merged)

    # Ensure date is parsed correctly
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Convert dates back to string
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Save back to JSON
    out_file = f"{stock}_merged.json"
    with open(out_file, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)

    print(f"{stock}: merged {len(data1)} + {len(data2)} → {len(df)} rows, saved {out_file}")
