import pandas as pd
import os

for country in ["India", "Italy"]:
    path = f"data/dataset anemia/{country}/{country}.xlsx"
    if os.path.exists(path):
        df = pd.read_excel(path)
        print(f"\n=== {country}.xlsx ===")
        print("Columns:", list(df.columns))
        print(df.head(3).to_string())
    else:
        print(f"\n=== {country}.xlsx NOT FOUND at {path} ===")