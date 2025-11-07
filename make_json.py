# make_json.py
import pandas as pd, json, sys
df = pd.read_csv("data/creditcard.csv")
idx = int(sys.argv[1]) if len(sys.argv)>1 else 0   # row index to use
row = df.iloc[idx].copy()
# drop Class column if present
if "Class" in row.index: row = row.drop("Class")
cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
out = {c: float(row[c]) for c in cols}
print(json.dumps(out, indent=2))
