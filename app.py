# app.py (Streamlit frontend - robust CSV parsing)
import streamlit as st
import requests
import pandas as pd
import re

st.set_page_config(page_title="Credit Card Fraud Detection (Demo)", layout="wide")
st.title("Credit Card Fraud Detection (Demo)")

st.markdown("Enter transaction features (or paste a CSV row) and click Predict.")
st.markdown("CSV row format: `Time,V1,V2,...,V28,Amount`  — if you paste a row that includes the Class label (0/1) it will be removed automatically.")

API_URL = "http://127.0.0.1:8000/predict"  # change if your API is on a different host/port

def parse_csv_row(csv_text: str):
    # remove surrounding whitespace and quotes
    s = csv_text.strip()
    if not s:
        return None, "Empty input"

    # remove any surrounding quotes and invisible characters
    s = s.replace('\ufeff', '')  # remove BOM if present
    s = s.replace('"', '').replace("'", "")
    # split on commas (allow whitespace around commas)
    vals = [v.strip() for v in re.split(r'\s*,\s*', s) if v.strip() != ""]

    # If last token looks like class label (0 or 1), drop it
    if len(vals) == 31 and vals[-1] in ("0", "1", "0.0", "1.0"):
        vals = vals[:-1]

    # final check
    if len(vals) != 30:
        return None, f"CSV row must have 30 values after cleaning (Time,V1..V28,Amount). Found {len(vals)} values."

    # try convert to floats
    try:
        float_vals = [float(x) for x in vals]
    except Exception as e:
        return None, f"Could not convert values to float. Error: {e}"

    # build transaction dict
    cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    tx = dict(zip(cols, float_vals))
    return tx, None

st.subheader("Paste CSV row (Time,V1..V28,Amount) — optional")
csv_text = st.text_area("Paste CSV row (single line) — do not include the header", height=140)

tx = None
error_msg = None

if csv_text:
    tx, error_msg = parse_csv_row(csv_text)
    if error_msg:
        st.error(error_msg)
    else:
        st.success("Parsed CSV row successfully. Preview of first 5 fields:")
        st.write({k: tx[k] for k in list(tx.keys())[:5]})  # show a small preview

# If no CSV paste or parsing failed, allow manual input
if not csv_text or tx is None:
    st.subheader("Or fill manually")
    cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    manual_vals = {}
    # show inputs in columns for compactness
    cols_per_row = 5
    for i in range(0, len(cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(cols[i:i+cols_per_row]):
            with row_cols[j]:
                manual_vals[col_name] = st.number_input(col_name, value=0.0, format="%.6f")
    # create tx from manual values if CSV not used
    if tx is None:
        tx = manual_vals

# Predict button
if st.button("Predict") and tx is not None:
    try:
        with st.spinner("Calling API..."):
            resp = requests.post(API_URL, json=tx, timeout=10)
        if resp.status_code == 200:
            res = resp.json()
            st.success(f"Result: {res}")
            # pretty display
            if "probability" in res:
                st.metric("Fraud probability", f"{res['probability']:.6f}")
                st.write("Predicted label:", res.get("label", "N/A"))
            else:
                st.write(res)
        else:
            st.error(f"API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Tip: copy a single CSV line from data/creditcard.csv (exclude header). If it contains the Class label (last column), the app will drop it automatically.")
