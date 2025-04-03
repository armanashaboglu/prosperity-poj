import pandas as pd
import numpy as np

# Load your CSV file
csv_path = "/Users/sinakitapci/Desktop/prosperity_trading/log_poj.csv"
df = pd.read_csv(csv_path, sep=';')

# Convert necessary columns
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
df = df.dropna(subset=["timestamp", "mid_price"])

# Split data per product
kelp_df = df[df["product"] == "KELP"].copy()
resin_df = df[df["product"] == "RAINFOREST_RESIN"].copy()

# Extended analysis function
def process_product(product_df, label, window=10, ewma_span=10):
    product_df = product_df.sort_values("timestamp").set_index("timestamp")

    # Moving averages
    product_df["MA"] = product_df["mid_price"].rolling(window=window).mean()
    product_df["EWMA"] = product_df["mid_price"].ewm(span=ewma_span).mean()
    product_df["LOG_MA"] = np.log(product_df["mid_price"]).rolling(window=window).mean()

    # Z-score (deviation from MA)
    product_df["STD"] = product_df["mid_price"].rolling(window=window).std()
    product_df["Z_SCORE"] = (product_df["mid_price"] - product_df["MA"]) / product_df["STD"]

    # Bid-Ask Imbalance (if columns exist)
    if "bid_volume_1" in product_df.columns and "ask_volume_1" in product_df.columns:
        product_df["bid_volume_1"] = pd.to_numeric(product_df["bid_volume_1"], errors="coerce")
        product_df["ask_volume_1"] = pd.to_numeric(product_df["ask_volume_1"], errors="coerce")
        product_df["imbalance"] = (
            product_df["bid_volume_1"] - product_df["ask_volume_1"]
        ) / (product_df["bid_volume_1"] + product_df["ask_volume_1"])
    else:
        product_df["imbalance"] = np.nan

    # Latest data
    latest = product_df.dropna(subset=["MA", "EWMA", "mid_price"]).iloc[-1]
    fair_value = np.mean([latest["MA"], latest["EWMA"]])

    # Print full metrics
    print(f"\n--- {label} ---")
    print(f"Fair Value Estimate     : {fair_value:.2f}")
    print(f"Latest Mid Price        : {latest['mid_price']:.2f}")
    print(f"{window}-period MA       : {latest['MA']:.2f}")
    print(f"EWMA (span={ewma_span})  : {latest['EWMA']:.2f}")
    print(f"Log MA                  : {latest['LOG_MA']:.4f}")
    print(f"Z-Score vs MA           : {latest['Z_SCORE']:.4f}")
    imbalance = latest['imbalance']
    if not pd.isna(imbalance):
        print(f"Bid-Ask Imbalance       : {imbalance:.4f}")
    else:
        print("Bid-Ask Imbalance       : N/A (missing volume data)")

    return product_df

# Run for both products
kelp_df = process_product(kelp_df, "KELP")
resin_df = process_product(resin_df, "RAINFOREST_RESIN")
