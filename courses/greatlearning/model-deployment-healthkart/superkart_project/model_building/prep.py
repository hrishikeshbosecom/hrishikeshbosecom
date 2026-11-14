"""Data preparation: load from the repo, clean, split, and save train/test sets."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "superkart_project/data/SuperKart.csv"
OUTPUT_DIR = "splits"
TARGET = "Product_Store_Sales_Total"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: {df.shape}")

    # Standardise inconsistent category labels
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # Drop the unique identifier column (no predictive value)
    df = df.drop(columns=["Product_Id"])

    # Guard against duplicates / missing values in future data refreshes
    before = len(df)
    df = df.drop_duplicates().dropna()
    print(f"Removed {before - len(df)} duplicate/incomplete rows")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 80/20 split, stratified by store so every store is represented in both sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["Store_Id"]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Xtrain.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
    Xtest.to_csv(f"{OUTPUT_DIR}/Xtest.csv", index=False)
    ytrain.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
    ytest.to_csv(f"{OUTPUT_DIR}/ytest.csv", index=False)

    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Splits saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
