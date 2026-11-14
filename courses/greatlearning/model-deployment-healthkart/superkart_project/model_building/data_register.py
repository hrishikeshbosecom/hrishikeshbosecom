"""Data registration: validate the dataset stored in the repo and print a summary."""
import sys
import pandas as pd

DATA_PATH = "superkart_project/data/SuperKart.csv"

EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]


def main():
    df = pd.read_csv(DATA_PATH)

    # 1. Schema check - every expected column must be present
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        sys.exit(f"Dataset validation FAILED - missing columns: {missing}")
    if extra:
        print(f"Warning - unexpected extra columns: {extra}")

    # 2. Basic sanity checks
    assert len(df) > 0, "Dataset is empty"
    assert (df["Product_Store_Sales_Total"] >= 0).all(), "Negative sales values found"

    # 3. Summary
    print(f"Dataset registered: {DATA_PATH}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"Missing values: {int(df.isnull().sum().sum())} | Duplicate rows: {int(df.duplicated().sum())}")
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nTarget (Product_Store_Sales_Total) summary:")
    print(df["Product_Store_Sales_Total"].describe().round(2).to_string())
    print("\nAll expected columns present - validation PASSED.")


if __name__ == "__main__":
    main()
