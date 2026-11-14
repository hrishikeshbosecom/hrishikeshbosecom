"""SuperKart sales forecasting - Streamlit front end."""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_sales_model.joblib")

# Store master data: each store has a fixed profile
STORES = {
    "OUT001": {"Store_Establishment_Year": 1987, "Store_Size": "High",
               "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type1"},
    "OUT002": {"Store_Establishment_Year": 1998, "Store_Size": "Small",
               "Store_Location_City_Type": "Tier 3", "Store_Type": "Food Mart"},
    "OUT003": {"Store_Establishment_Year": 1999, "Store_Size": "Medium",
               "Store_Location_City_Type": "Tier 1", "Store_Type": "Departmental Store"},
    "OUT004": {"Store_Establishment_Year": 2009, "Store_Size": "Medium",
               "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2"},
}
PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]
FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type",
            "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
            "Store_Location_City_Type", "Store_Type"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("🛒 SuperKart Sales Forecast")
st.write("Predict the total sales revenue of a product in a SuperKart store.")

single_tab, batch_tab = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with single_tab:
    st.subheader("Product details")
    c1, c2 = st.columns(2)
    product_type = c1.selectbox("Product type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Fruits and Vegetables"))
    sugar = c2.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    weight = c1.number_input("Product weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
    mrp = c2.number_input("Product MRP", min_value=10.0, max_value=400.0, value=147.0, step=1.0)
    area = st.slider("Allocated display area (share of store display)", 0.001, 0.300, 0.069, step=0.001, format="%.3f")

    st.subheader("Store details")
    store_id = st.selectbox("Store", list(STORES))
    profile = STORES[store_id]
    st.caption(f"{profile['Store_Type']} · {profile['Store_Size']} size · "
               f"{profile['Store_Location_City_Type']} city · established {profile['Store_Establishment_Year']}")

    # Collect the inputs into a single-row DataFrame in the training column order
    input_df = pd.DataFrame([{
        "Product_Weight": weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area,
        "Product_Type": product_type,
        "Product_MRP": mrp,
        "Store_Id": store_id,
        **profile,
    }])[FEATURES]

    if st.button("Predict sales", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Forecast total sales: **{prediction:,.2f}**")
        with st.expander("Model input"):
            st.dataframe(input_df, hide_index=True)

with batch_tab:
    st.write(f"Upload a CSV with these columns: `{', '.join(FEATURES)}`")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            batch["Product_Sugar_Content"] = batch["Product_Sugar_Content"].replace({"reg": "Regular"})
            batch["Predicted_Sales_Total"] = model.predict(batch[FEATURES]).round(2)
            st.dataframe(batch, hide_index=True)
            st.metric("Total forecast revenue", f"{batch['Predicted_Sales_Total'].sum():,.2f}")
            st.download_button("Download predictions", batch.to_csv(index=False),
                               "superkart_predictions.csv", "text/csv")
