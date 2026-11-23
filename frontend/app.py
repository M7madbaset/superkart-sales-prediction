
import os
import io
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (container name 'backend' on the shared Docker network; override with BACKEND_URL)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:7860")

# Page title
st.title("SuperKart System")
st.write(
    "Enter the product and store details below to predict the total sales."
)

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Departmental Store", "Food Mart"])
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

def api_error(response):
    # Show the error message returned by the API (if any)
    try:
        detail = response.json().get("error", response.text)
    except Exception:
        detail = response.text
    st.error(f"Prediction failed (HTTP {response.status_code}): {detail}")

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
        else:
            api_error(response)
    except requests.exceptions.RequestException:
        st.error("Unable to connect to the prediction API. Check that the backend is running.")

# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Predict for Batch", type='primary'):
        file_bytes = uploaded_file.getvalue()
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, file_bytes, "text/csv")},
                timeout=60,
            )
            if response.status_code == 200:
                results = response.json()
                # keys are row indices ("0", "1", ...): sort numerically and attach to the uploaded rows
                preds = [results[k] for k in sorted(results, key=int)]
                batch_df = pd.read_csv(io.BytesIO(file_bytes))
                batch_df.insert(0, "Predicted_Sales", preds)
                st.session_state["batch_result"] = batch_df
            else:
                st.session_state.pop("batch_result", None)
                api_error(response)
        except requests.exceptions.RequestException:
            st.error("Unable to connect to the prediction API. Check that the backend is running.")

# Show (and allow download of) the latest batch result
if "batch_result" in st.session_state:
    st.success("Predictions completed successfully!")
    st.dataframe(st.session_state["batch_result"], use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        st.session_state["batch_result"].to_csv(index=False).encode("utf-8"),
        file_name="superkart_batch_predictions.csv",
        mime="text/csv",
    )
