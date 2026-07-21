import pandas as pd
import streamlit as st
import joblib

model_lr = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Luxury Perfume Price Predictor",
    page_icon="🌷",
    layout="wide"
)

st.title("🌷Luxury Perfume Price Predictor~")
st.write("Discover the Estimated Price of Premium Perfumes.")

brand = st.text_input("Enter Brand Name")
product_title = st.text_input("Enter Product Title")
price_currency = st.text_input("Enter Price with Currency")
available_text = st.text_input("Enter Available Status")
last_updated = st.text_input("Enter Last Updated")
item_location = st.text_input("Enter Item Location")

perfume_type = st.selectbox(
    "Select Perfume Type",
    ["Eau de Parfum", "Eau de Toilette", "Perfume", "Cologne"]
)

available = st.number_input(
    "Available Quantity",
    min_value=2.0,
    max_value=842.0,
    value=100.0
)

sold = st.number_input(
    "Number of Products Sold",
    min_value=1.0,
    max_value=50000.0,
    value=100.0
)

predict = st.button("Predict Price")

if predict:

    data = {
        "brand":[brand],
        "product_title":[product_title],
        "price_currency":[price_currency],
        "available_text":[available_text],
        "last_updated":[last_updated],
        "item_location":[item_location],
        "perfume_type":[perfume_type],
        "available":[available],
        "sold":[sold]
    }

    df = pd.DataFrame(data)

    # One Hot Encoding
    df = pd.get_dummies(df)

     # Match training columns
    df = df.reindex(columns=encoded_columns, fill_value=0)

    prediction = model_lr.predict(df)

    st.success(
        label="Predicted Price",
        value=f"₹ {prediction[0]:,.2f}"
    )