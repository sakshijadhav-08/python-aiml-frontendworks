import streamlit as st
"""Streamlit is a Python library used to create web applications for data science,
machine learning, and Python projects without needing HTML, CSS, or JavaScript."""

import pandas as pd
# Pandas is a Python library used for data manipulation and analysis.

import joblib
# Joblib is a Python library used to save and load machine learning models and large Python objects.

model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

# st.set_page_config() – Used to customize the Streamlit app's page settings (such as the page title, icon, and layout).
# page_title – Sets the name of the application that appears on the browser tab.
# layout="wide" – Makes the app use the full width of the browser window, giving more space for displaying content.

st.title("Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")

year = st.number_input(
    "Year",
    min_value=2016,
    max_value=2026,
    value=2020
)

mileage = st.number_input(
    "Mileage",
    min_value=1,
    max_value=17000,
    value=5000
)

tax = st.number_input(
    "Road Tax", 
    min_value=0,
    max_value=500,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=201.0,
    value=50.0
)

engine = st.number_input(
    "Engine Size",
    min_value=1.0,
    max_value=5.0,
    value=2.0
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic","Manual","Semi-Auto"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol","Diesel", "Hybrid", "Electric", "Other"]
)

# selectbox() is used to provide a dropdown menu that makes data selection easy, accurate, and user-friendly.

car_model = st.text_input("Car Model")

predict = st.button("Predict Price")

if predict:

    data = {
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine]
    }

    df = pd.DataFrame(data)

    # One Hot Encoding
    df = pd.get_dummies(df)

     # Match training columns
    df = df.reindex(columns=encoded_columns, fill_value=0)

    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # Apply Standard Scaling
    df[numerical_columns] = scaler.transform(
        df[numerical_columns]
    )

    # Predict price
    prediction = model.predict(df)

    st.success(
        f"Predicted Price: £{prediction[0]:,.2f}"
    )
    