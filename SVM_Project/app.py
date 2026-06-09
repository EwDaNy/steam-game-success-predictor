import streamlit as st
import pandas as pd
import joblib

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("svm_features.pkl")

st.set_page_config(page_title="Steam Game Success Predictor", page_icon="🎮")

st.title("🎮 Steam Game Success Predictor")
st.subheader("SVM Deployment")

st.write("This app predicts Steam game success using Support Vector Machine.")

default_values = {
    "userscore": 0,
    "average_forever": 100,
    "average_2weeks": 10,
    "median_forever": 100,
    "median_2weeks": 10,
    "price": 0,
    "initialprice": 0,
    "discount": 0
}

input_values = {}

for feature in features:
    input_values[feature] = st.number_input(
        feature,
        value=float(default_values.get(feature, 0))
    )

input_data = pd.DataFrame([input_values])

input_scaled = scaler.transform(input_data)

if st.button("Predict Success"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("🔥 The game is predicted to be SUCCESSFUL")
    else:
        st.error("❌ The game is predicted to be NOT successful")