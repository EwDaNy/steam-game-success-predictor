import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("random_forest_model.pkl")
features = joblib.load("rf_features.pkl")

st.set_page_config(page_title="Steam Game Success Predictor", page_icon="🎮")

st.title("🎮 Steam Game Success Predictor")
st.subheader("Random Forest Deployment")

st.write(
    "This app predicts whether a Steam game can be considered successful "
    "based on game-related numerical features."
)

st.markdown("---")

st.write("### Enter game information:")

# Default values for common Steam dataset features
default_values = {
    "required_age": 0,
    "achievements": 0,
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

st.markdown("---")

if st.button("Predict Success"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("🔥 The game is predicted to be SUCCESSFUL")
    else:
        st.error("❌ The game is predicted to be NOT successful")

    st.write("### Prediction probability:")
    st.write(f"Not successful: {probability[0]:.2f}")
    st.write(f"Successful: {probability[1]:.2f}")

st.markdown("---")
st.caption("Model: Random Forest Classifier | Dataset: Steam Games")
