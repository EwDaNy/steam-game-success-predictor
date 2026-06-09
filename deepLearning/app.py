import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load model, scaler and features
model = tf.keras.models.load_model("deep_learning_model.keras")
scaler = joblib.load("dl_scaler.pkl")
features = joblib.load("dl_features.pkl")

st.set_page_config(page_title="Steam Game Success Predictor", page_icon="🎮")

st.title("🎮 Steam Game Success Predictor")
st.subheader("Deep Learning Deployment")

st.write(
    "This application predicts whether a Steam game can be considered successful "
    "using a neural network model."
)

st.markdown("---")
st.write("### Enter game information:")

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

# Scale input data
input_scaled = scaler.transform(input_data)

st.markdown("---")

if st.button("Predict Success"):
    probability = model.predict(input_scaled)[0][0]
    prediction = 1 if probability >= 0.5 else 0

    if prediction == 1:
        st.success("🔥 The game is predicted to be SUCCESSFUL")
    else:
        st.error("❌ The game is predicted to be NOT successful")

    st.write("### Prediction probability:")
    st.write(f"Successful: {probability:.2f}")
    st.write(f"Not successful: {1 - probability:.2f}")

st.markdown("---")
st.caption("Model: Deep Neural Network | Dataset: Steam Games")
