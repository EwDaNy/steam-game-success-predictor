import streamlit as st
import numpy as np
import joblib
import pandas as pd

# загрузка модели и scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎮 Steam Games Success Predictor (KNN)")

st.write("Введите параметры игры:")

# ввод данных (берём ключевые признаки)
positive = st.number_input("Positive reviews", value=1000)
negative = st.number_input("Negative reviews", value=100)
userscore = st.number_input("User score", value=0)
average_forever = st.number_input("Average playtime (forever)", value=100)
average_2weeks = st.number_input("Average playtime (2 weeks)", value=10)
median_forever = st.number_input("Median playtime forever", value=100)
median_2weeks = st.number_input("Median playtime 2 weeks", value=10)
price = st.number_input("Price", value=0)
initialprice = st.number_input("Initial price", value=0)
discount = st.number_input("Discount", value=0)
ccu = st.number_input("Concurrent users (CCU)", value=100)

# формируем input
input_data = pd.DataFrame([[
    positive, negative, userscore,
    average_forever, average_2weeks,
    median_forever, median_2weeks,
    price, initialprice, discount, ccu
]], columns=[
    'positive', 'negative', 'userscore',
    'average_forever', 'average_2weeks',
    'median_forever', 'median_2weeks',
    'price', 'initialprice', 'discount', 'ccu'
])

# масштабирование
input_data = scaler.transform(input_data)

# кнопка предсказания
if st.button("Predict Success"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🔥 Game is SUCCESSFUL")
    else:
        st.error("❌ Game is NOT successful")



#streamlit run app.py