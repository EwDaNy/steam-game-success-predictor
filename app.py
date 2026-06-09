from pathlib import Path
from math import exp

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_VALUES = {
    "positive": 1000,
    "negative": 100,
    "required_age": 0,
    "achievements": 0,
    "userscore": 0,
    "average_forever": 100,
    "average_2weeks": 10,
    "median_forever": 100,
    "median_2weeks": 10,
    "price": 0,
    "initialprice": 0,
    "discount": 0,
    "ccu": 100,
}

SUCCESSFUL_GAME_VALUES = {
    "positive": 358266,
    "negative": 22443,
    "required_age": 0,
    "achievements": 0,
    "userscore": 0,
    "average_forever": 3854,
    "average_2weeks": 835,
    "median_forever": 2213,
    "median_2weeks": 257,
    "price": 2999,
    "initialprice": 2999,
    "discount": 0,
    "ccu": 18028,
}

UNSUCCESSFUL_GAME_VALUES = {
    "positive": 1977,
    "negative": 1105,
    "required_age": 0,
    "achievements": 0,
    "userscore": 0,
    "average_forever": 639,
    "average_2weeks": 0,
    "median_forever": 229,
    "median_2weeks": 0,
    "price": 0,
    "initialprice": 0,
    "discount": 0,
    "ccu": 256,
}

KNN_FEATURES = [
    "positive",
    "negative",
    "userscore",
    "average_forever",
    "average_2weeks",
    "median_forever",
    "median_2weeks",
    "price",
    "initialprice",
    "discount",
    "ccu",
]


st.set_page_config(
    page_title="Steam ML Model Comparison",
    page_icon="ML",
    layout="wide",
)


@st.cache_resource
def load_sklearn_artifacts():
    return {
        "KNN": {
            "model": joblib.load(BASE_DIR / "KNN_Project" / "knn_model.pkl"),
            "scaler": joblib.load(BASE_DIR / "KNN_Project" / "scaler.pkl"),
            "features": KNN_FEATURES,
            "uses_scaler": True,
        },
        "SVM": {
            "model": joblib.load(BASE_DIR / "SVM_Project" / "svm_model.pkl"),
            "scaler": joblib.load(BASE_DIR / "SVM_Project" / "scaler.pkl"),
            "features": joblib.load(BASE_DIR / "SVM_Project" / "svm_features.pkl"),
            "uses_scaler": True,
        },
        "Random Forest": {
            "model": joblib.load(BASE_DIR / "randomForest" / "random_forest_model.pkl"),
            "scaler": None,
            "features": joblib.load(BASE_DIR / "randomForest" / "rf_features.pkl"),
            "uses_scaler": False,
        },
    }


@st.cache_resource
def load_deep_learning_artifacts():
    import tensorflow as tf

    return {
        "model": tf.keras.models.load_model(
            BASE_DIR / "deepLearning" / "deep_learning_model.keras"
        ),
        "scaler": joblib.load(BASE_DIR / "deepLearning" / "dl_scaler.pkl"),
        "features": joblib.load(BASE_DIR / "deepLearning" / "dl_features.pkl"),
        "uses_scaler": True,
    }


def build_input_frame(values, features):
    return pd.DataFrame([{feature: values.get(feature, 0) for feature in features}])


def sigmoid(score):
    if score >= 0:
        return 1 / (1 + exp(-score))
    return exp(score) / (1 + exp(score))


def predict_sklearn(name, artifacts, values):
    input_data = build_input_frame(values, artifacts["features"])

    if artifacts["uses_scaler"]:
        input_data = artifacts["scaler"].transform(input_data)

    prediction = int(artifacts["model"].predict(input_data)[0])
    probability = None

    if hasattr(artifacts["model"], "predict_proba"):
        probability = float(artifacts["model"].predict_proba(input_data)[0][1])
    elif hasattr(artifacts["model"], "decision_function"):
        decision_score = float(artifacts["model"].decision_function(input_data)[0])
        probability = sigmoid(decision_score)

    return {
        "model": name,
        "prediction": prediction,
        "probability": probability,
        "features": len(artifacts["features"]),
    }


def predict_deep_learning(values):
    artifacts = load_deep_learning_artifacts()
    input_data = build_input_frame(values, artifacts["features"])
    input_scaled = artifacts["scaler"].transform(input_data)
    probability = float(artifacts["model"].predict(input_scaled, verbose=0)[0][0])

    return {
        "model": "Deep Learning",
        "prediction": 1 if probability >= 0.5 else 0,
        "probability": probability,
        "features": len(artifacts["features"]),
    }


def render_result(result):
    label = "Successful" if result["prediction"] == 1 else "Not successful"
    probability = result["probability"]

    st.metric(result["model"], label)
    st.caption(f"Input features used: {result['features']}")

    if probability is not None:
        st.progress(max(0.0, min(1.0, probability)))
        st.write(f"Success probability: `{probability:.2f}`")
    else:
        st.write("Success probability: `not available`")


def apply_preset(values):
    for key, value in values.items():
        st.session_state[key] = value


for key, value in DEFAULT_VALUES.items():
    st.session_state.setdefault(key, value)

query_example = st.query_params.get("example")
query_compare = st.query_params.get("compare") == "1"

if query_example == "successful":
    apply_preset(SUCCESSFUL_GAME_VALUES)
elif query_example == "unsuccessful":
    apply_preset(UNSUCCESSFUL_GAME_VALUES)


st.title("Steam Game Success Predictor")
st.subheader("KNN, SVM, Random Forest and Deep Learning deployment")

st.write(
    "Enter Steam game parameters once and compare predictions from all four "
    "trained machine learning models."
)

st.divider()

st.write("### Game information")

preset_col1, preset_col2 = st.columns(2)

with preset_col1:
    st.button(
        "Use successful game example",
        on_click=apply_preset,
        args=(SUCCESSFUL_GAME_VALUES,),
        use_container_width=True,
    )

with preset_col2:
    st.button(
        "Use unsuccessful game example",
        on_click=apply_preset,
        args=(UNSUCCESSFUL_GAME_VALUES,),
        use_container_width=True,
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    positive = st.number_input("Positive reviews", key="positive")
    negative = st.number_input("Negative reviews", key="negative")
    userscore = st.number_input("User score", key="userscore")

with col2:
    average_forever = st.number_input(
        "Average playtime forever", key="average_forever"
    )
    average_2weeks = st.number_input(
        "Average playtime 2 weeks", key="average_2weeks"
    )
    median_forever = st.number_input(
        "Median playtime forever", key="median_forever"
    )

with col3:
    median_2weeks = st.number_input(
        "Median playtime 2 weeks", key="median_2weeks"
    )
    price = st.number_input("Price", key="price")
    initialprice = st.number_input("Initial price", key="initialprice")

with col4:
    discount = st.number_input("Discount", key="discount")
    ccu = st.number_input("Concurrent users (CCU)", key="ccu")
    required_age = st.number_input("Required age", key="required_age")
    achievements = st.number_input("Achievements", key="achievements")

input_values = {
    "positive": positive,
    "negative": negative,
    "required_age": required_age,
    "achievements": achievements,
    "userscore": userscore,
    "average_forever": average_forever,
    "average_2weeks": average_2weeks,
    "median_forever": median_forever,
    "median_2weeks": median_2weeks,
    "price": price,
    "initialprice": initialprice,
    "discount": discount,
    "ccu": ccu,
}

st.divider()

if st.button("Compare all models", type="primary", use_container_width=True) or query_compare:
    st.write("### Model predictions")

    try:
        sklearn_artifacts = load_sklearn_artifacts()
    except Exception as exc:
        st.error(f"Could not load Scikit-learn model files: {exc}")
        sklearn_artifacts = {}

    results = []

    for name, artifacts in sklearn_artifacts.items():
        try:
            results.append(predict_sklearn(name, artifacts, input_values))
        except Exception as exc:
            st.error(f"{name} prediction failed: {exc}")

    try:
        results.append(predict_deep_learning(input_values))
    except Exception as exc:
        st.error(f"Deep Learning prediction failed: {exc}")

    if results:
        columns = st.columns(len(results))

        for column, result in zip(columns, results):
            with column:
                render_result(result)

        st.write("### Summary")
        st.dataframe(
            pd.DataFrame(
                {
                    "Model": result["model"],
                    "Prediction": "Successful"
                    if result["prediction"] == 1
                    else "Not successful",
                    "Success probability": None
                    if result["probability"] is None
                    else round(result["probability"], 3),
                    "Features used": result["features"],
                }
                for result in results
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Fill in the game information and click the button to compare all models.")
