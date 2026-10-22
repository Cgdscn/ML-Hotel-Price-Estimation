import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==============================
# LOAD MODEL FILES
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

model = joblib.load(os.path.join(MODEL_DIR, "price_model.pkl"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "price_preprocessor.pkl"))
model_info = joblib.load(os.path.join(MODEL_DIR, "price_model_info.pkl"))
gui_options = joblib.load(os.path.join(MODEL_DIR, "gui_options.pkl"))

categorical_features = model_info["categorical_features"]
numeric_features = model_info["numeric_features"]
binary_features = model_info["binary_features"]

# ==============================
# PRICE CATEGORY FUNCTION
# ==============================

def get_price_category(price):
    if price <= 25000:
        return "Ekonomik", "0 – 25.000 TL"
    elif price <= 40000:
        return "Standart", "25.001 – 40.000 TL"
    elif price <= 70000:
        return "Premium", "40.001 – 70.000 TL"
    else:
        return "Lüks", "70.001 TL ve üzeri"

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Hotel Price Prediction",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Scandinavian Hotel Price Prediction")
st.write(
    "Bu uygulama otel bilgilerine göre önce tahmini fiyatı hesaplar, ardından fiyat kategorisini gösterir."
)

st.info(f"Loaded model: {model_info['model_name']}")

# ==============================
# USER INPUTS
# ==============================

st.header("Hotel Information")

col1, col2, col3 = st.columns(3)

with col1:
    country = st.selectbox("Country", gui_options["countries"])

with col2:
    city = st.selectbox("City", gui_options["cities"])

with col3:
    season = st.selectbox("Season", gui_options["seasons"])

st.header("Numerical Features")

col4, col5, col6, col7 = st.columns(4)

with col4:
    score = st.number_input(
        "Hotel Score",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.1
    )

with col5:
    review_count = st.number_input(
        "Review Count",
        min_value=0,
        value=500,
        step=10
    )

with col6:
    distance_to_center = st.number_input(
        "Distance to Center",
        min_value=0.0,
        value=2.0,
        step=0.1
    )

with col7:
    star_rating = st.number_input(
        "Star Rating",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=0.5
    )

st.header("Hotel Amenities")

amenity_values = {}

cols = st.columns(4)

for i, amenity in enumerate(binary_features):
    clean_name = (
        amenity
        .replace("has_", "")
        .replace("is_", "")
        .replace("no_", "no ")
        .replace("_", " ")
        .title()
    )

    with cols[i % 4]:
        amenity_values[amenity] = int(st.checkbox(clean_name, value=False))

amenity_count = sum(amenity_values.values())

# ==============================
# CREATE INPUT DATAFRAME
# ==============================

input_data = {
    "country": country,
    "city": city,
    "season": season,
    "score": score,
    "review_count": review_count,
    "distance_to_center": distance_to_center,
    "star_rating": star_rating,
    "amenity_count": amenity_count
}

for amenity in binary_features:
    input_data[amenity] = amenity_values[amenity]

input_df = pd.DataFrame([input_data])

# ==============================
# PREDICTION
# ==============================

st.header("Prediction Result")

if st.button("Predict Hotel Price"):

    input_processed = preprocessor.transform(input_df)

    predicted_log_price = model.predict(input_processed)[0]

    predicted_price = np.expm1(predicted_log_price)

    predicted_price = max(predicted_price, 0)

    category, price_range = get_price_category(predicted_price)

    formatted_price = f"{predicted_price:,.0f}".replace(",", ".")

    st.success(f"Tahmini Fiyat: {formatted_price} TL")
    st.success(f"Tahmini Kategori: {category}")
    st.success(f"Beklenen Fiyat Aralığı: {price_range}")

    st.subheader("Input Summary")
    st.dataframe(input_df)