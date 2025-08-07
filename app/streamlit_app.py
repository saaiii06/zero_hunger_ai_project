## Removed stray model loading line that caused NameError
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Get absolute path to src directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Load models and encoders using absolute paths
crop_model = joblib.load(os.path.join(SRC_DIR, 'crop_recommendation_model.joblib'))
crop_encoder = joblib.load(os.path.join(SRC_DIR, 'crop_label_encoder.joblib'))

# Load models and encoders using absolute paths
crop_model = joblib.load(os.path.join(SRC_DIR, 'crop_recommendation_model.joblib'))
crop_encoder = joblib.load(os.path.join(SRC_DIR, 'crop_label_encoder.joblib'))
yield_model = joblib.load(os.path.join(SRC_DIR, 'yield_prediction_model.joblib'))

st.set_page_config(page_title="Crop Recommendation & Yield Prediction", layout="centered")
st.title("🌱 Crop Recommendation & Yield Prediction")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("Crop Recommendation", "Yield Prediction"))

if section == "Crop Recommendation":
    st.header("Crop Recommendation")
    st.write("Enter soil and weather parameters to get the best crop suggestion.")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    
    if st.button("Recommend Crop"):
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        pred = crop_model.predict(input_data)
        crop_name = crop_encoder.inverse_transform(pred)[0]
        st.success(f"Recommended Crop: **{crop_name.capitalize()}**")

elif section == "Yield Prediction":
    st.header("Yield Prediction")
    st.write("Enter crop and weather details to predict expected yield.")
    crop_list = list(crop_encoder.classes_)
    crop = st.selectbox("Crop", crop_list)
    season = st.selectbox("Season", ["Kharif", "Rabi", "Whole Year"])
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    # Encode categorical variables as in training
    crop_idx = crop_encoder.transform([crop])[0]
    season_map = {"Kharif":0, "Rabi":1, "Whole Year":2}
    season_idx = season_map.get(season, 0)
    input_data = np.array([[rainfall, temperature, crop_idx, season_idx, year]])
    if st.button("Predict Yield"):
        pred_yield = yield_model.predict(input_data)[0]
        st.success(f"Predicted Yield: **{pred_yield:.2f} tons/hectare**")
