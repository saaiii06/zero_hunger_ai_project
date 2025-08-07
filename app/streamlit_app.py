## Removed stray model loading line that caused NameError
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stylable_container import stylable_container



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


# --- Custom Theme ---
st.set_page_config(page_title="Crop Recommendation & Yield Prediction", layout="wide", page_icon="🌱")


with st.sidebar:
    st.markdown("## 🌱 Zero Hunger AI")
    st.markdown("---")
    section = st.radio("Go to", ("Crop Recommendation", "Yield Prediction"))


# --- Custom CSS for dark theme only ---
dark_css = '''
<style>
body, .stApp { background: #181818 !important; color: #f7fafc !important; }
.stButton>button { background: #00bfae; color: #181818; border-radius: 8px; font-weight: bold; }
.stButton>button:hover { background: #008c7a; }
.stSelectbox>div>div { border-radius: 8px; }
.stNumberInput>div>div { border-radius: 8px; }
.stAlert { border-radius: 8px; }
.stSidebar { background: #23272f !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #00bfae; }
</style>
'''
st.markdown(dark_css, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-bottom: 2rem;'>
    <h1 style='font-size:2.5rem;'>🌱 Crop Recommendation & Yield Prediction</h1>
    <p style='font-size:1.2rem; color:#888;'>Empowering farmers with AI-driven insights for a sustainable future.</p>
</div>
""", unsafe_allow_html=True)

if section == "Crop Recommendation":
    with stylable_container(key="crop_rec", css_styles="background:rgba(76,175,80,0.07);padding:2rem 2rem 1rem 2rem;border-radius:16px;margin-bottom:2rem;"):
        st.subheader("🌾 Crop Recommendation")
        st.write("Enter soil and weather parameters to get the best crop suggestion.")
        col1, col2, col3 = st.columns(3)
        with col1:
            N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
            P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
            K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        with col3:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        if st.button("Recommend Crop"):
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            pred = crop_model.predict(input_data)
            crop_name = crop_encoder.inverse_transform(pred)[0]
            st.success(f"Recommended Crop: **{crop_name.capitalize()}**")

elif section == "Yield Prediction":
    with stylable_container(key="yield_pred", css_styles="background:rgba(0,191,174,0.07);padding:2rem 2rem 1rem 2rem;border-radius:16px;margin-bottom:2rem;"):
        st.subheader("🌾 Yield Prediction")
        st.write("Enter crop and weather details to predict expected yield.")
        col1, col2, col3 = st.columns(3)
        with col1:
            state = st.selectbox(
                "State",
                [
                    "Karnataka", "Punjab", "Maharashtra", "Uttar Pradesh", "Kerala", "Gujarat", "Andhra Pradesh", "Madhya Pradesh",
                    "West Bengal", "Tamil Nadu", "Rajasthan", "Bihar", "Odisha", "Assam", "Haryana", "Telangana", "Jharkhand", "Chhattisgarh"
                ]
            )
            district = st.text_input("District", "Bangalore")
            crop_list = list(crop_encoder.classes_)
            crop = st.selectbox("Crop", crop_list)
        with col2:
            season = st.selectbox("Season", ["Kharif", "Rabi", "Whole Year"])
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        with col3:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        # Encode categorical variables as in training
        from sklearn.preprocessing import LabelEncoder
        # For demo, use same encoders as in training (should be loaded from joblib in production)
        state_le = LabelEncoder(); state_le.classes_ = np.array(["Andhra Pradesh", "Gujarat", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Punjab", "Uttar Pradesh"])
        district_le = LabelEncoder(); district_le.classes_ = np.array(["Ahmedabad", "Bangalore", "Bhopal", "Guntur", "Kanpur", "Ludhiana", "Nagpur", "Trivandrum"])
        crop_idx = crop_encoder.transform([crop])[0]
        season_map = {"Kharif":0, "Rabi":1, "Whole Year":2}
        season_idx = season_map.get(season, 0)
        # Encode state and district
        try:
            state_idx = state_le.transform([state])[0]
        except Exception:
            state_idx = 0
        try:
            district_idx = district_le.transform([district])[0]
        except Exception:
            district_idx = 0
        input_data = np.array([[state_idx, district_idx, rainfall, temperature, humidity, ph, crop_idx, season_idx, year]])
        if st.button("Predict Yield"):
            pred_yield = yield_model.predict(input_data)[0]
            st.success(f"Predicted Yield: **{pred_yield:.2f} tons/hectare**")
