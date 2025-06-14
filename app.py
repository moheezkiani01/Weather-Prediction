# Streamlit used for building the web app interface
# Pandas used for handling data (tables, inputs, etc.)
# Joblib used for loading the trained machine learning model
# gdown used to fetch model from Google Drive

import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# ========== CONFIGURATION ==========
st.set_page_config(page_title="Temperature Prediction Model", page_icon="🌤️")

# ========== DOWNLOAD MODEL IF NOT EXISTS ==========
model_path = "temperature_predictor_v2.pkl"

if not os.path.exists(model_path):
    st.info("📥 Downloading model file...")
    file_id = "1yWrX0uK0_KyuNowHN-sP-W3AKXwE2Lk9"  # Replace with your file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ========== LOAD MODEL ==========
model = joblib.load(model_path)

# ========== PAGE HEADER ==========
st.title("🌤️ Smart Weather Temperature Predictor")
st.markdown("Use this app to get predictions for **Minimum**, **Maximum**, and **Average** temperatures based on your selected inputs.")

# ========== USER INPUT ==========
st.header("🔧 Input Weather Details")
city = st.selectbox("📍 Select City", ["Islamabad", "Lahore", "Karachi", "Peshawar", "Quetta", "Gilgit"])
season = st.selectbox("🗓️ Season", ["Winter", "Spring", "Summer", "Autumn"])
year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2025)
month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
day = st.number_input("📆 Day", min_value=1, max_value=31, value=12)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
dew_point = st.slider("🌫️ Dew Point (°C)", -20, 40, 10)
pressure = st.slider("🌡️ Pressure (hPa)", 900, 1100, 1010)
cloud_cover = st.slider("☁️ Cloud Cover (%)", 0, 100, 20)
wspd = st.slider("🌬️ Wind Speed (km/h)", 0, 100, 10)

# ========== CITY GEO DATA ==========
city_data = {
    "Islamabad": {"latitude": 33.6844, "longitude": 73.0479, "elevation": 540},
    "Lahore": {"latitude": 31.5204, "longitude": 74.3587, "elevation": 217},
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011, "elevation": 8},
    "Peshawar": {"latitude": 34.0151, "longitude": 71.5805, "elevation": 359},
    "Quetta": {"latitude": 30.1798, "longitude": 66.9750, "elevation": 1680},
    "Gilgit": {"latitude": 35.9208, "longitude": 74.3085, "elevation": 1500}
}

lat = city_data[city]["latitude"]
lon = city_data[city]["longitude"]
elevation = city_data[city]["elevation"]

# ========== PREDICTION ==========
if st.button("🚀 Predict Temperature"):
    input_df = pd.DataFrame([{
        "city": city,
        "season": season,
        "year": year,
        "month": month,
        "day": day,
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "humidity": humidity,
        "dew_point": dew_point,
        "pressure": pressure,
        "cloud_cover": cloud_cover,
        "wspd": wspd
    }])

    try:
        prediction = model.predict(input_df)
        tmin, tmax, tavg = prediction[0]

        st.header("📈 Predicted Temperatures")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Tmin", f"{tmin:.2f} °C", help="Minimum Temperature")
        col2.metric("🔥 Tmax", f"{tmax:.2f} °C", help="Maximum Temperature")
        col3.metric("🌤️ Tavg", f"{tavg:.2f} °C", help="Average Temperature")

        st.success("✅ Prediction complete!")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
