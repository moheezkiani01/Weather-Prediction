
# Streamlit used for building the web app interface
# Pandas used for handling data (tables, inputs, etc.)
# Joblit used for loading the trained machine learning model


import streamlit as st
import pandas as pd
import joblib



# Load the trained model from a .pkl file using joblib
# This model is a pipeline that includes preprocessing and a multi-output RandomForestRegressor.

st.set_page_config(page_title="Temperature Prediction Model", page_icon="🌤️")
model = joblib.load("temperature_predictor_v2.pkl")

# Set page metadata such as browser tab title and icon
# Display app title and description

st.title("🌤️ Smart Weather Temperature Predictor")
st.markdown("Use this app to get predictions for **Minimum**, **Maximum**, and **Average** temperatures based on your selected inputs.")


# Take input from the user in the form of parameters like humidity,dew point etc which user want to predict the temperature.
# User can predict the temperature of available cities by any season. 


st.header("🔧 Input Weather Details")
city = st.selectbox("📍 Select City", ["Islamabad", "Lahore", "Karachi", "Peshawar","Quetta", "Gilgit"])
season = st.selectbox("🗓️ Season", ["Winter", "Spring", "Summer", "Autumn"])
year = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2025)
month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
day = st.number_input("📆 Day", min_value=1, max_value=31, value=12)

humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
dew_point = st.slider("🌫️ Dew Point (°C)", -20, 40, 10)
pressure = st.slider("🌡️ Pressure (hPa)", 900, 1100, 1010)
cloud_cover = st.slider("☁️ Cloud Cover (%)", 0, 100, 20)
wspd = st.slider("🌬️ Wind Speed (km/h)", 0, 100, 10)


# Dictionary of city-specific geographical metadata
# Used to provide latitude, longitude, and elevation features for the model

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

#Button to trigger temperature prediction

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

    
    #Use the trained model to predict Tmin, Tmax, and Tavg
    
    try:
        prediction = model.predict(input_df)

        
        # The model gives results in a list inside a list — like a box with one row of answers.
        # We use [0] to open the box and get the three temperature values: Tmin, Tmax, and Tavg.
        
        tmin, tmax, tavg = prediction[0]

        
        #The model predict the temperature in three columns which are min, max, and avg temerature.
        
        st.header("📈 Predicted Temperatures")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Tmin", f"{tmin:.2f} °C", help="Minimum Temperature")
        col2.metric("🔥 Tmax", f"{tmax:.2f} °C", help="Maximum Temperature")
        col3.metric("🌤️ Tavg", f"{tavg:.2f} °C", help="Average Temperature")

        st.success("✅ Prediction complete!")

    except Exception as e:

        
        #In any case model fails to deliver the result it displays the error message.
        
        st.error(f"❌ Prediction failed: {e}")