import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, DateTime, Integer
import requests
from datetime import datetime
from geopy.geocoders import Nominatim

# Setup UI
st.set_page_config(page_title="Wakulima Weather Dashboard", layout="wide")
st.title("🌾 Wakulima Weather Intelligence Dashboard")

# Connect to SQLite DB 
@st.cache_resource
def get_engine():
    return create_engine("sqlite:///weather_data.db")

engine = get_engine()

# Define table schema and create table 
metadata = MetaData()
current_weather_table = Table(
    'current_weather', metadata,
    Column('city', String, primary_key=True),
    Column('temperature', Float),
    Column('precipitation', Float),
    Column('windspeed', Float),
    Column('timestamp', String), # Store as String
    Column('latitude', Float),
    Column('longitude', Float),
    Column('region', String)
)
metadata.create_all(engine)


# Load current weather
def load_current_data():
    df = pd.read_sql("SELECT * FROM current_weather ORDER BY timestamp DESC", engine)
    # df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True) # Remove conversion
    return df

# Define cities
cities = ["Nairobi", "Kisumu", "Nakuru", "Mombasa"," Kisii", "Limuru", "Eldoret", "Machakos", "Nyeri", "Garissa "]

# Geocoder
def get_coordinates(city):
    geolocator = Nominatim(user_agent="wakulima_weather")
    location = geolocator.geocode(city + ", Kenya")
    return location.latitude, location.longitude

# Fetch and store weather
@st.cache_data(ttl=600)
def fetch_and_store_cached():
    all_data = []
    for city in cities:
        lat, lon = get_coordinates(city)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url).json()
        weather = res.get("current_weather", {})

        if weather:
            row = {
                "city": city,
                "temperature": weather.get("temperature"),
                "precipitation": weather.get("precipitation", 0),
                "windspeed": weather.get("windspeed"),
                "timestamp": datetime.strptime(weather["time"], "%Y-%m-%dT%H:%M").strftime("%d/%m/%Y %H:%M:%S"), # Store as formatted string
                "latitude": lat,
                "longitude": lon
            }
            all_data.append(row)

    df = pd.DataFrame(all_data)
    with engine.connect() as connection:
        df.to_sql("current_weather", connection, if_exists="replace", index=False)
    print(f"Inserted {len(df)} records into current_weather.")
    return True # Return something to be cached

# Call the cached fetch and store function
fetch_and_store_cached()

df = load_current_data()

if df.empty:
    st.warning("No data found. Wait for the pipeline to collect records.")
    st.stop()


# Map regions to cities
region_map = {
    'Nairobi': 'Nairobi',
    'Kisumu': 'Nyanza',
    'Nakuru': 'Rift Valley',
    'Mombasa': 'Coastal',
    'Kisii': 'Nyanza',
    'Limuru': 'Cental',
    'Eldoret': 'Rift Valley',
    'Machakos': 'Eastern',
    'Nyeri': 'Central',
    'Garissa': 'North Eastern',
}
df['region'] = df['city'].map(region_map)

# Sidebar filters
regions = df['region'].dropna().unique()
selected_region = st.sidebar.selectbox("Select Region", regions)
selected_crop = st.sidebar.selectbox("Select Crop", ['maize', 'beans','sorghum', 'cassava', 'potatoes', 'tomatoes', 'cabbage', 'carrots', 'onions', 'spinach', 'kale', 'groundnuts', 'sunflower', 'millets', 'peanuts', 'Watermelon', 'pumpkin', 'Sweet potatoes', 'sugarcane', 'avocado', 'banana', 'passion_fruit', 'tea', 'coffee', 'macadamia', 'cashew', 'coconut', 'peach', 'mango', 'pawpaw', 'citrus', 'berries', 'herbs','chili','garlic','coriander','basil','oregano','thyme','rosemary','lavender','cilantro','parsley','mint'])

# Crop weather needs (mock)
crop_requirements = {
    'maize': {'temp_min': 18, 'temp_max': 30, 'rain_min': 10, 'rain_max': 100},
    'beans': {'temp_min': 16, 'temp_max': 28, 'rain_min': 15, 'rain_max': 120},
    'sorghum': {'temp_min': 20, 'temp_max': 35, 'rain_min': 20, 'rain_max': 150},
    'cassava': {'temp_min': 25, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'potatoes': {'temp_min': 10, 'temp_max': 25, 'rain_min': 50, 'rain_max': 150},
    'tomatoes': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 100},
    'cabbage': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 120},
    'carrots': {'temp_min': 10, 'temp_max': 20, 'rain_min': 40, 'rain_max': 100},
    'onions': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 80},
    'spinach': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 100},
    'kale': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 120},
    'groundnuts': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 150},
    'sunflower': {'temp_min': 20, 'temp_max': 35, 'rain_min': 20, 'rain_max': 100},
    'millets': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 150},
    'peanuts': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 150},
    'Watermelon': {'temp_min': 25, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'pumpkin': {'temp_min': 20, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'Sweet potatoes': {'temp_min': 25, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'sugarcane': {'temp_min': 20, 'temp_max': 35, 'rain_min': 50, 'rain_max': 300},
    'avocado': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'banana': {'temp_min': 20, 'temp_max': 35, '    rain_min': 30, 'rain_max': 200},
    'passion_fruit': {'temp_min': 20, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'tea': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 120},
    'coffee': {'temp_min': 15, 'temp_max': 25, 'rain_min': 20, 'rain_max': 100},
    'macadamia': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'cashew': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'coconut': {'temp_min': 25, 'temp_max': 35, 'rain_min': 50, 'rain_max': 300},
    'peach': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'mango': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'pawpaw': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'citrus': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'berries': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 120},
    'herbs': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'chili': {'temp_min': 20, 'temp_max': 35, 'rain_min': 30, 'rain_max': 200},
    'garlic': {'temp_min': 10, 'temp_max': 25, 'rain_min': 30, 'rain_max': 120},
    'coriander': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150}, 
    'basil': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'oregano': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'thyme': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'rosemary': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'lavender': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'cilantro': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'parsley': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150},
    'mint': {'temp_min': 15, 'temp_max': 30, 'rain_min': 20, 'rain_max': 150}
}
req = crop_requirements[selected_crop]

# Filtered data
filtered_df = df[df['region'] == selected_region]

# Manually get the latest entry for each city
latest_rows = []
for city in filtered_df['city'].unique():
    city_df = filtered_df[filtered_df['city'] == city].sort_values('timestamp', ascending=False)
    if not city_df.empty:
        latest_rows.append(city_df.iloc[0])

latest = pd.DataFrame(latest_rows)


# Explicitly convert timestamp to string using apply
latest['timestamp'] = latest['timestamp'].apply(str)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🌤 Live Weather", "📅 One-week Weather Forecast", "📊 Daily Summary", "⚠️ Alerts"])

# --- TAB 1: Live Weather ---
with tab1:
    st.subheader("📍 Current Weather Conditions with Map and Icons")

    def get_weather_icon(temp, rain):
        if rain > 20:
            return "🌧️"
        elif temp > 32:
            return "🔥"
        elif temp < 16:
            return "❄️"
        else:
            return "⛅"

    latest['weather_icon'] = latest.apply(lambda x: get_weather_icon(x['temperature'], x['precipitation']), axis=1)

    st.write("Data type of 'timestamp' column in 'latest' DataFrame:", latest['timestamp'].dtype) # Print dtype

    for _, row in latest.iterrows():
        timestamp_str = str(row['timestamp']) # Explicitly convert to string

        st.markdown(
            f"**{row['city']}** ({row['region']}) {row['weather_icon']}  \n"
            f"🌡️ {row['temperature']}°C | 💧 {row['precipitation']} mm | 💨 {row['windspeed']} km/h  \n"
            f"🕒 {timestamp_str}" # Use the explicitly converted string
        )
        st.markdown("---")

    st.map(latest[['latitude', 'longitude']])

# --- TAB 2: One-week Forecast ---
with tab2:
    st.subheader("📅 One-week Weather Forecast")
    forecast_query = f"SELECT * FROM forecast_7day WHERE city IN {tuple(filtered_df['city'].unique())}"
    try:
        forecast_df = pd.read_sql(forecast_query, engine)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], dayfirst=True)
        st.dataframe(forecast_df)
        st.line_chart(forecast_df.set_index("date")[["temperature", "precipitation", "windspeed"]])
    except:
        st.info("No forecast data available yet. Ensure your pipeline has added it.")

# --- TAB 3: Daily Summary ---
with tab3:
    st.subheader("📊 Daily Aggregated Weather (Past 30 Days)")
    try:
        daily_df = pd.read_sql("SELECT * FROM daily_weather ORDER BY date DESC LIMIT 30", engine)
        daily_df['date'] = pd.to_datetime(daily_df['date'], dayfirst=True)
        region_daily = daily_df[daily_df['city'].isin(filtered_df['city'])]
        st.dataframe(region_daily)
        st.line_chart(region_daily.set_index("date")[["avg_temperature", "min_temperature", "max_temperature"]])
    except:
        st.info("No daily summary data available yet.")

# --- TAB 4: Weather Alerts & Crop Fit ---
with tab4:
    st.subheader("⚠️ Weather Alerts")
    alerts = latest[
        (latest["temperature"] > 35) |
        (latest["precipitation"] > 20) |
        (latest["windspeed"] > 50)
    ]
    if alerts.empty:
        st.success("✅ Usual weather conditions.")
    else:
        st.warning("🚨 Unusual weather detected!")
        st.dataframe(alerts)

    st.subheader(f"🌱 Crop Suitability: {selected_crop.capitalize()} in {selected_region}")
    st.markdown(f"🌡️ {req['temp_min']}–{req['temp_max']} °C | 💧 {req['rain_min']}–{req['rain_max']} mm")

    for _, row in latest.iterrows():
        temp_ok = req['temp_min'] <= row['temperature'] <= req['temp_max']
        rain_ok = req['rain_min'] <= row['precipitation'] <= row['rain_max'] #  Check against rain_max
        result = "✅ Suitable" if temp_ok and rain_ok else "❌ Not Ideal"
        st.info(f"{row['city']}: {result}")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ by Davies Ochieng Owuor for Wakulima Inc.")