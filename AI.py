from pathlib import Path
from datetime import date, timedelta
import json

import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from auth import authentication_page
from feedback import feedback_page

st.set_page_config(
    page_title="Smart Irrigation AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    authentication_page()
    st.stop()

if st.session_state.authenticated:
    with st.sidebar:
        st.write(f"👤 Welcome, {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()


MODEL_FILE = Path("models/soil_moisture_model.joblib")
METRICS_FILE = Path("models/model_metrics.json")
DATA_FILE = Path("data/kolkata_unified_dataset.csv")



@st.cache_data(ttl=900)
def get_live_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("current"), None
    except (requests.RequestException, ValueError) as error:
        return None, str(error)


CROPS = {
    "Wheat": {"kc": 1.05},
    "Rice": {"kc": 1.20},
    "Maize": {"kc": 1.10},
    "Cotton": {"kc": 1.15},
    "Vegetables": {"kc": 1.00},
    "Sugarcane": {"kc": 1.25},
    "Pulses": {"kc": 0.90},
}

FEATURES = [
    "Volumetric_Moisture_Pct",
    "Sensor_Board_Temperature_C",
    "Raw_Capacitive_ADC",
    "Sensor_Voltage_V",
    "Atm_Pressure_inHg",
    "Atm_Temperature_C",
    "Soil_Temperature_C",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
]

st.sidebar.title("🌱 Smart Irrigation AI")
st.sidebar.caption("Farm water management")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Irrigation Planner",
        "AI Predictions",
        "7-Day Plan",
        "Reports & Insights",
        "Feedback & Suggestions",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("🏡 Farm Details")
farm_name = st.sidebar.text_input("Farm name", "My Farm")
crop = st.sidebar.selectbox("Select crop", list(CROPS.keys()))
area = st.sidebar.number_input(
    "Farm area (acres)", min_value=0.1, max_value=10000.0, value=1.0, step=0.1
)

st.sidebar.subheader("🌦️ Live Weather Settings")

farm_latitude = st.sidebar.number_input(
    "Farm Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=31.22,
    format="%.4f"
)

farm_longitude = st.sidebar.number_input(
    "Farm Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=75.77,
    format="%.4f"
)

soil_type = st.sidebar.selectbox("Soil type", ["Loamy", "Sandy", "Clay", "Silty"])
irrigation_method = st.sidebar.selectbox("Irrigation method", ["Drip", "Sprinkler", "Flood"])

st.sidebar.subheader("🌦️ Current Conditions")
soil_moisture_percent = st.sidebar.number_input(
    "Measured soil moisture (% VWC)",
    min_value=0.0,
    max_value=100.0,
    value=12.0,
    step=0.1,
    help="Enter a measured volumetric soil-moisture percentage. The training dataset values were about 8.9% to 19.6%.",
)
temperature = st.sidebar.number_input("Air temperature (°C)", -10.0, 60.0, 30.0, step=0.5)
humidity = st.sidebar.slider("Relative humidity (%)", 0, 100, 55)
rainfall = st.sidebar.number_input("Expected rainfall (mm)", 0.0, 500.0, 0.0, step=0.5)
et0 = st.sidebar.number_input("Reference evapotranspiration ET₀ (mm/day)", 0.0, 20.0, 4.0, step=0.1)
pump_flow = st.sidebar.number_input("Pump flow (litres/minute)", 1.0, 100000.0, 100.0, step=10.0)

st.sidebar.subheader("🧪 Sensor Inputs for AI")
sensor_board_temp = st.sidebar.number_input("Sensor board temperature (°C)", -10.0, 80.0, 25.0, step=0.5)
raw_adc = st.sidebar.number_input("Raw capacitive ADC reading", 0.0, 100000.0, 2500.0, step=10.0)
sensor_voltage = st.sidebar.number_input("Sensor voltage (V)", 0.0, 10.0, 2.5, step=0.01, format="%.2f")
pressure = st.sidebar.number_input("Atmospheric pressure (inHg)", 20.0, 35.0, 29.9, step=0.1)
soil_temp = st.sidebar.number_input("Soil temperature (°C)", -10.0, 60.0, 25.0, step=0.5)
farm_id = st.sidebar.selectbox("Dataset farm profile", ["Kolkata_Farm_1", "Kolkata_Farm_2"])
hour = st.sidebar.slider("Hour of day for sensor reading", 0, 23, 12)
day_of_year = st.sidebar.slider("Day of year (1–365)", 1, 365, date.today().timetuple().tm_yday if date.today().timetuple().tm_yday <= 365 else 365)
moisture_threshold = st.sidebar.number_input(
    "Low-moisture alert threshold (% VWC)",
    min_value=0.0,
    max_value=100.0,
    value=11.0,
    step=0.5,
    help="An editable demonstration threshold, not a universal agronomic recommendation.",
)

crop_kc = CROPS[crop]["kc"]
crop_water_need = et0 * crop_kc
effective_rainfall = rainfall * 0.8
net_water_need = max(0.0, crop_water_need - effective_rainfall)
area_square_metres = area * 4046.856
net_water_litres = net_water_need * area_square_metres
efficiency = {"Drip": 0.90, "Sprinkler": 0.75, "Flood": 0.55}[irrigation_method]
gross_water_litres = net_water_litres / efficiency
pump_minutes = gross_water_litres / pump_flow

if soil_moisture_percent < moisture_threshold:
    moisture_status = "Below selected threshold"
    recommendation = (
        "The entered moisture is below your selected alert threshold. "
        "Inspect the crop and soil, verify the sensor reading, and consider irrigation."
    )
elif soil_moisture_percent < moisture_threshold + 3:
    moisture_status = "Near selected threshold"
    recommendation = (
        "Moisture is near the selected alert threshold. Check crop stage, soil type, "
        "and local conditions before deciding whether to irrigate."
    )
else:
    moisture_status = "Above selected threshold"
    recommendation = "Moisture is above the selected alert threshold. Continue monitoring and verify field conditions."

# Load the trained machine learning model and metrics
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

try:
    model_bundle = load_model()
    model = model_bundle["model"] if isinstance(model_bundle, dict) and "model" in model_bundle else model_bundle
    model_features = model_bundle.get("features", FEATURES) if isinstance(model_bundle, dict) else FEATURES
    model_error = None
except Exception as e:
    model = None
    model_features = FEATURES
    model_error = str(e)

try:
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)
except Exception:
    metrics = {}

# Prepare the model input
day_angle = 2 * np.pi * day_of_year / 365
hour_angle = 2 * np.pi * hour / 24

sensor_input = {
    "Volumetric_Moisture_Pct": soil_moisture_percent,
    "Sensor_Board_Temperature_C": sensor_board_temp,
    "Raw_Capacitive_ADC": raw_adc,
    "Sensor_Voltage_V": sensor_voltage,
    "Atm_Pressure_inHg": pressure,
    "Atm_Temperature_C": temperature,
    "Soil_Temperature_C": soil_temp,
    "Hour_sin": np.sin(hour_angle),
    "Hour_cos": np.cos(hour_angle),
    "Day_sin": np.sin(day_angle),
    "Day_cos": np.cos(day_angle),
}

predicted_moisture = None

if model is not None:
    try:
        input_df = pd.DataFrame([sensor_input])[model_features]
        predicted_moisture = float(model.predict(input_df)[0])
        predicted_moisture = float(np.clip(predicted_moisture, 0, 100))
    except Exception as e:
        model_error = str(e)

# Page 1: Dashboard

if page == "Dashboard":
    st.markdown("""
    <style>
        .stApp {
        background:
            radial-gradient(ellipse at 5% 10%, #e5eadb 0%, transparent 40%),
            linear-gradient(135deg, #f5f3e8, #e9eee2);
    }
    
    .dashboard-hero {
        background: linear-gradient(120deg, #263d2b, #496344);
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #71866a;
        margin-bottom: 25px;
    }

    .dashboard-hero h1 {
        color: #f4f0df;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .dashboard-hero p {
        color: #d8e2ca;
        font-size: 15px;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: #edf0e5;
        border: 1px solid #cbd5bd;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetric"] label {
        color: #53634b !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #263d2b !important;
        font-weight: 750;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-hero">
        <h1>🌾 AgriSmart AI Dashboard</h1>
        <p>Monitor your farm, understand your soil and make smarter irrigation decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    
    st.markdown("## 🌤️ Live Farm Weather")
    st.caption("Real-time atmospheric conditions for your selected farm location.")

    weather_data, weather_error = get_live_weather(
        farm_latitude, farm_longitude
    )

    if weather_data:
        weather_col1, weather_col2, weather_col3 = st.columns(3)

        weather_col1.metric(
            "Air Temperature",
            f"{weather_data.get('temperature_2m', 'N/A')} °C"
        )
        weather_col2.metric(
            "Humidity",
            f"{weather_data.get('relative_humidity_2m', 'N/A')}%"
        )
        weather_col3.metric(
            "Current Precipitation",
            f"{weather_data.get('precipitation', 'N/A')} mm"
        )

        st.caption(
            "Weather is retrieved from Open-Meteo. "
            "Precipitation is a weather measurement, not a soil-moisture reading."
        )
    else:
        st.warning(
            "Live weather is currently unavailable. "
            "Check your internet connection and farm coordinates."
        )

    
    st.title("🌱 Smart Agriculture Irrigation Planner")
    st.write(f"Welcome, **{farm_name}**!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Soil Moisture", f"{soil_moisture_percent:.1f}%")
    col2.metric("Moisture Status", moisture_status)
    col3.metric("Selected Crop", crop)
        
    st.markdown("### 🌱 Soil Moisture Level")

    moisture_progress = min(
        max(soil_moisture_percent / 100, 0.0),
        1.0
    )

    st.progress(moisture_progress)

    st.caption(
        f"Current soil moisture: {soil_moisture_percent:.1f}%"
    )

    # st.subheader("Farm Overview")
    # st.write(f"**Farm area:** {area:.1f} acres")
    # st.write(f"**Soil type:** {soil_type}")
    # st.write(f"**Irrigation method:** {irrigation_method}")
    
    st.markdown("## 🌿 Farm Overview")
    st.caption("Essential information about your selected farm.")

    farm_col1, farm_col2, farm_col3 = st.columns(3)

    with farm_col1:
        st.metric("📐 Farm Area", f"{area:.1f} acres")

    with farm_col2:
        st.metric("🌱 Soil Type", soil_type)

    with farm_col3:
        st.metric("💧 Irrigation Method", irrigation_method)

    # st.subheader("Recommendation")
    
    st.markdown("## 💧 Smart Irrigation Recommendation")
    
    st.markdown("## 🌿 Crop Health & Irrigation Insights")
    st.caption("A quick overview of your selected crop and farming conditions.")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(f"🌾 **Selected Crop:** {crop}")
        st.info(f"🌱 **Soil Type:** {soil_type}")

    with insight_col2:
        st.info(f"💧 **Irrigation Method:** {irrigation_method}")
        st.info(
            "🔍 **Field Check:** Inspect soil and crop conditions "
            "before making irrigation decisions."
        )
    
    st.caption("AI-assisted guidance based on your current soil moisture.")
    
    if soil_moisture_percent < moisture_threshold:
        st.warning(recommendation)
    else:
        st.success(recommendation)

    # st.subheader("🚨 Smart Irrigation Alerts")
    
    st.markdown("## 🚨 Smart Irrigation Alerts")
    st.caption("Important notifications to help you monitor changing soil conditions.")

    if soil_moisture_percent < moisture_threshold:
        st.warning(
            "⚠️ LOW SOIL MOISTURE: Check field conditions and consider "
            "whether irrigation is needed."
        )
    else:
        st.success("✅ Current soil moisture is above your alert threshold.")

    if predicted_moisture is not None:
        if predicted_moisture < moisture_threshold:
            st.warning(
                "🔔 AI ALERT: Predicted soil moisture may fall below "
                "your threshold in 15 minutes. Verify conditions before acting."
            )
        else:
            st.info("🤖 AI alert: Predicted moisture is above your threshold.")

    if rainfall > 0:
        st.info(
            f"🌧️ Rainfall input: {rainfall:.1f} mm. "
            "Consider this rainfall when reviewing irrigation needs."
        )

    if predicted_moisture is not None:
        # st.metric("AI Predicted Moisture in 15 Minutes", f"{predicted_moisture:.2f}%")
        st.markdown("## 🤖 AI Soil Moisture Forecast")
        st.caption("Estimated soil moisture for the next 15 minutes.")
        st.metric(
            label="Predicted Soil Moisture",
            value=f"{predicted_moisture:.2f}%",
            help="This is an AI model prediction, not a live sensor reading.",
        )
    elif model_error:
        st.error(f"Unable to generate the soil-moisture forecast: {model_error}")

    st.caption("Predictions use entered sensor values; no live sensor connection is configured.")
    
elif page == "Feedback & Suggestions":
    feedback_page()

# Page 2: Irrigation Planner
    st.divider()

    st.markdown(
        """
        <div style="
            background: linear-gradient(120deg, #263d2b, #496344);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            color: #f4f0df;
        ">
            <h4>🌾 AgriSmart AI</h4>
            <p style="margin-bottom: 0;">
                Smart Decisions • Efficient Irrigation • Sustainable Agriculture
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif page == "Irrigation Planner":
    
    st.title("💧 Irrigation Planner")
    st.write("Estimated water requirement based on ET₀, crop coefficient, rainfall, area, and irrigation efficiency.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Crop Water Need", f"{crop_water_need:.2f} mm/day")
    c2.metric("Net Water Need", f"{net_water_need:.2f} mm/day")
    c3.metric("Estimated Water", f"{gross_water_litres:,.0f} litres/day")

    st.metric("Estimated Pump Running Time", f"{pump_minutes:.1f} minutes/day")
    st.info("This is an educational estimate. Actual irrigation needs depend on crop stage, soil, weather, and field measurements.")

# Page 3: AI Predictions
elif page == "AI Predictions":
    st.title("🤖 AI Soil-Moisture Prediction")

    if model_error:
        st.error(f"Could not load or run the model: {model_error}")
    elif predicted_moisture is not None:
        st.metric("Predicted Soil Moisture in 15 Minutes", f"{predicted_moisture:.3f}%")
        change = predicted_moisture - soil_moisture_percent
        st.metric("Predicted Change", f"{change:+.3f} percentage points")
        
        st.subheader("📈 Soil Moisture Prediction Graph")

        history = pd.read_csv(DATA_FILE)
        history["Timestamp"] = pd.to_datetime(history["Timestamp"], errors="coerce")
        history = history.dropna(subset=["Timestamp", "Volumetric_Moisture_Pct"])

        selected_history = history[history["Farm_ID"] == farm_id]
        selected_history = selected_history.tail(96)

        if not selected_history.empty:
            chart_data = selected_history.set_index("Timestamp")[
                ["Volumetric_Moisture_Pct"]
            ].rename(columns={
                "Volumetric_Moisture_Pct": "Historical Soil Moisture (%)"
            })

            st.line_chart(chart_data, use_container_width=True)

            st.caption(
                "Shows the last 96 recorded sensor readings (up to 24 hours "
                "at 15-minute intervals) for the selected farm. "
                "The AI prediction is shown separately above."
            )
        else:
            st.info("No historical readings are available for this farm.")

        if predicted_moisture < moisture_threshold:
            st.warning("The predicted moisture is below your selected threshold. Verify the sensor reading and inspect field conditions.")
        else:
            st.success("The predicted moisture is at or above your selected threshold.")

        # st.subheader("Model Evaluation")
        # for key in ["linear_regression", "random_forest", "baseline"]:
        #     if key in metrics:
        #         st.write(f"**{key.replace('_', ' ').title()}**")
        #         st.json(metrics[key])

        # st.caption("The prediction horizon is 15 minutes. The model was evaluated on a held-out time period from the supplied dataset.")
        
        st.markdown("## 📊 Model Performance Evaluation")
        st.caption(
            "Compare the performance of different machine learning models "
            "on the held-out test dataset."
        )

        model_names = {
            "linear_regression": "Linear Regression",
            "random_forest": "Random Forest",
            "baseline": "Persistence Baseline",
        }

        available_models = [
            key for key in model_names if key in metrics
        ]

        if available_models:
            best_model = min(
                available_models,
                key=lambda key: metrics[key].get("MAE", float("inf"))
            )

            for key in available_models:
                result = metrics[key]

                with st.container(border=True):
                    if key == best_model:
                        st.markdown(
                            f"### 🏆 {model_names[key]} — Best MAE"
                        )
                    else:
                        st.markdown(f"### 🤖 {model_names[key]}")

                    metric_col1, metric_col2, metric_col3 = st.columns(3)

                    metric_col1.metric(
                        "MAE",
                        f"{result.get('MAE', 0):.5f}"
                    )

                    metric_col2.metric(
                        "RMSE",
                        f"{result.get('RMSE', 0):.5f}"
                    )

                    metric_col3.metric(
                        "R² Score",
                        f"{result.get('R2', 0):.5f}"
                    )

        st.caption(
            "Prediction horizon: 15 minutes. Evaluation uses a held-out "
            "time period from the supplied dataset. Lower MAE and RMSE "
            "are better; higher R² is better."
        )
        
    else:
        st.warning("No prediction is available. Check that the trained model exists.")


    st.subheader("📊 AI Model Comparison Dashboard")

    comparison_rows = []

    model_names = {
        "linear_regression": "Linear Regression",
        "random_forest": "Random Forest",
        "baseline": "Persistence Baseline",
    }

    for key, display_name in model_names.items():
        if key in metrics:
            comparison_rows.append({
                "Model": display_name,
                "MAE": metrics[key].get("MAE"),
                "RMSE": metrics[key].get("RMSE"),
                "R² Score": metrics[key].get("R2"),
            })

    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df.round(5),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Lower MAE and RMSE are better. Higher R² is better. "
            "The persistence baseline predicts that moisture stays unchanged."
        )

        st.subheader("MAE Comparison")
        st.bar_chart(comparison_df.set_index("Model")[["MAE"]])

        st.subheader("RMSE Comparison")
        st.bar_chart(comparison_df.set_index("Model")[["RMSE"]])
    else:
        st.warning("Model comparison metrics are not available.")


# Page 4: 7-Day Plan
elif page == "7-Day Plan":
    st.title("📅 7-Day Irrigation Planning Estimate")

    rows = []
    for i in range(7):
        forecast_rain = rainfall if i == 0 else 0
        daily_net = max(0, et0 * crop_kc - forecast_rain * 0.8)
        daily_litres = daily_net * area_square_metres / efficiency
        rows.append({
            "Day": (date.today() + timedelta(days=i)).strftime("%d %b %Y"),
            "Estimated ET₀ (mm)": et0,
            "Assumed Rainfall (mm)": forecast_rain,
            "Estimated Water (litres)": round(daily_litres),
            "Current Moisture (%)": soil_moisture_percent if i == 0 else None,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("This is a scenario estimate, not a weather forecast. Rainfall after today is assumed to be zero.")

# Page 5: Reports & Insights
elif page == "Reports & Insights":
    st.title("📊 Reports & Insights")

    st.subheader("Current Farm Inputs")
    # st.write({
    #     "Farm": farm_name,
    #     "Crop": crop,
    #     "Area (acres)": area,
    #     "Soil type": soil_type,
    #     "Irrigation method": irrigation_method,
    #     "Moisture (%)": soil_moisture_percent,
    #     "Temperature (°C)": temperature,
    #     "Humidity (%)": humidity,
    #     "Expected rainfall (mm)": rainfall,
    # })
    
    st.markdown("## 🌾 Farm Overview")
    st.caption("A quick summary of your current farm configuration.")

    report_col1, report_col2, report_col3 = st.columns(3)

    with report_col1:
        st.metric("🌱 Selected Crop", crop)
        st.metric("📐 Farm Area", f"{area:.1f} acres")

    with report_col2:
        st.metric("💧 Soil Moisture", f"{soil_moisture_percent:.1f}%")
        st.metric("🌍 Soil Type", soil_type)

    with report_col3:
        st.metric("🚜 Irrigation Method", irrigation_method)
        st.metric("🌡️ Temperature", f"{temperature:.1f} °C")
    
  
    report_data = {
        "Farm Name": farm_name,
        "Crop": crop,
        "Farm Area (acres)": area,
        "Soil Type": soil_type,
        "Irrigation Method": irrigation_method,
        "Current Soil Moisture (%)": soil_moisture_percent,
        "Moisture Status": moisture_status,
        "Air Temperature (C)": temperature,
        "Relative Humidity (%)": humidity,
        "Expected Rainfall (mm)": rainfall,
        "ET0 (mm/day)": et0,
        "Estimated Crop Water Need (mm/day)": crop_water_need,
        "Estimated Net Water Need (mm/day)": net_water_need,
        "Estimated Water Required (litres/day)": gross_water_litres,
        "Estimated Pump Time (minutes/day)": pump_minutes,
        "AI Predicted Soil Moisture (%)": predicted_moisture,
        "Recommendation": recommendation,
    }

    report_df = pd.DataFrame([report_data])
    

    
    st.markdown("## 🌾 Crop-Wise Irrigation Comparison")
    st.caption(
        "Compare estimated irrigation needs across different crops "
        "to support better water-management decisions."
    )

    crop_comparison = []

    for crop_name, crop_info in CROPS.items():
        crop_water_need = et0 * crop_info["kc"]
        net_water_need = max(0.0, (et0 * crop_info["kc"]) - rainfall)

        daily_litres = net_water_need * area * 4046.86

        crop_comparison.append({
            "Crop": crop_name,
            "Daily Water Need (mm)": round(crop_water_need, 2),
            "After Rainfall (mm)": round(net_water_need, 2),
            "Estimated Water (litres/day)": round(daily_litres, 0)
        })

    crop_comparison_df = pd.DataFrame(crop_comparison)

    st.dataframe(
        crop_comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        crop_comparison_df.set_index("Crop")[
            "Estimated Water (litres/day)"
        ]
    )

    st.caption(
        "Estimates use configured crop water needs, selected farm area, "
        "and entered rainfall. Actual irrigation needs vary with crop "
        "stage, soil, weather, and field conditions."
    )
    
    # st.subheader("📊 Historical Data Analysis")
    st.markdown("## 📈 Historical Data Analysis")
    st.caption(
    "Explore historical soil moisture records, monitor trends, "
    "and understand changes in field conditions over time."
)

    try:
        historical_df = pd.read_csv(DATA_FILE)
        historical_df["Timestamp"] = pd.to_datetime(
            historical_df["Timestamp"], errors="coerce"
        )
        historical_df = historical_df.dropna(
            subset=["Timestamp", "Volumetric_Moisture_Pct"]
        )

        farm_options = ["All Farms"] + sorted(
            historical_df["Farm_ID"].dropna().unique().tolist()
        )

        selected_farm = st.selectbox(
            "Choose farm for historical analysis",
            farm_options,
            key="historical_analysis_farm",
        )

        if selected_farm != "All Farms":
            historical_df = historical_df[
                historical_df["Farm_ID"] == selected_farm
            ]

        if not historical_df.empty:
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Sensor Records", f"{len(historical_df):,}")
            col2.metric(
                "Average Soil Moisture",
                f"{historical_df['Volumetric_Moisture_Pct'].mean():.2f}%"
            )
            col3.metric(
                "Minimum Soil Moisture",
                f"{historical_df['Volumetric_Moisture_Pct'].min():.2f}%"
            )
            col4.metric(
                "Maximum Soil Moisture",
                f"{historical_df['Volumetric_Moisture_Pct'].max():.2f}%"
            )

            st.write("**Soil Moisture Over Time**")
            
            moisture_values = historical_df[
                "Volumetric_Moisture_Pct"
            ]

            first_moisture = moisture_values.iloc[0]
            latest_moisture = moisture_values.iloc[-1]
            moisture_change = latest_moisture - first_moisture

            if moisture_change > 0:
                st.success(
                    f"📈 Soil moisture increased by "
                    f"{moisture_change:.2f} percentage points."
                )
            elif moisture_change < 0:
                st.warning(
                    f"📉 Soil moisture decreased by "
                    f"{abs(moisture_change):.2f} percentage points."
                )
            else:
                st.info("➡️ Soil moisture remained unchanged.")

            trend_df = historical_df[
                ["Timestamp", "Volumetric_Moisture_Pct"]
            ].sort_values("Timestamp")

            st.line_chart(
                trend_df.set_index("Timestamp"),
                use_container_width=True,
            )

            st.write("**Soil Moisture Statistics**")

            stats_df = historical_df[
                "Volumetric_Moisture_Pct"
            ].describe().to_frame(name="Soil Moisture (%)")

            st.dataframe(stats_df.round(3), use_container_width=True)
            
            st.markdown("### 📝 Farm Data Summary")

            latest_reading = trend_df.iloc[-1][
                "Volumetric_Moisture_Pct"
            ]

            st.info(
                f"Your selected farm has {len(historical_df):,} sensor "
                f"records. Average soil moisture is "
                f"{historical_df['Volumetric_Moisture_Pct'].mean():.2f}%. "
                f"The latest recorded moisture is "
                f"{latest_reading:.2f}%."
            )

            st.write("**Recent Sensor Records**")

            st.dataframe(
                historical_df.tail(20).sort_values(
                    "Timestamp", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No historical records were found for this selection.")

    except Exception as e:
        st.error(f"Could not load historical data: {e}")




    # st.subheader("📥 Download Farm Report")
    st.markdown("## 📥 Download Your Farm Report")
    st.caption(
    "Export your current farm configuration and irrigation information "
    "for documentation and future reference."
)

    csv_data = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Farm Report (CSV)",
        data=csv_data,
        file_name="smart_irrigation_farm_report.csv",
        mime="text/csv",
    )


    if metrics:
        st.markdown("## 🤖 Model Performance Summary")
        st.caption("Key results from the machine-learning evaluation.")

        best_model = metrics.get("best_ml_model", "Not available")
        best_metrics = metrics.get("best_ml_model_metrics", {})

        st.info(f"🏆 Best Machine Learning Model: **{best_model}**")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "MAE",
            f"{best_metrics.get('MAE', 0):.5f}"
        )

        metric_col2.metric(
            "RMSE",
            f"{best_metrics.get('RMSE', 0):.5f}"
        )

        metric_col3.metric(
            "R² Score",
            f"{best_metrics.get('R2', 0):.5f}"
        )

        st.caption(
            "Lower MAE and RMSE are better. Higher R² is generally better. "
            "These results do not guarantee performance on new farms."
        )

    st.markdown("## 🌱 Moisture Status")
    st.write(moisture_status)
    st.write(recommendation)

# Page 6: About Project

elif page == "About Project":
    
    st.title("🌾 About AgriSmart AI")

    st.markdown("""
    ### AI-Based Smart Agriculture Irrigation Planner

    **AgriSmart AI** is an educational AI/ML project designed to
    support smarter irrigation decisions using soil moisture data,
    weather information, and crop characteristics.

    ### 🎯 Project Objective

    - Reduce unnecessary water usage.
    - Monitor soil moisture conditions.
    - Estimate future soil moisture.
    - Compare irrigation requirements across crops.
    - Support informed agricultural decisions.

    ### ⚙️ How the System Works

    1. The user enters farm and crop information.
    2. The system processes soil moisture and weather inputs.
    3. A machine-learning model estimates soil moisture
       15 minutes ahead.
    4. Rule-based logic generates irrigation guidance.
    5. Historical analysis and reports help users understand
       farm conditions.

    ### 🚀 Main Features

    - User registration and login.
    - Farm and crop configuration.
    - AI-based soil moisture prediction.
    - Smart irrigation recommendations and alerts.
    - Live weather integration when internet is available.
    - Seven-day estimated irrigation plan.
    - Crop-wise water requirement comparison.
    - Historical sensor data analysis.
    - Downloadable farm reports.
    - User feedback and ratings.

    ### 🧠 Artificial Intelligence and Machine Learning

    The project evaluates:
    - Linear Regression
    - Random Forest Regression
    - Persistence Baseline

    The models are evaluated using:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² Score

    The prediction target is measured soil moisture
    15 minutes into the future.

    ### 🛠️ Technologies Used

    - Python
    - Streamlit
    - Pandas and NumPy
    - Scikit-learn
    - Plotly
    - Requests
    - Joblib

    ### 📊 Dataset

    The project uses a historical soil-moisture and weather
    dataset for model development and evaluation.

    The evaluation uses a chronological train-test split.
    Results from this dataset do not guarantee performance
    on new farms or different environmental conditions.

    ### ⚠️ Limitations

    - The application is not connected to physical soil sensors.
    - Irrigation recommendations are estimates, not guaranteed
      agricultural advice.
    - Weather availability depends on an internet connection.
    - Model performance may differ on unseen farms.
    - The seven-day plan is an estimate, not a weather forecast.

    ### 🔮 Future Scope

    - Integration with real IoT soil moisture sensors.
    - Automatic irrigation control.
    - More farm-specific datasets.
    - Improved weather-based irrigation forecasting.
    - Mobile application and multilingual support.

    ### 🌱 Project Vision

    To explore how Artificial Intelligence and data-driven
    technologies can support efficient water management
    and sustainable agriculture.
    """)

    st.divider()

    st.success(
        "AgriSmart AI — Smart Decisions • Efficient Irrigation • "
        "Sustainable Agriculture"
    )