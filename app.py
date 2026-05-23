

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import os

if not os.path.exists("wbt_model.pkl"):
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1haVFCdwA3HnnipI3C6HH0na4UzUG0kcO",
        "wbt_model.pkl"
    )

if not os.path.exists("features.pkl"):
    urllib.request.urlretrieve(
        "https://drive.google.com/uc?export=download&id=1bGyqqoSfhsIU17nnJ21AqJozxNTtkmXA",
        "features.pkl"
    )

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Climate Intelligence Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to bottom right,#020617,#0f172a);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

h1,h2,h3,h4 {
    color: #38bdf8;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 20px rgba(0,0,0,0.25);
}

.metric-card h1 {
    color: white;
    font-size: 42px;
}

.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#2563eb);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

.info-box {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
models = joblib.load("wbt_model.pkl")
FEATURES = joblib.load("features.pkl")

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def engineer_features(df):

    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    df["temp_diff"] = df["T2M"] - df["TSOIL1"]
    df["ts_t2m"] = df["TS"] - df["T2M"]

    df["rad_ratio"] = (
        df["ALLSKY_SFC_SW_DWN"] /
        (df["CLRSKY_SFC_SW_DWN"] + 1e-6)
    )

    df["rad_net"] = (
        df["ALLSKY_SFC_SW_DWN"] -
        df["CLRSKY_SFC_SW_DWN"]
    )

    df["evap_wet"] = df["EVLAND"] * df["GWETTOP"]
    df["evap_root"] = df["EVLAND"] * df["GWETROOT"]

    df["wind_stress"] = (
        df["WS10M"] * df["PRECTOTCORR"]
    )

    df["wdir_sin"] = np.sin(np.radians(df["WD10M"]))
    df["wdir_cos"] = np.cos(np.radians(df["WD10M"]))

    df["ps_anom"] = df["PS"] - df["PS"].mean()

    return df


# =========================================================
# HEADER
# =========================================================
st.markdown("""
# 🌍 Climate Intelligence Dashboard

### Explainable Intelligence System
Predict Wet Bulb Temperature using Machine Learning + Explainable AI
""")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Climate Controls")

theme = st.sidebar.selectbox(
    "Prediction Mode",
    ["Standard", "Advanced", "Research"]
)

st.sidebar.markdown("---")

st.sidebar.info("""
This AI model predicts Wet Bulb Temperature using:
- Temperature
- Solar Radiation
- Humidity
- Wind
- Pressure
- Soil Wetness
""")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Prediction",
    "📊 Analytics",
    "🧠 AI Explanation",
    "ℹ️ About Model"
])

# =========================================================
# TAB 1 : PREDICTION
# =========================================================
with tab1:

    st.markdown("## 📥 Enter Environmental Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        rel_lat = st.number_input(
            "🌍 Relative Latitude",
            value=20.0
        )

        rel_lon = st.number_input(
            "🌍 Relative Longitude",
            value=80.0
        )

        T2M = st.slider(
            "🌡️ Air Temperature",
            0.0, 60.0, 30.0
        )

        TSOIL1 = st.slider(
            "🪨 Soil Temperature",
            0.0, 60.0, 28.0
        )

        TS = st.slider(
            "🔥 Surface Temperature",
            0.0, 70.0, 31.0
        )

    with col2:

        ALLSKY_SFC_SW_DWN = st.slider(
            "☀️ Solar Radiation",
            0.0, 500.0, 200.0
        )

        CLRSKY_SFC_SW_DWN = st.slider(
            "🌤️ Clear Sky Radiation",
            0.0, 500.0, 250.0
        )

        CLOUD_AMT = st.slider(
            "☁️ Cloud Amount",
            0.0, 1.0, 0.5
        )

        GWETTOP = st.slider(
            "💧 Surface Wetness",
            0.0, 1.0, 0.3
        )

        GWETROOT = st.slider(
            "🌱 Root Wetness",
            0.0, 1.0, 0.4
        )

    with col3:

        WS10M = st.slider(
            "💨 Wind Speed",
            0.0, 50.0, 5.0
        )

        PS = st.number_input(
            "📈 Pressure",
            value=1013.0
        )

        PRECTOTCORR = st.slider(
            "🌧️ Rainfall",
            0.0, 100.0, 0.0
        )

        EVLAND = st.slider(
            "🌿 Evaporation",
            0.0, 10.0, 0.2
        )

        WD10M = st.slider(
            "🧭 Wind Direction",
            0.0, 360.0, 180.0
        )

    selected_date = st.date_input(
        "📅 Select Date",
        date.today()
    )

    # =====================================================
    # PREDICTION
    # =====================================================
    if st.button("🚀 Predict WBT"):

        month = selected_date.month
        dayofyear = selected_date.timetuple().tm_yday
        year = selected_date.year

        input_df = pd.DataFrame({

            "rel_lat": [rel_lat],
            "rel_lon": [rel_lon],

            "T2M": [T2M],
            "TSOIL1": [TSOIL1],
            "TS": [TS],

            "ALLSKY_SFC_SW_DWN": [ALLSKY_SFC_SW_DWN],
            "CLRSKY_SFC_SW_DWN": [CLRSKY_SFC_SW_DWN],

            "CLOUD_AMT": [CLOUD_AMT],
            "GWETTOP": [GWETTOP],
            "GWETROOT": [GWETROOT],

            "WS10M": [WS10M],
            "PS": [PS],

            "PRECTOTCORR": [PRECTOTCORR],
            "EVLAND": [EVLAND],

            "WD10M": [WD10M],

            "month": [month],
            "dayofyear": [dayofyear],
            "year": [year]
        })

        input_df = engineer_features(input_df)
        input_df = input_df[FEATURES]

        prediction = np.mean([
            model.predict(input_df)[0]
            for model in models
        ])

        # =================================================
        # METRICS
        # =================================================
        st.markdown("---")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>🌡️ WBT</h3>
            <h1>{prediction:.2f}°C</h1>
            </div>
            """, unsafe_allow_html=True)

        with m2:

            humidity = CLOUD_AMT * 100

            st.markdown(f"""
            <div class="metric-card">
            <h3>💧 Humidity</h3>
            <h1>{humidity:.0f}%</h1>
            </div>
            """, unsafe_allow_html=True)

        with m3:

            risk = "LOW"

            if prediction >= 35:
                risk = "EXTREME"

            elif prediction >= 28:
                risk = "MODERATE"

            st.markdown(f"""
            <div class="metric-card">
            <h3>⚠️ Risk Level</h3>
            <h1>{risk}</h1>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # ALERTS
        # =================================================
        st.markdown("## 🚨 Heat Risk Analysis")

        if prediction >= 35:
            st.error("""
            Extreme Heat Stress Detected.
            High humidity + temperature combination
            may be dangerous for humans.
            """)

        elif prediction >= 28:
            st.warning("""
            Moderate Heat Stress.
            Outdoor activities should be reduced.
            """)

        else:
            st.success("""
            Environmental conditions are relatively safe.
            """)

        # =================================================
        # STORE FOR OTHER TABS
        # =================================================
        st.session_state["prediction"] = prediction
        st.session_state["features"] = {
            "Temperature": T2M,
            "Surface Temp": TS,
            "Soil Temp": TSOIL1,
            "Radiation": ALLSKY_SFC_SW_DWN,
            "Humidity": humidity,
            "Wind": WS10M
        }

# =========================================================
# TAB 2 : ANALYTICS
# =========================================================
with tab2:

    st.markdown("## 📊 Climate Analytics")

    if "features" in st.session_state:

        feature_dict = st.session_state["features"]

        chart_df = pd.DataFrame({
            "Feature": list(feature_dict.keys()),
            "Value": list(feature_dict.values())
        })

        fig = px.bar(
            chart_df,
            x="Feature",
            y="Value",
            title="Environmental Feature Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

        radar = go.Figure()

        radar.add_trace(go.Scatterpolar(
            r=list(feature_dict.values()),
            theta=list(feature_dict.keys()),
            fill='toself',
            name='Climate'
        ))

        radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=False
        )

        st.plotly_chart(radar, use_container_width=True)

    else:
        st.info("Run a prediction first.")

# =========================================================
# TAB 3 : AI EXPLANATION
# =========================================================
with tab3:

    st.markdown("## 🧠 Why Did AI Predict This?")

    if "prediction" in st.session_state:

        prediction = st.session_state["prediction"]

        explanation = []

        if T2M > 35:
            explanation.append(
                "High air temperature strongly increased WBT."
            )

        if CLOUD_AMT > 0.7:
            explanation.append(
                "High cloud amount increased humidity retention."
            )

        if WS10M < 3:
            explanation.append(
                "Low wind speed reduced cooling effect."
            )

        if GWETTOP > 0.5:
            explanation.append(
                "Surface wetness increased atmospheric moisture."
            )

        if ALLSKY_SFC_SW_DWN > 300:
            explanation.append(
                "High solar radiation raised surface heating."
            )

        if len(explanation) == 0:
            explanation.append(
                "Balanced environmental conditions produced stable WBT."
            )

        for i, exp in enumerate(explanation, start=1):

            st.markdown(f"""
            <div class="info-box">
            <h4>🔍 Reason {i}</h4>
            <p>{exp}</p>
            </div>
            """, unsafe_allow_html=True)

      

       

    else:
        st.info("Make prediction first.")

# =========================================================
# TAB 4 : ABOUT MODEL
# =========================================================
with tab4:

    st.markdown("## ℹ️ About This Model")

    st.markdown("""
    ### 🤖 Machine Learning Architecture

    This project uses:
    - LightGBM Regressor
    - Ensemble Learning
    - Feature Engineering
    - Climate Data Analytics

    ### 📚 Features Used

    The AI considers:
    - Air Temperature
    - Soil Temperature
    - Surface Temperature
    - Radiation
    - Humidity
    - Wind
    - Pressure
    - Rainfall
    - Seasonal Cycles

    ### 🌍 Applications

    - Climate Monitoring
    - Heat Stress Detection
    - Environmental Forecasting
    - Smart Agriculture
    - Weather Intelligence

    ### 🚀 Developed Using

    - Streamlit
    - Plotly
    - LightGBM
    - Python
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption("""
Built by team GENESIS using Streamlit + Explainable AI + Climate Intelligence
""")