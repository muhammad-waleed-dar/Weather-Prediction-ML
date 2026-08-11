"""
Phase 4/5: Streamlit interface for the Rain Tomorrow predictor.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import datetime

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from predict import RainPredictor
from utils import DECISION_THRESHOLD, month_to_season


@st.cache_resource
def load_predictor():
    return RainPredictor()


st.set_page_config(page_title="Rain Tomorrow Predictor", page_icon="🌦️", layout="centered")

# ------------------------------------------------------------------------------
# DESIGN TOKENS — a weather-station console: overcast neutrals, a storm-blue /
# sunbreak-amber pair as the two "states" the model predicts between, and a
# monospace face for the data readouts (temperature, pressure, probability)
# to read like real instrument output rather than generic app text.
# ------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --overcast: #EAEFF4;
    --panel: #FFFFFF;
    --ink: #24303D;
    --storm: #2E5C8A;
    --sunbreak: #E3A438;
    --mist: #B8C4D0;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, var(--overcast) 0%, #DCE4EC 100%);
}
[data-testid="stHeader"] { background: transparent; }

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }

.station-banner {
    background: linear-gradient(120deg, var(--storm) 0%, #244a70 100%);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    color: #F4F7FA;
    box-shadow: 0 8px 24px rgba(46, 92, 138, 0.25);
}
.station-banner h1 {
    font-size: 1.9rem; margin: 0 0 4px 0; color: #FFFFFF !important;
    letter-spacing: -0.5px;
}
.station-banner p { margin: 0; color: #C7D6E5; font-size: 0.95rem; }
.station-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
    letter-spacing: 1.5px; text-transform: uppercase; color: var(--sunbreak);
    margin-bottom: 6px; display: block;
}

[data-testid="stForm"] {
    background: var(--panel); border-radius: 14px; padding: 8px 4px;
    border: 1px solid var(--mist);
}

.section-label {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem;
    color: var(--storm); margin: 6px 0 2px 0; padding-bottom: 6px;
    border-bottom: 2px solid var(--overcast);
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(120deg, var(--storm), #244a70);
    color: white; border: none; border-radius: 10px; padding: 0.7rem 1rem;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem;
    width: 100%; box-shadow: 0 4px 14px rgba(46, 92, 138, 0.35);
}
[data-testid="stFormSubmitButton"] button:hover { background: #244a70; }

.result-card {
    border-radius: 16px; padding: 28px; margin-top: 20px;
    display: flex; align-items: center; gap: 28px; flex-wrap: wrap;
    justify-content: center;
}
.result-rain { background: linear-gradient(120deg, #2E5C8A 0%, #1c3a57 100%); color: #F4F7FA; }
.result-clear { background: linear-gradient(120deg, #E3A438 0%, #b97f26 100%); color: #2A1F0A; }

.result-text h2 { color: inherit !important; margin: 0 0 6px 0; font-size: 1.5rem; }
.result-text p { margin: 0; opacity: 0.9; font-size: 0.92rem; max-width: 320px; }

.readout {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
    margin-top: 10px; opacity: 0.85;
}

.gauge-outer { width: 200px; height: 100px; overflow: hidden; position: relative; flex-shrink: 0; }
.gauge-bg {
    width: 200px; height: 200px; border-radius: 50%;
    background: conic-gradient(from 180deg,
        var(--sunbreak) 0deg, #d9c26a 90deg, var(--storm) 180deg,
        transparent 180deg 360deg);
}
.gauge-needle {
    position: absolute; bottom: 0; left: 50%; width: 3px; height: 92px;
    background: var(--ink); transform-origin: bottom center;
    border-radius: 2px; box-shadow: 0 0 0 4px rgba(255,255,255,0.15);
}
.gauge-center {
    position: absolute; bottom: -8px; left: 50%; transform: translateX(-50%);
    width: 16px; height: 16px; border-radius: 50%; background: var(--ink);
}
.gauge-pct {
    position: absolute; bottom: 4px; left: 50%; transform: translateX(-50%);
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.4rem;
    color: inherit;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="station-banner">
        <span class="station-eyebrow">Weather Prediction ML — Zee Outsourcing Internship</span>
        <h1>🌦️ Rain Tomorrow Station</h1>
        <p>Enter today's readings below — the model estimates tomorrow's rain probability
        from a Random Forest trained on 10 years of Australian weather data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "Model files not found in `models/`. Run `python src/train_model.py` "
        "first (with `weatherAUS.csv` in `data/`) to generate them."
    )
    st.stop()

with st.form("weather_form"):
    st.markdown('<div class="section-label">📍 Location &amp; Date</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        location = st.selectbox("Location", predictor.categorical_options("Location"))
    with col2:
        obs_date = st.date_input("Observation date", datetime.date.today())

    st.markdown('<div class="section-label">🌡️ Temperature (°C)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        min_temp = st.number_input("MinTemp", value=15.0, step=0.5)
    with col2:
        max_temp = st.number_input("MaxTemp", value=25.0, step=0.5)
    with col3:
        temp_9am = st.number_input("Temp9am", value=18.0, step=0.5)
    temp_3pm = st.number_input("Temp3pm", value=23.0, step=0.5)

    st.markdown('<div class="section-label">🌧️ Rain, Evaporation &amp; Sunshine</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0, step=0.5)
    with col2:
        evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0, step=0.5)
    with col3:
        sunshine = st.number_input("Sunshine (hrs)", min_value=0.0, max_value=14.0, value=7.0, step=0.5)
    rain_today = st.selectbox("Rain today?", predictor.categorical_options("RainToday"))

    st.markdown('<div class="section-label">💨 Wind</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        wind_gust_dir = st.selectbox("Wind gust direction", predictor.categorical_options("WindGustDir"))
        wind_dir_9am = st.selectbox("Wind direction 9am", predictor.categorical_options("WindDir9am"))
    with col2:
        wind_gust_speed = st.number_input("Wind gust speed (km/h)", min_value=0.0, value=35.0, step=1.0)
        wind_dir_3pm = st.selectbox("Wind direction 3pm", predictor.categorical_options("WindDir3pm"))
    col1, col2 = st.columns(2)
    with col1:
        wind_speed_9am = st.number_input("Wind speed 9am (km/h)", min_value=0.0, value=12.0, step=1.0)
    with col2:
        wind_speed_3pm = st.number_input("Wind speed 3pm (km/h)", min_value=0.0, value=18.0, step=1.0)

    st.markdown('<div class="section-label">💧 Humidity, Pressure &amp; Cloud</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        humidity_9am = st.slider("Humidity 9am (%)", 0, 100, 70)
        pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0, step=0.5)
        cloud_9am = st.slider("Cloud cover 9am (oktas)", 0, 9, 4)
    with col2:
        humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1013.0, step=0.5)
        cloud_3pm = st.slider("Cloud cover 3pm (oktas)", 0, 9, 4)

    submitted = st.form_submit_button("Predict Rain Tomorrow")

if submitted:
    month_name = obs_date.strftime("%B")
    season = month_to_season(obs_date.month)

    raw_input = {
        "Location": location, "MinTemp": min_temp, "MaxTemp": max_temp,
        "Rainfall": rainfall, "Evaporation": evaporation, "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir, "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am, "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am, "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am, "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am, "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am, "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am, "Temp3pm": temp_3pm, "RainToday": rain_today,
        "Month": month_name, "Season": season,
        "TempRange": max_temp - min_temp,
        "HumidityChange": humidity_3pm - humidity_9am,
        "PressureChange": pressure_3pm - pressure_9am,
    }

    try:
        label, prob_rain = predictor.predict(raw_input)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    pct = prob_rain * 100
    needle_deg = -90 + (prob_rain * 180)  # -90deg = 0% (sunbreak side), +90deg = 100% (storm side)
    card_class = "result-rain" if label == "Rain" else "result-clear"
    headline = "Rain expected tomorrow" if label == "Rain" else "No rain expected tomorrow"
    icon = "🌧️" if label == "Rain" else "☀️"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="gauge-outer">
                <div class="gauge-bg"></div>
                <div class="gauge-needle" style="transform: rotate({needle_deg}deg);"></div>
                <div class="gauge-center"></div>
                <div class="gauge-pct">{pct:.0f}%</div>
            </div>
            <div class="result-text">
                <h2>{icon} {headline}</h2>
                <p>Predicted probability of rain: {pct:.1f}%. Decision threshold set at
                {DECISION_THRESHOLD} (tuned in Phase 3 from the default 0.5 to catch more
                actual rain days).</p>
                <div class="readout">LOC:{location} · {month_name.upper()} · {season.upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
