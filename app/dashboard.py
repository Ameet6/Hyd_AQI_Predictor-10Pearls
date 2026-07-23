"""
Pearls AQI Predictor — Dashboard
------------------------------------
Streamlit dashboard: live air quality + weather conditions, 3-day forecast,
pollutant breakdown, and model performance — all pulled from MongoDB.

Run with:
    streamlit run app/dashboard.py
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, db
from src.predict import get_latest_features, predict_all_horizons, aqi_category, load_active_model
from src.explain import explain_prediction
from src.alerts import check_alert

st.set_page_config(page_title="Hyderabad Air Quality", page_icon="🌤️", layout="wide")

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp { background: #f4f7fb; color: #1a2332; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #1a2332 !important; }
    p, [data-testid="stCaptionContainer"] { color: #5b6472 !important; }

    .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.6rem; }
    .topbar-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #1a2332; }
    .topbar-sub { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #8894a3; margin-top: 0.1rem; }
    .live-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #16a34a; margin-right: 6px;
        box-shadow: 0 0 0 4px rgba(22,163,74,0.15);
    }
    .live-badge {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: #16a34a; text-align: right;
    }
    .updated-time { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #8894a3; text-align: right; }

    .section-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: #8894a3; margin: 1.8rem 0 0.6rem 0;
    }
    div[role="radiogroup"] {
        gap: 0.5rem;
    }
    div[role="radiogroup"] label {
        background: #ffffff;
        border: 1px solid #e3e8ef;
        border-radius: 8px;
        padding: 0.4rem 1rem !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    .panel {
        background: #ffffff; border-radius: 14px; padding: 1.5rem 1.6rem;
        box-shadow: 0 4px 16px rgba(26,35,50,0.06); height: 100%;
    }
    .panel-title {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: #8894a3; margin-bottom: 0.8rem;
    }

    .aqi-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.85rem;
    }
    .aqi-big { font-family: 'JetBrains Mono', monospace; font-size: 3.6rem; font-weight: 700; line-height: 1; color: #1a2332; margin: 0.5rem 0 0.6rem 0; }
    .pm-note { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #8894a3; margin-top: 0.9rem; }

    .weather-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 1.5rem; margin-top: 0.4rem; }
    .weather-cell .w-val { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #1a2332; }
    .weather-cell .w-name { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.07em; text-transform: uppercase; color: #8894a3; }

    .pbar-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.55rem; }
    .pbar-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #5b6472; width: 52px; flex-shrink: 0; }
    .pbar-track { flex: 1; background: #eef1f6; border-radius: 999px; height: 8px; overflow: hidden; }
    .pbar-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
    .pbar-val { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #1a2332; width: 54px; text-align: right; flex-shrink: 0; }

    .forecast-card {
        background: #ffffff; border-radius: 14px; padding: 1.3rem 1.3rem 1.1rem 1.3rem;
        height: 100%; box-shadow: 0 4px 16px rgba(26,35,50,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .alert-banner {
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .alert-banner .alert-icon { font-size: 1.4rem; flex-shrink: 0; }
    .alert-banner .alert-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .alert-banner .alert-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 700;
        display: block;
        margin-bottom: 0.15rem;
    }
    .alert-caution { background: #fef3c7; border: 1px solid #fbbf24; }
    .alert-caution .alert-tag, .alert-caution .alert-text { color: #92400e; }
    .alert-warning { background: #fee2e2; border: 1px solid #ef4444; }
    .alert-warning .alert-tag, .alert-warning .alert-text { color: #991b1b; }
    .forecast-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(26,35,50,0.12);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #ffffff !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(26,35,50,0.06) !important;
        padding: 1.5rem 1.6rem !important;
    }
    }
    .forecast-card .day-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: #1a2332; }
    .forecast-card .day-date { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #8894a3; margin-bottom: 0.8rem; }
    .forecast-card .aqi-num { font-family: 'JetBrains Mono', monospace; font-size: 2.3rem; font-weight: 700; color: #1a2332; line-height: 1; display: inline-block; }
    .forecast-card .trend { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-left: 0.4rem; }
    .forecast-card .model-note { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #b0b8c4; margin-top: 0.7rem; }

    .legend-strip { display: flex; border-radius: 10px; overflow: hidden; height: 7px; }
    .legend-strip div { flex: 1; }
    .legend-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #8894a3; margin-top: 0.3rem; }

    .perf-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
    .perf-table th { text-align: left; color: #8894a3; font-weight: 500; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; padding: 0.5rem 0.8rem; border-bottom: 1px solid #e3e8ef; }
    .perf-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid #eef1f6; color: #1a2332; }
    .perf-table tr:nth-child(even) td { background: #fafbfd; }
    .algo-pill { background: #eef4ff; color: #2563eb; padding: 0.15rem 0.55rem; border-radius: 6px; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_client():
    return db.get_client()


def load_dashboard_data():
    client = get_db_client()
    features_collection = db.get_collection(config.FEATURES_COLLECTION, client)
    models_collection = db.get_collection(config.MODELS_COLLECTION, client)

    latest = get_latest_features(features_collection)
    forecasts = predict_all_horizons(latest, models_collection)

    history_cursor = features_collection.find(
        {"city": config.CITY_NAME}
    ).sort("timestamp", -1).limit(24 * 14)
    history_df = pd.DataFrame(list(history_cursor)).sort_values("timestamp")

    return latest, forecasts, history_df


latest, forecasts, history_df = load_dashboard_data()
current_label, current_bg, current_text = aqi_category(latest["aqi"])

alert = check_alert(latest["aqi"], forecasts)
if alert is not None:
    icon = "🔴" if alert["tier"] == "warning" else "🟠"
    tag = "Health Warning" if alert["tier"] == "warning" else "Health Caution"
    css_class = "alert-warning" if alert["tier"] == "warning" else "alert-caution"
    st.markdown(f"""
    <div class="alert-banner {css_class}">
        <div class="alert-icon">{icon}</div>
        <div class="alert-text"><span class="alert-tag">{tag}</span>{alert['message']}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown('<div class="topbar-title">Hyderabad, Pakistan — Air Quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="topbar-sub">Live monitoring &amp; 3-day forecast</div>', unsafe_allow_html=True)
with col_b:
    st.markdown('<div class="live-badge"><span class="live-dot"></span>LIVE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="updated-time">{latest["timestamp"].strftime("%a %b %d · %H:%M UTC")}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# AIR QUALITY + WEATHER — split panels
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

pollutants = [
    ("PM2.5", latest["pm2_5"], 75), ("PM10", latest["pm10"], 150),
    ("CO", latest["co"], 4000), ("NO2", latest["no2"], 100),
    ("SO2", latest["so2"], 40), ("O3", latest["o3"], 100),
]

with col1:
    pbar_html = ""
    for name, val, scale_max in pollutants:
        pct = min(100, (val / scale_max) * 100) if val is not None else 0
        bar_color = "#dc2626" if pct > 70 else ("#eab308" if pct > 40 else "#16a34a")
        pbar_html += (
            f'<div class="pbar-row">'
            f'<div class="pbar-label">{name}</div>'
            f'<div class="pbar-track"><div class="pbar-fill" style="width:{pct}%; background:{bar_color};"></div></div>'
            f'<div class="pbar-val">{val}</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Air Quality</div>
        <span class="aqi-badge" style="background:{current_bg}; color:{current_text};">{current_label}</span>
        <div class="aqi-big">{latest['aqi']}</div>
        <div class="legend-strip">
            <div style="background:#16a34a;"></div><div style="background:#eab308;"></div>
            <div style="background:#f97316;"></div><div style="background:#dc2626;"></div>
            <div style="background:#9333ea;"></div><div style="background:#7f1d1d;"></div>
        </div>
        <div class="legend-labels"><span>Good</span><span>Moderate</span><span>USG</span><span>Unhealthy</span><span>V.Unhealthy</span><span>Hazardous</span></div>
        <div class="pm-note">POLLUTANT LEVELS (µg/m³ · ppb)</div>
        {pbar_html}
    </div>
    """, unsafe_allow_html=True)

with col2:
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_label = weekday_names[latest["day_of_week"]]
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Weather</div>
        <div class="weather-grid">
            <div class="weather-cell"><div class="w-val">{latest['temperature']}°C</div><div class="w-name">🌡️ Temperature</div></div>
            <div class="weather-cell"><div class="w-val">{latest['humidity']}%</div><div class="w-name">💧 Humidity</div></div>
            <div class="weather-cell"><div class="w-val">{latest['pressure']} hPa</div><div class="w-name">📊 Pressure</div></div>
            <div class="weather-cell"><div class="w-val">{latest['wind_speed']} m/s</div><div class="w-name">💨 Wind Speed</div></div>
            <div class="weather-cell"><div class="w-val">{latest['hour']:02d}:00</div><div class="w-name">🕒 Hour (UTC)</div></div>
            <div class="weather-cell"><div class="w-val">{weekday_label}</div><div class="w-name">📅 Day</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3-DAY FORECAST
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Forecast</div>', unsafe_allow_html=True)
st.markdown("### Next 3 Days")

forecast_cols = st.columns(3)
for i, forecast in enumerate(forecasts):
    label, bg, text_color = aqi_category(forecast["predicted_aqi"])
    forecast_date = latest["timestamp"] + timedelta(hours=forecast["horizon_hours"])
    day_name = forecast_date.strftime("%A")
    day_date = forecast_date.strftime("%b %d")

    with forecast_cols[i]:
        if forecast["predicted_aqi"] is not None:
            delta = forecast["predicted_aqi"] - latest["aqi"]
            if delta > 2:
                trend = f'<span class="trend" style="color:#dc2626;">▲ {delta:+.0f}</span>'
            elif delta < -2:
                trend = f'<span class="trend" style="color:#16a34a;">▼ {delta:+.0f}</span>'
            else:
                trend = f'<span class="trend" style="color:#8894a3;">— {delta:+.0f}</span>'

            r2 = forecast["metrics"]["r2"] if forecast["metrics"] else None
            model_line = f"{forecast['algorithm'].replace('_', ' ').title()} · R² {r2:.2f}" if r2 is not None else ""
            st.markdown(f"""
            <div class="forecast-card">
                <div class="day-name">{day_name}</div>
                <div class="day-date">{day_date}</div>
                <div><span class="aqi-num">{forecast['predicted_aqi']}</span>{trend}</div>
                <span class="aqi-badge" style="background:{bg}; color:{text_color}; margin-top:0.5rem; display:inline-block;">{label}</span>
                <div class="model-note">{model_line}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="day-name">{day_name}</div>
                <div class="day-date">{day_date}</div>
                <span class="aqi-badge" style="background:#eef1f6; color:#8894a3;">No model yet</span>
            </div>
            """, unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# WHY THIS FORECAST — SHAP explanation
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
st.markdown("### Why This Forecast?")

day_choice = st.radio(
    "Select a day to explain:", ["Day 1", "Day 2", "Day 3"],
    horizontal=True, label_visibility="collapsed",
)
horizon_map = {"Day 1": 24, "Day 2": 48, "Day 3": 72}
selected_horizon = horizon_map[day_choice]

client_for_explain = get_db_client()
models_collection_explain = db.get_collection(config.MODELS_COLLECTION, client_for_explain)
model_info = load_active_model(selected_horizon, models_collection_explain)

if model_info is not None:
    from src.explain import FEATURE_LABELS
    contributions = explain_prediction(model_info["model"], model_info["feature_cols"], latest)
    top_contributions = contributions[:6]

    exp_fig = go.Figure()
    labels = [f"{FEATURE_LABELS.get(c['feature'], c['feature'])}  ·  {c['value']}" for c in top_contributions][::-1]
    impacts = [c["shap_value"] for c in top_contributions][::-1]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in impacts]

    exp_fig.add_trace(go.Bar(
        x=impacts, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:+.2f}" for v in impacts], textposition="outside",
        textfont=dict(color="#1a2332", family="JetBrains Mono", size=12),
    ))
    exp_fig.update_traces(marker_line_width=0, width=0.55)
    exp_fig.update_layout(
        height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#5b6472", size=12),
        xaxis=dict(title="Impact on predicted AQI (points)", gridcolor="#eef1f6", zeroline=True, zerolinecolor="#c7cedb", zerolinewidth=1.5,
                    title_font=dict(family="JetBrains Mono", size=11)),
        yaxis=dict(title=None, tickfont=dict(size=13, family="Space Grotesk")),
        margin=dict(l=130, r=55, t=15, b=45),
        bargap=0.4,
    )
    with st.container(border=True):
        st.plotly_chart(exp_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div class="pm-note">🔴 Red = pushed the forecast higher (worse air quality) &nbsp;·&nbsp; '
            '🟢 Green = pushed the forecast lower (better air quality)</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No model available yet for this horizon.")

# ---------------------------------------------------------------------------
# HISTORICAL TREND
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)
st.markdown("### Last 14 Days")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history_df["timestamp"], y=history_df["aqi"],
    mode="lines", name="AQI", line=dict(color="#2563eb", width=2),
    fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.06)",
))
fig.add_hline(y=100, line_dash="dot", line_color="#eab308", line_width=1,
              annotation_text="Unhealthy for Sensitive Groups", annotation_font_color="#b8860b", annotation_font_size=11)
fig.add_hline(y=150, line_dash="dot", line_color="#dc2626", line_width=1,
              annotation_text="Unhealthy", annotation_font_color="#dc2626", annotation_font_size=11)
fig.update_layout(
    height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#5b6472", size=11),
    xaxis=dict(title=None, gridcolor="#e3e8ef", showline=False),
    yaxis=dict(title="US AQI", gridcolor="#e3e8ef", showline=False),
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)
st.markdown("### Model Performance")

rows_html = ""
for forecast in forecasts:
    if forecast["metrics"] is None:
        continue
    m = forecast["metrics"]
    rows_html += (
        f'<tr>'
        f'<td>Day {forecast["day"]} ({forecast["horizon_hours"]}h)</td>'
        f'<td><span class="algo-pill">{forecast["algorithm"].replace("_", " ").title()}</span></td>'
        f'<td>{m["rmse"]:.2f}</td>'
        f'<td>{m["mae"]:.2f}</td>'
        f'<td>{m["r2"]:.3f}</td>'
        f'</tr>'
    )
st.markdown(f"""
<div class="panel">
    <table class="perf-table">
        <tr><th>Horizon</th><th>Algorithm</th><th>RMSE</th><th>MAE</th><th>R²</th></tr>
        {rows_html}
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-label" style="margin-top:1.4rem;">'
    'Open-Meteo · Feature pipeline hourly · Model retrains daily'
    '</div>',
    unsafe_allow_html=True,
)