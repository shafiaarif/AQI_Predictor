"""
app.py
------
Karachi AQI Predictor dashboard — dark theme version.

Features:
- Pure black dashboard background
- Gradient hero card for current AQI
- Custom-styled forecast cards (24h/48h/72h)
- Interactive Plotly line chart with colored AQI zone bands
- Plotly-based SHAP feature importance chart
- Styled hazard alert banner
- Last-updated timestamp + manual refresh

Run:
    streamlit run app.py
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from predict import (
    check_required_files,
    load_feature_columns,
    get_latest_features,
    load_models,
    prepare_features,
    predict_aqi,
    HORIZONS,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="centered",
)

# ============================================================
# CUSTOM DARK CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html,
    body,
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Main Streamlit container */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background-color: #000000 !important;
    }

    /* Main content */
    [data-testid="stMain"] {
        background-color: #000000 !important;
    }

    /* Header */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
    }

    /* Header decoration */
    [data-testid="stDecoration"] {
        background-color: #000000 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* ========================================================
       TEXT
       ======================================================== */

    p,
    span,
    label,
    div,
    .stMarkdown,
    .stText,
    .stCaption {
        font-family: 'Inter', sans-serif;
    }

    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 0;
        color: #FFFFFF !important;
    }

    .app-subtitle {
        color: #A1A1AA !important;
        font-size: 0.95rem;
        margin-top: 0.15rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-top: 1.8rem;
        margin-bottom: 0.6rem;
    }

    .footer-note {
        text-align: center;
        color: #71717A !important;
        font-size: 0.78rem;
        margin-top: 2rem;
    }

    /* ========================================================
       HERO CARD
       ======================================================== */

    .hero-card {
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: white !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.65);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.10);
    }

    .hero-card * {
        color: white !important;
    }

    .hero-aqi {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.3rem 0;
    }

    .hero-label {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.95;
    }

    .hero-sub {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 0.4rem;
    }

    /* ========================================================
       FORECAST CARDS
       ======================================================== */

    .forecast-card {
        border-radius: 16px;
        padding: 1.1rem 0.8rem;
        text-align: center;
        background: #111111 !important;
        border: 1px solid #27272A !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.45);
    }

    .forecast-horizon {
        font-size: 0.8rem;
        color: #A1A1AA !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .forecast-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin: 0.2rem 0;
    }

    .forecast-delta {
        font-size: 0.85rem;
        font-weight: 600;
    }

    .forecast-cat {
        font-size: 0.75rem;
        color: #A1A1AA !important;
        margin-top: 0.2rem;
    }

    /* ========================================================
       ALERT BANNER
       ======================================================== */

    .alert-banner {
        border-radius: 14px;
        padding: 0.9rem 1.2rem;
        font-weight: 600;
        font-size: 0.92rem;
        margin: 1.2rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #111111 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div {
        color: #E4E4E7 !important;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #27272A !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #1F1F1F !important;
        border-color: #00D4FF !important;
        color: #00D4FF !important;
    }

    /* ========================================================
       CAPTION
       ======================================================== */

    [data-testid="stCaptionContainer"] {
        color: #71717A !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #71717A !important;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #111111 !important;
        border: 1px solid #27272A !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #A1A1AA !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    /* ========================================================
       SPINNER
       ======================================================== */

    [data-testid="stSpinner"] p {
        color: #E4E4E7 !important;
    }

    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #000000;
    }

    ::-webkit-scrollbar-thumb {
        background: #27272A;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3F3F46;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def aqi_style(aqi: float):
    """Return (label, solid_color, gradient_css, emoji) for a given AQI value."""

    if aqi <= 50:
        return (
            "Good",
            "#16a34a",
            "linear-gradient(135deg,#22c55e,#15803d)",
            "🟢",
        )

    elif aqi <= 100:
        return (
            "Moderate",
            "#ca8a04",
            "linear-gradient(135deg,#facc15,#b45309)",
            "🟡",
        )

    elif aqi <= 150:
        return (
            "Unhealthy (Sensitive)",
            "#ea580c",
            "linear-gradient(135deg,#fb923c,#c2410c)",
            "🟠",
        )

    elif aqi <= 200:
        return (
            "Unhealthy",
            "#dc2626",
            "linear-gradient(135deg,#f87171,#991b1b)",
            "🔴",
        )

    elif aqi <= 300:
        return (
            "Very Unhealthy",
            "#9333ea",
            "linear-gradient(135deg,#c084fc,#6b21a8)",
            "🟣",
        )

    else:
        return (
            "Hazardous",
            "#7f1d1d",
            "linear-gradient(135deg,#7f1d1d,#450a0a)",
            "⚫",
        )


# ============================================================
# LOAD + CACHE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_everything():

    check_required_files()

    feature_columns = load_feature_columns()
    models = load_models()

    return feature_columns, models


@st.cache_data(ttl=1800, show_spinner=False)
def run_forecast():

    feature_columns, models = load_everything()

    row_df, current_aqi = get_latest_features(feature_columns)

    forecast = predict_aqi(
        row_df,
        current_aqi,
        models,
        feature_columns,
    )

    return (
        forecast,
        row_df,
        feature_columns,
        datetime.now().strftime("%d %b %Y, %I:%M %p"),
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🌫️ Karachi AQI Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    '3-day forecast · FS70 Ensemble (CatBoost + Neural Network) · live features'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# RUN FORECAST
# ============================================================

with st.spinner("Building live features and running predictions..."):

    forecast, row_df, feature_columns, last_updated = run_forecast()

    _, models = load_everything()


# ============================================================
# HERO CARD
# ============================================================

current_aqi = forecast["current_aqi"]

cat, solid_color, gradient, emoji = aqi_style(current_aqi)

st.markdown(
    f'<div class="hero-card" style="background:{gradient};">'
    f'<div class="hero-label">{emoji} Current AQI — Karachi</div>'
    f'<div class="hero-aqi">{current_aqi:.0f}</div>'
    f'<div class="hero-label">{cat}</div>'
    f'<div class="hero-sub">Last updated: {last_updated}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FORECAST CARDS
# ============================================================

st.markdown(
    '<div class="section-title">📈 3-Day Forecast</div>',
    unsafe_allow_html=True,
)

cols = st.columns(3)

for i, h in enumerate(HORIZONS):

    result = forecast[h]

    cat_h, color_h, _, emoji_h = aqi_style(
        result["predicted_aqi"]
    )

    change = result["predicted_change"]

    arrow = (
        "▲"
        if change > 0
        else ("▼" if change < 0 else "→")
    )

    delta_color = (
        "#ef4444"
        if change > 0
        else (
            "#22c55e"
            if change < 0
            else "#A1A1AA"
        )
    )

    with cols[i]:

        st.markdown(
            f'<div class="forecast-card">'
            f'<div class="forecast-horizon">{h}</div>'
            f'<div class="forecast-value">{result["predicted_aqi"]:.0f}</div>'
            f'<div class="forecast-delta" style="color:{delta_color};">{arrow} {change:+.1f}</div>'
            f'<div class="forecast-cat">{emoji_h} {cat_h}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# FORECAST TREND
# ============================================================

st.markdown(
    '<div class="section-title">Forecast Trend</div>',
    unsafe_allow_html=True,
)

x_labels = ["Now"] + HORIZONS

y_values = [
    current_aqi
] + [
    forecast[h]["predicted_aqi"]
    for h in HORIZONS
]

fig = go.Figure()


# ------------------------------------------------------------
# AQI ZONE BANDS
# ------------------------------------------------------------

zone_bands = [
    (0, 50, "rgba(34,197,94,0.08)"),
    (50, 100, "rgba(250,204,21,0.08)"),
    (100, 150, "rgba(251,146,60,0.08)"),
    (150, 200, "rgba(248,113,113,0.08)"),
]

y_max = max(y_values) + 20

for lo, hi, color in zone_bands:

    if lo < y_max:

        fig.add_hrect(
            y0=lo,
            y1=min(hi, y_max),
            fillcolor=color,
            line_width=0,
        )


# ------------------------------------------------------------
# FORECAST LINE
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=x_labels,
        y=y_values,
        mode="lines+markers",

        line=dict(
            color="#00D4FF",
            width=3,
            shape="spline",
        ),

        marker=dict(
            size=10,
            color="#00D4FF",
            line=dict(
                width=2,
                color="#FFFFFF",
            ),
        ),

        fill="tozeroy",

        fillcolor="rgba(0,212,255,0.06)",

        hovertemplate=(
            "<b>%{x}</b>"
            "<br>AQI: %{y:.1f}"
            "<extra></extra>"
        ),
    )
)


# ------------------------------------------------------------
# DARK PLOTLY LAYOUT
# ------------------------------------------------------------

fig.update_layout(

    height=340,

    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10,
    ),

    plot_bgcolor="#000000",

    paper_bgcolor="#000000",

    yaxis=dict(
        title="AQI (US)",
        range=[0, y_max],

        gridcolor="#27272A",

        zerolinecolor="#3F3F46",

        tickfont=dict(
            color="#A1A1AA",
        ),

        title_font=dict(
            color="#D4D4D8",
        ),
    ),

    xaxis=dict(

        title=None,

        gridcolor="#18181B",

        zerolinecolor="#27272A",

        tickfont=dict(
            color="#A1A1AA",
        ),
    ),

    font=dict(
        family="Inter, sans-serif",
        size=13,
        color="#E4E4E7",
    ),

    hoverlabel=dict(
        bgcolor="#111111",
        bordercolor="#3F3F46",
        font_color="#FFFFFF",
    ),
)

st.plotly_chart(
    fig,
    width='stretch',
    config={"displayModeBar": False},
)


# ============================================================
# MODEL BREAKDOWN
# ============================================================

with st.expander("Model breakdown (CatBoost vs Neural Net)"):

    for h in HORIZONS:

        result = forecast[h]

        st.write(
            f"**{h}** — "
            f"CatBoost: `{result['catboost_prediction']:+.2f}` · "
            f"NN: `{result['nn_prediction']:+.2f}` · "
            f"Ensemble change: `{result['predicted_change']:+.2f}`"
        )


# ============================================================
# HAZARD ALERT
# ============================================================

max_forecast = max(y_values)

if max_forecast >= 200:

    st.markdown(
        f'<div class="alert-banner" style="background:#2A0A0A;color:#FCA5A5;border-color:#7F1D1D;">'
        f'⚠️ ALERT: Predicted AQI reaches {max_forecast:.0f} — '
        f'Very Unhealthy/Hazardous levels expected. Avoid outdoor activity.'
        f'</div>',
        unsafe_allow_html=True,
    )

elif max_forecast >= 150:

    st.markdown(
        f'<div class="alert-banner" style="background:#2A1605;color:#FDBA74;border-color:#9A3412;">'
        f'⚠️ Predicted AQI reaches {max_forecast:.0f} — Unhealthy for sensitive groups.'
        f'</div>',
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        '<div class="alert-banner" style="background:#061A0D;color:#86EFAC;border-color:#166534;">'
        '✅ No hazardous AQI levels expected in the next 3 days.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔍 What\'s driving the 24h prediction?'
    '</div>',
    unsafe_allow_html=True,
)

X = prepare_features(
    row_df,
    feature_columns,
)

explainer = shap.TreeExplainer(
    models["catboost"]["24h"]
)

shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame(
    {
        "feature": X.columns,
        "impact": np.abs(shap_values[0]),
    }
).sort_values(
    "impact",
    ascending=True,
).tail(10)


# ------------------------------------------------------------
# SHAP FIGURE
# ------------------------------------------------------------

fig_shap = go.Figure(
    go.Bar(

        x=shap_df["impact"],

        y=shap_df["feature"],

        orientation="h",

        marker=dict(

            color=shap_df["impact"],

            colorscale=[
                [0, "#164E63"],
                [0.5, "#0891B2"],
                [1, "#22D3EE"],
            ],

            line=dict(
                width=0,
            ),
        ),

        hovertemplate=(
            "<b>%{y}</b>"
            "<br>|SHAP|: %{x:.3f}"
            "<extra></extra>"
        ),
    )
)


fig_shap.update_layout(

    height=380,

    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10,
    ),

    plot_bgcolor="#000000",

    paper_bgcolor="#000000",

    xaxis=dict(

        title="mean(|SHAP value|)",

        gridcolor="#27272A",

        zerolinecolor="#3F3F46",

        tickfont=dict(
            color="#A1A1AA",
        ),

        title_font=dict(
            color="#D4D4D8",
        ),
    ),

    yaxis=dict(

        title=None,

        tickfont=dict(
            color="#D4D4D8",
        ),
    ),

    font=dict(
        family="Inter, sans-serif",
        size=12,
        color="#E4E4E7",
    ),

    hoverlabel=dict(
        bgcolor="#111111",
        bordercolor="#3F3F46",
        font_color="#FFFFFF",
    ),
)

st.plotly_chart(
    fig_shap,
    width='stretch',
    config={"displayModeBar": False},
)


st.caption(
    "Top 10 features influencing the 24h AQI change "
    "prediction (CatBoost SHAP values)."
)


# ============================================================
# REFRESH + FOOTER
# ============================================================

col_a, col_b = st.columns([1, 3])

with col_a:

    if st.button("🔄 Refresh"):

        st.cache_data.clear()

        st.rerun()


st.markdown(
    '<div class="footer-note">Sh_AQI_Predictor · Hopsworks + CatBoost + TensorFlow · Karachi</div>',
    unsafe_allow_html=True,
)