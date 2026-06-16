"""
Energy Trading Agent — Premium Streamlit Dashboard
===================================================
A polished control room for the Prophet + LSTM ensemble forecaster and the
LangGraph trading agent.

Run:
    streamlit run dashboard.py
"""

import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Make the local `src` package importable regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from src.data_pipeline import load_data, fetch_and_save_data
from src.forecasting.prophet_model import (
    train_prophet_model,
    load_prophet_model,
    generate_prophet_forecast,
)
from src.forecasting.lstm_model import train_lstm_model
from src.forecasting.ensemble import generate_ensemble_forecast
from src.agent.langgraph_agent import run_trading_agent
from src.agent.report_generator import create_stakeholder_report


# ──────────────────────────────────────────────────────────────────────────
# Page config + theming
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Trading Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette: beige + shades of gold ───────────────────────────────────────
ACCENT = "#b8860b"          # dark goldenrod (primary gold)
GOLD_SOFT = "#d9b44a"       # light gold
INK = "#4a3a14"             # deep bronze — primary text (on beige)
INK_DIM = "#8a7434"         # muted gold-brown — secondary text

BUY_COLOR = "#b8860b"       # rich gold — accumulate
SELL_COLOR = "#6e5410"      # deep bronze — reduce
HOLD_COLOR = "#cdb86a"      # pale gold — hold
PROPHET_COLOR = "#9c7a2e"   # bronze
LSTM_COLOR = "#caa64b"      # gold
HIST_COLOR = "#a8966a"      # soft tan

SIGNAL_META = {
    "BUY":  {"color": BUY_COLOR,  "emoji": "🟢", "blurb": "Upside expected — accumulate."},
    "SELL": {"color": SELL_COLOR, "emoji": "🔴", "blurb": "Downside expected — reduce exposure."},
    "HOLD": {"color": HOLD_COLOR, "emoji": "🟡", "blurb": "Stable market — maintain position."},
    "UNKNOWN": {"color": HIST_COLOR, "emoji": "⚪", "blurb": "Signal unavailable."},
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    .stApp {
        background:
            radial-gradient(1200px 600px at 12% -8%, rgba(184,134,11,0.14), transparent 60%),
            radial-gradient(1000px 520px at 100% 0%, rgba(217,180,74,0.16), transparent 55%),
            linear-gradient(180deg, #f2ead8 0%, #e9dec4 100%);
        color: #4a3a14;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #4a3a14; }

    /* Hero */
    .hero {
        padding: 26px 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(184,134,11,0.20), rgba(217,180,74,0.10));
        border: 1px solid rgba(184,134,11,0.40);
        box-shadow: 0 8px 30px rgba(110,84,16,0.18);
        margin-bottom: 8px;
    }
    .hero h1 {
        font-size: 2.05rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #9c7a2e, #b8860b 55%, #6e5410);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .hero p { color:#8a7434; margin: 6px 0 0 0; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: rgba(255,250,235,0.65);
        border: 1px solid rgba(184,134,11,0.35);
        border-radius: 16px;
        padding: 18px 20px;
        height: 100%;
        backdrop-filter: blur(6px);
        transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(184,134,11,0.85);
        box-shadow: 0 6px 22px rgba(184,134,11,0.18);
    }
    .metric-label { color:#8a7434; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px; font-weight:600; }
    .metric-value { font-size:1.85rem; font-weight:800; margin-top:4px; color:#4a3a14; }
    .metric-delta { font-size:0.85rem; font-weight:600; margin-top:2px; }

    /* Signal banner */
    .signal-card {
        border-radius: 18px; padding: 22px 26px; margin-top: 4px;
        border: 1px solid rgba(184,134,11,0.40);
        display:flex; align-items:center; gap:18px;
    }
    .signal-badge {
        font-size: 1.6rem; font-weight: 800; padding: 8px 22px; border-radius: 999px;
        color:#fdf8ea;
    }

    /* Section titles */
    .section-title { font-size:1.05rem; font-weight:700; color:#9c7a2e; margin: 6px 0 2px 0; }

    /* Report box */
    .report-box {
        background: rgba(255,250,235,0.65);
        border: 1px solid rgba(184,134,11,0.35);
        border-radius: 16px; padding: 4px 22px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ece1c6 0%, #e3d4b0 100%);
        border-right:1px solid rgba(184,134,11,0.35);
    }
    div[data-testid="stMetric"] { background: transparent; }

    /* Gold-tinted controls */
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color:#9c7a2e; }
    .stButton > button {
        background: linear-gradient(135deg, #d9b44a, #b8860b);
        color:#3a2e10; font-weight:700; border:none; border-radius:10px;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #e3c163, #c9970c); color:#3a2e10; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background:#b8860b; }
    a, a:visited { color:#9c7a2e; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────
# Cached data / model layer
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(_cache_key: int = 0) -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner=False)
def ensure_models(_cache_key: int = 0):
    """Train Prophet + LSTM if artifacts are missing. Returns status messages."""
    df = load_data()
    msgs = []
    if load_prophet_model() is None:
        train_prophet_model(df)
        msgs.append("Trained Prophet model")
    if not os.path.exists(os.path.join(DATA_DIR, "lstm_model.pth")):
        train_lstm_model(df, epochs=15)
        msgs.append("Trained LSTM model")
    return msgs or ["Loaded cached models"]


@st.cache_data(show_spinner=False)
def get_forecast(periods: int, _cache_key: int = 0) -> pd.DataFrame:
    df = load_data()
    return generate_ensemble_forecast(df, periods)


@st.cache_data(show_spinner=False)
def get_prophet_band(periods: int, _cache_key: int = 0) -> pd.DataFrame:
    model = load_prophet_model()
    fc = generate_prophet_forecast(model, periods).tail(periods).reset_index(drop=True)
    return fc[["ds", "yhat_lower", "yhat_upper"]]


# ──────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Control Panel")
    st.caption("Prophet + LSTM ensemble · LangGraph agent")
    st.divider()

    horizon = st.slider("Forecast horizon (hours)", 6, 72, 24, step=6)
    history_window = st.slider("History shown (hours)", 24, 336, 168, step=24)
    show_band = st.toggle("Show confidence band", value=True)
    show_components = st.toggle("Show Prophet / LSTM lines", value=True)

    st.divider()
    st.markdown("**Data source**")
    source = st.radio(
        "Data source", ["Mock (synthetic)", "Live (EIA)"],
        label_visibility="collapsed",
    )
    is_live = source.startswith("Live")
    region = st.text_input("EIA region / RTO", value="PJM") if is_live else "PJM"

    # Key status indicators
    eia_ok = bool(os.environ.get("EIA_API_KEY"))
    claude_ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
    st.caption(f"{'🟢' if eia_ok else '⚪'} EIA_API_KEY    {'🟢' if claude_ok else '⚪'} ANTHROPIC_API_KEY")

    st.divider()
    if "cache_key" not in st.session_state:
        st.session_state.cache_key = 0

    if st.button("🔄 Refresh market data", use_container_width=True):
        try:
            fetch_and_save_data(source="live" if is_live else "mock", respondent=region)
            st.session_state.cache_key += 1
            st.cache_data.clear()
            st.toast(f"Market data refreshed ({'live EIA' if is_live else 'mock'})", icon="✅")
        except Exception as e:
            st.error(f"Live fetch failed — keeping existing data.\n\n{e}")

    if st.button("🧠 Retrain models", use_container_width=True):
        for f in ("prophet_model.pkl", "lstm_model.pth", "scaler.pkl"):
            p = os.path.join(DATA_DIR, f)
            if os.path.exists(p):
                os.remove(p)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.cache_key += 1
        st.toast("Models cleared — retraining…", icon="🧠")

    st.divider()
    st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M}")


# ──────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>⚡ Energy Trading Agent</h1>
        <p>AI-powered energy price forecasting &amp; autonomous trading signals —
        ensemble of Prophet &amp; LSTM, orchestrated by a LangGraph agent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

ck = st.session_state.cache_key

# Prepare everything with a single spinner
with st.spinner("Spinning up models and generating forecast…"):
    status = ensure_models(ck)
    df = get_data(ck)
    forecast = get_forecast(horizon, ck)
    current_price = float(df["price"].iloc[-1])
    agent_state = run_trading_agent(forecast, current_price)
    report = create_stakeholder_report(agent_state)
    signal = agent_state.get("trading_signal", "UNKNOWN")

avg_pred = float(forecast["ensemble_pred"].mean())
max_pred = float(forecast["ensemble_pred"].max())
min_pred = float(forecast["ensemble_pred"].min())
pct_change = (avg_pred - current_price) / current_price * 100 if current_price else 0.0
meta = SIGNAL_META.get(signal, SIGNAL_META["UNKNOWN"])


# ──────────────────────────────────────────────────────────────────────────
# KPI cards
# ──────────────────────────────────────────────────────────────────────────
def metric_card(label, value, delta_html=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


c1, c2, c3, c4 = st.columns(4)
delta_color = BUY_COLOR if pct_change >= 0 else SELL_COLOR
arrow = "▲" if pct_change >= 0 else "▼"

with c1:
    st.markdown(metric_card("Current Price", f"${current_price:,.2f}",
                            '<div class="metric-delta" style="color:#8a7434;">latest observed</div>'),
                unsafe_allow_html=True)
with c2:
    st.markdown(metric_card(f"Avg Forecast · {horizon}h", f"${avg_pred:,.2f}",
                            f'<div class="metric-delta" style="color:{delta_color};">{arrow} {pct_change:+.2f}% vs current</div>'),
                unsafe_allow_html=True)
with c3:
    st.markdown(metric_card(f"Forecast Range · {horizon}h", f"${min_pred:,.0f} – ${max_pred:,.0f}",
                            f'<div class="metric-delta" style="color:#8a7434;">spread ${max_pred-min_pred:,.2f}</div>'),
                unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Agent Signal",
                            f'<span style="color:{meta["color"]};">{meta["emoji"]} {signal}</span>',
                            f'<div class="metric-delta" style="color:#8a7434;">{meta["blurb"]}</div>'),
                unsafe_allow_html=True)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# Signal banner
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="signal-card" style="background:linear-gradient(135deg,{meta['color']}1f,transparent);">
        <span class="signal-badge" style="background:{meta['color']};">{signal}</span>
        <div>
            <div style="font-weight:700;font-size:1.05rem;color:#4a3a14;">Trading Recommendation</div>
            <div style="color:#8a7434;font-size:0.92rem;">{agent_state.get('reasoning','')}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# Forecast chart
# ──────────────────────────────────────────────────────────────────────────
left, right = st.columns([2.4, 1])

with left:
    st.markdown('<div class="section-title">📈 Price Forecast</div>', unsafe_allow_html=True)

    hist = df.tail(history_window)
    fig = go.Figure()

    # Confidence band
    if show_band:
        try:
            band = get_prophet_band(horizon, ck)
            fig.add_trace(go.Scatter(
                x=list(band["ds"]) + list(band["ds"][::-1]),
                y=list(band["yhat_upper"]) + list(band["yhat_lower"][::-1]),
                fill="toself", fillcolor="rgba(184,134,11,0.16)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="Confidence band", showlegend=True,
            ))
        except Exception:
            pass

    # History
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["price"], mode="lines",
        name="Historical", line=dict(color=HIST_COLOR, width=1.8),
    ))

    # Bridge last historical point into the forecast for visual continuity
    bridge_x = [hist["timestamp"].iloc[-1]]
    bridge_y = [hist["price"].iloc[-1]]

    if show_components:
        fig.add_trace(go.Scatter(
            x=bridge_x + list(forecast["timestamp"]), y=bridge_y + list(forecast["prophet_pred"]),
            mode="lines", name="Prophet", line=dict(color=PROPHET_COLOR, width=1.6, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=bridge_x + list(forecast["timestamp"]), y=bridge_y + list(forecast["lstm_pred"]),
            mode="lines", name="LSTM", line=dict(color=LSTM_COLOR, width=1.6, dash="dot"),
        ))

    fig.add_trace(go.Scatter(
        x=bridge_x + list(forecast["timestamp"]), y=bridge_y + list(forecast["ensemble_pred"]),
        mode="lines", name="Ensemble", line=dict(color=ACCENT, width=3.2),
    ))

    # "Now" marker
    fig.add_vline(x=hist["timestamp"].iloc[-1], line_width=1, line_dash="dash",
                  line_color="rgba(110,84,16,0.55)")

    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        height=440, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK_DIM)),
        hovermode="x unified",
        xaxis=dict(gridcolor="rgba(184,134,11,0.18)", linecolor="rgba(110,84,16,0.4)", title=None),
        yaxis=dict(gridcolor="rgba(184,134,11,0.18)", linecolor="rgba(110,84,16,0.4)", title="Price ($/MWh)"),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">🧠 Agent Report</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown('</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download report (Markdown)",
        data=report,
        file_name=f"trading_report_{datetime.now():%Y%m%d_%H%M}.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.write("")

# ──────────────────────────────────────────────────────────────────────────
# Forecast detail table
# ──────────────────────────────────────────────────────────────────────────
with st.expander("📋 Hourly forecast detail", expanded=False):
    table = forecast.copy()
    table.columns = ["Timestamp", "Prophet", "LSTM", "Ensemble"]
    table["Timestamp"] = pd.to_datetime(table["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        table.style.format({"Prophet": "${:.2f}", "LSTM": "${:.2f}", "Ensemble": "${:.2f}"})
            .background_gradient(cmap="YlOrBr", subset=["Ensemble"]),
        use_container_width=True, height=360,
    )
    st.download_button(
        "⬇️ Download forecast (CSV)",
        data=forecast.to_csv(index=False),
        file_name=f"forecast_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )

engine = agent_state.get("agent_engine", "rule-based")
st.caption("·  ".join(status) + f"  ·  Agent engine: {engine}  ·  for demonstration only.")
