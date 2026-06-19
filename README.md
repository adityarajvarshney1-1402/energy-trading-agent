# ⚡ Energy Trading Agent

AI-powered energy price forecasting and autonomous trading signals — an ensemble of
**Prophet** and **LSTM** models, orchestrated by a **LangGraph** agent, surfaced through
a premium **Streamlit** dashboard and a **FastAPI** service.

---

## Features

- **Ensemble forecasting** — combines a Facebook **Prophet** model (seasonality/trend) with a
  PyTorch **LSTM** for short-horizon price prediction, averaged into a single ensemble curve.
- **Autonomous trading agent** — a **LangGraph** workflow analyzes the forecast and emits a
  `BUY` / `SELL` / `HOLD` signal with reasoning. Uses **Claude** (`ChatAnthropic`) when an API
  key is present, and falls back to deterministic rule-based logic otherwise.
- **Premium Streamlit dashboard** — beige & gold themed control room with KPI cards,
  an interactive Plotly forecast chart (history + Prophet/LSTM/ensemble + confidence band),
  the agent's report, and CSV/Markdown exports.
- **Live or mock data** — synthetic generator out of the box, or live hourly data from the
  **EIA v2 open API** via a sidebar toggle.
- **REST API** — FastAPI endpoints for forecasts and trading signals.

---

## Project structure

```
energy-trading-agent/
├── dashboard.py              # Streamlit dashboard (main UI)
├── requirements.txt
└── src/
    ├── main.py               # FastAPI app
    ├── data_pipeline.py      # mock generator + live EIA fetch
    ├── forecasting/
    │   ├── prophet_model.py
    │   ├── lstm_model.py
    │   └── ensemble.py
    └── agent/
        ├── langgraph_agent.py  # LangGraph workflow + Claude/rule-based analysis
        └── report_generator.py
```

---

## Setup

```bash
# 1. (recommended) create & activate a virtual environment
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

> Models and data are **not** committed to the repo — they are trained/generated automatically
> on first run.

---

## Running

### Streamlit dashboard
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501. On first launch it trains the Prophet and LSTM models
(takes a moment), then renders the forecast and trading signal.

### FastAPI service
```bash
uvicorn src.main:app --reload
```
Interactive docs at http://localhost:8000/docs.

| Method | Endpoint           | Description                                  |
|--------|--------------------|----------------------------------------------|
| GET    | `/forecast`        | Ensemble forecast for the next `periods` hrs |
| GET    | `/trading-signal`  | Agent trading signal + stakeholder report    |
| GET    | `/refresh-data`    | Regenerate the underlying dataset            |

---

## Configuration

Set these as environment variables (or in a local `.env` file — gitignored):

| Variable             | Purpose                                                        |
|----------------------|---------------------------------------------------------------|
| `ANTHROPIC_API_KEY`  | Enables the Claude-powered trading agent (else rule-based)     |
| `ENERGY_AGENT_MODEL` | Override the Claude model (default `claude-sonnet-4-6`)        |
| `EIA_API_KEY`        | US hourly energy data ([free key](https://www.eia.gov/opendata/register.php)) |

### Data sources

Pick the source from the dashboard sidebar:

- **Mock (synthetic)** — realistic generated prices; no key needed. *Default.*
- **Commodities (yfinance)** — real market prices from Yahoo Finance with **no key**:
  Natural Gas (`NG=F`), WTI/Brent crude (`CL=F`/`BZ=F`), energy ETFs (`XLE`, `UNG`), and stocks.
- **EIA (US demand)** — hourly US grid demand used as a price proxy. Requires `EIA_API_KEY`.

---

## Notes & disclaimer

- The default dataset is **synthetic** mock data; the live EIA series is hourly grid data used
  as a price proxy. This project is for **demonstration and educational purposes only** and is
  **not financial advice**.
