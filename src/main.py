from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_pipeline import load_data, fetch_and_save_data
from src.forecasting.prophet_model import train_prophet_model, load_prophet_model
from src.forecasting.lstm_model import train_lstm_model
from src.forecasting.ensemble import generate_ensemble_forecast
from src.agent.langgraph_agent import run_trading_agent
from src.agent.report_generator import create_stakeholder_report

app = FastAPI(
    title="Energy Price Forecasting & Trading Agent API",
    description="API for fetching energy forecasts and agentic trading signals.",
    version="1.0.0"
)

class ForecastResponse(BaseModel):
    timestamp: list[str]
    prophet_pred: list[float]
    lstm_pred: list[float]
    ensemble_pred: list[float]

class SignalResponse(BaseModel):
    report: str
    signal: str

@app.on_event("startup")
async def startup_event():
    """
    On startup, generate mock data and train initial models if they don't exist.
    """
    print("Checking data and models...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Check data
    df = load_data()
    
    # 2. Check/Train Prophet
    if load_prophet_model() is None:
        print("Training Prophet model...")
        train_prophet_model(df)
        
    # 3. Check/Train LSTM
    if not os.path.exists(os.path.join(data_dir, "lstm_model.pth")):
        print("Training LSTM model...")
        train_lstm_model(df, epochs=10) # 10 epochs for quick startup
        
    print("Startup complete.")

@app.get("/refresh-data")
async def refresh_data():
    """
    Refreshes the historical data (generates new mock data).
    """
    fetch_and_save_data()
    return {"message": "Data refreshed successfully."}

@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(periods: int = 24):
    """
    Returns the ensemble forecast for the next `periods` hours.
    """
    try:
        df = load_data()
        forecast = generate_ensemble_forecast(df, periods)
        
        return {
            "timestamp": forecast['timestamp'].astype(str).tolist(),
            "prophet_pred": forecast['prophet_pred'].tolist(),
            "lstm_pred": forecast['lstm_pred'].tolist(),
            "ensemble_pred": forecast['ensemble_pred'].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading-signal", response_model=SignalResponse)
async def get_trading_signal():
    """
    Runs the LangGraph agent to analyze the forecast and returns a trading signal and report.
    """
    try:
        df = load_data()
        current_price = df['price'].iloc[-1]
        
        forecast = generate_ensemble_forecast(df, periods=24)
        
        # Run agent
        agent_state = run_trading_agent(forecast, current_price)
        
        # Generate report
        report = create_stakeholder_report(agent_state)
        
        return {
            "report": report,
            "signal": agent_state.get("trading_signal", "UNKNOWN")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
