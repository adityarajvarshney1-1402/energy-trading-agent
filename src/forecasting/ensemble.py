import pandas as pd
from .prophet_model import generate_prophet_forecast, load_prophet_model
from .lstm_model import generate_lstm_forecast, PriceLSTM
import torch
import pickle
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def generate_ensemble_forecast(df: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
    """
    Combines Prophet and LSTM forecasts to generate an ensemble prediction.
    """
    # 1. Prophet Forecast
    prophet_model = load_prophet_model()
    if prophet_model is None:
        raise ValueError("Prophet model not found. Please train models first.")
        
    prophet_forecast = generate_prophet_forecast(prophet_model, periods)
    
    # 2. LSTM Forecast
    lstm_model = PriceLSTM()
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.pth")
    if not os.path.exists(lstm_path):
         raise ValueError("LSTM model not found. Please train models first.")
         
    lstm_model.load_state_dict(torch.load(lstm_path))
    
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    lstm_forecast = generate_lstm_forecast(lstm_model, scaler, df, periods)
    
    # 3. Ensemble (Simple Average for now)
    # We only care about the future predictions, so we take the last `periods` from prophet
    prophet_future = prophet_forecast.tail(periods).reset_index(drop=True)
    lstm_future = lstm_forecast.reset_index(drop=True)
    
    ensemble_df = pd.DataFrame({
        'timestamp': prophet_future['ds'],
        'prophet_pred': prophet_future['yhat'],
        'lstm_pred': lstm_future['lstm_yhat'],
        'ensemble_pred': (prophet_future['yhat'] + lstm_future['lstm_yhat']) / 2.0
    })
    
    return ensemble_df
