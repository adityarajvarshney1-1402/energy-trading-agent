import pandas as pd
from prophet import Prophet
import os
import pickle

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def train_prophet_model(df: pd.DataFrame, save_path: str = "prophet_model.pkl"):
    """
    Trains a Prophet model on the given dataset.
    Requires dataframe to have 'ds' (datetime) and 'y' (target) columns.
    """
    # Prepare dataframe for Prophet
    prophet_df = df.rename(columns={'timestamp': 'ds', 'price': 'y'})
    
    # Initialize and fit model
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    model.fit(prophet_df)
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, save_path), 'wb') as f:
        pickle.dump(model, f)
        
    return model

def load_prophet_model(load_path: str = "prophet_model.pkl"):
    filepath = os.path.join(MODEL_DIR, load_path)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_prophet_forecast(model: Prophet, periods: int = 24) -> pd.DataFrame:
    """
    Generates a forecast for the next `periods` hours.
    """
    future = model.make_future_dataframe(periods=periods, freq='h')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
