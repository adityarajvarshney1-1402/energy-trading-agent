import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(PriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_lstm_data(df: pd.DataFrame, seq_length: int = 24):
    """
    Prepares data for LSTM training.
    Uses 'price' column.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:(i + seq_length)])
        y.append(scaled_data[i + seq_length])
        
    return torch.tensor(np.array(x), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32), scaler

def train_lstm_model(df: pd.DataFrame, epochs: int = 50, seq_length: int = 24):
    """
    Trains the LSTM model.
    """
    x_train, y_train, scaler = prepare_lstm_data(df, seq_length)
    
    model = PriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pth"))
    
    import pickle
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
        
    return model, scaler

def generate_lstm_forecast(model: PriceLSTM, scaler: MinMaxScaler, df: pd.DataFrame, periods: int = 24, seq_length: int = 24):
    """
    Generates a forecast for the next `periods` hours using the LSTM model.
    """
    model.eval()
    
    # Get the last seq_length data points
    last_data = df['price'].values[-seq_length:].reshape(-1, 1)
    scaled_last_data = scaler.transform(last_data)
    
    current_seq = torch.tensor(scaled_last_data, dtype=torch.float32).unsqueeze(0)
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(periods):
            pred = model(current_seq)
            predictions.append(pred.item())
            
            # Update sequence by appending the new prediction and dropping the oldest
            new_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1)), dim=1)
            current_seq = new_seq
            
    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Create forecast dataframe
    last_date = pd.to_datetime(df['timestamp'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(periods)]
    
    return pd.DataFrame({
        'ds': future_dates,
        'lstm_yhat': predictions.flatten()
    })
