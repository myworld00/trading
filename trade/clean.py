import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Load and Preprocess Data
df = pd.read_csv('C:/Users/junai/Desktop/trade/EURUSD_Candlestick_1_M_BID_01.01.2024-22.03.2025.csv')
df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')  # Fixed format
df = df.dropna().drop_duplicates(subset=['Gmt time'])

# Candlestick Patterns
def is_doji(open, high, low, close, body_threshold=0.1):
    body = abs(close - open)
    range = high - low
    return body <= body_threshold * range if range > 0 else False

def is_bullish_engulfing(prev_open, prev_close, curr_open, curr_close):
    prev_body = prev_close - prev_open
    curr_body = curr_close - curr_open
    return (prev_body < 0 and curr_body > 0 and curr_open <= prev_close and curr_close > prev_open)

def is_bearish_engulfing(prev_open, prev_close, curr_open, curr_close):
    prev_body = prev_close - prev_open
    curr_body = curr_close - curr_open
    return (prev_body > 0 and curr_body < 0 and curr_open >= prev_close and curr_close < prev_open)

df['prev_open'] = df['Open'].shift(1)
df['prev_close'] = df['Close'].shift(1)
df['is_doji'] = df.apply(lambda row: is_doji(row['Open'], row['High'], row['Low'], row['Close']), axis=1)
df['is_bullish_engulfing'] = df.apply(lambda row: is_bullish_engulfing(
    row['prev_open'], row['prev_close'], row['Open'], row['Close']), axis=1)
df['is_bearish_engulfing'] = df.apply(lambda row: is_bearish_engulfing(
    row['prev_open'], row['prev_close'], row['Open'], row['Close']), axis=1)

# Manual Technical Indicators
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

df['rsi'] = calculate_rsi(df['Close'])
df['macd'], df['macd_signal'] = calculate_macd(df['Close'])
df['bb_middle'] = df['Close'].rolling(window=20).mean()
df['bb_std'] = df['Close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# Normalize features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features].fillna(0))
pattern_features = ['is_doji', 'is_bullish_engulfing', 'is_bearish_engulfing']

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# Create sequences
sequence_length = 10
X, y = [], []
all_features = features + pattern_features
for i in range(len(df) - sequence_length):
    X.append(df[all_features].iloc[i:i+sequence_length].values)
    y.append(df['target'].iloc[i+sequence_length])
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 2: Define LSTM Model
class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)

model = HybridLSTM(input_size=len(all_features), hidden_size=64, num_layers=2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train Model
num_epochs = 30
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        with torch.no_grad():
            train_preds = model(X_train)
            train_acc = ((train_preds > 0.5).float() == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.2%}")

# Evaluate
with torch.no_grad():
    preds = model(X_test)
    preds_binary = (preds > 0.5).float()
    accuracy = (preds_binary == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.2%}")

    # Simulate profit
    trades = preds_binary.numpy().flatten()
    returns = np.where(trades == y_test.numpy().flatten(), 0.8, -1)
    profit = returns.sum()
    print(f"Simulated profit (assuming $1 trades): ${profit:.2f}")

# Save model
torch.save(model.state_dict(), 'hybrid_lstm.pth')