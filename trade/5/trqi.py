import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Step 1: Load and Preprocess 5-Minute Data
print("Loading data...")
df = pd.read_csv(r'C:\Users\junai\Desktop\trade\5\EURUSD_Candlestick_5_M_BID_08.01.2024-22.03.2025.csv')  # Update with your 5-min file name
df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
df = df.dropna().drop_duplicates(subset=['Gmt time'])

# Filter active hours (8:00-16:00 GMT)
df['hour'] = df['Gmt time'].dt.hour
df = df[df['hour'].between(8, 16)]

# Features
print("Calculating features...")
df['delta'] = (df['Close'] - df['Open']) / df['Open']
df['range'] = (df['High'] - df['Low']) / df['Open']
df['trend_3'] = df['Close'].pct_change(periods=3)
df['rsi'] = df['Close'].pct_change().rolling(window=14).apply(
    lambda x: np.mean(x[x > 0]) / -np.mean(x[x < 0]) if np.mean(x[x < 0]) != 0 else 50, raw=True)
df['rsi'] = 100 - (100 / (1 + df['rsi']))
df['ema_short'] = df['Close'].ewm(span=5, adjust=False).mean() - df['Close'].ewm(span=12, adjust=False).mean()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'delta', 'range', 'trend_3', 'rsi', 'ema_short']
print("Normalizing data...")
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features].fillna(0))

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# Create sequences
sequence_length = 10  # 50-minute lookback
print("Creating sequences...")
data_array = df[features].to_numpy()
target_array = df['target'].to_numpy()
n_samples = len(df) - sequence_length
X = np.zeros((n_samples, sequence_length, len(features)))
y = np.zeros((n_samples, 1))

for i in tqdm(range(n_samples), desc="Building sequences"):
    X[i] = data_array[i:i+sequence_length]
    y[i] = target_array[i+sequence_length]

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 2: Define MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size * sequence_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

model = MLPModel(input_size=len(features), hidden_size=128)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Custom profit-based loss
def profit_loss(outputs, targets):
    preds = (outputs > 0.65).float()
    returns = torch.where(preds == targets, torch.tensor(0.8), torch.tensor(-1.0))
    return -returns.mean()  # Maximize profit

# Step 3: Train Model
num_epochs = 50
batch_size = 128
print("Training model...")
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)  # Switch to profit_loss for profit focus
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train)
            train_acc = ((train_preds > 0.5).float() == y_train).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.2%}")

# Evaluate with Ensemble Filter
print("Evaluating model...")
model.eval()
with torch.no_grad():
    preds = model(X_test)
    # Add RSI filter: only trade when RSI < 30 (oversold) or > 70 (overbought)
    rsi_test = X_test[:, -1, features.index('rsi')]  # Last RSI in sequence
    trade_mask = (rsi_test < 0.3) | (rsi_test > 0.7)  # Normalized RSI thresholds
    preds_binary = (preds > 0.65).float() * trade_mask.float()  # Apply filter
    accuracy = (preds_binary == y_test).float().mean()
    print(f"Test Accuracy (threshold 0.65 + RSI filter): {accuracy.item():.2%}")

    trades = preds_binary.numpy().flatten()
    actual = y_test.numpy().flatten()
    returns = np.where(trades == actual, 0.8, -1) * (trades != 0)  # Only count executed trades
    profit = returns.sum()
    print(f"Simulated profit (assuming $1 trades): ${profit:.2f}")

    total_trades = len(trades)
    trades_made = np.sum(trades != 0)
    print(f"Total test trades: {total_trades}, Trades made: {trades_made}")

# Save model
torch.save(model.state_dict(), 'mlp_model.pth')
print("Model saved as 'mlp_model.pth'")