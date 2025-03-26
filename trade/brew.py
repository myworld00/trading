import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Step 1: Load and Preprocess Data
print("Loading data...")
df = pd.read_csv('C:/Users/junai/Desktop/trade/EURUSD_Candlestick_1_M_BID_01.01.2024-22.03.2025.csv')
df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
df = df.dropna().drop_duplicates(subset=['Gmt time'])

# Filter active hours (8:00-16:00 GMT)
df['hour'] = df['Gmt time'].dt.hour
df = df[df['hour'].between(8, 16)]

# Short-Term Features
print("Calculating features...")
df['delta'] = (df['Close'] - df['Open']) / df['Open']  # Candle direction
df['range'] = (df['High'] - df['Low']) / df['Open']    # Volatility
df['trend_3'] = df['Close'].pct_change(periods=3)      # 3-min trend
df['rsi_short'] = df['Close'].pct_change().rolling(window=5).apply(
    lambda x: np.mean(x[x > 0]) / -np.mean(x[x < 0]) if np.mean(x[x < 0]) != 0 else 50, raw=True)
df['rsi_short'] = 100 - (100 / (1 + df['rsi_short']))
df['volume_trend'] = df['Volume'].pct_change(periods=3)

# Normalize features
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'delta', 'range', 'trend_3', 'rsi_short', 'volume_trend']
print("Normalizing data...")
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features].fillna(0))

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# Create sequences
sequence_length = 5  # Shorter window
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

# Step 2: Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return self.sigmoid(out)

model = GRUModel(input_size=len(features), hidden_size=64, num_layers=2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Custom profit-based loss (optional)
def profit_loss(outputs, targets):
    preds = (outputs > 0.6).float()
    returns = torch.where(preds == targets, torch.tensor(0.8), torch.tensor(-1.0))
    return -returns.mean()  # Minimize loss = maximize profit

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

# Evaluate
print("Evaluating model...")
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_binary = (preds > 0.6).float()  # Higher threshold for confidence
    accuracy = (preds_binary == y_test).float().mean()
    print(f"Test Accuracy (threshold 0.6): {accuracy.item():.2%}")

    trades = preds_binary.numpy().flatten()
    actual = y_test.numpy().flatten()
    returns = np.where(trades == actual, 0.8, -1)
    profit = returns.sum()
    print(f"Simulated profit (assuming $1 trades): ${profit:.2f}")

    total_trades = len(trades)
    trades_made = np.sum(trades != 0)
    print(f"Total test trades: {total_trades}, Trades made: {trades_made}")

# Save model
torch.save(model.state_dict(), 'gru_model.pth')
print("Model saved as 'gru_model.pth'")