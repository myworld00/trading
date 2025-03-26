import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser
import numpy as np

# User settings
account_type = "PRACTICE"
trade_amount_input = "1%"

# Connect to IQ Option
api = IQ_Option("juanidh0452@gmail.com", "96aQ7Y3t93@B*rW")
api.connect()
api.change_balance(account_type)

# Function to calculate trade amount based on current balance
def calculate_trade_amount(balance, trade_amount_input):
    if trade_amount_input.endswith("%"):
        percentage = float(trade_amount_input.strip("%")) / 100
        trade_amount = int(balance * percentage)
        trade_amount = max(1, min(5, trade_amount))
    else:
        trade_amount = int(float(trade_amount_input))
    return trade_amount

# Function to calculate RSI
def calculate_rsi(closes, period=14):
    if len(closes) < period:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Initial balance and trade amount
balance = api.get_balance()
trade_amount = calculate_trade_amount(balance, trade_amount_input)
print(f"Using {account_type} account. Initial Balance: ${balance}. Initial Trade amount: ${trade_amount}")

# Launch IQ Option in browser
webbrowser.open("https://iqoption.com/en/login")

# Function to detect trade setup (with RSI)
def detect_trade_setup(candle, near_upper, near_lower, price_history, channel_range):
    body = abs(candle['open'] - candle['close'])
    wick_high = candle['max'] - max(candle['open'], candle['close'])
    wick_low = min(candle['open'], candle['close']) - candle['min']
    total_range = candle['max'] - candle['min']

    if total_range == 0:
        return None

    # Volatility filter
    if channel_range < 0.0005:
        print(f"[{time.strftime('%H:%M:%S')}] Channel range too narrow ({channel_range}), skipping trade.")
        return None

    # Momentum
    if len(price_history) >= 3:
        recent_closes = [c['close'] for c in price_history[-4:-1]]
        momentum = (recent_closes[-1] - recent_closes[0]) / 3
        strong_up_momentum = momentum > 0.00003
        strong_down_momentum = momentum < -0.00003
    else:
        strong_up_momentum = strong_down_momentum = False

    # Trend (3-period SMA)
    if len(price_history) >= 3:
        sma_closes = [c['close'] for c in price_history[-3:]]
        sma = sum(sma_closes) / 3
        trend_up = candle['close'] > sma
        trend_down = candle['close'] < sma
    else:
        trend_up = trend_down = False

    # RSI (14-period)
    closes = [c['close'] for c in price_history]
    rsi = calculate_rsi(closes, period=14)
    if rsi is None:
        return None
    overbought = rsi > 70
    oversold = rsi < 30
    rsi_trend_up = rsi > 50
    rsi_trend_down = rsi < 50

    print(f"[{time.strftime('%H:%M:%S')}] RSI: {rsi:.2f}, Overbought: {overbought}, Oversold: {oversold}")

    # Small body condition
    if body < 0.0002:
        if near_upper and overbought and not (strong_up_momentum or trend_up or rsi_trend_up):
            return "put"
        if near_lower and oversold and not (strong_down_momentum or trend_down or rsi_trend_down):
            return "call"
        if near_upper and (strong_up_momentum or trend_up or rsi_trend_up):
            return "call"
        if near_lower and (strong_down_momentum or trend_down or rsi_trend_down):
            return "put"

    # Classic pin bar
    if body < 0.0006 and total_range > body:
        if wick_high > 1.2 * body and wick_high > wick_low and near_upper and overbought and not (trend_up or rsi_trend_up):
            return "put"
        if wick_low > 1.2 * body and wick_low > wick_high and near_lower and oversold and not (trend_down or rsi_trend_down):
            return "call"
    return None

# Function to check trade status using async order
def check_trade_status(trade_id):
    for _ in range(3):
        result = api.get_async_order(trade_id)
        if result and 'option' in result:
            option = result['option']
            status = "closed" if option['closed_at'] else "open"
            profit = option.get('profit', 0) - option.get('amount', 0)
            return f"Status: {status}, Profit: ${profit}"
        time.sleep(5)
    return "Trade status not found after retries."

# Real-time candle fetching with historical data
pair = "EURUSD-OTC"
print(f"Monitoring {pair} for channel-based trades...")

last_candle_time = 0
price_history = []
no_movement_count = 0

try:
    while True:
        current_time = int(time.time())
        candle_time = current_time - (current_time % 60) - 60

        if candle_time != last_candle_time:
            candles = api.get_candles(pair, 60, 1, candle_time)
            if not candles:
                print(f"[{time.strftime('%H:%M:%S')}] No candle data available.")
                time.sleep(1)
                continue

            latest_candle = candles[-1]
            print("Raw candle data:", latest_candle)

            if latest_candle['open'] == latest_candle['close'] == latest_candle['min'] == latest_candle['max']:
                no_movement_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No price movement in candle. Count: {no_movement_count}")
                if no_movement_count > 5:
                    print("Too many candles with no movement. Possible API issue.")
            else:
                no_movement_count = 0

            # Update price history for channel and trend
            price_history.append({
                'max': latest_candle['max'],
                'min': latest_candle['min'],
                'close': latest_candle['close']
            })
            if len(price_history) > 30:
                price_history.pop(0)

            if len(price_history) >= 14:  # Need at least 14 candles for RSI
                highs = [x['max'] for x in price_history]
                lows = [x['min'] for x in price_history]
                upper_channel = max(highs)
                lower_channel = min(lows)
                channel_range = upper_channel - lower_channel
                print(f"Channel: Upper={upper_channel}, Lower={lower_channel}")

                current_price = latest_candle['close']
                proximity_threshold = max(channel_range * 0.3, 0.0003)
                near_upper = abs(current_price - upper_channel) < proximity_threshold
                near_lower = abs(current_price - lower_channel) < proximity_threshold
                print(f"Proximity: near_upper={near_upper}, near_lower={near_lower}, threshold={proximity_threshold}")

                direction = detect_trade_setup(latest_candle, near_upper, near_lower, price_history, channel_range)
                if direction:
                    balance = api.get_balance()
                    trade_amount = calculate_trade_amount(balance, trade_amount_input)
                    print(f"[{time.strftime('%H:%M:%S')}] Updated Balance: ${balance}, Trade amount: ${trade_amount}")
                    print(f"[{time.strftime('%H:%M:%S')}] Trade setup detected near channel! Attempting trade: {direction}")
                    trade_result = api.buy_digital_spot(pair, trade_amount, direction, 1)
                    print(f"Trade result: {trade_result}")
                    if isinstance(trade_result, tuple) and trade_result[0]:
                        trade_id = trade_result[1]
                        time.sleep(65)
                        print(f"Checking trade {trade_id}: {check_trade_status(trade_id)}")
                else:
                    body = abs(latest_candle['open'] - latest_candle['close'])
                    wick_high = latest_candle['max'] - max(latest_candle['open'], latest_candle['close'])
                    wick_low = min(latest_candle['open'], latest_candle['close']) - latest_candle['min']
                    print(f"[{time.strftime('%H:%M:%S')}] No trade setup. Body: {body}, Upper wick: {wick_high}, Lower wick: {wick_low}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Collecting price history...")

            last_candle_time = candle_time

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")