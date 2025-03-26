import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser

# User settings
account_type = "PRACTICE"
trade_amount_input = "5"

# Connect to IQ Option
api = IQ_Option("juanidh0452@gmail.com", "96aQ7Y3t93@B*rW")
api.connect()
api.change_balance(account_type)
balance = api.get_balance()

# Calculate trade amount
if trade_amount_input.endswith("%"):
    percentage = float(trade_amount_input.strip("%")) / 100
    trade_amount = int(balance * percentage)
else:
    trade_amount = int(float(trade_amount_input))

print(f"Using {account_type} account. Balance: ${balance}. Trade amount: ${trade_amount}")

# Launch IQ Option in browser
webbrowser.open("https://iqoption.com/en/login")

# Function to detect pin bar with direction (updated with your metrics)
def is_pin_bar(candle, near_channel=False):
    body = abs(candle['open'] - candle['close'])
    wick_high = candle['max'] - max(candle['open'], candle['close'])
    wick_low = min(candle['open'], candle['close']) - candle['min']
    total_range = candle['max'] - candle['min']

    body_limit = 0.0002  # Based on your avg body
    wick_multiplier = 1.5 if near_channel else 2

    if body < body_limit and total_range > body:
        # Bearish pin (put): Upper wick should be large
        if wick_high > wick_multiplier * body and wick_high > 0.0004 and wick_low < 0.0001:
            return "put"
        # Bullish pin (call): Lower wick should be large
        if wick_low > wick_multiplier * body and wick_low > 0.00035 and wick_high < 0.00015:
            return "call"
    return None

# Function to check trade status
def check_trade_status(trade_id):
    positions = api.get_digital_position(trade_id)
    if positions and 'orders' in positions:
        status = positions['orders'][0].get('status', 'unknown')
        profit = positions['orders'][0].get('profit', 0)
        return f"Status: {status}, Profit: ${profit}"
    return "Trade status not found."

# Real-time candle streaming
pair = "EURUSD-OTC"
api.start_candles_stream(pair, 60, 1)
print(f"Monitoring {pair} for pin bar setups with channel detection...")

last_candle_id = None
price_history = []

try:
    while True:
        candles = api.get_realtime_candles(pair, 60)
        if candles:
            latest_candle = list(candles.values())[0]
            candle_id = latest_candle['id']

            if candle_id != last_candle_id:
                print("Raw candle data:", latest_candle)
                df = pd.DataFrame([latest_candle])

                # Update price history for channel
                price_history.append({'max': latest_candle['max'], 'min': latest_candle['min']})
                if len(price_history) > 30:
                    price_history.pop(0)

                # Estimate channel boundaries
                if len(price_history) >= 10:
                    highs = [x['max'] for x in price_history]
                    lows = [x['min'] for x in price_history]
                    upper_channel = max(highs)
                    lower_channel = min(lows)
                    channel_range = upper_channel - lower_channel
                    print(f"Channel: Upper={upper_channel}, Lower={lower_channel}")

                    # Check if price is near channel boundary
                    current_price = latest_candle['close']
                    near_upper = abs(current_price - upper_channel) < channel_range * 0.1
                    near_lower = abs(current_price - lower_channel) < channel_range * 0.1
                    near_channel = near_upper or near_lower

                    # Detect pin bar
                    direction = is_pin_bar(df.iloc[0], near_channel)
                    if direction and near_channel:
                        print(f"[{time.strftime('%H:%M:%S')}] Pin bar detected near channel! Attempting trade: {direction}")
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
                        print(f"[{time.strftime('%H:%M:%S')}] No pin bar or not near channel. Body: {body}, Upper wick: {wick_high}, Lower wick: {wick_low}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Collecting price history...")

                last_candle_id = candle_id

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
    api.stop_candles_stream(pair, 60)