import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser

# User settings
account_type = "PRACTICE"
trade_amount_input = "1"

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

# Function to detect trade setup (with wider thresholds)
def detect_trade_setup(candle, near_upper, near_lower):
    body = abs(candle['open'] - candle['close'])
    wick_high = candle['max'] - max(candle['open'], candle['close'])
    wick_low = min(candle['open'], candle['close']) - candle['min']
    total_range = candle['max'] - candle['min']

    # Skip if no movement
    if total_range == 0:
        return None

    # Relaxed criteria: Wider small body threshold
    if body < 0.0002:  # Widened from 0.0001
        if near_upper:
            return "put"  # At upper channel, expect reversal down
        if near_lower:
            return "call"  # At lower channel, expect bounce up
    # Classic pin bar: Wider thresholds
    if body < 0.0006 and total_range > body:  # Widened from 0.0004
        if wick_high > 1.2 * body and wick_high > wick_low and near_upper:  # Reduced from 1.5
            return "put"
        if wick_low > 1.2 * body and wick_low > wick_high and near_lower:  # Reduced from 1.5
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

# Real-time candle fetching with historical data
pair = "EURUSD-OTC"
print(f"Monitoring {pair} for channel-based trades...")

last_candle_time = 0
price_history = []
no_movement_count = 0

try:
    while True:
        # Get current timestamp
        current_time = int(time.time())
        # Fetch the most recent closed candle (1 minute ago)
        candle_time = current_time - (current_time % 60) - 60  # Last closed candle

        if candle_time != last_candle_time:
            # Fetch the last closed candle
            candles = api.get_candles(pair, 60, 1, candle_time)
            if not candles:
                print(f"[{time.strftime('%H:%M:%S')}] No candle data available.")
                time.sleep(1)
                continue

            latest_candle = candles[-1]
            print("Raw candle data:", latest_candle)

            # Check for no movement
            if latest_candle['open'] == latest_candle['close'] == latest_candle['min'] == latest_candle['max']:
                no_movement_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No price movement in candle. Count: {no_movement_count}")
                if no_movement_count > 5:
                    print("Too many candles with no movement. Possible API issue.")
            else:
                no_movement_count = 0

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

                # Check if price is near channel boundary (wider range)
                current_price = latest_candle['close']
                near_upper = abs(current_price - upper_channel) < channel_range * 0.2  # Widened from 0.1
                near_lower = abs(current_price - lower_channel) < channel_range * 0.2  # Widened from 0.1

                # Detect trade setup
                direction = detect_trade_setup(latest_candle, near_upper, near_lower)
                if direction:
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

        # Sleep until the next minute
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")