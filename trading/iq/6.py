import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser
import numpy as np

# User settings
account_type = "PRACTICE"
trade_amount_input = "5%"  # Reduced to 5% to lower risk

# Connect to IQ Option
api = IQ_Option("juanidh0452@gmail.com", "96aQ7Y3t93@B*rW")
api.connect()
api.change_balance(account_type)

# Function to calculate trade amount based on current balance
def calculate_trade_amount(balance, trade_amount_input):
    if trade_amount_input.endswith("%"):
        percentage = float(trade_amount_input.strip("%")) / 100
        raw_trade_amount = balance * percentage
        trade_amount = int(raw_trade_amount)
        final_trade_amount = max(1, trade_amount)
        print(f"[{time.strftime('%H:%M:%S')}] Calculating trade amount: Balance=${balance}, Percentage={percentage*100}%, Raw Amount=${raw_trade_amount:.2f}, Final Amount=${final_trade_amount}")
        return final_trade_amount
    else:
        trade_amount = int(float(trade_amount_input))
        print(f"[{time.strftime('%H:%M:%S')}] Fixed trade amount: ${trade_amount}")
        return trade_amount

# Function to calculate SMA
def calculate_sma(closes, period=20):
    if len(closes) < period:
        return None
    return np.mean(closes[-period:])

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

# Function to identify SNR levels (15-candle lookback)
def identify_snr(price_history, lookback=15):
    if len(price_history) < lookback:
        return None, None
    highs = [candle['max'] for candle in price_history[-lookback:]]
    lows = [candle['min'] for candle in price_history[-lookback:]]
    resistance = max(highs)
    support = min(lows)
    return support, resistance

# Initial balance (for logging purposes)
initial_balance = api.get_balance()
print(f"Using {account_type} account. Initial Balance: ${initial_balance}")

# Launch IQ Option in browser
webbrowser.open("https://iqoption.com/en/login")

# Adaptive parameters with loss streak tracking
class AdaptiveParameters:
    def __init__(self):
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.proximity_factor = 0.3
        self.put_loss_streak = 0
        self.call_loss_streak = 0
        self.trade_history = []
        self.consecutive_losses = 0
        self.pause_until = 0

    def adjust_parameters(self, trade_result, direction):
        self.trade_history.append({'direction': direction, 'result': trade_result})
        if len(self.trade_history) > 20:
            self.trade_history.pop(0)

        if trade_result < 0:
            self.consecutive_losses += 1
            if direction == "put":
                self.put_loss_streak += 1
                self.call_loss_streak = 0
                self.rsi_overbought = max(60, self.rsi_overbought - 2)
                self.proximity_factor = max(0.2, self.proximity_factor - 0.02)
                print(f"[{time.strftime('%H:%M:%S')}] Put trade lost. Adjusted RSI overbought to {self.rsi_overbought}, proximity factor to {self.proximity_factor}")
            elif direction == "call":
                self.call_loss_streak += 1
                self.put_loss_streak = 0
                self.rsi_oversold = min(40, self.rsi_oversold + 2)
                self.proximity_factor = max(0.2, self.proximity_factor - 0.02)
                print(f"[{time.strftime('%H:%M:%S')}] Call trade lost. Adjusted RSI oversold to {self.rsi_oversold}, proximity factor to {self.proximity_factor}")
            
            # Pause trading after 3 consecutive losses
            if self.consecutive_losses >= 3:
                self.pause_until = time.time() + 300  # Pause for 5 minutes
                print(f"[{time.strftime('%H:%M:%S')}] 3 consecutive losses detected. Pausing trading until {time.strftime('%H:%M:%S', time.localtime(self.pause_until))}")
        else:
            self.consecutive_losses = 0
            if direction == "put":
                self.put_loss_streak = 0
            elif direction == "call":
                self.call_loss_streak = 0

    def get_bias(self):
        if self.put_loss_streak >= 2:
            return "call_bias"
        elif self.call_loss_streak >= 2:
            return "put_bias"
        return None

    def is_paused(self):
        return time.time() < self.pause_until

# Function to detect trade setup (with SNR, RSI, and trend filter)
def detect_trade_setup(candle, near_support, near_resistance, price_history, channel_range, params):
    body = abs(candle['open'] - candle['close'])
    wick_high = candle['max'] - max(candle['open'], candle['close'])
    wick_low = min(candle['open'], candle['close']) - candle['min']
    total_range = candle['max'] - candle['min']

    if total_range == 0:
        return None

    # Volatility filter
    if channel_range < 0.001:  # Increased threshold
        print(f"[{time.strftime('%H:%M:%S')}] Channel range too narrow ({channel_range}), skipping trade.")
        return None

    # RSI (14-period)
    closes = [c['close'] for c in price_history]
    rsi = calculate_rsi(closes, period=14)
    if rsi is None:
        return None
    overbought = rsi > params.rsi_overbought
    oversold = rsi < params.rsi_oversold
    rsi_trend_up = rsi > 50
    rsi_trend_down = rsi < 50

    # Trend filter using 20-SMA
    sma_20 = calculate_sma(closes, period=20)
    if sma_20 is None:
        return None
    current_price = candle['close']
    trend_up = current_price > sma_20
    trend_down = current_price < sma_20

    print(f"[{time.strftime('%H:%M:%S')}] RSI: {rsi:.2f}, Overbought: {overbought}, Oversold: {oversold}, 20-SMA: {sma_20:.5f}, Trend Up: {trend_up}, Trend Down: {trend_down}")

    # Check for bias due to loss streaks
    bias = params.get_bias()

    # Trade logic with trend filter
    if body < 0.0002:
        if near_resistance:
            if trend_down and overbought and not rsi_trend_up:
                return "put"
            if (bias == "call_bias" or trend_up) and not overbought:
                return "call"
        if near_support:
            if trend_up and oversold and not rsi_trend_down:
                return "call"
            if (bias == "put_bias" or trend_down) and not oversold:
                return "put"

    # Classic pin bar with trend filter
    if body < 0.0006 and total_range > body:
        if wick_high > 1.2 * body and wick_high > wick_low and near_resistance and overbought and not rsi_trend_up and trend_down:
            return "put"
        if wick_low > 1.2 * body and wick_low > wick_high and near_support and oversold and not rsi_trend_down and trend_up:
            return "call"
    return None

# Function to check trade status and return profit/loss
def check_trade_status(trade_id):
    for _ in range(3):
        result = api.get_async_order(trade_id)
        if result and 'option' in result:
            option = result['option']
            status = "closed" if option['closed_at'] else "open"
            profit = option.get('profit', 0) - option.get('amount', 0)
            return status, profit
        time.sleep(5)
    return "unknown", 0

# Real-time candle fetching with historical data
pair = "EURUSD-OTC"
print(f"Monitoring {pair} for channel-based trades...")

last_candle_time = 0
price_history = []
no_movement_count = 0
params = AdaptiveParameters()

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

            # Update price history for SNR and RSI
            price_history.append({
                'max': latest_candle['max'],
                'min': latest_candle['min'],
                'close': latest_candle['close']
            })
            if len(price_history) > 30:
                price_history.pop(0)

            if len(price_history) >= 20:  # Need at least 20 candles for SMA
                # Identify SNR levels (parallel channel boundaries)
                support, resistance = identify_snr(price_history, lookback=15)
                if support is None or resistance is None:
                    print(f"[{time.strftime('%H:%M:%S')}] Insufficient data for SNR.")
                    continue

                channel_range = resistance - support
                print(f"SNR Levels: Resistance={resistance}, Support={support}, Channel Range={channel_range}")

                current_price = latest_candle['close']
                proximity_threshold = max(channel_range * params.proximity_factor, 0.0003)
                near_resistance = abs(current_price - resistance) < proximity_threshold
                near_support = abs(current_price - support) < proximity_threshold
                print(f"Proximity: near_resistance={near_resistance}, near_support={near_support}, threshold={proximity_threshold}")

                # Check if trading is paused due to consecutive losses
                if params.is_paused():
                    print(f"[{time.strftime('%H:%M:%S')}] Trading paused due to consecutive losses. Resuming at {time.strftime('%H:%M:%S', time.localtime(params.pause_until))}")
                    continue

                direction = detect_trade_setup(latest_candle, near_support, near_resistance, price_history, channel_range, params)
                if direction:
                    # Check balance and calculate trade amount before each trade
                    balance = api.get_balance()
                    trade_amount = calculate_trade_amount(balance, trade_amount_input)
                    print(f"[{time.strftime('%H:%M:%S')}] Trade setup detected at SNR! Attempting trade: {direction} with amount ${trade_amount}")
                    trade_result = api.buy_digital_spot(pair, trade_amount, direction, 1)
                    print(f"Trade result: {trade_result}")
                    if isinstance(trade_result, tuple) and trade_result[0]:
                        trade_id = trade_result[1]
                        time.sleep(65)
                        status, profit = check_trade_status(trade_id)
                        print(f"Trade {trade_id}: Status={status}, Profit=${profit}")
                        params.adjust_parameters(profit, direction)
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