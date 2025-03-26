import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser
import numpy as np

# User settings
account_type = "PRACTICE"
trade_amount_input = "12%"  # 5% of balance

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
def calculate_sma(closes, period):
    if len(closes) < period:
        return None
    return np.mean(closes[-period:])

# Function to calculate EMA
def calculate_ema(closes, period):
    if len(closes) < period:
        return None
    ema = closes[-period]
    multiplier = 2 / (period + 1)
    for price in closes[-period + 1:]:
        ema = (price - ema) * multiplier + ema
    return ema

# Function to calculate MACD
def calculate_macd(closes):
    if len(closes) < 26:
        return None, None
    ema_12 = calculate_ema(closes, 12)
    ema_26 = calculate_ema(closes, 26)
    if ema_12 is None or ema_26 is None:
        return None, None
    macd_line = ema_12 - ema_26
    # Calculate Signal Line (9-period EMA of MACD Line)
    macd_values = []
    for i in range(len(closes) - 26, len(closes)):
        if len(closes[:i+1]) < 26:  # Ensure enough data for EMA_26
            continue
        ema_12 = calculate_ema(closes[:i+1], 12)
        ema_26 = calculate_ema(closes[:i+1], 26)
        if ema_12 is None or ema_26 is None:
            continue
        macd_values.append(ema_12 - ema_26)
    if len(macd_values) < 9:  # Ensure enough data for signal line
        return macd_line, None
    signal_line = calculate_ema(macd_values, 9)
    return macd_line, signal_line

# Function to calculate ATR
def calculate_atr(price_history, period=14):
    if len(price_history) < period:
        return None
    tr_values = []
    for i in range(1, len(price_history)):
        high = price_history[i]['max']
        low = price_history[i]['min']
        prev_close = price_history[i-1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    if len(tr_values) < period:
        return None
    return np.mean(tr_values[-period:])

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
        self.rsi_overbought = 70  # Changed from 75
        self.rsi_oversold = 30    # Changed from 25
        self.proximity_factor = 0.3
        self.put_loss_streak = 0
        self.call_loss_streak = 0
        self.trade_history = []
        self.consecutive_losses = 0
        self.pause_until = 0
        self.last_trade_time = 0

    def adjust_parameters(self, trade_result, direction):
        self.trade_history.append({'direction': direction, 'result': trade_result})
        if len(self.trade_history) > 20:
            self.trade_history.pop(0)

        if trade_result < 0:
            self.consecutive_losses += 1
            if direction == "put":
                self.put_loss_streak += 1
                self.call_loss_streak = 0
                self.rsi_overbought = max(65, self.rsi_overbought - 2)
                self.proximity_factor = max(0.2, self.proximity_factor - 0.02)
                print(f"[{time.strftime('%H:%M:%S')}] Put trade lost. Adjusted RSI overbought to {self.rsi_overbought}, proximity factor to {self.proximity_factor}")
            elif direction == "call":
                self.call_loss_streak += 1
                self.put_loss_streak = 0
                self.rsi_oversold = min(35, self.rsi_oversold + 2)
                self.proximity_factor = max(0.2, self.proximity_factor - 0.02)
                print(f"[{time.strftime('%H:%M:%S')}] Call trade lost. Adjusted RSI oversold to {self.rsi_oversold}, proximity factor to {self.proximity_factor}")
            
            if self.consecutive_losses >= 2:
                self.pause_until = time.time() + 900
                print(f"[{time.strftime('%H:%M:%S')}] 2 consecutive losses detected. Pausing trading until {time.strftime('%H:%M:%S', time.localtime(self.pause_until))}")
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

    def can_trade(self):
        return time.time() >= self.last_trade_time + 300

    def update_last_trade_time(self):
        self.last_trade_time = time.time()

# Function to detect trade setup (with loosened parameters)
def detect_trade_setup(candle, near_support, near_resistance, price_history, channel_range, params):
    body = abs(candle['open'] - candle['close'])
    total_range = candle['max'] - candle['min']

    if total_range == 0:
        return None

    # Adjusted volatility filter
    if channel_range < 0.0006:  # Lowered from 0.0008
        print(f"[{time.strftime('%H:%M:%S')}] Channel range too narrow ({channel_range}), skipping trade.")
        return None

    # Adjusted ATR filter
    atr = calculate_atr(price_history, period=14)
    if atr is None or atr < 0.00015:  # Lowered from 0.0002
        print(f"[{time.strftime('%H:%M:%S')}] ATR too low ({atr}), skipping trade.")
        return None

    # RSI (14-period)
    closes = [c['close'] for c in price_history]
    rsi = calculate_rsi(closes, period=14)
    if rsi is None:
        return None
    overbought = rsi > params.rsi_overbought
    oversold = rsi < params.rsi_oversold

    # Dual-SMA trend filter
    sma_20 = calculate_sma(closes, period=20)
    sma_30 = calculate_sma(closes, period=30)  # Changed from 50
    if sma_20 is None or sma_30 is None:
        return None
    trend_up = sma_20 > sma_30  # Removed current_price > sma_20
    trend_down = sma_20 < sma_30  # Removed current_price < sma_20

    # MACD confirmation (check direction instead of crossover)
    macd_line, signal_line = calculate_macd(closes)
    if macd_line is None or signal_line is None:
        return None
    macd_bullish = macd_line > signal_line
    macd_bearish = macd_line < signal_line

    print(f"[{time.strftime('%H:%M:%S')}] RSI: {rsi:.2f}, Overbought: {overbought}, Oversold: {oversold}, 20-SMA: {sma_20:.5f}, 30-SMA: {sma_30:.5f}, Trend Up: {trend_up}, Trend Down: {trend_down}, MACD: {macd_line:.5f}, Signal: {signal_line:.5f}")

    # Trade logic (removed pin bar requirement, increased body threshold)
    if body < 0.00015:  # Increased from 0.0001
        if near_resistance and trend_down and overbought and macd_bearish:
            return "put"
        if near_support and trend_up and oversold and macd_bullish:
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
            if len(price_history) > 35:  # Changed from 50
                price_history.pop(0)

            if len(price_history) >= 35:  # Changed from 50
                # Identify SNR levels
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

                if params.is_paused():
                    print(f"[{time.strftime('%H:%M:%S')}] Trading paused due to consecutive losses. Resuming at {time.strftime('%H:%M:%S', time.localtime(params.pause_until))}")
                    continue

                if not params.can_trade():
                    print(f"[{time.strftime('%H:%M:%S')}] Trade cooldown active. Next trade available at {time.strftime('%H:%M:%S', time.localtime(params.last_trade_time + 300))}")
                    continue

                direction = detect_trade_setup(latest_candle, near_support, near_resistance, price_history, channel_range, params)
                if direction:
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
                        params.update_last_trade_time()
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
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] An unexpected error occurred: {e}")
    raise  # Re-raise the exception for debugging purposes