import pandas as pd
from iqoptionapi.stable_api import IQ_Option
import time
import webbrowser

# Connect to IQ Option
api = IQ_Option("juanidh0452@gmail.com", "96aQ7Y3t93@B*rW")
api.connect()
api.change_balance("PRACTICE")  # Demo account

# Launch IQ Option in browser
webbrowser.open("https://iqoption.com/en/login")

# Function to detect pin bar
def is_pin_bar(candle):
    body = abs(candle['open'] - candle['close'])
    wick = candle['max'] - candle['min']
    return wick > body and body < 0.0003  # Loosened criteria

# Real-time candle streaming
pair = "EURUSD-OTC"
api.start_candles_stream(pair, 60, 1)
print(f"Monitoring {pair} for pin bar setups...")

last_candle_id = None

try:
    while True:
        candles = api.get_realtime_candles(pair, 60)
        if candles:
            latest_candle = list(candles.values())[0]
            candle_id = latest_candle['id']

            if candle_id != last_candle_id:
                print("Raw candle data:", latest_candle)
                df = pd.DataFrame([latest_candle])

                if is_pin_bar(df.iloc[0]):
                    direction = "call" if df['close'].iloc[0] > df['open'].iloc[0] else "put"
                    print(f"[{time.strftime('%H:%M:%S')}] Pin bar detected! Attempting trade: {direction}")
                    trade_result = api.buy_digital_spot(pair, 1, direction, 1)  # $1, 1-min expiry
                    print(f"Trade result: {trade_result}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No pin bar detected.")

                last_candle_id = candle_id

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
    api.stop_candles_stream(pair, 60)