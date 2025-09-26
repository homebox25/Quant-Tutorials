import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# --- Parameters ---
symbol = "AAPL"

# --- Download Data (last 1 year up to today) ---
aapl = yf.Ticker(symbol)
data = aapl.history(period="1y")  # Adjust the period as needed

# --- Calculate the 20-period Simple Moving Average (SMA) ---
data['SMA'] = data['Close'].rolling(window=20).mean()

# --- Calculate the 20-period Standard Deviation (SD) ---
data['SD'] = data['Close'].rolling(window=20).std()

# --- Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB) ---
data['UB'] = data['SMA'] + 2 * data['SD']
data['LB'] = data['SMA'] - 2 * data['SD']

# --- Plot Bollinger Bands ---
plt.figure(figsize=(12,6))
plt.title("Bollinger Bands for AAPL")
plt.plot(data["UB"], label="Upper Band", linestyle="--")
plt.plot(data["LB"], label="Lower Band", linestyle="--")
plt.show()
