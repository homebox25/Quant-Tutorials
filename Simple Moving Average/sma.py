import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
symbol = "AAPL"          # Stock ticker
short_window = 20        # Short SMA
long_window = 50         # Long SMA

# --- Download Data (last 1 year up to today) ---
data = yf.download(symbol, period="1y")
data["SMA20"] = data["Close"].rolling(window=short_window).mean()
data["SMA50"] = data["Close"].rolling(window=long_window).mean()

# --- Generate Trading Signals ---
data["Signal"] = 0
data.loc[data["SMA20"] > data["SMA50"], "Signal"] = 1
data["Position"] = data["Signal"].diff()  # +1 = Buy, -1 = Sell

# --- Plot Price + SMAs + Buy/Sell Markers ---
plt.figure(figsize=(12,6))
plt.plot(data["Close"], label=f"{symbol} Price", alpha=0.7)
plt.plot(data["SMA20"], label="20-day SMA", linestyle="--")
plt.plot(data["SMA50"], label="50-day SMA", linestyle="--")

# Buy markers
plt.plot(
    data[data["Position"] == 1].index,
    data[data["Position"] == 1]["Close"],
    "^", markersize=10, color="g", label="Buy Signal"
)

# Sell markers
plt.plot(
    data[data["Position"] == -1].index,
    data[data["Position"] == -1]["Close"],
    "v", markersize=10, color="r", label="Sell Signal"
)

plt.title(f"{symbol} Price with SMA Crossovers (20 vs 50)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# --- Strategy Returns ---
data["Daily Returns"] = data["Close"].pct_change()
data["Strategy Returns"] = data["Signal"].shift(1) * data["Daily Returns"]

# --- Plot Strategy vs Buy & Hold ---
(1 + data[["Daily Returns","Strategy Returns"]]).cumprod().plot(figsize=(12,6))
plt.title("Cumulative Returns: Buy & Hold vs SMA Strategy")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend(["Buy & Hold", "SMA Strategy"])
plt.grid(True)
plt.show()
