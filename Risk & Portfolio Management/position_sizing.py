import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
symbol = "SPY"
df = yf.download(symbol, start="2015-01-01", end="2025-01-01")

# 2. Signals (SMA crossover)
df["SMA50"] = df["Close"].rolling(50).mean()
df["SMA200"] = df["Close"].rolling(200).mean()
df["Signal"] = np.where(df["SMA50"] > df["SMA200"], 1, -1)

# 3. Daily returns
df["Return"] = df["Close"].pct_change()

# 4a. Equal-weight strategy (1x exposure)
df["Strategy_eq"] = df["Signal"].shift(1) * df["Return"]

# 4b. Volatility-adjusted strategy
# Scale exposure so more size in low-volatility periods, less in high-volatility
df["Volatility"] = df["Return"].rolling(20).std()
df["Vol_adj"] = (1 / df["Volatility"]) / (1 / df["Volatility"]).mean()  # normalize
df["Strategy_vol"] = df["Signal"].shift(1) * df["Return"] * df["Vol_adj"]

# 5. Cumulative returns
df["Cumulative_eq"] = (1 + df["Strategy_eq"]).cumprod()
df["Cumulative_vol"] = (1 + df["Strategy_vol"]).cumprod()
df["Cumulative_buyhold"] = (1 + df["Return"]).cumprod()

# 6. Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Cumulative_buyhold"], label="Buy & Hold", color="blue")
plt.plot(df.index, df["Cumulative_eq"], label="SMA Strategy (Equal Size)", color="red")
plt.plot(df.index, df["Cumulative_vol"], label="SMA Strategy (Volatility-Adjusted)", color="green")
plt.title(f"Risk Management & Position Sizing on {symbol}")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
