import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Get data
ticker = ["PEP", "KO"]  # Coca-Cola (good for mean reversion)
ko = yf.Ticker(ticker)
df = ko.history(start="2015-01-01", end="2025-01-01")

# 2. Bollinger Bands
df["SMA20"] = df["Close"].rolling(20).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["Upper"] = df["SMA20"] + 2 * df["STD20"]
df["Lower"] = df["SMA20"] - 2 * df["STD20"]

# Drop NaN rows created by rolling window
df = df.dropna().copy()

# 3. Mean reversion signals
df["Signal"] = 0
df.loc[df["Close"] < df["Lower"], "Signal"] = 1   # Buy
df.loc[df["Close"] > df["Upper"], "Signal"] = -1  # Sell

# 4. Daily returns
df["Return"] = df["Close"].pct_change()
df["Strategy"] = df["Signal"].shift(1) * df["Return"]

# 5. Estimate win rate and payoff from backtest
wins = df.loc[df["Strategy"] > 0, "Strategy"]
losses = df.loc[df["Strategy"] < 0, "Strategy"]
p = len(wins) / (len(wins) + len(losses)) if len(wins)+len(losses) > 0 else 0.5
b = abs(wins.mean() / losses.mean()) if len(losses) > 0 else 1

# Kelly fraction
f_kelly = (p * (b + 1) - 1) / b
f_half_kelly = f_kelly / 2
print(f"Win rate={p:.2f}, payoff ratio={b:.2f}, Kelly fraction={f_kelly:.2f}")

# 6. Apply fixed sizing vs Kelly sizing
df["Strategy_fixed"] = df["Strategy"]  # equal weight = 1x exposure
df["Strategy_kelly"] = df["Signal"].shift(1) * df["Return"] * f_half_kelly  # half-Kelly

# 7. Cumulative returns
df["Cumulative_fixed"] = (1 + df["Strategy_fixed"]).cumprod()
df["Cumulative_kelly"] = (1 + df["Strategy_kelly"]).cumprod()
df["Cumulative_buyhold"] = (1 + df["Return"]).cumprod()

# 8. Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Cumulative_buyhold"], label="Buy & Hold", color="blue")
plt.plot(df.index, df["Cumulative_fixed"], label="Mean Reversion (Fixed Size)", color="red")
plt.plot(df.index, df["Cumulative_kelly"], label="Mean Reversion (Half Kelly)", color="green")
plt.title(f"Mean Reversion + Kelly Sizing on {ticker}")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
