import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Fetch Data ---
tickers = ["KO", "PEP"]
kopep = yf.Ticker(tickers)
data = kopep.history(period="2y")["Close"]

# Normalize prices so we can compare (rebased to 1 at start)
normed = data / data.iloc[0]

# Spread = difference between normalized KO and PEP
spread = normed["KO"] - normed["PEP"]

# Calculate z-score of spread (standardized distance from mean)
zscore = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

# --- Generate Trading Signals ---
# Buy KO, Sell PEP if spread is too low (z < -1)
# Sell KO, Buy PEP if spread is too high (z > 1)
signals = pd.DataFrame(index=spread.index)
signals["zscore"] = zscore
signals["long_KO"] = (zscore < -1).astype(int)
signals["short_KO"] = (zscore > 1).astype(int)

# --- Plot Spread + Signals ---
plt.figure(figsize=(12,6))
plt.plot(spread.index, spread, label="KO - PEP Spread", color="blue")
plt.axhline(spread.mean(), color="black", linestyle="--", label="Mean")
plt.scatter(signals.index[signals["long_KO"]==1], 
            spread[signals["long_KO"]==1], 
            marker="^", color="green", s=100, label="Buy KO / Sell PEP")
plt.scatter(signals.index[signals["short_KO"]==1], 
            spread[signals["short_KO"]==1], 
            marker="v", color="red", s=100, label="Sell KO / Buy PEP")
plt.title("Pairs Trading: Coca-Cola vs Pepsi (Spread Mean Reversion)")
plt.legend()
plt.show()
