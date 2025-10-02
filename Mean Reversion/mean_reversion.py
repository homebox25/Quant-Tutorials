import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Get historical prices (Coke vs Pepsi example)
tickers = ["KO", "PEP"]
data = yf.download(tickers, start="2022-01-01", end="2025-01-01")
data.dropna(inplace=True)

# 2. Compute spread via linear regression (cointegration style)
X = sm.add_constant(data["PEP"])
model = sm.OLS(data["KO"], X).fit()
hedge_ratio = model.params["PEP"]
spread = data["KO"] - hedge_ratio * data["PEP"]

# 3. Estimate OU parameters (discretized approximation)
spread_diff = spread.diff().dropna()
spread_lag = spread.shift(1).dropna()
beta = np.polyfit(spread_lag, spread_diff, 1)  # regression slope/intercept

theta = -beta[0]
mu = -beta[1] / beta[0]
sigma = np.std(spread_diff - beta[0] * spread_lag - beta[1])

print(f"Estimated OU params:\nmu={mu:.4f}, theta={theta:.4f}, sigma={sigma:.4f}")

# 4. Trading signals (z-score)
z_score = (spread - mu) / sigma

entry_threshold = 2
exit_threshold = 0

longs = z_score < -entry_threshold
shorts = z_score > entry_threshold
exits = abs(z_score) < exit_threshold

# Strategy positions: 1 = long spread, -1 = short spread
position = np.zeros(len(spread))
position[longs] = 1
position[shorts] = -1
position[exits] = 0
position = pd.Series(position, index=spread.index).ffill().fillna(0)

# 5. Strategy PnL (simplified, not accounting for costs)
returns = spread.diff()
strategy_returns = position.shift(1) * returns
equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

# --- Plot results ---
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(spread, label="Spread")
plt.axhline(mu, color='red', linestyle='--', label="Mean")
plt.legend()
plt.title("Coke-Pepsi Spread with OU Mean Reversion")

plt.subplot(2,1,2)
plt.plot(equity_curve, label="Equity Curve", color="green")
plt.legend()
plt.title("Mean Reversion Strategy Backtest")
plt.tight_layout()
plt.show()
