import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Download data
tickers = ["KO", "PEP"]
ko = yf.Ticker("KO")
pep = yf.Ticker("PEP")
ko_data = ko.history(start="2018-01-01", end="2025-01-01")
pep = pep.history(start="2018-01-01", end="2025-01-01")
data = yf.download(tickers, start="2018-01-01", end="2025-01-01")["Adj Close"]

# 2. Regression hedge ratio (beta)
X = sm.add_constant(pep_data["Adj Close"])
model = sm.OLS(ko_data["Adj Close"], X).fit()
beta = model.params["PEP"]

# 3. Spread
data["Spread"] = data["KO"] - beta * data["PEP"]
data["Zscore"] = (data["Spread"] - data["Spread"].mean()) / data["Spread"].std()

# 4. Signals
data["Signal"] = 0
data.loc[data["Zscore"] > 2, "Signal"] = -1  # short spread
data.loc[data["Zscore"] < -2, "Signal"] = 1  # long spread

# 5. Returns of spread
data["SpreadReturn"] = data["Spread"].diff() * data["Signal"].shift()

# 6. Kelly fraction
win_rate = (data["SpreadReturn"] > 0).mean()
payoff_ratio = data.loc[data["SpreadReturn"] > 0, "SpreadReturn"].mean() / abs(
    data.loc[data["SpreadReturn"] < 0, "SpreadReturn"].mean()
)
kelly_f = win_rate - (1 - win_rate) / payoff_ratio
kelly_f = max(min(kelly_f, 1), 0)  # cap between 0 and 1

# 7. Apply Kelly sizing
data["KellyReturn"] = data["SpreadReturn"] * kelly_f

# 8. Equity curve
data[["SpreadReturn", "KellyReturn"]].cumsum().plot()
plt.title("Pairs Trading KO vs PEP (Mean Reversion + Kelly)")
plt.show()
