import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
symbol = "SPY"
period = "2y"
window = 20
num_std = 2
initial_cash = 10000

# --- Fetch Data ---
spy = yf.Ticker(symbol)
data = spy.history(period=period)
data["SMA"] = data["Close"].rolling(window=window).mean()
data["STD"] = data["Close"].rolling(window=window).std()
data["Upper"] = data["SMA"] + num_std * data["STD"]
data["Lower"] = data["SMA"] - num_std * data["STD"]

# --- Live Strategy Simulation ---
cash = initial_cash
position = 0
buy_dates, buy_prices = [], []
sell_dates, sell_prices = [], []
portfolio_values = []

for i in range(window, len(data)):
    price = data["Close"].iloc[i]
    upper = data["Upper"].iloc[i]
    lower = data["Lower"].iloc[i]
    date = data.index[i]

    # Generate signals
    if price < lower and cash > 0:  # Buy signal
        position = cash / price
        cash = 0
        buy_dates.append(date)
        buy_prices.append(price)
    elif price > upper and position > 0:  # Sell signal
        cash = position * price
        position = 0
        sell_dates.append(date)
        sell_prices.append(price)

    # Track portfolio value as if "live"
    portfolio_values.append(cash + position * price)

# Align portfolio values with dates
data = data.iloc[window:].copy()
data["Strategy_Value"] = portfolio_values
data["BuyHold_Value"] = initial_cash * (data["Close"] / data["Close"].iloc[0])

# --- Final Value ---
final_value = data["Strategy_Value"].iloc[-1]
buyhold_value = data["BuyHold_Value"].iloc[-1]
print(f"Final Strategy Value: ${final_value:.2f}")
print(f"Final Buy & Hold Value: ${buyhold_value:.2f}")

# --- Plot Price with Bollinger Bands & Trades ---
plt.figure(figsize=(12,6))
plt.plot(data.index, data["Close"], label="Close Price", color="blue", alpha=0.6)
plt.plot(data.index, data["SMA"], label="20-day SMA", color="orange")
plt.plot(data.index, data["Upper"], label="Upper Band", color="green", linestyle="--")
plt.plot(data.index, data["Lower"], label="Lower Band", color="red", linestyle="--")
plt.scatter(buy_dates, buy_prices, marker="^", color="green", s=100, label="Buy")
plt.scatter(sell_dates, sell_prices, marker="v", color="red", s=100, label="Sell")
plt.title("Mean Reversion Strategy (Bollinger Bands) VXX")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# --- Plot Cumulative Returns ---
plt.figure(figsize=(12,6))
plt.plot(data.index, data["Strategy_Value"], label="Strategy Portfolio", color="purple")
plt.plot(data.index, data["BuyHold_Value"], label="Buy & Hold", color="black", linestyle="--")
plt.title("Mean Reversion Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($) VXX")
plt.legend()
plt.grid(True)
plt.show()
