import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
np.random.seed(42)
n_flips = 200  # number of trades
n_sims = 10    # number of simulation runs to compare
p_win = 0.55   # probability of winning
b = 1          # 1:1 payoff (win = +100%, loss = -100%)

# Kelly fraction
f_kelly = (p_win * (b + 1) - 1) / b
f_half_kelly = f_kelly / 2
f_fixed = 0.05  # always risk 5%

def run_strategy(fraction):
    equity = [1]  # start with $1
    for _ in range(n_flips):
        bet = equity[-1] * fraction
        if np.random.rand() < p_win:
            equity.append(equity[-1] + bet)   # win
        else:
            equity.append(equity[-1] - bet)   # loss
    return equity

# Run multiple simulations
plt.figure(figsize=(12,6))
for i in range(n_sims):
    plt.plot(run_strategy(f_kelly), color="red", alpha=0.6)
    plt.plot(run_strategy(f_half_kelly), color="green", alpha=0.6)
    plt.plot(run_strategy(f_fixed), color="blue", alpha=0.6)

plt.title("Kelly vs Half-Kelly vs Fixed Fraction (p=55%, payoff=1:1)")
plt.xlabel("Number of Trades")
plt.ylabel("Equity ($)")
plt.legend(["Kelly", "Half Kelly", "Fixed Fraction"], loc="upper left")
plt.show()
