import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch data for EURGBP
ticker = 'EURGBP=X'
start_date = '2022-01-01'
end_date = '2024-08-30'
df = yf.download(ticker, start=start_date, end=end_date)

# Calculate mean and standard deviation (20-day rolling window)
window = 20
df['Mean'] = df['Close'].rolling(window=window).mean()
df['Std'] = df['Close'].rolling(window=window).std()

# Calculate Z-score
df['Z-score'] = (df['Close'] - df['Mean']) / df['Std']

# Define entry and exit thresholds
entry_threshold = -1.5  # Buy when Z-score < -1.5
exit_threshold = 1.5    # Sell when Z-score > 1.5

# Generate trading signals based on Z-score
df['Position'] = 0  # Default position is 0 (no position)
df.loc[df['Z-score'] < entry_threshold, 'Position'] = 1   # Long position (Buy)
df.loc[df['Z-score'] > exit_threshold, 'Position'] = -1   # Short position (Sell)
df['Position'] = df['Position'].shift()  # Shift position by one day

# Backtest - calculate daily returns
df['Strategy_Returns'] = df['Position'].shift(1) * df['Close'].pct_change()

# Calculate cumulative returns for strategy and buy-and-hold benchmark
df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod() - 1
df['Cumulative_Buy_and_Hold_Returns'] = (1 + df['Close'].pct_change()).cumprod() - 1

# Plot Z-score and trading signals
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df.index, df['Z-score'], label='Z-score')
plt.axhline(entry_threshold, color='green', linestyle='--', label='Buy Threshold (-1.5)')
plt.axhline(exit_threshold, color='red', linestyle='--', label='Sell Threshold (1.5)')
plt.title('Z-score and Trading Signals')
plt.legend()

# Plot cumulative returns for strategy and buy-and-hold
plt.subplot(2, 1, 2)
plt.plot(df.index, df['Cumulative_Strategy_Returns'], label='Strategy Returns', color='blue')
plt.plot(df.index, df['Cumulative_Buy_and_Hold_Returns'], label='Buy and Hold Returns', color='orange')
plt.title('Cumulative Returns: Strategy vs. Buy and Hold')
plt.legend()

plt.tight_layout()
plt.show()
