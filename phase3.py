# ==============================
# PHASE 3 — DATA VISUALIZATION
# ==============================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['axes.grid'] = True
print('Libraries ready')

# ==============================
# STEP 1 — Load enriched dataset from Phase 2
# ==============================
csv_path = Path('dataset/MSFT_enriched_phase2.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print('Rows, Cols:', df.shape)
print('Date range:', df['Date'].min(), '→', df['Date'].max())
display(df.head(3))

# =========================
# 1) Long-term price trend
# =========================
plt.plot(df['Date'], df['Close'])
plt.title('MSFT Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close')
plt.tight_layout()
plt.show()

# Interpretation:
# Clear long-term upward trend with some dips.
# Supports the hypothesis of a significant upward trend over time.

# =========================
# 2) Volume over time
# =========================
plt.plot(df['Date'], df['Volume'])
plt.title('MSFT Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.tight_layout()
plt.show()

# Interpretation:
# Volume varies with spikes during major events.
# Useful context for checking volume-price relationships.

# =========================================
# 3) Volume vs Daily Price Change (scatter)
# =========================================
plt.scatter(df['Volume'], df['Daily_Change'], s=5, alpha=0.5)
plt.title('Volume vs Daily Price Change (Close − Open)')
plt.xlabel('Volume')
plt.ylabel('Daily Change')
plt.tight_layout()
plt.show()

corr = df['Volume'].corr(df['Daily_Change'])
print('Correlation (Volume, Daily_Change):', round(corr, 4))

# Interpretation:
# Cloud is wide/flat — low correlation, meaning volume does not strongly predict same-day change.

# ===================================
# 4) Seasonality — Boxplots by Month & Day-of-Week
# ===================================
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name().str[:3]
order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=order, ordered=True)

# Boxplot by month
data = [df.loc[df['Month_Name']==m, 'Daily_Return'].dropna().values for m in order]
plt.boxplot(data, tick_labels=order, showfliers=False)
plt.title('Daily Returns by Month (boxplot)')
plt.xlabel('Month')
plt.ylabel('Daily Return')
plt.tight_layout()
plt.show()

# Boxplot by day-of-week
df['DOW'] = df['Date'].dt.day_name().str[:3]
dow_order = ['Mon','Tue','Wed','Thu','Fri']
df['DOW'] = pd.Categorical(df['DOW'], categories=dow_order, ordered=True)
data2 = [df.loc[df['DOW']==d, 'Daily_Return'].dropna().values for d in dow_order]
plt.boxplot(data2, tick_labels=dow_order, showfliers=False)
plt.title('Daily Returns by Day of Week (boxplot)')
plt.xlabel('Day of Week')
plt.ylabel('Daily Return')
plt.tight_layout()
plt.show()

# Interpretation:
# Boxes higher above zero = better average performance.
# Wider boxes = more risk/variation.

# =========================
# 5) Rolling Volatility
# =========================
plt.plot(df['Date'], df['Volatility_21d'])
plt.title('Rolling Annualized Volatility (21d window)')
plt.xlabel('Date')
plt.ylabel('Volatility (annualized)')
plt.tight_layout()
plt.show()

# Interpretation:
# Peaks mark high-stress market periods (e.g., crises, big news).

# ===================================
# 6) Regression — High & Low -> Close
# ===================================
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()

# Prepare data
X_train = np.column_stack([train['High'].values, train['Low'].values])
y_train = train['Close'].values
Xb = np.column_stack([np.ones(len(X_train)), X_train])

# Solve for coefficients: beta = (X'X)^(-1) X'y
beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y_train)
beta0, beta_high, beta_low = beta
print('Model: Close = b0 + b1*High + b2*Low')
print('b0, b1, b2 =', beta0, beta_high, beta_low)

def predict_close(high, low):
    return beta0 + beta_high*high + beta_low*low

# Evaluate on test
y_pred = predict_close(test['High'].values, test['Low'].values)
y_true = test['Close'].values
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - y_true.mean())**2)
r2 = 1 - ss_res/ss_tot
print('Test R^2:', round(r2, 6))

# Plot Actual vs Predicted Close (test segment)
plt.plot(test['Date'], y_true, label='Actual')
plt.plot(test['Date'], y_pred, label='Predicted')
plt.title('Actual vs Predicted Close (Test Period)')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.tight_layout()
plt.show()

# Interpretation:
# High R² — Close is highly predictable from High & Low.
# Supports the hypothesis that these features explain Close well.

# ==========================================
# Summary bullets for your report:
# ==========================================
print("""
- Trend: Closing price is strongly upward over decades.
- Volume: Activity spikes during major events; weak same-day link to price changes.
- Seasonality: Jan/Oct stronger on average; weekday differences visible.
- Volatility: Rolling volatility highlights market stress periods.
- Regression: High & Low explain most variance in Close (high R²).
""")
