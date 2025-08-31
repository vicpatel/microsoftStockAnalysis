# ==============================
# PHASE 2 — DATA UNDERSTANDING
# ==============================

import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print('Libraries loaded.')

# ==============================
# STEP 1 — Load & integrity checks
# ==============================
csv_path = Path('dataset/Microsoft_stock_data.csv')
df = pd.read_csv(csv_path)

# Parse dates and sort
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print('Shape (rows, cols):', df.shape)
print('Date range:', df['Date'].min(), '→', df['Date'].max())

# ==============================
# Missing & duplicates
# ==============================
print('\nMissing values by column:')
print(df.isna().sum())
dupe_dates = df['Date'].duplicated().sum()
print('Duplicate dates:', dupe_dates)

# ==============================
# Basic positivity checks
# ==============================
price_cols = ['Open','High','Low','Close']
non_positive_prices = (df[price_cols] <= 0).sum()
non_positive_volume = int((df['Volume'] <= 0).sum())
print('\nNon-positive counts:')
print(non_positive_prices)
print('Non-positive Volume:', non_positive_volume)

# ==============================
# Logical constraints for OHLC
# ==============================
viol_high = (df['High'] < df[['Open','Close','Low']].max(axis=1))
viol_low  = (df['Low']  > df[['Open','Close','High']].min(axis=1))
print('\nViolations:')
print('High < max(Open, Close, Low):', int(viol_high.sum()))
print('Low  > min(Open, Close, High):', int(viol_low.sum()))

print('\nPreview:')
display(pd.concat([df.head(3), df.tail(3)]))

# ==============================
# STEP 2 — Feature engineering
# ==============================
df['Daily_Change'] = df['Close'] - df['Open']
df['Daily_Return'] = df['Close'].pct_change()
df['Log_Return']   = np.log(df['Close']).diff()

# Rolling metrics (~21 trading days ≈ 1 month)
df['Volatility_21d'] = df['Daily_Return'].rolling(21).std() * np.sqrt(252)
df['AvgVol_21d']     = df['Volume'].rolling(21).mean()

print('\nEngineered columns added.')
display(df[['Date','Open','High','Low','Close','Volume','Daily_Change','Daily_Return','Volatility_21d','AvgVol_21d']].head(10))

# ==============================
# STEP 3 — Descriptive statistics & headline metrics
# ==============================
desc = df[['Open','High','Low','Close','Volume','Daily_Change','Daily_Return','Log_Return','Volatility_21d','AvgVol_21d']].describe(percentiles=[.25,.5,.75,.95]).T
print('\nDescriptive statistics:')
display(desc)

# Headline metrics
start_price = df.loc[0,'Close']
end_price = df.loc[df.index[-1],'Close']
years = (df.loc[df.index[-1],'Date'] - df.loc[0,'Date']).days / 365.25
cagr = (end_price / start_price) ** (1/years) - 1
mean_daily_ret = df['Daily_Return'].mean()
median_daily_ret = df['Daily_Return'].median()
ann_vol = df['Daily_Return'].std() * np.sqrt(252)

summary = pd.DataFrame({
    'Metric': ['Start Close','End Close','Span (years)','CAGR','Mean Daily Return','Median Daily Return','Annualized Volatility'],
    'Value': [round(start_price,4), round(end_price,4), round(years,2), round(cagr,4), round(mean_daily_ret,4), round(median_daily_ret,4), round(ann_vol,4)]
})
print('\nHeadline metrics:')
display(summary)

# ==============================
# STEP 4 — Correlations
# ==============================
corr_cols = ['Open','High','Low','Close','Volume','Daily_Change','Daily_Return']
corr_matrix = df[corr_cols].corr(method='pearson')
print('\nCorrelation matrix:')
display(corr_matrix.round(4))

# Volume vs next-day return
df['NextDay_Return'] = df['Daily_Return'].shift(-1)
pairs = {
    'High vs Close': df['High'].corr(df['Close']),
    'Low vs Close': df['Low'].corr(df['Close']),
    'Volume vs Daily_Return': df['Volume'].corr(df['Daily_Return']),
    'Volume vs NextDay_Return': df['Volume'].corr(df['NextDay_Return']),
    'Volume vs Daily_Change': df['Volume'].corr(df['Daily_Change']),
}
print('\nKey correlations:')
display(pd.DataFrame({'Pair': list(pairs.keys()), 'Pearson r': [round(v,4) for v in pairs.values()]}))

# ==============================
# STEP 5 — Calendar/seasonal profiles
# ==============================
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name().str[:3]
df['Quarter'] = df['Date'].dt.quarter
df['DOW'] = df['Date'].dt.day_name().str[:3]

monthly = df.groupby('Month', as_index=False).agg(
    Avg_Close   = ('Close','mean'),
    Avg_Return  = ('Daily_Return','mean'),
    Median_Return=('Daily_Return','median'),
    Volatility  = ('Daily_Return', lambda x: x.std()*np.sqrt(252))
)
monthly['Month_Name'] = monthly['Month'].map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
monthly = monthly[['Month','Month_Name','Avg_Close','Avg_Return','Median_Return','Volatility']]

print('\nMonthly profile:')
display(monthly.round(4))

dow = df.groupby('DOW', as_index=False).agg(
    Avg_Return=('Daily_Return','mean'),
    Median_Return=('Daily_Return','median'),
    Volatility=('Daily_Return', lambda x: x.std()*np.sqrt(252)),
    Avg_Volume=('Volume','mean')
).sort_values('DOW')

print('\nDay-of-Week profile:')
display(dow.round(4))

# ==============================
# STEP 6 — Extreme days
# ==============================
abs_ret_thr = df['Daily_Return'].abs().quantile(0.99)
vol_thr = df['Volume'].quantile(0.99)

extreme_moves = df.loc[df['Daily_Return'].abs() >= abs_ret_thr, ['Date','Open','Close','Daily_Return','Volume']].copy()
volume_spikes = df.loc[df['Volume'] >= vol_thr, ['Date','Open','Close','Daily_Return','Volume']].copy()

print(f'\nExtreme move threshold (|return|): {abs_ret_thr:.4f}')
print(f'Volume spike threshold: {vol_thr:.0f}')
print('\nExtreme daily moves (Top 1% by |Return|):')
display(extreme_moves.sort_values('Date').reset_index(drop=True))
print('\nVolume spikes (Top 1%):')
display(volume_spikes.sort_values('Date').reset_index(drop=True))

# ==============================
# STEP 7 — Export enriched dataset
# ==============================
export_path = Path('dataset/MSFT_enriched_phase2.csv')
df.to_csv(export_path, index=False)
print('Exported enriched dataset to:', export_path.resolve())
