# ==============================
# PHASE 4 & 5 — MODELING + EVALUATION
# ==============================

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox_fn
    HAS_RETURN_DF = True
except Exception:
    ljungbox_fn = None
    HAS_RETURN_DF = False
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['axes.grid'] = True

# ==============================
# STEP 0 — Utility helpers
# ==============================

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def parse_dates(df):
    candidate = next((c for c in df.columns if c.lower() in {'date','timestamp'}), None)
    if candidate:
        df[candidate] = pd.to_datetime(df[candidate], errors='coerce')
        df = df.dropna(subset=[candidate]).sort_values(candidate).set_index(candidate)
    else:
        df.index = pd.to_datetime(df.index, errors='raise')
        df = df.sort_index()
    return df

def add_features(df):
    require_columns(df, {'Open','High','Low','Close','Volume'})
    df['ret'] = df['Close'].pct_change()
    df['log_ret'] = np.log1p(df['ret'])
    df['vol_21'] = df['ret'].rolling(21).std() * np.sqrt(252)
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_21'] = df['Close'].rolling(21).mean()
    for k in (1,2,3,5,10,21):
        df[f'lag_close_{k}'] = df['Close'].shift(k)
        df[f'lag_ret_{k}'] = df['ret'].shift(k)
    df['month'] = df.index.month
    df['dow'] = df.index.dayofweek
    df['y_next_close'] = df['Close'].shift(-1)
    return df.dropna()

def split_by_date(df, cutoff):
    t = pd.Timestamp(cutoff)
    tr = df.loc[df.index < t].copy()
    te = df.loc[df.index >= t].copy()
    if tr.empty or te.empty:
        raise ValueError('Bad cutoff: one side is empty.')
    return tr, te

def eval_metrics(y, p):
    mae = mean_absolute_error(y, p)
    # Backward-compatible RMSE (avoid squared= argument)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    # MAPE safe (prices > 0, but keep cast)
    mape = float(np.mean(np.abs((y - p) / y)) * 100)
    r2 = r2_score(y, p)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape, 'R2': r2}

# ==============================
# STEP 1 — Train models
# ==============================

def train_lr_gbr(train, test, features):
    Xtr, ytr = train[features], train['y_next_close']
    Xte, yte = test[features], test['y_next_close']
    pre = ColumnTransformer([('num', StandardScaler(), features)], remainder='drop')
    lr = Pipeline([('pre', pre), ('est', LinearRegression())])
    lr.fit(Xtr, ytr)
    pred_lr = lr.predict(Xte)
    gbr = Pipeline([('pre', pre), ('est', GradientBoostingRegressor(random_state=42))])
    gbr.fit(Xtr, ytr)
    pred_gbr = gbr.predict(Xte)
    return pred_lr, pred_gbr, lr, gbr

def cross_val_timeseries(df, features, n_splits=5):
    X = df[features]; y = df['y_next_close']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pre = ColumnTransformer([('num', StandardScaler(), features)], remainder='drop')
    lr = Pipeline([('pre', pre), ('est', LinearRegression())])
    gbr = Pipeline([('pre', pre), ('est', GradientBoostingRegressor(random_state=42))])
    rows = []
    for model_name, model in [('LinearRegression', lr), ('GradientBoosting', gbr)]:
        fold = 1
        for tr_idx, va_idx in tscv.split(X):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            model.fit(Xtr, ytr)
            p = model.predict(Xva)
            rows.append({'Model': model_name, 'Fold': fold, **eval_metrics(yva, p)})
            fold += 1
    return pd.DataFrame(rows)

def train_sarimax_with_residuals(train_close, test_index):
    candidates = [(1,1,0),(0,1,1),(1,1,1),(2,1,1),(1,1,2)]
    best = {'aic': np.inf, 'order': None, 'res': None}
    for od in candidates:
        try:
            res = SARIMAX(train_close, order=od,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if res.aic < best['aic']:
                best = {'aic': res.aic, 'order': od, 'res': res}
        except Exception:
            pass
    if best['res'] is None:
        res = SARIMAX(train_close, order=(1,1,1),
                      enforce_stationarity=False,
                      enforce_invertibility=False).fit(disp=False)
        best = {'aic': res.aic, 'order': (1,1,1), 'res': res}

    fc = best['res'].forecast(steps=len(test_index))
    fc.index = test_index
    resid = best['res'].resid  # in-sample residuals
    return fc, best['order'], resid

# ==============================
# STEP 2 — Orchestrate pipeline
# ==============================

def run(data_path, out_dir, cutoff):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load → prepare → split
    df = pd.read_csv(data_path)
    df = parse_dates(df)
    df = add_features(df)
    train, test = split_by_date(df, cutoff)

    # Feature set
    feats = [
        'Open','High','Low','Volume',
        'ret','log_ret','vol_21','sma_5','sma_21',
        'lag_close_1','lag_close_2','lag_close_3','lag_close_5','lag_close_10','lag_close_21',
        'lag_ret_1','lag_ret_2','lag_ret_3','lag_ret_5','lag_ret_10','lag_ret_21',
        'month','dow'
    ]

    # CV on training
    cv_df = cross_val_timeseries(train, feats, n_splits=5)
    cv_summary = (cv_df.groupby('Model')[['MAE','RMSE','MAPE_%','R2']]
                      .agg(['mean','std']).reset_index())
    cv_summary.to_csv(out/'cv_metrics.csv', index=False)

    # Train final models on train, evaluate on test
    pred_lr, pred_gbr, lr_model, gbr_model = train_lr_gbr(train, test, feats)
    fc_smx, order_smx, resid_smx = train_sarimax_with_residuals(train['Close'], test.index)

    # Evaluate on test
    y = test['y_next_close']
    metrics_rows = [
        {'Model':'LinearRegression', **eval_metrics(y, pred_lr)},
        {'Model':'GradientBoosting', **eval_metrics(y, pred_gbr)},
        {'Model':'SARIMAX',         **eval_metrics(y, fc_smx.values)},
    ]
    metrics_df = pd.DataFrame(metrics_rows).set_index('Model')
    metrics_df.to_csv(out/'model_metrics.csv')

    # OLS summary on compact subset
    ols_feats = ['High','Low','Volume','lag_close_1','sma_5','sma_21','vol_21']
    Xtr = sm.add_constant(train[ols_feats])
    with open(out/'ols_summary.txt','w') as f:
        f.write(sm.OLS(train['y_next_close'], Xtr).fit().summary().as_text())

    # Seasonality tests (ANOVA)
    aov_month = smf.ols('ret ~ C(month)', data=df.dropna(subset=['ret'])).fit()
    sm.stats.anova_lm(aov_month, typ=2).to_csv(out/'anova_month_returns.csv')
    aov_wday = smf.ols('ret ~ C(dow)', data=df.dropna(subset=['ret'])).fit()
    sm.stats.anova_lm(aov_wday, typ=2).to_csv(out/'anova_weekday_returns.csv')

    # Persist predictions
    pred_frame = pd.DataFrame({
        'Date': test.index,
        'actual_next_close': y.values,
        'pred_lr': pred_lr,
        'pred_gbr': pred_gbr,
        'pred_sarimax': fc_smx.values
    })
    pred_frame.to_csv(out/'predictions.csv', index=False)

    # ==============================
    # STEP 3 — Figures
    # ==============================
    def save_png(actual, pred, title, fname):
        plt.figure()
        plt.plot(actual.index, actual.values, label='Actual')
        plt.plot(actual.index, pred, label='Predicted')
        plt.title(title); plt.xlabel('Date'); plt.ylabel('Price')
        plt.legend(); plt.tight_layout()
        plt.savefig(out/fname); plt.close()

    save_png(y, pred_lr,  'Pred vs Actual — Linear Regression', 'fig_pred_vs_actual_lr.png')
    save_png(y, pred_gbr, 'Pred vs Actual — Gradient Boosting', 'fig_pred_vs_actual_gbr.png')
    save_png(y, fc_smx.values, 'Pred vs Actual — SARIMAX', 'fig_pred_vs_actual_sarimax.png')

    # SARIMAX residual diagnostics
    fig = plt.figure(figsize=(10,4))
    plt.plot(resid_smx)
    plt.title('SARIMAX Residuals'); plt.xlabel('Date'); plt.ylabel('Residual')
    plt.tight_layout(); plt.savefig(out/'sarimax_residuals.png'); plt.close(fig)

    try:
        fig = plt.figure(figsize=(10,4))
        plot_acf(pd.Series(resid_smx).dropna(), lags=40)
        plt.tight_layout(); plt.savefig(out/'sarimax_residuals_acf.png'); plt.close(fig)
    except Exception:
        pass
    try:
        fig = plt.figure(figsize=(10,4))
        # No method arg for older statsmodels
        plot_pacf(pd.Series(resid_smx).dropna(), lags=40)
        plt.tight_layout(); plt.savefig(out/'sarimax_residuals_pacf.png'); plt.close(fig)
    except Exception:
        pass

    # Ljung-Box
    try:
        if ljungbox_fn is not None:
            lb = ljungbox_fn(pd.Series(resid_smx).dropna(), lags=[10,20,30], return_df=True)
            lb.to_csv(out/'sarimax_ljungbox.csv')
    except Exception:
        # Fallback if older API: skip
        pass

    # GBR feature importance
    try:
        gbr_est = gbr_model.named_steps['est']
        importances = gbr_est.feature_importances_
        feat_names = feats
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        imp_df.to_csv(out/'gbr_feature_importance.csv', index=False)

        plt.figure(figsize=(10,6))
        top = imp_df.head(20).iloc[::-1]
        plt.barh(top['feature'], top['importance'])
        plt.title('Gradient Boosting — Top 20 Feature Importances')
        plt.tight_layout(); plt.savefig(out/'gbr_feature_importance.png'); plt.close()
    except Exception:
        pass

    # PDF bundle
    with PdfPages(out/'figures_part2.pdf') as pdf:
        for title, series in [
            ('Pred vs Actual — Linear Regression', pred_lr),
            ('Pred vs Actual — Gradient Boosting', pred_gbr),
            ('Pred vs Actual — SARIMAX', fc_smx.values)
        ]:
            fig, ax = plt.subplots()
            ax.plot(y.index, y.values, label='Actual')
            ax.plot(y.index, series, label='Predicted')
            ax.set_title(title); ax.set_xlabel('Date'); ax.set_ylabel('Price'); ax.legend()
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for fname in ['sarimax_residuals.png','sarimax_residuals_acf.png','sarimax_residuals_pacf.png','gbr_feature_importance.png']:
            try:
                img = plt.imread(out/fname)
                fig, ax = plt.subplots()
                ax.imshow(img); ax.axis('off'); pdf.savefig(fig); plt.close(fig)
            except Exception:
                continue

    # ==============================
    # STEP 4 — Run summary
    # ==============================
    best_model = metrics_df['RMSE'].idxmin()
    summary = {
        'cutoff': str(cutoff),
        'best_model_by_RMSE': best_model,
        'sarimax_order': tuple(order_smx),
        'metrics': metrics_df.reset_index().to_dict(orient='records')
    }
    with open(out/'run_summary.json','w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(summary['metrics']).to_csv(out/'run_summary.csv', index=False)

    return order_smx

# ==============================
# STEP 5 — Entrypoint
# ==============================
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',   type=str, default='./dataset/MSFT_enriched_phase2.csv')
    ap.add_argument('--out',    type=str, default='./out_phase4_5')
    ap.add_argument('--cutoff', type=str, default='2017-01-01')
    args = ap.parse_args()

    print(f'Using data: {args.data}')
    print(f'Writing outputs to: {args.out}')
    order = run(args.data, args.out, args.cutoff)
    print('SARIMAX order:', order)
