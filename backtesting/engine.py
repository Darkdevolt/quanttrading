# backtesting/engine.py
import pandas as pd

def run_backtest(df_with_signals, initial_capital=100000):
    if df_with_signals is None or df_with_signals.empty:
        return None
    if 'Close' not in df_with_signals.columns or 'positions' not in df_with_signals.columns:
        return None

    portfolio = pd.DataFrame(index=df_with_signals.index)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = float(initial_capital)
    portfolio['total'] = float(initial_capital)

    shares = 0.0
    cash = float(initial_capital)

    for row in df_with_signals.itertuples():
        date = row.Index
        close_price = row.Close
        position_change = row.positions

        if position_change == 1.0 and cash > 0:
            shares = cash / close_price
            cash = 0.0
        elif position_change == -1.0 and shares > 0:
            cash += shares * close_price
            shares = 0.0

        portfolio.loc[date, 'total'] = cash + shares * close_price

    return portfolio['total']
