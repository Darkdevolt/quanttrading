# strategies/simple_ma.py
import pandas as pd
import numpy as np

def apply_strategy(df, short_window=40, long_window=100):
    df_strat = df.copy()
    
    df_strat['short_mavg'] = df_strat['Close'].rolling(window=short_window, min_periods=1).mean()
    df_strat['long_mavg'] = df_strat['Close'].rolling(window=long_window, min_periods=1).mean()
    
    df_strat['signal'] = 0.0
    df_strat['signal'][short_window:] = np.where(
        df_strat['short_mavg'][short_window:] > df_strat['long_mavg'][short_window:], 1.0, 0.0
    )
    
    df_strat['positions'] = df_strat['signal'].diff()
    return df_strat
