# strategies/simple_ma.py (Adapté pour utiliser la colonne 'Prix')
import pandas as pd
import numpy as np

def apply_strategy(df, short_window=40, long_window=100):
    """
    Applique une stratégie de croisement de moyennes mobiles simples au DataFrame.

    Args:
        df (pd.DataFrame): DataFrame des données historiques (doit contenir 'Prix' et avoir un DatetimeIndex).
        short_window (int): Période pour la moyenne mobile courte.
        long_window (int): Période pour la moyenne mobile longue.

    Returns:
        pd.DataFrame: Le DataFrame original avec les colonnes 'short_mavg',
                      'long_mavg' et 'signal'.
                      'signal' = 1 pour un signal d'achat, -1 pour un signal de vente, 0 sinon.
                      'positions' = 1 si en position, 0 si hors position (simplifié).
    """
    df_strat = df.copy()

    # Calcul des moyennes mobiles (utilise la colonne 'Prix' standardisée)
    df_strat['short_mavg'] = df_strat['Prix'].rolling(window=short_window, min_periods=1).mean()
    df_strat['long_mavg'] = df_strat['Prix'].rolling(window=long_window, min_periods=1).mean()

    # Génération des signaux de trading basés sur le croisement
    df_strat['signal'] = 0.0
    df_strat['signal'][short_window:] = np.where(df_strat['short_mavg'][short_window:] > df_strat['long_mavg'][short_window:], 1.0, 0.0)

    df_strat['positions'] = df_strat['signal'].diff()

    return df_strat
