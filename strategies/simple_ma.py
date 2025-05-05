# strategies/simple_ma.py
import pandas as pd
import numpy as np # S'assurer que numpy est importé si utilisé

def apply_strategy(df, short_window=40, long_window=100):
    """
    Applique une stratégie de croisement de moyennes mobiles simples au DataFrame.

    Args:
        df (pd.DataFrame): DataFrame des données historiques (doit contenir 'Close' et avoir un DatetimeIndex).
        short_window (int): Période pour la moyenne mobile courte.
        long_window (int): Période pour la moyenne mobile longue.

    Returns:
        pd.DataFrame: Le DataFrame original avec les colonnes 'short_mavg',
                      'long_mavg' et 'signal'.
                      'signal' = 1 pour un signal d'achat, -1 pour un signal de vente, 0 sinon.
                      'positions' = 1 si en position, 0 si hors position (simplifié).
    """
    df_strat = df.copy()

    # Calcul des moyennes mobiles (utilise la colonne 'Close')
    df_strat['short_mavg'] = df_strat['Close'].rolling(window=short_window, min_periods=1).mean()
    df_strat['long_mavg'] = df_strat['Close'].rolling(window=long_window, min_periods=1).mean()

    # Génération des signaux de trading basés sur le croisement
    # Créer une colonne 'signal' qui indique quand un croisement potentiel se produit
    df_strat['signal'] = 0.0

    # Quand la MA courte passe au-dessus de la MA longue (signal d'achat)
    df_strat['signal'][short_window:] = np.where(df_strat['short_mavg'][short_window:] > df_strat['long_mavg'][short_window:], 1.0, 0.0)

    # Prendre la différence pour identifier les moments précis des croisements
    # Un changement de signal de 0 à 1 indique un achat (+1 dans 'positions')
    # Un changement de signal de 1 à 0 indique une vente (-1 dans 'positions')
    df_strat['positions'] = df_strat['signal'].diff()

    # Les premières lignes auront des NaN à cause des moyennes mobiles et du diff, on peut les laisser
    # ou les gérer spécifiquement si nécessaire pour le backtest engine.

    return df_strat
