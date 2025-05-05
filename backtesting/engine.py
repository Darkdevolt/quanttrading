# backtesting/engine.py
import pandas as pd
import numpy as np # S'assurer que numpy est importé si utilisé

def run_backtest(df_with_signals, initial_capital=100000):
    """
    Exécute une simulation de backtesting très simplifiée basée sur les signaux générés.
    Achète avec tout le capital sur signal 1, vend tout sur signal -1.
    Ignore les frais, le slippage, les dividendes, etc.

    Args:
        df_with_signals (pd.DataFrame): DataFrame contenant 'Close' et 'positions' (généré par la stratégie).
                                        Doit avoir un index DatetimeIndex.
        initial_capital (float): Capital de départ pour le backtest.

    Returns:
        pd.Series: Série Pandas représentant l'évolution de la valeur du portefeuille (courbe d'équité),
                   indexée par date. Retourne None si les inputs ne sont pas valides.
    """
    if df_with_signals is None or df_with_signals.empty:
        print("Erreur: DataFrame with signals est vide ou None.")
        return None
    if 'Close' not in df_with_signals.columns or 'positions' not in df_with_signals.columns:
         print("Erreur: DataFrame must contain 'Close' and 'positions' columns.")
         return None
    if not isinstance(df_with_signals.index, pd.DatetimeIndex):
        print("Erreur: DataFrame index must be DatetimeIndex.")
        return None


    # Créer un DataFrame pour suivre le portefeuille, avec le même index que les données
    portfolio = pd.DataFrame(index=df_with_signals.index)
    portfolio['holdings'] = 0.0  # Nombre d'actions détenues (float pour simplifier, pas d'actions fractionnées ici)
    portfolio['cash'] = float(initial_capital) # Capital disponible
    portfolio['total'] = float(initial_capital) # Valeur totale du portefeuille

    shares = 0.0
    cash = float(initial_capital)

    # Parcourir les données chronologiquement pour simuler les trades
    # Utiliser itertuples est souvent plus rapide que iterrows pour les boucles
    for row in df_with_signals.itertuples():
        date = row.Index
        close_price = row.Close
        position_change = row.positions # 1.0 pour achat, -1.0 pour vente, NaN sinon

        # --- Logique d'exécution simplifiée ---
        # Achète quand signal passe de 0 à 1 (position_change == 1.0)
        if position_change == 1.0:
            if cash > 0 and close_price > 0: # S'assurer d'avoir du cash et prix non nul
                # Acheter avec tout le cash disponible
                shares_to_buy = cash / close_price
                shares += shares_to_buy
                cash = 0.0 # Tout le cash est investi
                # print(f"BUY: {date.date()} @ {close_price:.2f} | Shares: {shares:.2f}, Cash: {cash:.2f}")


        # Vend quand signal passe de 1 à 0 (position_change == -1.0)
        elif position_change == -1.0:
            if shares > 0 and close_price > 0: # S'assurer de détenir des actions et prix non nul
                # Vendre toutes les actions
                cash += shares * close_price
                shares = 0.0 # Toutes les actions sont vendues
                 # print(f"SELL: {date.date()} @ {close_price:.2f} | Shares: {shares:.2f}, Cash: {cash:.2f}")

        # Mettre à jour la valeur totale du portefeuille à la fin de cette journée
        # La valeur totale est le cash + la valeur actuelle des actions détenues
        portfolio.loc[date, 'holdings'] = shares
        portfolio.loc[date, 'cash'] = cash
        portfolio.loc[date, 'total'] = cash + shares * close_price


    # Retourne la série de la valeur totale du portefeuille au fil du temps
    return portfolio['total']
