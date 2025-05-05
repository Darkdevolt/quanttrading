# backtesting/metrics.py
import pandas as pd
import numpy as np # S'assurer que numpy est importé si utilisé

def calculate_performance_metrics(equity_curve):
    """
    Calcule des métriques de performance simples à partir de la courbe d'équité.

    Args:
        equity_curve (pd.Series): Série Pandas représentant la valeur du portefeuille au fil du temps.
                                   Doit avoir un index DatetimeIndex.

    Returns:
        dict: Dictionnaire contenant les métriques calculées.
              Retourne un dictionnaire avec une clé 'Erreur' si la courbe d'équité n'est pas valide.
    """
    if equity_curve is None or equity_curve.empty:
        return {"Erreur": "Courbe d'équité non valide ou vide."}

    # S'assurer que l'index est bien DatetimeIndex pour calculer le CAGR
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
         return {"Erreur": "L'index de la courbe d'équité doit être DatetimeIndex pour calculer les métriques."}

    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]

    # Éviter la division par zéro si le capital initial est 0
    if initial_value <= 0:
         return {"Erreur": "Capital initial invalide pour le calcul des métriques."}


    # Calcul du retour total
    total_return = (final_value / initial_value - 1) * 100

    # Calcul du CAGR (Compound Annual Growth Rate)
    # Nécessite au moins deux points dans la courbe
    if len(equity_curve) > 1:
        first_date = equity_curve.index[0]
        last_date = equity_curve.index[-1]
        # Calculer la durée en années
        # Utiliser days / 365.25 pour une année bissextile moyenne
        num_years = (last_date - first_date).days / 365.25
        # Éviter la division par zéro si la durée est très courte
        if num_years > 0:
            cagr = ((final_value / initial_value) ** (1 / num_years) - 1) * 100
        else:
            cagr = 0.0 # Si durée nulle, CAGR est 0 (ou infini, mais 0 est plus sûr ici)
    else:
        cagr = 0.0 # Pas assez de données pour calculer le CAGR


    # Ajout d'une métrique de drawdown simple (difficile de calculer le max drawdown sans une boucle ou une fonction dédiée)
    # On va laisser le drawdown pour un exemple plus avancé.

    metrics = {
        "Capital Initial": f"{initial_value:,.2f}",
        "Capital Final": f"{final_value:,.2f}",
        "Retour Total (%)": f"{total_return:.2f}%",
        "CAGR (%)": f"{cagr:.2f}%" if cagr != 0 else "N/A", # Afficher N/A si CAGR n'est pas calculable
        # Ajoutez d'autres métriques ici comme Volatilité, Sharpe Ratio, Max Drawdown (nécessite plus de code)
    }

    return metrics
