# backtesting/metrics.py
import pandas as pd

def calculate_performance_metrics(equity_curve):
    if equity_curve is None or equity_curve.empty:
        return {"Erreur": "Courbe d'équité invalide"}
    
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    
    total_return = (final_value / initial_value - 1) * 100
    
    if len(equity_curve) > 1:
        num_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        cagr = ((final_value / initial_value) ** (1/num_years) - 1) * 100 if num_years > 0 else 0
    else:
        cagr = 0

    return {
        "Capital Initial": f"{initial_value:,.2f}",
        "Capital Final": f"{final_value:,.2f}",
        "Retour Total (%)": f"{total_return:.2f}%",
        "CAGR (%)": f"{cagr:.2f}%" if cagr != 0 else "N/A"
    }
