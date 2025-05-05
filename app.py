# app.py
import streamlit as st
import pandas as pd
import numpy as np # Assurez-vous d'importer numpy si vos autres modules l'utilisent

# Importation des modules de votre application (les chemins restent les mêmes)
from data import loader
from strategies import simple_ma
from backtesting import engine, metrics

# --- Configuration de l'interface Streamlit ---
st.set_page_config(layout="wide", page_title="Application de Backtesting Simple")

st.title("Application de Backtesting Simple")

st.write("Uploader votre fichier de données historiques (CSV) et configurez les paramètres pour le backtest.")

# --- Input Fichier ---
uploaded_file = st.file_uploader("Uploader votre fichier de données historiques (CSV)", type=["csv"])

# --- Inputs de Configuration (affichés seulement si un fichier est potentiellement là) ---
if uploaded_file is not None:
    st.subheader("Paramètres du Backtest")

    # Charger les données immédiatement après l'upload pour obtenir la plage de dates
    df_initial = loader.load_historical_data_from_upload(uploaded_file)

    if df_initial is not None and not df_initial.empty:
        min_date = df_initial.index.min()
        max_date = df_initial.index.max()

        col1, col2 = st.columns(2)

        with col1:
            # On utilise les dates min/max du fichier comme bornes pour les sélecteurs de date
            start_date = st.date_input("Date de début de l'analyse", value=min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("Date de fin de l'analyse", value=max_date, min_value=min_date, max_value=max_date)

        with col2:
            initial_capital = st.number_input("Capital Initial", min_value=1000, value=100000, step=1000)
            short_window = st.slider("Période MA Courte", min_value=10, max_value=100, value=40)
            long_window = st.slider("Période MA Longue", min_value=50, max_value=250, value=100)

        # --- Bouton de Lancement du Backtest ---
        if st.button("Lancer le Backtest"):
            st.info("Préparation des données et exécution du backtest en cours...")

            # Re-charger les données (pour s'assurer qu'on travaille sur une copie propre si nécessaire)
            # ou simplement utiliser df_initial si vous savez que loader retourne une nouvelle copie
            df = loader.load_historical_data_from_upload(uploaded_file)

            if df is not None and not df.empty:
                 # Filtrer les données par la plage de dates sélectionnée par l'utilisateur
                 df_filtered = df.loc[str(start_date):str(end_date)].copy() # Utilisez .copy() pour éviter SettingWithCopyWarning

                 if df_filtered.empty:
                      st.warning(f"Aucune donnée disponible entre le {start_date} et le {end_date}. Ajustez la plage de dates.")
                 else:
                      st.success(f"Données filtrées chargées avec succès ({len(df_filtered)} lignes).")

                      # 2. Appliquer la stratégie pour générer les signaux
                      # La stratégie s'attend à un DataFrame avec une colonne 'Close' et un index DatetimeIndex
                      df_strat = simple_ma.apply_strategy(df_filtered, short_window, long_window)

                      # 3. Exécuter le backtest
                      # Le moteur run_backtest s'attend à un df avec 'Close' et 'positions'
                      equity_curve = engine.run_backtest(df_strat, initial_capital)

                      if equity_curve is not None and not equity_curve.empty:
                          st.success("Backtest terminé.")

                          # 4. Calculer les métriques de performance
                          performance_metrics = metrics.calculate_performance_metrics(equity_curve)

                          # --- Afficher les Résultats ---
                          st.header("Résultats du Backtest")

                          # Afficher les métriques
                          st.subheader("Métriques de Performance")
                          metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                          with metrics_col1:
                              st.metric("Capital Initial", performance_metrics.get("Capital Initial", "N/A"))
                          with metrics_col2:
                               st.metric("Capital Final", performance_metrics.get("Capital Final", "N/A"))
                          with metrics_col3:
                               st.metric("Retour Total", performance_metrics.get("Retour Total (%)", "N/A"))


                          st.subheader("Courbe d'Équité")
                          # Afficher la courbe d'équité
                          st.line_chart(equity_curve)

                          # Optionnel : Afficher le graphique des prix avec les signaux
                          st.subheader("Prix de l'Action avec Signaux")
                          df_plot = df_strat[['Close']].copy()

                          buy_dates = df_strat[df_strat['positions'] == 1].index
                          sell_dates = df_strat[df_strat['positions'] == -1].index

                          df_plot['Buy Signal'] = None
                          df_plot['Sell Signal'] = None

                          # Placer les points de signal sur le graphique des prix
                          # Utilisez la colonne 'Close' du DataFrame filtré/stratégie
                          df_plot.loc[buy_dates, 'Buy Signal'] = df_strat['Close'][buy_dates] * 0.95 # Légèrement en dessous du prix
                          df_plot.loc[sell_dates, 'Sell Signal'] = df_strat['Close'][sell_dates] * 1.05 # Légèrement au dessus du prix

                          st.line_chart(df_plot, use_container_width=True)
                          st.markdown("*(Les points bleus indiquent les achats, les points rouges indiquent les ventes selon la stratégie simplifiée)*")


                      else:
                          st.error("Une erreur est survenue pendant l'exécution du backtest ou la courbe d'équité est vide.")
            else:
                 st.error("Erreur lors du re-chargement ou du filtrage des données.")

    else:
        st.error("Impossible de charger ou de traiter le fichier uploadé. Vérifiez le format.")


st.sidebar.header("À Propos")
st.sidebar.info(
    "Cette application est un exemple *simplifié* de backtesting "
    "basé sur des données uploadées (CSV), démontrant l'organisation du code."
    "\n\nAssurez-vous que votre fichier CSV contient au moins "
    "une colonne 'Date' (format YYYY-MM-DD ou similaire) "
    "et une colonne 'Close' (prix de clôture)."
)
