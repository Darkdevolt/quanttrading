# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Importation des modules
from data import loader
from strategies import simple_ma
from backtesting import engine, metrics

# --- Configuration de l'interface Streamlit ---
st.set_page_config(layout="wide", page_title="Application de Backtesting (CSV/Excel)")

st.title("Application de Backtesting (Données CSV ou Excel)")

st.write("Uploader votre fichier de données historiques (CSV ou Excel) et configurez les paramètres pour le backtest.")

# --- Input Fichier ---
# Accepte maintenant les fichiers CSV, XLSX et XLS
uploaded_file = st.file_uploader("Uploader votre fichier de données historiques (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

# --- Inputs de Configuration (affichés seulement si un fichier est potentiellement là) ---
df_initial = None
if uploaded_file is not None:
    # Tenter de charger les données immédiatement après l'upload
    # Le loader s'occupe de détecter le format
    df_initial = loader.load_historical_data_from_upload(uploaded_file)

    if df_initial is not None and not df_initial.empty:
        st.success("Fichier chargé et traité avec succès.")
        st.write("Aperçu des données chargées :")
        st.dataframe(df_initial.head()) # Afficher les premières lignes

        st.subheader("Paramètres du Backtest")

        # Utiliser les dates min/max du fichier pour les sélecteurs de date
        min_date_data = df_initial.index.min().date()
        max_date_data = df_initial.index.max().date()

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Date de début de l'analyse", value=min_date_data, min_value=min_date_data, max_value=max_date_data)
            end_date = st.date_input("Date de fin de l'analyse", value=max_date_data, min_value=min_date_data, max_value=max_date_data)

        with col2:
            initial_capital = st.number_input("Capital Initial", min_value=1000, value=100000, step=1000)
            short_window = st.slider("Période MA Courte", min_value=10, max_value=100, value=40)
            long_window = st.slider("Période MA Longue", min_value=50, max_value=250, value=100)

        # --- Bouton de Lancement du Backtest ---
        if st.button("Lancer le Backtest"):
            st.info("Préparation des données et exécution du backtest en cours...")

            # Utiliser le DataFrame déjà chargé pour l'analyse
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                 st.error("La date de début ne peut pas être postérieure à la date de fin.")
            else:
                # Filtrer les données par la plage de dates sélectionnée
                df_filtered = df_initial.loc[str(start_date):str(end_date)].copy()

                if df_filtered.empty:
                    st.warning(f"Aucune donnée disponible entre le {start_date} et le {end_date}. Ajustez la plage de dates ou le fichier.")
                else:
                    st.success(f"Données filtrées chargées avec succès ({len(df_filtered)} lignes).")

                    # 2. Appliquer la stratégie
                    df_strat = simple_ma.apply_strategy(df_filtered, short_window, long_window)

                    # 3. Exécuter le backtest
                    equity_curve = engine.run_backtest(df_strat, initial_capital)

                    if equity_curve is not None and not equity_curve.empty:
                        st.success("Backtest terminé.")

                        # 4. Calculer les métriques
                        performance_metrics = metrics.calculate_performance_metrics(equity_curve)

                        # --- Afficher les Résultats ---
                        st.header("Résultats du Backtest")

                        st.subheader("Métriques de Performance")
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        with metrics_col1:
                            st.metric("Capital Initial", performance_metrics.get("Capital Initial", "N/A"))
                        with metrics_col2:
                             st.metric("Capital Final", performance_metrics.get("Capital Final", "N/A"))
                        with metrics_col3:
                             st.metric("Retour Total", performance_metrics.get("Retour Total (%)", "N/A"))
                        with metrics_col4:
                             st.metric("CAGR", performance_metrics.get("CAGR (%)", "N/A"))


                        st.subheader("Courbe d'Équité")
                        st.line_chart(equity_curve)

                        st.subheader("Prix de l'Action avec Signaux")
                        df_plot = df_strat[['Close']].copy()

                        buy_dates = df_strat[df_strat['positions'] == 1].index
                        sell_dates = df_strat[df_strat['positions'] == -1].index

                        df_plot['Buy Signal'] = None
                        df_plot['Sell Signal'] = None

                        df_plot.loc[buy_dates, 'Buy Signal'] = df_strat['Close'][buy_dates] * 0.95
                        df_plot.loc[sell_dates, 'Sell Signal'] = df_strat['Close'][sell_dates] * 1.05

                        df_final_plot = df_strat[['Close']].copy()
                        df_final_plot['Buy Signal'] = df_plot['Buy Signal']
                        df_final_plot['Sell Signal'] = df_plot['Sell Signal']


                        st.line_chart(df_final_plot, use_container_width=True)
                        st.markdown("*(Les points bleus indiquent les achats, les points rouges indiquent les ventes selon la stratégie simplifiée)*")


                    else:
                        st.error("Une erreur est survenue pendant l'exécution du backtest ou la courbe d'équité est vide.")

    # Si le fichier a été uploadé mais le chargement initial a échoué
    elif uploaded_file is not None and (df_initial is None or df_initial.empty):
         st.error("Impossible de charger ou de traiter le fichier uploadé. Veuillez vérifier qu'il est bien un fichier CSV ou Excel (.xlsx ou .xls) valide avec les colonnes 'Date' et 'Close'.")


# Message si aucun fichier n'a encore été uploadé
elif uploaded_file is None:
     st.info("Veuillez uploader un fichier CSV ou Excel (.csv, .xlsx, ou .xls) de données historiques pour commencer.")


st.sidebar.header("À Propos")
st.sidebar.info(
    "Cette application est un exemple *simplifié* de backtesting "
    "basé sur des données uploadées, démontrant l'organisation du code."
    "\n\nAssurez-vous que votre fichier (CSV ou Excel) contient au moins "
    "une colonne nommée exactement `Date` (format date/heure standard) "
    "et une colonne nommée exactement `Close` (prix de clôture numérique)."
)
