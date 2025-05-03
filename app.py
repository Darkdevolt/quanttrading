# -*- coding: utf-8 -*-
import streamlit as st
from utils.data_processing import initialize_session_state, handle_file_upload, process_data
from utils.visualization import display_data_preview, display_price_chart
from utils.indicators import calculate_technical_indicators
from utils.backtesting import run_backtest, display_backtest_results

# Configuration de la page
st.set_page_config(
    page_title="BRVM Quant Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': """
        ## BRVM Quant Backtest App
        **Version:** 1.7.1
        Cette application permet d'analyser et de backtester des stratégies d'investissement.
        """
    }
)

def main():
    # Initialisation de l'état de la session
    initialize_session_state()
    
    # Titre de l'application
    st.title("☑ BRVM Quant Backtest")
    
    # Section de chargement de fichier dans la sidebar
    current_uploaded_file = handle_file_upload()
    
    # Si des données sont disponibles
    if 'data' in st.session_state and not st.session_state.data.empty:
        data = st.session_state.data.copy()
        
        # Affichage des données et visualisations
        display_data_preview(data)
        display_price_chart(data)
        
        # Calcul des indicateurs techniques
        analysis_data = calculate_technical_indicators(data)
        
        # Section de backtesting
        if st.button("🔍 Lancer le Backtest", key="run_backtest", type="primary"):
            with st.spinner("Exécution du backtest en cours..."):
                portfolio_history, trades = run_backtest(analysis_data)
                display_backtest_results(portfolio_history, trades, data)
    
    # Messages d'information si aucune donnée n'est chargée
    elif current_uploaded_file is None:
        st.info("😊 Veuillez charger un fichier CSV de données historiques dans la barre latérale pour commencer.")
    elif 'data' not in st.session_state or st.session_state.data.empty:
        if current_uploaded_file is not None and st.session_state.all_columns:
            st.info("😊 Fichier chargé. Veuillez mapper les colonnes et cliquer sur 'Traiter les Données' dans la barre latérale.")

if __name__ == "__main__":
    main()
