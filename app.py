# -*- coding: utf-8 -*-
import streamlit as st
from utils.data_processing import initialize_session_state, handle_file_upload
from utils.visualization import display_data_preview, display_price_chart
from utils.indicators import calculate_technical_indicators
from utils.backtesting import run_backtest, display_backtest_results

# Page config
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
        """
    }
)

def main():
    initialize_session_state()
    st.title("☑ BRVM Quant Backtest")
    
    # File upload and processing
    current_uploaded_file = handle_file_upload()
    
    if 'data' in st.session_state and not st.session_state.data.empty:
        data = st.session_state.data.copy()
        display_data_preview(data)
        display_price_chart(data)
        
        # Technical analysis
        analysis_data = calculate_technical_indicators(data)
        
        # Backtesting
        if st.button("🔍 Lancer le Backtest", type="primary"):
            with st.spinner("Exécution du backtest en cours..."):
                portfolio_history, trades = run_backtest(analysis_data)
                display_backtest_results(portfolio_history, trades, data)
    
    elif current_uploaded_file is None:
        st.info("😊 Veuillez charger un fichier CSV pour commencer.")
    elif 'data' not in st.session_state or st.session_state.data.empty:
        st.info("😊 Fichier chargé. Veuillez mapper les colonnes.")

if __name__ == "__main__":
    main()
