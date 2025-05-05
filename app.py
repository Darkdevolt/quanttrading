# app.py
import streamlit as st
import pandas as pd
from data import loader
from strategies import simple_ma
from backtesting import engine, metrics

st.set_page_config(layout="wide", page_title="BRVM Quant")
st.title("📈 BRVM Backtesting Tool")

def initialize_session():
    defaults = {
        'uploaded_file': None,
        'df_processed': None,
        'column_mapping': {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]},
        'backtest_results': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session()

# Sidebar File Upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        
        # Auto-detect columns
        sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
        separator = csv.Sniffer().sniff(sample).delimiter if sample.strip() else ','
        uploaded_file.seek(0)
        df_sample = pd.read_csv(uploaded_file, sep=separator, nrows=1)
        st.session_state.all_columns = df_sample.columns.tolist()

        # Auto-map columns
        auto_mapping = {
            'Date': next((col for col in st.session_state.all_columns if 'date' in col.lower()), ''),
            'Close': next((col for col in st.session_state.all_columns if 'close' in col.lower()), '')
        }
        st.session_state.column_mapping.update(auto_mapping)

# Column Mapping UI
if st.session_state.get('all_columns'):
    with st.expander("Column Mapping"):
        for standard_col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            st.session_state.column_mapping[standard_col] = st.selectbox(
                f"Map {standard_col}",
                options=[""] + st.session_state.all_columns,
                index=st.session_state.all_columns.index(st.session_state.column_mapping[standard_col]) 
                if st.session_state.column_mapping[standard_col] in st.session_state.all_columns else 0
            )

# Processing
if st.button("Run Backtest"):
    if st.session_state.uploaded_file:
        df = loader.load_and_process_data(
            st.session_state.uploaded_file,
            st.session_state.column_mapping
        )
        
        if df is not None:
            df_strat = simple_ma.apply_strategy(df)
            equity_curve = engine.run_backtest(df_strat)
            st.session_state.backtest_results = metrics.calculate_performance_metrics(equity_curve)
            
            # Display Results
            st.subheader("Performance Metrics")
            st.write(st.session_state.backtest_results)
            
            st.subheader("Equity Curve")
            st.line_chart(equity_curve)
        else:
            st.error("Error processing data. Check column mapping.")
