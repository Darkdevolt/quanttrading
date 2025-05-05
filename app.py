# app.py
import streamlit as st
import pandas as pd
import numpy as np
import csv
import io

# Importation des modules (assurez-vous que ces fichiers existent et sont accessibles)
try:
    from data import loader
    from strategies import simple_ma
    from backtesting import engine, metrics
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez que les fichiers/dossiers existent.")
    st.stop()

# --- Configuration Streamlit ---
st.set_page_config(layout="wide", page_title="BRVM Quant Backtest")

st.title("\U0001F4C8 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")

# --- Initialisation session_state ---
def initialize_session():
    session_defaults = {
        'uploaded_file_content': None,
        'uploaded_file_name': None,
        'data': pd.DataFrame(),
        'all_columns': [],
        'column_mapping': {
            "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
        },
        'date_format_input': "",
        'backtest_results': None,
        'data_processed': False
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session()

st.sidebar.header("Paramètres Globaux")
st.sidebar.subheader("1. Chargement des Données")

def handle_upload():
    uploaded_file = st.session_state['new_uploaded_file']
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            st.session_state.uploaded_file_content = uploaded_file.read()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.sidebar.info(f"Fichier '{uploaded_file.name}' chargé. Détection des colonnes...")

            file_stream = io.BytesIO(st.session_state.uploaded_file_content)
            sample_bytes = file_stream.read(2048)
            file_stream.seek(0)

            try:
                sample_text = sample_bytes.decode('utf-8')
            except UnicodeDecodeError:
                sample_text = sample_bytes.decode('latin-1', errors='ignore')

            separator = ','
            try:
                if sample_text.strip():
                    dialect = csv.Sniffer().sniff(sample_text)
                    separator = dialect.delimiter
                else:
                    st.sidebar.warning("Echantillon vide. Utilisation de ',' par défaut.")
            except csv.Error:
                try:
                    header_line_bytes = io.BytesIO(st.session_state.uploaded_file_content).readline()
                    header_line = header_line_bytes.decode('utf-8', errors='ignore')
                    if header_line.count(';') >= header_line.count(','):
                        separator = ';'
                except Exception:
                    st.sidebar.warning("Fallback de détection échoué.")

            file_stream.seek(0)
            temp_df = pd.read_csv(file_stream, sep=separator, nrows=0)
            st.session_state.all_columns = list(temp_df.columns)
            st.sidebar.success(f"Colonnes détectées: {', '.join(st.session_state.all_columns)}")

            st.session_state.column_mapping = {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]}
            for standard_name in st.session_state.column_mapping.keys():
                matching_cols = [col for col in st.session_state.all_columns if standard_name.lower() in col.lower()]
                if matching_cols:
                    st.session_state.column_mapping[standard_name] = matching_cols[0]

            st.session_state.data = pd.DataFrame()
            st.session_state.data_processed = False
            st.session_state.backtest_results = None

        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement: {e}")
            initialize_session()
    else:
        if st.session_state.uploaded_file_name:
            st.sidebar.info("Fichier retiré.")
            initialize_session()

uploaded_file_widget = st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file',
    on_change=handle_upload
)

if st.session_state.uploaded_file_name and st.session_state.all_columns:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.write("Associez les colonnes de votre fichier aux noms standardisés.")
    options_list = [''] + st.session_state.all_columns

    for standard_name in st.session_state.column_mapping.keys():
        mapped_col = st.session_state.column_mapping.get(standard_name, "")
        selectbox_index = 0
        if mapped_col in options_list:
            try:
                selectbox_index = options_list.index(mapped_col)
            except ValueError:
                selectbox_index = 0

        selected_column = st.sidebar.selectbox(
            label=f"Colonne pour '{standard_name}'",
            options=options_list,
            index=selectbox_index,
            key=f'map_{standard_name}'
        )
        st.session_state.column_mapping[standard_name] = selected_column
