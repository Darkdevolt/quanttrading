# app.py

import streamlit as st
import pandas as pd
import numpy as np
import csv

from data import loader
from strategies import simple_ma
from backtesting import engine, metrics

# Configuration de l'application
st.set_page_config(layout="wide", page_title="BRVM Quant Backtest")

st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")

# Initialisation de l'état de session
if 'uploaded_file_obj' not in st.session_state:
    st.session_state.uploaded_file_obj = None
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()
if 'all_columns' not in st.session_state:
    st.session_state.all_columns = []
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
    }
if 'date_format_input' not in st.session_state:
    st.session_state.date_format_input = ""
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

st.sidebar.header("Paramètres Globaux")

# Fonction de gestion de l'upload
def handle_upload():
    uploaded_file = st.session_state['new_uploaded_file']
    st.session_state.uploaded_file_obj = uploaded_file

    if uploaded_file is not None:
        st.sidebar.info("Fichier uploadé. Détection des colonnes...")
        try:
            uploaded_file.seek(0)
            sample_bytes = uploaded_file.read(2048)
            uploaded_file.seek(0)
            try: sample_text = sample_bytes.decode('utf-8')
            except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

            separator = ','
            try:
                if sample_text.strip():
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample_text)
                    separator = dialect.delimiter
            except csv.Error:
                uploaded_file.seek(0)
                try: header_line_bytes = uploaded_file.readline()
                except Exception: header_line_bytes = b''
                try: header_line = header_line_bytes.decode('utf-8')
                except UnicodeDecodeError: header_line = header_line_bytes.decode('latin-1', errors='ignore')
                uploaded_file.seek(0)
                if header_line and header_line.count(';') >= header_line.count(','): separator = ';'

            uploaded_file.seek(0)
            temp_df = pd.read_csv(uploaded_file, sep=separator, nrows=0)
            st.session_state.all_columns = list(temp_df.columns)
            st.sidebar.success(f"Colonnes détectées : {', '.join(st.session_state.all_columns)}")

            st.session_state.column_mapping = {
                "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
            }
            for standard_name in st.session_state.column_mapping.keys():
                matching_cols = [col for col in st.session_state.all_columns if standard_name.lower() in col.lower()]
                if matching_cols:
                    st.session_state.column_mapping[standard_name] = matching_cols[0]

        except Exception as e:
            st.sidebar.error(f"Erreur lors de la détection des colonnes : {e}")
            st.session_state.all_columns = []
            st.session_state.uploaded_file_obj = None
            st.session_state.column_mapping = {
                "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
            }

    else:
        st.session_state.uploaded_file_obj = None
        st.session_state.data = pd.DataFrame()
        st.session_state.all_columns = []
        st.session_state.column_mapping = {
            "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
        }
        st.session_state.date_format_input = ""
        st.session_state.backtest_results = None
        st.sidebar.info("Aucun fichier chargé.")

# Upload du fichier
st.sidebar.subheader("1. Chargement des Données")
st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file',
    on_change=handle_upload
)

# Mapping des colonnes
if st.session_state.all_columns:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.write("Associez les colonnes de votre fichier aux noms standardisés.")

    for standard_name in st.session_state.column_mapping.keys():
        options = [''] + st.session_state.all_columns
        current_selection = st.session_state.column_mapping[standard_name]
        index = options.index(current_selection) if current_selection in options else 0

        selected_column = st.sidebar.selectbox(
            f"Colonne pour '{standard_name}'",
            options,
            index=index,
            key=f'map_{standard_name}'
        )
        st.session_state.column_mapping[standard_name] = selected_column

    st.sidebar.subheader("Format de Date (Optionnel)")
    st.session_state.date_format_input = st.sidebar.text_input(
        "Spécifiez le format de date (ex: %Y-%m-%d)",
        value=st.session_state.date_format_input,
        key='date_format_key',
        help="Laissez vide pour détection automatique. Exemples: %Y-%m-%d, %d/%m/%Y"
    )

    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    all_required_mapped = all(st.session_state.column_mapping.get(key) for key in required_keys)

    if st.sidebar.button("Processer les Données", disabled=not all_required_mapped):
        st.info("Traitement des données en cours...")
        processed_df = loader.load_and_process_data(
            st.session_state.uploaded_file_obj,
            st.session_state.column_mapping,
            st.session_state.date_format_input
        )

        if processed_df is not None and not processed_df.empty:
            st.session_state.data = processed_df
            st.success("Données traitées avec succès.")
            st.write("Aperçu des données traitées :")
            st.dataframe(st.session_state.data.head())
            st.info(f"Données disponibles du {st.session_state.data.index.min().date()} au {st.session_state.data.index.max().date()}.")
        else:
            st.session_state.data = pd.DataFrame()
            st.error("Impossible de traiter les données avec le mapping et le format fournis. Vérifiez vos sélections et le contenu du fichier.")

# (Suite à compléter avec la partie backtesting si nécessaire)
