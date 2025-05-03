import pandas as pd
import streamlit as st
import csv
import traceback
from datetime import datetime

def initialize_session_state():
    """Initialise les valeurs par défaut de session_state."""
    default_values = {
        'uploaded_file_obj': None,
        'all_columns': [],
        'column_mapping': {"Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""},
        'date_format_input': "",
        'data': pd.DataFrame(),
        'stock_name': "MonActionBRVM",
        # ... (toutes les autres valeurs par défaut)
    }
    
    for key, value in default_values.items():
        st.session_state.setdefault(key, value)

def handle_file_upload():
    """Gère le chargement et le mapping des fichiers."""
    st.sidebar.subheader("1. Chargement des Données")
    
    def handle_upload():
        if st.session_state['new_uploaded_file'] is not None:
            st.session_state.uploaded_file_obj = st.session_state['new_uploaded_file']
            # Réinitialiser les états dépendants du fichier
            st.session_state.data = pd.DataFrame()
            st.session_state.all_columns = []
            st.session_state.column_mapping = {"Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""}
            st.session_state.date_format_input = ""
            st.warning("Nouveau fichier chargé. Veuillez remapper les colonnes et retraiter les données si nécessaire.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Chargez votre fichier CSV d'historique",
        type=['csv'],
        key='new_uploaded_file',
        on_change=handle_upload,
        help="Assurez-vous que le fichier contient au moins les colonnes Date, Open, High, Low, Close, Volume."
    )
    
    current_uploaded_file = st.session_state.uploaded_file_obj
    
    if current_uploaded_file is not None:
        _handle_column_mapping(current_uploaded_file)
    
    return current_uploaded_file

def _handle_column_mapping(file):
    """Gère le mapping des colonnes dans la sidebar."""
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier CSV correspondant aux champs requis.")
    
    if not st.session_state.all_columns:
        try:
            file.seek(0)
            sample_bytes = file.read(2048)
            file.seek(0)
            
            try: 
                sample_text = sample_bytes.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError: 
                sample_text = sample_bytes.decode('latin-1', errors='ignore')
                encoding = 'latin-1'
            
            sep = ','
            try:
                if sample_text.strip():
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample_text)
                    sep = dialect.delimiter
            except csv.Error:
                if sample_text and sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','):
                    sep = ';'
            
            df_cols = pd.read_csv(file, sep=sep, encoding=encoding, nrows=0)
            st.session_state.all_columns = df_cols.columns.tolist()
            st.sidebar.write("Colonnes trouvées :", st.session_state.all_columns)
            file.seek(0)
        except Exception as e:
            st.sidebar.error(f"Impossible de lire les colonnes du fichier. Vérifiez le format et le séparateur. Erreur: {e}")
            st.session_state.all_columns = []
    
    if st.session_state.all_columns:
        _create_mapping_selectors()

def _create_mapping_selectors():
    """Crée les selectors pour le mapping des colonnes."""
    required_map = {
        "Date": "Date", 
        "Open": "Ouverture (Open)", 
        "High": "Plus Haut (High)",
        "Low": "Plus Bas (Low)", 
        "Close": "Clôture (Close/Prix)", 
        "Volume": "Volume"
    }
    
    options = [""] + st.session_state.all_columns
    pre_selected_values = {}
    
    # Logique de pré-sélection intelligente...
    
    for standard_name, display_name in required_map.items():
        current_selection = st.session_state.column_mapping.get(standard_name, "")
        default_value = pre_selected_values.get(standard_name, current_selection)
        
        if default_value not in options:
            default_value = ""
        
        st.session_state.column_mapping[standard_name] = st.sidebar.selectbox(
            f"Colonne pour '{display_name}'",
            options=options,
            index=options.index(default_value) if default_value in options else 0,
            key=f"map_{standard_name}",
            help=f"Sélectionnez la colonne CSV qui contient les données de {display_name}."
        )
    
    # Options avancées (format de date)
    with st.sidebar.expander("Options Avancées"):
        st.session_state.date_format_input = st.text_input(
            "Format de date (si conversion auto échoue)",
            value=st.session_state.date_format_input,
            key="date_format",
            help="Exemples: %d/%m/%Y, %Y-%m-%d %H:%M:%S. Laisser vide pour la détection auto."
        ).strip()
    
    if st.sidebar.button(" █ Traiter les Données", key="process_button", type="primary"):
        _process_uploaded_file()

def _process_uploaded_file():
    """Lance le traitement du fichier uploadé."""
    missing_maps = [name for name, col in st.session_state.column_mapping.items() if not col]
    
    if missing_maps:
        st.warning(f"Veuillez mapper toutes les colonnes requises : {', '.join(missing_maps)}")
    else:
        mapped_cols = [col for col in st.session_state.column_mapping.values() if col]
        if len(mapped_cols) != len(set(mapped_cols)):
            st.warning("Attention : La même colonne CSV a été sélectionnée pour plusieurs champs différents.")
        
        current_uploaded_file = st.session_state.uploaded_file_obj
        current_uploaded_file.seek(0)
        
        with st.spinner("Traitement des données en cours..."):
            processed_data = process_data(
                current_uploaded_file,
                st.session_state.column_mapping,
                st.session_state.date_format_input or None
            )
            
            if processed_data is not None:
                st.session_state.data = processed_data
                st.rerun()
            elif 'data' not in st.session_state:
                st.session_state.data = pd.DataFrame()

def process_data(file, column_mapping, date_format=None):
    """Charge, nettoie et standardise les données OHLCV depuis un fichier CSV."""
    if file is None:
        st.error("Aucun fichier n'a été chargé.")
        return None
    
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys):
        st.error("Veuillez mapper toutes les colonnes requises (Date, Open, High, Low, Close, Volume).")
        return None
    
    try:
        # Détection intelligente du séparateur et de l'encodage
        file.seek(0)
        sample_bytes = file.read(2048)
        file.seek(0)
        
        # ... (le reste de votre fonction process_data existante)
        
        return df_standardized
    
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        st.error(traceback.format_exc())
        return None
