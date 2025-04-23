# -*- coding: utf-8 -*- # Spécifier l'encodage utf-8
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64
from pandas.tseries.offsets import BDay # Pour gérer les jours ouvrés
import csv # <--- MODULE CSV IMPORTÉ ICI
import traceback # Pour les erreurs détaillées

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="BRVM Quant Backtest",
    layout="wide",
    initial_sidebar_state="expanded", # Garder la sidebar ouverte par défaut
    menu_items={
        'Get Help': 'https://www.example.com/help', # Lien Aide (à remplacer)
        'Report a bug': "https://www.example.com/bug", # Lien Bug (à remplacer)
        'About': """
        ## BRVM Quant Backtest App
        **Version:** 1.3 (Persistance Données via Session State)

        Cette application permet d'analyser et de backtester des stratégies d'investissement
        sur les actions cotées à la Bourse Régionale des Valeurs Mobilières (BRVM)
        en utilisant vos propres données historiques.

        **Auteur:** Votre Nom/Organisation
        **Note:** Les résultats du backtesting sont basés sur des données historiques
        et ne garantissent pas les performances futures. Utilisez cette application
        comme un outil d'aide à la décision et non comme un conseil financier direct.
        """
    }
)

# --- Initialisation de session_state ---
# C'est crucial pour stocker les variables qui doivent persister
if 'uploaded_file_obj' not in st.session_state:
    st.session_state.uploaded_file_obj = None
if 'all_columns' not in st.session_state:
    st.session_state.all_columns = []
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
    }
if 'date_format_input' not in st.session_state:
    st.session_state.date_format_input = ""
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame() # DataFrame vide initialement
if 'stock_name' not in st.session_state:
     st.session_state.stock_name = "MonActionBRVM"


# --- Titre et Introduction ---
st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")
st.sidebar.header("Paramètres Globaux")

# --- Section Upload de Fichier ---
st.sidebar.subheader("1. Chargement des Données")

# Utiliser un callback pour gérer le cas où un NOUVEAU fichier est uploadé
def handle_upload():
    """Réinitialise l'état quand un nouveau fichier est chargé."""
    if st.session_state['new_uploaded_file'] is not None:
        st.session_state.uploaded_file_obj = st.session_state['new_uploaded_file']
        # Réinitialiser les autres états liés aux données si un nouveau fichier arrive
        st.session_state.data = pd.DataFrame()
        st.session_state.all_columns = []
        st.session_state.column_mapping = {
            "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
        }
        st.session_state.date_format_input = ""
        # On pourrait aussi réinitialiser les paramètres de mapping ici si on voulait forcer le mapping à chaque nouvel upload

# Le widget file_uploader met sa valeur dans la session_state via la key 'new_uploaded_file'
# et déclenche handle_upload si un fichier est sélectionné.
st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file',
    on_change=handle_upload # Déclenche la fonction si la valeur change
)

# Traiter le fichier actuellement dans la session_state (celui qui a été uploadé et a persisté)
current_uploaded_file = st.session_state.uploaded_file_obj


# --- Fonction de Traitement des Données (Améliorée) ---
# Cette fonction ne change pas, elle prend des inputs et retourne un DataFrame
def process_data(file, column_mapping, date_format=None):
    """
    Charge, valide et traite les données CSV uploadées.

    Args:
        file: Objet fichier uploadé par Streamlit (ou équivalent avec seek/read).
        column_mapping (dict): Dictionnaire mappant les noms standardisés
                               aux noms de colonnes du fichier CSV.
        date_format (str, optional): Format de date à essayer si la conversion échoue.

    Returns:
        pd.DataFrame: DataFrame traité et standardisé, ou None en cas d'erreur.
    """
    if file is None:
        # st.error("Veuillez charger un fichier CSV.") # Message déjà géré par l'UI
        return None
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys):
         # missing_keys = [key for key in required_keys if key not in column_mapping or not column_mapping[key]]
         # st.warning(f"Veuillez mapper toutes les colonnes requises ({', '.join(missing_keys)}) dans la barre latérale.") # Message géré avant l'appel
         return None

    try:
        # Lire l'en-tête pour prévisualisation et détection du séparateur
        file.seek(0)
        sample_bytes = file.read(2048)
        file.seek(0)
        try:
            sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError:
             sample_text = sample_bytes.decode('latin-1', errors='ignore')

        sniffer = csv.Sniffer()
        separator = ',' # Default
        try:
            if sample_text.strip():
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
                # st.info(f"Séparateur détecté par Sniffer : '{separator}'") # Peut être verbeux
            # else: st.warning("L'échantillon du fichier est vide...") # Peut être verbeux
        except csv.Error:
            # st.warning("Sniffer n'a pas pu déterminer le séparateur. Essai manuel...") # Peut être verbeux
            # ... (fallback manuel pour séparateur)
            file.seek(0)
            try:
                header_line_bytes = file.readline()
                try: header_line = header_line_bytes.decode('utf-8')
                except UnicodeDecodeError: header_line = header_line_bytes.decode('latin-1', errors='ignore')
            except Exception as read_err:
                 st.error(f"Impossible de lire la première ligne pour la détection manuelle du séparateur: {read_err}")
                 return None
            file.seek(0)
            if header_line and header_line.count(';') >= header_line.count(','): separator = ';'
            else: separator = ','
            # st.info(f"Utilisation probable du séparateur '{separator}'...") # Peut être verbeux


        file.seek(0)
        try:
            df = pd.read_csv(file, sep=separator)
        except UnicodeDecodeError:
            st.warning("Échec de la lecture en UTF-8, tentative en Latin-1...")
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=separator, encoding='latin-1')
            except Exception as enc_err:
                 st.error(f"Impossible de lire le fichier CSV avec les encodages UTF-8 ou Latin-1. Erreur: {enc_err}")
                 return None
        except Exception as read_err:
            st.error(f"Erreur lors de la lecture du fichier CSV avec pandas : {read_err}")
            return None

        if df.empty:
            st.error("Le fichier CSV est vide ou n'a pas pu être lu correctement par Pandas.")
            return None

        # st.write("Colonnes détectées dans le fichier :", list(df.columns)) # Peut être verbeux

        missing_mapped_cols = []
        for standard_name, user_name in column_mapping.items():
            if not user_name:
                 missing_mapped_cols.append(standard_name)
            elif user_name not in df.columns:
                 st.error(f"La colonne mappée '{user_name}' pour '{standard_name}' n'existe pas dans le fichier.")
                 return None
        if missing_mapped_cols:
             st.error(f"Veuillez mapper les colonnes suivantes : {', '.join(missing_mapped_cols)}")
             return None

        df_standardized = pd.DataFrame()

        # Date Conversion
        date_col_name = column_mapping['Date']
        try:
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce', infer_datetime_format=True)
            if df_standardized['Date'].isnull().all() and date_format:
                 st.info(f"Tentative de conversion de date avec le format explicite : {date_format}")
                 try:
                     df_copy = df[[date_col_name]].copy()
                     df_standardized['Date'] = pd.to_datetime(df_copy[date_col_name], format=date_format, errors='coerce')
                 except Exception as fmt_e:
                     st.error(f"Erreur application format date '{date_format}' à '{date_col_name}': {fmt_e}")
                     return None
        except Exception as e:
            st.error(f"Erreur générale conversion colonne Date ('{date_col_name}'): {e}")
            return None

        if df_standardized['Date'].isnull().all():
             st.error(f"Impossible de convertir la colonne Date ('{date_col_name}') en dates valides.")
             return None
        if df_standardized['Date'].isnull().any():
            nan_dates_count = df_standardized['Date'].isnull().sum()
            st.warning(f"{nan_dates_count} valeur(s) Date ('{date_col_name}') invalides ou vides trouvées. Lignes correspondantes seront supprimées.")
            df_standardized = df_standardized.dropna(subset=['Date'])
            if df_standardized.empty:
                 st.error("Toutes les lignes supprimées après échec conversion dates.")
                 return None

        # Numeric Conversion
        standard_to_user_map = {
            'Ouverture': column_mapping['Open'], 'Plus_Haut': column_mapping['High'],
            'Plus_Bas': column_mapping['Low'], 'Prix': column_mapping['Close'],
            'Volume': column_mapping['Volume']
        }
        for standard_col_name, user_col_name in standard_to_user_map.items():
            try:
                # Gérer les virgules comme séparateur décimal et les espaces
                if df[user_col_name].dtype == 'object':
                     cleaned_series = df[user_col_name].astype(str).str.strip().str.replace(',', '.', regex=False).str.replace(r'\s+', '', regex=True)
                     converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                     # Fallback si la première tentative échoue (e.g., caractères non numériques restants)
                     if converted_series.isnull().all() and not df[user_col_name].isnull().all():
                          cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True) # Enlever tout sauf chiffres, point, tiret
                          cleaned_series = cleaned_series.str.replace(r'^(-?\.)?$', '', regex=True) # Enlever si ne reste que . ou -.
                          cleaned_series = cleaned_series.str.replace(r'(-.*)-', r'\1', regex=True) # Gérer multiples tirets (garder le premier si présent)
                          converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                     df_standardized[standard_col_name] = converted_series
                else:
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                # Optionnel: comparer avec original si besoin, mais NaN créés par coerce sont l'info utile ici
                if nan_after_conversion > 0:
                     st.warning(f"{nan_after_conversion} NaN créés dans '{user_col_name}' ({standard_col_name}) lors de la conversion numérique. Ces valeurs seront remplies par ffill/bfill si possible.")
            except Exception as e:
                st.error(f"Erreur conversion numérique colonne '{user_col_name}' ({standard_col_name}) : {e}")
                return None

        # --- Validation Post-Conversion ---
        numeric_standard_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        # Ne pas vérifier si ALL sont NaN tout de suite, car on va ffill/bfill.
        # On vérifiera après le remplissage.

        # --- Traitements Finaux ---
        # Trier par date
        df_standardized = df_standardized.sort_values('Date')

        # Définir l'index et gérer les duplicatas
        if df_standardized['Date'].duplicated().any():
            duplicates_count = df_standardized['Date'].duplicated().sum()
            st.warning(f"Il y a {duplicates_count} dates dupliquées dans vos données. Seule la dernière entrée pour chaque date sera conservée.")
            df_standardized = df_standardized.drop_duplicates(subset=['Date'], keep='last')

        df_standardized = df_standardized.set_index('Date')

        # Remplir les valeurs NaN restantes dans les colonnes OHLCV (méthode ffill puis bfill)
        cols_to_fill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        for col in cols_to_fill:
             if col in df_standardized.columns:
                 nan_before = df_standardized[col].isnull().sum()
                 if nan_before > 0:
                     df_standardized[col] = df_standardized[col].ffill() # Remplir avant
                     df_standardized[col] = df_standardized[col].bfill() # Remplir arrière (pour les NaN au début)
                     nan_after = df_standardized[col].isnull().sum()
                     if nan_after < nan_before:
                         st.info(f"{nan_before - nan_after} NaN dans '{col}' remplis par ffill/bfill.")
                     if nan_after > 0:
                          st.error(f"Attention: Il reste {nan_after} NaN dans la colonne '{col}' après ffill/bfill. Vérifiez vos données source, surtout au début de la série.")
                          # On pourrait choisir de retourner None ici ou laisser le DataFrame avec NaNs

        # --- Calculer Variation ET REMPLIR NaN INITIAL ---
        # Le calcul de variation doit se faire *après* le ffill/bfill sur 'Prix'
        if 'Prix' in df_standardized.columns:
            # Assurer que Prix n'est pas entièrement NaN après remplissage
            if df_standardized['Prix'].isnull().all():
                st.error("La colonne 'Prix' est entièrement NaN même après tentative de remplissage. Impossible de continuer.")
                return None

            df_standardized['Variation'] = df_standardized['Prix'].diff()
            df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
            # Remplir explicitement le NaN initial (première ligne) avec 0
            df_standardized['Variation'].fillna(0, inplace=True)
            df_standardized['Variation_%'].fillna(0, inplace=True)
        else:
             st.error("Colonne 'Prix' manquante après standardisation, impossible de calculer les variations.")
             return None


        # --- Re-vérifier les NaNs critiques après TOUS les remplissages ---
        # On doit s'assurer qu'il n'y a pas de NaNs dans les colonnes essentielles *après* toutes les tentatives.
        # 'Prix' a déjà été vérifié. Vérifions les autres si nécessaire, mais ffill/bfill devrait avoir aidé.
        # La vérification générique `isnull().any().any()` est utile ici.
        # On peut permettre des NaNs dans les MM si la fenêtre est plus grande que les données disponibles au début.
        # On va donc exclure les colonnes potentiellement NaN au début (comme les MMs) pour cette vérification critique.
        critical_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume', 'Variation', 'Variation_%']
        if df_standardized[critical_cols].isnull().any().any():
            cols_with_nan = df_standardized[critical_cols].columns[df_standardized[critical_cols].isnull().any()].tolist()
            st.error(f"Erreur critique: Il reste des valeurs manquantes inattendues dans les colonnes critiques après traitement : {cols_with_nan}. Vérifiez la qualité de vos données source.")
            st.dataframe(df_standardized[df_standardized[critical_cols].isnull().any(axis=1)])
            return None


        st.success("Données chargées et traitées avec succès !")
        return df_standardized

    except pd.errors.EmptyDataError:
        st.error("Erreur : Le fichier CSV semble vide après lecture.")
        return None
    except KeyError as e:
        st.error(f"Erreur : Problème d'accès à une colonne lors du traitement. Vérifiez le mapping et le contenu du fichier. La colonne '{e}' semble poser problème.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        st.error(traceback.format_exc()) # Afficher la pile d'appels
        return None


# --- Fonction pour Lien de Téléchargement CSV ---
# Pas de changement nécessaire
def get_csv_download_link(df, filename="rapport_backtest.csv", link_text="Télécharger le rapport (CSV)"):
    """Génère un lien pour télécharger un DataFrame en CSV."""
    if df.empty: return ""
    try:
        # Utiliser un buffer pour éviter les problèmes de position du fichier
        buffer = io.StringIO()
        df.to_csv(buffer, index=True, date_format='%Y-%m-%d %H:%M:%S')
        csv_string = buffer.getvalue()
        buffer.close()

        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        # Styles pour le bouton (identique à l'original)
        button_style = """
        display: inline-block;
        padding: 0.5em 1em;
        text-decoration: none;
        background-color: #4CAF50;
        color: white;
        border-radius: 0.25em;
        border: none;
        cursor: pointer;
        font-size: 1rem;
        margin-top: 1em;
        """
        button_hover_style = """
        <style>
        .download-button:hover {
            background-color: #45a049 !important;
            color: white !important;
            text-decoration: none !important;
        }
        </style>
        """
        st.markdown(f'{button_hover_style}', unsafe_allow_html=True)
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur création lien téléchargement : {e}")
        st.error(traceback.format_exc())
        return ""


# --- Interface Utilisateur dans la Sidebar (Après Upload) ---
# Cette section lit le fichier uploadé et configure le mapping
if current_uploaded_file is not None:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier correspondant aux champs requis.")

    # Lire l'en-tête et détecter les colonnes SEULEMENT si on ne l'a pas déjà fait pour ce fichier
    if not st.session_state.all_columns:
        try:
            current_uploaded_file.seek(0)
            sample_bytes = current_uploaded_file.read(2048)
            current_uploaded_file.seek(0)
            try: sample_text = sample_bytes.decode('utf-8')
            except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

            sniffer = csv.Sniffer()
            sep = ',' # défaut
            try:
                if sample_text.strip():
                    dialect = sniffer.sniff(sample_text)
                    sep = dialect.delimiter
            except csv.Error:
                 if sample_text and sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','): sep = ';'

            # Lire juste l'en-tête
            try:
                df_cols = pd.read_csv(current_uploaded_file, sep=sep, nrows=0)
                st.session_state.all_columns = df_cols.columns.tolist()
            except Exception as e:
                st.sidebar.error(f"Impossible lire colonnes (Vérifiez séparateur/format): {e}")
                st.session_state.all_columns = []

            current_uploaded_file.seek(0) # Revenir au début
        except Exception as e:
             st.sidebar.error(f"Erreur lecture initiale fichier pour mapping: {e}")
             st.sidebar.info("Assurez-vous que le fichier est un CSV valide (encodage UTF-8 ou Latin-1).")
             st.session_state.all_columns = [] # S'assurer que c'est vide en cas d'erreur


    if not st.session_state.all_columns:
        st.sidebar.warning("Impossible de lire les colonnes du fichier chargé.")
    else:
        st.sidebar.write("Colonnes trouvées :", st.session_state.all_columns)
        required_map = {
            "Date": "Date", "Open": "Ouverture", "High": "Plus Haut",
            "Low": "Plus Bas", "Close": "Clôture", "Volume": "Volume"
        }
        # Logique de pré-sélection (peut être affinée ou conservée si la mapping est déjà en state)
        # Pour que le mapping reste le même lors des reruns, on lit la valeur par défaut depuis session_state
        pre_selected_values = {}
        used_columns = set()
        # Priorité 1: Exact match
        for standard_name in required_map.keys():
             normalized_standard = standard_name.lower().replace('_','')
             for col in st.session_state.all_columns:
                 normalized_col = col.lower().replace('_','')
                 if normalized_standard == normalized_col and col not in used_columns:
                     pre_selected_values[standard_name] = col
                     used_columns.add(col)
                     break
        # Priorité 2: Partial match
        common_terms = { 'Date': ['date', 'time', 'jour'], 'Open': ['open', 'ouverture', 'ouv'], 'High': ['high', 'haut', 'max'], 'Low': ['low', 'bas', 'min'], 'Close': ['close', 'cloture', 'dernier', 'last', 'prix'], 'Volume': ['volume', 'vol', 'quantite', 'qty'] }
        for standard_name, terms in common_terms.items():
             if standard_name not in pre_selected_values:
                 for term in terms:
                     found_match = False
                     for col in st.session_state.all_columns:
                         if term in col.lower().replace('_','') and col not in used_columns:
                             pre_selected_values[standard_name] = col
                             used_columns.add(col)
                             found_match = True
                             break
                     if found_match: break

        # Création des selectbox, utilisant la valeur sauvegardée en session_state si elle existe
        # et la pré-sélection comme fallback initial
        for standard_name, display_name in required_map.items():
            default_value = st.session_state.column_mapping.get(standard_name, pre_selected_values.get(standard_name, ""))
            # S'assurer que la valeur par défaut est bien dans les options disponibles
            if default_value not in [""] + st.session_state.all_columns:
                 default_value = "" # Réinitialiser si la colonne n'existe plus (ex: nouveau fichier)

            st.session_state.column_mapping[standard_name] = st.sidebar.selectbox(
                f"Colonne pour '{display_name}'",
                options=[""] + st.session_state.all_columns,
                index=([""] + st.session_state.all_columns).index(default_value) if default_value else 0,
                key=f"map_{standard_name}" # Clé pour que Streamlit gère l'état du widget
            )

        with st.sidebar.expander("Options Avancées"):
             st.session_state.date_format_input = st.text_input(
                 "Format de date (si conversion auto échoue, ex: %d/%m/%Y)",
                 value=st.session_state.date_format_input, # Lire la valeur de session state
                 key="date_format",
                 help="Exemples: %Y-%m-%d, %d/%m/%Y %H:%M:%S. Voir Python strptime."
             )

        # Bouton pour déclencher le traitement
        if st.sidebar.button("▶️ Traiter les Données", key="process_button"):
            missing_maps = [name for name, col in st.session_state.column_mapping.items() if not col]
            if missing_maps:
                 st.warning(f"Veuillez mapper les colonnes requises avant de traiter : {', '.join(missing_maps)}")
            else:
                 mapped_cols = [col for col in st.session_state.column_mapping.values() if col]
                 if len(mapped_cols) != len(set(mapped_cols)):
                     st.warning("Attention: La même colonne CSV a été sélectionnée pour plusieurs champs. Vérifiez votre mapping.")
                 else:
                     # Important: Repositionner le pointeur du fichier au début avant de le lire
                     current_uploaded_file.seek(0)
                     with st.spinner("Traitement des données en cours..."):
                         # Appeler la fonction de traitement et stocker le résultat dans session_state
                         st.session_state.data = process_data(
                             current_uploaded_file,
                             st.session_state.column_mapping,
                             st.session_state.date_format_input or None # Passer le format si renseigné
                         )
                         # Le reste du script se relancera automatiquement si data n'est pas vide

# --- Exécution de l'Analyse (si les données sont chargées et traitées) ---
# Vérifier si les données traitées existent dans session_state et ne sont pas vides
if not st.session_state.data.empty:

    # Utiliser la variable stockée dans session_state
    data = st.session_state.data.copy() # Faire une copie pour éviter les SettingWithCopyWarning

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    # Lire/Écrire le nom de l'action depuis/vers session_state
    st.session_state.stock_name = st.sidebar.text_input(
        "Nom de l'action",
        st.session_state.stock_name, # Lire la valeur de session state
        key="stock_name_input" # Clé pour ce widget spécifique
    )
    st.title(f"📈 BRVM Quant Backtest - {st.session_state.stock_name}")

    # --- Affichage des Données Traitées ---
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)"):
        st.dataframe(data.tail(100).style.format({
            'Ouverture': '{:,.2f}', 'Plus_Haut': '{:,.2f}', 'Plus_Bas': '{:,.2f}',
            'Prix': '{:,.2f}', 'Volume': '{:,.0f}', 'Variation': '{:,.2f}',
            'Variation_%': '{:.2f}%'
        }))
        st.markdown(get_csv_download_link(data.tail(100), filename=f"data_preview_{st.session_state.stock_name}.csv", link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)

    # --- Visualisation du Cours ---
    st.subheader(f"Cours historique de {st.session_state.stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture')
        ax.set_title(f'Évolution du cours de {st.session_state.stock_name}')
        ax.set_xlabel('Date'); ax.set_ylabel('Prix (FCFA)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        # Formatteur pour éviter la notation scientifique sur les prix
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Tente de rendre les ticks de date lisibles
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate() # Incliner les labels si nécessaire
        plt.tight_layout(); st.pyplot(fig)
    except Exception as e: st.error(f"Erreur génération graphique cours : {e}")

    # --- Paramètres de la Stratégie ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")
    # Fondamental
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    # Les widgets avec des clés sont automatiquement gérés par session_state
    dividende_annuel = st.sidebar.number_input("Dernier dividende annuel (FCFA)", min_value=0.0, value=600.0, step=10.0, key="dividend")
    taux_croissance = st.sidebar.slider("Croissance annuelle dividende (%)", -10.0, 15.0, 3.0, 0.5, key="growth_rate") / 100
    rendement_exige = st.sidebar.slider("Taux d'actualisation (%)", 5.0, 30.0, 12.0, 0.5, key="discount_rate") / 100

    val_intrinseque = None
    if rendement_exige <= taux_croissance:
        st.sidebar.error("Le taux d'actualisation doit être supérieur au taux de croissance pour le modèle Gordon-Shapiro.")
    else:
        try:
             if dividende_annuel <= 0:
                 val_intrinseque = 0
                 st.sidebar.warning("Dividende annuel <= 0. Valeur Intrinsèque non calculable.")
             else:
                 D1 = dividende_annuel * (1 + taux_croissance)
                 val_intrinseque = D1 / (rendement_exige - taux_croissance)
             if val_intrinseque < 0:
                 st.sidebar.warning(f"VI négative calculée ({val_intrinseque:,.2f} FCFA).")
             elif pd.notna(val_intrinseque):
                 st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")
        except Exception as e:
            st.sidebar.error(f"Erreur calcul VI: {e}")

    # Déterminer si les signaux fondamentaux sont utilisables
    use_fundamental_signals = (val_intrinseque is not None and val_intrinseque > 0)
    if use_fundamental_signals:
         st.sidebar.info("Signaux de Valeur Intrinsèque activés.")
    else:
         st.sidebar.warning("Signaux de Valeur Intrinsèque désactivés (VI invalide ou <= 0).")


    # Technique
    st.sidebar.markdown("**Règles de Trading Techniques**")
    # Les marges sont activées si use_fundamental_signals est True
    if use_fundamental_signals:
        marge_achat = st.sidebar.slider("Marge achat / VI (%)", 0, 50, 20, key="buy_margin") / 100
        marge_vente = st.sidebar.slider("Prime sortie / VI (%)", 0, 50, 10, key="sell_premium") / 100
    else:
        marge_achat = 0
        marge_vente = 0
        st.sidebar.caption("Marges VI désactivées car VI invalide.")

    # Stop Loss / Take Profit
    stop_loss = st.sidebar.slider("Stop Loss / Prix Achat (%)", 1.0, 30.0, 10.0, 0.5, key="stop_loss") / 100
    take_profit = st.sidebar.slider("Take Profit / Prix Achat (%)", 5.0, 100.0, 20.0, 1.0, key="take_profit") / 100

    # Moyennes Mobiles
    st.sidebar.markdown("**Indicateurs Techniques (Moyennes Mobiles)**")
    window_court = st.sidebar.slider("Fenêtre MM Courte (j)", 5, 100, 20, key="short_ma")
    window_long = st.sidebar.slider("Fenêtre MM Longue (j)", 20, 250, 50, key="long_ma")
    if window_court >= window_long:
        st.sidebar.warning("La fenêtre de la MM Courte doit être inférieure à celle de la MM Longue.")

    # Marché
    st.sidebar.markdown("**Paramètres Marché (BRVM)**")
    plafond_variation = st.sidebar.slider("Plafond variation /j (%)", 5.0, 15.0, 7.5, 0.5, key="variation_cap") / 100
    delai_livraison = st.sidebar.slider("Délai livraison (j ouvrés)", 1, 5, 3, key="settlement_days")

    # Backtest
    st.sidebar.subheader("5. Paramètres du Backtest")
    capital_initial = st.sidebar.number_input("Capital initial (FCFA)", 100000, 100000000, 1000000, step=100000, key="initial_capital")
    frais_transaction = st.sidebar.slider("Frais transaction (%)", 0.0, 5.0, 0.5, 0.05, key="commission_rate") / 100
    taux_sans_risque = st.sidebar.slider("Taux sans risque annuel (%)", 0.0, 10.0, 3.0, 0.1, key="risk_free_rate") / 100


    # --- Calculs Techniques et Signaux ---
    st.subheader("Analyse Technique et Signaux")
    # Recalculer les indicateurs et signaux à chaque exécution (car les paramètres peuvent changer)
    try:
        if len(data) < window_long:
            st.warning(f"Pas assez données ({len(data)} jours) pour calculer la MM Longue ({window_long} jours).")
            # On peut continuer mais les MMs longues seront NaN
        elif len(data) < window_court:
             st.warning(f"Pas assez données ({len(data)} jours) pour calculer la MM Courte ({window_court} jours).")
             # On peut continuer mais les MMs seront NaN

        # Calcul MM
        # Utiliser min_periods pour gérer les NaNs au début
        data['MM_Court'] = data['Prix'].rolling(window=window_court, min_periods=1).mean() # min_periods=1 pour avoir des valeurs dès le début si besoin
        data['MM_Long'] = data['Prix'].rolling(window=window_long, min_periods=1).mean()


    except Exception as e: st.error(f"Erreur calcul MM : {e}"); # Ne pas st.stop() ici, juste afficher l'erreur


    # Niveaux Fondamentaux
    # Ces colonnes sont recalculées à chaque fois en fonction des sliders et de la VI
    data['val_intrinseque'] = val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan
    data['prix_achat_fondamental'] = (1 - marge_achat) * val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan
    data['prix_vente_fondamental'] = (1 + marge_vente) * val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan


    # Signaux Techniques MM
    # Recalculer les signaux à chaque fois
    data['signal_technique'] = 0
    # S'assurer que les colonnes MM existent et ne sont pas totalement NaN
    if 'MM_Court' in data.columns and 'MM_Long' in data.columns and not data['MM_Court'].isnull().all() and not data['MM_Long'].isnull().all():
         valid_ma = (data['MM_Court'].notna()) & (data['MM_Long'].notna())
         # Utiliser .shift(1) pour comparer avec la veille
         buy_cond = valid_ma & (data['MM_Court'] > data['MM_Long']) & (data['MM_Court'].shift(1) <= data['MM_Long'].shift(1))
         sell_cond = valid_ma & (data['MM_Court'] < data['MM_Long']) & (data['MM_Court'].shift(1) >= data['MM_Long'].shift(1))
         data.loc[buy_cond, 'signal_technique'] = 1
         data.loc[sell_cond, 'signal_technique'] = -1
    else:
         st.warning("Calcul des signaux techniques (MM) impossible car les Moyennes Mobiles ne sont pas valides.")


    # Signaux Combinés Achat/Vente
    # Recalculer les signaux combinés
    cond_achat_tech = (data['signal_technique'] == 1)
    cond_vente_tech = (data['signal_technique'] == -1)

    if use_fundamental_signals:
        # Assurer que les colonnes fondamentales existent et ne sont pas totalement NaN avant de les utiliser dans les conditions
        if 'prix_achat_fondamental' in data.columns and 'prix_vente_fondamental' in data.columns and \
           not data['prix_achat_fondamental'].isnull().all() and not data['prix_vente_fondamental'].isnull().all():

            # Gérer les NaNs potentiels dans les colonnes fondamentales lors de la comparaison
            # On utilise .fillna(method='ffill') ou .fillna(limit=...) si nécessaire, ou simplement s'assurer que les NaN sont traités
            # Ici, on utilise & avec les conditions pour que les lignes avec NaN dans les seuils fondamentaux ne déclenchent pas de signaux
            cond_achat_fond = (data['Prix'] < data['prix_achat_fondamental']) & data['prix_achat_fondamental'].notna()
            cond_vente_fond = (data['Prix'] > data['prix_vente_fondamental']) & data['prix_vente_fondamental'].notna()

            data['achat'] = cond_achat_tech & cond_achat_fond
            # Signal de vente est soit technique, soit fondamental
            data['vente_signal'] = cond_vente_tech | cond_vente_fond
        else:
             st.warning("Signaux fondamentaux désactivés car les seuils calculés sont invalides (NaN).")
             data['achat'] = cond_achat_tech # Revenir aux signaux techniques uniquement
             data['vente_signal'] = cond_vente_tech
             use_fundamental_signals = False # Désactiver pour la suite si les seuils sont inutilisables
    else:
        data['achat'] = cond_achat_tech
        data['vente_signal'] = cond_vente_tech

    # Graphique MM
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', lw=1, alpha=0.8, zorder=2)
        # Afficher les MMs seulement si elles ont été calculées (pas entièrement NaN)
        if 'MM_Court' in data.columns and not data['MM_Court'].isnull().all():
             ax2.plot(data.index[data['MM_Court'].first_valid_index():], data['MM_Court'].dropna(), label=f'MM {window_court}j', lw=1.5, zorder=3)
        if 'MM_Long' in data.columns and not data['MM_Long'].isnull().all():
             ax2.plot(data.index[data['MM_Long'].first_valid_index():], data['MM_Long'].dropna(), label=f'MM {window_long}j', lw=1.5, zorder=3)

        ax2.set_title('Analyse Technique - Moyennes Mobiles'); ax2.set_xlabel('Date'); ax2.set_ylabel('Prix (FCFA)')
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=1); ax2.legend()
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig2.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig2)
    except Exception as e: st.error(f"Erreur graphique MM : {e}")

    # Graphique Signaux
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', lw=1.5, zorder=2)

        if use_fundamental_signals and val_intrinseque is not None and val_intrinseque > 0:
            ax3.axhline(y=val_intrinseque, color='grey', ls='-', alpha=0.7, label=f'VI ({val_intrinseque:,.0f})', zorder=1)
            # Afficher les seuils fondamentaux s'ils sont valides (non NaN)
            if pd.notna(data['prix_achat_fondamental'].iloc[0]):
                ax3.axhline(y=data['prix_achat_fondamental'].iloc[0], color='green', ls='--', alpha=0.6, label=f'Seuil Achat VI ({data["prix_achat_fondamental"].iloc[0]:,.0f})', zorder=1)
            if pd.notna(data['prix_vente_fondamental'].iloc[0]):
                ax3.axhline(y=data['prix_vente_fondamental'].iloc[0], color='red', ls='--', alpha=0.6, label=f'Seuil Vente VI ({data["prix_vente_fondamental"].iloc[0]:,.0f})', zorder=1)


        # Afficher les signaux calculés (qui dépendent des conditions combinées)
        achats_sig = data[data['achat']]
        ventes_sig = data[data['vente_signal']]
        if not achats_sig.empty: ax3.scatter(achats_sig.index, achats_sig['Prix'], color='lime', edgecolor='green', s=70, marker='^', label='Signal Achat Strat', zorder=5)
        if not ventes_sig.empty: ax3.scatter(ventes_sig.index, ventes_sig['Prix'], color='tomato', edgecolor='red', s=70, marker='v', label='Signal Vente Strat', zorder=5)

        ax3.set_title('Prix et Signaux Trading Initiaux'); ax3.set_xlabel('Date'); ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, linestyle='--', alpha=0.6, zorder=1); ax3.legend(loc='best')
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig3.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig3)
    except Exception as e: st.error(f"Erreur graphique signaux : {e}"); st.error(traceback.format_exc())

    # --- Backtest ---
    st.subheader("🚀 Backtest de la Stratégie")
    st.markdown(f"Capital: **{capital_initial:,.0f} FCFA**, Frais: **{frais_transaction*100:.2f}%**, Plafond: **{plafond_variation*100:.1f}%**, Livraison: **{delai_livraison}j**.")

    # Fonction Backtest (Identique, s'assure de prendre les données calculées)
    def run_backtest(data_for_backtest, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison):
        # data_for_backtest est déjà la copie de data avec les signaux et MMs calculés
        if data_for_backtest.empty or not isinstance(data_for_backtest.index, pd.DatetimeIndex):
             st.error("Données invalides pour le backtest."); return pd.DataFrame(), [], [], pd.DataFrame()

        portfolio = pd.DataFrame(index=data_for_backtest.index)
        # Initialiser les colonnes avec les valeurs de départ
        portfolio['prix_effectif'] = 0.0 # Sera rempli jour par jour
        portfolio['actions'] = 0.0
        portfolio['cash'] = float(capital_initial)
        portfolio['valeur_actions'] = 0.0
        portfolio['valeur_totale'] = float(capital_initial)
        portfolio['rendement'] = 0.0
        portfolio['trade_en_cours'] = False
        portfolio['date_livraison_prevue'] = pd.NaT # Type NaT pour les dates
        portfolio['prix_achat_moyen'] = 0.0
        portfolio['stop_loss_price'] = np.nan # Prix Stop Loss pour le trade en cours
        portfolio['take_profit_price'] = np.nan # Prix Take Profit pour le trade en cours


        transactions = [] # Liste pour enregistrer les transactions
        achats_dates = [] # Liste pour marquer les dates d'achat sur le graphique
        ventes_dates = [] # Liste pour marquer les dates de vente sur le graphique

        # Variables pour l'état du portefeuille au jour le jour
        nb_actions_possedees = 0.0
        cash_disponible = float(capital_initial)
        prix_achat_moyen_actif = 0.0 # Prix moyen des actions détenues
        trade_en_cours_boucle = False # Indicateur si une position est ouverte
        date_livraison_boucle = pd.NaT # Date de fin du délai de livraison
        stop_loss_actif = np.nan # Niveau Stop Loss pour la position ouverte
        take_profit_actif = np.nan # Niveau Take Profit pour la position ouverte


        bday = BDay() # Business Day offset pour le délai de livraison

        for i, (jour, row) in enumerate(data_for_backtest.iterrows()):
            # --- Début de Journée ---
            prix_jour_brut = row['Prix']

            # Appliquer le plafond de variation par rapport au prix effectif de la VEILLE
            prix_veille_eff = portfolio.loc[data_for_backtest.index[i-1], 'prix_effectif'] if i > 0 else prix_jour_brut # Si c'est le 1er jour, pas de prix veille, utiliser le prix du jour

            # Éviter division par zéro si prix_veille_eff est 0
            if prix_veille_eff == 0:
                 variation = 0
            else:
                 variation = (prix_jour_brut - prix_veille_eff) / prix_veille_eff

            prix_effectif_jour = prix_jour_brut # Par défaut, le prix effectif est le prix brut

            if abs(variation) > plafond_variation:
                 # Calculer le prix plafonné
                 prix_effectif_jour = prix_veille_eff * (1 + (np.sign(variation) * plafond_variation))
                 # Optionnel: log ici pour voir quand le plafond est appliqué

            # --- Mettre à Jour le Portefeuille avec les valeurs de la veille pour le jour actuel ---
            # Pour la première ligne, on utilise les valeurs initiales
            if i > 0:
                 jour_prec = data_for_backtest.index[i-1]
                 # On prend les valeurs finales du jour précédent comme valeurs d'ouverture du jour actuel
                 nb_actions_possedees = portfolio.loc[jour_prec, 'actions']
                 cash_disponible = portfolio.loc[jour_prec, 'cash']
                 trade_en_cours_boucle = portfolio.loc[jour_prec, 'trade_en_cours']
                 date_livraison_boucle = portfolio.loc[jour_prec, 'date_livraison_prevue']
                 prix_achat_moyen_actif = portfolio.loc[jour_prec, 'prix_achat_moyen']
                 stop_loss_actif = portfolio.loc[jour_prec, 'stop_loss_price']
                 take_profit_actif = portfolio.loc[jour_prec, 'take_profit_price']
            # else: # Pour i=0, les valeurs sont déjà celles initialisées (0, capital_initial, False, NaT, 0)
                 # print(f"Jour 0: Cash: {cash_disponible}, Actions: {nb_actions_possedees}") # Debug initial


            # --- Gérer le Délai de Livraison ---
            # Les actions ne peuvent être vendues qu'après la date de livraison
            actions_disponibles_vente = nb_actions_possedees if jour >= date_livraison_boucle else 0

            # --- Conditions de Vente ---
            vente_trigger = False
            vente_raison = None

            # 1. Stop Loss (vérifier SEULEMENT s'il y a un trade en cours ET actions disponibles pour la vente)
            if trade_en_cours_boucle and actions_disponibles_vente > 0 and pd.notna(stop_loss_actif) and prix_effectif_jour <= stop_loss_actif:
                 vente_trigger = True
                 vente_raison = "Stop Loss"
                 # print(f"{jour.date()} - VENTE SL: Prix {prix_effectif_jour:.2f} <= SL {stop_loss_actif:.2f}") # Debug

            # 2. Take Profit (vérifier SEULEMENT s'il y a un trade en cours ET actions disponibles pour la vente et pas déjà vendu par SL)
            if not vente_trigger and trade_en_cours_boucle and actions_disponibles_vente > 0 and pd.notna(take_profit_actif) and prix_effectif_jour >= take_profit_actif:
                 vente_trigger = True
                 vente_raison = "Take Profit"
                 # print(f"{jour.date()} - VENTE TP: Prix {prix_effectif_jour:.2f} >= TP {take_profit_actif:.2f}") # Debug

            # 3. Signal de vente de la stratégie (vérifier si pas déjà vendu par SL/TP)
            if not vente_trigger and row['vente_signal'] and actions_disponibles_vente > 0:
                 vente_trigger = True
                 vente_raison = "Signal Stratégie"
                 # print(f"{jour.date()} - VENTE SIGNAL: Prix {prix_effectif_jour:.2f}") # Debug


            if vente_trigger:
                if nb_actions_possedees > 0: # S'assurer qu'on a bien des actions à vendre
                     montant_vente = nb_actions_possedees * prix_effectif_jour
                     frais = montant_vente * frais_transaction
                     cash_obtenu = montant_vente - frais
                     cash_disponible += cash_obtenu

                     transactions.append({
                         'Date': jour,
                         'Type': 'Vente',
                         'Quantité': nb_actions_possedees,
                         'Prix_Unitaire': prix_effectif_jour,
                         'Montant': montant_vente,
                         'Frais': frais,
                         'Cash_Net': cash_obtenu,
                         'Raison': vente_raison,
                         'Prix_Achat_Moyen': prix_achat_moyen_actif # Ajouter le PMA au moment de la vente
                     })
                     ventes_dates.append(jour) # Marquer la date de vente

                     # Réinitialiser l'état après la vente
                     nb_actions_possedees = 0.0
                     prix_achat_moyen_actif = 0.0
                     trade_en_cours_boucle = False
                     date_livraison_boucle = pd.NaT # Plus de date de livraison en attente
                     stop_loss_actif = np.nan
                     take_profit_actif = np.nan
                     # print(f"{jour.date()} - Vente exécutée. Cash: {cash_disponible:.2f}, Actions: {nb_actions_possedees}") # Debug


            # --- Conditions d'Achat ---
            # On achète seulement s'il n'y a pas de trade en cours ET qu'il y a un signal d'achat ET qu'on a du cash
            if not trade_en_cours_boucle and row['achat'] and cash_disponible > 0:
                # print(f"{jour.date()} - Achat potentiel: Cash {cash_disponible:.2f}, Prix {prix_effectif_jour:.2f}") # Debug
                 # Calculer combien d'actions on peut acheter
                 # Le montant maximum à investir est le cash disponible moins les frais potentiels
                 # montant_max_invest = cash_disponible / (1 + frais_transaction)
                 # nb_actions_a_acheter = np.floor(montant_max_invest / prix_effectif_jour)

                 # Simplification: investir tout le cash disponible, les frais sont calculés après
                 nb_actions_a_acheter = np.floor(cash_disponible / prix_effectif_jour)

                 if nb_actions_a_acheter >= 1: # S'assurer qu'on peut acheter au moins une action
                     montant_achat = nb_actions_a_acheter * prix_effectif_jour
                     frais = montant_achat * frais_transaction
                     if (montant_achat + frais) <= cash_disponible: # Vérifier qu'on a assez de cash AVEC les frais
                         cash_disponible -= (montant_achat + frais)
                         # Mettre à jour le prix d'achat moyen si on achetait par paliers.
                         # Ici, on achète tout en une fois, donc le PMA est juste le prix actuel.
                         # Si on faisait du DCA ou des achats partiels, il faudrait pondérer.
                         prix_achat_moyen_actif = prix_effectif_jour # Nouveau PMA après cet achat
                         nb_actions_possedees += nb_actions_a_acheter

                         transactions.append({
                             'Date': jour,
                             'Type': 'Achat',
                             'Quantité': nb_actions_a_acheter,
                             'Prix_Unitaire': prix_effectif_jour,
                             'Montant': montant_achat,
                             'Frais': frais,
                             'Cash_Net': -(montant_achat + frais), # Coût total de l'achat
                             'Raison': 'Signal Stratégie',
                             'Prix_Achat_Moyen': prix_achat_moyen_actif # Enregistrer le PMA
                         })
                         achats_dates.append(jour) # Marquer la date d'achat

                         # Activer le trade et calculer les niveaux de sortie
                         trade_en_cours_boucle = True
                         # Date de livraison = jour de l'achat + délai de livraison (en jours ouvrés)
                         date_livraison_boucle = jour + BDay(delai_livraison) # Utilise l'objet BDay

                         # Calculer les prix Stop Loss et Take Profit basés sur le prix d'achat EFFECTIF
                         stop_loss_actif = prix_effectif_jour * (1 - stop_loss)
                         take_profit_actif = prix_effectif_jour * (1 + take_profit)

                         # print(f"{jour.date()} - Achat exécuté. Cash: {cash_disponible:.2f}, Actions: {nb_actions_possedees}, PMA: {prix_achat_moyen_actif:.2f}, SL: {stop_loss_actif:.2f}, TP: {take_profit_actif:.2f}, Livraison: {date_livraison_boucle.date()}") # Debug
                     else:
                         st.warning(f"{jour.date()}: Pas assez de cash pour acheter {nb_actions_a_acheter:,.0f} actions ({montant_achat + frais:,.2f} FCFA nécessaires). Cash disponible: {cash_disponible:,.2f}.")
                         # print(f"{jour.date()} - Achat impossible: Cash insuffisant") # Debug

            # --- Fin de Journée ---
            # Mettre à jour les valeurs du portefeuille pour le jour actuel
            portfolio.loc[jour, 'actions'] = nb_actions_possedees
            portfolio.loc[jour, 'cash'] = cash_disponible
            portfolio.loc[jour, 'valeur_actions'] = nb_actions_possedees * prix_effectif_jour
            portfolio.loc[jour, 'valeur_totale'] = portfolio.loc[jour, 'cash'] + portfolio.loc[jour, 'valeur_actions']
            portfolio.loc[jour, 'trade_en_cours'] = trade_en_cours_boucle
            portfolio.loc[jour, 'date_livraison_prevue'] = date_livraison_boucle
            portfolio.loc[jour, 'prix_achat_moyen'] = prix_achat_moyen_actif
            portfolio.loc[jour, 'stop_loss_price'] = stop_loss_actif
            portfolio.loc[jour, 'take_profit_price'] = take_profit_actif

        # Calcul du rendement
        portfolio['rendement'] = portfolio['valeur_totale'].pct_change().fillna(0)

        # Si un trade est toujours en cours à la fin, simuler la vente au dernier prix effectif
        if nb_actions_possedees > 0:
            dernier_jour = data_for_backtest.index[-1]
            prix_dernier_jour = portfolio.loc[dernier_jour, 'prix_effectif']
            montant_vente = nb_actions_possedees * prix_dernier_jour
            frais = montant_vente * frais_transaction
            cash_obtenu = montant_vente - frais
            cash_disponible_final = portfolio.loc[dernier_jour, 'cash'] + cash_obtenu

            transactions.append({
                'Date': dernier_jour,
                'Type': 'Vente Fin', # Marquer comme vente de fin de backtest
                'Quantité': nb_actions_possedees,
                'Prix_Unitaire': prix_dernier_jour,
                'Montant': montant_vente,
                'Frais': frais,
                'Cash_Net': cash_obtenu,
                'Raison': 'Fin du Backtest',
                'Prix_Achat_Moyen': prix_achat_moyen_actif
            })
            # Mettre à jour la dernière ligne du portefeuille avec la vente simulée pour le calcul final
            portfolio.loc[dernier_jour, 'actions'] = 0.0
            portfolio.loc[dernier_jour, 'cash'] = cash_disponible_final
            portfolio.loc[dernier_jour, 'valeur_actions'] = 0.0
            portfolio.loc[dernier_jour, 'valeur_totale'] = cash_disponible_final # La valeur totale est maintenant tout en cash
            portfolio.loc[dernier_jour, 'trade_en_cours'] = False
            portfolio.loc[dernier_jour, 'date_livraison_prevue'] = pd.NaT
            portfolio.loc[dernier_jour, 'prix_achat_moyen'] = 0.0
            portfolio.loc[dernier_jour, 'stop_loss_price'] = np.nan
            portfolio.loc[dernier_jour, 'take_profit_price'] = np.nan
            # Recalculer le rendement pour la dernière journée
            portfolio.loc[dernier_jour, 'rendement'] = (portfolio.loc[dernier_jour, 'valeur_totale'] - portfolio.loc[data_for_backtest.index[-2], 'valeur_totale']) / portfolio.loc[data_for_backtest.index[-2], 'valeur_totale'] if len(data_for_backtest) > 1 else 0

        # Créer le DataFrame de transactions
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
             transactions_df = transactions_df.set_index('Date')

        return portfolio, achats_dates, ventes_dates, transactions_df

    # Lancer le Backtest
    try:
        portfolio_df, achat_dates_vis, vente_dates_vis, transactions_df = run_backtest(
            data, # Utiliser la copie de data qui contient les signaux
            capital_initial,
            frais_transaction,
            stop_loss,
            take_profit,
            plafond_variation,
            delai_livraison
        )

        if not portfolio_df.empty:
            st.success("Backtest terminé.")

            # --- Résultats du Backtest ---
            st.subheader("Synthèse des Résultats")
            valeur_finale = portfolio_df['valeur_totale'].iloc[-1] if not portfolio_df.empty else capital_initial
            gain_perte = valeur_finale - capital_initial
            rendement_pourcentage = (gain_perte / capital_initial) * 100 if capital_initial > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Capital Initial", f"{capital_initial:,.0f} FCFA")
            col2.metric("Valeur Finale", f"{valeur_finale:,.2f} FCFA")
            col3.metric("Gain / Perte Net", f"{gain_perte:,.2f} FCFA", delta=f"{rendement_pourcentage:.2f}%")
            col4.metric("Nombre de Trades", len([t for t in transactions_df['Type'] if t in ['Achat', 'Vente']]))


            # --- Indicateurs de Performance (Simplifiés) ---
            st.subheader("Indicateurs de Performance")
            if not portfolio_df.empty and capital_initial > 0:
                # Calcul du rendement cumulé
                portfolio_df['rendement_cumule'] = (portfolio_df['valeur_totale'] / capital_initial) - 1

                # Drawdown Maximum (exemple simple)
                # Valeur cumulée max jusqu'à présent
                portfolio_df['valeur_max'] = portfolio_df['valeur_totale'].cummax()
                # Drawdown journalier en %
                portfolio_df['drawdown'] = (portfolio_df['valeur_totale'] - portfolio_df['valeur_max']) / portfolio_df['valeur_max']
                max_drawdown = portfolio_df['drawdown'].min() * 100 if not portfolio_df['drawdown'].empty else 0

                # Nombre de jours de trading
                nb_jours_trading = len(portfolio_df)
                if nb_jours_trading > 0:
                     # Rendement annualisé (très simple, basé sur rendement total)
                     # Nécessite au moins un an de données pour être pertinent
                     nb_annees = nb_jours_trading / 252.0 # Environ 252 jours de trading par an
                     rendement_annualise = ((1 + rendement_pourcentage/100)**(1/nb_annees) - 1) * 100 if nb_annees > 0 else 0

                     # Volatilité Annualisée (simplifiée)
                     # Utilise l'écart type des rendements journaliers
                     volatilite_journaliere = portfolio_df['rendement'].std()
                     volatilite_annualisee = volatilite_journaliere * np.sqrt(252) * 100 if volatilite_journaliere > 0 else 0

                     # Taux sans risque journalier
                     taux_sans_risque_journalier = (1 + taux_sans_risque)**(1/252) - 1

                     # Sharpe Ratio (simplifié)
                     # (Rendement moyen journalier - Taux sans risque journalier) / Volatilité journalière
                     rendement_moyen_journalier = portfolio_df['rendement'].mean()
                     sharpe_ratio = (rendement_moyen_journalier - taux_sans_risque_journalier) / volatilite_journaliere if volatilite_journaliere > 0 else np.nan


                col1, col2, col3, col4_sharpe = st.columns(4)
                col1.metric("Rendement Annualisé (estimé)", f"{rendement_annualise:.2f}%" if nb_jours_trading > 0 else "N/A")
                col2.metric("Volatilité Annualisée (estimée)", f"{volatilite_annualisee:.2f}%" if nb_jours_trading > 0 and not np.isnan(volatilite_annualisee) else "N/A")
                col3.metric("Drawdown Max (%)", f"{max_drawdown:.2f}%")
                col4_sharpe.metric("Sharpe Ratio (estimé)", f"{sharpe_ratio:.2f}" if nb_jours_trading > 0 and not np.isnan(sharpe_ratio) else "N/A")


            # --- Graphique de la Valeur du Portefeuille ---
            st.subheader("Évolution de la Valeur du Portefeuille")
            try:
                fig_port, ax_port = plt.subplots(figsize=(12, 6))
                ax_port.plot(portfolio_df.index, portfolio_df['valeur_totale'], label='Valeur du Portefeuille', lw=2)
                # Optionnel: ajouter le cours de l'action normalisé ou l'indice BRVM pour comparaison
                ax_port.set_title('Évolution de la Valeur du Portefeuille'); ax_port.set_xlabel('Date'); ax_port.set_ylabel('Valeur (FCFA)')
                ax_port.grid(True, linestyle='--', alpha=0.6); ax_port.legend()
                ax_port.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax_port.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_port.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig_port.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig_port)
            except Exception as e: st.error(f"Erreur graphique portefeuille : {e}")

            # --- Graphique des Rendements Cumulés ---
            st.subheader("Rendement Cumulé (%)")
            try:
                fig_ret, ax_ret = plt.subplots(figsize=(12, 6))
                # Convertir le rendement cumulé en pourcentage pour l'affichage
                ax_ret.plot(portfolio_df.index, portfolio_df['rendement_cumule'] * 100, label='Rendement Cumulé (%)', lw=2)
                ax_ret.axhline(y=0, color='grey', linestyle='--') # Ligne de zéro
                ax_ret.set_title('Rendement Cumulé du Portefeuille'); ax_ret.set_xlabel('Date'); ax_ret.set_ylabel('Rendement (%)')
                ax_ret.grid(True, linestyle='--', alpha=0.6); ax_ret.legend()
                # Formatter l'axe Y en pourcentage
                ax_ret.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f}%'))
                ax_ret.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_ret.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig_ret.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig_ret)
            except Exception as e: st.error(f"Erreur graphique rendement : {e}")


            # --- Graphique des Transactions sur le Cours ---
            st.subheader("Transactions sur le Cours")
            try:
                fig_trades, ax_trades = plt.subplots(figsize=(12, 6))
                ax_trades.plot(data.index, data['Prix'], label='Prix de Clôture', lw=1.5, alpha=0.7, zorder=2)

                # Ajouter les points d'achat et de vente basés sur les dates enregistrées
                # S'assurer que les dates existent bien dans l'index du DataFrame 'data' pour le plotting
                achats_plot_df = data.loc[achat_dates_vis]
                ventes_plot_df = data.loc[vente_dates_vis]

                if not achats_plot_df.empty:
                     ax_trades.scatter(achats_plot_df.index, achats_plot_df['Prix'], color='lime', edgecolor='green', s=100, marker='^', label='Achat', zorder=5)
                if not ventes_plot_df.empty:
                     ax_trades.scatter(ventes_plot_df.index, ventes_plot_df['Prix'], color='tomato', edgecolor='red', s=100, marker='v', label='Vente', zorder=5)

                ax_trades.set_title('Cours avec Transactions'); ax_trades.set_xlabel('Date'); ax_trades.set_ylabel('Prix (FCFA)')
                ax_trades.grid(True, linestyle='--', alpha=0.6, zorder=1); ax_trades.legend()
                ax_trades.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax_trades.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_trades.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig_trades.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig_trades)
            except Exception as e: st.error(f"Erreur graphique transactions : {e}"); st.error(traceback.format_exc())


            # --- Tableau des Transactions ---
            st.subheader("Détail des Transactions")
            if not transactions_df.empty:
                 # Formatter les colonnes numériques pour l'affichage
                 formatted_transactions = transactions_df.copy()
                 num_cols_to_format = ['Quantité', 'Prix_Unitaire', 'Montant', 'Frais', 'Cash_Net', 'Prix_Achat_Moyen']
                 format_dict = {col: '{:,.2f}' for col in num_cols_to_format}
                 format_dict['Quantité'] = '{:,.0f}' # Quantité en entier

                 st.dataframe(formatted_transactions.style.format(format_dict))

                 # Lien de téléchargement
                 st.markdown(get_csv_download_link(transactions_df, filename=f"transactions_{st.session_state.stock_name}.csv", link_text="Télécharger les transactions (CSV)"), unsafe_allow_html=True)
            else:
                 st.info("Aucune transaction effectuée pendant le backtest.")


        else:
            st.warning("Impossible d'exécuter le backtest avec les données fournies.")

    except Exception as e:
        st.error(f"Erreur lors de l'exécution du backtest : {e}")
        st.error(traceback.format_exc()) # Afficher la pile d'appels
