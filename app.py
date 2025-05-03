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
import csv
import traceback

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="BRVM Quant Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help', # Mettez votre lien d'aide
        'Report a bug': "https://www.example.com/bug", # Mettez votre lien de bug
        'About': """
        ## BRVM Quant Backtest App
        **Version:** 1.7.0 (Correction Erreurs + Logique Backtest)

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
# Utilisation de .setdefault pour éviter les erreurs si la clé existe déjà
# et pour initialiser si elle n'existe pas.
default_values = {
    'uploaded_file_obj': None,
    'all_columns': [],
    'column_mapping': {"Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""},
    'date_format_input': "",
    'data': pd.DataFrame(),
    'stock_name': "MonActionBRVM",
    'dividend': 600.0,
    'growth_rate': 3.0,
    'discount_rate': 12.0,
    'use_mm_signal': True,
    'short_ma': 20,
    'long_ma': 50,
    'use_rsi_signal': True,
    'rsi_window': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'use_macd_signal': True,
    'macd_fast_window': 12,
    'macd_slow_window': 26,
    'macd_signal_window': 9,
    'tech_signal_method': "MM OU RSI OU MACD",
    'buy_margin': 20.0,
    'sell_premium': 10.0,
    'stop_loss': 10.0,
    'take_profit': 20.0,
    'use_trailing_stop': True,
    'trailing_stop_pct': 5.0,
    'variation_cap': 7.5,
    'settlement_days': 3,
    'initial_capital': 1000000.0,
    'commission_rate': 0.5,
    'risk_free_rate': 3.0,
    'invest_percentage': 100.0
}

for key, value in default_values.items():
    st.session_state.setdefault(key, value)


# --- Titre et Introduction ---
st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")
st.sidebar.header("Paramètres Globaux")

# --- Section Upload de Fichier ---
st.sidebar.subheader("1. Chargement des Données")

def handle_upload():
    """Réinitialise l'état quand un nouveau fichier est chargé."""
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

# Utiliser l'objet fichier stocké dans session_state
current_uploaded_file = st.session_state.uploaded_file_obj

# --- Fonction de Traitement des Données ---
def process_data(file, column_mapping, date_format=None):
    """
    Charge, nettoie, et standardise les données OHLCV depuis un fichier CSV.
    Gère la détection de séparateur, l'encodage, la conversion de types,
    les doublons, et les valeurs manquantes.
    """
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
        sample_bytes = file.read(2048) # Lire un échantillon pour l'analyse
        file.seek(0) # Rembobiner le fichier pour la lecture par Pandas

        # Essayer de décoder en UTF-8, sinon Latin-1
        try:
            sample_text = sample_bytes.decode('utf-8')
            encoding = 'utf-8'
        except UnicodeDecodeError:
            try:
                sample_text = sample_bytes.decode('latin-1')
                encoding = 'latin-1'
                st.info("Détection de l'encodage Latin-1.")
            except Exception as e:
                 st.error(f"Impossible de décoder l'échantillon du fichier avec UTF-8 ou Latin-1: {e}")
                 return None

        # Détection du séparateur
        separator = ',' # Défaut
        try:
            if sample_text.strip(): # S'assurer qu'il y a du contenu
                # Utiliser Sniffer pour détecter le dialecte (séparateur, etc.)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
                st.info(f"Séparateur détecté : '{separator}'")
            else:
                st.warning("L'échantillon du fichier semble vide.")
        except csv.Error:
            # Heuristique simple si Sniffer échoue
            if sample_text and sample_text.count(';') >= sample_text.count(','):
                separator = ';'
                st.info(f"Sniffer a échoué, utilisation de l'heuristique. Séparateur probable : '{separator}'")
            else:
                 st.info(f"Sniffer a échoué, utilisation du séparateur par défaut : '{separator}'")


        # Lecture du CSV avec Pandas
        file.seek(0) # Rembobiner à nouveau
        try:
            df = pd.read_csv(file, sep=separator, encoding=encoding, low_memory=False)
        except Exception as read_err:
            st.error(f"Erreur lors de la lecture du fichier CSV avec pandas (sep='{separator}', encoding='{encoding}') : {read_err}")
            st.error(traceback.format_exc())
            return None

        if df.empty:
            st.error("Le fichier CSV est vide ou n'a pas pu être lu correctement par Pandas.")
            return None

        st.info(f"Fichier lu. {len(df)} lignes et {len(df.columns)} colonnes trouvées.")

        # --- Standardisation et Nettoyage ---
        df_standardized = pd.DataFrame()
        original_row_count = len(df)

        # 1. Conversion de la Date
        date_col_name = column_mapping['Date']
        if date_col_name not in df.columns:
             st.error(f"La colonne Date mappée '{date_col_name}' n'existe pas dans le fichier.")
             return None
        try:
            # Tentative de conversion automatique
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce', infer_datetime_format=True)

            # Si la conversion auto échoue et qu'un format est fourni, essayer avec le format
            if df_standardized['Date'].isnull().all() and date_format:
                st.info(f"La conversion automatique de la date a échoué. Tentative avec le format explicite : '{date_format}'")
                try:
                    # Utiliser une copie pour éviter SettingWithCopyWarning
                    df_copy = df[[date_col_name]].copy()
                    df_standardized['Date'] = pd.to_datetime(df_copy[date_col_name], format=date_format, errors='coerce')
                except Exception as fmt_e:
                    st.error(f"Erreur lors de l'application du format de date '{date_format}' à la colonne '{date_col_name}': {fmt_e}")
                    return None
        except Exception as e:
            st.error(f"Erreur générale lors de la conversion de la colonne Date ('{date_col_name}'): {e}")
            st.error(traceback.format_exc())
            return None

        # Vérifier les NaN après conversion de date
        if df_standardized['Date'].isnull().all():
            st.error(f"Impossible de convertir la colonne Date ('{date_col_name}') en dates valides, même avec le format optionnel. Vérifiez la colonne et le format.")
            return None
        if df_standardized['Date'].isnull().any():
            nan_dates_count = df_standardized['Date'].isnull().sum()
            st.warning(f"{nan_dates_count} valeur(s) dans la colonne Date ('{date_col_name}') sont invalides ou vides. Les lignes correspondantes seront supprimées.")
            # Associer les NaNs de date au DataFrame original pour suppression
            df = df[df_standardized['Date'].notna()]
            df_standardized = df_standardized.dropna(subset=['Date'])
            if df_standardized.empty:
                st.error("Toutes les lignes ont été supprimées après l'échec de la conversion des dates.")
                return None
            st.info(f"{original_row_count - len(df_standardized)} lignes supprimées en raison de dates invalides.")

        # 2. Conversion Numérique (Open, High, Low, Close, Volume)
        standard_to_user_map = {
            'Ouverture': column_mapping['Open'], 'Plus_Haut': column_mapping['High'],
            'Plus_Bas': column_mapping['Low'], 'Prix': column_mapping['Close'], # 'Prix' est notre nom standard pour Close
            'Volume': column_mapping['Volume']
        }

        for standard_col_name, user_col_name in standard_to_user_map.items():
            if user_col_name not in df.columns:
                 st.error(f"La colonne mappée '{user_col_name}' pour '{standard_col_name}' n'existe pas dans le fichier.")
                 return None
            try:
                # Nettoyage robuste pour les colonnes numériques (gestion des espaces, virgules comme décimales)
                if df[user_col_name].dtype == 'object':
                    # 1. Supprimer les espaces blancs au début/fin
                    cleaned_series = df[user_col_name].astype(str).str.strip()
                    # 2. Remplacer les virgules par des points pour les décimales
                    cleaned_series = cleaned_series.str.replace(',', '.', regex=False)
                    # 3. Supprimer les espaces internes (ex: séparateurs de milliers)
                    cleaned_series = cleaned_series.str.replace(r'\s+', '', regex=True)
                    # 4. Tentative de conversion en numérique
                    converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                    # Si la conversion échoue toujours, tenter un nettoyage plus agressif
                    # (supprimer tout sauf chiffres, point, signe moins au début)
                    if converted_series.isnull().all() and not df[user_col_name].isnull().all():
                        st.info(f"Nettoyage avancé pour la colonne '{user_col_name}'...")
                        # Garde seulement chiffres, point, et un éventuel '-' au début
                        cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                        # Gère les cas comme '.' ou '-' seuls qui ne sont pas valides
                        cleaned_series = cleaned_series.str.replace(r'^(-?\.)?$', '', regex=True)
                        # Gère les cas comme '12-34' -> '1234' (enlève les '-' non initiaux)
                        cleaned_series = cleaned_series.str.replace(r'(?<=\d)-', '', regex=True)
                        converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                    df_standardized[standard_col_name] = converted_series
                else:
                    # Si déjà numérique, convertir directement (gère les types entiers/flottants)
                    df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                # Vérifier les NaNs introduits par la conversion
                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                original_nan = df[user_col_name].isnull().sum() # NaN originaux dans la colonne user
                newly_created_nan = nan_after_conversion - original_nan

                if newly_created_nan > 0:
                    st.warning(f"{newly_created_nan} valeur(s) dans '{user_col_name}' n'ont pas pu être converties en nombre et sont devenues NaN (en plus des {original_nan} NaN d'origine).")

            except Exception as e:
                st.error(f"Erreur lors de la conversion numérique de la colonne '{user_col_name}' (mappée à '{standard_col_name}') : {e}")
                st.error(traceback.format_exc())
                return None

        # Supprimer les lignes où les colonnes OHLC sont NaN APRES conversion
        ohlc_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix']
        rows_before_ohlc_nan_drop = len(df_standardized)
        df_standardized.dropna(subset=ohlc_cols, inplace=True)
        rows_after_ohlc_nan_drop = len(df_standardized)
        if rows_before_ohlc_nan_drop > rows_after_ohlc_nan_drop:
             st.warning(f"{rows_before_ohlc_nan_drop - rows_after_ohlc_nan_drop} lignes supplémentaires supprimées car les valeurs Open/High/Low/Close n'étaient pas numériques.")

        if df_standardized.empty:
            st.error("Toutes les lignes ont été supprimées après nettoyage des valeurs OHLC non numériques.")
            return None

        # 3. Tri par Date et Gestion des Doublons
        df_standardized = df_standardized.sort_values('Date')
        if df_standardized['Date'].duplicated().any():
            duplicates_count = df_standardized['Date'].duplicated().sum()
            st.warning(f"Il y a {duplicates_count} dates dupliquées dans les données. Seule la dernière entrée pour chaque date sera conservée.")
            df_standardized = df_standardized.drop_duplicates(subset=['Date'], keep='last')

        # 4. Définir l'Index sur la Date
        df_standardized = df_standardized.set_index('Date')

        # 5. Remplissage des NaN restants (Forward Fill puis Backward Fill)
        # Appliquer seulement aux colonnes où c'est logique (OHLCV)
        cols_to_fill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        for col in cols_to_fill:
            if col in df_standardized.columns:
                nan_before = df_standardized[col].isnull().sum()
                if nan_before > 0:
                    df_standardized[col] = df_standardized[col].ffill() # Remplir avec la valeur précédente
                    df_standardized[col] = df_standardized[col].bfill() # Remplir les NaN restants (au début) avec la valeur suivante
                    nan_after = df_standardized[col].isnull().sum()
                    if nan_after < nan_before:
                        st.info(f"{nan_before - nan_after} NaN dans la colonne '{col}' ont été remplis en utilisant les valeurs adjacentes (ffill/bfill).")
                    if nan_after > 0:
                        # Ceci ne devrait arriver que si TOUTE la colonne est NaN
                        st.error(f"Attention: Il reste {nan_after} NaN dans la colonne '{col}' même après ffill/bfill. La colonne est probablement entièrement vide ou invalide.")
                        return None # Erreur critique si une colonne essentielle est vide

        # 6. Calcul des Variations (après nettoyage et remplissage)
        if 'Prix' in df_standardized.columns:
            df_standardized['Variation'] = df_standardized['Prix'].diff()
            df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
            # Remplir les NaN initiaux (première ligne) pour les variations
            df_standardized['Variation'].fillna(0, inplace=True)
            df_standardized['Variation_%'].fillna(0, inplace=True)
        else:
            st.error("La colonne 'Prix' (Close) est manquante ou n'a pas pu être traitée. Impossible de calculer les variations.")
            return None

        # 7. Vérification Finale des NaN Critiques
        critical_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume', 'Variation', 'Variation_%']
        # S'assurer que les colonnes existent avant de vérifier les NaN
        existing_critical_cols = [col for col in critical_cols if col in df_standardized.columns]
        if df_standardized[existing_critical_cols].isnull().any().any():
            cols_with_nan = df_standardized[existing_critical_cols].columns[df_standardized[existing_critical_cols].isnull().any()].tolist()
            st.error(f"Erreur critique : Des valeurs manquantes inattendues subsistent dans les colonnes critiques après traitement : {cols_with_nan}. Veuillez vérifier vos données source.")
            # Afficher les lignes avec des NaN restants pour le débogage
            st.dataframe(df_standardized[df_standardized[existing_critical_cols].isnull().any(axis=1)])
            return None

        st.success(f"Données chargées et traitées avec succès ! {len(df_standardized)} lignes finales.")
        return df_standardized

    except pd.errors.EmptyDataError:
        st.error("Erreur : Le fichier CSV semble vide ou ne contient que des en-têtes.")
        return None
    except KeyError as e:
        st.error(f"Erreur : Problème d'accès à une colonne lors du traitement. Vérifiez le mapping des colonnes et le contenu du fichier. Colonne manquante ou mal mappée : '{e}'.")
        st.error(traceback.format_exc())
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        st.error(traceback.format_exc())
        return None


# --- Fonction pour Lien de Téléchargement CSV ---
def get_csv_download_link(df, filename="rapport.csv", link_text="Télécharger les données (CSV)"):
    """Génère un lien de téléchargement HTML pour un DataFrame."""
    if df is None or df.empty:
        return ""
    try:
        # Utiliser io.StringIO pour créer un fichier texte en mémoire
        buffer = io.StringIO()
        # Écrire le DataFrame dans le buffer au format CSV
        # index=True pour inclure l'index (Date) dans le CSV
        # date_format pour contrôler le format des dates dans le CSV
        df.to_csv(buffer, index=True, date_format='%Y-%m-%d %H:%M:%S', sep=';', decimal=',') # Utilisation de ; et , pour compatibilité Excel FR
        csv_string = buffer.getvalue()
        buffer.close()

        # Encoder la chaîne CSV en base64
        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')

        # Styles CSS pour le bouton (peut être personnalisé)
        button_style = """
        display: inline-block;
        padding: 0.5em 1em;
        text-decoration: none;
        background-color: #4CAF50; /* Couleur verte */
        color: white;
        border-radius: 0.25em;
        border: none;
        cursor: pointer;
        font-size: 0.9rem; /* Taille de police légèrement réduite */
        margin-top: 0.5em; /* Marge supérieure réduite */
        transition: background-color 0.3s ease; /* Transition douce au survol */
        """
        # Style au survol (facultatif mais améliore l'UX)
        button_hover_style = """
        <style>
        .download-button:hover {
            background-color: #45a049 !important; /* Vert plus foncé au survol */
            color: white !important;
            text-decoration: none !important;
        }
        </style>
        """
        # Injecter le style de survol dans la page
        st.markdown(f'{button_hover_style}', unsafe_allow_html=True)

        # Créer le lien HTML
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur lors de la création du lien de téléchargement : {e}")
        st.error(traceback.format_exc())
        return ""


# --- Interface Utilisateur dans la Sidebar (Mapping etc.) ---
if current_uploaded_file is not None:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier CSV correspondant aux champs requis.")

    # Lire les colonnes seulement si elles n'ont pas déjà été lues pour ce fichier
    if not st.session_state.all_columns:
        try:
            current_uploaded_file.seek(0)
            # Détecter séparateur (simplifié ici, car déjà fait dans process_data)
            # On suppose que le séparateur détecté précédemment est correct
            # Lire juste les en-têtes (nrows=0) est plus rapide
            # On réutilise la logique de détection d'encodage et de séparateur
            sample_bytes = current_uploaded_file.read(2048)
            current_uploaded_file.seek(0)
            try: sample_text = sample_bytes.decode('utf-8'); encoding = 'utf-8'
            except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore'); encoding = 'latin-1'

            sep = ','
            try:
                if sample_text.strip():
                     sniffer = csv.Sniffer(); dialect = sniffer.sniff(sample_text); sep = dialect.delimiter
            except csv.Error:
                 if sample_text and sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','): sep = ';'

            df_cols = pd.read_csv(current_uploaded_file, sep=sep, encoding=encoding, nrows=0)
            st.session_state.all_columns = df_cols.columns.tolist()
            st.sidebar.write("Colonnes trouvées :", st.session_state.all_columns)
            current_uploaded_file.seek(0) # Important de rembobiner après lecture
        except Exception as e:
            st.sidebar.error(f"Impossible de lire les colonnes du fichier. Vérifiez le format et le séparateur. Erreur: {e}")
            st.session_state.all_columns = [] # Réinitialiser si erreur

    # Afficher les sélecteurs de mapping si les colonnes ont été lues
    if st.session_state.all_columns:
        required_map = {
            "Date": "Date", "Open": "Ouverture (Open)", "High": "Plus Haut (High)",
            "Low": "Plus Bas (Low)", "Close": "Clôture (Close/Prix)", "Volume": "Volume"
        }
        options = [""] + st.session_state.all_columns # Ajouter une option vide

        # Tentative de pré-sélection intelligente
        pre_selected_values = {}
        used_columns_for_preselect = set() # Garder trace des colonnes déjà utilisées pour la pré-sélection

        # 1. Correspondance exacte (insensible à la casse, ignore _)
        for standard_name in required_map.keys():
            normalized_standard = standard_name.lower().replace('_','').replace('(','').replace(')','').replace('/','')
            for col in st.session_state.all_columns:
                normalized_col = col.lower().replace('_','').replace('(','').replace(')','').replace('/','')
                if normalized_standard == normalized_col and col not in used_columns_for_preselect:
                    pre_selected_values[standard_name] = col
                    used_columns_for_preselect.add(col)
                    break # Passer au standard_name suivant

        # 2. Correspondance par termes courants (si pas déjà trouvé)
        common_terms = {
            'Date': ['date', 'time', 'jour', 'séance'],
            'Open': ['open', 'ouverture', 'ouv'],
            'High': ['high', 'haut', 'max'],
            'Low': ['low', 'bas', 'min'],
            'Close': ['close', 'cloture', 'dernier', 'last', 'prix', 'cours'],
            'Volume': ['volume', 'vol', 'quantite', 'qty', 'échangé']
        }
        for standard_name, terms in common_terms.items():
            if standard_name not in pre_selected_values: # Seulement si pas déjà trouvé par correspondance exacte
                for term in terms:
                    found_match = False
                    for col in st.session_state.all_columns:
                        # Vérifier si le terme est DANS le nom de colonne (insensible casse, ignore _)
                        normalized_col = col.lower().replace('_','').replace('(','').replace(')','').replace('/','')
                        if term in normalized_col and col not in used_columns_for_preselect:
                            pre_selected_values[standard_name] = col
                            used_columns_for_preselect.add(col)
                            found_match = True
                            break # Passer au terme suivant pour ce standard_name
                    if found_match:
                        break # Passer au standard_name suivant

        # Créer les widgets selectbox pour chaque champ requis
        for standard_name, display_name in required_map.items():
            # Utiliser la valeur pré-sélectionnée OU la valeur déjà dans session_state (si rechargement) OU vide
            current_selection = st.session_state.column_mapping.get(standard_name, "")
            default_value = pre_selected_values.get(standard_name, current_selection)

            # S'assurer que la valeur par défaut est bien dans les options disponibles
            if default_value not in options:
                default_value = "" # Revenir à vide si la colonne n'existe plus/pas

            # Trouver l'index de la valeur par défaut pour le widget selectbox
            try:
                 default_index = options.index(default_value)
            except ValueError:
                 default_index = 0 # Index de l'option vide ""

            # Créer le selectbox et stocker la sélection dans st.session_state.column_mapping
            st.session_state.column_mapping[standard_name] = st.sidebar.selectbox(
                f"Colonne pour '{display_name}'",
                options=options,
                index=default_index,
                key=f"map_{standard_name}", # Clé unique pour chaque widget
                help=f"Sélectionnez la colonne CSV qui contient les données de {display_name}."
            )

        # Options Avancées (Format de Date)
        with st.sidebar.expander("Options Avancées"):
            st.session_state.date_format_input = st.text_input(
                "Format de date (si conversion auto échoue)",
                value=st.session_state.date_format_input, # Utiliser la valeur de session_state
                key="date_format", # Clé unique
                help="Exemples: %d/%m/%Y, %Y-%m-%d %H:%M:%S. Laisser vide pour la détection auto. Voir la doc Python `strptime` pour les codes."
            ).strip() # .strip() pour enlever les espaces inutiles

        # Bouton pour lancer le traitement
        if st.sidebar.button("▶️ Traiter les Données", key="process_button", type="primary"):
            # Vérifier si toutes les colonnes requises sont mappées
            missing_maps = [name for name, col in st.session_state.column_mapping.items() if not col]
            if missing_maps:
                st.warning(f"Veuillez mapper toutes les colonnes requises : {', '.join(missing_maps)}")
            else:
                # Vérifier si la même colonne CSV est utilisée pour plusieurs champs (avertissement)
                mapped_cols = [col for col in st.session_state.column_mapping.values() if col]
                if len(mapped_cols) != len(set(mapped_cols)):
                    st.warning("Attention : La même colonne CSV a été sélectionnée pour plusieurs champs différents. Assurez-vous que c'est intentionnel.")

                # Lancer le traitement
                current_uploaded_file.seek(0) # S'assurer que le pointeur est au début
                with st.spinner("Traitement des données en cours..."):
                    processed_data = process_data(
                        current_uploaded_file,
                        st.session_state.column_mapping,
                        st.session_state.date_format_input or None # Passer None si vide
                    )
                    # Stocker les données traitées (ou None si erreur) dans session_state
                    if processed_data is not None:
                        st.session_state.data = processed_data
                        st.rerun() # Forcer le rechargement pour afficher les résultats
                    else:
                        # Garder les données précédentes s'il y en avait, sinon vider
                        if 'data' not in st.session_state:
                             st.session_state.data = pd.DataFrame()
                        # Ne pas faire rerun si process_data échoue, pour voir les messages d'erreur

    else:
        # Afficher un message si les colonnes n'ont pas pu être lues
        st.sidebar.warning("Impossible de lire les colonnes du fichier. Veuillez vérifier le fichier ou réessayer.")


# --- Exécution de l'Analyse et du Backtest (si les données sont chargées et traitées) ---
if 'data' in st.session_state and not st.session_state.data.empty:

    data = st.session_state.data.copy() # Utiliser une copie pour éviter de modifier l'état

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    st.session_state.stock_name = st.sidebar.text_input(
        "Nom de l'action",
        value=st.session_state.stock_name, # Utiliser la valeur de session_state
        key="stock_name_input" # Clé unique
    )
    # Mettre à jour le titre principal dynamiquement
    st.title(f"📈 BRVM Quant Backtest - {st.session_state.stock_name}")

    # --- Affichage des Données Traitées ---
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)", expanded=False):
        # Formattage pour une meilleure lisibilité
        st.dataframe(data.tail(100).style.format({
            'Ouverture': '{:,.2f}',
            'Plus_Haut': '{:,.2f}',
            'Plus_Bas': '{:,.2f}',
            'Prix': '{:,.2f}', # Colonne Close renommée en Prix
            'Volume': '{:,.0f}',
            'Variation': '{:,.2f}',
            'Variation_%': '{:.2f}%'
        }))
        # Ajouter le lien de téléchargement pour l'aperçu
        st.markdown(get_csv_download_link(data.tail(100), filename=f"apercu_data_{st.session_state.stock_name}.csv", link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)
        st.markdown(get_csv_download_link(data, filename=f"data_completes_{st.session_state.stock_name}.csv", link_text="Télécharger les données complètes (CSV)"), unsafe_allow_html=True)


    # --- Visualisation du Cours ---
    st.subheader(f"Graphique du Cours Historique de {st.session_state.stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture', color='royalblue')

        # Améliorations esthétiques
        ax.set_title(f'Évolution du cours de {st.session_state.stock_name}', fontsize=14)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Prix (FCFA)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # Formatter l'axe Y pour afficher les milliers avec des virgules
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Formatter l'axe X pour afficher les dates clairement
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12)) # Ajuster le nombre de ticks

        fig.tight_layout() # Ajuster automatiquement la mise en page
        st.pyplot(fig)
        plt.close(fig) # Fermer la figure pour libérer la mémoire

    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique du cours : {e}")
        st.error(traceback.format_exc())


    # --- Paramètres de la Stratégie ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")

    # --- Section Fondamentale (Gordon-Shapiro) ---
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    # Utiliser les clés de session_state pour les valeurs par défaut et la persistance
    dividende_annuel = st.sidebar.number_input(
        "Dernier dividende annuel (FCFA)",
        min_value=0.0,
        value=st.session_state.dividend, # Utilise la clé 'dividend'
        step=10.0,
        key="dividend", # Clé unique pour le widget
        format="%.2f"
    )
    taux_croissance_pct = st.sidebar.slider(
        "Croissance annuelle dividende (%)",
        min_value=-10.0, max_value=20.0, # Ajusté la plage max
        value=st.session_state.growth_rate, # Utilise la clé 'growth_rate'
        step=0.5,
        key="growth_rate", # Clé unique
        format="%.1f%%"
    )
    taux_croissance = taux_croissance_pct / 100.0 # Convertir en décimal

    rendement_exige_pct = st.sidebar.slider(
        "Taux d'actualisation requis (%)",
        min_value=1.0, max_value=30.0, # Ajusté la plage min
        value=st.session_state.discount_rate, # Utilise la clé 'discount_rate'
        step=0.5,
        key="discount_rate", # Clé unique
        format="%.1f%%"
    )
    rendement_exige = rendement_exige_pct / 100.0 # Convertir en décimal

    # Calcul de la Valeur Intrinsèque (VI)
    val_intrinseque = None
    vi_error = False
    if rendement_exige <= taux_croissance:
        st.sidebar.error("Le taux d'actualisation doit être strictement supérieur au taux de croissance pour le modèle Gordon-Shapiro.")
        vi_error = True
    elif dividende_annuel <= 0:
         st.sidebar.warning("Dividende annuel nul ou négatif. La VI est considérée comme 0.")
         val_intrinseque = 0.0 # VI est 0 si pas de dividende
    else:
        try:
            # Calcul du dividende attendu l'année prochaine (D1)
            D1 = dividende_annuel * (1 + taux_croissance)
            # Calcul de la VI
            val_intrinseque = D1 / (rendement_exige - taux_croissance)

            if val_intrinseque < 0:
                st.sidebar.warning(f"La VI calculée est négative ({val_intrinseque:,.2f} FCFA). Cela peut arriver si D1 < 0 (croissance très négative). Signaux VI désactivés.")
                vi_error = True
                val_intrinseque = None # Invalider la VI négative pour les signaux
            elif pd.notna(val_intrinseque):
                st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")
            else:
                 st.sidebar.error("Erreur inconnue lors du calcul de la VI.")
                 vi_error = True

        except ZeroDivisionError:
             st.sidebar.error("Division par zéro lors du calcul de la VI (rendement exigé = taux croissance).")
             vi_error = True
        except Exception as e:
            st.sidebar.error(f"Erreur lors du calcul de la VI : {e}")
            vi_error = True

    # Activer/Désactiver les signaux fondamentaux basés sur la VI valide
    use_fundamental_signals = (val_intrinseque is not None) # VI est valide si elle est calculée et >= 0
    if use_fundamental_signals:
        st.sidebar.success("Signaux basés sur la Valeur Intrinsèque activés.")
    else:
        st.sidebar.warning("Signaux basés sur la Valeur Intrinsèque désactivés (VI invalide, négative ou nulle).")


    # --- Section Technique ---
    st.sidebar.markdown("**Règles de Trading Techniques**")

    # Moyennes Mobiles (MM)
    st.sidebar.markdown("###### Paramètres Moyennes Mobiles (MM)")
    use_mm = st.sidebar.checkbox(
        "Utiliser les signaux MM Crossover",
        value=st.session_state.use_mm_signal, # Utilise la clé 'use_mm_signal'
        key="use_mm_signal" # Clé unique
    )
    if use_mm:
        window_court = st.sidebar.slider(
            "Fenêtre MM Courte (jours)", min_value=5, max_value=100,
            value=st.session_state.short_ma, # Utilise la clé 'short_ma'
            key="short_ma" # Clé unique
        )
        window_long = st.sidebar.slider(
            "Fenêtre MM Longue (jours)", min_value=10, max_value=250, # Assurer min > min court
            value=st.session_state.long_ma, # Utilise la clé 'long_ma'
            key="long_ma" # Clé unique
        )
        if window_court >= window_long:
            st.sidebar.warning("La fenêtre MM Courte doit être inférieure à la fenêtre MM Longue.")
    else:
        # Garder des valeurs par défaut même si désactivé pour éviter erreurs potentielles
        window_court = st.session_state.short_ma
        window_long = st.session_state.long_ma
        st.sidebar.caption("Signaux MM Crossover désactivés.")


    # RSI (Relative Strength Index)
    st.sidebar.markdown("###### Paramètres RSI")
    use_rsi = st.sidebar.checkbox(
        "Utiliser les signaux RSI",
        value=st.session_state.use_rsi_signal, # Utilise la clé 'use_rsi_signal'
        key="use_rsi_signal" # Clé unique
    )
    if use_rsi:
        rsi_window = st.sidebar.slider(
            "Fenêtre RSI (jours)", min_value=5, max_value=50, # Plage ajustée
            value=st.session_state.rsi_window, # Utilise la clé 'rsi_window'
            key="rsi_window" # Clé unique
        )
        rsi_oversold = st.sidebar.slider(
            "Seuil Survente RSI", min_value=10, max_value=50, # Plage ajustée
            value=st.session_state.rsi_oversold, # Utilise la clé 'rsi_oversold'
            key="rsi_oversold" # Clé unique
        )
        rsi_overbought = st.sidebar.slider(
            "Seuil Surachat RSI", min_value=50, max_value=90, # Plage ajustée
            value=st.session_state.rsi_overbought, # Utilise la clé 'rsi_overbought'
            key="rsi_overbought" # Clé unique
        )
        if rsi_oversold >= rsi_overbought:
            st.sidebar.warning("Le seuil de Survente RSI doit être inférieur au seuil de Surachat.")
    else:
        # Garder des valeurs par défaut
        rsi_window = st.session_state.rsi_window
        rsi_oversold = st.session_state.rsi_oversold
        rsi_overbought = st.session_state.rsi_overbought
        st.sidebar.caption("Signaux RSI désactivés.")


    # MACD (Moving Average Convergence Divergence)
    st.sidebar.markdown("###### Paramètres MACD")
    use_macd = st.sidebar.checkbox(
        "Utiliser les signaux MACD Crossover",
        value=st.session_state.use_macd_signal, # Utilise la clé 'use_macd_signal'
        key="use_macd_signal" # Clé unique
    )
    if use_macd:
        macd_fast_window = st.sidebar.slider(
            "Fenêtre Rapide MACD (jours)", min_value=5, max_value=50,
            value=st.session_state.macd_fast_window, # Utilise la clé 'macd_fast_window'
            key="macd_fast_window" # Clé unique
        )
        macd_slow_window = st.sidebar.slider(
            "Fenêtre Lente MACD (jours)", min_value=10, max_value=100, # Assurer min > min rapide
            value=st.session_state.macd_slow_window, # Utilise la clé 'macd_slow_window'
            key="macd_slow_window" # Clé unique
        )
        macd_signal_window = st.sidebar.slider(
            "Fenêtre Signal MACD (jours)", min_value=3, max_value=50, # Plage ajustée
            value=st.session_state.macd_signal_window, # Utilise la clé 'macd_signal_window'
            key="macd_signal_window" # Clé unique
        )
        if macd_fast_window >= macd_slow_window:
            st.sidebar.warning("La fenêtre Rapide MACD doit être inférieure à la fenêtre Lente.")
    else:
        # Garder des valeurs par défaut
        macd_fast_window = st.session_state.macd_fast_window
        macd_slow_window = st.session_state.macd_slow_window
        macd_signal_window = st.session_state.macd_signal_window
        st.sidebar.caption("Signaux MACD Crossover désactivés.")


    # Combinaison des Signaux Techniques
    st.sidebar.markdown("###### Combinaison des Signaux Techniques")
    # Options possibles pour combiner les signaux (si plusieurs sont activés)
    technical_signal_options = [
        "MM OU RSI OU MACD", # Achat si au moins un signal est positif
        "MM ET RSI ET MACD", # Achat seulement si TOUS les signaux sont positifs
        "MM Seulement",
        "RSI Seulement",
        "MACD Seulement",
        "MM OU RSI",
        "MM ET RSI",
        "MM OU MACD",
        "MM ET MACD",
        "RSI OU MACD",
        "RSI ET MACD"
    ]
    # Filtrer les options pour ne montrer que celles qui sont possibles
    # en fonction des indicateurs activés
    possible_options = []
    active_indicators = []
    if use_mm: active_indicators.append("MM")
    if use_rsi: active_indicators.append("RSI")
    if use_macd: active_indicators.append("MACD")

    if not active_indicators:
         possible_options = ["Aucun Indicateur Actif"]
         st.sidebar.warning("Aucun indicateur technique n'est activé. Les signaux techniques seront ignorés.")
         technical_signal_method = possible_options[0]
         tech_signal_method_active = False
    else:
        for option in technical_signal_options:
            required_for_option = []
            if "MM" in option: required_for_option.append("MM")
            if "RSI" in option: required_for_option.append("RSI")
            if "MACD" in option: required_for_option.append("MACD")

            # Si l'option ne nécessite aucun indicateur ou si tous les indicateurs requis sont actifs
            if not required_for_option or all(indicator in active_indicators for indicator in required_for_option):
                 # Et si l'option n'utilise que des indicateurs actifs (cas "Seulement")
                 is_only_option = "Seulement" in option
                 if not is_only_option or (is_only_option and len(required_for_option) == 1):
                      possible_options.append(option)

        # S'assurer que la méthode sélectionnée précédemment est toujours valide
        current_method = st.session_state.tech_signal_method
        if current_method not in possible_options:
            # Si l'ancienne méthode n'est plus valide, choisir la première option possible
            st.session_state.tech_signal_method = possible_options[0] if possible_options else "Aucun Indicateur Actif"


        technical_signal_method = st.sidebar.selectbox(
            "Méthode de combinaison des signaux actifs",
            options=possible_options,
            index=possible_options.index(st.session_state.tech_signal_method) if st.session_state.tech_signal_method in possible_options else 0,
            key="tech_signal_method", # Clé unique
            help="Comment combiner les signaux des indicateurs techniques activés ? 'OU' signifie qu'un seul signal suffit, 'ET' signifie que tous les signaux doivent concorder."
        )
        tech_signal_method_active = True # Au moins un indicateur est actif et une méthode valide est choisie


    # --- Marges Fondamentales & Sorties ---
    st.sidebar.markdown("###### Marges Fondamentales & Sorties")
    if use_fundamental_signals:
        marge_achat_pct = st.sidebar.slider(
            "Marge d'achat vs VI (%)", min_value=0.0, max_value=50.0,
            value=st.session_state.buy_margin, # Utilise la clé 'buy_margin'
            step=1.0, key="buy_margin", # Clé unique
            format="%.1f%%",
            help="Acheter si le prix est inférieur à VI * (1 - Marge Achat). Ex: 20% -> acheter si Prix < 0.8 * VI"
        )
        marge_achat = marge_achat_pct / 100.0

        marge_vente_pct = st.sidebar.slider(
            "Prime de vente vs VI (%)", min_value=0.0, max_value=50.0,
            value=st.session_state.sell_premium, # Utilise la clé 'sell_premium'
            step=1.0, key="sell_premium", # Clé unique
            format="%.1f%%",
            help="Vendre si le prix est supérieur à VI * (1 + Prime Vente). Ex: 10% -> vendre si Prix > 1.1 * VI"
        )
        marge_vente = marge_vente_pct / 100.0
    else:
        marge_achat = 0.0
        marge_vente = 0.0
        st.sidebar.caption("Marges liées à la VI désactivées car la VI n'est pas valide.")

    # Stop Loss / Take Profit / Trailing Stop
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%) / Prix Achat", min_value=1.0, max_value=50.0, # Plage ajustée
        value=st.session_state.stop_loss, # Utilise la clé 'stop_loss'
        step=0.5, key="stop_loss", # Clé unique
        format="%.1f%%",
        help="Vendre si le prix baisse de ce % par rapport au prix d'achat."
    )
    stop_loss = stop_loss_pct / 100.0

    take_profit_pct = st.sidebar.slider(
        "Take Profit (%) / Prix Achat", min_value=1.0, max_value=100.0, # Plage ajustée
        value=st.session_state.take_profit, # Utilise la clé 'take_profit'
        step=1.0, key="take_profit", # Clé unique
        format="%.1f%%",
        help="Vendre si le prix augmente de ce % par rapport au prix d'achat."
    )
    take_profit = take_profit_pct / 100.0

    # Trailing Stop Loss
    use_trailing_stop = st.sidebar.checkbox(
        "Utiliser le Trailing Stop Loss",
        value=st.session_state.use_trailing_stop, # Utilise la clé 'use_trailing_stop'
        key="use_trailing_stop" # Clé unique
    )
    if use_trailing_stop:
        trailing_stop_loss_pct_val = st.sidebar.slider(
            "Trailing Stop Loss (%)", min_value=1.0, max_value=30.0, # Plage ajustée
            value=st.session_state.trailing_stop_pct, # Utilise la clé 'trailing_stop_pct'
            step=0.5, key="trailing_stop_pct", # Clé unique
            format="%.1f%%",
            help="Ajuste le seuil de Stop Loss à la hausse lorsque le prix monte. Vente si le prix baisse de ce % par rapport au plus haut atteint depuis l'achat."
        )
        trailing_stop_loss_pct = trailing_stop_loss_pct_val / 100.0
    else:
        trailing_stop_loss_pct = 0.0 # Pas de trailing stop
        st.sidebar.caption("Trailing Stop Loss désactivé.")


    # --- Paramètres Marché (BRVM) ---
    st.sidebar.markdown("**Paramètres Marché (BRVM)**")
    plafond_variation_pct = st.sidebar.slider(
        "Plafond variation / jour (%)", min_value=5.0, max_value=15.0,
        value=st.session_state.variation_cap, # Utilise la clé 'variation_cap'
        step=0.5, key="variation_cap", # Clé unique
        format="%.1f%%",
        help="Plafond de variation journalière maximum autorisé par la BRVM (ex: 7.5%)."
    )
    plafond_variation = plafond_variation_pct / 100.0

    delai_livraison = st.sidebar.slider(
        "Délai livraison (jours ouvrés, T+)", min_value=0, max_value=5, # 0 pour T+0
        value=st.session_state.settlement_days, # Utilise la clé 'settlement_days'
        key="settlement_days", # Clé unique
        help="Nombre de jours ouvrés entre la transaction et le règlement/livraison (ex: 3 pour T+3)."
    )


    # --- Paramètres du Backtest ---
    st.sidebar.subheader("5. Paramètres du Backtest")
    capital_initial = st.sidebar.number_input(
        "Capital initial (FCFA)", min_value=10000, max_value=1000000000, # Plage ajustée
        value=st.session_state.initial_capital, # Utilise la clé 'initial_capital'
        step=10000, key="initial_capital", # Clé unique
        format="%d"
    )
    frais_transaction_pct = st.sidebar.slider(
        "Frais de transaction (%) par ordre", min_value=0.0, max_value=5.0,
        value=st.session_state.commission_rate, # Utilise la clé 'commission_rate'
        step=0.05, key="commission_rate", # Clé unique
        format="%.2f%%",
        help="Pourcentage de commission appliqué sur la valeur de chaque transaction (achat et vente)."
    )
    frais_transaction = frais_transaction_pct / 100.0

    taux_sans_risque_pct = st.sidebar.slider(
        "Taux sans risque annuel (%)", min_value=0.0, max_value=15.0, # Plage ajustée
        value=st.session_state.risk_free_rate, # Utilise la clé 'risk_free_rate'
        step=0.1, key="risk_free_rate", # Clé unique
        format="%.1f%%",
        help="Taux de rendement annuel d'un placement sans risque (ex: bons du trésor), utilisé pour calculer le ratio de Sharpe."
    )
    taux_sans_risque_annuel = taux_sans_risque_pct / 100.0

    invest_percentage_pct = st.sidebar.slider(
        "Investir (%) du cash dispo par trade", min_value=10, max_value=100,
        value=st.session_state.invest_percentage, # Utilise la clé 'invest_percentage'
        step=5, key="invest_percentage", # Clé unique
        format="%d%%",
        help="Pourcentage du capital disponible à investir lors de chaque signal d'achat. Le montant investi inclut les frais de transaction estimés."
    )
    invest_percentage = invest_percentage_pct / 100.0


    # --- Calculs Techniques et Signaux ---
    st.subheader("Analyse Technique et Signaux")
    analysis_data = data.copy() # Travailler sur une copie pour l'analyse

    # Fonction pour calculer le RSI
    def calculate_rsi(df, column='Prix', window=14):
        """Calcule le Relative Strength Index (RSI)."""
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        # Utiliser EWM (Exponential Weighted Moving Average) pour plus de réactivité
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        # Remplacer les infinis (si avg_loss est 0) par 100 (ou 0 si avg_gain est 0)
        rsi = rsi.replace([np.inf, -np.inf], np.nan) # Remplacer inf par NaN d'abord
        rsi = rsi.fillna(100) # Si avg_loss était 0 et avg_gain > 0, RSI -> 100
        # Si avg_gain et avg_loss étaient 0 (pas de changement), RSI peut être NaN, on le met à 50? Ou ffill?
        # Pour l'instant, on laisse les NaN initiaux dus à min_periods
        return rsi

    # Fonction pour calculer le MACD
    def calculate_macd(df, column='Prix', fast_window=12, slow_window=26, signal_window=9):
        """Calcule la ligne MACD, la ligne de Signal, et l'Histogramme."""
        price_series = df[column]
        # Calcul des Moyennes Mobiles Exponentielles (EMA)
        fast_ema = price_series.ewm(span=fast_window, adjust=False).mean()
        slow_ema = price_series.ewm(span=slow_window, adjust=False).mean()
        # Calcul de la ligne MACD
        macd_line = fast_ema - slow_ema
        # Calcul de la ligne de Signal (EMA de la ligne MACD)
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        # Calcul de l'Histogramme MACD
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # Calculer les indicateurs si activés
    # Initialiser les colonnes avec NaN pour éviter les erreurs si non calculées
    analysis_data['MM_Courte'] = np.nan
    analysis_data['MM_Longue'] = np.nan
    analysis_data['RSI'] = np.nan
    analysis_data['MACD_Line'] = np.nan
    analysis_data['MACD_Signal'] = np.nan
    analysis_data['MACD_Hist'] = np.nan

    min_data_needed = 0 # Pour vérifier si on a assez de données

    try:
        if use_mm and window_long > window_court:
            analysis_data['MM_Courte'] = analysis_data['Prix'].rolling(window=window_court).mean()
            analysis_data['MM_Longue'] = analysis_data['Prix'].rolling(window=window_long).mean()
            min_data_needed = max(min_data_needed, window_long)
        if use_rsi:
            analysis_data['RSI'] = calculate_rsi(analysis_data, window=rsi_window)
            min_data_needed = max(min_data_needed, rsi_window)
        if use_macd and macd_slow_window > macd_fast_window:
            macd_line, signal_line, hist = calculate_macd(
                analysis_data,
                fast_window=macd_fast_window,
                slow_window=macd_slow_window,
                signal_window=macd_signal_window
            )
            analysis_data['MACD_Line'] = macd_line
            analysis_data['MACD_Signal'] = signal_line
            analysis_data['MACD_Hist'] = hist
            # Le MACD a besoin de plus de données pour se stabiliser, typiquement slow_window + signal_window
            min_data_needed = max(min_data_needed, macd_slow_window + signal_window)

        # Vérifier si on a assez de données
        if len(analysis_data) < min_data_needed:
             st.warning(f"Pas assez de données historiques ({len(analysis_data)} jours) pour calculer tous les indicateurs techniques avec les fenêtres choisies (minimum {min_data_needed} jours requis). Les signaux techniques pourraient être non fiables au début.")

        # --- Génération des Signaux Techniques ---
        # Initialiser les colonnes de signaux à 0 (Neutre)
        analysis_data['Signal_MM'] = 0
        analysis_data['Signal_RSI'] = 0
        analysis_data['Signal_MACD'] = 0
        analysis_data['Signal_Technique_Combine'] = 0 # Signal final combiné

        # Signal MM Crossover
        if use_mm and window_long > window_court:
            # Achat: MM Courte croise AU-DESSUS de MM Longue
            analysis_data.loc[(analysis_data['MM_Courte'] > analysis_data['MM_Longue']) & (analysis_data['MM_Courte'].shift(1) <= analysis_data['MM_Longue'].shift(1)), 'Signal_MM'] = 1
            # Vente: MM Courte croise EN-DESSOUS de MM Longue
            analysis_data.loc[(analysis_data['MM_Courte'] < analysis_data['MM_Longue']) & (analysis_data['MM_Courte'].shift(1) >= analysis_data['MM_Longue'].shift(1)), 'Signal_MM'] = -1

        # Signal RSI
        if use_rsi and rsi_overbought > rsi_oversold:
            # Achat: RSI croise AU-DESSUS du seuil de survente
            analysis_data.loc[(analysis_data['RSI'] > rsi_oversold) & (analysis_data['RSI'].shift(1) <= rsi_oversold), 'Signal_RSI'] = 1
            # Vente: RSI croise EN-DESSOUS du seuil de surachat
            analysis_data.loc[(analysis_data['RSI'] < rsi_overbought) & (analysis_data['RSI'].shift(1) >= rsi_overbought), 'Signal_RSI'] = -1

        # Signal MACD Crossover
        if use_macd and macd_slow_window > macd_fast_window:
            # Achat: Ligne MACD croise AU-DESSUS de la ligne Signal
            analysis_data.loc[(analysis_data['MACD_Line'] > analysis_data['MACD_Signal']) & (analysis_data['MACD_Line'].shift(1) <= analysis_data['MACD_Signal'].shift(1)), 'Signal_MACD'] = 1
            # Vente: Ligne MACD croise EN-DESSOUS de la ligne Signal
            analysis_data.loc[(analysis_data['MACD_Line'] < analysis_data['MACD_Signal']) & (analysis_data['MACD_Line'].shift(1) >= analysis_data['MACD_Signal'].shift(1)), 'Signal_MACD'] = -1
            # On pourrait aussi utiliser le croisement de l'histogramme avec zéro

        # Combinaison des signaux techniques selon la méthode choisie
        if tech_signal_method_active:
            buy_signals = pd.Series(False, index=analysis_data.index)
            sell_signals = pd.Series(False, index=analysis_data.index)

            # Déterminer les conditions d'achat et de vente pour chaque indicateur actif
            mm_buy = analysis_data['Signal_MM'] == 1 if use_mm else pd.Series(False, index=analysis_data.index)
            mm_sell = analysis_data['Signal_MM'] == -1 if use_mm else pd.Series(False, index=analysis_data.index)
            rsi_buy = analysis_data['Signal_RSI'] == 1 if use_rsi else pd.Series(False, index=analysis_data.index)
            rsi_sell = analysis_data['Signal_RSI'] == -1 if use_rsi else pd.Series(False, index=analysis_data.index)
            macd_buy = analysis_data['Signal_MACD'] == 1 if use_macd else pd.Series(False, index=analysis_data.index)
            macd_sell = analysis_data['Signal_MACD'] == -1 if use_macd else pd.Series(False, index=analysis_data.index)

            # Appliquer la logique de combinaison
            method = technical_signal_method # Simplifier le nom

            if method == "MM OU RSI OU MACD":
                buy_signals = mm_buy | rsi_buy | macd_buy
                sell_signals = mm_sell | rsi_sell | macd_sell
            elif method == "MM ET RSI ET MACD":
                 # Ne peut être vrai que si tous sont actifs
                 if use_mm and use_rsi and use_macd:
                     buy_signals = mm_buy & rsi_buy & macd_buy
                     sell_signals = mm_sell & rsi_sell & macd_sell
            elif method == "MM Seulement":
                buy_signals = mm_buy
                sell_signals = mm_sell
            elif method == "RSI Seulement":
                buy_signals = rsi_buy
                sell_signals = rsi_sell
            elif method == "MACD Seulement":
                buy_signals = macd_buy
                sell_signals = macd_sell
            elif method == "MM OU RSI":
                buy_signals = mm_buy | rsi_buy
                sell_signals = mm_sell | rsi_sell
            elif method == "MM ET RSI":
                if use_mm and use_rsi:
                     buy_signals = mm_buy & rsi_buy
                     sell_signals = mm_sell & rsi_sell
            elif method == "MM OU MACD":
                buy_signals = mm_buy | macd_buy
                sell_signals = mm_sell | macd_sell
            elif method == "MM ET MACD":
                 if use_mm and use_macd:
                     buy_signals = mm_buy & macd_buy
                     sell_signals = mm_sell & macd_sell
            elif method == "RSI OU MACD":
                buy_signals = rsi_buy | macd_buy
                sell_signals = rsi_sell | macd_sell
            elif method == "RSI ET MACD":
                 if use_rsi and use_macd:
                     buy_signals = rsi_buy & macd_buy
                     sell_signals = rsi_sell & macd_sell

            # Appliquer les signaux combinés
            analysis_data.loc[buy_signals, 'Signal_Technique_Combine'] = 1
            analysis_data.loc[sell_signals, 'Signal_Technique_Combine'] = -1


        # --- Affichage des Indicateurs Techniques ---
        st.markdown("### Indicateurs Techniques Calculés")
        cols_to_show = ['Prix', 'MM_Courte', 'MM_Longue', 'RSI', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'Signal_Technique_Combine']
        # Garder seulement les colonnes qui existent réellement (au cas où un indicateur est désactivé)
        cols_to_show = [col for col in cols_to_show if col in analysis_data.columns]
        st.dataframe(analysis_data[cols_to_show].tail(100).style.format({
             'Prix': '{:,.2f}', 'MM_Courte': '{:,.2f}', 'MM_Longue': '{:,.2f}',
             'RSI': '{:.2f}', 'MACD_Line': '{:.2f}', 'MACD_Signal': '{:.2f}', 'MACD_Hist': '{:.2f}',
             'Signal_Technique_Combine': '{:d}' # Afficher comme entier
        }))
        st.markdown(get_csv_download_link(analysis_data[cols_to_show], filename=f"indicateurs_{st.session_state.stock_name}.csv", link_text="Télécharger les indicateurs (CSV)"), unsafe_allow_html=True)


        # --- Graphiques des Indicateurs ---
        st.markdown("### Graphiques des Indicateurs")
        fig_indicators, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True) # 3 lignes, 1 colonne, partager l'axe X

        # 1. Prix et Moyennes Mobiles
        axes[0].plot(analysis_data.index, analysis_data['Prix'], label='Prix Clôture', color='black', linewidth=1)
        if use_mm and 'MM_Courte' in analysis_data.columns and 'MM_Longue' in analysis_data.columns:
            axes[0].plot(analysis_data.index, analysis_data['MM_Courte'], label=f'MM {window_court} jours', color='orange', linewidth=1)
            axes[0].plot(analysis_data.index, analysis_data['MM_Longue'], label=f'MM {window_long} jours', color='blue', linewidth=1)
        axes[0].set_ylabel('Prix (FCFA)')
        axes[0].set_title(f'Prix et Moyennes Mobiles - {st.session_state.stock_name}', fontsize=12)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


        # 2. RSI
        if use_rsi and 'RSI' in analysis_data.columns:
            axes[1].plot(analysis_data.index, analysis_data['RSI'], label=f'RSI ({rsi_window})', color='purple', linewidth=1)
            axes[1].axhline(rsi_overbought, color='red', linestyle='--', linewidth=0.8, label=f'Surachat ({rsi_overbought})')
            axes[1].axhline(rsi_oversold, color='green', linestyle='--', linewidth=0.8, label=f'Survente ({rsi_oversold})')
            axes[1].fill_between(analysis_data.index, rsi_overbought, rsi_oversold, color='grey', alpha=0.1) # Zone neutre
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100) # Limites standard du RSI
            axes[1].set_title('Relative Strength Index (RSI)', fontsize=12)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, linestyle='--', alpha=0.5)
        else:
             axes[1].text(0.5, 0.5, 'RSI désactivé', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


        # 3. MACD
        if use_macd and 'MACD_Line' in analysis_data.columns and 'MACD_Signal' in analysis_data.columns and 'MACD_Hist' in analysis_data.columns:
            axes[2].plot(analysis_data.index, analysis_data['MACD_Line'], label='MACD', color='blue', linewidth=1)
            axes[2].plot(analysis_data.index, analysis_data['MACD_Signal'], label='Signal', color='red', linewidth=1)
            # Utiliser bar pour l'histogramme, colorer en fonction du signe
            colors = ['green' if x >= 0 else 'red' for x in analysis_data['MACD_Hist']]
            axes[2].bar(analysis_data.index, analysis_data['MACD_Hist'], label='Histogramme', color=colors, width=1.0, alpha=0.6) # Ajuster width si nécessaire
            axes[2].axhline(0, color='grey', linestyle='--', linewidth=0.8)
            axes[2].set_ylabel('MACD')
            axes[2].set_title(f'MACD ({macd_fast_window},{macd_slow_window},{macd_signal_window})', fontsize=12)
            axes[2].legend(fontsize=8)
            axes[2].grid(True, linestyle='--', alpha=0.5)
        else:
             axes[2].text(0.5, 0.5, 'MACD désactivé', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)


        # Formatage final
        axes[-1].set_xlabel('Date') # Label X seulement sur le dernier graphe
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format de date plus court
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=15))
        plt.xticks(rotation=45, ha='right')
        fig_indicators.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajuster pour éviter chevauchement titres/labels
        st.pyplot(fig_indicators)
        plt.close(fig_indicators) # Libérer mémoire

    except Exception as e:
        st.error(f"Erreur lors du calcul ou de l'affichage des indicateurs techniques : {e}")
        st.error(traceback.format_exc())


    # --- Section Backtesting ---
    st.subheader("Backtesting de la Stratégie")

    if st.button("🚀 Lancer le Backtest", key="run_backtest", type="primary"):
        with st.spinner("Exécution du backtest en cours..."):
            try:
                # --- Initialisation du Backtest ---
                cash = capital_initial
                position = 0 # Nombre d'actions détenues
                portfolio_value = capital_initial
                last_buy_price = 0
                peak_portfolio_value_since_buy = 0 # Pour le trailing stop
                trades = [] # Liste pour enregistrer les transactions
                portfolio_history = pd.DataFrame(index=analysis_data.index, columns=['Cash', 'Position', 'Position Value', 'Total Value'])

                # --- Boucle de Backtesting ---
                for i in range(len(analysis_data)):
                    current_date = analysis_data.index[i]
                    current_price = analysis_data['Prix'].iloc[i]
                    current_open_price = analysis_data['Ouverture'].iloc[i] # Utiliser l'ouverture pour l'exécution? Ou la clôture? Clôture ici.
                    current_high_price = analysis_data['Plus_Haut'].iloc[i]
                    current_low_price = analysis_data['Plus_Bas'].iloc[i]

                    # Vérifier si les données sont valides pour ce jour
                    if pd.isna(current_price) or pd.isna(current_open_price) or pd.isna(current_high_price) or pd.isna(current_low_price):
                        # Si données invalides, on reporte la valeur du portefeuille du jour précédent
                        if i > 0:
                            portfolio_history.iloc[i] = portfolio_history.iloc[i-1]
                        else:
                            portfolio_history.iloc[i] = [cash, position, 0, cash]
                        continue # Passer au jour suivant

                    # Initialiser la valeur du portefeuille pour aujourd'hui
                    position_value = position * current_price
                    portfolio_value = cash + position_value
                    portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]

                    # --- Logique de Vente (prioritaire sur l'achat pour éviter achat/vente le même jour) ---
                    sell_signal_triggered = False
                    sell_reason = ""

                    if position > 0: # On ne peut vendre que si on a une position
                        # 1. Vérifier Stop Loss
                        stop_loss_price = last_buy_price * (1 - stop_loss)
                        if current_low_price <= stop_loss_price: # Utiliser le plus bas du jour pour le stop loss
                            sell_signal_triggered = True
                            sell_reason = f"Stop Loss ({stop_loss_pct:.1f}%)"
                            execution_price = stop_loss_price # On suppose l'exécution au seuil SL

                        # 2. Vérifier Trailing Stop Loss (si activé et pas déjà stoppé)
                        if use_trailing_stop and not sell_signal_triggered:
                            # Mettre à jour le plus haut atteint depuis l'achat
                            peak_portfolio_value_since_buy = max(peak_portfolio_value_since_buy, current_high_price)
                            trailing_stop_price = peak_portfolio_value_since_buy * (1 - trailing_stop_loss_pct)
                            if current_low_price <= trailing_stop_price:
                                sell_signal_triggered = True
                                sell_reason = f"Trailing Stop ({trailing_stop_loss_pct_val:.1f}%)"
                                execution_price = trailing_stop_price # Exécution au seuil trailing

                        # 3. Vérifier Take Profit (si pas déjà stoppé)
                        take_profit_price = last_buy_price * (1 + take_profit)
                        if not sell_signal_triggered and current_high_price >= take_profit_price: # Utiliser le plus haut pour TP
                            sell_signal_triggered = True
                            sell_reason = f"Take Profit ({take_profit_pct:.1f}%)"
                            execution_price = take_profit_price # Exécution au seuil TP

                        # 4. Vérifier Signal de Vente Fondamental (si activé et pas déjà stoppé)
                        if use_fundamental_signals and not sell_signal_triggered:
                            sell_vi_price = val_intrinseque * (1 + marge_vente)
                            if current_price > sell_vi_price: # Basé sur le prix de clôture pour VI
                                sell_signal_triggered = True
                                sell_reason = f"Valeur Intrinsèque (Prime {marge_vente_pct:.1f}%)"
                                execution_price = current_price # Exécution à la clôture

                        # 5. Vérifier Signal de Vente Technique (si activé et pas déjà stoppé)
                        if tech_signal_method_active and not sell_signal_triggered:
                            if analysis_data['Signal_Technique_Combine'].iloc[i] == -1:
                                sell_signal_triggered = True
                                sell_reason = f"Signal Technique ({technical_signal_method})"
                                execution_price = current_price # Exécution à la clôture

                        # Exécuter la vente si un signal est déclenché
                        if sell_signal_triggered:
                            # Calculer le montant de la vente
                            sell_value = position * execution_price
                            commission = sell_value * frais_transaction
                            cash += sell_value - commission
                            trade_info = {
                                'Date': current_date, 'Type': 'Vente', 'Prix': execution_price,
                                'Quantité': position, 'Valeur': sell_value, 'Frais': commission,
                                'Cash Après': cash, 'Raison': sell_reason
                            }
                            trades.append(trade_info)
                            st.write(f"Debug - Vente: {trade_info}") # Debug
                            position = 0
                            last_buy_price = 0
                            peak_portfolio_value_since_buy = 0
                            # Mettre à jour la valeur du portefeuille après la vente pour ce jour
                            position_value = 0
                            portfolio_value = cash
                            portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]


                    # --- Logique d'Achat ---
                    buy_signal_triggered = False
                    buy_reason = ""

                    if position == 0: # On ne peut acheter que si on n'a pas de position
                        # 1. Vérifier Signal d'Achat Fondamental (si activé)
                        buy_vi_price = val_intrinseque * (1 - marge_achat) if use_fundamental_signals else -1 # -1 si désactivé
                        fundamental_buy = use_fundamental_signals and current_price < buy_vi_price

                        # 2. Vérifier Signal d'Achat Technique (si activé)
                        technical_buy = tech_signal_method_active and analysis_data['Signal_Technique_Combine'].iloc[i] == 1

                        # Combiner les signaux d'achat (ici on suppose qu'on achète si l'un OU l'autre est vrai)
                        # On pourrait ajouter une option pour ET
                        if fundamental_buy or technical_buy:
                            buy_signal_triggered = True
                            reasons = []
                            if fundamental_buy: reasons.append(f"Valeur Intrinsèque (Marge {marge_achat_pct:.1f}%)")
                            if technical_buy: reasons.append(f"Signal Technique ({technical_signal_method})")
                            buy_reason = " & ".join(reasons)
                            execution_price = current_price # Achat à la clôture

                            # Exécuter l'achat si signal déclenché
                            # Calculer le montant à investir
                            amount_to_invest = cash * invest_percentage
                            # Estimer les frais pour calculer la quantité achetable
                            # Qty * Price + Qty * Price * Fee <= Amount => Qty * Price * (1 + Fee) <= Amount
                            # Qty <= Amount / (Price * (1 + Fee))
                            if execution_price > 0 and (1 + frais_transaction) > 0:
                                quantity_to_buy = int(amount_to_invest / (execution_price * (1 + frais_transaction)))
                            else:
                                quantity_to_buy = 0

                            if quantity_to_buy > 0:
                                buy_value = quantity_to_buy * execution_price
                                commission = buy_value * frais_transaction
                                total_cost = buy_value + commission

                                if total_cost <= cash: # Vérifier si on a assez de cash
                                    cash -= total_cost
                                    position = quantity_to_buy
                                    last_buy_price = execution_price
                                    peak_portfolio_value_since_buy = execution_price # Initialiser le pic au prix d'achat

                                    trade_info = {
                                        'Date': current_date, 'Type': 'Achat', 'Prix': execution_price,
                                        'Quantité': position, 'Valeur': buy_value, 'Frais': commission,
                                        'Cash Après': cash, 'Raison': buy_reason
                                    }
                                    trades.append(trade_info)
                                    st.write(f"Debug - Achat: {trade_info}") # Debug

                                    # Mettre à jour la valeur du portefeuille après l'achat pour ce jour
                                    position_value = position * current_price # Recalculer avec le prix actuel
                                    portfolio_value = cash + position_value
                                    portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]
                                else:
                                     st.write(f"Debug - Achat annulé {current_date}: Pas assez de cash ({cash:.2f}) pour acheter {quantity_to_buy} actions à {execution_price:.2f} (Coût total: {total_cost:.2f})") # Debug
                            else:
                                 st.write(f"Debug - Achat annulé {current_date}: Quantité à acheter nulle ou négative.") # Debug


                    # Mettre à jour la valeur du portefeuille à la fin de la journée (si aucun trade n'a eu lieu ce jour)
                    # Si un trade a eu lieu, la mise à jour a déjà été faite dans la section achat/vente
                    if not buy_signal_triggered and not sell_signal_triggered:
                         position_value = position * current_price
                         portfolio_value = cash + position_value
                         portfolio_history.loc[current_date, ['Cash', 'Position', 'Position Value', 'Total Value']] = [cash, position, position_value, portfolio_value]


                # --- Fin de la Boucle ---
                st.success("Backtest terminé !")

                # --- Affichage des Résultats ---
                st.markdown("### Résultats du Backtest")

                # 1. Historique du Portefeuille
                st.markdown("#### Évolution de la Valeur du Portefeuille")
                portfolio_history.dropna(subset=['Total Value'], inplace=True) # Supprimer les lignes potentiellement vides au début

                if not portfolio_history.empty:
                    fig_portfolio, ax_portfolio = plt.subplots(figsize=(12, 6))
                    ax_portfolio.plot(portfolio_history.index, portfolio_history['Total Value'], label='Valeur Totale Portefeuille', color='green', linewidth=1.5)
                    # Ajouter une ligne pour Buy & Hold
                    buy_hold_value = (capital_initial / analysis_data['Prix'].iloc[0]) * analysis_data['Prix']
                    ax_portfolio.plot(analysis_data.index, buy_hold_value, label='Stratégie Buy & Hold', color='grey', linestyle='--', linewidth=1)

                    ax_portfolio.set_title('Évolution de la Valeur du Portefeuille vs Buy & Hold', fontsize=14)
                    ax_portfolio.set_xlabel('Date', fontsize=10)
                    ax_portfolio.set_ylabel('Valeur Portefeuille (FCFA)', fontsize=10)
                    ax_portfolio.grid(True, linestyle='--', alpha=0.6)
                    ax_portfolio.legend(fontsize=10)
                    ax_portfolio.tick_params(axis='x', rotation=45, labelsize=9)
                    ax_portfolio.tick_params(axis='y', labelsize=9)
                    ax_portfolio.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                    fig_portfolio.tight_layout()
                    st.pyplot(fig_portfolio)
                    plt.close(fig_portfolio)

                    # Afficher les 100 dernières lignes de l'historique
                    with st.expander("Historique détaillé du portefeuille (100 dernières lignes)"):
                        st.dataframe(portfolio_history.tail(100).style.format({
                            'Cash': '{:,.2f}',
                            'Position': '{:,.0f}',
                            'Position Value': '{:,.2f}',
                            'Total Value': '{:,.2f}'
                        }))
                        st.markdown(get_csv_download_link(portfolio_history, filename=f"historique_portefeuille_{st.session_state.stock_name}.csv", link_text="Télécharger l'historique du portefeuille (CSV)"), unsafe_allow_html=True)

                else:
                    st.warning("L'historique du portefeuille est vide. Le backtest n'a peut-être pas pu s'exécuter.")


                # 2. Liste des Transactions
                st.markdown("#### Liste des Transactions")
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_df.set_index('Date', inplace=True)
                    st.dataframe(trades_df.style.format({
                        'Prix': '{:,.2f}',
                        'Quantité': '{:,.0f}',
                        'Valeur': '{:,.2f}',
                        'Frais': '{:,.2f}',
                        'Cash Après': '{:,.2f}'
                    }))
                    st.markdown(get_csv_download_link(trades_df, filename=f"trades_{st.session_state.stock_name}.csv", link_text="Télécharger la liste des trades (CSV)"), unsafe_allow_html=True)
                else:
                    st.info("Aucune transaction n'a été effectuée pendant la période de backtest.")


                # 3. Métriques de Performance
                st.markdown("#### Métriques de Performance Clés")
                if not portfolio_history.empty and len(portfolio_history) > 1:
                    # Calculs de base
                    final_portfolio_value = portfolio_history['Total Value'].iloc[-1]
                    total_return_pct = ((final_portfolio_value / capital_initial) - 1) * 100
                    start_date = portfolio_history.index[0]
                    end_date = portfolio_history.index[-1]
                    duration_years = (end_date - start_date).days / 365.25

                    # Rendement Annualisé (CAGR)
                    cagr = ((final_portfolio_value / capital_initial) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0

                    # Volatilité Annualisée
                    daily_returns = portfolio_history['Total Value'].pct_change().dropna()
                    volatility_annualized = daily_returns.std() * np.sqrt(252) * 100 # 252 jours de trading par an

                    # Ratio de Sharpe
                    # Taux sans risque journalier
                    risk_free_rate_daily = (1 + taux_sans_risque_annuel)**(1/252) - 1
                    excess_returns_daily = daily_returns - risk_free_rate_daily
                    sharpe_ratio = (excess_returns_daily.mean() / excess_returns_daily.std()) * np.sqrt(252) if excess_returns_daily.std() != 0 else 0

                    # Max Drawdown
                    rolling_max = portfolio_history['Total Value'].cummax()
                    daily_drawdown = (portfolio_history['Total Value'] / rolling_max) - 1
                    max_drawdown = daily_drawdown.min() * 100

                    # Performance Buy & Hold
                    final_buy_hold_value = buy_hold_value.iloc[-1]
                    bh_total_return_pct = ((final_buy_hold_value / capital_initial) - 1) * 100
                    bh_cagr = ((final_buy_hold_value / capital_initial) ** (1 / duration_years) - 1) * 100 if duration_years > 0 else 0

                    # Affichage des métriques
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Valeur Finale Portefeuille", f"{final_portfolio_value:,.2f} FCFA", f"{total_return_pct:+.2f}% Total")
                    col1.metric("Rendement Annualisé (CAGR)", f"{cagr:.2f}%")
                    col2.metric("Volatilité Annualisée", f"{volatility_annualized:.2f}%")
                    col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
                    col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                    col3.metric("Nombre de Trades", f"{len(trades)}")

                    st.markdown("---")
                    st.markdown("##### Comparaison Buy & Hold")
                    col1b, col2b = st.columns(2)
                    col1b.metric("Valeur Finale Buy & Hold", f"{final_buy_hold_value:,.2f} FCFA", f"{bh_total_return_pct:+.2f}% Total")
                    col2b.metric("CAGR Buy & Hold", f"{bh_cagr:.2f}%")

                else:
                    st.warning("Impossible de calculer les métriques de performance (historique de portefeuille vide ou trop court).")

            except Exception as e:
                st.error(f"Une erreur est survenue lors de l'exécution du backtest : {e}")
                st.error(traceback.format_exc())

# --- Message si aucune donnée n'est chargée ---
elif current_uploaded_file is None:
    st.info("👈 Veuillez charger un fichier CSV de données historiques dans la barre latérale pour commencer.")
# --- Message si données chargées mais pas traitées ---
elif 'data' not in st.session_state or st.session_state.data.empty:
     if current_uploaded_file is not None and st.session_state.all_columns:
          st.info("👈 Fichier chargé. Veuillez mapper les colonnes et cliquer sur 'Traiter les Données' dans la barre latérale.")
     # Ne rien afficher si le fichier est chargé mais les colonnes n'ont pas pu être lues (erreur déjà affichée)

