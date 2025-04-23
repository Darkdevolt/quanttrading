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
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': """
        ## BRVM Quant Backtest App
        **Version:** 1.5 (Corrections, Ajout % Investissement)

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
    st.session_state.data = pd.DataFrame()
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

def handle_upload():
    """Réinitialise l'état quand un nouveau fichier est chargé."""
    if st.session_state['new_uploaded_file'] is not None:
        st.session_state.uploaded_file_obj = st.session_state['new_uploaded_file']
        st.session_state.data = pd.DataFrame() # Reset processed data
        st.session_state.all_columns = [] # Reset columns list
        st.session_state.column_mapping = { # Reset mapping
            "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
        }
        st.session_state.date_format_input = ""

st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file',
    on_change=handle_upload
)

current_uploaded_file = st.session_state.uploaded_file_obj

# --- Fonction de Traitement des Données (Pas de changement majeur) ---
def process_data(file, column_mapping, date_format=None):
    if file is None: return None
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys): return None

    try:
        file.seek(0)
        sample_bytes = file.read(2048)
        file.seek(0)
        try: sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

        sniffer = csv.Sniffer()
        separator = ','
        try:
            if sample_text.strip():
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
        except csv.Error:
             if sample_text and sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','): separator = ';'

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
                if df[user_col_name].dtype == 'object':
                     cleaned_series = df[user_col_name].astype(str).str.strip().str.replace(',', '.', regex=False).str.replace(r'\s+', '', regex=True)
                     converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                     if converted_series.isnull().all() and not df[user_col_name].isnull().all():
                          cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                          cleaned_series = cleaned_series.str.replace(r'^(-?\.)?$', '', regex=True)
                          cleaned_series = cleaned_series.str.replace(r'(-.*)-', r'\1', regex=True)
                          converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                     df_standardized[standard_col_name] = converted_series
                else:
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                if nan_after_conversion > 0:
                     st.warning(f"{nan_after_conversion} NaN créés dans '{user_col_name}' ({standard_col_name}) lors de la conversion numérique. Ces valeurs seront remplies par ffill/bfill si possible.")
            except Exception as e:
                st.error(f"Erreur conversion numérique colonne '{user_col_name}' ({standard_col_name}) : {e}")
                return None

        df_standardized = df_standardized.sort_values('Date')

        if df_standardized['Date'].duplicated().any():
            duplicates_count = df_standardized['Date'].duplicated().sum()
            st.warning(f"Il y a {duplicates_count} dates dupliquées. Seule la dernière entrée sera conservée.")
            df_standardized = df_standardized.drop_duplicates(subset=['Date'], keep='last')

        df_standardized = df_standardized.set_index('Date')

        cols_to_fill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        for col in cols_to_fill:
             if col in df_standardized.columns:
                 nan_before = df_standardized[col].isnull().sum()
                 if nan_before > 0:
                     df_standardized[col] = df_standardized[col].ffill()
                     df_standardized[col] = df_standardized[col].bfill()
                     nan_after = df_standardized[col].isnull().sum()
                     if nan_after < nan_before:
                         st.info(f"{nan_before - nan_after} NaN dans '{col}' remplis par ffill/bfill.")
                     if nan_after > 0:
                          st.error(f"Attention: Il reste {nan_after} NaN dans la colonne '{col}' après ffill/bfill. Vérifiez vos données source.")


        if 'Prix' in df_standardized.columns:
            if df_standardized['Prix'].isnull().all():
                st.error("La colonne 'Prix' est entièrement NaN même après tentative de remplissage. Impossible de continuer.")
                return None
            df_standardized['Variation'] = df_standardized['Prix'].diff()
            df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
            df_standardized['Variation'].fillna(0, inplace=True)
            df_standardized['Variation_%'].fillna(0, inplace=True)
        else:
             st.error("Colonne 'Prix' manquante, impossible de calculer les variations.")
             return None

        critical_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume', 'Variation', 'Variation_%']
        if df_standardized[critical_cols].isnull().any().any():
            cols_with_nan = df_standardized[critical_cols].columns[df_standardized[critical_cols].isnull().any()].tolist()
            st.error(f"Erreur critique: Il reste des valeurs manquantes inattendues dans les colonnes critiques : {cols_with_nan}. Vérifiez vos données source.")
            st.dataframe(df_standardized[df_standardized[critical_cols].isnull().any(axis=1)])
            return None

        st.success("Données chargées et traitées avec succès !")
        return df_standardized

    except pd.errors.EmptyDataError:
        st.error("Erreur : Le fichier CSV semble vide.")
        return None
    except KeyError as e:
        st.error(f"Erreur : Problème d'accès à une colonne. Vérifiez mapping/fichier. Colonne '{e}'.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        st.error(traceback.format_exc())
        return None

# --- Fonction pour Lien de Téléchargement CSV (Pas de changement) ---
def get_csv_download_link(df, filename="rapport_backtest.csv", link_text="Télécharger le rapport (CSV)"):
    if df.empty: return ""
    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=True, date_format='%Y-%m-%d %H:%M:%S')
        csv_string = buffer.getvalue()
        buffer.close()
        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        button_style = """
        display: inline-block; padding: 0.5em 1em; text-decoration: none;
        background-color: #4CAF50; color: white; border-radius: 0.25em;
        border: none; cursor: pointer; font-size: 1rem; margin-top: 1em;
        """
        button_hover_style = """
        <style>
        .download-button:hover { background-color: #45a049 !important; color: white !important; text-decoration: none !important; }
        </style>
        """
        st.markdown(f'{button_hover_style}', unsafe_allow_html=True)
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur création lien téléchargement : {e}")
        st.error(traceback.format_exc())
        return ""

# --- Interface Utilisateur dans la Sidebar (Mapping etc.) ---
if current_uploaded_file is not None:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier.")

    if not st.session_state.all_columns:
        try:
            current_uploaded_file.seek(0)
            sample_bytes = current_uploaded_file.read(2048)
            current_uploaded_file.seek(0)
            try: sample_text = sample_bytes.decode('utf-8')
            except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

            sniffer = csv.Sniffer()
            sep = ','
            try:
                if sample_text.strip():
                    dialect = sniffer.sniff(sample_text)
                    sep = dialect.delimiter
            except csv.Error:
                 if sample_text and sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','): sep = ';'

            try:
                df_cols = pd.read_csv(current_uploaded_file, sep=sep, nrows=0)
                st.session_state.all_columns = df_cols.columns.tolist()
            except Exception as e:
                st.sidebar.error(f"Impossible lire colonnes (Vérifiez séparateur/format): {e}")
                st.session_state.all_columns = []

            current_uploaded_file.seek(0)
        except Exception as e:
             st.sidebar.error(f"Erreur lecture initiale fichier pour mapping: {e}")
             st.sidebar.info("Assurez-vous que le fichier est un CSV valide.")
             st.session_state.all_columns = []

    if not st.session_state.all_columns:
        st.sidebar.warning("Impossible de lire les colonnes du fichier chargé.")
    else:
        st.sidebar.write("Colonnes trouvées :", st.session_state.all_columns)
        required_map = {
            "Date": "Date", "Open": "Ouverture", "High": "Plus Haut",
            "Low": "Plus Bas", "Close": "Clôture", "Volume": "Volume"
        }
        pre_selected_values = {} # Logic identical to previous version for pre-selection

        used_columns = set()
        for standard_name in required_map.keys():
             normalized_standard = standard_name.lower().replace('_','')
             for col in st.session_state.all_columns:
                 normalized_col = col.lower().replace('_','')
                 if normalized_standard == normalized_col and col not in used_columns:
                     pre_selected_values[standard_name] = col
                     used_columns.add(col)
                     break
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

        for standard_name, display_name in required_map.items():
            default_value = st.session_state.column_mapping.get(standard_name, pre_selected_values.get(standard_name, ""))
            if default_value not in [""] + st.session_state.all_columns:
                 default_value = ""

            st.session_state.column_mapping[standard_name] = st.sidebar.selectbox(
                f"Colonne pour '{display_name}'",
                options=[""] + st.session_state.all_columns,
                index=([""] + st.session_state.all_columns).index(default_value) if default_value else 0,
                key=f"map_{standard_name}"
            )

        with st.sidebar.expander("Options Avancées"):
             st.session_state.date_format_input = st.text_input(
                 "Format de date (si conversion auto échoue, ex: %d/%m/%Y)",
                 value=st.session_state.date_format_input,
                 key="date_format",
                 help="Exemples: %Y-%m-%d, %d/%m/%Y %H:%M:%S. Voir Python strptime."
             )

        if st.sidebar.button("▶️ Traiter les Données", key="process_button"):
            missing_maps = [name for name, col in st.session_state.column_mapping.items() if not col]
            if missing_maps:
                 st.warning(f"Veuillez mapper les colonnes requises : {', '.join(missing_maps)}")
            else:
                 mapped_cols = [col for col in st.session_state.column_mapping.values() if col]
                 if len(mapped_cols) != len(set(mapped_cols)):
                     st.warning("Attention: La même colonne CSV a été sélectionnée pour plusieurs champs.")
                 else:
                     current_uploaded_file.seek(0)
                     with st.spinner("Traitement des données en cours..."):
                         st.session_state.data = process_data(
                             current_uploaded_file,
                             st.session_state.column_mapping,
                             st.session_state.date_format_input or None
                         )

# --- Exécution de l'Analyse (si les données sont chargées et traitées) ---
if not st.session_state.data.empty:

    data = st.session_state.data.copy() # Use a copy from session state

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    st.session_state.stock_name = st.sidebar.text_input(
        "Nom de l'action",
        st.session_state.stock_name,
        key="stock_name_input"
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

    # --- Visualisation du Cours (Pas de changement) ---
    st.subheader(f"Cours historique de {st.session_state.stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture')
        ax.set_title(f'Évolution du cours de {st.session_state.stock_name}')
        ax.set_xlabel('Date'); ax.set_ylabel('Prix (FCFA)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig)
    except Exception as e: st.error(f"Erreur génération graphique cours : {e}")

    # --- Paramètres de la Stratégie ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")

    # --- Fondamental (Pas de changement, sauf message pour VI) ---
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    dividende_annuel = st.sidebar.number_input("Dernier dividende annuel (FCFA)", min_value=0.0, value=600.0, step=10.0, key="dividend")
    taux_croissance = st.sidebar.slider("Croissance annuelle dividende (%)", -10.0, 15.0, 3.0, 0.5, key="growth_rate") / 100
    rendement_exige = st.sidebar.slider("Taux d'actualisation (%)", 5.0, 30.0, 12.0, 0.5, key="discount_rate") / 100

    val_intrinseque = None
    if rendement_exige <= taux_croissance:
        st.sidebar.error("Le taux d'actualisation doit être supérieur au taux de croissance.")
    else:
        try:
             if dividende_annuel <= 0: val_intrinseque = 0
             else: D1 = dividende_annuel * (1 + taux_croissance); val_intrinseque = D1 / (rendement_exige - taux_croissance)
             if val_intrinseque < 0: st.sidebar.warning(f"VI négative ({val_intrinseque:,.2f}). Signaux VI désactivés.")
             elif pd.notna(val_intrinseque): st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")
        except Exception as e: st.sidebar.error(f"Erreur calcul VI: {e}")

    use_fundamental_signals = (val_intrinseque is not None and val_intrinseque > 0)
    if use_fundamental_signals: st.sidebar.info("Signaux de Valeur Intrinsèque activés.")
    else: st.sidebar.warning("Signaux de Valeur Intrinsèque désactivés (VI invalide ou <= 0).")

    # --- Technique (Incluant RSI) ---
    st.sidebar.markdown("**Règles de Trading Techniques**")

    # MM Parameters
    st.sidebar.markdown("###### Paramètres Moyennes Mobiles")
    window_court = st.sidebar.slider("Fenêtre MM Courte (j)", 5, 100, 20, key="short_ma")
    window_long = st.sidebar.slider("Fenêtre MM Longue (j)", 20, 250, 50, key="long_ma")
    if window_court >= window_long: st.sidebar.warning("La fenêtre MM Courte doit être < à la MM Longue.")

    # RSI Parameters
    st.sidebar.markdown("###### Paramètres RSI (Relative Strength Index)")
    use_rsi = st.sidebar.checkbox("Utiliser les signaux RSI", value=True, key="use_rsi_signal")
    if use_rsi:
        rsi_window = st.sidebar.slider("Fenêtre RSI (j)", 5, 30, 14, key="rsi_window")
        rsi_oversold = st.sidebar.slider("Seuil Survente RSI", 10, 40, 30, key="rsi_oversold")
        rsi_overbought = st.sidebar.slider("Seuil Surachat RSI", 60, 90, 70, key="rsi_overbought")
        if rsi_oversold >= rsi_overbought: st.sidebar.warning("Seuil Survente >= Seuil Surachat.")
    else:
        st.sidebar.info("Signaux RSI désactivés.")
        rsi_window = 14
        rsi_oversold = 30
        rsi_overbought = 70

    # Fundamental Margins (dependent on use_fundamental_signals)
    st.sidebar.markdown("###### Marges Fundamentales & Sorties")
    if use_fundamental_signals:
        marge_achat = st.sidebar.slider("Marge achat / VI (%)", 0, 50, 20, key="buy_margin") / 100
        marge_vente = st.sidebar.slider("Prime sortie / VI (%)", 0, 50, 10, key="sell_premium") / 100
    else: marge_achat = 0; marge_vente = 0; st.sidebar.caption("Marges VI désactivées.")

    # Stop Loss / Take Profit (Pas de changement)
    stop_loss = st.sidebar.slider("Stop Loss / Prix Achat (%)", 1.0, 30.0, 10.0, 0.5, key="stop_loss") / 100
    take_profit = st.sidebar.slider("Take Profit / Prix Achat (%)", 5.0, 100.0, 20.0, 1.0, key="take_profit") / 100

    # --- Marché (Pas de changement) ---
    st.sidebar.markdown("**Paramètres Marché (BRVM)**")
    plafond_variation = st.sidebar.slider("Plafond variation /j (%)", 5.0, 15.0, 7.5, 0.5, key="variation_cap") / 100
    delai_livraison = st.sidebar.slider("Délai livraison (j ouvrés)", 1, 5, 3, key="settlement_days")

    # --- Backtest Parameters (MODIFIÉ pour quantité) ---
    st.sidebar.subheader("5. Paramètres du Backtest")
    capital_initial = st.sidebar.number_input("Capital initial (FCFA)", 100000, 100000000, 1000000, step=100000, key="initial_capital")
    frais_transaction = st.sidebar.slider("Frais transaction (%)", 0.0, 5.0, 0.5, 0.05, key="commission_rate") / 100
    taux_sans_risque = st.sidebar.slider("Taux sans risque annuel (%)", 0.0, 10.0, 3.0, 0.1, key="risk_free_rate") / 100

    # Nouveau paramètre pour la quantité à investir
    invest_percentage = st.sidebar.slider("Investir (%) du cash dispo par trade", 10, 100, 100, 5, key="invest_percentage") / 100
    st.sidebar.caption("Le cash investi inclut les frais.")


    # --- Calculs Techniques et Signaux ---
    st.subheader("Analyse Technique et Signaux")

    # Function to calculate RSI
    def calculate_rsi(df, column='Prix', window=14):
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Wilders' smoothing (equivalent to EMA alpha=1/window)
        avg_gain = gain.ewm(com=window-1, adjust=False).mean()
        avg_loss = loss.ewm(com=window-1, adjust=False).mean()

        rs = avg_gain / avg_loss
        # Handle division by zero: if avg_loss is 0, rs is inf. 100/(1+inf) is 0. RSI becomes 100.
        # Fill any potential NaNs (e.g. if both avg_gain and avg_loss are NaN initially) with 0 for the division result
        # Use .replace([np.inf, -np.inf], np.nan) before fillna to handle edge cases if division results in inf/neg_inf
        rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan).fillna(0))).fillna(0) # Chained fillna/replace for safety
        return rsi

    # Calculate MM (Recalculé à chaque exécution)
    try:
        min_data_needed = window_long
        if use_rsi:
             min_data_needed = max(min_data_needed, rsi_window)

        if len(data) < min_data_needed:
             st.warning(f"Pas assez données ({len(data)}j) pour certains indicateurs (MM Longue {window_long}j, RSI {rsi_window}j). Minimum requis: {min_data_needed} jours. Certains indicateurs/signaux seront NaN au début.")
             # Continue execution, NaNs will appear where data is insufficient

        data['MM_Court'] = data['Prix'].rolling(window=window_court, min_periods=1).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long, min_periods=1).mean()

        # Calculate RSI (NEW - Recalculé à chaque exécution)
        if use_rsi:
            data['RSI'] = calculate_rsi(data, column='Prix', window=rsi_window)
        else:
            data['RSI'] = np.nan # Ensure column exists even if not used


    except Exception as e: st.error(f"Erreur calcul Indicateurs (MM/RSI) : {e}");


    # Niveaux Fondamentaux (Recalculé à chaque exécution)
    data['val_intrinseque'] = val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan
    data['prix_achat_fondamental'] = (1 - marge_achat) * val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan
    data['prix_vente_fondamental'] = (1 + marge_vente) * val_intrinseque if use_fundamental_signals and val_intrinseque > 0 else np.nan


    # Signaux Techniques MM (Recalculé à chaque exécution)
    data['signal_technique_mm'] = 0
    if 'MM_Court' in data.columns and 'MM_Long' in data.columns and not data[['MM_Court', 'MM_Long']].isnull().all().all():
         valid_ma = (data['MM_Court'].notna()) & (data['MM_Long'].notna())
         buy_cond_mm = valid_ma & (data['MM_Court'] > data['MM_Long']) & (data['MM_Court'].shift(1) <= data['MM_Long'].shift(1))
         sell_cond_mm = valid_ma & (data['MM_Court'] < data['MM_Long']) & (data['MM_Court'].shift(1) >= data['MM_Long'].shift(1))
         data.loc[buy_cond_mm, 'signal_technique_mm'] = 1
         data.loc[sell_cond_mm, 'signal_technique_mm'] = -1


    # Signaux Techniques RSI (Recalculé à chaque exécution)
    data['signal_technique_rsi'] = 0
    if use_rsi and 'RSI' in data.columns and not data['RSI'].isnull().all():
        valid_rsi = data['RSI'].notna()
        buy_cond_rsi = valid_rsi & (data['RSI'] > rsi_oversold) & (data['RSI'].shift(1) <= rsi_oversold)
        sell_cond_rsi = valid_rsi & (data['RSI'] < rsi_overbought) & (data['RSI'].shift(1) >= rsi_overbought)
        data.loc[buy_cond_rsi, 'signal_technique_rsi'] = 1
        data.loc[sell_cond_rsi, 'signal_technique_rsi'] = -1


    # Signaux Combinés Achat/Vente (LOGIQUE MODIFIÉE POUR INCLURE RSI)
    cond_achat_tech = (data['signal_technique_mm'] == 1)
    cond_vente_tech = (data['signal_technique_mm'] == -1)

    if use_rsi:
        cond_achat_tech = cond_achat_tech | (data['signal_technique_rsi'] == 1)
        cond_vente_tech = cond_vente_tech | (data['signal_technique_rsi'] == -1)

    if use_fundamental_signals:
        if 'prix_achat_fondamental' in data.columns and 'prix_vente_fondamental' in data.columns and \
           not data[['prix_achat_fondamental', 'prix_vente_fondamental']].isnull().all().all():

            cond_achat_fond = (data['Prix'] < data['prix_achat_fondamental']) & data['prix_achat_fondamental'].notna()
            cond_vente_fond = (data['Prix'] > data['prix_vente_fondamental']) & data['prix_vente_fondamental'].notna()

            data['achat'] = cond_achat_tech & cond_achat_fond
            data['vente_signal'] = cond_vente_tech | cond_vente_fond
        else:
             st.warning("Signaux fondamentaux désactivés car les seuils calculés sont invalides (NaN).")
             data['achat'] = cond_achat_tech
             data['vente_signal'] = cond_vente_tech
             use_fundamental_signals = False
    else:
        data['achat'] = cond_achat_tech
        data['vente_signal'] = cond_vente_tech


    # Graphique Indicateurs Techniques (CORRIGÉ l'erreur de slicing pour le plot)
    st.subheader("Analyse Technique - Indicateurs")
    try:
        fig_tech, ax_tech = plt.subplots(figsize=(12, 8), nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax_price, ax_rsi = ax_tech

        # Price and MM Plot - CORRECTION ici, tracer la série dropna directement
        ax_price.plot(data.index, data['Prix'], label='Prix', lw=1, alpha=0.8, zorder=2)
        if 'MM_Court' in data.columns and not data['MM_Court'].isnull().all():
             # data['MM_Court'].dropna() fournit les valeurs et l'index correspondant sans NaN initiaux
             ax_price.plot(data['MM_Court'].dropna().index, data['MM_Court'].dropna(), label=f'MM {window_court}j', lw=1.5, zorder=3)
        if 'MM_Long' in data.columns and not data['MM_Long'].isnull().all():
             # CORRECTION similaire pour MM Longue
             ax_price.plot(data['MM_Long'].dropna().index, data['MM_Long'].dropna(), label=f'MM {window_long}j', lw=1.5, zorder=3)

        ax_price.set_title('Prix & Moyennes Mobiles'); ax_price.set_ylabel('Prix (FCFA)')
        ax_price.grid(True, linestyle='--', alpha=0.6, zorder=1); ax_price.legend(loc='upper left')
        ax_price.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


        # RSI Plot - CORRECTION ici si le RSI a aussi des NaNs au début
        if use_rsi and 'RSI' in data.columns and not data['RSI'].isnull().all():
            # CORRECTION similaire pour RSI
            ax_rsi.plot(data['RSI'].dropna().index, data['RSI'].dropna(), label=f'RSI ({rsi_window}j)', lw=1.5, color='purple')
            ax_rsi.axhline(rsi_overbought, color='red', linestyle='--', lw=1, alpha=0.7, label=f'Surachat ({rsi_overbought})')
            ax_rsi.axhline(rsi_oversold, color='green', linestyle='--', lw=1, alpha=0.7, label=f'Survente ({rsi_oversold})')
            ax_rsi.axhline(50, color='grey', linestyle=':', lw=1, alpha=0.5, label='50')
            ax_rsi.set_title('Indicateur RSI'); ax_rsi.set_ylabel('RSI')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, linestyle='--', alpha=0.6); ax_rsi.legend(loc='upper left')
        else:
             ax_rsi.set_title('Indicateur RSI (Désactivé ou Indisponible)'); ax_rsi.set_ylabel('RSI')
             ax_rsi.text(0.5, 0.5, "RSI désactivé ou calcul impossible", horizontalalignment='center', verticalalignment='center', transform=ax_rsi.transAxes, color='grey')


        ax_rsi.set_xlabel('Date')
        ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_rsi.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig_tech.autofmt_xdate()
        plt.tight_layout(); st.pyplot(fig_tech)
    except Exception as e: st.error(f"Erreur génération graphique indicateurs : {e}"); st.error(traceback.format_exc())


    # Graphique Signaux (Pas de changement)
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', lw=1.5, zorder=2)

        if use_fundamental_signals and val_intrinseque is not None and val_intrinseque > 0:
            ax3.axhline(y=val_intrinseque, color='grey', ls='-', alpha=0.7, label=f'VI ({val_intrinseque:,.0f})', zorder=1)
            if 'prix_achat_fondamental' in data.columns and pd.notna(data['prix_achat_fondamental'].iloc[0]): ax3.axhline(y=data['prix_achat_fondamental'].iloc[0], color='green', ls='--', alpha=0.6, label=f'Seuil Achat VI ({data["prix_achat_fondamental"].iloc[0]:,.0f})', zorder=1)
            if 'prix_vente_fondamental' in data.columns and pd.notna(data['prix_vente_fondamental'].iloc[0]): ax3.axhline(y=data['prix_vente_fondamental'].iloc[0], color='red', ls='--', alpha=0.6, label=f'Seuil Vente VI ({data["prix_vente_fondamental"].iloc[0]:,.0f})', zorder=1)


        achats_sig = data[data['achat'] & data['Prix'].notna()]
        ventes_sig = data[data['vente_signal'] & data['Prix'].notna()]

        if not achats_sig.empty: ax3.scatter(achats_sig.index, achats_sig['Prix'], color='lime', edgecolor='green', s=70, marker='^', label='Signal Achat Strat', zorder=5)
        if not ventes_sig.empty: ax3.scatter(ventes_sig.index, ventes_sig['Prix'], color='tomato', edgecolor='red', s=70, marker='v', label='Signal Vente Strat', zorder=5)

        ax3.set_title('Prix et Signaux Trading Stratégie'); ax3.set_xlabel('Date'); ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, linestyle='--', alpha=0.6, zorder=1); ax3.legend(loc='best')
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig3.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig3)
    except Exception as e: st.error(f"Erreur graphique signaux : {e}"); st.error(traceback.format_exc())

    # --- Backtest ---
    st.subheader("🚀 Backtest de la Stratégie")
    st.markdown(f"Capital: **{capital_initial:,.0f} FCFA**, Frais: **{frais_transaction*100:.2f}%**, Plafond: **{plafond_variation*100:.1f}%**, Livraison: **{delai_livraison}j**.")
    st.markdown(f"Investissement par trade : **{invest_percentage*100:.0f}%** du cash disponible.") # Afficher le nouveau paramètre


    # Fonction Backtest (MODIFIÉE pour la quantité et la gestion du cash insuffisant)
    def run_backtest(data_for_backtest, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison, invest_percentage):
         if data_for_backtest.empty or not isinstance(data_for_backtest.index, pd.DatetimeIndex):
              st.error("Données invalides pour le backtest."); return pd.DataFrame(), [], [], pd.DataFrame()

         portfolio = pd.DataFrame(index=data_for_backtest.index)
         portfolio['prix_effectif'] = 0.0
         portfolio['actions'] = 0.0
         portfolio['cash'] = float(capital_initial)
         portfolio['valeur_actions'] = 0.0
         portfolio['valeur_totale'] = float(capital_initial)
         portfolio['rendement'] = 0.0
         portfolio['trade_en_cours'] = False
         portfolio['date_livraison_prevue'] = pd.NaT
         portfolio['prix_achat_moyen'] = 0.0
         portfolio['stop_loss_price'] = np.nan
         portfolio['take_profit_price'] = np.nan

         transactions = []
         achats_dates = []
         ventes_dates = []

         nb_actions_possedees = 0.0
         cash_disponible = float(capital_initial)
         prix_achat_moyen_actif = 0.0
         trade_en_cours_boucle = False
         date_livraison_boucle = pd.NaT
         stop_loss_actif = np.nan
         take_profit_actif = np.nan

         bday = BDay()

         for i, (jour, row) in enumerate(data_for_backtest.iterrows()):
             # --- Début de Journée ---
             prix_jour_brut = row['Prix']

             prix_veille_eff = portfolio.loc[data_for_backtest.index[i-1], 'prix_effectif'] if i > 0 else prix_jour_brut

             if prix_veille_eff == 0: variation = 0
             else: variation = (prix_jour_brut - prix_veille_eff) / prix_veille_eff

             prix_effectif_jour = prix_jour_brut

             if abs(variation) > plafond_variation:
                  prix_effectif_jour = prix_veille_eff * (1 + (np.sign(variation) * plafond_variation))

             portfolio.loc[jour, 'prix_effectif'] = prix_effectif_jour

             if i > 0:
                  jour_prec = data_for_backtest.index[i-1]
                  nb_actions_possedees = portfolio.loc[jour_prec, 'actions']
                  cash_disponible = portfolio.loc[jour_prec, 'cash']
                  trade_en_cours_boucle = portfolio.loc[jour_prec, 'trade_en_cours']
                  date_livraison_boucle = portfolio.loc[jour_prec, 'date_livraison_prevue']
                  prix_achat_moyen_actif = portfolio.loc[jour_prec, 'prix_achat_moyen']
                  stop_loss_actif = portfolio.loc[jour_prec, 'stop_loss_price']
                  take_profit_actif = portfolio.loc[jour_prec, 'take_profit_price']

             # --- Gérer le Délai de Livraison ---
             actions_disponibles_vente = nb_actions_possedees if jour >= date_livraison_boucle else 0

             # --- Conditions de Vente ---
             vente_trigger = False
             vente_raison = None

             # 1. Stop Loss
             if trade_en_cours_boucle and actions_disponibles_vente > 0 and pd.notna(stop_loss_actif) and prix_effectif_jour <= stop_loss_actif:
                  vente_trigger = True; vente_raison = "Stop Loss"

             # 2. Take Profit
             if not vente_trigger and trade_en_cours_boucle and actions_disponibles_vente > 0 and pd.notna(take_profit_actif) and prix_effectif_jour >= take_profit_actif:
                  vente_trigger = True; vente_raison = "Take Profit"

             # 3. Signal de vente de la stratégie (incluant MM ou Funda si activé)
             if not vente_trigger and row['vente_signal'] and actions_disponibles_vente > 0:
                  vente_trigger = True; vente_raison = "Signal Stratégie"


             if vente_trigger:
                 if nb_actions_possedees > 0:
                      montant_vente = nb_actions_possedees * prix_effectif_jour
                      frais = montant_vente * frais_transaction
                      cash_obtenu = montant_vente - frais
                      cash_disponible += cash_obtenu

                      transactions.append({
                          'Date': jour, 'Type': 'Vente', 'Quantité': nb_actions_possedees,
                          'Prix_Unitaire': prix_effectif_jour, 'Montant': montant_vente,
                          'Frais': frais, 'Cash_Net': cash_obtenu, 'Raison': vente_raison,
                          'Prix_Achat_Moyen': prix_achat_moyen_actif
                      })
                      ventes_dates.append(jour)

                      nb_actions_possedees = 0.0; prix_achat_moyen_actif = 0.0
                      trade_en_cours_boucle = False; date_livraison_boucle = pd.NaT
                      stop_loss_actif = np.nan; take_profit_actif = np.nan


             # --- Conditions d'Achat ---
             # row['achat'] inclut maintenant la logique MM et/ou RSI et/ou Funda
             if not trade_en_cours_boucle and row['achat'] and cash_disponible > 0:
                 # MODIFICATION ICI : Calcul de la quantité en incluant les frais dès le départ
                 # et en utilisant le pourcentage d'investissement du cash disponible
                 amount_to_spend = cash_disponible * invest_percentage
                 cost_per_share_with_fees = prix_effectif_jour * (1 + frais_transaction)

                 if cost_per_share_with_fees > 0: # Éviter la division par zéro
                     # Calculer le nombre maximal d'actions entières achetable avec le montant alloué
                     nb_actions_a_acheter = np.floor(amount_to_spend / cost_per_share_with_fees)
                 else:
                     nb_actions_a_acheter = 0 # Ne peut rien acheter si le coût est nul ou négatif

                 if nb_actions_a_acheter >= 1: # S'assurer qu'on peut acheter au moins une action
                     montant_achat = nb_actions_a_acheter * prix_effectif_jour
                     frais = montant_achat * frais_transaction # Recalculer frais précis pour l'enregistrement
                     total_cost = montant_achat + frais

                     # Cette vérification devrait maintenant toujours passer si nb_actions_a_acheter >= 1,
                     # grâce à la nouvelle formule de calcul.
                     if total_cost <= cash_disponible:
                         cash_disponible -= total_cost
                         prix_achat_moyen_actif = prix_effectif_jour
                         nb_actions_possedees += nb_actions_a_acheter

                         transactions.append({
                             'Date': jour, 'Type': 'Achat', 'Quantité': nb_actions_a_acheter,
                             'Prix_Unitaire': prix_effectif_jour, 'Montant': montant_achat,
                             'Frais': frais, 'Cash_Net': -total_cost,
                             'Raison': 'Signal Stratégie', 'Prix_Achat_Moyen': prix_achat_moyen_actif
                         })
                         achats_dates.append(jour)

                         trade_en_cours_boucle = True
                         date_livraison_boucle = jour + BDay(delai_livraison)
                         stop_loss_actif = prix_effectif_jour * (1 - stop_loss)
                         take_profit_actif = prix_effectif_jour * (1 + take_profit)

                     else:
                          # Ce cas devient très rare avec la nouvelle formule, peut-être dû à des
                          # légers problèmes de précision flottante ou si amount_to_spend était > cash_disponible
                          st.warning(f"{jour.date()}: Logique achat inattendue - Coût total ({total_cost:,.2f}) > Cash ({cash_disponible:,.2f}) après calcul de quantité ({nb_actions_a_acheter}).")
                 else:
                      # Ce warning est normal si le cash alloué ne permet pas d'acheter une action entière
                      st.info(f"{jour.date()}: Pas assez de cash ({cash_disponible*invest_percentage:,.2f} alloué) pour acheter au moins une action au prix {prix_effectif_jour:,.2f} (incluant frais).")


             # --- Fin de Journée ---
             portfolio.loc[jour, 'actions'] = nb_actions_possedees
             portfolio.loc[jour, 'cash'] = cash_disponible
             portfolio.loc[jour, 'valeur_actions'] = nb_actions_possedees * prix_effectif_jour
             portfolio.loc[jour, 'valeur_totale'] = portfolio.loc[jour, 'cash'] + portfolio.loc[jour, 'valeur_actions']
             portfolio.loc[jour, 'trade_en_cours'] = trade_en_cours_boucle
             portfolio.loc[jour, 'date_livraison_prevue'] = date_livraison_boucle
             portfolio.loc[jour, 'prix_achat_moyen'] = prix_achat_moyen_actif
             portfolio.loc[jour, 'stop_loss_price'] = stop_loss_actif
             portfolio.loc[jour, 'take_profit_price'] = take_profit_actif


         # Simulation de la vente finale si position ouverte
         if nb_actions_possedees > 0:
             dernier_jour = data_for_backtest.index[-1]
             # Utiliser le dernier prix effectif enregistré dans le portefeuille
             prix_dernier_jour = portfolio.loc[dernier_jour, 'prix_effectif']
             montant_vente = nb_actions_possedees * prix_dernier_jour
             frais = montant_vente * frais_transaction
             cash_obtenu = montant_vente - frais
             # Ajouter le cash de la vente au cash *existant* le dernier jour dans le portefeuille
             cash_disponible_final = portfolio.loc[dernier_jour, 'cash'] + cash_obtenu


             transactions.append({
                 'Date': dernier_jour, 'Type': 'Vente Fin', 'Quantité': nb_actions_possedees,
                 'Prix_Unitaire': prix_dernier_jour, 'Montant': montant_vente,
                 'Frais': frais, 'Cash_Net': cash_obtenu, 'Raison': 'Fin du Backtest',
                 'Prix_Achat_Moyen': prix_achat_moyen_actif
             })
             # Update the last day's portfolio state to reflect liquidation
             portfolio.loc[dernier_jour, 'actions'] = 0.0
             portfolio.loc[dernier_jour, 'cash'] = cash_disponible_final
             portfolio.loc[dernier_jour, 'valeur_actions'] = 0.0
             portfolio.loc[dernier_jour, 'valeur_totale'] = cash_disponible_final
             portfolio.loc[dernier_jour, 'trade_en_cours'] = False
             portfolio.loc[dernier_jour, 'date_livraison_prevue'] = pd.NaT
             portfolio.loc[dernier_jour, 'prix_achat_moyen'] = 0.0
             portfolio.loc[dernier_jour, 'stop_loss_price'] = np.nan
             portfolio.loc[dernier_jour, 'take_profit_price'] = np.nan

             # Recalculer le rendement pour la dernière journée après liquidation simulée
             if len(data_for_backtest) > 1:
                portfolio.loc[dernier_jour, 'rendement'] = (portfolio.loc[dernier_jour, 'valeur_totale'] - portfolio.loc[data_for_backtest.index[-2], 'valeur_totale']) / portfolio.loc[data_for_backtest.index[-2], 'valeur_totale']
             else:
                portfolio.loc[dernier_jour, 'rendement'] = 0


         transactions_df = pd.DataFrame(transactions)
         if not transactions_df.empty:
              transactions_df = transactions_df.set_index('Date')

         return portfolio, achats_dates, ventes_dates, transactions_df


    # Lancer le Backtest
    try:
        portfolio_df, achat_dates_vis, vente_dates_vis, transactions_df = run_backtest(
            data,
            capital_initial,
            frais_transaction,
            stop_loss,
            take_profit,
            plafond_variation,
            delai_livraison,
            invest_percentage # Passer le nouveau paramètre
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
                portfolio_df['rendement_cumule'] = (portfolio_df['valeur_totale'] / capital_initial) - 1
                portfolio_df['valeur_max'] = portfolio_df['valeur_totale'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['valeur_totale'] - portfolio_df['valeur_max']) / portfolio_df['valeur_max']
                max_drawdown = portfolio_df['drawdown'].min() * 100 if not portfolio_df['drawdown'].empty else 0

                nb_jours_trading = len(portfolio_df)
                if nb_jours_trading > 0:
                     nb_annees = nb_jours_trading / 252.0
                     rendement_annualise = ((1 + rendement_pourcentage/100)**(1/nb_annees) - 1) * 100 if nb_annees > 0 else 0
                     volatilite_journaliere = portfolio_df['rendement'].std()
                     volatilite_annualisee = volatilite_journaliere * np.sqrt(252) * 100 if volatilite_journaliere > 0 else np.nan
                     taux_sans_risque_journalier = (1 + taux_sans_risque)**(1/252) - 1
                     rendement_moyen_journalier = portfolio_df['rendement'].mean()
                     sharpe_ratio = (rendement_moyen_journalier - taux_sans_risque_journalier) / volatilite_journaliere if volatilite_journaliere > 0 else np.nan

                col1, col2, col3, col4_sharpe = st.columns(4)
                col1.metric("Rendement Annualisé (estimé)", f"{rendement_annualise:.2f}%" if nb_jours_trading > 0 else "N/A")
                col2.metric("Volatilité Annualisée (estimée)", f"{volatilite_annualisee:.2f}%" if nb_jours_trading > 0 and not np.isnan(volatilite_annualisee) else "N/A")
                col3.metric("Drawdown Max (%)", f"{max_drawdown:.2f}%")
                col4_sharpe.metric("Sharpe Ratio (estimé)", f"{sharpe_ratio:.2f}" if nb_jours_trading > 0 and not np.isnan(sharpe_ratio) else "N/A")


            # --- Graphique de la Valeur du Portefeuille (Pas de changement) ---
            st.subheader("Évolution de la Valeur du Portefeuille")
            try:
                fig_port, ax_port = plt.subplots(figsize=(12, 6))
                ax_port.plot(portfolio_df.index, portfolio_df['valeur_totale'], label='Valeur du Portefeuille', lw=2)
                ax_port.set_title('Évolution de la Valeur du Portefeuille'); ax_port.set_xlabel('Date'); ax_port.set_ylabel('Valeur (FCFA)')
                ax_port.grid(True, linestyle='--', alpha=0.6); ax_port.legend()
                ax_port.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax_port.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_port.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig_port.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig_port)
            except Exception as e: st.error(f"Erreur graphique portefeuille : {e}")

            # --- Graphique des Rendements Cumulés (Pas de changement) ---
            st.subheader("Rendement Cumulé (%)")
            try:
                fig_ret, ax_ret = plt.subplots(figsize=(12, 6))
                ax_ret.plot(portfolio_df.index, portfolio_df['rendement_cumule'] * 100, label='Rendement Cumulé (%)', lw=2)
                ax_ret.axhline(y=0, color='grey', linestyle='--')
                ax_ret.set_title('Rendement Cumulé du Portefeuille'); ax_ret.set_xlabel('Date'); ax_ret.set_ylabel('Rendement (%)')
                ax_ret.grid(True, linestyle='--', alpha=0.6); ax_ret.legend()
                ax_ret.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.2f}%'))
                ax_ret.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax_ret.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig_ret.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig_ret)
            except Exception as e: st.error(f"Erreur graphique rendement : {e}")


            # --- Graphique des Transactions sur le Cours (Pas de changement dans la logique de plotting, utilise les dates enregistrées) ---
            st.subheader("Transactions sur le Cours")
            try:
                fig_trades, ax_trades = plt.subplots(figsize=(12, 6))
                ax_trades.plot(data.index, data['Prix'], label='Prix de Clôture', lw=1.5, alpha=0.7, zorder=2)

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


            # --- Tableau des Transactions (Pas de changement) ---
            st.subheader("Détail des Transactions")
            if not transactions_df.empty:
                 formatted_transactions = transactions_df.copy()
                 num_cols_to_format = ['Quantité', 'Prix_Unitaire', 'Montant', 'Frais', 'Cash_Net', 'Prix_Achat_Moyen']
                 format_dict = {col: '{:,.2f}' for col in num_cols_to_format}
                 format_dict['Quantité'] = '{:,.0f}'
                 st.dataframe(formatted_transactions.style.format(format_dict))
                 st.markdown(get_csv_download_link(transactions_df, filename=f"transactions_{st.session_state.stock_name}.csv", link_text="Télécharger les transactions (CSV)"), unsafe_allow_html=True)
            else:
                 st.info("Aucune transaction effectuée.")

        else:
            st.warning("Impossible d'exécuter le backtest avec les données fournies.")

    except Exception as e:
        st.error(f"Erreur lors de l'exécution du backtest : {e}")
        st.error(traceback.format_exc())
