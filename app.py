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
        **Version:** 1.2 (Correction NaN Variation)

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

# --- Titre et Introduction ---
st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")
st.sidebar.header("Paramètres Globaux")

# --- Section Upload de Fichier ---
st.sidebar.subheader("1. Chargement des Données")
uploaded_file = st.sidebar.file_uploader("Chargez votre fichier CSV d'historique", type=['csv'])

# --- Fonction de Traitement des Données (Améliorée) ---
def process_data(file, column_mapping, date_format=None):
    """
    Charge, valide et traite les données CSV uploadées.

    Args:
        file: Objet fichier uploadé par Streamlit.
        column_mapping (dict): Dictionnaire mappant les noms standardisés
                                aux noms de colonnes du fichier CSV.
        date_format (str, optional): Format de date à essayer si la conversion échoue.

    Returns:
        pd.DataFrame: DataFrame traité et standardisé, ou None en cas d'erreur.
    """
    if file is None:
        st.error("Veuillez charger un fichier CSV.")
        return None
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys):
         missing_keys = [key for key in required_keys if key not in column_mapping or not column_mapping[key]]
         st.warning(f"Veuillez mapper toutes les colonnes requises ({', '.join(missing_keys)}) dans la barre latérale.")
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
                st.info(f"Séparateur détecté par Sniffer : '{separator}'")
            else:
                 st.warning("L'échantillon du fichier est vide, impossible d'utiliser Sniffer. Utilisation du séparateur par défaut ','.")
        except csv.Error:
            st.warning("Sniffer n'a pas pu déterminer le séparateur. Essai manuel sur la première ligne.")
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
            if header_line.count(';') >= header_line.count(','): separator = ';'
            else: separator = ','
            st.info(f"Utilisation probable du séparateur '{separator}' basé sur le comptage.")


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

        st.write("Colonnes détectées dans le fichier :", list(df.columns))

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
            st.warning(f"{nan_dates_count} valeur(s) Date ('{date_col_name}') invalides supprimées.")
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
                          cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True).str.replace(r'^(-?\.)?$', '', regex=True).str.replace(r'(-.*)-', r'\1', regex=True)
                          converted_series = pd.to_numeric(cleaned_series, errors='coerce')
                     df_standardized[standard_col_name] = converted_series
                else:
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                original_nan_count = df[user_col_name].isnull().sum()
                newly_created_nan = nan_after_conversion - original_nan_count
                if newly_created_nan > 0:
                    st.warning(f"{newly_created_nan} NaN créés dans '{user_col_name}' ({standard_col_name}) lors de la conversion numérique.")
            except Exception as e:
                st.error(f"Erreur conversion numérique colonne '{user_col_name}' ({standard_col_name}) : {e}")
                return None

        # --- Validation Post-Conversion ---
        numeric_standard_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        if df_standardized[numeric_standard_cols].isnull().all().all():
             st.error("Colonnes numériques mappées entièrement vides ou invalides après conversion.")
             return None

        initial_rows = len(df_standardized)
        df_standardized = df_standardized.dropna(subset=['Prix']) # Prix (Close) est essentiel
        rows_dropped = initial_rows - len(df_standardized)
        if rows_dropped > 0:
            st.warning(f"{rows_dropped} lignes supprimées car 'Prix' (colonne '{column_mapping['Close']}') était NaN après conversion.")
        if df_standardized.empty:
            st.error("DataFrame vide après suppression lignes avec 'Prix' NaN.")
            return None

        # --- Traitements Finaux ---
        # Trier par date
        df_standardized = df_standardized.sort_values('Date')

        # Définir l'index et gérer les duplicatas
        # Faire la vérification des duplicats *avant* de définir l'index si 'Date' est encore une colonne
        if df_standardized['Date'].duplicated().any():
            duplicates_count = df_standardized['Date'].duplicated().sum()
            st.warning(f"Il y a {duplicates_count} dates dupliquées dans vos données. Seule la dernière entrée pour chaque date sera conservée.")
            df_standardized = df_standardized.drop_duplicates(subset=['Date'], keep='last')

        df_standardized = df_standardized.set_index('Date')

        # --- Calculer Variation ET REMPLIR NaN INITIAL ---
        if 'Prix' in df_standardized.columns:
            df_standardized['Variation'] = df_standardized['Prix'].diff()
            df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
            # Remplir explicitement le NaN initial (première ligne) avec 0
            df_standardized['Variation'].fillna(0, inplace=True)
            df_standardized['Variation_%'].fillna(0, inplace=True)
            # st.info("NaN initial pour 'Variation' et 'Variation_%' remplacé par 0.") # Optionnel
        else:
             st.error("Colonne 'Prix' manquante, impossible de calculer les variations.")
             return None


        # Remplir les valeurs NaN restantes dans les colonnes OHLCV (méthode ffill puis bfill)
        cols_to_fill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        for col in cols_to_fill:
             if col in df_standardized.columns and df_standardized[col].isnull().any():
                  nan_before = df_standardized[col].isnull().sum()
                  df_standardized[col] = df_standardized[col].ffill() # Remplir avant
                  df_standardized[col] = df_standardized[col].bfill() # Remplir arrière (pour les NaN au début)
                  nan_after = df_standardized[col].isnull().sum()
                  if nan_after < nan_before:
                      st.info(f"{nan_before - nan_after} NaN dans '{col}' remplis par ffill/bfill.")


        # --- Re-vérifier les NaNs après TOUS les remplissages ---
        # Maintenant, Variation et Variation_% ne devraient plus causer cette erreur
        if df_standardized.isnull().any().any():
            cols_with_nan = df_standardized.columns[df_standardized.isnull().any()].tolist()
            st.error(f"Erreur critique: Il reste des valeurs manquantes inattendues après toutes les tentatives de remplissage dans : {cols_with_nan}. Vérifiez la qualité de vos données source, notamment les colonnes OHLCV au début de la série.")
            st.dataframe(df_standardized[df_standardized.isnull().any(axis=1)])
            return None


        st.success("Données chargées et traitées avec succès !")
        return df_standardized

    except pd.errors.EmptyDataError:
        st.error("Erreur : Le fichier CSV semble vide après lecture de l'en-tête.")
        return None
    except KeyError as e:
        st.error(f"Erreur : Problème d'accès à une colonne lors du traitement. Vérifiez le mapping et le contenu du fichier. La colonne '{e}' semble poser problème.")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# --- Fonction pour Lien de Téléchargement CSV ---
def get_csv_download_link(df, filename="rapport_backtest.csv", link_text="Télécharger le rapport (CSV)"):
    """Génère un lien pour télécharger un DataFrame en CSV."""
    try:
        csv_string = df.to_csv(index=True, date_format='%Y-%m-%d %H:%M:%S')
        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        button_style = """...""" # Style CSS (identique)
        button_hover_style = """...""" # Style CSS (identique)
        st.markdown(f'<style>{button_hover_style}</style>', unsafe_allow_html=True)
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur création lien téléchargement : {e}")
        return ""


# --- Interface Utilisateur dans la Sidebar (Après Upload) ---
column_mapping = {}
date_format_input = None
data = None

if uploaded_file is not None:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier correspondant aux champs requis.")
    try:
        # ... (logique de lecture entête et détection séparateur identique) ...
        uploaded_file.seek(0)
        sample_bytes = uploaded_file.read(2048)
        uploaded_file.seek(0)
        try: sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

        sniffer = csv.Sniffer()
        sep = ',' # défaut
        try:
            if sample_text.strip():
                dialect = sniffer.sniff(sample_text)
                sep = dialect.delimiter
        except csv.Error:
             if sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','): sep = ';'

        # Lire juste l'en-tête
        try:
            df_cols = pd.read_csv(uploaded_file, sep=sep, nrows=0)
            all_columns = df_cols.columns.tolist()
        except Exception as e:
            st.sidebar.error(f"Impossible lire colonnes (Vérifiez séparateur/format): {e}")
            all_columns = []

        uploaded_file.seek(0) # Revenir au début

        if not all_columns:
            st.sidebar.error("Aucune colonne lue depuis le fichier.")
        else:
            st.sidebar.write("Colonnes trouvées :", all_columns)
            required_map = {
                "Date": "Date", "Open": "Ouverture", "High": "Plus Haut",
                "Low": "Plus Bas", "Close": "Clôture", "Volume": "Volume"
            }
            # ... (logique de pré-sélection identique) ...
            pre_selected_indices = {}
            used_columns = set()
            # Priorité 1: Exact match
            for standard_name in required_map.keys():
                normalized_standard = standard_name.lower().replace('_','')
                for idx, col in enumerate(all_columns):
                    normalized_col = col.lower().replace('_','')
                    if normalized_standard == normalized_col and col not in used_columns:
                        pre_selected_indices[standard_name] = idx + 1
                        used_columns.add(col)
                        break
            # Priorité 2: Partial match
            common_terms = { 'Date': ['date', 'time', 'jour'], 'Open': ['open', 'ouverture', 'ouv'], 'High': ['high', 'haut', 'max'], 'Low': ['low', 'bas', 'min'], 'Close': ['close', 'cloture', 'dernier', 'last', 'prix'], 'Volume': ['volume', 'vol', 'quantite', 'qty'] }
            for standard_name, terms in common_terms.items():
                 if standard_name not in pre_selected_indices:
                    for term in terms:
                        found_match = False
                        for idx, col in enumerate(all_columns):
                            if term in col.lower().replace('_','') and col not in used_columns:
                                pre_selected_indices[standard_name] = idx + 1
                                used_columns.add(col)
                                found_match = True
                                break
                        if found_match: break

            # Création des selectbox
            for standard_name, display_name in required_map.items():
                 default_index = pre_selected_indices.get(standard_name, 0)
                 column_mapping[standard_name] = st.sidebar.selectbox(f"Colonne pour '{display_name}'", options=[""] + all_columns, index=default_index, key=f"map_{standard_name}")

            with st.sidebar.expander("Options Avancées"):
                 date_format_input = st.text_input("Format de date (si conversion auto échoue, ex: %d/%m/%Y)", key="date_format", help="Exemples: %Y-%m-%d, %d/%m/%Y %H:%M:%S. Voir Python strptime.")

            if st.sidebar.button("▶️ Traiter les Données", key="process_button"):
                missing_maps = [name for name, col in column_mapping.items() if not col]
                if missing_maps:
                     st.warning(f"Veuillez mapper les colonnes suivantes : {', '.join(missing_maps)}")
                else:
                     mapped_cols = [col for col in column_mapping.values() if col]
                     if len(mapped_cols) != len(set(mapped_cols)):
                          st.warning("Attention: La même colonne CSV a été sélectionnée pour plusieurs champs. Vérifiez votre mapping.")
                     else:
                          with st.spinner("Traitement des données en cours..."):
                              data = process_data(uploaded_file, column_mapping, date_format_input or None)

    except Exception as e:
        st.sidebar.error(f"Erreur lecture initiale fichier : {e}")
        st.sidebar.info("Assurez-vous que le fichier est un CSV valide (encodage UTF-8 ou Latin-1).")


# --- Exécution de l'Analyse (si les données sont chargées) ---
# (Le reste du code à partir d'ici est identique à la version précédente)
if data is not None and not data.empty:

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    stock_name = st.sidebar.text_input("Nom de l'action", "MonActionBRVM", key="stock_name")
    st.title(f"📈 BRVM Quant Backtest - {stock_name}")

    # --- Affichage des Données Traitées ---
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)"):
        st.dataframe(data.tail(100).style.format({
            'Ouverture': '{:,.2f}', 'Plus_Haut': '{:,.2f}', 'Plus_Bas': '{:,.2f}',
            'Prix': '{:,.2f}', 'Volume': '{:,.0f}', 'Variation': '{:,.2f}',
            'Variation_%': '{:.2f}%'
        }))
        st.markdown(get_csv_download_link(data.tail(100), filename=f"data_preview_{stock_name}.csv", link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)

    # --- Visualisation du Cours ---
    st.subheader(f"Cours historique de {stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture')
        ax.set_title(f'Évolution du cours de {stock_name}')
        ax.set_xlabel('Date'); ax.set_ylabel('Prix (FCFA)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate()
        plt.tight_layout(); st.pyplot(fig)
    except Exception as e: st.error(f"Erreur génération graphique cours : {e}")

    # --- Paramètres de la Stratégie ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")
    # Fondamental
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    dividende_annuel = st.sidebar.number_input("Dernier dividende annuel (FCFA)", min_value=0.0, value=600.0, step=10.0, key="dividend")
    taux_croissance = st.sidebar.slider("Croissance annuelle dividende (%)", -10.0, 15.0, 3.0, 0.5, key="growth_rate") / 100
    rendement_exige = st.sidebar.slider("Taux d'actualisation (%)", 5.0, 30.0, 12.0, 0.5, key="discount_rate") / 100

    val_intrinseque = None
    if rendement_exige <= taux_croissance: st.sidebar.error("Actualisation <= Croissance!")
    else:
        try:
             if dividende_annuel <= 0: val_intrinseque = 0
             else:
                 D1 = dividende_annuel * (1 + taux_croissance)
                 val_intrinseque = D1 / (rendement_exige - taux_croissance)
             if val_intrinseque < 0: st.sidebar.warning(f"VI négative ({val_intrinseque:,.2f}).")
             elif pd.notna(val_intrinseque): st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")
        except Exception as e: st.sidebar.error(f"Erreur calcul VI: {e}")

    use_fundamental_signals = False
    if val_intrinseque is not None and val_intrinseque > 0: use_fundamental_signals = True; st.sidebar.info("Signaux VI activés.")
    else: st.sidebar.warning("VI invalide (<=0). Signaux VI désactivés.")

    # Technique
    st.sidebar.markdown("**Règles de Trading Techniques**")
    if use_fundamental_signals:
        marge_achat = st.sidebar.slider("Marge achat / VI (%)", 0, 50, 20, key="buy_margin") / 100
        marge_vente = st.sidebar.slider("Prime sortie / VI (%)", 0, 50, 10, key="sell_premium") / 100
    else: marge_achat = 0; marge_vente = 0; st.sidebar.caption("Marges VI désactivées.")
    stop_loss = st.sidebar.slider("Stop Loss / Prix Achat (%)", 1.0, 30.0, 10.0, 0.5, key="stop_loss") / 100
    take_profit = st.sidebar.slider("Take Profit / Prix Achat (%)", 5.0, 100.0, 20.0, 1.0, key="take_profit") / 100

    # Moyennes Mobiles
    st.sidebar.markdown("**Indicateurs Techniques (Moyennes Mobiles)**")
    window_court = st.sidebar.slider("Fenêtre MM Courte (j)", 5, 100, 20, key="short_ma")
    window_long = st.sidebar.slider("Fenêtre MM Longue (j)", 20, 250, 50, key="long_ma")
    if window_court >= window_long: st.sidebar.warning("MM Courte >= MM Longue.")

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
    # Calcul MM
    try:
        if len(data) < window_long: st.warning(f"Pas assez données ({len(data)}j) pour MM Longue ({window_long}j)."); st.stop()
        elif len(data) < window_court: st.warning(f"Pas assez données ({len(data)}j) pour MM Courte ({window_court}j)."); st.stop()
        data['MM_Court'] = data['Prix'].rolling(window=window_court, min_periods=window_court).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long, min_periods=window_long).mean()
    except Exception as e: st.error(f"Erreur calcul MM : {e}"); st.stop()

    # Graphique MM
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', lw=1, alpha=0.8, zorder=2)
        ax2.plot(data.index[window_court-1:], data['MM_Court'].dropna(), label=f'MM {window_court}j', lw=1.5, zorder=3)
        ax2.plot(data.index[window_long-1:], data['MM_Long'].dropna(), label=f'MM {window_long}j', lw=1.5, zorder=3)
        ax2.set_title('Analyse Technique - Moyennes Mobiles'); ax2.set_xlabel('Date'); ax2.set_ylabel('Prix (FCFA)')
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=1); ax2.legend()
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig2.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig2)
    except Exception as e: st.error(f"Erreur graphique MM : {e}")

    # Niveaux Fondamentaux
    if use_fundamental_signals:
        data['val_intrinseque'] = val_intrinseque
        data['prix_achat_fondamental'] = (1 - marge_achat) * val_intrinseque
        data['prix_vente_fondamental'] = (1 + marge_vente) * val_intrinseque
    else: data['val_intrinseque'], data['prix_achat_fondamental'], data['prix_vente_fondamental'] = np.nan, np.nan, np.nan

    # Signaux Techniques MM
    data['signal_technique'] = 0
    valid_ma = (data['MM_Court'].notna()) & (data['MM_Long'].notna())
    buy_cond = valid_ma & (data['MM_Court'] > data['MM_Long']) & (data['MM_Court'].shift(1) <= data['MM_Long'].shift(1))
    sell_cond = valid_ma & (data['MM_Court'] < data['MM_Long']) & (data['MM_Court'].shift(1) >= data['MM_Long'].shift(1))
    data.loc[buy_cond, 'signal_technique'] = 1
    data.loc[sell_cond, 'signal_technique'] = -1

    # Signaux Combinés Achat/Vente
    cond_achat_tech = (data['signal_technique'] == 1)
    cond_vente_tech = (data['signal_technique'] == -1)
    if use_fundamental_signals:
        cond_achat_fond = (data['Prix'] < data['prix_achat_fondamental'])
        cond_vente_fond = (data['Prix'] > data['prix_vente_fondamental'])
        data['achat'] = cond_achat_tech & cond_achat_fond
        data['vente_signal'] = cond_vente_tech | cond_vente_fond
    else:
        data['achat'] = cond_achat_tech
        data['vente_signal'] = cond_vente_tech

    # Graphique Signaux
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', lw=1.5, zorder=2)
        if use_fundamental_signals and pd.notna(val_intrinseque):
            ax3.axhline(y=val_intrinseque, color='grey', ls='-', alpha=0.7, label=f'VI ({val_intrinseque:,.0f})', zorder=1)
            if pd.notna(data['prix_achat_fondamental'].iloc[0]): ax3.axhline(y=data['prix_achat_fondamental'].iloc[0], color='green', ls='--', alpha=0.6, label=f'Seuil Achat VI ({data["prix_achat_fondamental"].iloc[0]:,.0f})', zorder=1)
            if pd.notna(data['prix_vente_fondamental'].iloc[0]): ax3.axhline(y=data['prix_vente_fondamental'].iloc[0], color='red', ls='--', alpha=0.6, label=f'Seuil Vente VI ({data["prix_vente_fondamental"].iloc[0]:,.0f})', zorder=1)

        achats_sig = data[data['achat']]
        ventes_sig = data[data['vente_signal']]
        if not achats_sig.empty: ax3.scatter(achats_sig.index, achats_sig['Prix'], color='lime', edgecolor='green', s=70, marker='^', label='Signal Achat Strat', zorder=5)
        if not ventes_sig.empty: ax3.scatter(ventes_sig.index, ventes_sig['Prix'], color='tomato', edgecolor='red', s=70, marker='v', label='Signal Vente Strat', zorder=5)

        ax3.set_title('Prix et Signaux Trading Initiaux'); ax3.set_xlabel('Date'); ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, linestyle='--', alpha=0.6, zorder=1); ax3.legend(loc='best')
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig3.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig3)
    except Exception as e: st.error(f"Erreur graphique signaux : {e}"); import traceback; st.error(traceback.format_exc())

    # --- Backtest ---
    st.subheader("🚀 Backtest de la Stratégie")
    st.markdown(f"Capital: **{capital_initial:,.0f} FCFA**, Frais: **{frais_transaction*100:.2f}%**, Plafond: **{plafond_variation*100:.1f}%**, Livraison: **{delai_livraison}j**.")

    # Fonction Backtest (identique à la version précédente)
    def run_backtest(data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison, use_fondamental_signals):
        if not isinstance(data.index, pd.DatetimeIndex):
            st.error("Index non DatetimeIndex pour backtest."); return pd.DataFrame(), [], [], pd.DataFrame()
        portfolio = pd.DataFrame(index=data.index)
        portfolio['prix_effectif'], portfolio['actions'], portfolio['cash'] = 0.0, 0.0, float(capital_initial)
        portfolio['valeur_actions'], portfolio['valeur_totale'] = 0.0, float(capital_initial)
        portfolio['rendement'], portfolio['trade_en_cours'] = 0.0, False
        portfolio['date_livraison_prevue'], portfolio['prix_achat_moyen'] = pd.NaT, 0.0
        transactions, achats_dates, ventes_dates = [], [], []
        nb_actions_possedees, cash_disponible = 0.0, float(capital_initial)
        prix_achat_moyen_actif = 0.0
        trade_en_cours_boucle, date_livraison_boucle = False, pd.NaT
        bday = BDay()

        for i, (jour, row) in enumerate(data.iterrows()):
            prix_jour_brut = row['Prix']
            prix_veille_eff = portfolio.loc[data.index[i-1], 'prix_effectif'] if i > 0 else prix_jour_brut
            variation = (prix_jour_brut - prix_veille_eff) / prix_veille_eff if prix_veille_eff != 0 else 0
            prix_effectif_jour = prix_jour_brut
            log_plafond = ""
            if abs(variation) > plafond_variation:
                prix_effectif_jour = prix_veille_eff * (1 + (np.sign(variation) * plafond_variation))
                log_plafond = f"(Plafond {plafond_variation*100:.1f}%, Prix->{prix_effectif_jour:,.2f})"
            portfolio.loc[jour, 'prix_effectif'] = prix_effectif_jour

            if i > 0:
                 jour_prec = data.index[i-1]
                 nb_actions_possedees = portfolio.loc[jour_prec, 'actions']; cash_disponible = portfolio.loc[jour_prec, 'cash']
                 trade_en_cours_boucle = portfolio.loc[jour_prec, 'trade_en_cours']; date_livraison_boucle = portfolio.loc[jour_prec, 'date_livraison_prevue']
                 prix_achat_moyen_actif = portfolio.loc[jour_prec, 'prix_achat_moyen']
                 if trade_en_cours_boucle and pd.notna(date_livraison_boucle) and jour >= date_livraison_boucle:
                      trade_en_cours_boucle = False; date_livraison_boucle = pd.NaT

            if not trade_en_cours_boucle:
                vendre, raison_vente = False, ""
                if nb_actions_possedees > 0:
                    if prix_achat_moyen_actif > 0 and prix_effectif_jour < prix_achat_moyen_actif * (1 - stop_loss): vendre, raison_vente = True, f"Stop Loss ({stop_loss*100:.1f}%)"
                    elif prix_achat_moyen_actif > 0 and prix_effectif_jour > prix_achat_moyen_actif * (1 + take_profit): vendre, raison_vente = True, f"Take Profit ({take_profit*100:.1f}%)"
                    elif row['vente_signal']: vendre, raison_vente = True, "Signal Vente Strat"
                    if vendre:
                        montant_brut = nb_actions_possedees * prix_effectif_jour; frais = montant_brut * frais_transaction; montant_net = montant_brut - frais
                        cash_disponible += montant_net; ventes_dates.append(jour); date_livraison_op = jour + bday * delai_livraison
                        transactions.append({'Date Ordre': jour, 'Date Livraison': date_livraison_op, 'Type': 'Vente', 'Raison': raison_vente, 'Quantité': nb_actions_possedees, 'Prix Unitaire': prix_effectif_jour, 'Frais': frais, 'Montant Net': montant_net})
                        # st.write(f"🔔 {jour.date()}: {raison_vente}. Vente {nb_actions_possedees:.0f}@{prix_effectif_jour:,.2f} {log_plafond}. Net: +{montant_net:,.2f}. Liv: {date_livraison_op.date()}")
                        nb_actions_possedees, prix_achat_moyen_actif = 0.0, 0.0
                        trade_en_cours_boucle, date_livraison_boucle = True, date_livraison_op

                if not vendre and nb_actions_possedees == 0 and row['achat']:
                    if cash_disponible > 0:
                        cout_action_frais = prix_effectif_jour * (1 + frais_transaction)
                        if cout_action_frais > 0:
                            nb_actions = int(cash_disponible // cout_action_frais)
                            if nb_actions > 0:
                                cout_brut = nb_actions * prix_effectif_jour; frais = cout_brut * frais_transaction; cout_total = cout_brut + frais
                                if cash_disponible >= cout_total:
                                    cash_disponible -= cout_total; achats_dates.append(jour); date_livraison_op = jour + bday * delai_livraison
                                    transactions.append({'Date Ordre': jour, 'Date Livraison': date_livraison_op, 'Type': 'Achat', 'Raison': 'Signal Achat Strat', 'Quantité': nb_actions, 'Prix Unitaire': prix_effectif_jour, 'Frais': frais, 'Montant Net': -cout_total})
                                    # st.write(f"🔔 {jour.date()}: Signal Achat. Achat {nb_actions:.0f}@{prix_effectif_jour:,.2f} {log_plafond}. Coût: {cout_total:,.2f}. Liv: {date_livraison_op.date()}")
                                    nb_actions_possedees, prix_achat_moyen_actif = nb_actions, prix_effectif_jour
                                    trade_en_cours_boucle, date_livraison_boucle = True, date_livraison_op

            portfolio.loc[jour, 'actions'], portfolio.loc[jour, 'cash'] = nb_actions_possedees, cash_disponible
            portfolio.loc[jour, 'valeur_actions'] = nb_actions_possedees * prix_effectif_jour
            portfolio.loc[jour, 'valeur_totale'] = portfolio.loc[jour, 'cash'] + portfolio.loc[jour, 'valeur_actions']
            portfolio.loc[jour, 'trade_en_cours'], portfolio.loc[jour, 'date_livraison_prevue'] = trade_en_cours_boucle, date_livraison_boucle
            portfolio.loc[jour, 'prix_achat_moyen'] = prix_achat_moyen_actif

            if i > 0:
                valeur_veille = portfolio.loc[data.index[i-1], 'valeur_totale']
                portfolio.loc[jour, 'rendement'] = (portfolio.loc[jour, 'valeur_totale'] / valeur_veille - 1) if valeur_veille is not None and valeur_veille != 0 else 0.0
            else: portfolio.loc[jour, 'rendement'] = 0.0

        portfolio['rendement'] = portfolio['rendement'].fillna(0.0)
        portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty: transactions_df = transactions_df.sort_values('Date Ordre').set_index('Date Ordre')
        return portfolio if portfolio is not None else pd.DataFrame(), achats_dates, ventes_dates, transactions_df if transactions_df is not None else pd.DataFrame()
    # --- Fin fonction backtest ---

    # Exécution Backtest
    try:
        with st.spinner("Exécution du backtest..."):
            backtest_results = run_backtest(data.copy(), capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison, use_fundamental_signals)
        if backtest_results is None or not isinstance(backtest_results, tuple) or len(backtest_results) != 4 or backtest_results[0] is None:
             st.error("Backtest n'a pas retourné de résultat valide."); st.stop()
        else:
             portfolio_history, achats_dates, ventes_dates, journal_transactions = backtest_results
             if portfolio_history.empty: st.warning("Backtest exécuté, mais aucun historique de portefeuille produit.")
             else: st.success("Backtest terminé.")
             if journal_transactions is None: journal_transactions = pd.DataFrame() # Ensure df for later steps

    except Exception as e: st.error(f"Erreur durant backtest : {e}"); import traceback; st.error(traceback.format_exc()); st.stop()

    # --- Affichage Résultats Backtest ---
    st.subheader("📊 Résultats du Backtest")
    if not portfolio_history.empty:
        # Statistiques Clés
        try:
            valeur_finale = portfolio_history['valeur_totale'].iloc[-1]
            rendement_total_pct = (valeur_finale / capital_initial - 1) * 100 if capital_initial != 0 else 0
            start_date, end_date = portfolio_history.index[0], portfolio_history.index[-1]
            jours_total = (end_date - start_date).days
            rendement_annualise_pct = ((valeur_finale / capital_initial) ** (365.25 / jours_total) - 1) * 100 if jours_total > 0 and capital_initial != 0 and valeur_finale > 0 else (-100.0 if valeur_finale <= 0 else 0.0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Valeur Finale", f"{valeur_finale:,.2f} FCFA", f"{valeur_finale-capital_initial:,.2f} FCFA")
            col2.metric("Rendement Total", f"{rendement_total_pct:.2f}%")
            col3.metric("Rendt Annualisé", f"{rendement_annualise_pct:.2f}%" if pd.notna(rendement_annualise_pct) else "N/A")
        except Exception as e: st.error(f"Erreur calcul stats perf : {e}")
        # Graphique Évolution Portefeuille
        try:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.plot(portfolio_history.index, portfolio_history['valeur_totale'], lw=2, label='Valeur Portefeuille', color='blue')
            ax4.axhline(y=capital_initial, ls='--', lw=1, color='grey', label='Capital Initial')
            ax4.set_title('Évolution Valeur Portefeuille'); ax4.set_xlabel('Date'); ax4.set_ylabel('Valeur (FCFA)')
            ax4.grid(True, linestyle='--', alpha=0.6); ax4.legend()
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax4.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            fig4.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig4)
        except Exception as e: st.error(f"Erreur graphique évolution portefeuille : {e}")
        # Composition Finale
        st.subheader("💼 Composition Portefeuille Final")
        try:
            last = portfolio_history.iloc[-1]
            col1, col2, col3 = st.columns(3)
            col1.metric("Actions détenues", f"{last['actions']:,.0f}")
            col2.metric("Liquidités", f"{last['cash']:,.2f} FCFA")
            status = "Oui" if last['trade_en_cours'] else "Non"
            if last['trade_en_cours'] and pd.notna(last['date_livraison_prevue']): status += f" (Liv: {last['date_livraison_prevue']:%Y-%m-%d})"
            col3.metric("Trade en attente?", status)
        except Exception as e: st.error(f"Erreur affichage composition finale : {e}")
        # Journal Transactions
        with st.expander("📜 Journal des Transactions"):
            if journal_transactions is not None and not journal_transactions.empty:
                st.dataframe(journal_transactions.style.format({'Date Livraison': '{:%Y-%m-%d}', 'Quantité': '{:,.0f}', 'Prix Unitaire': '{:,.2f}', 'Frais': '{:,.2f}', 'Montant Net': '{:,.2f}'}))
                st.markdown(get_csv_download_link(journal_transactions, filename=f"transactions_{stock_name}.csv", link_text="Télécharger Journal (CSV)"), unsafe_allow_html=True)
            else: st.info("Aucune transaction effectuée.")
        # Métriques Avancées
        st.subheader("⚙️ Métriques Performance Avancées")
        try:
            ret = portfolio_history['rendement'].dropna()
            vol_strat = ret.std(ddof=1) * np.sqrt(252) * 100 if len(ret) >= 2 else np.nan
            vol_ann = vol_strat / 100 if pd.notna(vol_strat) else np.nan
            sharpe = ((rendement_annualise_pct/100 - taux_sans_risque) / vol_ann) if pd.notna(rendement_annualise_pct) and pd.notna(vol_ann) and vol_ann != 0 else np.nan
            portfolio_history['peak'] = portfolio_history['valeur_totale'].cummax()
            portfolio_history['drawdown'] = ((portfolio_history['valeur_totale'] - portfolio_history['peak']) / portfolio_history['peak']).replace([np.inf, -np.inf], np.nan).fillna(0)
            max_dd = portfolio_history['drawdown'].min() * 100 if not portfolio_history['drawdown'].empty else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Volatilité Ann.", f"{vol_strat:.2f}%" if pd.notna(vol_strat) else "N/A")
            col2.metric("Ratio Sharpe", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A", help=f"TSR={taux_sans_risque*100:.1f}%")
            col3.metric("Drawdown Max", f"{max_dd:.2f}%")
            # Graphique Drawdown
            fig5, ax5 = plt.subplots(figsize=(12, 4))
            ax5.fill_between(portfolio_history.index, portfolio_history['drawdown']*100, 0, color='red', alpha=0.3)
            ax5.set_title('Drawdown Portefeuille'); ax5.set_xlabel('Date'); ax5.set_ylabel('Drawdown (%)')
            ax5.grid(True, linestyle='--', alpha=0.6)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax5.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            fig5.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig5)
            # Graphique Distribution Rendements
            if not ret.empty:
                fig6, ax6 = plt.subplots(figsize=(10, 4)); ax6.hist(ret * 100, bins=50, alpha=0.75, density=True)
                ax6.set_title('Distribution Rendements Journaliers Stratégie'); ax6.set_xlabel('Rendt Journalier (%)'); ax6.set_ylabel('Densité')
                ax6.grid(True, alpha=0.3); rendt_moyen = ret.mean() * 100
                ax6.axvline(rendt_moyen, color='red', ls='dashed', lw=1, label=f'Moy: {rendt_moyen:.3f}%'); ax6.legend()
                plt.tight_layout(); st.pyplot(fig6)
        except Exception as e: st.error(f"Erreur calcul/affichage métriques avancées : {e}"); import traceback; st.error(traceback.format_exc())
        # Comparaison Buy & Hold
        st.subheader("⚖️ Comparaison avec 'Buy & Hold'")
        try:
            px_init, px_fin = data['Prix'].iloc[0], data['Prix'].iloc[-1]
            rendt_bh = (px_fin / px_init - 1) * 100 if px_init != 0 else 0
            rendt_ann_bh = ((px_fin / px_init) ** (365.25 / jours_total) - 1) * 100 if jours_total > 0 and px_init != 0 and px_fin > 0 else (-100.0 if px_fin <= 0 else 0.0)
            ret_bh = data['Prix'].pct_change().fillna(0.0)
            vol_bh = ret_bh.std(ddof=1) * np.sqrt(252) * 100 if len(ret_bh) >= 2 else np.nan
            data['peak_bh'] = data['Prix'].cummax()
            data['dd_bh'] = ((data['Prix'] - data['peak_bh']) / data['peak_bh']).replace([np.inf, -np.inf], np.nan).fillna(0) if px_init != 0 else 0
            max_dd_bh = data['dd_bh'].min() * 100 if not data['dd_bh'].empty else 0
            st.markdown("### Performance Buy & Hold")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendt Total B&H", f"{rendt_bh:.2f}%")
            col2.metric("Rendt Ann. B&H", f"{rendt_ann_bh:.2f}%" if pd.notna(rendt_ann_bh) else "N/A")
            col3.metric("Max Drawdown B&H", f"{max_dd_bh:.2f}%")
            st.markdown("### Comparaison Directe")
            sup_tot = (rendement_total_pct - rendt_bh) if pd.notna(rendement_total_pct) and pd.notna(rendt_bh) else np.nan
            sup_ann = (rendement_annualise_pct - rendt_ann_bh) if pd.notna(rendement_annualise_pct) and pd.notna(rendt_ann_bh) else np.nan
            diff_vol = (vol_strat - vol_bh) if pd.notna(vol_strat) and pd.notna(vol_bh) else np.nan
            col1, col2, col3 = st.columns(3)
            col1.metric("Surperf. (Total)", f"{sup_tot:.2f}%" if pd.notna(sup_tot) else "N/A")
            col2.metric("Surperf. (Ann.)", f"{sup_ann:.2f}%" if pd.notna(sup_ann) else "N/A")
            col3.metric("Diff. Volatilité", f"{diff_vol:.2f}%" if pd.notna(diff_vol) else "N/A", help="Négatif = Stratégie moins volatile")
            # Graphique Comparatif
            fig7, ax7 = plt.subplots(figsize=(12, 6)); plot_strat, plot_bh = False, False
            if 'rendement_cumule' in portfolio_history.columns and not portfolio_history.empty:
                ax7.plot(portfolio_history.index, (1 + portfolio_history['rendement_cumule']) * capital_initial, label=f'Stratégie ({stock_name})', lw=2, color='blue'); plot_strat = True
            if px_init != 0:
                ax7.plot(data.index, (data['Prix'] / px_init) * capital_initial, label=f'Buy & Hold ({stock_name})', lw=2, ls='--', color='orange'); plot_bh = True
            if plot_strat or plot_bh:
                ax7.set_title('Comparaison Perf. Normalisées'); ax7.set_xlabel('Date'); ax7.set_ylabel(f'Valeur (Base {capital_initial:,.0f} FCFA)')
                ax7.grid(True, linestyle='--', alpha=0.6); ax7.legend()
                ax7.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')); ax7.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig7.autofmt_xdate(); plt.tight_layout(); st.pyplot(fig7)
        except Exception as e: st.error(f"Erreur comparaison B&H : {e}"); import traceback; st.error(traceback.format_exc())
        # Téléchargement Rapport
        st.subheader("📥 Télécharger le Rapport Complet")
        st.markdown(get_csv_download_link(portfolio_history, filename=f"rapport_backtest_{stock_name}.csv", link_text="Télécharger Historique Portefeuille (CSV)"), unsafe_allow_html=True)
        # Note interprétation
        st.info("""**Note sur l'interprétation :** ... (identique) ...""")
    else:
         st.warning("Le backtest n'a pas produit de résultats à afficher (historique portefeuille vide).")

else:
    # Messages initiaux
    if uploaded_file is None: st.info("👈 Veuillez charger un fichier CSV via la barre latérale.")
    elif data is None and uploaded_file is not None: pass # Erreurs gérées ailleurs
    elif data is not None and data.empty: st.error("❌ Traitement a résulté en DataFrame vide. Vérifiez fichier/mapping.")

# --- Pied de page ---
st.markdown("---"); st.markdown("Application Backtesting BRVM v1.2")
