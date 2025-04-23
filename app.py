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
        **Version:** 1.1 (Améliorée)

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
    if not all(col in column_mapping and column_mapping[col] for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
         st.warning("Veuillez mapper toutes les colonnes requises (Date, Open, High, Low, Close, Volume) dans la barre latérale.")
         return None

    try:
        # Lire l'en-tête pour prévisualisation et détection du séparateur
        file.seek(0)
        # Lire quelques octets pour éviter de lire une ligne potentiellement énorme
        sample_bytes = file.read(2048)
        file.seek(0)
        # Essayer de décoder en utf-8, fallback sur latin-1 si besoin
        try:
            sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError:
             sample_text = sample_bytes.decode('latin-1', errors='ignore')

        # Utiliser Sniffer sur le sample texte
        sniffer = csv.Sniffer()
        separator = ',' # Séparateur par défaut
        try:
            # Utiliser StringIO pour que Sniffer lise le texte comme un fichier
            # Vérifier que sample_text n'est pas vide
            if sample_text.strip():
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
                st.info(f"Séparateur détecté par Sniffer : '{separator}'")
            else:
                 st.warning("L'échantillon du fichier est vide, impossible d'utiliser Sniffer. Utilisation du séparateur par défaut ','.")

        except csv.Error:
            # Si Sniffer échoue, tenter une détection manuelle simple sur la première ligne
            st.warning("Sniffer n'a pas pu déterminer le séparateur. Essai manuel sur la première ligne.")
            file.seek(0)
            try:
                header_line_bytes = file.readline()
                try:
                    header_line = header_line_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    header_line = header_line_bytes.decode('latin-1', errors='ignore')
            except Exception as read_err:
                 st.error(f"Impossible de lire la première ligne pour la détection manuelle du séparateur: {read_err}")
                 return None
            file.seek(0)

            if header_line.count(';') >= header_line.count(','):
                 separator = ';'
                 st.info("Utilisation probable du séparateur ';' basé sur le comptage.")
            else:
                 separator = ','
                 st.info("Utilisation probable du séparateur ',' basé sur le comptage.")


        file.seek(0) # Revenir au début pour la lecture par pandas
        # Essayer avec l'encodage utf-8, puis latin-1 si ça échoue
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

        # --- Validation Initiale ---
        if df.empty:
            st.error("Le fichier CSV est vide ou n'a pas pu être lu correctement par Pandas.")
            return None

        st.write("Colonnes détectées dans le fichier :", list(df.columns))

        # Vérifier si les colonnes mappées existent
        missing_mapped_cols = []
        for standard_name, user_name in column_mapping.items():
            if not user_name: # Si l'utilisateur n'a rien sélectionné pour ce champ
                missing_mapped_cols.append(standard_name)
            elif user_name not in df.columns:
                st.error(f"La colonne mappée '{user_name}' pour '{standard_name}' n'existe pas dans le fichier.")
                return None
        if missing_mapped_cols:
             st.error(f"Veuillez mapper les colonnes suivantes : {', '.join(missing_mapped_cols)}")
             return None


        # --- Création du DataFrame Standardisé ---
        df_standardized = pd.DataFrame()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Date
        date_col_name = column_mapping['Date']
        try:
            # Essayer la conversion directe, inférer le format si possible
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce', infer_datetime_format=True)

            # Si échec et format spécifié, essayer avec le format
            # Vérifier si TOUTES les dates sont NaT ET qu'un format est donné
            if df_standardized['Date'].isnull().all() and date_format:
                 st.info(f"Tentative de conversion de date avec le format explicite : {date_format}")
                 try:
                    # Passer une copie pour éviter SettingWithCopyWarning potentiel
                    df_copy = df[[date_col_name]].copy()
                    df_standardized['Date'] = pd.to_datetime(df_copy[date_col_name], format=date_format, errors='coerce')
                 except ValueError as ve:
                      st.error(f"Le format de date spécifié '{date_format}' est invalide ou ne correspond pas aux données dans la colonne '{date_col_name}'. Erreur: {ve}")
                      return None
                 except Exception as fmt_e:
                      st.error(f"Erreur inattendue lors de l'application du format de date '{date_format}' à la colonne '{date_col_name}': {fmt_e}")
                      return None

        except Exception as e:
            st.error(f"Erreur générale lors de la conversion de la colonne Date ('{date_col_name}'): {e}")
            st.info("Assurez-vous que le format de date est cohérent dans la colonne ou spécifiez-le explicitement dans les options avancées si nécessaire.")
            return None

        # Vérification après conversion de date
        if df_standardized['Date'].isnull().all():
             st.error(f"Impossible de convertir la colonne Date ('{date_col_name}') en dates valides, même avec le format optionnel.")
             st.info("Vérifiez le contenu de la colonne et le format de date.")
             return None
        if df_standardized['Date'].isnull().any():
            nan_dates_count = df_standardized['Date'].isnull().sum()
            st.warning(f"{nan_dates_count} valeur(s) dans la colonne Date ('{date_col_name}') n'ont pas pu être converties et les lignes correspondantes ont été supprimées.")
            df_standardized = df_standardized.dropna(subset=['Date'])
            if df_standardized.empty:
                 st.error("Toutes les lignes ont été supprimées après échec de conversion des dates.")
                 return None

        # Colonnes Numériques
        standard_to_user_map = {
            'Ouverture': column_mapping['Open'],
            'Plus_Haut': column_mapping['High'],
            'Plus_Bas': column_mapping['Low'],
            'Prix': column_mapping['Close'], # 'Prix' est le nom standard pour 'Close'
            'Volume': column_mapping['Volume']
        }

        for standard_col_name, user_col_name in standard_to_user_map.items():
            try:
                # Rendre le nettoyage plus robuste
                if df[user_col_name].dtype == 'object':
                     # 1. Supprimer les espaces de début/fin
                     # 2. Remplacer les virgules par des points (pour les décimaux)
                     # 3. Supprimer TOUT ce qui n'est pas chiffre, point, ou signe moins (placé correctement)
                     #    Attention: ceci peut être trop agressif si des milliers sont séparés par des espaces ou points.
                     #    Une approche plus sûre pourrait cibler des caractères spécifiques à enlever (ex: ' FCFA', '$', etc.)
                     #    Ici, on essaie une version plus générale.
                     cleaned_series = df[user_col_name].astype(str).str.strip()
                     # Remplacer la virgule décimale
                     cleaned_series = cleaned_series.str.replace(',', '.', regex=False)
                     # Supprimer les séparateurs de milliers (espaces)
                     cleaned_series = cleaned_series.str.replace(r'\s+', '', regex=True)
                     # Essayer de convertir directement après nettoyage simple
                     converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                     # Si la conversion simple échoue, tenter un nettoyage plus agressif (supprimer tout sauf chiffres, point, signe -)
                     if converted_series.isnull().all() and not df[user_col_name].isnull().all():
                          st.info(f"Nettoyage simple insuffisant pour '{user_col_name}', tentative de nettoyage agressif.")
                          cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                          # Gérer les cas comme "--", ".-", etc.
                          cleaned_series = cleaned_series.str.replace(r'^(-?\.)?$', '', regex=True) # Vide si juste point ou -
                          cleaned_series = cleaned_series.str.replace(r'(-.*)-', r'\1', regex=True) # Enlève les '-' supplémentaires
                          converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                     df_standardized[standard_col_name] = converted_series

                else:
                     # Si ce n'est pas 'object', tenter la conversion directe
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                # Compter les NaN introduits par la conversion
                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                original_nan_count = df[user_col_name].isnull().sum()
                newly_created_nan = nan_after_conversion - original_nan_count

                if newly_created_nan > 0:
                    st.warning(f"{newly_created_nan} valeur(s) dans la colonne '{user_col_name}' ({standard_col_name}) n'ont pas pu être converties en numérique et ont été remplacées par NaN.")

            except Exception as e:
                st.error(f"Erreur lors du traitement de la colonne numérique '{user_col_name}' ({standard_col_name}) : {e}")
                return None

        # --- Validation Post-Conversion ---
        numeric_standard_cols = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        # Vérifier si toutes les colonnes numériques sont entièrement NaN
        if df_standardized[numeric_standard_cols].isnull().all().all():
             st.error("Toutes les valeurs dans les colonnes numériques mappées sont manquantes ou invalides après conversion.")
             return None

        # Supprimer les lignes où le prix est manquant (essentiel)
        initial_rows = len(df_standardized)
        df_standardized = df_standardized.dropna(subset=['Prix'])
        rows_dropped = initial_rows - len(df_standardized)
        if rows_dropped > 0:
            st.warning(f"{rows_dropped} lignes supprimées car la valeur 'Prix' (colonne '{column_mapping['Close']}') était manquante ou invalide après conversion.")

        if df_standardized.empty:
            st.error("Le DataFrame est vide après suppression des lignes avec 'Prix' manquant.")
            return None

        # --- Traitements Finaux ---
        # Trier par date (important pour les calculs temporels)
        df_standardized = df_standardized.sort_values('Date')

        # Vérifier s'il y a des dates dupliquées après tri
        if df_standardized['Date'].duplicated().any():
            duplicates_count = df_standardized['Date'].duplicated().sum()
            st.warning(f"Il y a {duplicates_count} dates dupliquées dans vos données. Seule la dernière entrée pour chaque date sera conservée.")
            # Conserver la dernière occurrence pour chaque date
            df_standardized = df_standardized[~df_standardized['Date'].duplicated(keep='last')]


        # Définir la date comme index
        df_standardized = df_standardized.set_index('Date')

        # Calculer Variation
        df_standardized['Variation'] = df_standardized['Prix'].diff()
        df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100

        # Remplir les valeurs NaN restantes (méthode ffill puis bfill pour plus de robustesse)
        # Ffill remplit avec la valeur précédente, bfill avec la suivante (utile pour les NaN au début)
        cols_to_fill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']
        for col in cols_to_fill:
             if col in df_standardized.columns and df_standardized[col].isnull().any():
                  nan_before = df_standardized[col].isnull().sum()
                  # Remplir d'abord avec la valeur précédente
                  df_standardized[col] = df_standardized[col].ffill()
                  # Puis remplir les NaN restants (au début) avec la valeur suivante
                  df_standardized[col] = df_standardized[col].bfill()
                  nan_after = df_standardized[col].isnull().sum()
                  if nan_after < nan_before:
                      fill_method = "ffill/bfill"
                      if col == 'Volume':
                          # Optionnel: Remplir le Volume restant par 0 au lieu de bfill
                          # df_standardized[col] = df_standardized[col].fillna(0)
                          # nan_after = df_standardized[col].isnull().sum()
                          # if nan_after < nan_before:
                          #    fill_method = "ffill/0"
                          pass # Garder ffill/bfill pour Volume aussi pour l'instant

                      st.info(f"{nan_before - nan_after} valeur(s) manquante(s) dans '{col}' remplies par {fill_method}.")


        # Re-vérifier les NaNs après remplissage (ne devrait plus y en avoir si bfill a fonctionné)
        if df_standardized.isnull().any().any():
            cols_with_nan = df_standardized.columns[df_standardized.isnull().any()].tolist()
            st.error(f"Erreur critique: Il reste des valeurs manquantes après les tentatives de remplissage dans : {cols_with_nan}. Cela peut arriver si une colonne est entièrement vide. Le traitement ne peut continuer.")
            # Optionnel: Afficher les lignes avec NaN
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
        st.error(traceback.format_exc()) # Afficher la trace complète pour le débogage
        return None


# --- Fonction pour Lien de Téléchargement CSV ---
def get_csv_download_link(df, filename="rapport_backtest.csv", link_text="Télécharger le rapport (CSV)"):
    """Génère un lien pour télécharger un DataFrame en CSV."""
    try:
        # S'assurer que l'index est inclus et correctement formaté
        csv_string = df.to_csv(index=True, date_format='%Y-%m-%d %H:%M:%S') # Format ISO pour date/heure si applicable
        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8') # Encodage UTF-8
        # Style CSS pour un bouton plus joli
        button_style = """
        display: inline-block;
        padding: 8px 15px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 5px;
        text-align: center;
        text-decoration: none;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        """
        # Ajouter un effet au survol
        button_hover_style = """
        a.download-button:hover {
            background-color: #0056b3;
        }
        """
        st.markdown(f'<style>{button_hover_style}</style>', unsafe_allow_html=True)
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur lors de la création du lien de téléchargement : {e}")
        return ""


# --- Interface Utilisateur dans la Sidebar (Après Upload) ---
column_mapping = {}
date_format_input = None
data = None # Initialiser data à None

if uploaded_file is not None:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.info("Sélectionnez les colonnes de votre fichier correspondant aux champs requis.")

    try:
        # Essayer de lire juste l'en-tête pour obtenir les colonnes
        uploaded_file.seek(0)
        # Lire un échantillon pour le Sniffer
        sample_bytes = uploaded_file.read(2048)
        uploaded_file.seek(0)
        try:
            sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError:
             sample_text = sample_bytes.decode('latin-1', errors='ignore')

        # Détecter le séparateur
        sniffer = csv.Sniffer()
        sep = ',' # défaut
        try:
            if sample_text.strip():
                dialect = sniffer.sniff(sample_text)
                sep = dialect.delimiter
        except csv.Error:
             # Fallback simple si Sniffer échoue
             if sample_text.split('\n')[0].count(';') >= sample_text.split('\n')[0].count(','):
                  sep = ';'

        # Lire juste l'en-tête avec le séparateur détecté
        try:
            df_cols = pd.read_csv(uploaded_file, sep=sep, nrows=0)
            all_columns = df_cols.columns.tolist()
        except Exception as e:
            st.sidebar.error(f"Impossible de lire les colonnes du fichier (Vérifiez séparateur/format): {e}")
            all_columns = [] # Pas de colonnes à mapper

        uploaded_file.seek(0) # Important de revenir au début pour la lecture finale

        if not all_columns:
            st.sidebar.error("Aucune colonne n'a pu être lue depuis le fichier.")
        else:
            st.sidebar.write("Colonnes trouvées :", all_columns)
            required_map = {
                "Date": "Date",
                "Open": "Ouverture",
                "High": "Plus Haut",
                "Low": "Plus Bas",
                "Close": "Clôture",
                "Volume": "Volume"
            }

            # Logique de pré-sélection améliorée
            pre_selected_indices = {}
            used_columns = set() # Pour éviter de mapper la même colonne CSV à plusieurs champs

            # Priorité 1: Correspondance exacte (ignorant casse et _)
            for standard_name in required_map.keys():
                normalized_standard = standard_name.lower().replace('_','')
                for idx, col in enumerate(all_columns):
                    normalized_col = col.lower().replace('_','')
                    if normalized_standard == normalized_col and col not in used_columns:
                        pre_selected_indices[standard_name] = idx + 1 # +1 car on a ""
                        used_columns.add(col)
                        break

            # Priorité 2: Correspondance partielle commune (ignorant casse et _)
            common_terms = {
                'Date': ['date', 'time', 'jour'],
                'Open': ['open', 'ouverture', 'ouv'],
                'High': ['high', 'haut', 'max'],
                'Low': ['low', 'bas', 'min'],
                'Close': ['close', 'cloture', 'dernier', 'last', 'prix'],
                'Volume': ['volume', 'vol', 'quantite', 'qty']
            }
            for standard_name, terms in common_terms.items():
                 if standard_name not in pre_selected_indices: # Si pas déjà trouvé
                    for term in terms:
                        found_match = False
                        for idx, col in enumerate(all_columns):
                            if term in col.lower().replace('_','') and col not in used_columns:
                                pre_selected_indices[standard_name] = idx + 1
                                used_columns.add(col)
                                found_match = True
                                break # Passer au standard_name suivant dès qu'un match est trouvé
                        if found_match:
                            break


            # Création des selectbox avec pré-sélection
            for standard_name, display_name in required_map.items():
                 default_index = pre_selected_indices.get(standard_name, 0) # 0 est pour l'option vide ""
                 column_mapping[standard_name] = st.sidebar.selectbox(
                      f"Colonne pour '{display_name}'",
                      options=[""] + all_columns, # Ajouter option vide
                      index=default_index,
                      key=f"map_{standard_name}"
                 )

            with st.sidebar.expander("Options Avancées"):
                 date_format_input = st.text_input("Format de date (si conversion auto échoue, ex: %d/%m/%Y)", key="date_format",
                                                   help="Exemples: %Y-%m-%d, %d/%m/%Y %H:%M:%S. Voir Python strptime.")

            # Bouton pour lancer le traitement
            if st.sidebar.button("▶️ Traiter les Données", key="process_button"):
                # Vérification si tous les mappings sont faits
                missing_maps = [name for name, col in column_mapping.items() if not col]
                if missing_maps:
                     st.warning(f"Veuillez mapper les colonnes suivantes avant de traiter : {', '.join(missing_maps)}")
                else:
                     # Vérifier si la même colonne CSV est mappée plusieurs fois
                     mapped_cols = [col for col in column_mapping.values() if col]
                     if len(mapped_cols) != len(set(mapped_cols)):
                          st.warning("Attention: La même colonne CSV a été sélectionnée pour plusieurs champs. Veuillez vérifier votre mapping.")
                     else:
                          with st.spinner("Traitement des données en cours..."):
                              data = process_data(uploaded_file, column_mapping, date_format_input or None) # Passer None si vide

    except pd.errors.EmptyDataError:
        st.sidebar.error("Le fichier CSV semble vide.")
    except Exception as e:
        st.sidebar.error(f"Erreur lors de la lecture initiale du fichier : {e}")
        st.sidebar.info("Assurez-vous que le fichier est un CSV valide et que l'encodage est compatible (UTF-8 ou Latin-1).")


# --- Exécution de l'Analyse (si les données sont chargées) ---
if data is not None and not data.empty:

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    stock_name = st.sidebar.text_input("Nom de l'action", "MonActionBRVM", key="stock_name")
    st.title(f"📈 BRVM Quant Backtest - {stock_name}") # Mettre à jour le titre principal

    # --- Affichage des Données Traitées ---
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)"):
        # Afficher avec un formatage amélioré pour les nombres
        st.dataframe(data.tail(100).style.format({
            'Ouverture': '{:,.2f}',
            'Plus_Haut': '{:,.2f}',
            'Plus_Bas': '{:,.2f}',
            'Prix': '{:,.2f}',
            'Volume': '{:,.0f}',
            'Variation': '{:,.2f}',
            'Variation_%': '{:.2f}%'
        }))
        st.markdown(get_csv_download_link(data.tail(100), filename=f"data_preview_{stock_name}.csv", link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)

    # --- Visualisation du Cours ---
    st.subheader(f"Cours historique de {stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture') # Ligne un peu plus fine
        ax.set_title(f'Évolution du cours de {stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix (FCFA)') # Supposer FCFA, pourrait être paramétrable
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        # Améliorer la lisibilité de l'axe Y (formatage des grands nombres)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


        # Amélioration du format des dates sur l'axe X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Format plus précis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10)) # Ajustement automatique
        fig.autofmt_xdate() # Rotation et alignement automatique des dates

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique de cours : {e}")


    # --- Paramètres de la Stratégie (dans la Sidebar) ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")

    # Paramètres fondamentaux
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    dividende_annuel = st.sidebar.number_input("Dernier dividende annuel (FCFA)", min_value=0.0, value=600.0, step=10.0, help="Dividende versé lors de la dernière période.", key="dividend")
    taux_croissance = st.sidebar.slider("Croissance annuelle dividende (%)", -10.0, 15.0, 3.0, 0.5, help="Taux de croissance annuel attendu des dividendes (peut être négatif).", key="growth_rate") / 100
    rendement_exige = st.sidebar.slider("Taux d'actualisation (%)", 5.0, 30.0, 12.0, 0.5, help="Taux de rendement minimum exigé par l'investisseur.", key="discount_rate") / 100

    # Validation Gordon-Shapiro
    val_intrinseque = None # Initialiser
    if rendement_exige <= taux_croissance:
         st.sidebar.error("Le taux d'actualisation doit être supérieur au taux de croissance des dividendes pour le modèle de Gordon-Shapiro.")
         # Ne pas stopper ici, permettre à l'utilisateur d'ajuster
    else:
        # Calcul Valeur Intrinsèque
        try:
             # Assurer que le dividende est positif pour que le modèle ait du sens
             if dividende_annuel <= 0:
                 st.sidebar.warning("Le dividende annuel est nul ou négatif. La valeur intrinsèque basée sur Gordon-Shapiro sera nulle ou négative.")
                 val_intrinseque = 0 # Ou np.nan, selon la préférence
             else:
                 D1 = dividende_annuel * (1 + taux_croissance)
                 val_intrinseque = D1 / (rendement_exige - taux_croissance)

             if val_intrinseque < 0:
                 st.sidebar.warning(f"Valeur intrinsèque calculée négative ({val_intrinseque:,.2f} FCFA). Vérifiez les paramètres (dividende > 0, croissance < actualisation). Le modèle peut ne pas être adapté.")
             elif pd.notna(val_intrinseque):
                 st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")

        except Exception as e:
             st.sidebar.error(f"Erreur calcul valeur intrinsèque: {e}")
             # Ne pas stopper, val_intrinseque restera None

    # Si la valeur intrinsèque n'est pas valide pour la stratégie
    use_fundamental_signals = False
    if val_intrinseque is not None and val_intrinseque > 0:
        use_fundamental_signals = True
        st.sidebar.info("Signaux fondamentaux (Achat/Vente basés sur VI) activés.")
    else:
        st.sidebar.warning("Valeur intrinsèque non calculée ou invalide (<=0). Les signaux d'achat/vente basés sur la VI seront désactivés. Seuls les signaux techniques (MM) et SL/TP seront utilisés.")


    # Paramètres Techniques et Trading Rules
    st.sidebar.markdown("**Règles de Trading Techniques**")
    if use_fundamental_signals:
        marge_achat = st.sidebar.slider("Marge de sécurité à l'achat / VI (%)", 0, 50, 20, help="Achat si Prix < (1 - Marge) * Valeur Intrinsèque (combiné avec signal MM).", key="buy_margin") / 100
        marge_vente = st.sidebar.slider("Prime de sortie / VI (%)", 0, 50, 10, help="Signal de Vente si Prix > (1 + Prime) * Valeur Intrinsèque (combiné avec signal MM).", key="sell_premium") / 100
    else:
        # Mettre des valeurs par défaut non utilisées si signaux fonda désactivés
        marge_achat = 0
        marge_vente = 0
        st.sidebar.caption("Marge d'achat / VI et Prime de sortie / VI désactivées.")

    stop_loss = st.sidebar.slider("Stop Loss / Prix Achat (%)", 1.0, 30.0, 10.0, 0.5, help="Vente si le prix baisse de ce % par rapport au prix d'achat moyen.", key="stop_loss") / 100
    take_profit = st.sidebar.slider("Take Profit / Prix Achat (%)", 5.0, 100.0, 20.0, 1.0, help="Vente si le prix augmente de ce % par rapport au prix d'achat moyen.", key="take_profit") / 100

    # Moyennes Mobiles
    st.sidebar.markdown("**Indicateurs Techniques (Moyennes Mobiles)**")
    window_court = st.sidebar.slider("Fenêtre Moyenne Mobile Courte (jours)", 5, 100, 20, key="short_ma")
    window_long = st.sidebar.slider("Fenêtre Moyenne Mobile Longue (jours)", 20, 250, 50, key="long_ma")

    # Vérification fenêtres MA
    if window_court >= window_long:
        st.sidebar.warning("La fenêtre courte doit être inférieure à la fenêtre longue.")
        # Optionnel: ajuster automatiquement ou laisser l'utilisateur corriger
        # window_court = max(5, window_long - 10) # Exemple d'ajustement

    # Paramètres spécifiques à la BRVM
    st.sidebar.markdown("**Paramètres Marché (BRVM)**")
    plafond_variation = st.sidebar.slider("Plafond variation journalière (%)", 5.0, 15.0, 7.5, 0.5, help="Variation maximale autorisée par jour (ex: 7.5%). Simulé sur le prix de clôture.", key="variation_cap") / 100
    delai_livraison = st.sidebar.slider("Délai de livraison (jours ouvrés)", 1, 5, 3, help="Nombre de jours ouvrés pour la livraison des titres après transaction (T+N). Bloque les trades pendant ce délai.", key="settlement_days")

    # Paramètres Backtest
    st.sidebar.subheader("5. Paramètres du Backtest")
    capital_initial = st.sidebar.number_input("Capital initial (FCFA)", 100000, 100000000, 1000000, step=100000, key="initial_capital")
    frais_transaction = st.sidebar.slider("Frais de transaction (%) par ordre", 0.0, 5.0, 0.5, 0.05, help="Pourcentage de frais appliqué à chaque achat et vente.", key="commission_rate") / 100
    taux_sans_risque = st.sidebar.slider("Taux sans risque annuel (%)", 0.0, 10.0, 3.0, 0.1, help="Taux utilisé pour le calcul du Ratio de Sharpe.", key="risk_free_rate") / 100


    # --- Calculs Techniques et Signaux ---
    st.subheader("Analyse Technique et Signaux")

    # Calcul des moyennes mobiles
    try:
        # S'assurer qu'il y a assez de données pour les fenêtres
        if len(data) < window_long:
             st.warning(f"Pas assez de données ({len(data)} jours) pour calculer la moyenne mobile longue sur {window_long} jours.")
             st.stop()
        elif len(data) < window_court:
             st.warning(f"Pas assez de données ({len(data)} jours) pour calculer la moyenne mobile courte sur {window_court} jours.")
             st.stop()

        data['MM_Court'] = data['Prix'].rolling(window=window_court, min_periods=window_court).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long, min_periods=window_long).mean()
    except Exception as e:
        st.error(f"Erreur lors du calcul des moyennes mobiles : {e}")
        st.stop()

    # Affichage du graphique avec moyennes mobiles
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', linewidth=1, alpha=0.8, zorder=2)
        # Afficher les MM seulement où elles sont valides (non-NaN)
        # Utiliser zorder pour mettre les MM au-dessus de la grille mais sous le prix si alpha<1
        ax2.plot(data.index[window_court-1:], data['MM_Court'].dropna(), label=f'MM {window_court} jours', linewidth=1.5, zorder=3)
        ax2.plot(data.index[window_long-1:], data['MM_Long'].dropna(), label=f'MM {window_long} jours', linewidth=1.5, zorder=3)
        ax2.set_title('Analyse Technique - Moyennes Mobiles')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prix (FCFA)')
        ax2.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax2.legend()
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig2.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique des moyennes mobiles : {e}")


    # Calcul des niveaux de prix pour signaux fondamentaux (si applicable)
    if use_fundamental_signals:
        data['val_intrinseque'] = val_intrinseque
        data['prix_achat_fondamental'] = (1 - marge_achat) * val_intrinseque
        data['prix_vente_fondamental'] = (1 + marge_vente) * val_intrinseque
    else:
        # Mettre NaN si non utilisés pour éviter erreurs/confusion
        data['val_intrinseque'] = np.nan
        data['prix_achat_fondamental'] = np.nan
        data['prix_vente_fondamental'] = np.nan

    # Signal technique: croisement des moyennes mobiles
    data['signal_technique'] = 0
    valid_ma_rows = (data['MM_Court'].notna()) & (data['MM_Long'].notna())
    # Acheter QUAND la courte CROISE AU DESSUS la longue
    buy_signal_cond = valid_ma_rows & (data['MM_Court'] > data['MM_Long']) & (data['MM_Court'].shift(1) <= data['MM_Long'].shift(1))
    data.loc[buy_signal_cond, 'signal_technique'] = 1
    # Vendre QUAND la courte CROISE EN DESSOUS la longue
    sell_signal_cond = valid_ma_rows & (data['MM_Court'] < data['MM_Long']) & (data['MM_Court'].shift(1) >= data['MM_Long'].shift(1))
    data.loc[sell_signal_cond, 'signal_technique'] = -1

    # --- Conditions de Trading (Logique combinée) ---
    # Achat: Signal technique d'achat ET (si signaux fonda actifs, Prix < Seuil Fondamental)
    condition_achat_technique = (data['signal_technique'] == 1)
    if use_fundamental_signals:
        condition_achat_fondamentale = (data['Prix'] < data['prix_achat_fondamental'])
        data['achat'] = condition_achat_technique & condition_achat_fondamentale
    else:
        data['achat'] = condition_achat_technique # Seul le signal technique compte

    # Vente (Signal): Signal technique de vente OU (si signaux fonda actifs, Prix > Seuil Fondamental)
    condition_vente_technique = (data['signal_technique'] == -1)
    if use_fundamental_signals:
        condition_vente_fondamentale = (data['Prix'] > data['prix_vente_fondamental'])
        data['vente_signal'] = condition_vente_technique | condition_vente_fondamentale
    else:
        data['vente_signal'] = condition_vente_technique # Seul le signal technique compte
    # Note: La vente réelle dans le backtest inclura aussi Stop Loss / Take Profit


    # Affichage du graphique avec zones et signaux
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', linewidth=1.5, zorder=2)

        # Afficher les lignes fondamentales seulement si elles sont utilisées
        if use_fundamental_signals and pd.notna(val_intrinseque):
            ax3.axhline(y=val_intrinseque, color='grey', linestyle='-', alpha=0.7, label=f'Valeur Intrinsèque ({val_intrinseque:,.0f})', zorder=1)
            if pd.notna(data['prix_achat_fondamental'].iloc[0]):
                ax3.axhline(y=data['prix_achat_fondamental'].iloc[0], color='green', linestyle='--', alpha=0.6, label=f'Seuil Achat Fondam. ({data["prix_achat_fondamental"].iloc[0]:,.0f})', zorder=1)
            if pd.notna(data['prix_vente_fondamental'].iloc[0]):
                 ax3.axhline(y=data['prix_vente_fondamental'].iloc[0], color='red', linestyle='--', alpha=0.6, label=f'Seuil Vente Fondam. ({data["prix_vente_fondamental"].iloc[0]:,.0f})', zorder=1)

        # Marquage des signaux déclenchés (basé sur la logique combinée 'achat' et 'vente_signal')
        achats_signaux = data[data['achat']]
        ventes_signaux = data[data['vente_signal']] # Signaux initiaux (avant SL/TP)

        if not achats_signaux.empty:
            ax3.scatter(achats_signaux.index, achats_signaux['Prix'], color='lime', edgecolor='green', s=70, marker='^', label='Signal Achat Stratégie', zorder=5)

        if not ventes_signaux.empty:
            # Filtrer pour plus de clarté si souhaité
            ventes_visibles = ventes_signaux # Afficher tous les signaux de vente initiaux
            ax3.scatter(ventes_visibles.index, ventes_visibles['Prix'], color='tomato', edgecolor='red', s=70, marker='v', label='Signal Vente Stratégie', zorder=5)

        ax3.set_title('Prix et Signaux de Trading Initiaux')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, linestyle='--', alpha=0.6, zorder=1)
        ax3.legend(loc='best') # Ajustement auto de la légende
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig3.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig3)
    except Exception as e:
         st.error(f"Erreur lors de la génération du graphique des signaux : {e}")
         import traceback
         st.error(traceback.format_exc())

    # --- Backtest ---
    st.subheader("🚀 Backtest de la Stratégie")
    st.markdown(f"Exécution du backtest avec un capital initial de **{capital_initial:,.0f} FCFA** et des frais de **{frais_transaction*100:.2f}%**.")

    # Fonction pour exécuter le backtest (Améliorée avec Jours Ouvrés et gestion cash)
    def run_backtest(data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison, use_fondamental_signals):
        """
        Exécute le backtest en tenant compte des jours ouvrés pour la livraison.
        Args:
            data (pd.DataFrame): Données avec prix et signaux.
            ... (autres args)
            use_fondamental_signals (bool): Indique si les signaux fondamentaux sont actifs.
        Returns: tuple: (pd.DataFrame: historique portefeuille, list: dates achat, list: dates vente, pd.DataFrame: journal transactions)
        """
        # S'assurer que l'index est bien de type DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            st.error("L'index du DataFrame doit être de type DatetimeIndex pour le calcul des jours ouvrés.")
            # Retourner une structure vide pour éviter un crash plus loin
            return pd.DataFrame(), [], [], pd.DataFrame()


        portfolio = pd.DataFrame(index=data.index)
        portfolio['prix_effectif'] = 0.0  # Prix utilisé pour la transaction du jour (après plafond)
        portfolio['actions'] = 0.0        # Actions réellement détenues
        portfolio['cash'] = float(capital_initial) # Liquidités disponibles
        portfolio['valeur_actions'] = 0.0 # Valeur des actions détenues
        portfolio['valeur_totale'] = float(capital_initial) # Valeur totale (cash + actions)
        portfolio['rendement'] = 0.0      # Rendement journalier
        portfolio['trade_en_cours'] = False # Indicateur si une opération est en attente de livraison
        portfolio['date_livraison_prevue'] = pd.NaT # Date prévue pour la fin du blocage
        portfolio['prix_achat_moyen'] = 0.0 # Suivi du prix d'achat moyen pour SL/TP

        transactions = [] # Pour stocker les détails des trades
        achats_dates = []
        ventes_dates = []

        # Initialisation des variables suivies dans la boucle
        nb_actions_possedees = 0.0
        cash_disponible = float(capital_initial)
        prix_achat_moyen_actif = 0.0 # Prix moyen de la position actuelle
        trade_en_cours_boucle = False
        date_livraison_boucle = pd.NaT

        # Utiliser BDay pour les jours ouvrés
        bday = BDay()

        # Itérer sur les jours où les données sont disponibles
        for i, (jour, row) in enumerate(data.iterrows()):

            # --- Appliquer le Plafond de Variation (Simulation BRVM) ---
            prix_jour_brut = row['Prix']
            if i == 0: # Premier jour, pas de veille
                prix_veille_eff = prix_jour_brut # Ou utiliser 'Ouverture' si disponible?
            else:
                # Utiliser le prix effectif de la veille stocké dans le portfolio
                prix_veille_eff = portfolio.loc[data.index[i-1], 'prix_effectif']

            variation = (prix_jour_brut - prix_veille_eff) / prix_veille_eff if prix_veille_eff != 0 else 0
            prix_effectif_jour = prix_jour_brut # Prix utilisé pour les transactions
            log_plafond = ""
            if abs(variation) > plafond_variation:
                if variation > 0:
                    prix_effectif_jour = prix_veille_eff * (1 + plafond_variation)
                else:
                    prix_effectif_jour = prix_veille_eff * (1 - plafond_variation)
                log_plafond = f"(Plafond {plafond_variation*100:.1f}% atteint, Prix ajusté à {prix_effectif_jour:,.2f})"

            # Stocker le prix effectif du jour
            portfolio.loc[jour, 'prix_effectif'] = prix_effectif_jour

            # --- Vérification Livraison et état de la veille ---
            if i > 0:
                 jour_prec = data.index[i-1]
                 # Récupérer l'état *avant* la transaction potentielle du jour
                 nb_actions_possedees = portfolio.loc[jour_prec, 'actions']
                 cash_disponible = portfolio.loc[jour_prec, 'cash']
                 trade_en_cours_boucle = portfolio.loc[jour_prec, 'trade_en_cours']
                 date_livraison_boucle = portfolio.loc[jour_prec, 'date_livraison_prevue']
                 prix_achat_moyen_actif = portfolio.loc[jour_prec, 'prix_achat_moyen'] # Récupérer le prix moyen de la veille

                 # Vérifier si une livraison a lieu AUJOURD'HUI
                 if trade_en_cours_boucle and pd.notna(date_livraison_boucle) and jour >= date_livraison_boucle:
                      trade_en_cours_boucle = False # Livraison effectuée, on peut trader à nouveau
                      date_livraison_boucle = pd.NaT
                      # Les actions/cash sont déjà à jour au moment de l'ordre dans cette logique
                      # Le flag trade_en_cours est juste pour bloquer/débloquer

            # --- Logique de Trading (seulement si pas de trade en cours) ---
            if not trade_en_cours_boucle:
                vendre = False
                raison_vente = ""
                # === Vérification VENTE (priorité sur l'achat si les deux signaux le même jour) ===
                if nb_actions_possedees > 0:
                    # 1. Stop Loss
                    if prix_achat_moyen_actif > 0 and prix_effectif_jour < prix_achat_moyen_actif * (1 - stop_loss):
                        vendre = True
                        raison_vente = f"Stop Loss ({stop_loss*100:.1f}%)"
                    # 2. Take Profit
                    elif prix_achat_moyen_actif > 0 and prix_effectif_jour > prix_achat_moyen_actif * (1 + take_profit):
                        vendre = True
                        raison_vente = f"Take Profit ({take_profit*100:.1f}%)"
                    # 3. Signal de Vente Stratégie (basé sur 'vente_signal' du jour)
                    elif row['vente_signal']:
                        vendre = True
                        raison_vente = "Signal Vente Stratégie"

                    if vendre:
                        montant_vente_brut = nb_actions_possedees * prix_effectif_jour
                        frais = montant_vente_brut * frais_transaction
                        montant_vente_net = montant_vente_brut - frais
                        cash_disponible += montant_vente_net # Argent reçu après livraison
                        ventes_dates.append(jour)
                        date_livraison_op = jour + bday * delai_livraison # Calcul date livraison ouvrée

                        transactions.append({
                            'Date Ordre': jour, 'Date Livraison': date_livraison_op, 'Type': 'Vente',
                            'Raison': raison_vente, 'Quantité': nb_actions_possedees,
                            'Prix Unitaire': prix_effectif_jour, 'Frais': frais, 'Montant Net': montant_vente_net
                        })
                        st.write(f"🔔 {jour.date()}: {raison_vente}. Vente de {nb_actions_possedees:.0f} actions à {prix_effectif_jour:,.2f} {log_plafond}. Cash net: +{montant_vente_net:,.2f}. Livraison: {date_livraison_op.date()}")

                        nb_actions_possedees = 0.0 # Actions vendues
                        prix_achat_moyen_actif = 0.0 # Reset prix moyen
                        trade_en_cours_boucle = True # Bloquer nouvelles opérations
                        date_livraison_boucle = date_livraison_op

                # === Vérification ACHAT (seulement si on n'a pas vendu et qu'on n'a pas d'actions) ===
                if not vendre and nb_actions_possedees == 0 and row['achat']:
                    if cash_disponible > 0:
                        cout_par_action_avec_frais = prix_effectif_jour * (1 + frais_transaction)
                        if cout_par_action_avec_frais > 0:
                            nb_actions_a_acheter = int(cash_disponible // cout_par_action_avec_frais) # // pour division entière

                            if nb_actions_a_acheter > 0:
                                cout_achat_brut = nb_actions_a_acheter * prix_effectif_jour
                                frais = cout_achat_brut * frais_transaction
                                cout_achat_total = cout_achat_brut + frais

                                # Vérifier si on a assez de cash (double check)
                                if cash_disponible >= cout_achat_total:
                                    cash_disponible -= cout_achat_total # Cash débité
                                    achats_dates.append(jour)
                                    date_livraison_op = jour + bday * delai_livraison

                                    transactions.append({
                                        'Date Ordre': jour, 'Date Livraison': date_livraison_op, 'Type': 'Achat',
                                        'Raison': 'Signal Achat Stratégie', 'Quantité': nb_actions_a_acheter,
                                        'Prix Unitaire': prix_effectif_jour, 'Frais': frais, 'Montant Net': -cout_achat_total
                                    })
                                    st.write(f"🔔 {jour.date()}: Signal Achat. Achat de {nb_actions_a_acheter:.0f} actions à {prix_effectif_jour:,.2f} {log_plafond}. Coût total: {cout_achat_total:,.2f}. Livraison: {date_livraison_op.date()}")

                                    # Mettre à jour l'état *après* la transaction
                                    nb_actions_possedees = nb_actions_a_acheter # On les aura à la livraison
                                    prix_achat_moyen_actif = prix_effectif_jour # MAJ prix moyen
                                    trade_en_cours_boucle = True # Bloquer nouvelles opérations
                                    date_livraison_boucle = date_livraison_op
                                else:
                                     st.warning(f"{jour.date()}: Fonds insuffisants pour acheter {nb_actions_a_acheter} actions (Besoin: {cout_achat_total:.2f}, Dispo: {cash_disponible:.2f})")


            # --- Mise à jour quotidienne du portefeuille (état à la fin de la journée) ---
            portfolio.loc[jour, 'actions'] = nb_actions_possedees
            portfolio.loc[jour, 'cash'] = cash_disponible
            # La valeur des actions est basée sur le prix *effectif* du jour
            portfolio.loc[jour, 'valeur_actions'] = nb_actions_possedees * prix_effectif_jour
            portfolio.loc[jour, 'valeur_totale'] = portfolio.loc[jour, 'cash'] + portfolio.loc[jour, 'valeur_actions']
            portfolio.loc[jour, 'trade_en_cours'] = trade_en_cours_boucle
            portfolio.loc[jour, 'date_livraison_prevue'] = date_livraison_boucle
            portfolio.loc[jour, 'prix_achat_moyen'] = prix_achat_moyen_actif # Sauvegarder le prix moyen pour le jour suivant


            # Calcul du rendement quotidien (basé sur la valeur totale de la veille)
            if i > 0:
                jour_prec = data.index[i-1]
                valeur_totale_veille = portfolio.loc[jour_prec, 'valeur_totale']
                if valeur_totale_veille is not None and valeur_totale_veille != 0:
                     portfolio.loc[jour, 'rendement'] = (portfolio.loc[jour, 'valeur_totale'] / valeur_totale_veille) - 1
                else:
                     portfolio.loc[jour, 'rendement'] = 0.0
            else: # Premier jour
                 portfolio.loc[jour, 'rendement'] = 0.0


        # --- Fin de la boucle ---

        # Calcul des rendements cumulés à la fin
        portfolio['rendement'] = portfolio['rendement'].fillna(0.0)
        portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1

        # Créer DataFrame des transactions
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df = transactions_df.sort_values('Date Ordre').set_index('Date Ordre')

        # Retourner les résultats, s'assurer que portfolio n'est pas None
        return portfolio if portfolio is not None else pd.DataFrame(), achats_dates, ventes_dates, transactions_df if transactions_df is not None else pd.DataFrame()
    # --- Fin fonction run_backtest ---


    # Exécution
    try:
        with st.spinner("Exécution du backtest..."):
            backtest_results = run_backtest(
                data.copy(), # Passer une copie pour éviter modif de l'original
                capital_initial,
                frais_transaction,
                stop_loss,
                take_profit,
                plafond_variation,
                delai_livraison,
                use_fundamental_signals # Passer l'info si signaux fonda sont actifs
            )
        # Vérifier si le backtest a retourné une structure valide
        if backtest_results is None or not isinstance(backtest_results, tuple) or len(backtest_results) != 4 or backtest_results[0] is None:
             st.error("Le backtest n'a pas pu s'exécuter correctement ou a retourné un résultat invalide.")
             st.stop()
        else:
             portfolio_history, achats_dates, ventes_dates, journal_transactions = backtest_results
             # Vérifier si le dataframe du portfolio est vide
             if portfolio_history.empty:
                  st.warning("Le backtest s'est exécuté mais n'a produit aucun historique de portefeuille (peut arriver si les données sont très courtes ou aucun trade n'a eu lieu).")
                  # Initialiser journal_transactions s'il est None pour éviter erreurs suivantes
                  if journal_transactions is None: journal_transactions = pd.DataFrame()
             else:
                  st.success("Backtest terminé.")

    except Exception as e:
        st.error(f"Une erreur est survenue durant l'exécution du backtest : {e}")
        import traceback
        st.error(traceback.format_exc()) # Afficher la trace pour le débogage
        st.stop()


    # --- Affichage des Résultats du Backtest ---
    st.subheader("📊 Résultats du Backtest")

    # Continuer seulement si portfolio_history n'est pas vide
    if not portfolio_history.empty:
        # Statistiques Clés
        try:
            valeur_finale = portfolio_history['valeur_totale'].iloc[-1]
            rendement_total_pct = (valeur_finale / capital_initial - 1) * 100 if capital_initial != 0 else 0
            # Rendement annualisé (plus robuste)
            start_date = portfolio_history.index[0]
            end_date = portfolio_history.index[-1]
            jours_total = (end_date - start_date).days
            if jours_total > 0 and capital_initial != 0:
                 ratio_valeur = valeur_finale / capital_initial
                 if ratio_valeur > 0:
                      rendement_annualise_pct = (ratio_valeur ** (365.25 / jours_total) - 1) * 100
                 else:
                      rendement_annualise_pct = -100.0 # Perte totale
            else:
                 rendement_annualise_pct = 0.0 # Pas de durée ou capital nul

            col1, col2, col3 = st.columns(3)
            col1.metric("Valeur Finale Portefeuille", f"{valeur_finale:,.2f} FCFA", f"{valeur_finale-capital_initial:,.2f} FCFA")
            col2.metric("Rendement Total", f"{rendement_total_pct:.2f}%")
            col3.metric("Rendement Annualisé", f"{rendement_annualise_pct:.2f}%" if pd.notna(rendement_annualise_pct) else "N/A")

        except IndexError:
             st.error("Erreur: Impossible d'accéder aux résultats du portefeuille (probablement vide ou index corrompu).")
        except Exception as e:
            st.error(f"Erreur lors du calcul des statistiques de performance : {e}")

        # Graphique Évolution Portefeuille
        try:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            ax4.plot(portfolio_history.index, portfolio_history['valeur_totale'], linewidth=2, label='Valeur du portefeuille', color='blue')
            ax4.axhline(y=capital_initial, linestyle='--', linewidth=1, color='grey', label='Capital initial')

            ax4.set_title('Évolution de la Valeur du Portefeuille')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Valeur (FCFA)')
            ax4.grid(True, linestyle='--', alpha=0.6)
            ax4.legend()
            ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax4.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            fig4.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique d'évolution du portefeuille : {e}")

        # Composition Finale Portefeuille
        st.subheader("💼 Composition du Portefeuille Final")
        try:
            last_row = portfolio_history.iloc[-1]
            actions_finales = last_row['actions']
            cash_final = last_row['cash']
            trade_en_cours_final = last_row['trade_en_cours']
            date_livraison_finale = last_row['date_livraison_prevue']

            col1, col2, col3 = st.columns(3)
            col1.metric("Nombre d'actions détenues", f"{actions_finales:,.0f}")
            col2.metric("Liquidités (Cash)", f"{cash_final:,.2f} FCFA")
            status_trade = "Oui" if trade_en_cours_final else "Non"
            if trade_en_cours_final and pd.notna(date_livraison_finale):
                 status_trade += f" (Livraison prévue: {date_livraison_finale.strftime('%Y-%m-%d')})"
            col3.metric("Trade en attente de livraison?", status_trade)
        except IndexError:
             st.error("Erreur: Impossible d'accéder aux données finales du portefeuille.")
        except Exception as e:
            st.error(f"Erreur lors de l'affichage de la composition finale : {e}")


        # Journal des Transactions
        with st.expander("📜 Journal des Transactions"):
            if journal_transactions is not None and not journal_transactions.empty:
                # Afficher avec formatage
                st.dataframe(journal_transactions.style.format({
                    'Date Livraison': lambda dt: dt.strftime('%Y-%m-%d') if pd.notna(dt) else 'N/A',
                    'Quantité': '{:,.0f}',
                    'Prix Unitaire': '{:,.2f}',
                    'Frais': '{:,.2f}',
                    'Montant Net': '{:,.2f}' # Garder signe pour achat/vente
                }))
                st.markdown(get_csv_download_link(journal_transactions, filename=f"transactions_{stock_name}.csv", link_text="Télécharger le journal (CSV)"), unsafe_allow_html=True)
            else:
                st.info("Aucune transaction n'a été effectuée pendant la période de backtest.")

        # Métriques Avancées
        st.subheader("⚙️ Métriques de Performance Avancées")
        try:
            # Volatilité Annualisée
            # S'assurer qu'il y a au moins 2 rendements pour calculer std dev
            valid_returns = portfolio_history['rendement'].dropna()
            if len(valid_returns) >= 2:
                volatilite_strat_pct = valid_returns.std(ddof=1) * np.sqrt(252) * 100
            else:
                volatilite_strat_pct = np.nan

            # Ratio de Sharpe
            if len(valid_returns) >= 2 and volatilite_strat_pct is not None and volatilite_strat_pct != 0 and pd.notna(rendement_annualise_pct):
                # Utiliser le rendement annualisé déjà calculé
                volatilite_annualisee_strat = volatilite_strat_pct / 100
                sharpe_ratio = (rendement_annualise_pct/100 - taux_sans_risque) / volatilite_annualisee_strat
            else:
                sharpe_ratio = np.nan # Indicateur que le calcul n'est pas possible

            # Drawdown Maximum
            portfolio_history['peak'] = portfolio_history['valeur_totale'].cummax()
            portfolio_history['drawdown'] = (portfolio_history['valeur_totale'] - portfolio_history['peak']) / portfolio_history['peak']
            portfolio_history['drawdown'] = portfolio_history['drawdown'].replace([np.inf, -np.inf], np.nan).fillna(0)
            max_drawdown_pct = portfolio_history['drawdown'].min() * 100 if not portfolio_history['drawdown'].empty else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Volatilité Annualisée", f"{volatilite_strat_pct:.2f}%" if pd.notna(volatilite_strat_pct) else "N/A")
            col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}" if pd.notna(sharpe_ratio) else "N/A", help=f"Basé sur un taux sans risque de {taux_sans_risque*100:.1f}%")
            col3.metric("Drawdown Maximum", f"{max_drawdown_pct:.2f}%")

            # Graphique Drawdown
            fig5, ax5 = plt.subplots(figsize=(12, 4))
            ax5.fill_between(portfolio_history.index, portfolio_history['drawdown']*100, 0, color='red', alpha=0.3)
            ax5.set_title('Drawdown du Portefeuille')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Drawdown (%)')
            ax5.grid(True, linestyle='--', alpha=0.6)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax5.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
            fig5.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig5)

            # Graphique Distribution Rendements
            if not valid_returns.empty:
                fig6, ax6 = plt.subplots(figsize=(10, 4))
                ax6.hist(valid_returns * 100, bins=50, alpha=0.75, density=True)
                ax6.set_title('Distribution des Rendements Journaliers de la Stratégie')
                ax6.set_xlabel('Rendement Journalier (%)')
                ax6.set_ylabel('Densité de Fréquence')
                ax6.grid(True, alpha=0.3)
                rendement_moyen_pct = valid_returns.mean() * 100
                ax6.axvline(rendement_moyen_pct, color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {rendement_moyen_pct:.3f}%')
                ax6.legend()
                plt.tight_layout()
                st.pyplot(fig6)
            else:
                st.info("Pas de rendements journaliers valides à afficher pour la distribution.")

        except Exception as e:
            st.error(f"Erreur lors du calcul ou de l'affichage des métriques avancées : {e}")
            import traceback
            st.error(traceback.format_exc())

        # Comparaison Buy & Hold
        st.subheader("⚖️ Comparaison avec Stratégie 'Buy & Hold'")
        try:
            # Calcul performance Buy & Hold
            prix_initial_bh = data['Prix'].iloc[0]
            prix_final_bh = data['Prix'].iloc[-1]
            rendement_total_bh_pct = (prix_final_bh / prix_initial_bh - 1) * 100 if prix_initial_bh != 0 else 0

            if jours_total > 0 and prix_initial_bh != 0:
                 ratio_bh = prix_final_bh / prix_initial_bh
                 if ratio_bh > 0:
                    rendement_annualise_bh_pct = (ratio_bh ** (365.25 / jours_total) - 1) * 100
                 else:
                     rendement_annualise_bh_pct = -100.0
            else:
                 rendement_annualise_bh_pct = 0.0

            # Calcul Volatilité et Drawdown Buy & Hold
            data['rendement_bh'] = data['Prix'].pct_change().fillna(0.0)
            valid_returns_bh = data['rendement_bh']
            if len(valid_returns_bh) >= 2:
                volatilite_bh_pct = valid_returns_bh.std(ddof=1) * np.sqrt(252) * 100
            else:
                 volatilite_bh_pct = np.nan

            data['peak_bh'] = data['Prix'].cummax()
            # Gérer le cas où le prix initial est 0 pour le drawdown B&H
            if prix_initial_bh != 0 :
                data['drawdown_bh'] = ((data['Prix'] - data['peak_bh']) / data['peak_bh']).replace([np.inf, -np.inf], np.nan).fillna(0)
                max_drawdown_bh_pct = data['drawdown_bh'].min() * 100 if not data['drawdown_bh'].empty else 0
            else:
                 data['drawdown_bh'] = 0
                 max_drawdown_bh_pct = 0


            st.markdown("### Performance Buy & Hold")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendement Total B&H", f"{rendement_total_bh_pct:.2f}%")
            col2.metric("Rendement Annualisé B&H", f"{rendement_annualise_bh_pct:.2f}%" if pd.notna(rendement_annualise_bh_pct) else "N/A")
            col3.metric("Max Drawdown B&H", f"{max_drawdown_bh_pct:.2f}%")

            st.markdown("### Comparaison Directe")
            # Calculer les différences seulement si les métriques sont valides
            surperf_total = (rendement_total_pct - rendement_total_bh_pct) if pd.notna(rendement_total_pct) and pd.notna(rendement_total_bh_pct) else np.nan
            surperf_annuel = (rendement_annualise_pct - rendement_annualise_bh_pct) if pd.notna(rendement_annualise_pct) and pd.notna(rendement_annualise_bh_pct) else np.nan
            diff_vol = (volatilite_strat_pct - volatilite_bh_pct) if pd.notna(volatilite_strat_pct) and pd.notna(volatilite_bh_pct) else np.nan

            col1, col2, col3 = st.columns(3)
            col1.metric("Surperformance (Total)", f"{surperf_total:.2f}%" if pd.notna(surperf_total) else "N/A")
            col2.metric("Surperformance (Annualisée)", f"{surperf_annuel:.2f}%" if pd.notna(surperf_annuel) else "N/A")
            col3.metric("Différence Volatilité", f"{diff_vol:.2f}%" if pd.notna(diff_vol) else "N/A", help="Négatif = Stratégie moins volatile")


            # Graphique Comparatif Performances Normalisées
            fig7, ax7 = plt.subplots(figsize=(12, 6))
            plot_strat = False
            plot_bh = False
            # Performance normalisée base 1
            if 'rendement_cumule' in portfolio_history.columns and not portfolio_history['rendement_cumule'].empty:
                perf_strategie = (1 + portfolio_history['rendement_cumule'])
                ax7.plot(portfolio_history.index, perf_strategie * capital_initial, label=f'Stratégie ({stock_name})', linewidth=2, color='blue')
                plot_strat = True

            if prix_initial_bh != 0:
                perf_buy_hold = (data['Prix'] / prix_initial_bh)
                ax7.plot(data.index, perf_buy_hold * capital_initial, label=f'Buy & Hold ({stock_name})', linewidth=2, linestyle='--', color='orange')
                plot_bh = True

            if plot_strat or plot_bh:
                ax7.set_title('Comparaison Normalisée des Performances (Base = Capital Initial)')
                ax7.set_xlabel('Date')
                ax7.set_ylabel(f'Valeur Portefeuille (Base {capital_initial:,.0f} FCFA)')
                ax7.grid(True, linestyle='--', alpha=0.6)
                ax7.legend()
                ax7.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
                ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax7.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
                fig7.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig7)
            else:
                 st.warning("Impossible d'afficher le graphique comparatif des performances.")

        except Exception as e:
            st.error(f"Erreur lors de la comparaison avec Buy & Hold : {e}")
            import traceback
            st.error(traceback.format_exc())


        # --- Téléchargement du Rapport Complet du Portefeuille ---
        st.subheader("📥 Télécharger le Rapport Complet")
        st.markdown(get_csv_download_link(portfolio_history, filename=f"rapport_backtest_{stock_name}.csv", link_text="Télécharger l'historique du portefeuille (CSV)"), unsafe_allow_html=True)

        st.info("""
        **Note sur l'interprétation :**
        * **Ratio de Sharpe :** Mesure le rendement ajusté au risque (plus élevé = mieux). Un ratio > 1 est souvent considéré comme bon.
        * **Drawdown Maximum :** La perte maximale historique du pic au creux (plus faible = mieux). Indique le risque de perte potentielle.
        * **Volatilité :** Mesure l'ampleur des variations de prix/rendement (plus faible = moins risqué/variable en général).
        * La comparaison avec **Buy & Hold** montre si la stratégie active a ajouté de la valeur (alpha) par rapport à un simple achat initial et conservation.
        * **Hypothèses Clés :** Frais de transaction fixes appliqués à chaque ordre. Ordres exécutés au prix du jour (clôture ajustée par le plafond de variation simulé). Délai de livraison en jours ouvrés appliqué, bloquant les transactions pendant ce délai. Pas de prise en compte de la liquidité du marché (possibilité d'acheter/vendre la quantité désirée au prix affiché) ni du slippage (différence entre prix attendu et prix exécuté). Les dividendes ne sont pas réinvestis dans cette simulation (sauf implicitement via le calcul de la valeur intrinsèque).
        """)
    else:
         st.warning("Le backtest n'a pas produit de résultats à afficher (aucun trade ou historique de portefeuille vide).")

else:
    # Message si aucune donnée n'est chargée ou traitée
    if uploaded_file is None:
        st.info("👈 Veuillez charger un fichier CSV via la barre latérale pour commencer l'analyse.")
    # Ne plus afficher le warning si le bouton n'a pas été cliqué, seulement s'il y a eu une erreur
    elif data is None and uploaded_file is not None:
         # Ce cas est maintenant couvert par les erreurs affichées dans process_data ou l'interface de mapping
         pass
    elif data is not None and data.empty:
         # Ce cas est couvert par l'erreur affichée dans process_data
         st.error("❌ Le traitement des données a résulté en un DataFrame vide. Vérifiez votre fichier CSV, le mapping des colonnes et les éventuels messages d'avertissement/erreur lors du traitement.")


# --- Pied de page ---
st.markdown("---")
st.markdown("Application de Backtesting BRVM v1.1 - Améliorations Implémentées")
