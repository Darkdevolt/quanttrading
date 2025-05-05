# app.py
import streamlit as st
import pandas as pd
import numpy as np
import csv # Nécessaire pour le sniffer dans le loader, bien que la logique soit dans loader.py maintenant

# Importation des modules
from data import loader # Importe le module loader
from strategies import simple_ma
from backtesting import engine, metrics

# --- Configuration de l'interface Streamlit ---
st.set_page_config(layout="wide", page_title="BRVM Quant Backtest")

st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")

# --- Initialisation de l'état de session ---
# Utiliser st.session_state pour persister les données et les paramètres
if 'uploaded_file_obj' not in st.session_state:
    st.session_state.uploaded_file_obj = None
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame() # DataFrame traité et standardisé
if 'all_columns' not in st.session_state:
    st.session_state.all_columns = [] # Noms des colonnes détectées dans le fichier uploadé
if 'column_mapping' not in st.session_state:
    # Mapping par défaut. L'utilisateur le modifiera.
    st.session_state.column_mapping = {
        "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
    }
if 'date_format_input' not in st.session_state:
    st.session_state.date_format_input = "" # Input utilisateur pour le format de date
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None # Résultats du backtest (courbe d'équité, métriques)


st.sidebar.header("Paramètres Globaux")

# --- Section Upload de Fichier ---
st.sidebar.subheader("1. Chargement des Données")

# Callback pour gérer le cas où un NOUVEAU fichier est uploadé
def handle_upload():
    """Gère l'upload d'un nouveau fichier et détecte les colonnes."""
    uploaded_file = st.session_state['new_uploaded_file']
    st.session_state.uploaded_file_obj = uploaded_file # Stocker l'objet fichier

    if uploaded_file is not None:
        st.sidebar.info("Fichier uploadé. Détection des colonnes...")
        try:
            # Lire juste l'en-tête pour obtenir les noms de colonnes
            # Utiliser le même sniffer que dans le loader pour la cohérence
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
                 # Fallback manuel si sniffer échoue
                 uploaded_file.seek(0)
                 try: header_line_bytes = uploaded_file.readline()
                 except Exception: header_line_bytes = b'' # Handle potential read errors
                 try: header_line = header_line_bytes.decode('utf-8')
                 except UnicodeDecodeError: header_line = header_line_bytes.decode('latin-1', errors='ignore')
                 uploaded_file.seek(0)
                 if header_line and header_line.count(';') >= header_line.count(','): separator = ';'
                 else: separator = ','

            uploaded_file.seek(0) # Revenir au début avant de lire l'en-tête
            # Lire l'en-tête avec pandas pour obtenir les noms de colonnes
            temp_df = pd.read_csv(uploaded_file, sep=separator, nrows=0) # Lire 0 ligne pour juste l'en-tête
            st.session_state.all_columns = list(temp_df.columns)
            st.sidebar.success(f"Colonnes détectées : {', '.join(st.session_state.all_columns)}")

            # Réinitialiser le mapping par défaut ou essayer de deviner
            st.session_state.column_mapping = {
                "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
            }
            # Tentative de mapping automatique basique
            for standard_name in st.session_state.column_mapping.keys():
                 # Chercher une colonne qui contient le nom standard (insensible à la casse)
                 matching_cols = [col for col in st.session_state.all_columns if standard_name.lower() in col.lower()]
                 if matching_cols:
                     # Utiliser la première correspondance trouvée
                     st.session_state.column_mapping[standard_name] = matching_cols[0]


        except Exception as e:
            st.sidebar.error(f"Erreur lors de la détection des colonnes : {e}")
            st.session_state.all_columns = [] # Vider la liste des colonnes
            st.session_state.uploaded_file_obj = None # Invalider le fichier uploadé
            st.session_state.column_mapping = { # Réinitialiser le mapping
                "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
            }


    else:
        # Si le fichier est retiré
        st.session_state.uploaded_file_obj = None
        st.session_state.data = pd.DataFrame()
        st.session_state.all_columns = []
        st.session_state.column_mapping = {
            "Date": "", "Open": "", "High": "", "Low": "", "Close": "", "Volume": ""
        }
        st.session_state.date_format_input = ""
        st.session_state.backtest_results = None
        st.sidebar.info("Aucun fichier chargé.")


# Le widget file_uploader met sa valeur dans la session_state via la key 'new_uploaded_file'
# et déclenche handle_upload si un fichier est sélectionné ou retiré.
st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file', # La valeur de l'uploader est stockée ici
    on_change=handle_upload # Déclenche la fonction si la valeur change
)

# --- Section Mapping des Colonnes (Affichée si colonnes détectées) ---
if st.session_state.all_columns:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.write("Associez les colonnes de votre fichier aux noms standardisés.")

    # Créer les selectbox pour le mapping
    for standard_name in st.session_state.column_mapping.keys():
        # Utiliser value=st.session_state.column_mapping[standard_name] pour que la valeur soit persistante
        selected_column = st.sidebar.selectbox(
            f"Colonne pour '{standard_name}'",
            [''] + st.session_state.all_columns, # Ajouter une option vide
            index=[''] + st.session_state.all_columns.index(st.session_state.column_mapping[standard_name]) if st.session_state.column_mapping[standard_name] in st.session_state.all_columns else 0,
            key=f'map_{standard_name}' # Clé unique pour chaque selectbox dans session_state
        )
        # Mettre à jour le mapping dans la session_state quand une selectbox change
        st.session_state.column_mapping[standard_name] = selected_column

    # Option pour spécifier le format de date
    st.sidebar.subheader("Format de Date (Optionnel)")
    st.session_state.date_format_input = st.sidebar.text_input(
        "Spécifiez le format de date (ex: %Y-%m-%d)",
        value=st.session_state.date_format_input,
        key='date_format_key',
        help="Laissez vide pour détection automatique. Spécifiez si la détection échoue. Exemples: %Y-%m-%d, %d/%m/%Y"
    )

    # --- Bouton pour Traiter les Données ---
    # Vérifier si toutes les colonnes requises sont mappées avant d'activer le bouton
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    all_required_mapped = all(st.session_state.column_mapping.get(key) for key in required_keys)

    if st.sidebar.button("Processer les Données", disabled=not all_required_mapped):
        st.info("Traitement des données en cours...")
        # Appeler la fonction du loader avec le fichier uploadé et le mapping
        processed_df = loader.load_and_process_data(
            st.session_state.uploaded_file_obj,
            st.session_state.column_mapping,
            st.session_state.date_format_input
        )

        if processed_df is not None and not processed_df.empty:
            st.session_state.data = processed_df # Stocker le DataFrame traité
            st.success("Données traitées avec succès.")
            st.write("Aperçu des données traitées :")
            st.dataframe(st.session_state.data.head()) # Afficher les premières lignes
            st.info(f"Données disponibles du {st.session_state.data.index.min().date()} au {st.session_state.data.index.max().date()}.")

        else:
            st.session_state.data = pd.DataFrame() # Vider les données si le traitement échoue
            st.error("Impossible de traiter les données avec le mapping et le format fournis. Vérifiez vos sélections et le contenu du fichier.")
            # Les messages d'erreur spécifiques sont dans les logs (ou dans le loader si vous y avez laissé des prints)

# --- Section Paramètres de Backtest (Affichée si données traitées) ---
if not st.session_state.data.empty:
    st.sidebar.subheader("3. Paramètres du Backtest")

    # Utiliser les dates min/max des données traitées pour les sélecteurs de date
    min_date_data = st.session_state.data.index.min().date()
    max_date_data = st.session_state.data.index.max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Date de début de l'analyse", value=min_date_data, min_value=min_date_data, max_value=max_date_data)
        end_date = st.date_input("Date de fin de l'analyse", value=max_date_data, min_value=min_date_data, max_value=max_date_data)

    with col2:
        initial_capital = st.number_input("Capital Initial", min_value=1000, value=100000, step=1000)
        short_window = st.slider("Période MA Courte", min_value=10, max_value=100, value=40)
        long_window = st.slider("Période MA Longue", min_value=50, max_value=250, value=100)

    # --- Bouton de Lancement du Backtest ---
    if st.button("Lancer le Backtest"):
        st.info("Exécution du backtest en cours...")

        # Filtrer les données traitées par la plage de dates sélectionnée par l'utilisateur
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
             st.error("La date de début de l'analyse ne peut pas être postérieure à la date de fin.")
        else:
            # Utiliser .loc avec des strings pour filtrer sur l'index DatetimeIndex
            # Ajouter .copy() pour éviter SettingWithCopyWarning
            df_for_backtest = st.session_state.data.loc[str(start_date):str(end_date)].copy()

            if df_for_backtest.empty:
                st.warning(f"Aucune donnée disponible dans la plage de dates sélectionnée ({start_date} au {end_date}).")
                st.session_state.backtest_results = None
            else:
                st.success(f"Exécution du backtest sur {len(df_for_backtest)} jours de données.")

                # 1. Appliquer la stratégie pour générer les signaux
                # La stratégie s'attend à un DataFrame avec une colonne 'Close' (qui est 'Prix' dans nos données standardisées)
                # Adapter l'appel de la stratégie pour utiliser la colonne 'Prix'
                # NOTE: La stratégie simple_ma.py actuelle s'attend à une colonne nommée EXACTEMENT 'Close'.
                # Si vous avez renommé la colonne en 'Prix' dans le loader, vous devez adapter la stratégie
                # ou renommer 'Prix' en 'Close' temporairement pour la stratégie.
                # Option 1 (Adapter la stratégie - Mieux à long terme): Modifier simple_ma.py pour utiliser 'Prix'
                # Option 2 (Renommer temporairement - Plus rapide pour cet exemple):
                df_for_backtest_strategy = df_for_backtest.rename(columns={'Prix': 'Close'}).copy()


                df_strat = simple_ma.apply_strategy(df_for_backtest_strategy, short_window, long_window)

                # 2. Exécuter le backtest
                # Le moteur run_backtest s'attend à un df avec 'Close' et 'positions'.
                # Il faut donc lui passer le df_strat qui a la colonne 'Close' (temporairement renommée) et 'positions'.
                equity_curve = engine.run_backtest(df_strat, initial_capital)


                if equity_curve is not None and not equity_curve.empty:
                    st.success("Backtest terminé.")

                    # 3. Calculer les métriques de performance
                    performance_metrics = metrics.calculate_performance_metrics(equity_curve)

                    # Stocker les résultats dans l'état de session
                    st.session_state.backtest_results = {
                        'equity_curve': equity_curve,
                        'performance_metrics': performance_metrics,
                        'df_strat_for_plot': df_strat # Stocker aussi le df avec signaux pour le graphique
                    }

                else:
                    st.error("Une erreur est survenue pendant l'exécution du backtest ou la courbe d'équité est vide.")
                    st.session_state.backtest_results = None

# --- Afficher les Résultats du Backtest (Si disponibles) ---
if st.session_state.backtest_results:
    st.header("Résultats du Backtest")

    # Afficher les métriques
    st.subheader("Métriques de Performance")
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    metrics_data = st.session_state.backtest_results['performance_metrics']
    with metrics_col1:
        st.metric("Capital Initial", metrics_data.get("Capital Initial", "N/A"))
    with metrics_col2:
         st.metric("Capital Final", metrics_data.get("Capital Final", "N/A"))
    with metrics_col3:
         st.metric("Retour Total", metrics_data.get("Retour Total (%)", "N/A"))
    with metrics_col4:
         st.metric("CAGR", metrics_data.get("CAGR (%)", "N/A"))


    st.subheader("Courbe d'Équité")
    st.line_chart(st.session_state.backtest_results['equity_curve'])

    st.subheader("Prix de l'Action avec Signaux")
    df_strat_for_plot = st.session_state.backtest_results['df_strat_for_plot']

    # Préparer le DataFrame pour le graphique des signaux
    # Utiliser la colonne 'Close' temporairement renommée par la stratégie
    df_plot = df_strat_for_plot[['Close']].copy()

    # S'assurer que 'positions' est numérique pour la comparaison
    df_strat_for_plot['positions'] = pd.to_numeric(df_strat_for_plot['positions'], errors='coerce')

    buy_dates = df_strat_for_plot[df_strat_for_plot['positions'] == 1].index
    sell_dates = df_strat_for_plot[df_strat_for_plot['positions'] == -1].index

    df_plot['Buy Signal'] = None
    df_plot['Sell Signal'] = None

    # Placer les points de signal sur le graphique des prix
    # Utilisez la colonne 'Close' du DataFrame de stratégie
    df_plot.loc[buy_dates, 'Buy Signal'] = df_strat_for_plot['Close'][buy_dates] * 0.95 # Légèrement en dessous du prix
    df_plot.loc[sell_dates, 'Sell Signal'] = df_strat_for_plot['Close'][sell_dates] * 1.05 # Légèrement au dessus du prix

    # Combiner Close et les signaux dans un seul DataFrame pour st.line_chart
    df_final_plot = df_strat_for_plot[['Close']].copy()
    df_final_plot['Buy Signal'] = df_plot['Buy Signal']
    df_final_plot['Sell Signal'] = df_plot['Sell Signal']


    st.line_chart(df_final_plot, use_container_width=True)
    st.markdown("*(Les points bleus indiquent les achats, les points rouges indiquent les ventes selon la stratégie simplifiée)*")


# --- Message si aucun fichier n'a encore été uploadé ---
if st.session_state.uploaded_file_obj is None:
     st.info("Veuillez uploader un fichier CSV de données historiques dans la barre latérale pour commencer.")
elif st.session_state.data.empty and st.session_state.uploaded_file_obj is not None:
     # Message si fichier uploadé mais pas encore traité ou traitement échoué
     st.warning("Fichier uploadé. Veuillez mapper les colonnes et cliquer sur 'Processer les Données' dans la barre latérale.")


st.sidebar.markdown("---") # Séparateur visuel
st.sidebar.header("À Propos")
st.sidebar.info(
    "Cet outil permet de backtester des stratégies simples sur des données historiques de la BRVM."
    "\n\nAssurez-vous que votre fichier CSV contient les colonnes nécessaires "
    "(Date, Ouverture, Plus Haut, Plus Bas, Clôture, Volume) "
    "et mappez-les correctement après l'upload."
)
