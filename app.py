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

    options_list = [''] + st.session_state.all_columns # Define options once

    # Créer les selectbox pour le mapping
    for standard_name in st.session_state.column_mapping.keys():
        # Get the currently mapped column name safely
        mapped_col = st.session_state.column_mapping.get(standard_name, "")

        # Calculate the correct index for the selectbox
        selectbox_index = 0 # Default to index 0 ('')
        if mapped_col and mapped_col in st.session_state.all_columns:
            try:
                # Find index in original list and add 1 for the prepended ''
                original_index = st.session_state.all_columns.index(mapped_col)
                selectbox_index = original_index + 1
            except ValueError:
                # Should not happen due to 'in' check, but good practice
                selectbox_index = 0
        # Handle case where the mapping is explicitly set to '' after being something else
        elif not mapped_col:
            selectbox_index = 0


        # Create the selectbox with the calculated index
        selected_column = st.sidebar.selectbox(
            f"Colonne pour '{standard_name}'",
            options=options_list, # Use the defined options list
            index=selectbox_index, # Use the correctly calculated integer index
            key=f'map_{standard_name}' # Clé unique pour chaque selectbox dans session_state
        )
        # Mettre à jour le mapping dans la session_state quand une selectbox change
        # (This happens automatically thanks to the key and Streamlit's rerun logic)
        # Check if the value actually changed to update the session state correctly
        if st.session_state.column_mapping.get(standard_name) != selected_column:
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
        # Assurer que le fichier est prêt à être lu à nouveau
        if st.session_state.uploaded_file_obj:
            st.session_state.uploaded_file_obj.seek(0) # Important!

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
            st.error("Impossible de traiter les données avec le mapping et le format fournis. Vérifiez vos sélections, le format de date et le contenu du fichier.")
            # Les messages d'erreur spécifiques sont dans les logs (ou dans le loader si vous y avez laissé des prints)

# --- Section Paramètres de Backtest (Affichée si données traitées) ---
if not st.session_state.data.empty:
    st.sidebar.subheader("3. Paramètres du Backtest")

    # Utiliser les dates min/max des données traitées pour les sélecteurs de date
    min_date_data = st.session_state.data.index.min().date()
    max_date_data = st.session_state.data.index.max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Date de début de l'analyse", value=min_date_data, min_value=min_date_data, max_value=max_date_data, key='start_date')
        end_date = st.date_input("Date de fin de l'analyse", value=max_date_data, min_value=min_date_data, max_value=max_date_data, key='end_date')

    with col2:
        initial_capital = st.number_input("Capital Initial", min_value=1000, value=100000, step=1000, key='initial_capital')
        short_window = st.slider("Période MA Courte", min_value=1, max_value=100, value=40, key='short_window') # Min value should be 1 or more
        long_window = st.slider("Période MA Longue", min_value=2, max_value=250, value=100, key='long_window') # Min value should be > short_window

    # --- Bouton de Lancement du Backtest ---
    if st.button("Lancer le Backtest"):
        # Validation des fenêtres de moyenne mobile
        if short_window >= long_window:
            st.error("La période de Moyenne Mobile Courte doit être inférieure à la période Longue.")
        else:
            st.info("Exécution du backtest en cours...")

            # Filtrer les données traitées par la plage de dates sélectionnée par l'utilisateur
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                st.error("La date de début de l'analyse ne peut pas être postérieure à la date de fin.")
            else:
                # Utiliser .loc avec des strings pour filtrer sur l'index DatetimeIndex
                # Assurer que les dates sont bien converties en datetime pour la comparaison
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)

                # Filtrer en s'assurant que l'index est aussi datetime
                df_for_backtest = st.session_state.data[(st.session_state.data.index >= start_dt) & (st.session_state.data.index <= end_dt)].copy()


                if df_for_backtest.empty:
                    st.warning(f"Aucune donnée disponible dans la plage de dates sélectionnée ({start_date} au {end_date}).")
                    st.session_state.backtest_results = None
                elif len(df_for_backtest) <= long_window:
                     st.warning(f"Pas assez de données ({len(df_for_backtest)} jours) pour la période de MA longue ({long_window} jours) dans la plage sélectionnée.")
                     st.session_state.backtest_results = None
                else:
                    st.success(f"Exécution du backtest sur {len(df_for_backtest)} jours de données ({start_date} au {end_date}).")

                    # 1. Appliquer la stratégie pour générer les signaux
                    # Assumant que le loader a bien standardisé la colonne en 'Prix' ou 'Close'.
                    # Vérifions le nom de la colonne de clôture dans les données traitées.
                    # Si load_and_process_data retourne un df avec 'Prix', il faut le gérer.
                    # Supposons ici que load_and_process_data retourne un df avec la colonne 'Close' standardisée.
                    # Si ce n'est pas le cas, il faudra adapter soit le loader, soit ici.

                    # Assumons que la colonne standardisée est 'Close' comme attendu par simple_ma
                    if 'Close' not in df_for_backtest.columns:
                         st.error("La colonne 'Close' standardisée est manquante dans les données traitées. Vérifiez le mapping et la fonction 'load_and_process_data'.")
                         st.session_state.backtest_results = None
                    else:
                        try:
                            df_strat = simple_ma.apply_strategy(df_for_backtest.copy(), short_window, long_window) # Passer une copie

                            # 2. Exécuter le backtest
                            # Le moteur run_backtest s'attend à un df avec 'Close' et 'positions'.
                            equity_curve = engine.run_backtest(df_strat, initial_capital) # df_strat contient déjà 'Close' et 'positions'

                            if equity_curve is not None and not equity_curve.empty:
                                st.success("Backtest terminé.")

                                # 3. Calculer les métriques de performance
                                performance_metrics = metrics.calculate_performance_metrics(equity_curve, initial_capital) # Passer le capital initial aux métriques

                                # Stocker les résultats dans l'état de session
                                st.session_state.backtest_results = {
                                    'equity_curve': equity_curve['Equity'], # Stocker seulement la série d'équité
                                    'performance_metrics': performance_metrics,
                                    'df_strat_for_plot': df_strat # Stocker aussi le df avec signaux pour le graphique
                                }

                            else:
                                st.error("Une erreur est survenue pendant l'exécution du backtest ou la courbe d'équité est vide.")
                                st.session_state.backtest_results = None
                        except Exception as e:
                            st.error(f"Erreur lors de l'application de la stratégie ou du backtest : {e}")
                            st.session_state.backtest_results = None


# --- Afficher les Résultats du Backtest (Si disponibles) ---
if st.session_state.backtest_results:
    st.header("Résultats du Backtest")

    results = st.session_state.backtest_results
    metrics_data = results['performance_metrics']
    equity_curve_series = results['equity_curve']
    df_strat_for_plot = results['df_strat_for_plot']

    # Afficher les métriques
    st.subheader("Métriques de Performance")
    # Utiliser des colonnes pour mieux organiser l'affichage
    met_col1, met_col2 = st.columns(2)

    with met_col1:
        st.metric("Capital Initial", f"{metrics_data.get('Capital Initial', 'N/A'):,.0f}")
        st.metric("Capital Final", f"{metrics_data.get('Capital Final', 'N/A'):,.2f}")
        st.metric("Retour Total (%)", f"{metrics_data.get('Retour Total (%)', 'N/A'):.2f}%")
        st.metric("CAGR (%)", f"{metrics_data.get('CAGR (%)', 'N/A'):.2f}%")


    with met_col2:
        st.metric("Max Drawdown (%)", f"{metrics_data.get('Max Drawdown (%)', 'N/A'):.2f}%")
        st.metric("Ratio de Sharpe", f"{metrics_data.get('Sharpe Ratio', 'N/A'):.2f}")
        # Ajoutez d'autres métriques si disponibles (ex: Volatilité, Sortino, etc.)
        # st.metric("Volatilité Ann. (%)", f"{metrics_data.get('Annual Volatility (%)', 'N/A'):.2f}%")


    st.subheader("Courbe d'Équité")
    st.line_chart(equity_curve_series) # Afficher la série d'équité

    st.subheader("Prix de l'Action avec Signaux et Moyennes Mobiles")

    # Préparer le DataFrame pour le graphique des signaux/prix/MAs
    df_plot = df_strat_for_plot[['Close', 'Short_MA', 'Long_MA']].copy() # Inclure les MAs

    # S'assurer que 'positions' est numérique pour la comparaison
    df_strat_for_plot['positions'] = pd.to_numeric(df_strat_for_plot['positions'], errors='coerce')

    # Trouver les dates où un signal d'achat ou de vente a été généré (changement de position)
    # Signal d'achat: position passe de 0 ou -1 à 1
    # Signal de vente: position passe de 1 ou 0 à -1 (ou juste 1 à -1 si on ne vend qu'après achat)
    # Simplifions : signal d'achat là où position == 1 pour la première fois après != 1
    # signal de vente là où position == -1 pour la première fois après != -1

    # On prend les points où la position change
    buy_dates = df_strat_for_plot[(df_strat_for_plot['positions'] == 1) & (df_strat_for_plot['positions'].shift(1) != 1)].index
    sell_dates = df_strat_for_plot[(df_strat_for_plot['positions'] == -1) & (df_strat_for_plot['positions'].shift(1) != -1)].index


    # Créer des colonnes pour les marqueurs de signaux sur le graphique
    df_plot['Achat'] = np.nan # Utiliser np.nan pour que line_chart ne trace pas de lignes
    df_plot['Vente'] = np.nan

    # Placer les points de signal sur le graphique des prix (à la valeur du prix 'Close')
    if not buy_dates.empty:
        df_plot.loc[buy_dates, 'Achat'] = df_plot.loc[buy_dates, 'Close']
    if not sell_dates.empty:
        df_plot.loc[sell_dates, 'Vente'] = df_plot.loc[sell_dates, 'Close']


    # Configurer le graphique avec st.line_chart
    # On trace Close, Short_MA, Long_MA comme lignes
    # On trace Achat et Vente comme points (st.line_chart le fait automatiquement si valeurs non-nan isolées)
    st.line_chart(df_plot[['Close', 'Short_MA', 'Long_MA', 'Achat', 'Vente']], use_container_width=True)
    st.markdown("*(Les points verts indiquent les achats, les points rouges indiquent les ventes lors des croisements de moyennes mobiles)*")


# --- Message si aucun fichier n'a encore été uploadé ou traité ---
if st.session_state.uploaded_file_obj is None:
     st.info("Veuillez uploader un fichier CSV de données historiques dans la barre latérale pour commencer.")
elif st.session_state.data.empty and st.session_state.uploaded_file_obj is not None and 'Processer les Données' not in st.session_state.get('buttons_clicked', []): # Simple way to check if process was attempted
    # Message si fichier uploadé mais pas encore traité ou traitement échoué
    st.warning("Fichier uploadé. Veuillez mapper les colonnes requises et cliquer sur 'Processer les Données' dans la barre latérale.")


st.sidebar.markdown("---") # Séparateur visuel
st.sidebar.header("À Propos")
st.sidebar.info(
    "Cet outil permet de backtester une stratégie simple de croisement de moyennes mobiles sur des données historiques de la BRVM."
    "\n\nAssurez-vous que votre fichier CSV contient les colonnes nécessaires "
    "(Date, Ouverture, Plus Haut, Plus Bas, Clôture, Volume) "
    "et mappez-les correctement après l'upload."
    "\n\nLes résultats du backtest sont indicatifs et ne garantissent pas les performances futures."
)

# Pour suivre quel bouton a été cliqué (simple gestion d'état)
if 'buttons_clicked' not in st.session_state:
    st.session_state['buttons_clicked'] = []

# Log button clicks (peut être utile pour le débogage de l'état)
if st.sidebar.button("Processer les Données", key="process_btn_state_tracker", disabled=not all_required_mapped if 'all_required_mapped' in locals() else True): # Need to check if var exists
    if 'Processer les Données' not in st.session_state['buttons_clicked']:
       st.session_state['buttons_clicked'].append('Processer les Données')
# Note: Streamlit reruns the script, so tracking button clicks needs careful state management.
# The logic above is a basic example and might need refinement depending on exact flow desired.
