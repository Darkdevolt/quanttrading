# app.py
import streamlit as st
import pandas as pd
import numpy as np
import csv
import io # Nécessaire pour gérer l'objet fichier uploadé correctement

# Importation des modules (assurez-vous que ces fichiers existent et sont accessibles)
try:
    from data import loader
    from strategies import simple_ma
    from backtesting import engine, metrics
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez que les fichiers/dossiers existent.")
    st.stop() # Arrête l'exécution si les modules ne peuvent pas être importés

# --- Configuration de l'interface Streamlit ---
st.set_page_config(layout="wide", page_title="BRVM Quant Backtest")

st.title("📈 BRVM Quant Backtest")
st.markdown("""
Bienvenue sur l'outil d'analyse et de backtesting quantitatif pour la BRVM.
Chargez vos données historiques au format CSV pour commencer.
""")

# --- Initialisation de l'état de session ---
# Utiliser st.session_state pour persister les données et les paramètres
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None # Stocke le contenu binaire du fichier
if 'uploaded_file_name' not in st.session_state:
     st.session_state.uploaded_file_name = None
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
if 'data_processed' not in st.session_state:
     st.session_state.data_processed = False # Flag pour savoir si les données ont été traitées


st.sidebar.header("Paramètres Globaux")

# --- Section Upload de Fichier ---
st.sidebar.subheader("1. Chargement des Données")

# Callback pour gérer le cas où un NOUVEAU fichier est uploadé
def handle_upload():
    """Gère l'upload d'un nouveau fichier, stocke son contenu, et détecte les colonnes."""
    uploaded_file = st.session_state['new_uploaded_file'] # Récupère l'objet fichier depuis la clé du widget

    if uploaded_file is not None:
        # Lire le contenu une fois et le stocker dans session_state
        try:
            uploaded_file.seek(0) # S'assurer qu'on lit depuis le début
            st.session_state.uploaded_file_content = uploaded_file.read()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.sidebar.info(f"Fichier '{uploaded_file.name}' chargé. Détection des colonnes...")

            # Utiliser le contenu stocké pour la détection
            file_stream = io.BytesIO(st.session_state.uploaded_file_content)
            sample_bytes = file_stream.read(2048)
            file_stream.seek(0) # Revenir au début pour pandas

            try: sample_text = sample_bytes.decode('utf-8')
            except UnicodeDecodeError: sample_text = sample_bytes.decode('latin-1', errors='ignore')

            separator = ',' # Default separator
            try:
                if sample_text.strip():
                    # Essayer de détecter le séparateur avec Sniffer
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample_text)
                    separator = dialect.delimiter
                else:
                    # Si l'échantillon est vide, garder la virgule par défaut mais avertir
                    st.sidebar.warning("Échantillon de fichier vide ou non lisible pour la détection du séparateur, utilisation de ',' par défaut.")
            except csv.Error:
                 # Fallback manuel si sniffer échoue
                 file_stream_fallback = io.BytesIO(st.session_state.uploaded_file_content)
                 try:
                     header_line_bytes = file_stream_fallback.readline()
                     try: header_line = header_line_bytes.decode('utf-8')
                     except UnicodeDecodeError: header_line = header_line_bytes.decode('latin-1', errors='ignore')

                     if header_line and header_line.count(';') >= header_line.count(','):
                         separator = ';'
                     # Garder ',' sinon
                 except Exception:
                     st.sidebar.warning("Impossible de lire l'en-tête pour le fallback du séparateur.")
                 finally:
                    file_stream_fallback.close()


            # Lire l'en-tête avec pandas pour obtenir les noms de colonnes, en utilisant le séparateur détecté
            file_stream.seek(0) # Revenir au début du stream pour pandas
            temp_df = pd.read_csv(file_stream, sep=separator, nrows=0)
            st.session_state.all_columns = list(temp_df.columns)
            st.sidebar.success(f"Colonnes détectées ({len(st.session_state.all_columns)}): {', '.join(st.session_state.all_columns)}")

            # Réinitialiser le mapping et tenter un mapping automatique
            st.session_state.column_mapping = {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]}
            for standard_name in st.session_state.column_mapping.keys():
                 matching_cols = [col for col in st.session_state.all_columns if standard_name.lower() in col.lower()]
                 if matching_cols:
                      st.session_state.column_mapping[standard_name] = matching_cols[0]

            # Réinitialiser les étapes suivantes car un nouveau fichier est chargé
            st.session_state.data = pd.DataFrame()
            st.session_state.data_processed = False
            st.session_state.backtest_results = None


        except Exception as e:
            st.sidebar.error(f"Erreur lors de la lecture/détection des colonnes : {e}")
            # Réinitialiser tout en cas d'erreur critique
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.session_state.all_columns = []
            st.session_state.column_mapping = {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]}
            st.session_state.data = pd.DataFrame()
            st.session_state.data_processed = False
            st.session_state.backtest_results = None

    else:
        # Si le fichier est retiré (l'utilisateur clique sur le 'x')
        if st.session_state.uploaded_file_name: # Ne réinitialiser que si un fichier était chargé
            st.sidebar.info("Fichier retiré.")
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.session_state.all_columns = []
            st.session_state.column_mapping = {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]}
            st.session_state.data = pd.DataFrame()
            st.session_state.data_processed = False
            st.session_state.backtest_results = None


# Le widget file_uploader
# Utilise une clé différente pour déclencher le on_change uniquement lorsqu'un fichier est uploadé/retiré
uploaded_file_widget = st.sidebar.file_uploader(
    "Chargez votre fichier CSV d'historique",
    type=['csv'],
    key='new_uploaded_file', # Clé pour le widget, on_change met à jour l'état
    on_change=handle_upload # Déclenche la fonction si un fichier est ajouté ou retiré
)

# --- Section Mapping des Colonnes (Affichée si colonnes détectées) ---
# Utiliser uploaded_file_name pour conditionner l'affichage, car all_columns peut être vide même si un fichier est chargé (erreur)
if st.session_state.uploaded_file_name and st.session_state.all_columns:
    st.sidebar.subheader("2. Mapping des Colonnes")
    st.sidebar.write("Associez les colonnes de votre fichier aux noms standardisés.")

    options_list = [''] + st.session_state.all_columns # Liste des options pour les selectbox

    # Créer les selectbox pour le mapping
    for standard_name in st.session_state.column_mapping.keys():
        # Get the currently mapped column name safely from session state
        mapped_col = st.session_state.column_mapping.get(standard_name, "")

        # *** CORRECTION POINT CLÉ POUR TypeError ***
        # Calcule l'index (entier) de la sélection actuelle dans la liste d'options.
        # L'index doit être un entier.
        selectbox_index = 0 # Par défaut, index 0 (l'option vide '')
        if mapped_col and mapped_col in options_list: # Vérifie si la colonne mappée existe dans les options
             try:
                 # Trouve l'index de la colonne mappée DANS LA LISTE DES OPTIONS
                 selectbox_index = options_list.index(mapped_col)
             except ValueError:
                 # Si par hasard la colonne n'est pas trouvée (ne devrait pas arriver avec le 'in' check),
                 # on garde l'index 0 par sécurité.
                 selectbox_index = 0
        # Si mapped_col est '' ou non trouvé, selectbox_index reste 0.

        # Créer le selectbox en utilisant l'index entier calculé
        selected_column = st.sidebar.selectbox(
            label=f"Colonne pour '{standard_name}'", # Utiliser 'label'
            options=options_list,
            index=selectbox_index, # Passe l'index entier calculé
            key=f'map_{standard_name}' # Clé unique pour la gestion d'état
        )

        # Mettre à jour le mapping dans session_state SI la sélection a changé
        # Streamlit gère cela via la clé, mais on peut s'assurer que l'état est cohérent
        if st.session_state.column_mapping[standard_name] != selected_column:
            st.session_state.column_mapping[standard_name] = selected_column
            # Si le mapping change, considérer que les données ne sont plus à jour
            st.session_state.data_processed = False
            st.session_state.backtest_results = None


    # Option pour spécifier le format de date
    st.sidebar.subheader("Format de Date (Optionnel)")
    date_format_value = st.session_state.get('date_format_input', "") # Lire la valeur actuelle
    new_date_format = st.sidebar.text_input(
        "Spécifiez le format de date (ex: %Y-%m-%d)",
        value=date_format_value,
        key='date_format_key',
        help="Laissez vide pour détection automatique. Exemples: %Y-%m-%d, %d/%m/%Y, %d-%b-%y"
    )
    # Mettre à jour l'état si la valeur change
    if date_format_value != new_date_format:
        st.session_state.date_format_input = new_date_format
        # Si le format de date change, considérer que les données ne sont plus à jour
        st.session_state.data_processed = False
        st.session_state.backtest_results = None


    # --- Bouton pour Traiter les Données ---
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # Vérifier que chaque clé requise a une valeur non vide dans le mapping
    all_required_mapped = all(st.session_state.column_mapping.get(key, "") != "" for key in required_keys)

    if st.sidebar.button("Processer les Données", disabled=(not all_required_mapped or not st.session_state.uploaded_file_content)):
        if not st.session_state.uploaded_file_content:
             st.error("Aucun fichier n'est chargé.")
        elif not all_required_mapped:
            st.error("Veuillez mapper toutes les colonnes requises (Date, Open, High, Low, Close, Volume).")
        else:
            st.info("Traitement des données en cours...")
            # Utiliser le contenu stocké pour le traitement
            file_stream = io.BytesIO(st.session_state.uploaded_file_content)
            try:
                processed_df = loader.load_and_process_data(
                    file_stream, # Passer le stream BytesIO
                    st.session_state.column_mapping,
                    st.session_state.date_format_input or None # Passer None si vide
                )

                if processed_df is not None and not processed_df.empty:
                    st.session_state.data = processed_df
                    st.session_state.data_processed = True # Marquer comme traité
                    st.success("Données traitées avec succès.")
                    st.write("Aperçu des données traitées :")
                    st.dataframe(st.session_state.data.head())
                    min_date_str = st.session_state.data.index.min().strftime('%Y-%m-%d')
                    max_date_str = st.session_state.data.index.max().strftime('%Y-%m-%d')
                    st.info(f"Données disponibles du {min_date_str} au {max_date_str}.")
                    st.session_state.backtest_results = None # Réinitialiser les anciens résultats

                else:
                    # Si processed_df est None ou vide après appel de la fonction
                    st.session_state.data = pd.DataFrame()
                    st.session_state.data_processed = False
                    st.error("Impossible de traiter les données. Vérifiez le mapping, le format de date et le contenu du fichier CSV.")

            except Exception as e:
                st.session_state.data = pd.DataFrame()
                st.session_state.data_processed = False
                st.error(f"Erreur lors du traitement des données : {e}")
                st.exception(e) # Affiche la traceback complète dans l'app pour le debug
            finally:
                 file_stream.close() # Fermer le stream

# --- Section Paramètres de Backtest (Affichée si données traitées) ---
# Utiliser le flag data_processed pour conditionner l'affichage
if st.session_state.data_processed and not st.session_state.data.empty:
    st.sidebar.subheader("3. Paramètres du Backtest")

    min_date_data = st.session_state.data.index.min().date()
    max_date_data = st.session_state.data.index.max().date()

    # Utiliser des clés différentes pour les widgets de date pour éviter les conflits
    start_date = st.sidebar.date_input("Date de début", value=min_date_data, min_value=min_date_data, max_value=max_date_data, key='backtest_start_date')
    end_date = st.sidebar.date_input("Date de fin", value=max_date_data, min_value=min_date_data, max_value=max_date_data, key='backtest_end_date')

    initial_capital = st.sidebar.number_input("Capital Initial", min_value=1000, value=100000, step=1000, key='initial_capital')
    short_window = st.sidebar.slider("Période MA Courte", min_value=1, max_value=100, value=40, key='short_window')
    long_window = st.sidebar.slider("Période MA Longue", min_value=2, max_value=250, value=100, key='long_window')

    # --- Bouton de Lancement du Backtest ---
    if st.sidebar.button("Lancer le Backtest", key='run_backtest_button'):
        # Validations avant de lancer
        valid_params = True
        if short_window >= long_window:
            st.error("La période de Moyenne Mobile Courte doit être inférieure à la période Longue.")
            valid_params = False
        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            st.error("La date de début ne peut pas être postérieure à la date de fin.")
            valid_params = False

        if valid_params:
            st.info("Exécution du backtest en cours...")
            try:
                # Filtrer les données sur la plage de dates sélectionnée
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                # S'assurer que l'index est bien de type datetime pour le filtrage
                df_filtered = st.session_state.data.loc[start_dt:end_dt].copy()

                if df_filtered.empty:
                    st.warning(f"Aucune donnée disponible entre {start_date} et {end_date}.")
                    st.session_state.backtest_results = None
                elif len(df_filtered) <= long_window:
                     st.warning(f"Pas assez de données ({len(df_filtered)} jours) pour la période MA longue ({long_window}) dans la plage sélectionnée.")
                     st.session_state.backtest_results = None
                else:
                    st.success(f"Exécution sur {len(df_filtered)} jours ({start_date} à {end_date}).")

                    # Vérifier que la colonne 'Close' existe (devrait être assurée par le loader)
                    if 'Close' not in df_filtered.columns:
                         raise ValueError("Colonne 'Close' standardisée non trouvée dans les données traitées.")

                    # 1. Appliquer la stratégie
                    df_strat = simple_ma.apply_strategy(df_filtered, short_window, long_window)

                    # 2. Exécuter le backtest
                    equity_curve_df = engine.run_backtest(df_strat, initial_capital) # Attend df avec 'Close' et 'positions'

                    if equity_curve_df is not None and not equity_curve_df.empty:
                        # 3. Calculer les métriques
                        performance_metrics = metrics.calculate_performance_metrics(equity_curve_df, initial_capital)

                        # Stocker les résultats
                        st.session_state.backtest_results = {
                            'equity_curve': equity_curve_df['Equity'], # Garder la Series
                            'performance_metrics': performance_metrics,
                            'df_strat_for_plot': df_strat # Garder le df avec signaux, MAs etc.
                        }
                        st.success("Backtest terminé.")
                    else:
                        st.error("Le backtest n'a pas retourné de courbe d'équité valide.")
                        st.session_state.backtest_results = None

            except Exception as e:
                st.error(f"Erreur lors de l'exécution du backtest : {e}")
                st.exception(e) # Affiche la traceback pour debug
                st.session_state.backtest_results = None

# --- Afficher les Résultats du Backtest (Si disponibles) ---
if st.session_state.get('backtest_results'): # Vérifier si la clé existe et n'est pas None
    st.header("Résultats du Backtest")

    results = st.session_state.backtest_results
    metrics_data = results['performance_metrics']
    equity_curve_series = results['equity_curve']
    df_strat_for_plot = results['df_strat_for_plot']

    # Afficher les métriques
    st.subheader("Métriques de Performance")
    met_col1, met_col2 = st.columns(2)
    with met_col1:
        st.metric("Capital Initial", f"{metrics_data.get('Capital Initial', 'N/A'):,.0f}")
        st.metric("Capital Final", f"{metrics_data.get('Capital Final', 'N/A'):,.2f}")
        st.metric("Retour Total (%)", f"{metrics_data.get('Retour Total (%)', 'N/A'):.2f}%")
    with met_col2:
        st.metric("CAGR (%)", f"{metrics_data.get('CAGR (%)', 'N/A'):.2f}%")
        st.metric("Max Drawdown (%)", f"{metrics_data.get('Max Drawdown (%)', 'N/A'):.2f}%")
        st.metric("Ratio de Sharpe", f"{metrics_data.get('Sharpe Ratio', 'N/A'):.2f}")
        # Ajouter d'autres métriques ici si calculées

    st.subheader("Courbe d'Équité")
    st.line_chart(equity_curve_series, use_container_width=True)

    # Afficher le graphique Prix + Signaux + MAs
    st.subheader("Prix, Signaux et Moyennes Mobiles")
    try:
        df_plot = df_strat_for_plot[['Close', 'Short_MA', 'Long_MA']].copy()

        # Identifier les points d'achat/vente (uniquement les changements)
        df_strat_for_plot['position_change'] = df_strat_for_plot['positions'].diff()
        buy_dates = df_strat_for_plot[df_strat_for_plot['position_change'] > 0].index # Entrée en position longue
        sell_dates = df_strat_for_plot[df_strat_for_plot['position_change'] < 0].index # Sortie de position longue

        df_plot['Achat'] = np.nan
        df_plot['Vente'] = np.nan
        df_plot.loc[buy_dates, 'Achat'] = df_plot.loc[buy_dates, 'Close']
        df_plot.loc[sell_dates, 'Vente'] = df_plot.loc[sell_dates, 'Close']

        # Colonnes à afficher dans le graphique
        plot_cols = ['Close', 'Short_MA', 'Long_MA', 'Achat', 'Vente']
        # S'assurer que les colonnes existent avant de les afficher
        cols_to_plot_final = [col for col in plot_cols if col in df_plot.columns]

        st.line_chart(df_plot[cols_to_plot_final], use_container_width=True)
        st.markdown("*(Ligne bleue: Prix de clôture, Autres lignes: Moyennes Mobiles, Points verts: Achat, Points rouges: Vente)*")

    except Exception as e:
        st.error(f"Erreur lors de la préparation du graphique des signaux : {e}")


# --- Messages d'aide initiaux ---
if not st.session_state.get('uploaded_file_name'):
     st.info("⬅️ Veuillez uploader un fichier CSV dans la barre latérale pour commencer.")
elif not st.session_state.get('data_processed'):
     # Message si fichier uploadé mais pas encore traité ou traitement échoué/mapping modifié
     st.warning("⬅️ Fichier chargé. Veuillez vérifier/compléter le mapping des colonnes et cliquer sur 'Processer les Données'.")


# --- Section À Propos ---
st.sidebar.markdown("---")
st.sidebar.header("À Propos")
st.sidebar.info(
    "Outil de backtesting quantitatif simple pour la BRVM."
    "\n\n**Instructions:**"
    "\n1. Chargez un fichier CSV contenant l'historique des cours."
    "\n2. Mappez les colonnes (Date, Open, High, Low, Close, Volume)."
    "\n3. (Optionnel) Spécifiez le format de date si la détection auto échoue."
    "\n4. Cliquez sur 'Processer les Données'."
    "\n5. Ajustez les paramètres du backtest (dates, capital, MAs)."
    "\n6. Cliquez sur 'Lancer le Backtest'."
    "\n\n*Les résultats sont indicatifs.*"
)
