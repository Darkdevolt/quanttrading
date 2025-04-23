import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64
from pandas.tseries.offsets import BDay # Pour gérer les jours ouvrés

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
        header_line = file.readline().decode('utf-8')
        file.seek(0)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(header_line)
            separator = dialect.delimiter
            st.info(f"Séparateur détecté : '{separator}'")
        except csv.Error:
            st.warning("Impossible de détecter automatiquement le séparateur. Essai avec ',' et ';'.")
            # Essayer les séparateurs courants
            try:
                file.seek(0)
                df_test_comma = pd.read_csv(file, nrows=5, sep=',')
                if len(df_test_comma.columns) > 1:
                    separator = ','
                else:
                    file.seek(0)
                    df_test_semi = pd.read_csv(file, nrows=5, sep=';')
                    if len(df_test_semi.columns) > 1:
                        separator = ';'
                    else:
                        separator = ',' # Défaut
                st.info(f"Utilisation du séparateur : '{separator}'")
            except Exception:
                 separator = ',' # fallback

        file.seek(0)
        df = pd.read_csv(file, sep=separator)

        # --- Validation Initiale ---
        if df.empty:
            st.error("Le fichier CSV est vide.")
            return None

        st.write("Colonnes détectées dans le fichier :", list(df.columns))

        # Vérifier si les colonnes mappées existent
        for standard_name, user_name in column_mapping.items():
            if user_name not in df.columns:
                st.error(f"La colonne mappée '{user_name}' pour '{standard_name}' n'existe pas dans le fichier.")
                return None

        # --- Création du DataFrame Standardisé ---
        df_standardized = pd.DataFrame()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Date
        date_col_name = column_mapping['Date']
        try:
            # Essayer la conversion directe
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce')
            # Si échec et format spécifié, essayer avec le format
            if df_standardized['Date'].isnull().all() and date_format:
                 st.info(f"Tentative de conversion de date avec le format : {date_format}")
                 df_standardized['Date'] = pd.to_datetime(df[date_col_name], format=date_format, errors='coerce')
        except Exception as e:
            st.error(f"Erreur lors de la conversion de la colonne Date ('{date_col_name}'): {e}")
            st.info("Assurez-vous que le format de date est correct ou spécifiez-le dans les options avancées.")
            return None

        if df_standardized['Date'].isnull().all():
             st.error(f"Impossible de convertir la colonne Date ('{date_col_name}') en dates valides.")
             st.info("Vérifiez le contenu de la colonne et le format de date.")
             return None
        if df_standardized['Date'].isnull().any():
            st.warning(f"Certaines valeurs dans la colonne Date ('{date_col_name}') n'ont pas pu être converties et ont été supprimées.")
            df_standardized = df_standardized.dropna(subset=['Date'])
            if df_standardized.empty:
                 st.error("Toutes les lignes ont été supprimées après échec de conversion des dates.")
                 return None

        # Colonnes Numériques
        for col in numeric_cols:
            user_col_name = column_mapping[col]
            standard_col_name = col.replace('Close', 'Prix').replace('Open', 'Ouverture') \
                                 .replace('High', 'Plus_Haut').replace('Low', 'Plus_Bas') # Standardiser les noms

            try:
                # Essayer de nettoyer les caractères non numériques (espaces, devises)
                if df[user_col_name].dtype == 'object':
                     cleaned_col = df[user_col_name].astype(str).str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '.', regex=False)
                     df_standardized[standard_col_name] = pd.to_numeric(cleaned_col, errors='coerce')
                else:
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                if df_standardized[standard_col_name].isnull().any():
                    st.warning(f"Certaines valeurs dans la colonne '{user_col_name}' ({standard_col_name}) ne sont pas numériques et ont été remplacées par NaN.")

            except Exception as e:
                st.error(f"Erreur lors de la conversion de la colonne '{user_col_name}' ({standard_col_name}) en numérique : {e}")
                return None

        # --- Validation Post-Conversion ---
        if df_standardized[numeric_cols].isnull().all().all():
             st.error("Toutes les valeurs dans les colonnes numériques (Open, High, Low, Close, Volume) sont manquantes ou invalides après conversion.")
             return None

        # Supprimer les lignes où le prix est manquant (essentiel)
        initial_rows = len(df_standardized)
        df_standardized = df_standardized.dropna(subset=['Prix'])
        if len(df_standardized) < initial_rows:
            st.warning(f"{initial_rows - len(df_standardized)} lignes supprimées car la valeur 'Prix' (Close) était manquante.")

        if df_standardized.empty:
            st.error("Le DataFrame est vide après suppression des lignes avec 'Prix' manquant.")
            return None

        # --- Traitements Finaux ---
        # Trier par date (important pour les calculs temporels)
        df_standardized = df_standardized.sort_values('Date')

        # Définir la date comme index
        df_standardized = df_standardized.set_index('Date')

        # Calculer Variation
        df_standardized['Variation'] = df_standardized['Prix'].diff()
        df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100

        # Remplir les valeurs NaN restantes (peut être dangereux pour OHLC, mais ok pour Volume si nécessaire)
        # Remplissage prudent : ffill pour les prix (si nécessaire), 0 pour le volume
        cols_to_ffill = ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix']
        for col in cols_to_ffill:
             if col in df_standardized.columns and df_standardized[col].isnull().any():
                  df_standardized[col] = df_standardized[col].ffill()
                  st.info(f"Valeurs manquantes dans '{col}' remplies par propagation avant (ffill).")

        if 'Volume' in df_standardized.columns and df_standardized['Volume'].isnull().any():
             df_standardized['Volume'] = df_standardized['Volume'].fillna(0)
             st.info("Valeurs manquantes dans 'Volume' remplies par 0.")

        # Re-vérifier les NaNs après remplissage
        if df_standardized.isnull().any().any():
            st.warning(f"Il reste des valeurs manquantes après traitement dans les colonnes : {df_standardized.columns[df_standardized.isnull().any()].tolist()}")
            # Optionnel: supprimer les lignes restantes avec NaN si critique
            # df_standardized = df_standardized.dropna()
            # if df_standardized.empty:
            #     st.error("Le DataFrame est vide après suppression finale des lignes avec NaN.")
            #     return None

        st.success("Données chargées et traitées avec succès !")
        return df_standardized

    except pd.errors.EmptyDataError:
        st.error("Erreur : Le fichier CSV est vide ou mal formaté.")
        return None
    except KeyError as e:
        st.error(f"Erreur : Colonne manquante dans le fichier CSV ou le mapping : {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du traitement des données : {e}")
        # Pour le débogage, afficher plus d'infos si nécessaire
        # import traceback
        # st.error(traceback.format_exc())
        return None

# --- Fonction pour Lien de Téléchargement CSV ---
def get_csv_download_link(df, filename="rapport_backtest.csv", link_text="Télécharger le rapport (CSV)"):
    """Génère un lien pour télécharger un DataFrame en CSV."""
    try:
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none; padding: 5px 10px; background-color: #007bff; color: white; border-radius: 5px;">{link_text}</a>'
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
        header_line = uploaded_file.readline().decode('utf-8')
        uploaded_file.seek(0)
        # Simple détection pour la prévisualisation (peut être améliorée)
        sep = ',' if header_line.count(',') >= header_line.count(';') else ';'
        all_columns = pd.read_csv(uploaded_file, sep=sep, nrows=0).columns.tolist()
        uploaded_file.seek(0) # Important de revenir au début

        if not all_columns:
            st.sidebar.error("Impossible de lire les colonnes du fichier.")
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
            for standard_name, display_name in required_map.items():
                 # Essayer de pré-sélectionner si un nom correspond (ignorer la casse)
                 default_selection = None
                 for col in all_columns:
                      if standard_name.lower() in col.lower():
                           default_selection = col
                           break

                 column_mapping[standard_name] = st.sidebar.selectbox(
                      f"Colonne pour '{display_name}'",
                      options=[""] + all_columns, # Ajouter option vide
                      index=all_columns.index(default_selection)+1 if default_selection else 0,
                      key=f"map_{standard_name}"
                 )

            with st.sidebar.expander("Options Avancées"):
                 date_format_input = st.text_input("Format de date (si nécessaire, ex: %d/%m/%Y)", key="date_format")

            # Bouton pour lancer le traitement
            if st.sidebar.button("▶️ Traiter les Données", key="process_button"):
                if not all(col in column_mapping and column_mapping[col] for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']):
                     st.warning("Veuillez mapper toutes les colonnes requises avant de traiter.")
                else:
                     with st.spinner("Traitement des données en cours..."):
                          data = process_data(uploaded_file, column_mapping, date_format_input)

    except Exception as e:
        st.sidebar.error(f"Erreur lors de la lecture de l'en-tête du fichier : {e}")
        st.sidebar.info("Assurez-vous que le fichier est un CSV valide.")

# --- Exécution de l'Analyse (si les données sont chargées) ---
if data is not None and not data.empty:

    # --- Nom de l'action ---
    st.sidebar.subheader("3. Informations Action")
    stock_name = st.sidebar.text_input("Nom de l'action", "MonActionBRVM", key="stock_name")
    st.title(f"📈 BRVM Quant Backtest - {stock_name}") # Mettre à jour le titre principal

    # --- Affichage des Données Traitées ---
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)"):
        st.dataframe(data.tail(100))
        st.markdown(get_csv_download_link(data.tail(100), filename=f"data_preview_{stock_name}.csv", link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)

    # --- Visualisation du Cours ---
    st.subheader(f"Cours historique de {stock_name}")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=2, label='Prix de Clôture')
        ax.set_title(f'Évolution du cours de {stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix (FCFA)') # Supposer FCFA, pourrait être paramétrable
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Amélioration du format des dates sur l'axe X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Format plus précis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10)) # Ajustement automatique
        plt.xticks(rotation=30, ha='right') # Rotation légère pour lisibilité
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique de cours : {e}")


    # --- Paramètres de la Stratégie (dans la Sidebar) ---
    st.sidebar.subheader("4. Paramètres de la Stratégie")

    # Paramètres fondamentaux
    st.sidebar.markdown("**Analyse Fondamentale (Modèle Gordon-Shapiro)**")
    rendement_exige = st.sidebar.slider("Taux d'actualisation (%)", 5.0, 25.0, 12.0, 0.5, help="Taux de rendement minimum exigé par l'investisseur.", key="discount_rate") / 100
    taux_croissance = st.sidebar.slider("Croissance annuelle dividende (%)", 0.0, 15.0, 3.0, 0.5, help="Taux de croissance annuel attendu des dividendes.", key="growth_rate") / 100
    dividende_annuel = st.sidebar.number_input("Dernier dividende annuel (FCFA)", min_value=0.0, value=600.0, step=10.0, help="Dividende versé lors de la dernière période.", key="dividend")

    # Validation Gordon-Shapiro
    if rendement_exige <= taux_croissance:
         st.sidebar.error("Le taux d'actualisation doit être supérieur au taux de croissance des dividendes pour le modèle de Gordon-Shapiro.")
         st.stop() # Arrêter l'exécution si condition invalide

    # Calcul Valeur Intrinsèque
    try:
         D1 = dividende_annuel * (1 + taux_croissance)
         val_intrinseque = D1 / (rendement_exige - taux_croissance)
         st.sidebar.metric("Valeur Intrinsèque (estimée)", f"{val_intrinseque:,.2f} FCFA")
    except ZeroDivisionError:
         st.sidebar.error("Division par zéro : Le taux d'actualisation ne peut être égal au taux de croissance.")
         st.stop()
    except Exception as e:
         st.sidebar.error(f"Erreur calcul valeur intrinsèque: {e}")
         st.stop()

    # Paramètres Techniques et Trading Rules
    st.sidebar.markdown("**Règles de Trading Techniques**")
    marge_achat = st.sidebar.slider("Marge de sécurité à l'achat (%)", 0, 50, 20, help="Achat si Prix < (1 - Marge) * Valeur Intrinsèque.", key="buy_margin") / 100
    marge_vente = st.sidebar.slider("Prime de sortie (%)", 0, 50, 10, help="Vente si Prix > (1 + Prime) * Valeur Intrinsèque (signal fondamental).", key="sell_premium") / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 25.0, 10.0, 0.5, help="Vente si le prix baisse de ce % par rapport au prix d'achat moyen.", key="stop_loss") / 100
    take_profit = st.sidebar.slider("Take Profit (%)", 5.0, 100.0, 20.0, 1.0, help="Vente si le prix augmente de ce % par rapport au prix d'achat moyen.", key="take_profit") / 100

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
    plafond_variation = st.sidebar.slider("Plafond variation journalière (%)", 5.0, 15.0, 7.5, 0.5, help="Variation maximale autorisée par jour (ex: 7.5%).", key="variation_cap") / 100
    delai_livraison = st.sidebar.slider("Délai de livraison (jours ouvrés)", 1, 5, 3, help="Nombre de jours ouvrés pour la livraison des titres après transaction (T+N).", key="settlement_days")

    # Paramètres Backtest
    st.sidebar.subheader("5. Paramètres du Backtest")
    capital_initial = st.sidebar.number_input("Capital initial (FCFA)", 100000, 100000000, 1000000, step=100000, key="initial_capital")
    frais_transaction = st.sidebar.slider("Frais de transaction (%) par ordre", 0.0, 5.0, 0.5, 0.05, help="Pourcentage de frais appliqué à chaque achat et vente.", key="commission_rate") / 100
    taux_sans_risque = st.sidebar.slider("Taux sans risque annuel (%)", 0.0, 10.0, 3.0, 0.1, help="Taux utilisé pour le calcul du Ratio de Sharpe.", key="risk_free_rate") / 100


    # --- Calculs Techniques et Signaux ---
    st.subheader("Analyse Technique et Signaux")

    # Calcul des moyennes mobiles
    try:
        data['MM_Court'] = data['Prix'].rolling(window=window_court, min_periods=window_court).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long, min_periods=window_long).mean()
    except Exception as e:
        st.error(f"Erreur lors du calcul des moyennes mobiles : {e}")
        st.stop()

    # Affichage du graphique avec moyennes mobiles
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', linewidth=1, alpha=0.8)
        ax2.plot(data.index, data['MM_Court'], label=f'MM {window_court} jours', linewidth=1.5)
        ax2.plot(data.index, data['MM_Long'], label=f'MM {window_long} jours', linewidth=1.5)
        ax2.set_title('Analyse Technique - Moyennes Mobiles')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prix (FCFA)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique des moyennes mobiles : {e}")


    # Calcul des niveaux de prix pour signaux fondamentaux
    data['val_intrinseque'] = val_intrinseque
    data['prix_achat_fondamental'] = (1 - marge_achat) * val_intrinseque
    data['prix_vente_fondamental'] = (1 + marge_vente) * val_intrinseque

    # Signal technique: croisement des moyennes mobiles
    # Initialiser avec 0, puis assigner 1 (achat) ou -1 (vente)
    data['signal_technique'] = 0
    # Condition d'achat : MM Courte passe AU-DESSUS de MM Longue
    buy_signal_cond = (data['MM_Court'] > data['MM_Long']) & (data['MM_Court'].shift(1) <= data['MM_Long'].shift(1))
    data.loc[buy_signal_cond, 'signal_technique'] = 1

    # Condition de vente : MM Courte passe EN DESSOUS de MM Longue
    sell_signal_cond = (data['MM_Court'] < data['MM_Long']) & (data['MM_Court'].shift(1) >= data['MM_Long'].shift(1))
    data.loc[sell_signal_cond, 'signal_technique'] = -1

    # Signaux combinés (Exemple : Fondamental ET Technique pour achat)
    # Achat: Prix sous la marge ET croisement haussier des MM vient de se produire
    data['achat'] = (data['Prix'] < data['prix_achat_fondamental']) & (data['signal_technique'] == 1)

    # Vente: Prix au-dessus de la prime OU croisement baissier des MM vient de se produire
    data['vente_signal'] = (data['Prix'] > data['prix_vente_fondamental']) | (data['signal_technique'] == -1)
    # Note: La vente réelle dans le backtest sera aussi déclenchée par Stop Loss / Take Profit

    # Affichage du graphique avec zones et signaux
    try:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
        ax3.axhline(y=val_intrinseque, color='grey', linestyle='-', alpha=0.7, label=f'Valeur Intrinsèque ({val_intrinseque:,.0f})')
        ax3.axhline(y=data['prix_achat_fondamental'].iloc[0], color='green', linestyle='--', alpha=0.6, label=f'Seuil Achat Fondamental ({data["prix_achat_fondamental"].iloc[0]:,.0f})')
        ax3.axhline(y=data['prix_vente_fondamental'].iloc[0], color='red', linestyle='--', alpha=0.6, label=f'Seuil Vente Fondamental ({data["prix_vente_fondamental"].iloc[0]:,.0f})')

        # Marquage des signaux déclenchés (basé sur la logique combinée)
        achats_signaux = data[data['achat']]
        ventes_signaux = data[data['vente_signal']] # Signaux initiaux (avant SL/TP)

        if not achats_signaux.empty:
            ax3.scatter(achats_signaux.index, achats_signaux['Prix'], color='green', s=60, marker='^', label='Signal Achat Combiné', zorder=5)

        if not ventes_signaux.empty:
            ax3.scatter(ventes_signaux.index, ventes_signaux['Prix'], color='red', s=60, marker='v', label='Signal Vente Combiné/Fondamental', zorder=5)

        ax3.set_title('Prix, Valeur Intrinsèque et Signaux de Trading Initiaux')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper left')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)
    except Exception as e:
         st.error(f"Erreur lors de la génération du graphique des signaux : {e}")

    # --- Backtest ---
    st.subheader("🚀 Backtest de la Stratégie")
    st.markdown(f"Exécution du backtest avec un capital initial de **{capital_initial:,.0f} FCFA** et des frais de **{frais_transaction*100:.2f}%**.")

    # Fonction pour exécuter le backtest (Améliorée avec Jours Ouvrés et gestion cash)
    def run_backtest(data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison):
        """
        Exécute le backtest en tenant compte des jours ouvrés pour la livraison.

        Args:
            data (pd.DataFrame): DataFrame contenant les prix et signaux.
            capital_initial (float): Capital de départ.
            frais_transaction (float): Pourcentage de frais par transaction.
            stop_loss (float): Pourcentage de perte max avant vente.
            take_profit (float): Pourcentage de gain avant vente.
            plafond_variation (float): Variation journalière max autorisée.
            delai_livraison (int): Nombre de jours ouvrés pour la livraison.

        Returns:
            tuple: (pd.DataFrame: historique du portefeuille, list: dates d'achat, list: dates de vente, pd.DataFrame: journal des transactions)
        """
        portfolio = pd.DataFrame(index=data.index)
        portfolio['prix'] = data['Prix'] # Prix de référence du jour
        portfolio['actions'] = 0.0        # Actions détenues à la fin de la journée (après livraison)
        portfolio['cash'] = float(capital_initial) # Liquidités disponibles
        portfolio['valeur_actions'] = 0.0 # Valeur des actions détenues
        portfolio['valeur_totale'] = float(capital_initial) # Valeur totale (cash + actions)
        portfolio['rendement'] = 0.0      # Rendement journalier
        portfolio['trade_en_cours'] = False # Indicateur si une opération est en attente de livraison

        transactions = [] # Pour stocker les détails des trades
        achats_dates = []
        ventes_dates = []
        prix_achat_moyen = 0.0
        nb_actions_achetees = 0.0
        date_livraison_attendue = None

        # Utiliser BDay pour les jours ouvrés
        bday = BDay()

        for i in range(1, len(data)):
            jour = data.index[i]
            jour_prec = data.index[i-1]

            # --- Initialisation pour la journée ---
            portfolio.loc[jour, ['actions', 'cash', 'valeur_actions', 'valeur_totale', 'trade_en_cours']] = portfolio.loc[jour_prec, ['actions', 'cash', 'valeur_actions', 'valeur_totale', 'trade_en_cours']]

            # --- Appliquer le Plafond de Variation (Simulation BRVM) ---
            prix_jour = data['Prix'].iloc[i]
            prix_veille = data['Prix'].iloc[i-1]
            variation = (prix_jour - prix_veille) / prix_veille if prix_veille != 0 else 0

            prix_effectif = prix_jour # Prix utilisé pour les transactions
            if abs(variation) > plafond_variation:
                if variation > 0:
                    prix_effectif = prix_veille * (1 + plafond_variation)
                else:
                    prix_effectif = prix_veille * (1 - plafond_variation)
                # Note: On pourrait logguer cet ajustement si besoin
                # print(f"{jour.date()}: Plafond atteint. Prix ajusté de {prix_jour:.2f} à {prix_effectif:.2f}")

            portfolio.loc[jour, 'prix'] = prix_effectif # Stocker le prix effectif

            # --- Vérification Livraison ---
            actions_actuelles = portfolio.loc[jour, 'actions']
            cash_actuel = portfolio.loc[jour, 'cash']

            if date_livraison_attendue is not None and jour >= date_livraison_attendue:
                 # Livraison effectuée
                 portfolio.loc[jour, 'trade_en_cours'] = False
                 date_livraison_attendue = None
                 # Note: Les actions/cash sont déjà ajustés au moment de l'ordre dans cette logique

            # --- Évaluation des Ventes (Stop Loss, Take Profit, Signal) ---
            vendre = False
            raison_vente = ""
            if actions_actuelles > 0 and not portfolio.loc[jour, 'trade_en_cours']: # Ne pas trader si une op est en cours
                # 1. Stop Loss
                if prix_achat_moyen > 0 and prix_effectif < prix_achat_moyen * (1 - stop_loss):
                    vendre = True
                    raison_vente = f"Stop Loss ({stop_loss*100:.1f}%)"
                # 2. Take Profit
                elif prix_achat_moyen > 0 and prix_effectif > prix_achat_moyen * (1 + take_profit):
                    vendre = True
                    raison_vente = f"Take Profit ({take_profit*100:.1f}%)"
                # 3. Signal de Vente (combiné/fondamental)
                elif data['vente_signal'].iloc[i]:
                    vendre = True
                    raison_vente = "Signal Vente Stratégie"

                if vendre:
                    montant_vente = actions_actuelles * prix_effectif * (1 - frais_transaction)
                    cash_actuel += montant_vente # Cash reçu à la livraison
                    ventes_dates.append(jour)
                    date_livraison_attendue = jour + bday * delai_livraison # Calcul date livraison ouvrée

                    transactions.append({
                        'Date Ordre': jour,
                        'Date Livraison': date_livraison_attendue,
                        'Type': 'Vente',
                        'Raison': raison_vente,
                        'Quantité': actions_actuelles,
                        'Prix Unitaire': prix_effectif,
                        'Frais': actions_actuelles * prix_effectif * frais_transaction,
                        'Montant Net': montant_vente
                    })

                    st.write(f"🔔 {jour.date()}: {raison_vente} déclenché. Vente de {actions_actuelles:.0f} actions à {prix_effectif:,.2f}. Livraison prévue: {date_livraison_attendue.date()}")

                    actions_actuelles = 0.0
                    nb_actions_achetees = 0.0
                    prix_achat_moyen = 0.0
                    portfolio.loc[jour, 'trade_en_cours'] = True # Bloquer nouvelles opérations

            # --- Évaluation Achat ---
            # Acheter seulement si on n'a pas d'actions et pas de trade en cours
            if actions_actuelles == 0 and data['achat'].iloc[i] and not portfolio.loc[jour, 'trade_en_cours']:
                if cash_actuel > 0:
                    # Calcul du nombre d'actions achetables
                    nb_actions_a_acheter = int(cash_actuel / (prix_effectif * (1 + frais_transaction)))

                    if nb_actions_a_acheter > 0:
                        cout_achat = nb_actions_a_acheter * prix_effectif * (1 + frais_transaction)
                        cash_actuel -= cout_achat # Cash débité immédiatement
                        achats_dates.append(jour)
                        date_livraison_attendue = jour + bday * delai_livraison

                        # Mettre à jour actions et prix moyen APRÈS livraison (simplifié ici, ajusté à l'ordre)
                        actions_actuelles = nb_actions_a_acheter # Actions reçues à la livraison
                        nb_actions_achetees = nb_actions_a_acheter
                        prix_achat_moyen = prix_effectif # Prix de cette transaction devient le prix moyen

                        transactions.append({
                            'Date Ordre': jour,
                            'Date Livraison': date_livraison_attendue,
                            'Type': 'Achat',
                            'Raison': 'Signal Achat Stratégie',
                            'Quantité': nb_actions_a_acheter,
                            'Prix Unitaire': prix_effectif,
                            'Frais': nb_actions_a_acheter * prix_effectif * frais_transaction,
                            'Montant Net': -cout_achat
                        })

                        st.write(f"🔔 {jour.date()}: Signal Achat. Achat de {nb_actions_a_acheter:.0f} actions à {prix_effectif:,.2f}. Livraison prévue: {date_livraison_attendue.date()}")

                        portfolio.loc[jour, 'trade_en_cours'] = True # Bloquer nouvelles opérations


            # --- Mise à jour quotidienne du portefeuille ---
            portfolio.loc[jour, 'actions'] = actions_actuelles
            portfolio.loc[jour, 'cash'] = cash_actuel
            portfolio.loc[jour, 'valeur_actions'] = actions_actuelles * prix_effectif # Valorisation au prix effectif du jour
            portfolio.loc[jour, 'valeur_totale'] = portfolio.loc[jour, 'cash'] + portfolio.loc[jour, 'valeur_actions']

            # Calcul du rendement quotidien
            valeur_totale_veille = portfolio.loc[jour_prec, 'valeur_totale']
            if valeur_totale_veille != 0:
                 portfolio.loc[jour, 'rendement'] = (portfolio.loc[jour, 'valeur_totale'] / valeur_totale_veille) - 1
            else:
                 portfolio.loc[jour, 'rendement'] = 0

        # Calcul des rendements cumulés à la fin
        portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1

        # Créer DataFrame des transactions
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df = transactions_df.sort_values('Date Ordre').set_index('Date Ordre')


        return portfolio, achats_dates, ventes_dates, transactions_df
    # --- Fin fonction run_backtest ---

    # Exécution
    try:
        with st.spinner("Exécution du backtest..."):
            portfolio_history, achats_dates, ventes_dates, journal_transactions = run_backtest(
                data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison
            )
        st.success("Backtest terminé.")
    except Exception as e:
        st.error(f"Une erreur est survenue durant l'exécution du backtest : {e}")
        import traceback
        st.error(traceback.format_exc()) # Afficher la trace pour le débogage
        st.stop()


    # --- Affichage des Résultats du Backtest ---
    st.subheader("📊 Résultats du Backtest")

    # Statistiques Clés
    try:
        valeur_finale = portfolio_history['valeur_totale'].iloc[-1]
        rendement_total_pct = (valeur_finale / capital_initial - 1) * 100
        # Rendement annualisé (plus robuste)
        jours_total = (portfolio_history.index[-1] - portfolio_history.index[0]).days
        if jours_total > 0:
             rendement_annualise_pct = ((valeur_finale / capital_initial) ** (365.25 / jours_total) - 1) * 100
        else:
             rendement_annualise_pct = 0.0 # ou NaN

        col1, col2, col3 = st.columns(3)
        col1.metric("Valeur Finale Portefeuille", f"{valeur_finale:,.2f} FCFA", f"{valeur_finale-capital_initial:,.2f} FCFA")
        col2.metric("Rendement Total", f"{rendement_total_pct:.2f}%")
        col3.metric("Rendement Annualisé", f"{rendement_annualise_pct:.2f}%")

    except Exception as e:
        st.error(f"Erreur lors du calcul des statistiques de performance : {e}")

    # Graphique Évolution Portefeuille
    try:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(portfolio_history.index, portfolio_history['valeur_totale'], linewidth=2, label='Valeur du portefeuille', color='blue')
        ax4.axhline(y=capital_initial, linestyle='--', linewidth=1, color='grey', label='Capital initial')

        # Marquage achats/ventes (optionnel, peut surcharger)
        # y_min, y_max = ax4.get_ylim()
        # if achats_dates:
        #     ax4.scatter(achats_dates, [portfolio_history.loc[d, 'valeur_totale'] for d in achats_dates], color='green', marker='^', s=50, label='Achats', zorder=5)
        # if ventes_dates:
        #      ax4.scatter(ventes_dates, [portfolio_history.loc[d, 'valeur_totale'] for d in ventes_dates], color='red', marker='v', s=50, label='Ventes', zorder=5)
        # ax4.vlines(achats_dates, ymin=y_min, ymax=y_max, color='green', linestyle='--', alpha=0.3, linewidth=1)
        # ax4.vlines(ventes_dates, ymin=y_min, ymax=y_max, color='red', linestyle='--', alpha=0.3, linewidth=1)


        ax4.set_title('Évolution de la Valeur du Portefeuille')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Valeur (FCFA)')
        ax4.grid(True, linestyle='--', alpha=0.6)
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig4)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique d'évolution du portefeuille : {e}")

    # Composition Finale Portefeuille
    st.subheader("💼 Composition du Portefeuille Final")
    try:
        actions_finales = portfolio_history['actions'].iloc[-1]
        cash_final = portfolio_history['cash'].iloc[-1]
        # actions_en_attente_final = portfolio_history['actions_en_attente'].iloc[-1] # Si colonne existe
        trade_en_cours_final = portfolio_history['trade_en_cours'].iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre d'actions détenues", f"{actions_finales:,.0f}")
        col2.metric("Liquidités (Cash)", f"{cash_final:,.2f} FCFA")
        col3.metric("Trade en attente de livraison?", "Oui" if trade_en_cours_final else "Non")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de la composition finale : {e}")


    # Journal des Transactions
    with st.expander("📜 Journal des Transactions"):
        if journal_transactions is not None and not journal_transactions.empty:
            st.dataframe(journal_transactions.style.format({
                'Quantité': '{:,.0f}',
                'Prix Unitaire': '{:,.2f}',
                'Frais': '{:,.2f}',
                'Montant Net': '{:,.2f}'
            }))
            st.markdown(get_csv_download_link(journal_transactions, filename=f"transactions_{stock_name}.csv", link_text="Télécharger le journal (CSV)"), unsafe_allow_html=True)
        else:
            st.info("Aucune transaction n'a été effectuée pendant la période de backtest.")

    # Métriques Avancées
    st.subheader("⚙️ Métriques de Performance Avancées")
    try:
        # Volatilité Annualisée
        volatilite_strat_pct = portfolio_history['rendement'].std() * np.sqrt(252) * 100 # 252 jours de trading approx.

        # Ratio de Sharpe
        rendement_journalier_moyen = portfolio_history['rendement'].mean()
        rendement_annualise_strat = (1 + rendement_journalier_moyen)**252 - 1
        volatilite_annualisee_strat = portfolio_history['rendement'].std() * np.sqrt(252)
        sharpe_ratio = (rendement_annualise_strat - taux_sans_risque) / volatilite_annualisee_strat if volatilite_annualisee_strat != 0 else 0

        # Drawdown Maximum
        portfolio_history['peak'] = portfolio_history['valeur_totale'].cummax()
        portfolio_history['drawdown'] = (portfolio_history['valeur_totale'] - portfolio_history['peak']) / portfolio_history['peak']
        max_drawdown_pct = portfolio_history['drawdown'].min() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Volatilité Annualisée", f"{volatilite_strat_pct:.2f}%")
        col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}", help=f"Basé sur un taux sans risque de {taux_sans_risque*100:.1f}%")
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
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig5)

        # Graphique Distribution Rendements
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        ax6.hist(portfolio_history['rendement'].dropna() * 100, bins=50, alpha=0.75, density=True)
        ax6.set_title('Distribution des Rendements Journaliers de la Stratégie')
        ax6.set_xlabel('Rendement Journalier (%)')
        ax6.set_ylabel('Densité de Fréquence')
        ax6.grid(True, alpha=0.3)
        # Ajouter une ligne pour moyenne = 0
        ax6.axvline(portfolio_history['rendement'].mean() * 100, color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {portfolio_history["rendement"].mean()*100:.3f}%')
        ax6.legend()
        plt.tight_layout()
        st.pyplot(fig6)

    except Exception as e:
        st.error(f"Erreur lors du calcul ou de l'affichage des métriques avancées : {e}")

    # Comparaison Buy & Hold
    st.subheader("⚖️ Comparaison avec Stratégie 'Buy & Hold'")
    try:
        # Calcul performance Buy & Hold
        prix_initial_bh = data['Prix'].iloc[0]
        prix_final_bh = data['Prix'].iloc[-1]
        rendement_total_bh_pct = (prix_final_bh / prix_initial_bh - 1) * 100

        if jours_total > 0:
             rendement_annualise_bh_pct = ((prix_final_bh / prix_initial_bh) ** (365.25 / jours_total) - 1) * 100
        else:
             rendement_annualise_bh_pct = 0.0

        # Calcul Volatilité et Drawdown Buy & Hold
        data['rendement_bh'] = data['Prix'].pct_change()
        volatilite_bh_pct = data['rendement_bh'].std() * np.sqrt(252) * 100
        data['peak_bh'] = data['Prix'].cummax()
        data['drawdown_bh'] = (data['Prix'] - data['peak_bh']) / data['peak_bh']
        max_drawdown_bh_pct = data['drawdown_bh'].min() * 100

        st.markdown("### Performance Buy & Hold")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement Total B&H", f"{rendement_total_bh_pct:.2f}%")
        col2.metric("Rendement Annualisé B&H", f"{rendement_annualise_bh_pct:.2f}%")
        col3.metric("Max Drawdown B&H", f"{max_drawdown_bh_pct:.2f}%")

        st.markdown("### Comparaison Directe")
        col1, col2, col3 = st.columns(3)
        col1.metric("Surperformance (Total)", f"{rendement_total_pct - rendement_total_bh_pct:.2f}%")
        col2.metric("Surperformance (Annualisée)", f"{rendement_annualise_pct - rendement_annualise_bh_pct:.2f}%")
        col3.metric("Différence Volatilité", f"{volatilite_strat_pct - volatilite_bh_pct:.2f}%", help="Négatif = Stratégie moins volatile")


        # Graphique Comparatif Performances Normalisées
        fig7, ax7 = plt.subplots(figsize=(12, 6))
        # Performance normalisée base 1
        perf_strategie = (1 + portfolio_history['rendement_cumule']) # * capital_initial # Déjà calculé
        perf_buy_hold = (data['Prix'] / prix_initial_bh) # * capital_initial

        ax7.plot(portfolio_history.index, perf_strategie * capital_initial, label=f'Stratégie ({stock_name})', linewidth=2, color='blue')
        ax7.plot(data.index, perf_buy_hold * capital_initial, label=f'Buy & Hold ({stock_name})', linewidth=2, linestyle='--', color='orange')

        ax7.set_title('Comparaison Normalisée des Performances (Base = Capital Initial)')
        ax7.set_xlabel('Date')
        ax7.set_ylabel(f'Valeur Portefeuille (Base {capital_initial:,.0f} FCFA)')
        ax7.grid(True, linestyle='--', alpha=0.6)
        ax7.legend()
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax7.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig7)

    except Exception as e:
        st.error(f"Erreur lors de la comparaison avec Buy & Hold : {e}")


    # --- Téléchargement du Rapport Complet du Portefeuille ---
    st.subheader("📥 Télécharger le Rapport Complet")
    st.markdown(get_csv_download_link(portfolio_history, filename=f"rapport_backtest_{stock_name}.csv", link_text="Télécharger l'historique du portefeuille (CSV)"), unsafe_allow_html=True)

    st.info("""
    **Note sur l'interprétation :**
    * **Ratio de Sharpe :** Mesure le rendement ajusté au risque (plus élevé = mieux).
    * **Drawdown Maximum :** La perte maximale historique du pic au creux (plus faible = mieux).
    * **Volatilité :** Mesure l'ampleur des variations de prix/rendement (plus faible = moins risqué/variable).
    * La comparaison avec **Buy & Hold** montre si la stratégie active a ajouté de la valeur par rapport à un simple achat initial.
    * **Hypothèses :** Les frais sont fixes, les ordres sont exécutés au prix du jour (avec plafond), la livraison prend N jours ouvrés. Pas de prise en compte de la liquidité ou du slippage.
    """)

else:
    # Message si aucune donnée n'est chargée ou traitée
    if uploaded_file is None:
        st.info("👈 Veuillez charger un fichier CSV via la barre latérale pour commencer l'analyse.")
    elif data is None and uploaded_file is not None:
         # Si fichier chargé mais data est None, c'est qu'il y a eu une erreur de traitement ou que le bouton n'a pas été cliqué
         st.warning("⚠️ Veuillez mapper les colonnes et cliquer sur 'Traiter les Données' dans la barre latérale.")
    elif data is not None and data.empty:
         st.error("❌ Le traitement des données a résulté en un DataFrame vide. Vérifiez votre fichier et le mapping des colonnes.")


# --- Pied de page ---
st.markdown("---")
st.markdown("Application de Backtesting BRVM v1.1 - Améliorations Implémentées")
