import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

# Configuration de la page Streamlit
st.set_page_config(
    page_title="BRVM Quant Backtest",
    layout="wide",
    menu_items={
        'About': "Analyse quantitative des actions sur la BRVM"
    }
)

# Titre et introduction dynamique
title_placeholder = st.empty()  # Placeholder pour le titre qui sera mis à jour
st.markdown("""
Cette application permet d'analyser et de backtester des stratégies d'investissement
sur les actions cotées à la Bourse Régionale des Valeurs Mobilières (BRVM).
""")

# Upload de fichier CSV
uploaded_file = st.file_uploader("Chargez votre fichier CSV d'historique de cours", type=['csv'])

# Fonction pour charger et traiter les données
def process_data(file):
    try:
        # Tenter de lire le fichier avec différents séparateurs
        try:
            # Essayer de lire les premières lignes pour déterminer le format
            sample = file.read(1024)
            file.seek(0)  # Revenir au début du fichier
            
            # Détecter le séparateur
            if b'\t' in sample:
                separator = '\t'
            elif b';' in sample:
                separator = ';'
            else:
                separator = ','
                
            # Lire le CSV avec le séparateur identifié
            df = pd.read_csv(file, sep=separator)
            
            # Vérifier si la première colonne est numérique (parfois un index)
            if df.columns[0].isdigit() or df.iloc[0, 0].isdigit():
                file.seek(0)
                df = pd.read_csv(file, sep=separator, index_col=0)
                
        except Exception as e:
            st.error(f"Erreur lors de la lecture initiale: {e}")
            try:
                file.seek(0)
                df = pd.read_csv(file)
            except:
                file.seek(0)
                df = pd.read_csv(file, sep=None, engine='python')  # Détection automatique du séparateur
        
        # Afficher les colonnes détectées pour le débogage
        st.write("Colonnes détectées:", list(df.columns))
        
        # Identifier les colonnes nécessaires peu importe la casse
        columns_lower = [col.lower() for col in df.columns]
        
        date_col = None
        open_col = None
        high_col = None
        low_col = None
        close_col = None
        volume_col = None
        
        for i, col_name in enumerate(columns_lower):
            if 'date' in col_name:
                date_col = df.columns[i]
            elif 'open' in col_name:
                open_col = df.columns[i]
            elif 'high' in col_name:
                high_col = df.columns[i]
            elif 'low' in col_name:
                low_col = df.columns[i]
            elif 'close' in col_name:
                close_col = df.columns[i]
            elif 'vol' in col_name:
                volume_col = df.columns[i]
        
        # Si les colonnes n'ont pas été trouvées par nom, essayer par position
        if date_col is None and len(df.columns) >= 1:
            date_col = df.columns[0]
        if open_col is None and len(df.columns) >= 2:
            open_col = df.columns[1]
        if high_col is None and len(df.columns) >= 3:
            high_col = df.columns[2]
        if low_col is None and len(df.columns) >= 4:
            low_col = df.columns[3]
        if close_col is None and len(df.columns) >= 5:
            close_col = df.columns[4]
        if volume_col is None and len(df.columns) >= 6:
            volume_col = df.columns[5]
        
        missing_columns = []
        if date_col is None:
            missing_columns.append("Date")
        if open_col is None:
            missing_columns.append("Open")
        if high_col is None:
            missing_columns.append("High")
        if low_col is None:
            missing_columns.append("Low")
        if close_col is None:
            missing_columns.append("Close")
        if volume_col is None:
            missing_columns.append("Volume")
        
        if missing_columns:
            st.error(f"Colonnes manquantes: {', '.join(missing_columns)}. Assurez-vous que votre CSV contient ces informations.")
            # Afficher les premières lignes pour aider à diagnostiquer
            st.write("Aperçu des données:", df.head())
            return None
        
        # Créer un nouveau DataFrame avec les colonnes standardisées
        df_standardized = pd.DataFrame()
        df_standardized['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df_standardized['Ouverture'] = pd.to_numeric(df[open_col], errors='coerce')
        df_standardized['Plus_Haut'] = pd.to_numeric(df[high_col], errors='coerce')
        df_standardized['Plus_Bas'] = pd.to_numeric(df[low_col], errors='coerce')
        df_standardized['Prix'] = pd.to_numeric(df[close_col], errors='coerce')
        df_standardized['Volume'] = pd.to_numeric(df[volume_col], errors='coerce')
        
        # Supprimer les lignes avec des dates manquantes
        df_standardized = df_standardized.dropna(subset=['Date'])
        
        # Trier par date (du plus ancien au plus récent)
        df_standardized = df_standardized.sort_values('Date')
        
        # Définir la date comme index
        df_standardized = df_standardized.set_index('Date')
        
        # Ajouter la colonne Variation
        df_standardized['Variation'] = df_standardized['Prix'].diff()
        df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
        
        # Remplir les valeurs NaN
        df_standardized = df_standardized.fillna(method='ffill')
        
        return df_standardized
    
    except Exception as e:
        st.error(f"Erreur lors du traitement des données: {e}")
        return None

# Si un fichier est uploadé, traiter les données
if uploaded_file is not None:
    # Demander le titre de l'action
    stock_name = st.text_input("Nom de l'action", "Action")
    
    # Mettre à jour le titre avec le nom de l'action
    title_placeholder.title(f"📈 BRVM Quant Backtest - {stock_name}")
    
    # Traiter les données
    data = process_data(uploaded_file)
    
    if data is not None:
        # Affichage des données brutes (optionnel, avec un bouton pour afficher/masquer)
        with st.expander("Afficher les données brutes"):
            st.dataframe(data.tail(100))
            
        # Visualisation du cours de l'action
        st.subheader(f"Cours historique de {stock_name}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=2)
        ax.set_title(f'Évolution du cours de {stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix (XOF)')
        ax.grid(True, alpha=0.3)
        
        # Amélioration du format des dates sur l'axe X
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Paramètres de la stratégie
        st.subheader("Paramètres de la stratégie")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Paramètres fondamentaux
            st.markdown("### Analyse fondamentale")
            rendement_exige = st.slider("Taux d'actualisation (%)", 5, 20, 12) / 100
            taux_croissance = st.slider("Croissance annuelle dividende (%)", 0, 10, 3) / 100
            dividende_annuel = st.number_input("Dernier dividende annuel (XOF)", 200, 1000, 600)
        
        with col2:
            # Paramètres techniques
            st.markdown("### Règles de trading")
            marge_achat = st.slider("Marge de sécurité à l'achat (%)", 0, 50, 20) / 100
            marge_vente = st.slider("Prime de sortie (%)", 0, 50, 10) / 100
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 10) / 100
        
        # Calcul des moyennes mobiles
        st.subheader("Analyse technique")
        window_court = st.slider("Fenêtre de la moyenne mobile courte", 5, 50, 20)
        window_long = st.slider("Fenêtre de la moyenne mobile longue", 20, 200, 50)
        
        # Calcul des moyennes mobiles
        data['MM_Court'] = data['Prix'].rolling(window=window_court).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long).mean()
        
        # Affichage du graphique avec moyennes mobiles
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
        ax2.plot(data.index, data['MM_Court'], label=f'MM {window_court} jours', linewidth=1.5)
        ax2.plot(data.index, data['MM_Long'], label=f'MM {window_long} jours', linewidth=1.5)
        ax2.set_title('Analyse technique - Moyennes Mobiles')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prix (XOF)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Calcul de la valeur intrinsèque avec le modèle de Gordon
        D1 = dividende_annuel * (1 + taux_croissance)
        val_intrinseque = D1 / (rendement_exige - taux_croissance)
        st.markdown(f"### Valeur intrinsèque calculée: **{val_intrinseque:.2f} XOF**")
        
        # Calcul des signaux d'achat/vente
        data['val_intrinseque'] = val_intrinseque
        data['prix_achat'] = (1 - marge_achat) * val_intrinseque
        data['prix_vente'] = (1 + marge_vente) * val_intrinseque
        
        # Signal technique: croisement des moyennes mobiles
        data['signal_technique'] = 0
        data.loc[data['MM_Court'] > data['MM_Long'], 'signal_technique'] = 1
        data.loc[data['MM_Court'] < data['MM_Long'], 'signal_technique'] = -1
        
        # Combinaison des signaux fondamentaux et techniques
        data['achat'] = (data['Prix'] < data['prix_achat']) & (data['signal_technique'] == 1)
        data['vente'] = (data['Prix'] > data['prix_vente']) | (data['signal_technique'] == -1)
        
        # Affichage du graphique avec zones d'achat/vente
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
        ax3.axhline(y=val_intrinseque, color='g', linestyle='-', alpha=0.5, label='Valeur intrinsèque')
        ax3.axhline(y=data['prix_achat'][0], color='g', linestyle='--', alpha=0.5, label='Prix d\'achat')
        ax3.axhline(y=data['prix_vente'][0], color='r', linestyle='--', alpha=0.5, label='Prix de vente')
        
        # Marquage des signaux d'achat/vente
        achats = data[data['achat'] == True]
        ventes = data[data['vente'] == True]
        
        if not achats.empty:
            ax3.scatter(achats.index, achats['Prix'], color='g', s=50, marker='^', label='Signal d\'achat')
        
        if not ventes.empty:
            ax3.scatter(ventes.index, ventes['Prix'], color='r', s=50, marker='v', label='Signal de vente')
        
        ax3.set_title('Signaux d\'achat et de vente')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Prix (XOF)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Backtest
        st.subheader("Backtest de la stratégie")
        
        capital_initial = st.number_input("Capital initial (XOF)", 100000, 10000000, 1000000, step=100000)
        frais_transaction = st.slider("Frais de transaction (%)", 0.0, 2.0, 0.5) / 100
        
        # Fonction pour exécuter le backtest
        def run_backtest(data, capital_initial, frais_transaction, stop_loss):
            capital = capital_initial
            positions = []
            achats_dates = []
            ventes_dates = []
            prix_achats = []
            prix_ventes = []
            portefeuille_valeur = []
            
            # On utilise un DataFrame pour suivre l'évolution du portefeuille
            portfolio = pd.DataFrame(index=data.index)
            portfolio['prix'] = data['Prix']
            portfolio['actions'] = 0
            portfolio['cash'] = capital_initial
            portfolio['valeur_actions'] = 0
            portfolio['valeur_totale'] = capital_initial
            portfolio['rendement'] = 0
            
            for i in range(1, len(data)):
                jour = data.index[i]
                jour_prec = data.index[i-1]
                prix = data['Prix'].iloc[i]
                
                # Initialisation pour ce jour
                actions = portfolio.loc[jour_prec, 'actions']
                cash = portfolio.loc[jour_prec, 'cash']
                
                # Vérification du stop loss pour les positions existantes
                if actions > 0:
                    prix_achat_moyen = sum(prix_achats) / len(prix_achats) if prix_achats else 0
                    if prix < (1 - stop_loss) * prix_achat_moyen:
                        # Vente forcée (stop loss)
                        vente_montant = actions * prix * (1 - frais_transaction)
                        cash += vente_montant
                        ventes_dates.append(jour)
                        prix_ventes.append(prix)
                        actions = 0
                        prix_achats = []
                
                # Signal d'achat
                if data['achat'].iloc[i] and cash >= prix:
                    # Calcul du nombre d'actions à acheter (maximum possible avec le cash disponible)
                    max_actions = int(cash / (prix * (1 + frais_transaction)))
                    if max_actions > 0:
                        # Achat
                        cout_achat = max_actions * prix * (1 + frais_transaction)
                        cash -= cout_achat
                        actions += max_actions
                        achats_dates.append(jour)
                        prix_achats.append(prix)
                
                # Signal de vente
                elif data['vente'].iloc[i] and actions > 0:
                    # Vente
                    vente_montant = actions * prix * (1 - frais_transaction)
                    cash += vente_montant
                    ventes_dates.append(jour)
                    prix_ventes.append(prix)
                    actions = 0
                    prix_achats = []
                
                # Mise à jour du portfolio pour ce jour
                portfolio.loc[jour, 'actions'] = actions
                portfolio.loc[jour, 'cash'] = cash
                portfolio.loc[jour, 'valeur_actions'] = actions * prix
                portfolio.loc[jour, 'valeur_totale'] = cash + (actions * prix)
                
                # Calcul du rendement quotidien
                if i > 0:
                    rendement_jour = (portfolio.loc[jour, 'valeur_totale'] / portfolio.loc[jour_prec, 'valeur_totale']) - 1
                    portfolio.loc[jour, 'rendement'] = rendement_jour
            
            # Calcul des rendements cumulés
            portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1
            
            return portfolio, achats_dates, ventes_dates
        
        # Exécution du backtest
        portfolio, achats_dates, ventes_dates = run_backtest(data, capital_initial, frais_transaction, stop_loss)
        
        # Affichage des résultats du backtest
        st.subheader("Résultats du backtest")
        
        # Statistiques de performance
        rendement_total = (portfolio['valeur_totale'].iloc[-1] / capital_initial - 1) * 100
        rendement_annualise = ((1 + rendement_total/100) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement total", f"{rendement_total:.2f}%")
        col2.metric("Rendement annualisé", f"{rendement_annualise:.2f}%")
        col3.metric("Valeur finale du portefeuille", f"{portfolio['valeur_totale'].iloc[-1]:,.2f} XOF")
        
        # Graphique de l'évolution du portefeuille
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(portfolio.index, portfolio['valeur_totale'], linewidth=2, label='Valeur du portefeuille')
        ax4.plot(portfolio.index, [capital_initial] * len(portfolio), '--', linewidth=1, color='gray', label='Capital initial')
        
        # Marquage des achats et ventes sur le graphique
        for date in achats_dates:
            ax4.axvline(x=date, color='g', linestyle='--', alpha=0.3)
        for date in ventes_dates:
            ax4.axvline(x=date, color='r', linestyle='--', alpha=0.3)
        
        ax4.set_title('Évolution de la valeur du portefeuille')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Valeur (XOF)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Composition du portefeuille final
        st.subheader("Composition du portefeuille final")
        col1, col2 = st.columns(2)
        col1.metric("Nombre d'actions", f"{portfolio['actions'].iloc[-1]}")
        col2.metric("Liquidités", f"{portfolio['cash'].iloc[-1]:,.2f} XOF")
        
        # Affichage des transactions
        st.subheader("Journal des transactions")
        if achats_dates or ventes_dates:
            # Créer un DataFrame pour les transactions
            transactions = []
            for date in achats_dates:
                prix = data.loc[date, 'Prix']
                transactions.append({
                    'Date': date,
                    'Type': 'Achat',
                    'Prix': prix,
                    'Montant': prix
                })
            
            for date in ventes_dates:
                prix = data.loc[date, 'Prix']
                transactions.append({
                    'Date': date,
                    'Type': 'Vente',
                    'Prix': prix,
                    'Montant': prix
                })
            
            transactions_df = pd.DataFrame(transactions)
            if not transactions_df.empty:
                transactions_df = transactions_df.sort_values('Date')
                st.dataframe(transactions_df)
            else:
                st.info("Aucune transaction n'a été effectuée pendant la période analysée.")
        
        else:
            st.info("Aucune transaction n'a été effectuée pendant la période analysée.")
        
        # Métriques avancées
        st.subheader("Métriques avancées")
        
        # Calcul des rendements journaliers du marché
        data['rendement_marche'] = data['Prix'].pct_change()
        
        # Calcul de la volatilité
        volatilite_strat = portfolio['rendement'].std() * (252 ** 0.5) * 100  # Annualisée
        volatilite_marche = data['rendement_marche'].std() * (252 ** 0.5) * 100  # Annualisée
        
        # Calcul du ratio de Sharpe (en supposant un taux sans risque de 3%)
        taux_sans_risque = 0.03
        sharpe_ratio = (rendement_annualise/100 - taux_sans_risque) / (volatilite_strat/100) if volatilite_strat != 0 else 0
        
        # Calcul du drawdown
        portfolio['peak'] = portfolio['valeur_totale'].cummax()
        portfolio['drawdown'] = (portfolio['valeur_totale'] - portfolio['peak']) / portfolio['peak'] * 100
        max_drawdown = portfolio['drawdown'].min()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Volatilité annualisée", f"{volatilite_strat:.2f}%")
        col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
        col3.metric("Drawdown maximum", f"{max_drawdown:.2f}%")
        
        # Graphique du drawdown
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        ax5.fill_between(portfolio.index, portfolio['drawdown'], 0, color='red', alpha=0.3)
        ax5.set_title('Drawdown du portefeuille')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig5)
        
        # Conclusion et notes
        st.subheader("Conclusion")
        st.markdown(f"""
        Sur la base de ce backtest, la stratégie a généré un rendement total de **{rendement_total:.2f}%**
        sur toute la période analysée, soit un rendement annualisé de **{rendement_annualise:.2f}%**.
        
        **Points clés:**
        
        - Valeur intrinsèque calculée: {val_intrinseque:.2f} XOF
        - Nombre total d'achats: {len(achats_dates)}
        - Nombre total de ventes: {len(ventes_dates)}
        - Volatilité annualisée: {volatilite_strat:.2f}%
        - Ratio de Sharpe: {sharpe_ratio:.2f}
        
        Cette stratégie combine l'analyse fondamentale (valorisation par dividendes) et l'analyse technique
        (croisement de moyennes mobiles) pour identifier les points d'entrée et de sortie optimaux.
        """)
        
        # Avertissement
        st.warning("""
        **Avertissement:** Les performances passées ne préjugent pas des performances futures.
        Cette application est fournie à des fins éducatives uniquement et ne constitue pas un conseil en investissement.
        """)
        
        # Pied de page
        st.markdown("---")
        st.markdown("© 2025 BRVM Quant - Analyse quantitative des marchés financiers africains")
    
else:
    # Si aucun fichier n'est uploadé, afficher un message d'instruction
    title_placeholder.title("📈 BRVM Quant Backtest")
    
    st.info("""
    ### Comment utiliser cette application:
    
    1. Préparez votre fichier CSV avec au minimum les colonnes suivantes: Date, Open, High, Low, Close, Volume
    2. Uploadez votre fichier en utilisant le sélecteur ci-dessus
    3. Entrez le nom de l'action que vous analysez
    4. Ajustez les paramètres de votre stratégie
    5. Analysez les résultats du backtest
    
    **Format attendu du CSV:**
    ```
    Date,Open,High,Low,Close,Volume
    4/22/2025,24000,24000,24000,24000,26210
    4/18/2025,24895,24000,24000,24000,8517
    ...
    ```
    """)
    
    # Exemple de structure de fichier CSV
    st.markdown("""
    ### Exemple de structure de données:
    
    | Date | Open | High | Low | Close | Volume |
    |------|------|------|-----|-------|--------|
    | 4/22/2025 | 24000 | 24000 | 24000 | 24000 | 26210 |
    | 4/18/2025 | 24895 | 24000 | 24000 | 24000 | 8517 |
    | 4/17/2025 | 24890 | 24895 | 24895 | 24895 | 5381 |
    | ... | ... | ... | ... | ... | ... |
    """)
