import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io
import base64

# Configuration de la page Streamlit
st.set_page_config(
    page_title="BRVM Quant Backtest",
    layout="wide",
    menu_items={
        'About': "Analyse quantitative des actions sur la BRVM"
    }
)

# Titre et introduction dynamique
title_placeholder = st.empty()
st.markdown("""
Cette application permet d'analyser et de backtester des stratégies d'investissement
sur les actions cotées à la Bourse Régionale des Valeurs Mobilières (BRVM).
""")

# Upload de fichier CSV
uploaded_file = st.file_uploader("Chargez votre fichier CSV d'historique de cours", type=['csv'])

# Fonction pour charger et traiter les données
def process_data(file):
    try:
        try:
            sample = file.read(1024)
            file.seek(0)
            
            if b'\t' in sample:
                separator = '\t'
            elif b';' in sample:
                separator = ';'
            else:
                separator = ','
                
            df = pd.read_csv(file, sep=separator)
            
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
                df = pd.read_csv(file, sep=None, engine='python')
        
        st.write("Colonnes détectées:", list(df.columns))
        
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
            st.error(f"Colonnes manquantes: {', '.join(missing_columns)}")
            st.write("Aperçu des données:", df.head())
            return None
        
        df_standardized = pd.DataFrame()
        df_standardized['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df_standardized['Ouverture'] = pd.to_numeric(df[open_col], errors='coerce')
        df_standardized['Plus_Haut'] = pd.to_numeric(df[high_col], errors='coerce')
        df_standardized['Plus_Bas'] = pd.to_numeric(df[low_col], errors='coerce')
        df_standardized['Prix'] = pd.to_numeric(df[close_col], errors='coerce')
        df_standardized['Volume'] = pd.to_numeric(df[volume_col], errors='coerce')
        
        df_standardized = df_standardized.dropna(subset=['Date'])
        df_standardized = df_standardized.sort_values('Date')
        df_standardized = df_standardized.set_index('Date')
        
        df_standardized['Variation'] = df_standardized['Prix'].diff()
        df_standardized['Variation_%'] = df_standardized['Prix'].pct_change() * 100
        df_standardized = df_standardized.fillna(method='ffill')
        
        return df_standardized
    
    except Exception as e:
        st.error(f"Erreur lors du traitement des données: {e}")
        return None

def get_csv_download_link(df, filename="rapport_backtest.csv"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger le rapport (CSV)</a>'
    return href

if uploaded_file is not None:
    stock_name = st.text_input("Nom de l'action", "Action")
    title_placeholder.title(f"📈 BRVM Quant Backtest - {stock_name}")
    data = process_data(uploaded_file)
    
    if data is not None:
        with st.expander("Afficher les données brutes"):
            st.dataframe(data.tail(100))
            
        st.subheader(f"Cours historique de {stock_name}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=2)
        ax.set_title(f'Évolution du cours de {stock_name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix (FCFA)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Paramètres de la stratégie")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Analyse fondamentale")
            rendement_exige = st.slider("Taux d'actualisation (%)", 5, 20, 12) / 100
            taux_croissance = st.slider("Croissance annuelle dividende (%)", 0, 10, 3) / 100
            dividende_annuel = st.number_input("Dernier dividende annuel (FCFA)", min_value=0, value=600)
        
        with col2:
            st.markdown("### Règles de trading")
            marge_achat = st.slider("Marge de sécurité à l'achat (%)", 0, 50, 20) / 100
            marge_vente = st.slider("Prime de sortie (%)", 0, 50, 10) / 100
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 10) / 100
            take_profit = st.slider("Take Profit (%)", 5, 50, 20) / 100
        
        st.subheader("Paramètres spécifiques à la BRVM")
        col1, col2 = st.columns(2)
        with col1:
            plafond_variation = st.slider("Plafond de variation journalière (%)", 5, 15, 10) / 100
        with col2:
            delai_livraison = st.slider("Délai de livraison (jours ouvrés)", 1, 5, 3)
        
        st.subheader("Analyse technique")
        window_court = st.slider("Fenêtre de la moyenne mobile courte", 5, 50, 20)
        window_long = st.slider("Fenêtre de la moyenne mobile longue", 20, 200, 50)
        
        data['MM_Court'] = data['Prix'].rolling(window=window_court).mean()
        data['MM_Long'] = data['Prix'].rolling(window=window_long).mean()
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
        ax2.plot(data.index, data['MM_Court'], label=f'MM {window_court} jours', linewidth=1.5)
        ax2.plot(data.index, data['MM_Long'], label=f'MM {window_long} jours', linewidth=1.5)
        ax2.set_title('Analyse technique - Moyennes Mobiles')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prix (FCFA)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        
        D1 = dividende_annuel * (1 + taux_croissance)
        val_intrinseque = D1 / (rendement_exige - taux_croissance)
        st.markdown(f"### Valeur intrinsèque calculée: **{val_intrinseque:.2f} FCFA**")
        
        data['val_intrinseque'] = val_intrinseque
        data['prix_achat'] = (1 - marge_achat) * val_intrinseque
        data['prix_vente'] = (1 + marge_vente) * val_intrinseque
        
        data['signal_technique'] = 0
        data.loc[data['MM_Court'] > data['MM_Long'], 'signal_technique'] = 1
        data.loc[data['MM_Court'] < data['MM_Long'], 'signal_technique'] = -1
        
        data['achat'] = (data['Prix'] < data['prix_achat']) & (data['signal_technique'] == 1)
        data['vente'] = (data['Prix'] > data['prix_vente']) | (data['signal_technique'] == -1)
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
        ax3.axhline(y=val_intrinseque, color='g', linestyle='-', alpha=0.5, label='Valeur intrinsèque')
        ax3.axhline(y=data['prix_achat'][0], color='g', linestyle='--', alpha=0.5, label='Prix d\'achat')
        ax3.axhline(y=data['prix_vente'][0], color='r', linestyle='--', alpha=0.5, label='Prix de vente')
        
        achats = data[data['achat'] == True]
        ventes = data[data['vente'] == True]
        
        if not achats.empty:
            ax3.scatter(achats.index, achats['Prix'], color='g', s=50, marker='^', label='Signal d\'achat')
        
        if not ventes.empty:
            ax3.scatter(ventes.index, ventes['Prix'], color='r', s=50, marker='v', label='Signal de vente')
        
        ax3.set_title('Signaux d\'achat et de vente')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Prix (FCFA)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.subheader("Backtest de la stratégie")
        capital_initial = st.number_input("Capital initial (FCFA)", 100000, 10000000, 1000000, step=100000)
        frais_transaction = st.slider("Frais de transaction (%)", 0.0, 2.0, 0.5) / 100
        
        def run_backtest(data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison):
            portfolio = pd.DataFrame(index=data.index)
            portfolio['prix'] = data['Prix']
            portfolio['actions'] = 0
            portfolio['actions_en_attente'] = 0
            portfolio['date_livraison'] = None
            portfolio['cash'] = capital_initial
            portfolio['cash_reserve'] = 0
            portfolio['valeur_actions'] = 0
            portfolio['valeur_totale'] = capital_initial
            portfolio['rendement'] = 0
            
            prix_achats = []
            achats_dates = []
            ventes_dates = []
            
            for i in range(1, len(data)):
                jour = data.index[i]
                jour_prec = data.index[i-1]
                prix = data['Prix'].iloc[i]
                prix_prec = data['Prix'].iloc[i-1]
                
                variation = (prix - prix_prec) / prix_prec
                if abs(variation) > plafond_variation:
                    prix = prix_prec * (1 + plafond_variation) if variation > 0 else prix_prec * (1 - plafond_variation)
                
                actions = portfolio.loc[jour_prec, 'actions']
                actions_en_attente = portfolio.loc[jour_prec, 'actions_en_attente']
                date_livraison = portfolio.loc[jour_prec, 'date_livraison']
                cash = portfolio.loc[jour_prec, 'cash']
                cash_reserve = portfolio.loc[jour_prec, 'cash_reserve']
                
                if date_livraison is not None and jour >= date_livraison:
                    if actions_en_attente > 0:
                        actions += actions_en_attente
                        st.write(f"Livraison de {actions_en_attente} actions achetées")
                    elif actions_en_attente < 0:
                        cash += cash_reserve
                        cash_reserve = 0
                        st.write(f"Livraison de {-actions_en_attente} actions vendues")
                    actions_en_attente = 0
                    date_livraison = None
                
                if actions > 0:
                    prix_achat_moyen = sum(prix_achats) / len(prix_achats) if prix_achats else 0
                    
                    if prix < (1 - stop_loss) * prix_achat_moyen and actions_en_attente == 0:
                        vente_montant = actions * prix * (1 - frais_transaction)
                        cash_reserve += vente_montant
                        ventes_dates.append(jour)
                        prix_ventes.append(prix)
                        date_livraison = jour + timedelta(days=delai_livraison)
                        actions_en_attente = -actions
                        st.write(f"Stop Loss: Vente de {actions} actions à {prix:.2f}")
                    
                    elif prix > (1 + take_profit) * prix_achat_moyen and actions_en_attente == 0:
                        vente_montant = actions * prix * (1 - frais_transaction)
                        cash_reserve += vente_montant
                        ventes_dates.append(jour)
                        prix_ventes.append(prix)
                        date_livraison = jour + timedelta(days=delai_livraison)
                        actions_en_attente = -actions
                        st.write(f"Take Profit: Vente de {actions} actions à {prix:.2f}")
                
                if data['achat'].iloc[i] and cash > 0 and actions == 0 and actions_en_attente == 0:
                    max_actions = int(cash / (prix * (1 + frais_transaction)))
                    if max_actions > 0:
                        cout_achat = max_actions * prix * (1 + frais_transaction)
                        cash -= cout_achat
                        cash_reserve -= cout_achat
                        actions_en_attente = max_actions
                        date_livraison = jour + timedelta(days=delai_livraison)
                        achats_dates.append(jour)
                        prix_achats.append(prix)
                        st.write(f"Achat de {max_actions} actions à {prix:.2f}")
                
                elif data['vente'].iloc[i] and actions > 0 and actions_en_attente == 0:
                    vente_montant = actions * prix * (1 - frais_transaction)
                    cash_reserve += vente_montant
                    ventes_dates.append(jour)
                    prix_ventes.append(prix)
                    date_livraison = jour + timedelta(days=delai_livraison)
                    actions_en_attente = -actions
                    st.write(f"Vente stratégique de {actions} actions à {prix:.2f}")
                
                portfolio.loc[jour, 'actions'] = actions
                portfolio.loc[jour, 'actions_en_attente'] = actions_en_attente
                portfolio.loc[jour, 'date_livraison'] = date_livraison
                portfolio.loc[jour, 'cash'] = cash
                portfolio.loc[jour, 'cash_reserve'] = cash_reserve
                portfolio.loc[jour, 'valeur_actions'] = actions * prix
                portfolio.loc[jour, 'valeur_totale'] = cash + cash_reserve + (actions * prix)
                
                if i > 0:
                    rendement_jour = (portfolio.loc[jour, 'valeur_totale'] / portfolio.loc[jour_prec, 'valeur_totale']) - 1
                    portfolio.loc[jour, 'rendement'] = rendement_jour
            
            portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1
            return portfolio, achats_dates, ventes_dates
        
        portfolio, achats_dates, ventes_dates = run_backtest(data, capital_initial, frais_transaction, stop_loss, take_profit, plafond_variation, delai_livraison)
        
        st.subheader("Résultats du backtest")
        rendement_total = (portfolio['valeur_totale'].iloc[-1] / capital_initial - 1) * 100
        rendement_annualise = ((1 + rendement_total/100) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement total", f"{rendement_total:.2f}%")
        col2.metric("Rendement annualisé", f"{rendement_annualise:.2f}%")
        col3.metric("Valeur finale", f"{portfolio['valeur_totale'].iloc[-1]:,.2f} FCFA")
        
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(portfolio.index, portfolio['valeur_totale'], linewidth=2, label='Portefeuille')
        ax4.plot(portfolio.index, [capital_initial] * len(portfolio), '--', label='Capital initial')
        
        for date in achats_dates:
            ax4.axvline(x=date, color='g', linestyle='--', alpha=0.3)
        for date in ventes_dates:
            ax4.axvline(x=date, color='r', linestyle='--', alpha=0.3)
        
        ax4.set_title('Évolution du portefeuille')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Valeur (FCFA)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.subheader("Composition finale")
        col1, col2, col3 = st.columns(3)
        col1.metric("Actions", portfolio['actions'].iloc[-1])
        col2.metric("En attente", portfolio['actions_en_attente'].iloc[-1])
        col3.metric("Liquidités", f"{portfolio['cash'].iloc[-1]:,.2f} FCFA")
        
        st.subheader("Journal des transactions")
        transactions = []
        for date in achats_dates:
            transactions.append({
                'Date': date,
                'Type': 'Achat',
                'Prix': data.loc[date, 'Prix'],
                'Livraison': date + timedelta(days=delai_livraison)
            })
        
        for date in ventes_dates:
            transactions.append({
                'Date': date,
                'Type': 'Vente', 
                'Prix': data.loc[date, 'Prix'],
                'Livraison': date + timedelta(days=delai_livraison)
            })
        
        if transactions:
            transactions_df = pd.DataFrame(transactions).sort_values('Date')
            st.dataframe(transactions_df)
        else:
            st.info("Aucune transaction")
        
        st.subheader("Métriques avancées")
        data['rendement_marche'] = data['Prix'].pct_change()
        volatilite_strat = portfolio['rendement'].std() * (252 ** 0.5) * 100
        volatilite_marche = data['rendement_marche'].std() * (252 ** 0.5) * 100
        taux_sans_risque = 0.03
        sharpe_ratio = (rendement_annualise/100 - taux_sans_risque) / (volatilite_strat/100) if volatilite_strat != 0 else 0
        
        portfolio['peak'] = portfolio['valeur_totale'].cummax()
        portfolio['drawdown'] = (portfolio['valeur_totale'] - portfolio['peak']) / portfolio['peak'] * 100
        max_drawdown = portfolio['drawdown'].min()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Volatilité", f"{volatilite_strat:.2f}%")
        col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
        col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
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
        
        fig6, ax6 = plt.subplots(figsize=(12, 4))
        ax6.hist(portfolio['rendement'] * 100, bins=50, alpha=0.7)
        ax6.set_title('Distribution des rendements')
        ax6.set_xlabel('Rendement (%)')
        ax6.set_ylabel('Fréquence')
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6)
        
        st.subheader("Comparaison avec Buy & Hold")
        prix_initial = data['Prix'].iloc[0]
        prix_final = data['Prix'].iloc[-1]
        rendement_bh = (prix_final / prix_initial - 1) * 100
        rendement_bh_annualise = ((1 + rendement_bh/100) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rendement BH", f"{rendement_bh:.2f}%")
        col2.metric("Rend. annualisé BH", f"{rendement_bh_annualise:.2f}%")
        col3.metric("Surperformance", f"{rendement_total - rendement_bh:.2f}%")
        
        fig7, ax7 = plt.subplots(figsize=(12, 6))
        perf_strat = (1 + portfolio['rendement_cumule'])
        perf_bh = data['Prix'] / data['Prix'].iloc[0]
        
        ax7.plot(data.index, perf_strat, label='Stratégie', linewidth=2)
        ax7.plot(data.index, perf_bh, label='Buy & Hold', linewidth=2, linestyle='--')
        ax7.set_title('Comparaison des performances')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Performance (base 1)')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax7.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig7)
        
        st.markdown(get_csv_download_link(portfolio), unsafe_allow_html=True)
