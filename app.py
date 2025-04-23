import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import base64

# Configuration de la page
st.set_page_config(
    page_title="BRVM Quant Backtest Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et introduction
st.title("📈 BRVM Quant Backtest Pro")
st.markdown("""
Application professionnelle de backtesting pour la Bourse Régionale des Valeurs Mobilières (BRVM)
""")

# Fonction de traitement des données
def process_data(file):
    try:
        # Détection automatique du format
        sample = file.read(1024)
        file.seek(0)
        separator = '\t' if b'\t' in sample else ';' if b';' in sample else ','
        
        df = pd.read_csv(file, sep=separator)
        if df.columns[0].isdigit():
            file.seek(0)
            df = pd.read_csv(file, sep=separator, index_col=0)

        # Standardisation des colonnes
        cols = {col.lower(): col for col in df.columns}
        col_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        df_standard = pd.DataFrame()
        for std_col, alt_names in col_map.items():
            found = False
            for alt in [std_col] + [alt_names.lower()]:
                if alt in cols:
                    df_standard[std_col] = df[cols[alt]]
                    found = True
                    break
            if not found:
                st.error(f"Colonne {std_col} introuvable")
                return None

        # Nettoyage des données
        df_standard['Date'] = pd.to_datetime(df_standard['Date'])
        df_standard = df_standard.sort_values('Date').set_index('Date')
        df_standard = df_standard.apply(pd.to_numeric, errors='coerce')
        df_standard['Returns'] = df_standard['Close'].pct_change()
        
        return df_standard.dropna()

    except Exception as e:
        st.error(f"Erreur de traitement: {str(e)}")
        return None

# Fonction de backtest améliorée
def run_backtest(data, capital, params):
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Close'] = data['Close']
    
    # Initialisation
    portfolio['Shares'] = 0
    portfolio['Pending'] = 0
    portfolio['Settlement'] = None
    portfolio['Cash'] = capital
    portfolio['Reserved'] = 0
    portfolio['Total'] = capital
    
    buy_prices = []
    transactions = []
    
    for i in range(1, len(data)):
        current = data.index[i]
        prev = data.index[i-1]
        
        # Paramètres courants
        price = data['Close'].iloc[i]
        shares = portfolio.at[prev, 'Shares']
        pending = portfolio.at[prev, 'Pending']
        settle = portfolio.at[prev, 'Settlement']
        cash = portfolio.at[prev, 'Cash']
        reserved = portfolio.at[prev, 'Reserved']
        
        # Gestion des règlements
        if settle and current >= settle:
            if pending > 0:  # Livraison achat
                shares += pending
                transactions.append(('Settlement', current, 'Buy', pending, price))
            else:  # Livraison vente
                cash += reserved
                reserved = 0
                transactions.append(('Settlement', current, 'Sell', -pending, price))
            pending = 0
            settle = None
        
        # Règles de trading unifiées
        if shares > 0:
            avg_price = np.mean(buy_prices) if buy_prices else 0
            
            # Conditions de sortie combinées
            exit_cond = (
                (price <= (1 - params['stop_loss']) * avg_price) or  # Stop Loss
                (price >= (1 + params['take_profit']) * avg_price) or  # Take Profit
                (data['Signal'].iloc[i] == -1)  # Signal technique
            )
            
            if exit_cond and pending == 0:
                sale_amount = shares * price * (1 - params['fees'])
                reserved += sale_amount
                settle = current + timedelta(days=params['settle_days'])
                pending = -shares
                
                reason = ("Stop Loss" if price <= (1 - params['stop_loss']) * avg_price else
                         "Take Profit" if price >= (1 + params['take_profit']) * avg_price else
                         "Technical Signal")
                
                transactions.append(('Exit', current, reason, shares, price))
                buy_prices = []  # Reset après vente
        
        # Condition d'entrée
        entry_cond = 
            (data['Signal'].iloc[i] == 1) and  # Signal technique
            (cash > 0) and 
            (shares == 0) and 
            (pending == 0)
        
        if entry_cond:
            max_shares = int(cash / (price * (1 + params['fees'])))
            if max_shares > 0:
                cost = max_shares * price * (1 + params['fees'])
                cash -= cost
                reserved -= cost
                pending = max_shares
                settle = current + timedelta(days=params['settle_days'])
                buy_prices.append(price)
                transactions.append(('Entry', current, 'Buy', max_shares, price))
        
        # Mise à jour du portefeuille
        portfolio.at[current, 'Shares'] = shares
        portfolio.at[current, 'Pending'] = pending
        portfolio.at[current, 'Settlement'] = settle
        portfolio.at[current, 'Cash'] = cash
        portfolio.at[current, 'Reserved'] = reserved
        portfolio.at[current, 'Total'] = cash + reserved + (shares * price)
    
    # Calcul des performances
    portfolio['Returns'] = portfolio['Total'].pct_change()
    portfolio['Cumulative'] = (1 + portfolio['Returns']).cumprod()
    
    return portfolio, pd.DataFrame(transactions, columns=['Type', 'Date', 'Reason', 'Amount', 'Price'])

# Interface utilisateur
uploaded_file = st.file_uploader("Importer les données historiques", type=['csv'])
if uploaded_file:
    data = process_data(uploaded_file)
    if data is not None:
        st.success(f"Données chargées: {len(data)} points du {data.index[0].date()} au {data.index[-1].date()}")
        
        # Paramètres de stratégie
        st.sidebar.header("Paramètres de stratégie")
        capital = st.sidebar.number_input("Capital initial (FCFA)", 100000, 100000000, 1000000)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 10) / 100
            take_profit = st.slider("Take Profit (%)", 5, 50, 20) / 100
            fees = st.slider("Frais de transaction (%)", 0.1, 2.0, 0.5) / 100
        with col2:
            short_window = st.slider("MM courte (jours)", 5, 50, 20)
            long_window = st.slider("MM longue (jours)", 50, 200, 100)
            settle_days = st.slider("Délai règlement (jours)", 1, 5, 3)
        
        # Calcul des signaux
        data['SMA_Short'] = data['Close'].rolling(short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(long_window).mean()
        data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, -1)
        
        # Exécution du backtest
        params = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'fees': fees,
            'settle_days': settle_days
        }
        
        portfolio, transactions = run_backtest(data, capital, params)
        
        # Affichage des résultats
        st.header("Résultats du backtesting")
        
        # Métriques clés
        total_return = (portfolio['Total'].iloc[-1] / capital - 1) * 100
        annual_return = ((1 + total_return/100) ** (365/(data.index[-1] - data.index[0]).days) - 1) * 100
        max_drawdown = (portfolio['Total'] / portfolio['Total'].cummax() - 1).min() * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Valeur finale", f"{portfolio['Total'].iloc[-1]:,.0f} FCFA")
        col2.metric("Rendement total", f"{total_return:.1f}%")
        col3.metric("Rendement annualisé", f"{annual_return:.1f}%")
        
        # Graphique de performance
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio.index, portfolio['Total'], label='Stratégie', linewidth=2)
        ax.plot(portfolio.index, [capital] * len(portfolio), '--', label='Capital initial')
        ax.set_title("Performance du portefeuille")
        ax.set_xlabel("Date")
        ax.set_ylabel("Valeur (FCFA)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Transactions
        st.subheader("Journal des transactions")
        st.dataframe(transactions.sort_values('Date', ascending=False))
        
        # Téléchargement des résultats
        csv = portfolio.to_csv().encode()
        st.download_button(
            label="Télécharger les résultats",
            data=csv,
            file_name='brvm_backtest_results.csv',
            mime='text/csv'
        )
