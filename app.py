import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from datetime import datetime

# Indicateurs techniques (SMA, RSI, MACD)
def calcul_sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def calcul_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def calcul_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line, 'Histogram': hist})

# Backtest simplifié BRVM avec variation max et délai de règlement

def run_backtest(df, capital, commission, stop_loss_pct, take_profit_pct, variation_cap, settlement_days):
    data = df.copy().reset_index(drop=True)
    n = len(data)
    capital_init = capital
    shares = 0
    cash = capital
    portfolio = []
    prev_price = data.loc[0, 'Close']

    position_open_day = -settlement_days - 1  # valeur initiale en dehors du range
    buy_price = 0

    for i, row in data.iterrows():
        price = row['Close']
        # limiter variation journalière
        var = (price - prev_price) / prev_price if prev_price > 0 else 0
        if abs(var) > variation_cap:
            price = prev_price * (1 + np.sign(var) * variation_cap)
        prev_price = price

        signal = 0
        if i > 0 and data.loc[i, 'SMA_20'] > data.loc[i, 'SMA_50'] and data.loc[i - 1, 'SMA_20'] <= data.loc[i - 1, 'SMA_50']:
            signal = 1
        if i > 0 and data.loc[i, 'SMA_20'] < data.loc[i, 'SMA_50'] and data.loc[i - 1, 'SMA_20'] >= data.loc[i - 1, 'SMA_50']:
            signal = -1

        # exécution ordre
        if signal == 1 and cash > price:
            shares = cash // price
            cost = shares * price * (1 + commission)
            cash -= cost
            buy_price = price
            position_open_day = i

        # Ne peut vendre que si 3 jours se sont écoulés
        if signal == -1 and shares > 0 and i - position_open_day >= settlement_days:
            proceeds = shares * price * (1 - commission)
            cash += proceeds
            shares = 0

        total = cash + shares * price
        portfolio.append(total)

    # calcul metrics
    port = pd.Series(portfolio, index=df['Date'])
    returns = port.pct_change().fillna(0)
    cum_ret = (port / capital_init - 1) * 100
    max_dd = (port.cummax() - port).max() / port.cummax().max() * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan

    return port, cum_ret, max_dd, sharpe

# Fonctions d'affichage

def afficher_statistiques(df):
    st.subheader("Statistiques descriptives")
    st.dataframe(df[['Open','High','Low','Close','Volume']].describe())

def afficher_charts(df):
    st.subheader("Cours et Moyennes Mobiles")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close')
    ax.plot(df['Date'], df['SMA_20'], label='SMA20')
    ax.plot(df['Date'], df['SMA_50'], label='SMA50')
    ax.legend(); st.pyplot(fig)

    st.subheader("RSI (14)")
    st.line_chart(df.set_index('Date')['RSI_14'])

    st.subheader("MACD")
    st.line_chart(df.set_index('Date')[['MACD','Signal']])
    st.subheader("Histogramme MACD")
    fig2, ax2 = plt.subplots()
    ax2.bar(df['Date'], df['Histogram']); st.pyplot(fig2)

# Traitement CSV

def process_data(file):
    df = pd.read_csv(file)
    df.columns=df.columns.str.strip().str.lower()
    req=['date','open','high','low','close','volume']
    if any(col not in df.columns for col in req):
        st.error('Colonnes manquantes: '+', '.join([c for c in req if c not in df.columns]))
        return None
    df=df.rename(columns={'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    df['Date']=pd.to_datetime(df['Date'], errors='coerce')
    df=df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df['SMA_20']=calcul_sma(df['Close'],20)
    df['SMA_50']=calcul_sma(df['Close'],50)
    df['RSI_14']=calcul_rsi(df['Close'],14)
    macd=calcul_macd(df['Close'])
    df=pd.concat([df,macd],axis=1)
    return df

# UI principale
st.title('Backtest BRVM - Analyse technique avancée')
file=st.file_uploader('CSV (Date,Open,High,Low,Close,Volume)',type='csv')
if file:
    df=process_data(file)
    if df is not None:
        st.success('Données chargées')
        st.subheader('Aperçu')
        st.dataframe(df.head(10))

        afficher_statistiques(df)
        afficher_charts(df)

        # Paramètres backtest
        st.sidebar.header('Paramètres Backtest')
        capital=st.sidebar.number_input('Capital initial FCFA', min_value=100_000, max_value=int(1e9), value=100_000, step=10_000)
        commission=st.sidebar.slider('Commission %',0.0,1.0,0.5
