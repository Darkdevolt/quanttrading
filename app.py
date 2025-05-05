import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from datetime import datetime

# Fonctions d'indicateurs techniques (SMA, RSI, MACD)
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

# Affichage de statistiques et graphiques

def afficher_statistiques(df):
    st.subheader("Statistiques descriptives")
    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

def afficher_graphique_plotly(df):
    st.subheader("Graphique interactif (Close)")
    fig = px.line(df, x='Date', y='Close')
    st.plotly_chart(fig)

def afficher_graphique_matplotlib(df):
    st.subheader("Graphique statique (Close)")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close')
    ax.set_xlabel('Date'); ax.set_ylabel('Prix de clôture')
    ax.legend(); st.pyplot(fig)

def afficher_indicateurs(df):
    st.subheader("Indicateurs techniques")
    df['SMA_20'] = calcul_sma(df['Close'], 20)
    df['RSI_14'] = calcul_rsi(df['Close'], 14)
    macd_df = calcul_macd(df['Close'])
    df = pd.concat([df.reset_index(drop=True), macd_df.reset_index(drop=True)], axis=1)

    # SMA et Close
    st.line_chart(df.set_index('Date')[['Close', 'SMA_20']])
    # RSI
    st.line_chart(df.set_index('Date')['RSI_14'])
    # MACD + Signal
    st.line_chart(df.set_index('Date')[['MACD', 'Signal']])
    # Histogramme MACD
    st.subheader("Histogramme MACD")
    fig, ax = plt.subplots()
    ax.bar(df['Date'], df['Histogram'])
    st.pyplot(fig)

# Téléchargement de résumé

def telecharger_resume(df):
    st.subheader("Télécharger le résumé (CSV)")
    buffer = BytesIO()
    df.describe().to_csv(buffer)
    buffer.seek(0)
    st.download_button(label="Télécharger", data=buffer, file_name="resume.csv", mime="text/csv")

# Traitement des données CSV

def process_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        # Normalisation des noms de colonnes en minuscules
        df.columns = df.columns.str.strip().str.lower()
        required = ['date','open','high','low','close','volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error("Colonnes manquantes : " + ", ".join(missing))
            return None
        # Renommage en noms standard
        df = df.rename(columns={
            'date':'Date','open':'Open','high':'High',
            'low':'Low','close':'Close','volume':'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
        return None

# Interface principale
st.title("Backtest BRVM - Analyse technique")
fichier = st.file_uploader("Fichier CSV (Date,Open,High,Low,Close,Volume)", type='csv')
if fichier:
    df = process_data(fichier)
    if df is not None:
        st.success("Fichier chargé.")
        st.subheader("Aperçu")
        n = st.slider("Lignes", 5, min(50,len(df)), 10)
        st.dataframe(df.head(n))
        sel = st.multiselect("Colonnes", df.columns.tolist(), default=['Date','Close'])
        st.dataframe(df[sel])
        afficher_statistiques(df)
        afficher_graphique_plotly(df)
        afficher_graphique_matplotlib(df)
        afficher_indicateurs(df)

# Fin du script
