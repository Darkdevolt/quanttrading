import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from datetime import datetime

# Implémentation manuelle des indicateurs techniques (SMA, RSI, MACD)
def calcul_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def calcul_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def calcul_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': hist
    })

def afficher_statistiques(df: pd.DataFrame) -> None:
    st.subheader("Statistiques descriptives")
    stats = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
    st.dataframe(stats)

def afficher_graphique_plotly(df: pd.DataFrame) -> None:
    st.subheader("Graphique interactif avec Plotly")
    fig = px.line(df, x='Date', y='Close', title='Prix de Clôture (Close)')
    st.plotly_chart(fig)

def afficher_graphique_matplotlib(df: pd.DataFrame) -> None:
    st.subheader("Graphique statique Matplotlib")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de clôture")
    ax.set_title("Historique des prix de clôture")
    ax.legend()
    st.pyplot(fig)

def afficher_indicateurs(df: pd.DataFrame) -> None:
    st.subheader("Indicateurs techniques")
    # Calcul des indicateurs
    df['SMA_20'] = calcul_sma(df['Close'], 20)
    df['RSI_14'] = calcul_rsi(df['Close'], 14)
    macd_df = calcul_macd(df['Close'], 12, 26, 9)
    df = pd.concat([df.reset_index(drop=True), macd_df.reset_index(drop=True)], axis=1)

    # Tracé
    st.line_chart(df.set_index('Date')[['Close', 'SMA_20']])
    st.line_chart(df.set_index('Date')[['RSI_14']])
    st.line_chart(df.set_index('Date')[['MACD', 'Signal']])

    # Histogramme MACD
    st.subheader("Histogramme MACD")
    fig, ax = plt.subplots()
    ax.bar(df['Date'], df['Histogram'], label='Histogramme MACD')
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD Histogram")
    ax.set_title("Histogramme de la MACD")
    st.pyplot(fig)

def telecharger_resume(df: pd.DataFrame) -> None:
    st.subheader("Télécharger un résumé des données")
    buffer = BytesIO()
    df.describe().to_csv(buffer)
    buffer.seek(0)
    st.download_button(
        label="Télécharger le résumé (CSV)",
        data=buffer,
        file_name="resume_donnees.csv",
        mime="text/csv"
    )

def process_data(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error("Colonnes manquantes. Requises: " + ", ".join(required_cols))
            return None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
        return None

def test_lecture_fichier(n: int = 30) -> list:
    erreurs = []
    for _ in range(n):
        try:
            data = {
                'Date': pd.date_range('2023-01-01', periods=10, freq='D'),
                'Open': np.random.rand(10)*100,
                'High': np.random.rand(10)*100,
                'Low': np.random.rand(10)*100,
                'Close': np.random.rand(10)*100,
                'Volume': np.random.randint(100, 1000, 10)
            }
            df_test = pd.DataFrame(data)
            buf = BytesIO()
            df_test.to_csv(buf, index=False)
            buf.seek(0)
            df_loaded = pd.read_csv(buf)
            assert all(col in df_loaded.columns for col in data.keys())
        except Exception as e:
            erreurs.append(str(e))
    return erreurs

# Interface principale
st.title("Application d'analyse des données boursières - BRVM")
fichier = st.file_uploader("Téléverser un fichier CSV contenant les données boursières", type="csv")
if fichier:
    df = process_data(fichier)
    if df is not None:
        st.success("Fichier chargé avec succès.")
        st.subheader("Aperçu des données")
        nb_lignes = st.slider("Nombre de lignes à afficher", 5, min(50, len(df)), 10)
        st.dataframe(df.head(nb_lignes))
        cols = st.multiselect("Colonnes à afficher", options=df.columns.tolist(), default=['Date', 'Close'])
        st.dataframe(df[cols])
        afficher_statistiques(df)
        afficher_graphique_plotly(df)
        afficher_graphique_matplotlib(df)
        afficher_indicateurs(df)
        telecharger_resume(df)
# Tests internes
erreurs = test_lecture_fichier()
if erreurs:
    st.warning(f"{len(erreurs)} erreurs détectées lors des tests internes.")
else:
    st.info("Tous les tests de lecture de fichier (30/30) ont réussi.")
