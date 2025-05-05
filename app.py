import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from datetime import datetime

def afficher_statistiques(df):
    st.subheader("Statistiques descriptives")
    stats = df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
    st.dataframe(stats)

def afficher_graphique_plotly(df):
    st.subheader("Graphique interactif avec Plotly")
    fig = px.line(df, x='Date', y='Close', title='Prix de Clôture (Close)')
    st.plotly_chart(fig)

def afficher_graphique_matplotlib(df):
    st.subheader("Graphique statique Matplotlib")
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Close'], label='Close', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de clôture")
    ax.set_title("Historique des prix de clôture")
    ax.legend()
    st.pyplot(fig)

def telecharger_resume(df):
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

def process_data(file):
    try:
        df = pd.read_csv(file)

        colonnes_attendues = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in colonnes_attendues):
            st.error("Colonnes manquantes. Colonnes requises : " + ", ".join(colonnes_attendues))
            return None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date')

        return df

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None

def test_lecture_fichier(n=30):
    erreurs = []
    for i in range(n):
        try:
            data = {
                'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
                'Open': np.random.rand(10) * 100,
                'High': np.random.rand(10) * 100,
                'Low': np.random.rand(10) * 100,
                'Close': np.random.rand(10) * 100,
                'Volume': np.random.randint(100, 1000, size=10)
            }
            df_test = pd.DataFrame(data)
            buffer = BytesIO()
            df_test.to_csv(buffer, index=False)
            buffer.seek(0)

            df_loaded = pd.read_csv(buffer)
            assert all(col in df_loaded.columns for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        except Exception as e:
            erreurs.append(str(e))
    return erreurs

# Interface principale
st.title("Application d'analyse des données boursières")

fichier = st.file_uploader("Téléverser un fichier CSV contenant les données boursières", type="csv")

if fichier:
    df = process_data(fichier)
    if df is not None:
        st.success("Fichier chargé avec succès.")

        st.subheader("Aperçu des données")
        nb_lignes = st.slider("Nombre de lignes à afficher", 5, min(50, len(df)), 10)
        st.dataframe(df.head(nb_lignes))

        colonnes = st.multiselect("Sélectionnez les colonnes à afficher", options=df.columns.tolist(), default=['Date', 'Close'])
        st.dataframe(df[colonnes])

        afficher_statistiques(df)
        afficher_graphique_plotly(df)
        afficher_graphique_matplotlib(df)
        telecharger_resume(df)

# Tests automatiques (résumé non visible pour l'utilisateur final)
erreurs_test = test_lecture_fichier()
if erreurs_test:
    st.warning(f"{len(erreurs_test)} erreurs détectées lors des tests internes.")
else:
    st.info("Tous les tests de lecture de fichier (30/30) ont réussi.")
