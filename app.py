import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")
st.title("Visualisation des données BRVM")

uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    try:
        if file_extension == ".csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension == ".xlsx":
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Type de fichier non supporté. Veuillez importer un fichier CSV ou Excel.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # Conversion de la colonne de dates
    try:
        if not pd.api.types.is_datetime64_any_dtype(data.iloc[:, 0]):
            data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], errors='coerce')
    except Exception as e:
        st.error(f"Erreur lors de la conversion des dates : {e}")
        st.stop()

    # Supprimer les lignes avec dates nulles
    data = data.dropna(subset=[data.columns[0]])
    data = data.set_index(data.columns[0])

    # Sauvegarde dans la session
    st.session_state.data = data

    st.success("Fichier chargé avec succès.")

# Affichage des données et sélection de l’actif
if "data" in st.session_state:
    st.subheader("Aperçu des données")
    st.dataframe(st.session_state.data.head())

    asset_options = st.session_state.data.columns.tolist()
    selected_asset = st.selectbox("Choisir un actif à visualiser", asset_options)

    # Assurer que l'index est de type datetime
    if not isinstance(st.session_state.data.index, pd.DatetimeIndex):
        st.session_state.data.index = pd.to_datetime(st.session_state.data.index, errors='coerce')

    min_date = st.session_state.data.index.min()
    max_date = st.session_state.data.index.max()

    if pd.notnull(min_date) and pd.notnull(max_date):
        st.info(f"Données disponibles du {min_date.date()} au {max_date.date()}.")
    else:
        st.warning("Les dates du fichier ne sont pas valides ou sont manquantes.")

    # Affichage du graphique
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.data.index, y=st.session_state.data[selected_asset], mode='lines', name=selected_asset))
    fig.update_layout(title=f"Évolution de {selected_asset}", xaxis_title="Date", yaxis_title="Valeur", height=600)
    st.plotly_chart(fig, use_container_width=True)
