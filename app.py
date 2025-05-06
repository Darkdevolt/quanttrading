# app.py
import streamlit as st
import pandas as pd
from utils import parse_boc_pdf, classify_titles, calculate_intrinsic_value, gordon_shapiro_value

st.set_page_config(page_title="Analyse BRVM", layout="wide")
st.title("Analyse automatique des titres cotés BRVM")

uploaded_file = st.file_uploader("Uploader un Bulletin Officiel de la Cote (BOC)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extraction des données en cours..."):
        try:
            # Extraction et classification des données
            df = parse_boc_pdf(uploaded_file)
            df = classify_titles(df)
        except Exception as e:
            st.error(f"Erreur lors du parsing du PDF : {e}")
            st.stop()

    if df.empty:
        st.warning("Aucune donnée extraite du PDF.")
    else:
        st.success("Extraction terminée.")

        # Sélection du type de titre
        title_type = st.selectbox(
            "Choisir le type de titre", 
            ["Tous", "Propriété (Actions)", "Créance (Obligations)"]
        )

        df_display = df.copy()
        if title_type != "Tous":
            df_display = df[df["type"] == title_type]

        st.dataframe(df_display)

        # Section valorisation pour les actions
        if title_type == "Propriété (Actions)":
            st.subheader("Valeur intrinsèque des actions")

            # Paramètres ajustables
            market_per = st.number_input(
                "PER du marché ou du secteur (par défaut : 12)", value=12.0)
            required_return = st.slider(
                "Taux de rentabilité exigé (r)", min_value=0.01, max_value=0.30,
                step=0.005, value=0.12)
            growth_rate = st.slider(
                "Taux de croissance (g)", min_value=0.00, max_value=0.20,
                step=0.005, value=0.05)

            # Calcul des valeurs intrinsèques
            df_actions = df_display.copy()
            df_actions["valeur_PER"] = df_actions.apply(
                lambda row: calculate_intrinsic_value(row, market_per), axis=1)
            df_actions["valeur_GS"] = df_actions.apply(
                lambda row: gordon_shapiro_value(row, required_return, growth_rate), axis=1)
            df_actions["% sous-évaluation"] = (
                (df_actions["valeur_GS"] - df_actions["cours"]) / df_actions["valeur_GS"] * 100
            )

            # Filtrer les actions sous-évaluées
            under_valued = df_actions[df_actions["cours"] < df_actions["valeur_GS"]]

            st.write("### Actions potentiellement sous-évaluées selon Gordon-Shapiro")
            st.dataframe(
                under_valued.sort_values(by="% sous-évaluation", ascending=False)
            )
