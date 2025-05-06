# app.py
import streamlit as st
import pandas as pd
from utils import parse_boc_pdf, classify_titles, calculate_intrinsic_value, gordon_shapiro_value

st.set_page_config(page_title="Analyse BRVM", layout="wide")
st.title("Analyse automatique des titres cotés BRVM")

uploaded_file = st.file_uploader("Uploader un Bulletin Officiel de la Cote (BOC)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extraction des données en cours..."):
        df = parse_boc_pdf(uploaded_file)
        df = classify_titles(df)

    st.success("Extraction terminée.")

    title_type = st.selectbox("Choisir le type de titre ", ["Tous", "Propriété (Actions)", "Créance (Obligations)"])

    if title_type != "Tous":
        df = df[df["type"] == title_type]

    st.dataframe(df)

    if title_type == "Propriété (Actions)":
        st.subheader("Valeur intrinsèque des actions")

        market_per = st.number_input("PER du marché ou du secteur (par défaut : 12)", value=12.0)
        required_return = st.slider("Taux de rentabilité exigé r", min_value=0.05, max_value=0.25, step=0.005, value=0.12)
        growth_rate = st.slider("Taux de croissance g", min_value=0.00, max_value=0.20, step=0.005, value=0.05)

        df["valeur_PER"] = df.apply(lambda row: calculate_intrinsic_value(row, market_per), axis=1)
        df["valeur_GS"] = df.apply(lambda row: gordon_shapiro_value(row, required_return, growth_rate), axis=1)
        df["% sous-évaluation"] = (df["valeur_GS"] - df["cours"]) / df["valeur_GS"] * 100

        under_valued = df[df["cours"] < df["valeur_GS"]]

        st.write("Actions potentiellement sous-évaluées selon Gordon-Shapiro")
        st.dataframe(under_valued.sort_values(by="% sous-évaluation", ascending=False))
