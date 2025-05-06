# app.py
import streamlit as st
import pandas as pd
from utils import parse_boc_pdf, classify_titles, calculate_intrinsic_value, gordon_shapiro_value

st.set_page_config(page_title="Analyse BRVM", layout="wide")
st.title("Analyse automatique des titres cotés BRVM")

# 1. Upload du BOC
uploaded_file = st.file_uploader("Uploader un Bulletin Officiel de la Cote (BOC)", type=["pdf"])

if uploaded_file:
    with st.spinner("Extraction des données en cours..."):
        try:
            df_raw = parse_boc_pdf(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors du parsing du PDF : {e}")
            st.stop()

    if df_raw.empty:
        st.warning("Aucune donnée extraite du PDF.")
        st.stop()

    st.success("Extraction brute terminée.")

    # 2. Choix des colonnes essentielles
    st.subheader("Mapper les colonnes du tableau extrait")
    cols = df_raw.columns.tolist()
    mapping = {}
    required_fields = {
        'symbole': 'Symbole (ex: UNXC, TPBF.O10)',
        'titre': 'Intitulé du titre',
        'cours': 'Cours actuel',
        'per': 'PER (optionnel)',
        'dernier_dividende': 'Dernier dividende (optionnel)',
        'bnpa': 'BNPA (optionnel)'
    }
    for field, label in required_fields.items():
        mapping[field] = st.selectbox(
            label,
            options=[None] + cols,
            index=0,
            help=f"Choisir la colonne correspondant à {label}"
        )

    # Vérification mapping minimal
    if not mapping['symbole'] or not mapping['titre'] or not mapping['cours']:
        st.error("Vous devez mapper au moins les colonnes 'symbole', 'titre' et 'cours'.")
        st.stop()

    # 3. Création du DataFrame structuré
    df = pd.DataFrame()
    df['symbole'] = df_raw[mapping['symbole']]
    df['titre'] = df_raw[mapping['titre']]
    # Conversion du cours en float
    df['cours'] = df_raw[mapping['cours']].astype(str).str.replace(',', '').astype(float)
    # Champs optionnels
    if mapping['per']:
        df['per'] = pd.to_numeric(df_raw[mapping['per']], errors='coerce')
    if mapping['dernier_dividende']:
        df['dernier_dividende'] = pd.to_numeric(df_raw[mapping['dernier_dividende']], errors='coerce')
    if mapping['bnpa']:
        df['bnpa'] = pd.to_numeric(df_raw[mapping['bnpa']], errors='coerce')

    # 4. Classification des titres
    df = classify_titles(df)

    # 5. Affichage et filtrage
    title_type = st.selectbox(
        "Choisir le type de titre",
        ["Tous", "Propriété (Actions)", "Créance (Obligations)"]
    )
    df_display = df.copy()
    if title_type != "Tous":
        df_display = df[df['type'] == title_type]

    st.dataframe(df_display)

    # 6. Valorisation actions
    if title_type == "Propriété (Actions)":
        st.subheader("Calcul de la valeur intrinsèque des actions")

        market_per = st.number_input(
            "PER du marché ou secteur (par défaut : 12)", value=12.0
        )
        required_return = st.slider(
            "Taux de rentabilité exigé (r)", min_value=0.01, max_value=0.30,
            step=0.005, value=0.12
        )
        growth_rate = st.slider(
            "Taux de croissance (g)", min_value=0.00, max_value=0.20,
            step=0.005, value=0.05
        )

        df_actions = df_display.copy()
        df_actions['valeur_PER'] = df_actions.apply(
            lambda x: calculate_intrinsic_value(x, market_per), axis=1
        )
        df_actions['valeur_GS'] = df_actions.apply(
            lambda x: gordon_shapiro_value(x, required_return, growth_rate), axis=1
        )
        df_actions['% sous-évaluation'] = (
            (df_actions['valeur_GS'] - df_actions['cours']) / df_actions['valeur_GS'] * 100
        )

        under_valued = df_actions[df_actions['cours'] < df_actions['valeur_GS']]

        st.write("### Actions potentiellement sous-évaluées selon Gordon-Shapiro")
        st.dataframe(
            under_valued.sort_values(by='% sous-évaluation', ascending=False)
        )
