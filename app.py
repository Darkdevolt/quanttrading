import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz

# Configuration de la page
st.set_page_config(page_title="Analyse BRVM", layout="wide", page_icon="📈")

# Titre de l'application
st.title("📊 Plateforme d'Analyse des Marchés BRVM")

# Fonction pour charger les données (simulée ici - en pratique, vous utiliseriez une connexion à votre base de données cloud)
@st.cache_data(ttl=3600)  # Cache les données pendant 1 heure
def load_data():
    # Ici vous implémenteriez la logique pour charger depuis votre source cloud
    # Exemple avec des données simulées basées sur le PDF
    indices_data = {
        "Indice": ["BRVM COMPOSITE", "BRVM PRESTIGE", "BRVM 30"],
        "Valeur": [290.62, 121.60, 146.21],
        "Var. Jour": [1.00, 0.59, 1.16],
        "Var. Annuelle": [5.29, 5.89, 5.38]
    }
    
    actions_data = {
        "Titre": ["UNIWAX CI", "ECOBANK TRANS. INCORP. TG", "AFRICA GLOBAL LOGISTICS CI"],
        "Symbole": ["UNXC", "ETIT", "SDSC"],
        "Secteur": ["Industriels", "Services Financiers", "Logistique"],
        "Cours": [515, 16, 1450],
        "Var. Jour": [7.29, 6.67, 6.23],
        "Var. Annuelle": [25.61, 0.00, 8.21],
        "Volume": [11317, 91891, 8035],
        "Capitalisation (M FCFA)": [12500, 3200, 45000]
    }
    
    return pd.DataFrame(indices_data), pd.DataFrame(actions_data)

# Chargement des données
indices_df, actions_df = load_data()

# Sidebar avec filtres
with st.sidebar:
    st.header("Filtres")
    secteur = st.multiselect(
        "Secteur d'activité",
        options=actions_df["Secteur"].unique(),
        default=actions_df["Secteur"].unique()
    )
    
    variation_min, variation_max = st.slider(
        "Variation journalière (%)",
        min_value=-10.0,
        max_value=10.0,
        value=(-10.0, 10.0)
    
    date_analyse = st.date_input("Date d'analyse", datetime.now(pytz.timezone('Africa/Abidjan'))

# Section des indices
st.header("📈 Performance des Indices")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("BRVM COMPOSITE", 
              f"{indices_df[indices_df['Indice']=='BRVM COMPOSITE']['Valeur'].values[0]}", 
              f"{indices_df[indices_df['Indice']=='BRVM COMPOSITE']['Var. Jour'].values[0]}%")

with col2:
    st.metric("BRVM PRESTIGE", 
              f"{indices_df[indices_df['Indice']=='BRVM PRESTIGE']['Valeur'].values[0]}", 
              f"{indices_df[indices_df['Indice']=='BRVM PRESTIGE']['Var. Jour'].values[0]}%")

with col3:
    st.metric("BRVM 30", 
              f"{indices_df[indices_df['Indice']=='BRVM 30']['Valeur'].values[0]}", 
              f"{indices_df[indices_df['Indice']=='BRVM 30']['Var. Jour'].values[0]}%")

# Graphique d'évolution des indices
st.plotly_chart(
    px.line(indices_df, x="Indice", y="Valeur", title="Valeur des Indices"),
    use_container_width=True
)

# Section des actions
st.header("📊 Analyse des Actions")

# Appliquer les filtres
filtered_df = actions_df[
    (actions_df["Secteur"].isin(secteur)) & 
    (actions_df["Var. Jour"] >= variation_min) & 
    (actions_df["Var. Jour"] <= variation_max)
]

# Afficher le dataframe filtré
st.dataframe(filtered_df.sort_values("Var. Jour", ascending=False), 
             use_container_width=True,
             column_config={
                 "Var. Jour": st.column_config.ProgressColumn(
                     "Variation Jour",
                     format="%.2f%%",
                     min_value=-10,
                     max_value=10,
                 )
             })

# Visualisations
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        px.bar(filtered_df, 
               x="Symbole", 
               y="Var. Jour", 
               color="Var. Jour",
               title="Variation Journalière par Action"),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        px.scatter(filtered_df, 
                  x="Volume", 
                  y="Var. Jour", 
                  size="Capitalisation (M FCFA)",
                  color="Secteur",
                  hover_name="Titre",
                  title="Volume vs Variation"),
        use_container_width=True
    )

# Section des plus fortes hausses/baisses
st.header("🎢 Performances Extrêmes")

tab1, tab2 = st.tabs(["Plus fortes hausses", "Plus fortes baisses"])

with tab1:
    top_gainers = filtered_df.nlargest(5, "Var. Jour")
    st.dataframe(top_gainers[["Titre", "Symbole", "Var. Jour", "Var. Annuelle"]])

with tab2:
    top_losers = filtered_df.nsmallest(5, "Var. Jour")
    st.dataframe(top_losers[["Titre", "Symbole", "Var. Jour", "Var. Annuelle"]])

# Section des indicateurs techniques
st.header("📊 Indicateurs Techniques")
st.write("""
- **PER moyen du marché**: 11.15
- **Taux de rendement moyen**: 7.92%
- **Taux de rentabilité moyen**: 9.38%
- **Ratio de liquidité moyen**: 7.86
""")

# Pied de page
st.divider()
st.caption(f"Dernière mise à jour: {datetime.now(pytz.timezone('Africa/Abidjan')).strftime('%d/%m/%Y %H:%M')} | Source: BRVM")
