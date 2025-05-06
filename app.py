import streamlit as st
import pandas as pd
from io import StringIO
import re

# Configuration de la page
st.set_page_config(layout="wide", page_title="Analyse BRVM", page_icon="📈")
st.title("📊 Analyse du Bulletin Officiel de la BRVM")

# Données brutes
raw_data = """
BRVM COMPOSITE290,62Variation Jour1,00 %
Variation annuelle5,29 %
BRVM PRESTIGE121,60Variation Jour0,59 %
Variation annuelle5,89 %
... (toutes vos données brutes ici) ...
"""

# Fonctions de traitement
def parse_main_indices(data):
    pattern = r"(BRVM \w+)([\d,]+)Variation Jour([\d,]+ %)Variation annuelle([\d,]+ %)"
    matches = re.findall(pattern, data.replace("\n", ""))
    return pd.DataFrame([{
        "Indice": m[0].strip(),
        "Valeur": float(m[1].replace(",", ".")),
        "Var. Jour": m[2].strip(),
        "Var. Annuelle": m[3].strip()
    } for m in matches])

def parse_top_movements(data, movement_type):
    section_pattern = {
        "hausses": r"PLUS FORTES HAUSSES(.*?)PLUS FORTES BAISSES",
        "baisses": r"PLUS FORTES BAISSES(.*?)(?:Base =)"
    }[movement_type]
    
    section = re.search(section_pattern, data, re.DOTALL)
    if not section:
        return pd.DataFrame()
    
    lines = [line.strip() for line in section.group(1).split("\n") if line.strip()]
    return pd.DataFrame([{
        "Titre": lines[i],
        "Cours": lines[i+1],
        "Var. Jour": lines[i+2],
        "Var. Annuelle": lines[i+3]
    } for i in range(0, len(lines), 4) if i+3 < len(lines)])

def parse_sector_indices(data):
    pattern = r"BRVM - (\w+)(\d+)([\d,]+)([\d,-]+ %)([\d,-]+ %)([\d,]+)([\d,]+)([\d,]+)"
    matches = re.findall(pattern, data.replace("\n", ""))
    return pd.DataFrame([{
        "Secteur": m[0],
        "Nb Sociétés": int(m[1]),
        "Valeur": float(m[2].replace(",", ".")),
        "Var. Jour": m[3],
        "Var. Annuelle": m[4],
        "Volume": int(m[5].replace(",", "")),
        "Valeur Transigée": int(m[6].replace(",", "")),
        "PER": float(m[7].replace(",", ".")) if m[7] else None
    } for m in matches])

# Interface Streamlit
tab1, tab2, tab3 = st.tabs(["Indices Principaux", "Mouvements des Titres", "Indices Sectoriels"])

with tab1:
    st.header("📊 Indices Clés")
    indices_df = parse_main_indices(raw_data)
    if not indices_df.empty:
        cols = st.columns(len(indices_df))
        for idx, row in indices_df.iterrows():
            with cols[idx]:
                st.metric(
                    label=row["Indice"],
                    value=row["Valeur"],
                    delta=row["Var. Jour"],
                    help=f"Variation annuelle: {row['Var. Annuelle']}"
                )
    else:
        st.warning("Aucun indice trouvé dans les données.")

with tab2:
    st.header("📌 Top Mouvements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Hausses")
        hausses_df = parse_top_movements(raw_data, "hausses")
        st.dataframe(
            hausses_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Var. Jour": st.column_config.NumberColumn(format="%+.2f %%"),
                "Var. Annuelle": st.column_config.NumberColumn(format="%+.2f %%")
            }
        )
    
    with col2:
        st.subheader("🔻 Baisses")
        baisses_df = parse_top_movements(raw_data, "baisses")
        st.dataframe(
            baisses_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Var. Jour": st.column_config.NumberColumn(format="%+.2f %%"),
                "Var. Annuelle": st.column_config.NumberColumn(format="%+.2f %%")
            }
        )

with tab3:
    st.header("🏭 Indices Sectoriels")
    sector_df = parse_sector_indices(raw_data)
    if not sector_df.empty:
        st.dataframe(
            sector_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Var. Jour": st.column_config.NumberColumn(format="%+.2f %%"),
                "Var. Annuelle": st.column_config.NumberColumn(format="%+.2f %%"),
                "Valeur Transigée": st.column_config.NumberColumn(format="%,d"),
                "PER": st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        # Graphique des variations
        st.subheader("Variations par Secteur")
        selected_metric = st.selectbox("Choisir la métrique", ["Var. Jour", "Var. Annuelle"])
        st.bar_chart(sector_df.set_index("Secteur")[selected_metric].str.replace("%", "").astype(float))
    else:
        st.warning("Aucun indice sectoriel trouvé dans les données.")

# Section des données brutes
with st.expander("📄 Voir les données brutes"):
    st.text(raw_data[:5000] + "..." if len(raw_data) > 5000 else raw_data)

# Pied de page
st.caption("Application développée pour l'analyse des bulletins BRVM | Données du 6 mai 2025")
