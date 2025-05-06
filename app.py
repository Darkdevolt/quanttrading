import streamlit as st
import PyPDF2
import pandas as pd
import re
from io import BytesIO

# Configuration de la page
st.set_page_config(layout="wide", page_title="Analyse BRVM", page_icon="📈")
st.title("📊 Analyse du Bulletin Officiel de la BRVM")

# Fonctions d'extraction
def extract_text_from_pdf(uploaded_file):
    """Extrait le texte brut du PDF."""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    return "\n".join(page.extract_text() for page in pdf_reader.pages)

def parse_indices(text):
    """Extrait les indices principaux avec leurs variations."""
    pattern = r"(BRVM \w+)\n([\d,]+)\nVariation Jour\n([\d,]+ %)\nVariation annuelle\n([\d,]+ %)"
    return {m[0]: {"Valeur": m[1], "Var. Jour": m[2], "Var. Annuelle": m[3]} 
            for m in re.findall(pattern, text)}

def parse_top_movements(text, movement_type):
    """Extrait les tops hausses/baisses."""
    section_pattern = {
        "hausses": r"PLUS FORTES HAUSSES(.*?)PLUS FORTES BAISSES",
        "baisses": r"PLUS FORTES BAISSES(.*?)(?:\n\n|Base =)"
    }[movement_type]
    
    section = re.search(section_pattern, text, re.DOTALL)
    if not section:
        return []
    
    lines = [line.strip() for line in section.group(1).split("\n") if line.strip()]
    return [{
        "Titre": lines[i],
        "Cours": lines[i+1],
        "Var. Jour": lines[i+2],
        "Var. Annuelle": lines[i+3]
    } for i in range(0, len(lines), 4) if i+3 < len(lines)]

# Interface Streamlit
uploaded_file = st.file_uploader("Téléverser le bulletin BRVM (PDF)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    
    # Section 1: Indices Principaux
    st.header("📊 Indices Clés")
    indices = parse_indices(text)
    if indices:
        cols = st.columns(len(indices))
        for idx, (name, data) in enumerate(indices.items()):
            with cols[idx]:
                st.metric(
                    label=name,
                    value=data["Valeur"],
                    delta=data["Var. Jour"],
                    help=f"Variation annuelle: {data['Var. Annuelle']}"
                )
    else:
        st.warning("Aucun indice trouvé dans le document.")

    # Section 2: Mouvements des Titres
    st.header("📌 Top Mouvements")
    tab_hausses, tab_baisses = st.tabs(["🚀 Hausses", "🔻 Baisses"])
    
    with tab_hausses:
        hausses = parse_top_movements(text, "hausses")
        st.dataframe(
            pd.DataFrame(hausses),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Var. Jour": st.column_config.NumberColumn(format="%+.2f %%"),
                "Var. Annuelle": st.column_config.NumberColumn(format="%+.2f %%")
            }
        )
    
    with tab_baisses:
        baisses = parse_top_movements(text, "baisses")
        st.dataframe(
            pd.DataFrame(baisses),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Var. Jour": st.column_config.NumberColumn(format="%+.2f %%"),
                "Var. Annuelle": st.column_config.NumberColumn(format="%+.2f %%")
            }
        )

    # Section 3: Données Brutes (Optionnel)
    with st.expander("Voir les données brutes"):
        st.text(text[:5000] + "..." if len(text) > 5000 else text)
else:
    st.info("Veuillez téléverser un fichier PDF pour commencer l'analyse.")

# Pied de page
st.caption("Application développée pour l'analyse des bulletins BRVM | Données du 6 mai 2025")
