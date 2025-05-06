import streamlit as st
import PyPDF2
import pandas as pd

st.title("Analyse du Bulletin Officiel de la Côte (BRVM)")

# Fonction pour extraire le texte du PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Téléchargement du fichier PDF
uploaded_file = st.file_uploader("Téléchargez le fichier PDF", type="pdf")

if uploaded_file:
    st.success("Fichier PDF téléchargé avec succès !")
    
    # Extraire le texte
    text = extract_text_from_pdf(uploaded_file)
    
    # Afficher les données pertinentes (exemple : indices)
    st.header("Indices Principaux")
    if "BRVM COMPOSITE" in text:
        st.write("**BRVM COMPOSITE**: " + text.split("BRVM COMPOSITE")[1].split("\n")[0].strip())
    if "BRVM PRESTIGE" in text:
        st.write("**BRVM PRESTIGE**: " + text.split("BRVM PRESTIGE")[1].split("\n")[0].strip())
    
    # Exemple d'extraction des plus fortes hausses/baisses
    st.header("Plus Fortes Variations")
    if "PLUS FORTES HAUSSES" in text:
        hausses = text.split("PLUS FORTES HAUSSES")[1].split("PLUS FORTES BAISSES")[0]
        st.write(hausses)
    
    if "PLUS FORTES BAISSES" in text:
        baisses = text.split("PLUS FORTES BAISSES")[1].split("\n\n")[0]
        st.write(baisses)

    # Option pour afficher le texte brut
    if st.checkbox("Afficher le texte brut"):
        st.text_area("Contenu du PDF", text, height=300)
