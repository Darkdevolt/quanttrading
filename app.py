import streamlit as st
import pdfplumber
import pandas as pd
import re
from io import BytesIO

# Titre de l'application
st.title("Analyse des Bulletins Officiels de la Côte BRVM")

# Fonction pour extraire les données du PDF
def extract_data_from_pdf(uploaded_file):
    data = {
        "Actions": [],
        "Obligations": [],
        "Indices": [],
        "OPCVM": []
    }
    
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            
            # Extraction des actions (simplifié)
            if "ACTIONS" in text:
                action_lines = re.findall(r'([A-Z]{2,5})\s+(.+?)\s+(\d[\d,.]*)\s+([-+]?\d*\.\d+%)', text)
                for line in action_lines:
                    data["Actions"].append({
                        "Symbole": line[0],
                        "Titre": line[1],
                        "Cours": line[2],
                        "Variation": line[3]
                    })
            
            # Extraction des obligations (simplifié)
            if "OBLIGATIONS D'ETAT" in text:
                obligation_lines = re.findall(r'([A-Z]{2,5}\.\w+)\s+(.+?)\s+(\d[\d,.]*)\s+(\d[\d,.]*)', text)
                for line in obligation_lines:
                    data["Obligations"].append({
                        "Code": line[0],
                        "Description": line[1],
                        "Valeur Nominale": line[2],
                        "Cours": line[3]
                    })
    
    return data

# Téléchargement du fichier
uploaded_file = st.file_uploader("Téléchargez le bulletin BRVM (PDF)", type="pdf")

if uploaded_file is not None:
    # Extraction des données
    data = extract_data_from_pdf(BytesIO(uploaded_file.read()))
    
    # Sélection du type de données à afficher
    data_type = st.selectbox(
        "Sélectionnez le type de données à afficher",
        ["Actions", "Obligations", "Indices", "OPCVM"]
    )
    
    # Affichage des données
    if data[data_type]:
        df = pd.DataFrame(data[data_type])
        st.dataframe(df)
        
        # Options d'analyse
        st.subheader("Options d'analyse")
        
        if data_type == "Actions":
            selected_stock = st.selectbox(
                "Sélectionnez une action pour analyse détaillée",
                [a["Symbole"] for a in data["Actions"]]
            )
            
            # Afficher les détails de l'action sélectionnée
            stock_details = next((a for a in data["Actions"] if a["Symbole"] == selected_stock), None)
            if stock_details:
                st.write(f"**Détails de {selected_stock}**")
                st.json(stock_details)
                
        elif data_type == "Obligations":
            selected_bond = st.selectbox(
                "Sélectionnez une obligation pour analyse détaillée",
                [o["Code"] for o in data["Obligations"]]
            )
            
            # Afficher les détails de l'obligation sélectionnée
            bond_details = next((o for o in data["Obligations"] if o["Code"] == selected_bond), None)
            if bond_details:
                st.write(f"**Détails de {selected_bond}**")
                st.json(bond_details)
    else:
        st.warning(f"Aucune donnée {data_type} trouvée dans le document.")
