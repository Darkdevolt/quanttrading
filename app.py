import streamlit as st
import pdfplumber
import pandas as pd
import base64
import tempfile
from io import BytesIO
import datetime

# Configuration initiale
st.set_page_config(layout="wide", page_title="BRVM Trading Expert", page_icon="📈")

# Fonctions de traitement PDF
def parse_equity_data(text):
    equities = []
    current_section = None
    lines = text.split('\n')
    
    for line in lines:
        if 'ACTIONS' in line:
            current_section = 'actions'
        elif 'OBLIGATIONS' in line:
            current_section = 'obligations'
        
        if current_section == 'actions' and 'CB' in line:
            parts = line.split()
            if len(parts) >= 12:
                try:
                    equity = {
                        'Symbole': parts[1],
                        'Titre': ' '.join(parts[2:-10]),
                        'Cours': float(parts[-10].replace(' ', '')),
                        'Var. Jour': float(parts[-9].replace('%', '')),
                        'Var. annuelle': float(parts[-8].replace('%', '')),
                        'Dividende': float(parts[-7]),
                        'PER': float(parts[-6]),
                        'Secteur': parts[0]
                    }
                    equities.append(equity)
                except:
                    continue
    return pd.DataFrame(equities)

def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return parse_equity_data(text)

# Modèles de valuation
def gordon_model(dividend, growth, required_return):
    if required_return <= growth:
        return 0
    return (dividend * (1 + growth/100)) / ((required_return/100 - growth/100))

def shapiro_model(dividend, growth, required_return, sector_per):
    gordon = gordon_model(dividend, growth, required_return)
    return gordon * (sector_per / (sector_per + (required_return/100 - growth/100)))

# Interface Streamlit
def main():
    st.title("📊 BRVM Trading Expert System")
    
    # Upload PDF
    uploaded_file = st.file_uploader("Déposer le bulletin officiel (PDF)", type="pdf")
    
    if uploaded_file:
        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            df = parse_pdf(tmp_file.name)
        
        # Section d'analyse
        st.sidebar.header("Paramètres d'Analyse")
        required_return = st.sidebar.slider("Taux de rendement requis (%)", 5.0, 20.0, 7.92)
        margin_safety = st.sidebar.slider("Marge de sécurité (%)", 0, 30, 15)
        selected_sector = st.sidebar.selectbox("Filtrer par secteur", df['Secteur'].unique())
        
        # Calcul des valuations
        df['Valeur Gordon'] = df.apply(lambda x: gordon_model(x['Dividende'], x['Var. annuelle'], required_return), axis=1)
        df['Valeur Shapiro'] = df.apply(lambda x: shapiro_model(x['Dividende'], x['Var. annuelle'], required_return, x['PER']), axis=1)
        df['Décotage'] = ((df[['Valeur Gordon', 'Valeur Shapiro']].mean(axis=1) - df['Cours']) / df['Cours']) * 100
        
        # Filtrage
        filtered_df = df[df['Secteur'] == selected_sector]
        undervalued = filtered_df[filtered_df['Décotage'] > margin_safety]
        
        # Affichage principal
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("🔍 Sélection de Valeur")
            selected = st.selectbox("Choisir un titre", undervalued['Symbole'])
            stock = undervalued[undervalued['Symbole'] == selected].iloc[0]
            
            st.metric("Cours actuel", f"{stock['Cours']:,} XOF")
            st.metric("Valeur intrinsèque (moyenne)", 
                      f"{(stock['Valeur Gordon'] + stock['Valeur Shapiro']) / 2:,.0f} XOF",
                      delta=f"{stock['Décotage']:.1f}%")
            
            st.download_button(
                label="💾 Sauvegarder l'analyse",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"brvm_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        
        with col2:
            st.subheader("📉 Top Opportunités d'Achat")
            st.dataframe(
                undervalued.sort_values('Décotage', ascending=False)[['Symbole', 'Cours', 'Valeur Gordon', 'Valeur Shapiro', 'Décotage']],
                column_config={
                    "Cours": st.column_config.NumberColumn(format="%,d XOF"),
                    "Valeur Gordon": st.column_config.NumberColumn(format="%,d XOF"),
                    "Valeur Shapiro": st.column_config.NumberColumn(format="%,d XOF"),
                    "Décotage": st.column_config.NumberColumn(format="+%.1f %%")
                },
                height=500
            )
            
            st.subheader("🧮 Comparaison des Modèles")
            st.bar_chart(
                undervalued.set_index('Symbole')[['Valeur Gordon', 'Valeur Shapiro', 'Cours']],
                height=300
            )
    
    else:
        st.info("Veuillez uploader le bulletin officiel BRVM au format PDF")

if __name__ == "__main__":
    main()
