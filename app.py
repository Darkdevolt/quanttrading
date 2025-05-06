import streamlit as st
import pdfplumber
import pandas as pd
import tempfile
import datetime

# Configuration initiale
st.set_page_config(layout="wide", page_title="BRVM Trading Expert", page_icon="📈")

def parse_pdf(file):
    """Fonction améliorée pour extraire les données du PDF BRVM"""
    data = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            for line in text.split('\n'):
                if len(line.split()) > 8 and line.split()[0] in ['CB', 'CD', 'ENE', 'FIN', 'TEL', 'IND', 'SPU']:
                    parts = line.split()
                    try:
                        record = {
                            'Type': parts[0],
                            'Symbole': parts[1],
                            'Titre': ' '.join(parts[2:-10]),
                            'Cours': float(parts[-10].replace(' ', '').replace(',', '')),
                            'Var_Jour': float(parts[-9].replace('%', '').replace(',', '')),
                            'Var_Annuelle': float(parts[-8].replace('%', '').replace(',', '')),
                            'Dividende': float(parts[-7].replace(',', '')),
                            'PER': float(parts[-6].replace(',', '')) if parts[-6] != '-' else 0,
                            'Volume': int(parts[-5].replace(',', ''))
                        }
                        data.append(record)
                    except (ValueError, IndexError):
                        continue
    
    # Mapping des types aux secteurs
    sector_mapping = {
        'CB': 'Consommation de Base',
        'CD': 'Consommation Discrétionnaire',
        'ENE': 'Énergie',
        'FIN': 'Financier',
        'TEL': 'Télécoms',
        'IND': 'Industriel',
        'SPU': 'Services Publics'
    }
    
    df = pd.DataFrame(data)
    df['Secteur'] = df['Type'].map(sector_mapping)
    return df

def calculate_valuations(df, required_return):
    """Calcule les différentes valuations"""
    df['Valeur_Gordon'] = df.apply(
        lambda x: (x['Dividende'] * (1 + x['Var_Annuelle']/100)) / 
        (required_return/100 - x['Var_Annuelle']/100) 
        if (required_return/100 > x['Var_Annuelle']/100) else 0,
        axis=1
    )
    
    df['Valeur_Shapiro'] = df.apply(
        lambda x: x['Valeur_Gordon'] * (x['PER'] / (x['PER'] + (required_return/100 - x['Var_Annuelle']/100)) 
        if x['PER'] > 0 else 0,
        axis=1
    )
    
    df['Decotage'] = ((df[['Valeur_Gordon', 'Valeur_Shapiro']].mean(axis=1) - df['Cours']) / df['Cours']) * 100
    return df

def main():
    st.title("📊 BRVM Trading Expert System")
    
    uploaded_file = st.file_uploader("Déposer le bulletin officiel BRVM (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyse du PDF en cours..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                df = parse_pdf(tmp_file.name)
        
        if not df.empty:
            st.sidebar.header("Paramètres d'Analyse")
            required_return = st.sidebar.slider(
                "Taux de rendement requis (%)", 
                min_value=5.0, 
                max_value=20.0, 
                value=7.92,
                step=0.1
            )
            
            margin_safety = st.sidebar.slider(
                "Marge de sécurité (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=1
            )
            
            # Vérification que la colonne Secteur existe
            if 'Secteur' in df.columns:
                selected_sector = st.sidebar.selectbox(
                    "Filtrer par secteur",
                    options=['Tous'] + sorted(df['Secteur'].unique().tolist())
            else:
                selected_sector = 'Tous'
                st.warning("Information sur le secteur non détectée dans le PDF")
            
            df = calculate_valuations(df, required_return)
            
            if selected_sector != 'Tous':
                filtered_df = df[df['Secteur'] == selected_sector]
            else:
                filtered_df = df.copy()
            
            undervalued = filtered_df[filtered_df['Decotage'] > margin_safety].sort_values('Decotage', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔍 Analyse Individuelle")
                if not undervalued.empty:
                    selected = st.selectbox(
                        "Choisir un titre sous-évalué",
                        options=undervalued['Symbole'].tolist()
                    )
                    
                    stock = undervalued[undervalued['Symbole'] == selected].iloc[0]
                    
                    st.metric("Cours actuel", f"{stock['Cours']:,.0f} XOF")
                    st.metric("Valeur Gordon", f"{stock['Valeur_Gordon']:,.0f} XOF", 
                              delta=f"{((stock['Valeur_Gordon']/stock['Cours'])-1)*100:.1f}%")
                    st.metric("Valeur Shapiro", f"{stock['Valeur_Shapiro']:,.0f} XOF",
                              delta=f"{((stock['Valeur_Shapiro']/stock['Cours'])-1)*100:.1f}%")
                    
                    st.download_button(
                        label="💾 Exporter les données",
                        data=df.to_csv(index=False, sep=';').encode('utf-8'),
                        file_name=f"brvm_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )
                else:
                    st.warning("Aucune opportunité trouvée avec les critères actuels")
            
            with col2:
                st.subheader("📊 Top Opportunités")
                if not undervalued.empty:
                    st.dataframe(
                        undervalued[['Symbole', 'Titre', 'Secteur', 'Cours', 
                                    'Valeur_Gordon', 'Valeur_Shapiro', 'Decotage']],
                        column_config={
                            "Cours": st.column_config.NumberColumn(format="%,d XOF"),
                            "Valeur_Gordon": st.column_config.NumberColumn(format="%,d XOF"),
                            "Valeur_Shapiro": st.column_config.NumberColumn(format="%,d XOF"),
                            "Decotage": st.column_config.NumberColumn(
                                label="Décotage (%)",
                                format="+%.1f %%",
                                help="Pourcentage de décotage par rapport à la moyenne des modèles"
                            )
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=500
                    )
                    
                    st.subheader("Comparaison des Valuations")
                    st.bar_chart(
                        undervalued.set_index('Symbole')[['Cours', 'Valeur_Gordon', 'Valeur_Shapiro']],
                        height=400
                    )
                else:
                    st.info("Ajustez les paramètres pour trouver des opportunités")
        else:
            st.error("Aucune donnée valide n'a pu être extraite du PDF")
    else:
        st.info("Veuillez uploader le dernier bulletin officiel BRVM (PDF)")

if __name__ == "__main__":
    main()
