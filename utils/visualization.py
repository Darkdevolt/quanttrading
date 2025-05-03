import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64

def display_data_preview(data):
    """Affiche un aperçu des données traitées."""
    with st.expander("Aperçu des Données Traitées (100 dernières lignes)", expanded=False):
        st.dataframe(data.tail(100).style.format({
            'Ouverture': '{:.2f}',
            'Plus_Haut': '{:.2f}',
            'Plus_Bas': '{:.2f}',
            'Prix': '{:.2f}',
            'Volume': '{:.0f}',
            'Variation': '{:.2f}',
            'Variation_%': '{:.2f}%'
        }))
        
        st.markdown(get_csv_download_link(data.tail(100),
                   filename=f"apercu_data_{st.session_state.stock_name}.csv",
                   link_text="Télécharger l'aperçu (CSV)"), unsafe_allow_html=True)
        st.markdown(get_csv_download_link(data),
                   filename=f"data_completes_{st.session_state.stock_name}.csv",
                   link_text="Télécharger les données complètes (CSV)"), unsafe_allow_html=True)

def display_price_chart(data):
    """Affiche le graphique du cours historique."""
    st.subheader(f"Graphique du Cours Historique de {st.session_state.stock_name}")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Prix'], linewidth=1.5, label='Prix de Clôture', color='royalblue')
        
        ax.set_title(f"Évolution du cours de {st.session_state.stock_name}", fontsize=14)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Prix (FCFA)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Erreur lors de la génération du graphique du cours : {e}")
        st.error(traceback.format_exc())

def get_csv_download_link(df, filename="rapport.csv", link_text="Télécharger les données (CSV)"):
    """Génère un lien de téléchargement HTML pour un DataFrame."""
    if df is None or df.empty:
        return ""
    
    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=True, date_format='%Y-%m-%d %H:%M:%S', sep=';', decimal=',')
        csv_string = buffer.getvalue()
        buffer.close()
        
        b64 = base64.b64encode(csv_string.encode('utf-8')).decode('utf-8')
        
        button_style = """
            display: inline-block;
            padding: 0.5em 1em;
            text-decoration: none;
            background-color: #4CAF50;
            color: white;
            border-radius: 0.25em;
            border: none;
            cursor: pointer;
            font-size: 0.9rem;
            margin-top: 0.5em;
            transition: background-color 0.3s ease;
        """
        
        button_hover_style = """
            <style>
                .download-button:hover {
                    background-color: #45a049 !important;
                    color: white !important;
                    text-decoration: none !important;
                }
            </style>
        """
        
        st.markdown(f"{button_hover_style}", unsafe_allow_html=True)
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="{button_style}">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Erreur lors de la création du lien de téléchargement : {e}")
        st.error(traceback.format_exc())
        return ""
