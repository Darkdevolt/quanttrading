# app.py
import streamlit as st
import pandas as pd
import plotly.express as px  # <-- Nouvelle importation
from data import loader
from strategies import simple_ma
from backtesting import engine, metrics

st.set_page_config(layout="wide", page_title="BRVM Quant", page_icon="📈")

def initialize_session():
    session_defaults = {
        'uploaded_file': None,
        'df_processed': None,
        'all_columns': [],
        'column_mapping': {k: "" for k in ["Date", "Open", "High", "Low", "Close", "Volume"]},
        'backtest_results': None,
        'show_history_chart': False
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session()

# --- Interface principale ---
st.title("📈 BRVM - Analyse Technique Interactive")
st.markdown("""
Chargez votre fichier CSV historique et explorez les données avec des outils interactifs.
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Upload de fichier
    uploaded_file = st.file_uploader("📤 Téléverser un fichier CSV", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        
        # Détection automatique des colonnes
        try:
            sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
            separator = csv.Sniffer().sniff(sample).delimiter if sample.strip() else ','
            uploaded_file.seek(0)
            df_sample = pd.read_csv(uploaded_file, sep=separator, nrows=1)
            st.session_state.all_columns = df_sample.columns.tolist()
            
            # Mapping automatique amélioré
            auto_mapping = {
                'Date': next((col for col in st.session_state.all_columns if 'date' in col.lower()), ''),
                'Close': next((col for col in st.session_state.all_columns if 'close' in col.lower()), ''),
                'Open': next((col for col in st.session_state.all_columns if 'open' in col.lower()), ''),
                'High': next((col for col in st.session_state.all_columns if 'high' in col.lower()), ''),
                'Low': next((col for col in st.session_state.all_columns if 'low' in col.lower()), '')
            }
            st.session_state.column_mapping.update(auto_mapping)
            st.success("Colonnes détectées avec succès!")
            
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {str(e)}")

    # Mapping manuel des colonnes
    if st.session_state.get('all_columns'):
        st.divider()
        st.subheader("Mapping des colonnes")
        for standard_col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            st.session_state.column_mapping[standard_col] = st.selectbox(
                f"{standard_col} →",
                options=[""] + st.session_state.all_columns,
                index=st.session_state.all_columns.index(st.session_state.column_mapping[standard_col]) 
                if st.session_state.column_mapping[standard_col] in st.session_state.all_columns else 0,
                key=f"map_{standard_col}"
            )

# --- Contenu principal ---
if st.session_state.uploaded_file:
    try:
        # Chargement des données
        df = loader.load_and_process_data(
            st.session_state.uploaded_file,
            st.session_state.column_mapping
        )
        
        if df is not None:
            # --- Graphique interactif ---
            st.header("Visualisation Interactive")
            
            # Préparer les données pour Plotly
            plot_df = df.reset_index().rename(columns={'index':'Date'})
            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
            
            # Créer le graphique avec Plotly
            fig = px.line(plot_df, 
                         x='Date', 
                         y='Close',
                         title='Évolution du Cours Historique',
                         labels={'Close': 'Prix de Clôture (FCFA)'},
                         template='plotly_dark')
            
            # Personnalisation avancée
            fig.update_layout(
                hovermode='x unified',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1A", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                yaxis=dict(
                    tickprefix='FCFA ',
                    gridcolor='rgba(100,100,100,0.2)'
                ),
                height=600
            )
            
            # Ajouter une moyenne mobile rapide
            fig.add_scatter(x=plot_df['Date'], 
                          y=plot_df['Close'].rolling(20).mean(),
                          mode='lines',
                          name='MM20',
                          line=dict(color='#00FF00', width=1))
            
            # Afficher le graphique
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Fonctionnalités de zoom ---
            with st.expander("🔎 Outils d'Analyse"):
                col1, col2 = st.columns(2)
                with col1:
                    date_range = st.date_input("Période Spécifique", 
                                             value=[plot_df['Date'].min(), plot_df['Date'].max()],
                                             min_value=plot_df['Date'].min(),
                                             max_value=plot_df['Date'].max())
                with col2:
                    selected_ma = st.slider("Moyenne Mobile (jours)", 5, 200, 50)
                
                # Appliquer les filtres
                filtered_df = plot_df[
                    (plot_df['Date'] >= pd.to_datetime(date_range[0])) & 
                    (plot_df['Date'] <= pd.to_datetime(date_range[1]))
                ]
                
                # Mettre à jour le graphique filtré
                fig_filtered = px.line(filtered_df, 
                                      x='Date', 
                                      y='Close',
                                      title=f'Zoom: {date_range[0]} à {date_range[1]}',
                                      template='plotly_dark')
                
                fig_filtered.add_scatter(x=filtered_df['Date'], 
                                       y=filtered_df['Close'].rolling(selected_ma).mean(),
                                       mode='lines',
                                       name=f'MM{selected_ma}',
                                       line=dict(color='#FF00FF', width=2))
                
                st.plotly_chart(fig_filtered, use_container_width=True)
            
            # --- Backtest Section ---
            st.header("Backtesting Automatisé")
            if st.button("Exécuter le Backtest"):
                df_strat = simple_ma.apply_strategy(df)
                equity_curve = engine.run_backtest(df_strat)
                st.session_state.backtest_results = metrics.calculate_performance_metrics(equity_curve)
                
                # Afficher les résultats
                st.subheader("📊 Performance du Portefeuille")
                cols = st.columns(4)
                metrics_mapping = {
                    "Capital Initial": "green",
                    "Capital Final": "blue",
                    "Retour Total (%)": "orange",
                    "CAGR (%)": "purple"
                }
                
                for i, (metric, color) in enumerate(metrics_mapping.items()):
                    cols[i%4].metric(
                        label=metric,
                        value=st.session_state.backtest_results.get(metric, 'N/A'),
                        delta_color="off" if i > 1 else "normal"
                    )
                
                # Courbe d'équité
                st.subheader("Courbe de Performance")
                st.line_chart(equity_curve)
            
        else:
            st.error("Erreur lors du traitement des données. Vérifiez le mapping des colonnes.")
            
    except Exception as e:
        st.error(f"Erreur critique : {str(e)}")
        st.stop()

else:
    st.info("Veuillez téléverser un fichier CSV pour commencer l'analyse.")
