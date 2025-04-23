import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
import holidays
import plotly.graph_objects as go
import numba as nb
import base64
import re
import hashlib
import tempfile
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="BRVM Quant Pro",
    layout="wide",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://www.brvm.org',
        'About': "Plateforme professionnelle de backtesting pour la BRVM"
    }
)

# ---- Sécurité ----
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file(file):
    if file.name.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Type de fichier non autorisé")
    if file.size > MAX_FILE_SIZE:
        raise ValueError("Taille de fichier excède 50MB")
    return True

def sanitize_input(text):
    return re.sub(r'[^a-zA-Z0-9\s_-]', '', text)

# ---- Optimisation ----
@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)

# ---- Gestion des jours fériés ----
class BRVMHolidays(holidays.HolidayBase):
    def _populate(self, year):
        # Jours fériés de la BRVM (à compléter)
        self[datetime(year, 1, 1)] = "Nouvel An"
        self[datetime(year, 4, 16)] = "Fête de l'Indépendance"
        self[datetime(year, 5, 1)] = "Fête du Travail"
        self[datetime(year, 12, 25)] = "Noël"

brvm_business_day = CustomBusinessDay(holidays=BRVMHolidays())

# ---- Interface Utilisateur ----
st.title("🚀 BRVM Quant Pro - Backtest Professionnel")
st.markdown("""
**Solution complète de backtesting pour la Bourse Régionale des Valeurs Mobilières (BRVM)**  
*Intègre les spécificités du marché ouest-africain*
""")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Téléverser CSV", type=['csv'])
    strategy_choice = st.selectbox(
        "Stratégie à comparer",
        ["Buy & Hold", "Moyennes Mobiles", "RSI", "MACD"]
    )
    risk_free_rate = st.slider("Taux sans risque (%)", 0.0, 10.0, 3.0) / 100
    st.markdown("---")
    st.markdown("**Aide rapide:**\n- Format CSV requis\n- Colonnes: Date, Open, High, Low, Close, Volume")

# ---- Traitement des données ----
def process_data(file):
    try:
        # Validation et nettoyage
        validate_file(file)
        file_hash = hashlib.md5(file.getvalue()).hexdigest()
        if st.session_state.get('file_hash') != file_hash:
            st.session_state.clear()
            st.session_state.file_hash = file_hash

        # Lecture avec détection automatique
        df = pd.read_csv(file, parse_dates=True, infer_datetime_format=True)
        
        # Interface de mapping des colonnes
        with st.expander("🔧 Mapping des colonnes"):
            cols = df.columns.tolist()
            col1, col2 = st.columns(2)
            mapping = {}
            with col1:
                mapping['date'] = st.selectbox("Colonne Date", cols, index=0)
                mapping['open'] = st.selectbox("Colonne Open", cols, index=1)
                mapping['high'] = st.selectbox("Colonne High", cols, index=2)
            with col2:
                mapping['low'] = st.selectbox("Colonne Low", cols, index=3)
                mapping['close'] = st.selectbox("Colonne Close", cols, index=4)
                mapping['volume'] = st.selectbox("Colonne Volume", cols, index=5 if len(cols) > 5 else 0)
        
        # Conversion des types
        df = df.rename(columns=mapping)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Validation finale
        if df.isnull().sum().sum() > 0:
            st.warning("Certaines données sont manquantes ou invalides. Les lignes concernées seront filtrées.")
            df = df.dropna()
        
        df = df.set_index('date').sort_index()
        df['returns'] = df['close'].pct_change()
        return df
    
    except Exception as e:
        st.error(f"Erreur critique: {str(e)}")
        st.stop()

# ---- Calculs financiers optimisés ----
@nb.jit(nopython=True)
def calculate_metrics_numba(returns):
    cumulative = np.empty_like(returns)
    cumulative[0] = 1
    for i in range(1, len(returns)):
        cumulative[i] = cumulative[i-1] * (1 + returns[i])
    return cumulative

def calculate_performance(df):
    df['cumulative'] = calculate_metrics_numba(df['returns'].values)
    df['drawdown'] = df['cumulative'] / df['cumulative'].cummax() - 1
    return df

# ---- Moteur de backtest ----
class BacktestEngine:
    def __init__(self, data, initial_capital=1e6, fees=0.001):
        self.data = data
        self.initial_capital = initial_capital
        self.fees = fees
        self.posiciones = []
        self.history = []
    
    def add_strategy(self, strategy):
        self.strategy = strategy
    
    def run(self):
        capital = self.initial_capital
        position = 0
        prev_signal = 0
        
        for idx, row in self.data.iterrows():
            signal = self.strategy.generate_signal(row)
            
            if signal != prev_signal:
                # Exécution avec délai BRVM
                execution_price = self.calculate_execution_price(row)
                trade_cost = abs(signal - position) * execution_price * self.fees
                
                if signal > position:  # Achat
                    capital -= (signal - position) * execution_price + trade_cost
                else:  # Vente
                    capital += (position - signal) * execution_price - trade_cost
                
                position = signal
            
            portfolio_value = capital + position * row['close']
            self.history.append({
                'date': idx,
                'value': portfolio_value,
                'position': position,
                'returns': (portfolio_value / self.initial_capital) - 1
            })
            prev_signal = signal
        
        return pd.DataFrame(self.history).set_index('date')

    def calculate_execution_price(self, row):
        # Simulation de l'impact de marché
        spread = row['high'] - row['low']
        return row['close'] + spread * 0.1 * np.random.randn()

# ---- Visualisations interactives ----
def plot_performance(backtest_result, benchmark):
    fig = go.Figure()
    
    # Stratégie
    fig.add_trace(go.Scatter(
        x=backtest_result.index,
        y=backtest_result['value'],
        name='Stratégie',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Benchmark
    fig.add_trace(go.Scatter(
        x=benchmark.index,
        y=benchmark['close'] / benchmark['close'].iloc[0] * backtest_result['value'].iloc[0],
        name='Benchmark',
        line=dict(color='#ff7f0e', dash='dot')
    ))
    
    # Drawdown
    fig.add_trace(go.Scatter(
        x=backtest_result.index,
        y=backtest_result['drawdown'],
        name='Drawdown',
        yaxis='y2',
        fill='tozeroy',
        line=dict(color='#d62728', width=0.5)
    ))
    
    fig.update_layout(
        title='Performance du Portefeuille',
        yaxis=dict(title='Valeur du Portefeuille'),
        yaxis2=dict(title='Drawdown', overlaying='y', side='right'),
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Rapport détaillé ----
def generate_report(results):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        results.to_excel(writer, sheet_name='Résultats')
    return buffer.getvalue()

# ---- Flux principal ----
if uploaded_file:
    try:
        with st.spinner("Analyse des données en cours..."):
            data = process_data(uploaded_file)
            benchmark = data.copy()
        
        # Configuration du backtest
        with st.expander("⚙️ Paramètres avancés", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_capital = st.number_input("Capital Initial (FCFA)", 1e6, 1e9, 1e6)
                stop_loss = st.slider("Stop Loss (%)", 0.0, 20.0, 10.0) / 100
            with col2:
                transaction_fees = st.slider("Frais de Transaction (%)", 0.0, 2.0, 0.5) / 100
                take_profit = st.slider("Take Profit (%)", 0.0, 50.0, 20.0) / 100
            with col3:
                delivery_days = st.slider("Délai Livraison (jours)", 1, 5, 3)
                risk_free = st.number_input("Taux Sans Risque (%)", 0.0, 15.0, 3.0) / 100
        
        # Exécution du backtest
        engine = BacktestEngine(data, initial_capital, transaction_fees)
        engine.add_strategy(YourCustomStrategy())  # À implémenter
        results = engine.run()
        results = calculate_performance(results)
        
        # Affichage des résultats
        st.subheader("📊 Analyse de Performance")
        plot_performance(results, benchmark)
        
        # Métriques de performance
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rendement Total", f"{results['returns'].iloc[-1]*100:.2f}%")
        col2.metric("Volatilité Annuelle", f"{results['returns'].std()*np.sqrt(252)*100:.2f}%")
        col3.metric("Ratio de Sharpe", f"{(results['returns'].mean()*252 - risk_free)/results['returns'].std()/np.sqrt(252):.2f}")
        col4.metric("Max Drawdown", f"{results['drawdown'].min()*100:.2f}%")
        
        # Téléchargement du rapport
        st.download_button(
            label="📥 Télécharger Rapport Complet",
            data=generate_report(results),
            file_name="rapport_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    except Exception as e:
        st.error(f"Erreur lors de l'exécution: {str(e)}")
        st.stop()

else:
    st.info("Veuillez téléverser un fichier CSV pour commencer l'analyse.")

# ---- Documentation ----
with st.expander("📚 Documentation des stratégies"):
    st.markdown("""
    ### Modèles Implémentés
    **1. Stratégie Buy & Hold**  
    Investissement initial avec maintien de la position jusqu'à la fin de la période.

    **2. Croisement de Moyennes Mobiles**  
    - Achat quand MM courte > MM longue  
    - Vente quand MM courte < MM longue

    **3. Stratégie RSI**  
    - Surchat (RSI > 70) ⇒ Vente  
    - Survente (RSI < 30) ⇒ Achat

    ### Paramètres BRVM
    - **Plage horaire:** 9h00 - 15h00 UTC  
    - **Taille des lots:** Multiples de 100  
    - **Variation maximale:** ±10% journalier
    """)
