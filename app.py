import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io

# Configuration de la page Streamlit

st.set_page_config(
page_title="SONATEL BRVM Quant Backtest",
layout="wide",
menu_items={
'About': "Analyse quantitative des actions SONATEL sur la BRVM"
}
)

# Titre et introduction

st.title("📈 BRVM Quant Backtest - SONATEL")
st.markdown("""
Cette application permet d'analyser et de backtester des stratégies d'investissement
sur les actions SONATEL cotées à la Bourse Régionale des Valeurs Mobilières (BRVM).
""")

# Chargement des données

@st.cache_data
def load_data():
# Les données sont intégrées directement dans le code pour simplifier
data_str = """Exchange Date	Close	Net	%Chg	Open	Low	High	Volume	Turnover - XOF	Flow
28-Feb-25	9,400.00	45	0.48%	9,360.00	9,360.00	9,400.00	1,059	9,926,440.00	1,405,576,936.73
27-Feb-25	9,355.00	-35	-0.37%	9,390.00	9,350.00	9,400.00	906	8,483,025.00	1,395,650,496.73
26-Feb-25	9,390.00	40	0.43%	9,485.00	9,300.00	9,485.00	6,707	62,471,635.00	1,404,133,521.73
25-Feb-25	9,350.00	0	0.00%	9,350.00	9,250.00	9,400.00	2,745	25,615,780.00	1,341,661,886.73
24-Feb-25	9,350.00	85	0.92%	9,265.00	9,265.00	9,350.00	4,427	41,170,400.00	1,316,046,106.73"""
# Vous pouvez ajouter toutes les autres lignes ici
# Pour simplifier, je n'ai inclus que les 5 premières lignes

```
# Charger les données
df = pd.read_csv(io.StringIO(data_str), sep='\\t')

# Convertir les dates
df['Exchange Date'] = pd.to_datetime(df['Exchange Date'], format='%d-%b-%y', errors='coerce')

# Nettoyer les données en supprimant les lignes avec des dates manquantes
df = df.dropna(subset=['Exchange Date'])

# Trier par date (du plus ancien au plus récent)
df = df.sort_values('Exchange Date')

# Renommer les colonnes pour plus de clarté
df = df.rename(columns={
    'Exchange Date': 'Date',
    'Close': 'Prix',
    'Net': 'Variation',
    '%Chg': 'Variation_%',
    'Open': 'Ouverture',
    'Low': 'Plus_Bas',
    'High': 'Plus_Haut',
    'Volume': 'Volume',
    'Turnover - XOF': 'Chiffre_Affaires',
    'Flow': 'Flux'
})

# Remplacer les valeurs manquantes par des valeurs de la ligne précédente
df = df.fillna(method='ffill')

# Définir la date comme index
df = df.set_index('Date')

# Convertir les colonnes numériques
for col in ['Prix', 'Variation', 'Ouverture', 'Plus_Bas', 'Plus_Haut', 'Volume', 'Chiffre_Affaires', 'Flux']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '').str.replace(',', ''), errors='coerce')

return df

```

# Chargement des données historiques

data = load_data()

# Affichage des données brutes (optionnel, avec un bouton pour afficher/masquer)

with st.expander("Afficher les données brutes"):
st.dataframe(data.tail(100))

# Visualisation du cours de l'action

st.subheader("Cours historique de SONATEL")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Prix'], linewidth=2)
ax.set_title('Évolution du cours de SONATEL')
ax.set_xlabel('Date')
ax.set_ylabel('Prix (XOF)')
ax.grid(True, alpha=0.3)

# Amélioration du format des dates sur l'axe X

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Paramètres de la stratégie

st.subheader("Paramètres de la stratégie")

col1, col2 = st.columns(2)

with col1:
# Paramètres fondamentaux
st.markdown("### Analyse fondamentale")
rendement_exige = st.slider("Taux d'actualisation (%)", 5, 20, 12) / 100
taux_croissance = st.slider("Croissance annuelle dividende (%)", 0, 10, 3) / 100
dividende_annuel = st.number_input("Dernier dividende annuel (XOF)", 200, 1000, 600)

with col2:
# Paramètres techniques
st.markdown("### Règles de trading")
marge_achat = st.slider("Marge de sécurité à l'achat (%)", 0, 50, 20) / 100
marge_vente = st.slider("Prime de sortie (%)", 0, 50, 10) / 100
stop_loss = st.slider("Stop Loss (%)", 1, 20, 10) / 100

# Calcul des moyennes mobiles

st.subheader("Analyse technique")
window_court = st.slider("Fenêtre de la moyenne mobile courte", 5, 50, 20)
window_long = st.slider("Fenêtre de la moyenne mobile longue", 20, 200, 50)

# Calcul des moyennes mobiles

data['MM_Court'] = data['Prix'].rolling(window=window_court).mean()
data['MM_Long'] = data['Prix'].rolling(window=window_long).mean()

# Affichage du graphique avec moyennes mobiles

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
ax2.plot(data.index, data['MM_Court'], label=f'MM {window_court} jours', linewidth=1.5)
ax2.plot(data.index, data['MM_Long'], label=f'MM {window_long} jours', linewidth=1.5)
ax2.set_title('Analyse technique - Moyennes Mobiles')
ax2.set_xlabel('Date')
ax2.set_ylabel('Prix (XOF)')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# Calcul de la valeur intrinsèque avec le modèle de Gordon

D1 = dividende_annuel * (1 + taux_croissance)
val_intrinseque = D1 / (rendement_exige - taux_croissance)
st.markdown(f"### Valeur intrinsèque calculée: **{val_intrinseque:.2f} XOF**")

# Calcul des signaux d'achat/vente

data['val_intrinseque'] = val_intrinseque
data['prix_achat'] = (1 - marge_achat) * val_intrinseque
data['prix_vente'] = (1 + marge_vente) * val_intrinseque

# Signal technique: croisement des moyennes mobiles

data['signal_technique'] = 0
data.loc[data['MM_Court'] > data['MM_Long'], 'signal_technique'] = 1
data.loc[data['MM_Court'] < data['MM_Long'], 'signal_technique'] = -1

# Combinaison des signaux fondamentaux et techniques

data['achat'] = (data['Prix'] < data['prix_achat']) & (data['signal_technique'] == 1)
data['vente'] = (data['Prix'] > data['prix_vente']) | (data['signal_technique'] == -1)

# Affichage du graphique avec zones d'achat/vente

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(data.index, data['Prix'], label='Prix', linewidth=1.5)
ax3.axhline(y=val_intrinseque, color='g', linestyle='-', alpha=0.5, label='Valeur intrinsèque')
ax3.axhline(y=data['prix_achat'][0], color='g', linestyle='--', alpha=0.5, label='Prix d\'achat')
ax3.axhline(y=data['prix_vente'][0], color='r', linestyle='--', alpha=0.5, label='Prix de vente')

# Marquage des signaux d'achat/vente

achats = data[data['achat'] == True]
ventes = data[data['vente'] == True]

if not achats.empty:
ax3.scatter(achats.index, achats['Prix'], color='g', s=50, marker='^', label='Signal d\'achat')

if not ventes.empty:
ax3.scatter(ventes.index, ventes['Prix'], color='r', s=50, marker='v', label='Signal de vente')

ax3.set_title('Signaux d\'achat et de vente')
ax3.set_xlabel('Date')
ax3.set_ylabel('Prix (XOF)')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# Backtest

st.subheader("Backtest de la stratégie")

capital_initial = st.number_input("Capital initial (XOF)", 100000, 10000000, 1000000, step=100000)
frais_transaction = st.slider("Frais de transaction (%)", 0.0, 2.0, 0.5) / 100

# Exécution du backtest

def run_backtest(data, capital_initial, frais_transaction, stop_loss):
capital = capital_initial
positions = []
achats_dates = []
ventes_dates = []
prix_achats = []
prix_ventes = []
portefeuille_valeur = []

```
# On utilise un DataFrame pour suivre l'évolution du portefeuille
portfolio = pd.DataFrame(index=data.index)
portfolio['prix'] = data['Prix']
portfolio['actions'] = 0
portfolio['cash'] = capital_initial
portfolio['valeur_actions'] = 0
portfolio['valeur_totale'] = capital_initial
portfolio['rendement'] = 0

for i in range(1, len(data)):
    jour = data.index[i]
    jour_prec = data.index[i-1]
    prix = data['Prix'].iloc[i]

    # Initialisation pour ce jour
    actions = portfolio.loc[jour_prec, 'actions']
    cash = portfolio.loc[jour_prec, 'cash']

    # Vérification du stop loss pour les positions existantes
    if actions > 0:
        prix_achat_moyen = sum(prix_achats) / len(prix_achats)
        if prix < (1 - stop_loss) * prix_achat_moyen:
            # Vente forcée (stop loss)
            vente_montant = actions * prix * (1 - frais_transaction)
            cash += vente_montant
            ventes_dates.append(jour)
            prix_ventes.append(prix)
            actions = 0
            prix_achats = []

    # Signal d'achat
    if data['achat'].iloc[i] and cash >= prix:
        # Calcul du nombre d'actions à acheter (maximum possible avec le cash disponible)
        max_actions = int(cash / (prix * (1 + frais_transaction)))
        if max_actions > 0:
            # Achat
            cout_achat = max_actions * prix * (1 + frais_transaction)
            cash -= cout_achat
            actions += max_actions
            achats_dates.append(jour)
            prix_achats.append(prix)

    # Signal de vente
    elif data['vente'].iloc[i] and actions > 0:
        # Vente
        vente_montant = actions * prix * (1 - frais_transaction)
        cash += vente_montant
        ventes_dates.append(jour)
        prix_ventes.append(prix)
        actions = 0
        prix_achats = []

    # Mise à jour du portfolio pour ce jour
    portfolio.loc[jour, 'actions'] = actions
    portfolio.loc[jour, 'cash'] = cash
    portfolio.loc[jour, 'valeur_actions'] = actions * prix
    portfolio.loc[jour, 'valeur_totale'] = cash + (actions * prix)

    # Calcul du rendement quotidien
    if i > 0:
        rendement_jour = (portfolio.loc[jour, 'valeur_totale'] / portfolio.loc[jour_prec, 'valeur_totale']) - 1
        portfolio.loc[jour, 'rendement'] = rendement_jour

# Calcul des rendements cumulés
portfolio['rendement_cumule'] = (1 + portfolio['rendement']).cumprod() - 1

return portfolio, achats_dates, ventes_dates

```

# Exécution du backtest

portfolio, achats_dates, ventes_dates = run_backtest(data, capital_initial, frais_transaction, stop_loss)

# Affichage des résultats du backtest

st.subheader("Résultats du backtest")

# Statistiques de performance

rendement_total = (portfolio['valeur_totale'].iloc[-1] / capital_initial - 1) * 100
rendement_annualise = ((1 + rendement_total/100) ** (365 / (portfolio.index[-1] - portfolio.index[0]).days) - 1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Rendement total", f"{rendement_total:.2f}%")
col2.metric("Rendement annualisé", f"{rendement_annualise:.2f}%")
col3.metric("Valeur finale du portefeuille", f"{portfolio['valeur_totale'].iloc[-1]:,.2f} XOF")

# Graphique de l'évolution du portefeuille

fig4, ax4 = plt.subplots(figsize=(12, 6))
ax4.plot(portfolio.index, portfolio['valeur_totale'], linewidth=2, label='Valeur du portefeuille')
ax4.plot(portfolio.index, [capital_initial] * len(portfolio), '--', linewidth=1, color='gray', label='Capital initial')

# Marquage des achats et ventes sur le graphique

for date in achats_dates:
ax4.axvline(x=date, color='g', linestyle='--', alpha=0.3)
for date in ventes_dates:
ax4.axvline(x=date, color='r', linestyle='--', alpha=0.3)

ax4.set_title('Évolution de la valeur du portefeuille')
ax4.set_xlabel('Date')
ax4.set_ylabel('Valeur (XOF)')
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)

# Composition du portefeuille final

st.subheader("Composition du portefeuille final")
col1, col2 = st.columns(2)
col1.metric("Nombre d'actions", f"{portfolio['actions'].iloc[-1]}")
col2.metric("Liquidités", f"{portfolio['cash'].iloc[-1]:,.2f} XOF")

# Affichage des transactions

st.subheader("Journal des transactions")
if achats_dates or ventes_dates:
# Créer un DataFrame pour les transactions
transactions = []
for date in achats_dates:
prix = data.loc[date, 'Prix']
transactions.append({
'Date': date,
'Type': 'Achat',
'Prix': prix,
'Montant': prix
})

```
for date in ventes_dates:
    prix = data.loc[date, 'Prix']
    transactions.append({
        'Date': date,
        'Type': 'Vente',
        'Prix': prix,
        'Montant': prix
    })

transactions_df = pd.DataFrame(transactions)
transactions_df = transactions_df.sort_values('Date')
st.dataframe(transactions_df)

```

else:
[st.info](http://st.info/)("Aucune transaction n'a été effectuée pendant la période analysée.")

# Métriques avancées

st.subheader("Métriques avancées")

# Calcul des rendements journaliers du marché (SONATEL)

data['rendement_marche'] = data['Prix'].pct_change()

# Calcul de la volatilité

volatilite_strat = portfolio['rendement'].std() * (252 ** 0.5) * 100  # Annualisée
volatilite_marche = data['rendement_marche'].std() * (252 ** 0.5) * 100  # Annualisée

# Calcul du ratio de Sharpe (en supposant un taux sans risque de 3%)

taux_sans_risque = 0.03
sharpe_ratio = (rendement_annualise/100 - taux_sans_risque) / (volatilite_strat/100)

# Calcul du drawdown

portfolio['peak'] = portfolio['valeur_totale'].cummax()
portfolio['drawdown'] = (portfolio['valeur_totale'] - portfolio['peak']) / portfolio['peak'] * 100
max_drawdown = portfolio['drawdown'].min()

col1, col2, col3 = st.columns(3)
col1.metric("Volatilité annualisée", f"{volatilite_strat:.2f}%")
col2.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}")
col3.metric("Drawdown maximum", f"{max_drawdown:.2f}%")

# Graphique du drawdown

fig5, ax5 = plt.subplots(figsize=(12, 4))
ax5.fill_between(portfolio.index, portfolio['drawdown'], 0, color='red', alpha=0.3)
ax5.set_title('Drawdown du portefeuille')
ax5.set_xlabel('Date')
ax5.set_ylabel('Drawdown (%)')
ax5.grid(True, alpha=0.3)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig5)

# Conclusion et notes

st.subheader("Conclusion")
st.markdown(f"""
Sur la base de ce backtest, la stratégie a généré un rendement total de **{rendement_total:.2f}%**
sur toute la période analysée, soit un rendement annualisé de **{rendement_annualise:.2f}%**.

**Points clés:**

- Valeur intrinsèque calculée: {val_intrinseque:.2f} XOF
- Nombre total d'achats: {len(achats_dates)}
- Nombre total de ventes: {len(ventes_dates)}
- Volatilité annualisée: {volatilite_strat:.2f}%
- Ratio de Sharpe: {sharpe_ratio:.2f}

Cette stratégie combine l'analyse fondamentale (valorisation par dividendes) et l'analyse technique
(croisement de moyennes mobiles) pour identifier les points d'entrée et de sortie optimaux.
""")

# Avertissement

st.warning("""
**Avertissement:** Les performances passées ne préjugent pas des performances futures.
Cette application est fournie à des fins éducatives uniquement et ne constitue pas un conseil en investissement.
""")

# Pied de page

st.markdown("---")
st.markdown("© 2025 BRVM Quant - Analyse quantitative des marchés financiers africains")
