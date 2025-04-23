# 📈 BRVM Quant - Backtest SONATEL

Une application d'analyse quantitative pour les actions SONATEL cotées à la Bourse Régionale des Valeurs Mobilières (BRVM).

## Présentation

Cette application Streamlit permet de backtester une stratégie d'investissement combinant analyse fondamentale et technique sur les actions SONATEL. L'objectif est de démontrer qu'une stratégie disciplinée peut générer des rendements supérieurs à 20% par an sur ce titre liquide du marché boursier ouest-africain.

## Fonctionnalités

- 📊 **Analyse des données historiques** SONATEL
- 💰 **Calcul de la valeur intrinsèque** par le modèle de Gordon-Shapiro
- 📉 **Analyse technique** avec croisement de moyennes mobiles
- 🧮 **Backtest complet** avec simulation de portefeuille
- 📈 **Métriques de performance** (rendement, volatilité, Sharpe ratio, drawdown)
- 📒 **Journal des transactions** détaillé

## Stratégie d'investissement

La stratégie combine deux approches:

1. **Analyse fondamentale**: Calcul d'une valeur intrinsèque basée sur les dividendes futurs estimés (modèle de Gordon)
2. **Analyse technique**: Utilisation du croisement de moyennes mobiles comme confirmation

Un achat est généré lorsque:
- Le prix est inférieur à la valeur intrinsèque (avec une marge de sécurité)
- ET le signal technique est positif (MM courte > MM longue)

Une vente est générée lorsque:
- Le prix dépasse la valeur intrinsèque (avec une prime)
- OU le signal technique devient négatif
- OU le stop-loss est déclenché

## Installation et utilisation

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/votre-utilisateur/brvm-quant.git
cd brvm-quant

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### Déploiement sur Streamlit Cloud

1. Créez un compte sur [Streamlit Cloud](https://streamlit.io/cloud)
2. Connectez votre compte GitHub
3. Sélectionnez le repository et déployez l'application

## Structure du projet

```
brvm-quant/
│
├── app.py               ← Application Streamlit principale
├── requirements.txt     ← Liste des dépendances Python
├── README.md            ← Ce fichier de documentation
└── .streamlit/          ← Configuration Streamlit (optionnel)
    └── config.toml      ← Configuration pour Streamlit Cloud
```

## Résultats attendus

Avec les paramètres par défaut, cette stratégie vise à:
- Générer un rendement annualisé > 20%
- Maintenir un ratio de Sharpe > 1.5
- Limiter le drawdown maximum à moins de 25%

## Personnalisation

L'application permet de personnaliser plusieurs paramètres:
- Taux d'actualisation et de croissance pour le modèle fondamental
- Fenêtres des moyennes mobiles pour l'analyse technique
- Marges de sécurité pour l'achat et la vente
- Stop-loss et frais de transaction

## Limites et avertissements

- Les performances passées ne préjugent pas des performances futures
- Cette application est fournie à des fins éducatives uniquement
- Ne constitue pas un conseil en investissement

## Contact et contribution

N'hésitez pas à contribuer à ce projet en soumettant des pull requests ou en signalant des problèmes via GitHub.

## Licence

Ce projet est sous licence MIT.
