import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# URL de la page du palmarès Sikafinance BRVM
SIKAFINANCE_URL = "https://www.sikafinance.com/marches/palmares"

# --- Fonction de scraping (mise en cache avec Streamlit) ---
@st.cache_data(ttl=3600) # Cache les données pendant 1 heure (3600 secondes)
def scrape_brvm_palmares(url):
    """
    Scrape les données du palmarès (Top 5 Hausses et Baisses) depuis Sikafinance.
    Retourne un dictionnaire de DataFrames pandas.
    """
    data = {}
    try:
        response = requests.get(url)
        response.raise_for_status() # Lève une exception pour les codes d'état d'erreur (4xx ou 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Recherche des tables ---
        # Sikafinance utilise des titres h2 pour les sections "Top 5 Hausses" et "Top 5 Baisses"
        # Nous allons chercher ces titres puis les tables associées.

        # Top 5 Hausses
        hausses_title = soup.find('h2', string='Top 5 Hausses')
        if hausses_title:
            hausses_table = hausses_title.find_next('table')
            if hausses_table:
                data['Top 5 Hausses'] = parse_table(hausses_table)

        # Top 5 Baisses
        baisses_title = soup.find('h2', string='Top 5 Baisses')
        if baisses_title:
            baisses_table = baisses_title.find_next('table')
            if baisses_table:
                data['Top 5 Baisses'] = parse_table(baisses_table)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des données depuis Sikafinance : {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur est survenue lors du parsing des données : {e}")
        return None

    return data

# --- Fonction utilitaire pour parser une table HTML en DataFrame ---
def parse_table(table_element):
    """Parse une table HTML en DataFrame pandas."""
    headers = [th.get_text(strip=True) for th in table_element.find_all('th')]
    rows = []
    for tr in table_element.find_all('tr')[1:]: # Ignorer la ligne d'en-tête si présente dans le tbody
        cells = [td.get_text(strip=True) for td in tr.find_all('td')]
        if cells: # S'assurer que la ligne n'est pas vide
            rows.append(cells)

    # Créer le DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Tenter de convertir les colonnes numériques
    for col in ['Cours', 'Variation', 'Volume', 'Transactions', 'Capitalisation']:
        if col in df.columns:
            # Nettoyer les données (ex: remplacer ',' par '.', supprimer ' ', 'XOF')
            df[col] = df[col].str.replace(' ', '').str.replace(',', '.', regex=False).str.replace('XOF', '', regex=False)
            # Convertir en numérique, les erreurs seront NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# --- Application Streamlit ---
st.title("🚀 Données Financières BRVM - Palmarès")
st.markdown("📈 Palmarès (Top 5 Hausses et Baisses) de la Bourse Régionale des Valeurs Mobilières (BRVM) scanné depuis [Sikafinance](https://www.sikafinance.com/marches/palmares).")

# Récupérer les données
palmares_data = scrape_brvm_palmares(SIKAFINANCE_URL)

if palmares_data:
    st.write(f"Dernière mise à jour des données : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.warning("Les données sont mises en cache pendant 1 heure pour éviter de surcharger le site source.")

    # Afficher les tables
    for title, df in palmares_data.items():
        st.subheader(title)
        if not df.empty:
            # Formatage des colonnes numériques pour une meilleure lisibilité
            # Détecter les colonnes numériques présentes
            numeric_cols = [col for col in ['Cours', 'Variation', 'Volume', 'Transactions', 'Capitalisation'] if col in df.columns]
            
            # Appliquer le formatage
            st.dataframe(df.style.format({
                'Cours': '{:,.2f}'.format, # 2 décimales, séparateur de milliers
                'Variation': '{:,.2f}%'.format, # Pourcentage
                'Volume': '{:,.0f}'.format, # Entier, séparateur de milliers
                'Transactions': '{:,.0f}'.format, # Entier, séparateur de milliers
                'Capitalisation': '{:,.0f}'.format # Entier, séparateur de milliers
            }, na_rep="-"), use_container_width=True)
        else:
            st.info(f"Aucune donnée trouvée pour '{title}'.")

else:
    st.error("Impossible de charger les données du palmarès pour le moment.")

st.markdown("---")
st.markdown("Créé pour répondre à une demande.")
