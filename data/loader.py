# data/loader.py
import pandas as pd
import io # Utile pour lire un fichier uploadé

def load_historical_data_from_upload(uploaded_file):
    """
    Charge les données historiques depuis un fichier uploadé (CSV attendu).
    Supppose que le fichier CSV a une colonne 'Date' et une colonne 'Close'.

    Args:
        uploaded_file: L'objet fichier obtenu depuis st.file_uploader.

    Returns:
        pd.DataFrame: DataFrame avec les données historiques (Index=Date, Colonne 'Close'),
                      ou None si erreur.
    """
    if uploaded_file is None:
        return None

    try:
        # Lire le fichier uploadé. Utilisez io.StringIO pour les CSV texte.
        # Adaptez si votre format est différent (par ex: Excel -> pd.read_excel)
        dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))

        # --- Nettoyage et préparation des données (À adapter selon votre fichier CSV !) ---
        # Assurez-vous d'avoir une colonne de date et une colonne de prix de clôture.
        # Renommez si nécessaire pour avoir 'Date' et 'Close'.
        # Exemple : si votre colonne de date s'appelle 'TradingDate' et la clôture 'PrixCloture':
        # dataframe.rename(columns={'TradingDate': 'Date', 'PrixCloture': 'Close'}, inplace=True)

        # Convertir la colonne 'Date' en datetime et la définir comme index
        # Assurez-vous que le format de date dans votre CSV est correctement interprété par pd.to_datetime
        if 'Date' not in dataframe.columns:
             print("Erreur: La colonne 'Date' n'est pas trouvée dans le fichier.")
             return None
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe.set_index('Date', inplace=True)

        # Trier par index (date) pour s'assurer de l'ordre chronologique
        dataframe.sort_index(inplace=True)

        # S'assurer que la colonne 'Close' existe et est numérique
        if 'Close' not in dataframe.columns:
            print("Erreur: La colonne 'Close' n'est pas trouvée dans le fichier.")
            return None
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='coerce')
        dataframe.dropna(subset=['Close'], inplace=True) # Supprimer les lignes sans prix de clôture valide


        if dataframe.empty:
             print("Erreur: Aucune donnée valide trouvée après chargement et nettoyage.")
             return None

        return dataframe

    except Exception as e:
        print(f"Erreur lors du chargement ou traitement du fichier : {e}")
        return None

# Fichier __init__.py dans le dossier data (peut être vide)
# data/__init__.py
