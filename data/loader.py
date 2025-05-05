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
                      ou None si erreur ou si le fichier n'a pas le bon format/colonnes.
    """
    if uploaded_file is None:
        return None

    try:
        # Lire le fichier uploadé. Utilisez io.StringIO pour les CSV texte.
        # Adaptez si votre format est différent (par ex: Excel -> pd.read_excel)
        # decode('utf-8') suppose que le fichier est encodé en UTF-8
        dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))

        # --- Validation et préparation des données (À adapter si vos noms de colonnes sont différents !) ---
        # Attendre les colonnes nommées 'Date' et 'Close'
        required_columns = ['Date', 'Close']
        if not all(col in dataframe.columns for col in required_columns):
             print(f"Erreur: Le fichier CSV doit contenir les colonnes: {required_columns}")
             # Afficher les colonnes trouvées pour aider au débogage
             print(f"Colonnes trouvées: {dataframe.columns.tolist()}")
             return None

        # Convertir la colonne 'Date' en datetime et la définir comme index
        # Assurez-vous que le format de date dans votre CSV est correctement interprété par pd.to_datetime
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe.set_index('Date', inplace=True)

        # Trier par index (date) pour s'assurer de l'ordre chronologique
        dataframe.sort_index(inplace=True)

        # S'assurer que la colonne 'Close' est numérique et gérer les erreurs
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='coerce')
        # Supprimer les lignes où 'Close' n'est pas un nombre valide après conversion
        dataframe.dropna(subset=['Close'], inplace=True)

        if dataframe.empty:
             print("Erreur: Aucune donnée valide trouvée après chargement et nettoyage.")
             return None

        return dataframe

    except Exception as e:
        # Capturer toute autre erreur lors de la lecture ou du traitement
        print(f"Erreur lors du chargement ou traitement du fichier : {e}")
        return None

# Note : Le fichier __init__.py dans le dossier data est nécessaire même s'il est vide.
