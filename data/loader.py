# data/loader.py
import pandas as pd
import io # Utile pour lire un fichier uploadé

def load_historical_data_from_upload(uploaded_file):
    """
    Charge les données historiques depuis un fichier Excel uploadé (.xlsx/.xls attendu).
    Supppose que le fichier a une colonne 'Date' et une colonne 'Close'.

    Args:
        uploaded_file: L'objet fichier obtenu depuis st.file_uploader.

    Returns:
        pd.DataFrame: DataFrame avec les données historiques (Index=Date, Colonne 'Close'),
                      ou None si erreur ou si le fichier n'a pas le bon format/colonnes.
    """
    if uploaded_file is None:
        return None

    try:
        # Lire le fichier Excel uploadé
        # pandas.read_excel gère directement l'objet fichier uploadé
        dataframe = pd.read_excel(uploaded_file)

        # --- Validation et préparation des données ---
        # Vos colonnes 'Date' et 'Close' correspondent à ce que le code attend, c'est parfait.
        required_columns = ['Date', 'Close']
        if not all(col in dataframe.columns for col in required_columns):
             print(f"Erreur: Le fichier Excel doit contenir les colonnes: {required_columns}")
             print(f"Colonnes trouvées: {dataframe.columns.tolist()}") # Afficher les colonnes trouvées
             return None

        # Convertir la colonne 'Date' en datetime et la définir comme index
        # pandas.read_excel essaie généralement de détecter le format de date, mais assurez-vous qu'il est clair.
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe.set_index('Date', inplace=True)

        # Trier par index (date) pour s'assurer de l'ordre chronologique
        dataframe.sort_index(inplace=True)

        # S'assurer que la colonne 'Close' est numérique et gérer les erreurs
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='coerce')
        # Supprimer les lignes où 'Close' n'est pas un nombre valide après conversion
        dataframe.dropna(subset=['Close'], inplace=True)

        # Inclure d'autres colonnes si vous en avez besoin plus tard (Open, High, Low, Volume)
        # Par exemple, pour les graphiques avancés, vous pourriez vouloir garder:
        # dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Mais pour le backtest MA simple, seule 'Close' est requise par défaut.

        if dataframe.empty:
             print("Erreur: Aucune donnée valide trouvée après chargement et nettoyage.")
             return None

        return dataframe

    except Exception as e:
        print(f"Erreur lors du chargement ou traitement du fichier Excel : {e}")
        # Afficher un message d'erreur plus spécifique si possible
        if "No sheet named" in str(e):
             print("Erreur: Vérifiez que le fichier Excel contient au moins une feuille de calcul valide.")
        return None

# data/__init__.py reste vide
