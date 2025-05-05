# data/loader.py
import pandas as pd
import io # Utile pour lire les fichiers uploadés
import os # Utile pour obtenir l'extension du nom de fichier

def load_historical_data_from_upload(uploaded_file):
    """
    Charge les données historiques depuis un fichier uploadé (CSV ou Excel attendu).
    Détecte le type de fichier par son extension.
    Supppose que le fichier a une colonne 'Date' et une colonne 'Close'.

    Args:
        uploaded_file: L'objet fichier obtenu depuis st.file_uploader.

    Returns:
        pd.DataFrame: DataFrame avec les données historiques (Index=Date, Colonne 'Close'),
                      ou None si erreur, format non supporté, ou colonnes manquantes.
    """
    if uploaded_file is None:
        return None

    # Obtenir le nom et l'extension du fichier
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower() # Ex: '.csv', '.xlsx'

    dataframe = None

    try:
        if file_extension == '.csv':
            # Lire le fichier CSV
            # Utilise io.StringIO et decode pour lire depuis l'objet BytesIO de Streamlit
            dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            print("Fichier détecté comme CSV.")

        elif file_extension in ['.xlsx', '.xls']:
            # Lire le fichier Excel
            # pandas.read_excel gère directement l'objet fichier uploadé
            dataframe = pd.read_excel(uploaded_file)
            print(f"Fichier détecté comme Excel ({file_extension}).")

        else:
            print(f"Format de fichier non supporté : {file_extension}")
            return None

        # --- Validation et préparation des données (Appliqué après lecture, quel que soit le format) ---
        if dataframe is None or dataframe.empty:
            print("Le fichier a été lu mais le DataFrame est vide.")
            return None

        # Attendre les colonnes nommées 'Date' et 'Close'
        required_columns = ['Date', 'Close']
        if not all(col in dataframe.columns for col in required_columns):
             print(f"Erreur: Le fichier doit contenir les colonnes: {required_columns}")
             print(f"Colonnes trouvées: {dataframe.columns.tolist()}")
             return None

        # Convertir la colonne 'Date' en datetime et la définir comme index
        # Ceci fonctionne pour les dates lues depuis CSV ou Excel, tant que le format est standard.
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

# data/__init__.py reste vide
