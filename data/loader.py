# data/loader.py
import pandas as pd
import io
import os

def load_historical_data_from_upload(uploaded_file):
    # ... (début de la fonction) ...

    print(f"Tentative de chargement du fichier : {file_name}")
    print(f"Extension détectée : {file_extension}")

    dataframe = None

    try:
        if file_extension == '.csv':
            # ... lecture CSV ...
            print("Lecture CSV réussie (si pas d'exception ici).")

        elif file_extension in ['.xlsx', '.xls']:
            # ... lecture Excel ...
            print("Lecture Excel réussie (si pas d'exception ici).")

        else:
            print(f"Format de fichier non supporté : {file_extension}")
            return None

        # --- Validation et préparation ---
        if dataframe is None or dataframe.empty:
            print("DataFrame est None ou vide juste après la lecture.")
            return None

        print(f"DataFrame chargé. Forme : {dataframe.shape}")
        print(f"Colonnes trouvées : {dataframe.columns.tolist()}")
        print(f"Index actuel : {dataframe.index.name}")

        required_columns = ['Date', 'Close']
        if not all(col in dataframe.columns for col in required_columns):
             print(f"Erreur: Colonnes requises {required_columns} non trouvées.")
             print(f"Colonnes présentes : {dataframe.columns.tolist()}")
             return None

        # Convertir la colonne 'Date'
        print(f"Tentative de conversion de la colonne 'Date' (type actuel: {dataframe['Date'].dtype})...")
        original_date_col = dataframe['Date'] # Garder l'original pour débogage
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], errors='coerce') # Utilisez errors='coerce' pour voir quelles valeurs échouent

        if dataframe['Date'].isnull().any():
             print("Attention: Certaines dates n'ont pas pu être parsées.")
             # Optionnel: Afficher les valeurs qui n'ont pas pu être parsées
             # print(original_date_col[dataframe['Date'].isnull()])
             # Vous pourriez décider de supprimer ces lignes si nécessaire
             # dataframe.dropna(subset=['Date'], inplace=True)


        dataframe.set_index('Date', inplace=True)
        print("Colonne 'Date' définie comme index.")

        dataframe.sort_index(inplace=True)
        print("DataFrame trié par date.")

        # S'assurer que 'Close' est numérique
        print(f"Tentative de conversion de la colonne 'Close' (type actuel: {dataframe['Close'].dtype})...")
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='coerce')
        initial_rows = len(dataframe)
        dataframe.dropna(subset=['Close'], inplace=True)
        if len(dataframe) < initial_rows:
             print(f"Attention: {initial_rows - len(dataframe)} lignes supprimées car 'Close' n'était pas numérique.")


        if dataframe.empty:
             print("Le DataFrame est devenu vide après le nettoyage des dates/prix.")
             return None

        print("Chargement et préparation réussis.")
        return dataframe

    except Exception as e:
        print(f"Une erreur inattendue s'est produite pendant le traitement : {e}")
        return None
