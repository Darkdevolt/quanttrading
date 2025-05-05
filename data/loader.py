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
        print("Aucun fichier n'a été uploadé.") # Debug print
        return None

    # Obtenir le nom et l'extension du fichier
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower() # Ex: '.csv', '.xlsx'

    # --- Le print déplacé APRÈS la définition de file_name ---
    print(f"Tentative de chargement du fichier : {file_name}")
    print(f"Extension détectée : {file_extension}")
    # -----------------------------------------------------

    dataframe = None

    try:
        if file_extension == '.csv':
            print("Lecture du fichier comme CSV...") # Debug print
            # Utilise io.StringIO et decode pour lire depuis l'objet BytesIO de Streamlit
            dataframe = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
            print("Lecture CSV réussie (si pas d'exception ici).") # Debug print

        elif file_extension in ['.xlsx', '.xls']:
            print(f"Lecture du fichier comme Excel ({file_extension})...") # Debug print
            # pandas.read_excel gère directement l'objet fichier uploadé
            dataframe = pd.read_excel(uploaded_file)
            print("Lecture Excel réussie (si pas d'exception ici).") # Debug print

        else:
            print(f"Format de fichier non supporté : {file_extension}") # Debug print
            return None

        # --- Validation et préparation (Appliqué après lecture) ---
        if dataframe is None or dataframe.empty:
            print("DataFrame est None ou vide juste après la lecture.") # Debug print
            return None

        print(f"DataFrame chargé. Forme : {dataframe.shape}") # Debug print
        print(f"Colonnes trouvées : {dataframe.columns.tolist()}") # Debug print
        # L'index n'est pas encore défini, donc on ne peut pas print son nom ici sans risque d'erreur si set_index échoue

        required_columns = ['Date', 'Close']
        if not all(col in dataframe.columns for col in required_columns):
             print(f"Erreur: Colonnes requises {required_columns} non trouvées.") # Debug print
             print(f"Colonnes présentes : {dataframe.columns.tolist()}") # Debug print
             return None

        # Convertir la colonne 'Date'
        print(f"Tentative de conversion de la colonne 'Date' (type actuel: {dataframe['Date'].dtype})...") # Debug print
        original_date_col_for_debug = dataframe['Date'] # Garder l'original pour débogage si besoin
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], errors='coerce') # Utilisez errors='coerce' pour voir quelles valeurs échouent

        if dataframe['Date'].isnull().any():
             print("Attention: Certaines dates n'ont pas pu être parsées et sont NaN.") # Debug print
             # Optionnel: Afficher les valeurs qui n'ont pas pu être parsées
             # print("Exemples de dates non parsées:")
             # print(original_date_col_for_debug[dataframe['Date'].isnull()].head())
             # Vous pourriez décider de supprimer ces lignes si nécessaire
             # dataframe.dropna(subset=['Date'], inplace=True) # Si vous décommentez ceci, ajoutez un print avant/après

        # IMPORTANT : S'assurer qu'aucune date n'est NaT (Not a Time) avant de définir l'index
        dataframe.dropna(subset=['Date'], inplace=True) # Supprime les lignes où la conversion de date a échoué
        if dataframe.empty:
            print("Le DataFrame est vide après suppression des dates non valides.") # Debug print
            return None


        dataframe.set_index('Date', inplace=True)
        print("Colonne 'Date' définie comme index.") # Debug print
        print(f"Index actuel après set_index : {dataframe.index.name}") # Debug print


        dataframe.sort_index(inplace=True)
        print("DataFrame trié par date.") # Debug print

        # S'assurer que 'Close' est numérique
        print(f"Tentative de conversion de la colonne 'Close' (type actuel: {dataframe['Close'].dtype})...") # Debug print
        dataframe['Close'] = pd.to_numeric(dataframe['Close'], errors='coerce')
        initial_rows_close_check = len(dataframe)
        dataframe.dropna(subset=['Close'], inplace=True) # Supprime les lignes où 'Close' n'est pas numérique
        if len(dataframe) < initial_rows_close_check:
             print(f"Attention: {initial_rows_close_check - len(dataframe)} lignes supprimées car 'Close' n'était pas numérique.") # Debug print


        if dataframe.empty:
             print("Le DataFrame est devenu vide après le nettoyage des dates/prix.") # Debug print
             return None

        print("Chargement et préparation réussis.") # Debug print
        print(f"DataFrame final shape: {dataframe.shape}") # Debug print
        print(f"DataFrame final head:\n{dataframe.head()}") # Debug print


        return dataframe

    except Exception as e:
        print(f"Une erreur inattendue s'est produite pendant le traitement : {e}") # Debug print
        # Afficher un message d'erreur plus spécifique si possible
        if "No sheet named" in str(e):
             print("Erreur: Vérifiez que le fichier Excel contient au moins une feuille de calcul valide.") # Debug print
        return None
