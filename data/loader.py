# data/loader.py
import pandas as pd
import io
import os
import csv # Nécessaire pour le sniffer de séparateur

def load_and_process_data(uploaded_file, column_mapping, date_format=None):
    """
    Charge, valide et traite les données historiques depuis un fichier uploadé.
    Gère le mapping des colonnes, la conversion des dates/nombres, et le nettoyage.

    Args:
        uploaded_file: Objet fichier uploadé par Streamlit.
        column_mapping (dict): Dictionnaire mappant les noms standardisés
                               aux noms de colonnes du fichier source.
                               Ex: {"Date": "Date", "Open": "Open", ...}
        date_format (str, optional): Format de date explicite à essayer si la conversion échoue.

    Returns:
        pd.DataFrame: DataFrame traité et standardisé (index = Date, colonnes = standardisées),
                      ou None en cas d'erreur.
    """
    if uploaded_file is None:
        # Le message d'erreur "Veuillez charger un fichier" sera géré dans l'UI (app.py)
        return None

    # Vérifier que le mapping minimal est fourni
    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys):
         # Le message d'avertissement sur les colonnes manquantes sera géré dans l'UI (app.py)
         return None

    try:
        # --- Détection du séparateur et lecture du fichier CSV ---
        # Revenir au début du fichier pour la lecture
        uploaded_file.seek(0)
        sample_bytes = uploaded_file.read(2048) # Lire un échantillon pour le sniffer
        uploaded_file.seek(0) # Revenir au début pour la lecture complète

        try:
            sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError:
             sample_text = sample_bytes.decode('latin-1', errors='ignore') # Essayer Latin-1 si UTF-8 échoue

        separator = ',' # Séparateur par défaut
        try:
            if sample_text.strip():
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
                # print(f"Séparateur détecté par Sniffer : '{separator}'") # Logs de débogage
            # else: print("L'échantillon du fichier est vide...") # Logs de débogage
        except csv.Error:
            # print("Sniffer n'a pas pu déterminer le séparateur. Essai manuel...") # Logs de débogage
            # Fallback manuel pour séparateur (simple : compte ';' vs ',')
            uploaded_file.seek(0)
            try:
                header_line_bytes = uploaded_file.readline()
                try: header_line = header_line_bytes.decode('utf-8')
                except UnicodeDecodeError: header_line = header_line_bytes.decode('latin-1', errors='ignore')
            except Exception as read_err:
                 print(f"Impossible de lire la première ligne pour la détection manuelle du séparateur: {read_err}") # Logs de débogage
                 return None
            uploaded_file.seek(0) # Revenir au début
            if header_line and header_line.count(';') >= header_line.count(','):
                 separator = ';'
            else:
                 separator = ','
            # print(f"Utilisation probable du séparateur '{separator}'...") # Logs de débogage

        # Lire le fichier CSV complet avec le séparateur détecté
        uploaded_file.seek(0) # Revenir au début une dernière fois avant la lecture pandas
        try:
            df = pd.read_csv(uploaded_file, sep=separator)
        except UnicodeDecodeError:
            print("Échec de la lecture en UTF-8, tentative en Latin-1...") # Logs de débogage
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=separator, encoding='latin-1')
            except Exception as enc_err:
                 print(f"Impossible de lire le fichier CSV avec les encodages UTF-8 ou Latin-1. Erreur: {enc_err}") # Logs de débogage
                 # Afficher l'erreur dans l'UI sera géré dans app.py
                 return None
        except Exception as read_err:
            print(f"Erreur lors de la lecture du fichier CSV avec pandas : {read_err}") # Logs de débogage
            # Afficher l'erreur dans l'UI sera géré dans app.py
            return None

        if df.empty:
            print("Le fichier CSV est vide ou n'a pas pu être lu correctement par Pandas.") # Logs de débogage
            # Afficher l'erreur dans l'UI sera géré dans app.py
            return None

        # print("Colonnes détectées dans le fichier :", list(df.columns)) # Logs de débogage

        # --- Validation des colonnes mappées ---
        missing_in_file = []
        for standard_name, user_name in column_mapping.items():
            if user_name and user_name not in df.columns:
                 missing_in_file.append(user_name)

        if missing_in_file:
             print(f"Les colonnes mappées suivantes n'existent pas dans le fichier : {', '.join(missing_in_file)}") # Logs de débogage
             # Afficher l'erreur dans l'UI sera géré dans app.py
             return None


        # --- Standardisation et Conversion ---
        df_standardized = pd.DataFrame()

        # Conversion de la colonne Date
        date_col_name = column_mapping['Date']
        try:
            # Tenter d'abord la conversion automatique
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce', infer_datetime_format=True)

            # Si toutes les dates sont NaT (conversion automatique échouée) et qu'un format explicite est donné
            if df_standardized['Date'].isnull().all() and date_format:
                 print(f"Conversion automatique de date échouée. Tentative avec le format explicite : {date_format}") # Logs de débogage
                 try:
                     # Utiliser une copie pour éviter SettingWithCopyWarning si df original est une vue
                     df_copy_for_date = df[[date_col_name]].copy()
                     df_standardized['Date'] = pd.to_datetime(df_copy_for_date[date_col_name], format=date_format, errors='coerce')
                 except Exception as fmt_e:
                     print(f"Erreur application format date '{date_format}' à '{date_col_name}': {fmt_e}") # Logs de débogage
                     # Afficher l'erreur dans l'UI sera géré dans app.py
                     return None
        except Exception as e:
            print(f"Erreur générale conversion colonne Date ('{date_col_name}'): {e}") # Logs de débogage
            # Afficher l'erreur dans l'UI sera géré dans app.py
            return None

        # Vérifier si la conversion de date a réussi pour au moins une ligne
        if df_standardized['Date'].isnull().all():
             print(f"Impossible de convertir la colonne Date ('{date_col_name}') en dates valides pour toutes les lignes.") # Logs de débogage
             # Afficher l'erreur dans l'UI sera géré dans app.py
             return None

        # Gérer les valeurs NaN créées par errors='coerce' dans la colonne Date
        nan_dates_count = df_standardized['Date'].isnull().sum()
        if nan_dates_count > 0:
            print(f"{nan_dates_count} valeur(s) Date ('{date_col_name}') invalides ou vides trouvées. Lignes correspondantes seront supprimées.") # Logs de débogage
            df_standardized = df_standardized.dropna(subset=['Date'])
            if df_standardized.empty:
                 print("Toutes les lignes supprimées après échec conversion dates.") # Logs de débogage
                 # Afficher l'erreur dans l'UI sera géré dans app.py
                 return None


        # Conversion des colonnes numériques (Open, High, Low, Close, Volume)
        standard_to_user_map = {
            'Ouverture': column_mapping.get('Open'),
            'Plus_Haut': column_mapping.get('High'),
            'Plus_Bas': column_mapping.get('Low'),
            'Prix': column_mapping.get('Close'), # Utilisez 'Prix' pour la colonne Close standardisée
            'Volume': column_mapping.get('Volume')
        }
        # Filtrer les entrées où le nom utilisateur est vide (colonne non mappée)
        standard_to_user_map = {k: v for k, v in standard_to_user_map.items() if v}

        for standard_col_name, user_col_name in standard_to_user_map.items():
            try:
                # Gérer les virgules comme séparateur décimal et les espaces
                if df[user_col_name].dtype == 'object':
                     # Nettoyer la série : enlever les espaces, remplacer virgule par point
                     cleaned_series = df[user_col_name].astype(str).str.strip().str.replace(',', '.', regex=False).str.replace(r'\s+', '', regex=True)
                     # Tenter la conversion numérique
                     converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                     # Fallback plus agressif si la première tentative échoue (e.g., caractères non numériques restants)
                     if converted_series.isnull().all() and not df[user_col_name].isnull().all():
                          print(f"Conversion numérique simple de '{user_col_name}' échouée, tentative de nettoyage agressif...") # Logs de débogage
                          # Enlever tout sauf chiffres, point, tiret (pour les nombres négatifs)
                          cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                          # Gérer les cas où il ne reste que . ou -.
                          cleaned_series = cleaned_series.str.replace(r'^(-?\.)?$', '', regex=True)
                          # Gérer les multiples tirets (garder le premier si présent, ex: --5 -> -5)
                          cleaned_series = cleaned_series.str.replace(r'(-.*)-', r'\1', regex=True)
                          converted_series = pd.to_numeric(cleaned_series, errors='coerce')

                     df_standardized[standard_col_name] = converted_series
                else:
                     # Si la colonne est déjà numérique, tenter la conversion directe avec coerce
                     df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')

                nan_after_conversion = df_standardized[standard_col_name].isnull().sum()
                if nan_after_conversion > 0:
                     print(f"{nan_after_conversion} NaN créés dans '{user_col_name}' ({standard_col_name}) lors de la conversion numérique.") # Logs de débogage

            except KeyError:
                 # Cette colonne n'était pas mappée, on l'ignore.
                 pass
            except Exception as e:
                print(f"Erreur conversion numérique colonne '{user_col_name}' ({standard_col_name}) : {e}") # Logs de débogage
                # Afficher l'erreur dans l'UI sera géré dans app.py
                return None

        # --- Traitements Finaux ---
        # Trier par date (l'index est déjà la date après set_index)
        df_standardized = df_standardized.sort_index()

        # Gérer les duplicatas d'index (dates) - Conserver la dernière entrée
        if df_standardized.index.duplicated().any():
            duplicates_count = df_standardized.index.duplicated().sum()
            print(f"Il y a {duplicates_count} dates dupliquées dans vos données. Seule la dernière entrée pour chaque date sera conservée.") # Logs de débogage
            df_standardized = df_standardized[~df_standardized.index.duplicated(keep='last')]


        # Remplir les valeurs NaN restantes dans les colonnes numériques (méthode ffill puis bfill)
        # Appliquer uniquement aux colonnes qui existent dans le DataFrame standardisé
        cols_to_fill = [col for col in ['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume'] if col in df_standardized.columns]

        for col in cols_to_fill:
             nan_before = df_standardized[col].isnull().sum()
             if nan_before > 0:
                 df_standardized[col] = df_standardized[col].ffill() # Remplir avant
                 df_standardized[col] = df_standardized[col].bfill() # Remplir arrière (pour les NaN au début)
                 nan_after = df_standardized[col].isnull().sum()
                 if nan_after < nan_before:
                     print(f"{nan_before - nan_after} NaN dans '{col}' remplis par ffill/bfill.") # Logs de débogage
                 if nan_after > 0:
                      print(f"Attention: Il reste {nan_after} NaN dans la colonne '{col}' après ffill/bfill. Vérifiez vos données source, surtout au début et à la fin de la série.") # Logs de débogage
                      # On pourrait choisir de retourner None ici si on ne veut aucun NaN final

        # Vérifier si la colonne 'Prix' (Close standardisée) est présente et n'est pas entièrement NaN après remplissage
        if 'Prix' not in df_standardized.columns or df_standardized['Prix'].isnull().all():
             print("Erreur critique: La colonne 'Close' ('Prix' standardisé) est manquante ou contient uniquement des NaN après traitement.") # Logs de débogage
             # Afficher l'erreur dans l'UI sera géré dans app.py
             return None


        print("Chargement et traitement des données réussis.") # Logs de débogage
        print(f"DataFrame standardisé final shape: {df_standardized.shape}") # Logs de débogage
        # print(f"DataFrame standardisé final head:\n{df_standardized.head()}") # Logs de débogage


        return df_standardized[['Ouverture', 'Plus_Haut', 'Plus_Bas', 'Prix', 'Volume']] # Retourner les colonnes standardisées

    except Exception as e:
        # Capturer toute autre erreur inattendue
        print(f"Une erreur inattendue s'est produite pendant le traitement : {e}") # Logs de débogage
        # Afficher l'erreur dans l'UI sera géré dans app.py
        return None

# data/__init__.py reste vide
