# utils.py
import pandas as pd
import pdfplumber


def parse_boc_pdf(pdf_file) -> pd.DataFrame:
    """
    Extrait les tableaux d'un PDF BOC et retourne un DataFrame structuré.
    """
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            for table in extracted:
                # Convertir en DataFrame, en ignorant les lignes vides
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)

    # Nettoyage des noms de colonnes
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    # Conversion en colonnes standard
    # Repérer les colonnes utiles
    cols = {}
    if 'cours' in df.columns:
        # Remplacer les virgules et convertir en float
        df['cours'] = df['cours'].str.replace(',', '').astype(float)
        cols['cours'] = 'cours'
    if 'per' in df.columns:
        df['per'] = pd.to_numeric(df['per'], errors='coerce')
        cols['per'] = 'per'
    if 'dernier_dividende' in df.columns:
        df['dernier_dividende'] = pd.to_numeric(
            df['dernier_dividende'], errors='coerce')
        cols['dernier_dividende'] = 'dernier_dividende'

    # Garder uniquement symbôle, titre, cours et autres
    keep = ['symbole', 'titre', 'cours'] + [v for v in cols.values() if v not in ['symbole','titre','cours']]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={'symbole': 'symbole', 'titre': 'titre'})
    return df


def classify_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'type' : 'Propriété (Actions)' ou 'Créance (Obligations)'.
    Créance si le symbole contient un point (ex.: TPBF.O10), sinon Propriété.
    """
    df['type'] = df['symbole'].apply(
        lambda x: 'Créance (Obligations)' if '.' in str(x) else 'Propriété (Actions)'
    )
    return df


def calculate_intrinsic_value(row, market_per: float):
    """
    Calcule la valeur intrinsèque via la méthode des multiples (PER).
    Si BNPA présent, use bnpa*market_per, sinon estime BNPA = cours/PER.
    """
    # Si la colonne 'bnpa' existe
    if 'bnpa' in row and not pd.isna(row['bnpa']):
        return row['bnpa'] * market_per
    # Sinon, estimer BNPA à partir du PER
    per = row.get('per', None)
    cours = row.get('cours', None)
    if per and per > 0:
        bnpa_estime = cours / per
        return bnpa_estime * market_per
    return None


def gordon_shapiro_value(row, r: float, g: float):
    """
    Calcule la valeur via le modèle de Gordon-Shapiro ajusté.
    V = D1 / (r - g) où D1 = dernier_dividende.
    """
    d = row.get('dernier_dividende', None)
    if d is None or pd.isna(d):
        return None
    # Eviter division par zéro
    if r <= g:
        return None
    return d * (1 + g) / (r - g)
