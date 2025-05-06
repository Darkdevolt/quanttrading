# utils.py
import pandas as pd
import pdfplumber

def parse_boc_pdf(pdf_file) -> pd.DataFrame:
    """
    Extrait les tableaux d'un PDF BOC et retourne un DataFrame structuré.
    Gère les colonnes dupliquées et structure les données pertinentes.
    """
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if not table or len(table) < 2:
                    continue
                df_page = pd.DataFrame(table[1:], columns=table[0])
                # Dédoublonnage des colonnes
                cols = list(df_page.columns)
                new_cols = []
                counts = {}
                for col in cols:
                    col_str = str(col).strip()
                    if col_str in counts:
                        counts[col_str] += 1
                        new_cols.append(f"{col_str}_{counts[col_str]}")
                    else:
                        counts[col_str] = 0
                        new_cols.append(col_str)
                df_page.columns = new_cols
                tables.append(df_page)
    if not tables:
        return pd.DataFrame()
    # Concaténation robuste
    df = pd.concat(tables, ignore_index=True, sort=False)
    # Suppression colonnes complètement dupliquées
    df = df.loc[:, ~df.columns.duplicated()]

    # Normalisation des noms de colonnes
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]

    # Conversion des colonnes numériques
    if 'cours' in df.columns:
        df['cours'] = df['cours'].str.replace(',', '').astype(float)
    if 'per' in df.columns:
        df['per'] = pd.to_numeric(df['per'], errors='coerce')
    if 'dernier_dividende' in df.columns:
        df['dernier_dividende'] = pd.to_numeric(df['dernier_dividende'], errors='coerce')
    if 'bnpa' in df.columns:
        df['bnpa'] = pd.to_numeric(df['bnpa'], errors='coerce')

    # Filtrer et renommer les colonnes essentielles
    cols_keep = ['symbole', 'titre', 'cours', 'per', 'dernier_dividende', 'bnpa']
    available = [c for c in cols_keep if c in df.columns]
    df = df[available]
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
    Si BNPA présent, utilise bnpa*market_per, sinon estime BNPA = cours/PER.
    """
    # Utiliser BNPA si disponible
    bnpa = row.get('bnpa', None)
    if bnpa and not pd.isna(bnpa):
        return bnpa * market_per
    # Sinon estimer BNPA depuis PER
    per = row.get('per', None)
    cours = row.get('cours', None)
    if per and per > 0 and cours:
        return (cours / per) * market_per
    return None


def gordon_shapiro_value(row, r: float, g: float):
    """
    Calcule la valeur via le modèle de Gordon-Shapiro ajusté.
    V = D1*(1+g)/(r - g) où D1 = dernier_dividende.
    """
    d = row.get('dernier_dividende', None)
    if d is None or pd.isna(d):
        return None
    if r <= g:
        return None
    return d * (1 + g) / (r - g)
