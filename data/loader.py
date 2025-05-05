# data/loader.py
import pandas as pd
import io
import os
import csv

def load_and_process_data(uploaded_file, column_mapping, date_format=None):
    if uploaded_file is None:
        return None

    required_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(key in column_mapping and column_mapping[key] for key in required_keys):
        return None

    try:
        uploaded_file.seek(0)
        sample_bytes = uploaded_file.read(2048)
        uploaded_file.seek(0)

        try:
            sample_text = sample_bytes.decode('utf-8')
        except UnicodeDecodeError:
            sample_text = sample_bytes.decode('latin-1', errors='ignore')

        separator = ','
        try:
            if sample_text.strip():
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text)
                separator = dialect.delimiter
        except csv.Error:
            uploaded_file.seek(0)
            try:
                header_line_bytes = uploaded_file.readline()
                header_line = header_line_bytes.decode('utf-8', errors='ignore')
            except Exception:
                return None
            uploaded_file.seek(0)
            separator = ';' if header_line.count(';') >= header_line.count(',') else ','

        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep=separator)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=separator, encoding='latin-1')
            except Exception:
                return None

        if df.empty:
            return None

        missing_in_file = []
        for standard_name, user_name in column_mapping.items():
            if user_name and user_name not in df.columns:
                missing_in_file.append(user_name)

        if missing_in_file:
            return None

        df_standardized = pd.DataFrame()
        date_col_name = column_mapping['Date']
        
        try:
            df_standardized['Date'] = pd.to_datetime(df[date_col_name], errors='coerce')
            if df_standardized['Date'].isnull().all() and date_format:
                df_standardized['Date'] = pd.to_datetime(df[date_col_name], format=date_format, errors='coerce')
        except Exception:
            return None

        if df_standardized['Date'].isnull().all():
            return None

        nan_dates_count = df_standardized['Date'].isnull().sum()
        if nan_dates_count > 0:
            df_standardized = df_standardized.dropna(subset=['Date'])

        standard_to_user_map = {
            'Open': column_mapping.get('Open'),
            'High': column_mapping.get('High'),
            'Low': column_mapping.get('Low'),
            'Close': column_mapping.get('Close'),
            'Volume': column_mapping.get('Volume')
        }

        for standard_col_name, user_col_name in standard_to_user_map.items():
            try:
                if df[user_col_name].dtype == 'object':
                    cleaned_series = df[user_col_name].astype(str).str.replace(',', '.', regex=False)
                    df_standardized[standard_col_name] = pd.to_numeric(cleaned_series, errors='coerce')
                else:
                    df_standardized[standard_col_name] = pd.to_numeric(df[user_col_name], errors='coerce')
            except Exception:
                return None

        df_standardized.set_index('Date', inplace=True)
        df_standardized = df_standardized.sort_index()

        if df_standardized.index.duplicated().any():
            df_standardized = df_standardized[~df_standardized.index.duplicated(keep='last')]

        cols_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_fill:
            df_standardized[col] = df_standardized[col].ffill().bfill()

        if 'Close' not in df_standardized.columns or df_standardized['Close'].isnull().all():
            return None

        return df_standardized[['Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception:
        return None
