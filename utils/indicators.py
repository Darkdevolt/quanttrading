import pandas as pd
import numpy as np
import streamlit as st

def calculate_technical_indicators(data):
    """Calcule les indicateurs techniques en fonction des paramètres de l'utilisateur."""
    analysis_data = data.copy()
    
    # Initialiser les colonnes avec NaN
    analysis_data['MM_Courte'] = np.nan
    analysis_data['MM_Longue'] = np.nan
    analysis_data['RSI'] = np.nan
    analysis_data['MACD_Line'] = np.nan
    analysis_data['MACD_Signal'] = np.nan
    analysis_data['MACD_Hist'] = np.nan
    
    try:
        # Récupérer les paramètres de session_state
        use_mm = st.session_state.get('use_mm_signal', False)
        window_court = st.session_state.get('short_ma', 20)
        window_long = st.session_state.get('long_ma', 50)
        
        use_rsi = st.session_state.get('use_rsi_signal', False)
        rsi_window = st.session_state.get('rsi_window', 14)
        
        use_macd = st.session_state.get('use_macd_signal', False)
        macd_fast_window = st.session_state.get('macd_fast_window', 12)
        macd_slow_window = st.session_state.get('macd_slow_window', 26)
        macd_signal_window = st.session_state.get('macd_signal_window', 9)
        
        # Calculer les indicateurs si activés
        if use_mm and window_long > window_court:
            analysis_data['MM_Courte'] = analysis_data['Prix'].rolling(window=window_court).mean()
            analysis_data['MM_Longue'] = analysis_data['Prix'].rolling(window=window_long).mean()
        
        if use_rsi:
            analysis_data['RSI'] = calculate_rsi(analysis_data, window=rsi_window)
        
        if use_macd and macd_slow_window > macd_fast_window:
            macd_line, signal_line, hist = calculate_macd(
                analysis_data,
                fast_window=macd_fast_window,
                slow_window=macd_slow_window,
                signal_window=macd_signal_window
            )
            analysis_data['MACD_Line'] = macd_line
            analysis_data['MACD_Signal'] = signal_line
            analysis_data['MACD_Hist'] = hist
        
        # Générer les signaux techniques
        analysis_data = generate_technical_signals(analysis_data)
        
        return analysis_data
    
    except Exception as e:
        st.error(f"Erreur lors du calcul des indicateurs techniques : {e}")
        st.error(traceback.format_exc())
        return data

def calculate_rsi(df, column='Prix', window=14):
    """Calcule le Relative Strength Index (RSI)."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(100)
    
    return rsi

def calculate_macd(df, column='Prix', fast_window=12, slow_window=26, signal_window=9):
    """Calcule la ligne MACD, la ligne de Signal, et l'Histogramme."""
    price_series = df[column]
    fast_ema = price_series.ewm(span=fast_window, adjust=False).mean()
    slow_ema = price_series.ewm(span=slow_window, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def generate_technical_signals(df):
    """Génère les signaux techniques en fonction des indicateurs calculés."""
    # Initialiser les colonnes de signaux
    df["Signal_MM"] = 0
    df["Signal_RSI"] = 0
    df["Signal_MACD"] = 0
    df["Signal_Technique_Combine"] = 0
    
    # Récupérer les paramètres de session_state
    use_mm = st.session_state.get('use_mm_signal', False)
    window_court = st.session_state.get('short_ma', 20)
    window_long = st.session_state.get('long_ma', 50)
    
    use_rsi = st.session_state.get('use_rsi_signal', False)
    rsi_overbought = st.session_state.get('rsi_overbought', 70)
    rsi_oversold = st.session_state.get('rsi_oversold', 30)
    
    use_macd = st.session_state.get('use_macd_signal', False)
    tech_signal_method = st.session_state.get('tech_signal_method', "MM OU RSI OU MACD")
    
    # Générer les signaux individuels
    if use_mm and 'MM_Courte' in df.columns and 'MM_Longue' in df.columns:
        df.loc[(df['MM_Courte'] > df['MM_Longue']) & (df['MM_Courte'].shift(1) <= df['MM_Longue'].shift(1)), 'Signal_MM'] = 1
        df.loc[(df['MM_Courte'] < df['MM_Longue']) & (df['MM_Courte'].shift(1) >= df['MM_Longue'].shift(1)), 'Signal_MM'] = -1
    
    if use_rsi and 'RSI' in df.columns:
        df.loc[(df['RSI'] > rsi_oversold) & (df['RSI'].shift(1) <= rsi_oversold), 'Signal_RSI'] = 1
        df.loc[(df['RSI'] < rsi_overbought) & (df['RSI'].shift(1) >= rsi_overbought), 'Signal_RSI'] = -1
    
    if use_macd and 'MACD_Line' in df.columns and 'MACD_Signal' in df.columns:
        df.loc[(df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1)), 'Signal_MACD'] = 1
        df.loc[(df['MACD_Line'] < df['MACD_Signal']) & (df['MACD_Line'].shift(1) >= df['MACD_Signal'].shift(1)), 'Signal_MACD'] = -1
    
    # Combiner les signaux selon la méthode choisie
    active_indicators = []
    if use_mm: active_indicators.append("MM")
    if use_rsi: active_indicators.append("RSI")
    if use_macd: active_indicators.append("MACD")
    
    if active_indicators:
        buy_signals = pd.Series(False, index=df.index)
        sell_signals = pd.Series(False, index=df.index)
        
        mm_buy = df['Signal_MM'] == 1 if use_mm else pd.Series(False, index=df.index)
        mm_sell = df['Signal_MM'] == -1 if use_mm else pd.Series(False, index=df.index)
        
        rsi_buy = df['Signal_RSI'] == 1 if use_rsi else pd.Series(False, index=df.index)
        rsi_sell = df['Signal_RSI'] == -1 if use_rsi else pd.Series(False, index=df.index)
        
        macd_buy = df['Signal_MACD'] == 1 if use_macd else pd.Series(False, index=df.index)
        macd_sell = df['Signal_MACD'] == -1 if use_macd else pd.Series(False, index=df.index)
        
        # Logique de combinaison
        if "OU" in tech_signal_method:
            buy_signals = mm_buy | rsi_buy | macd_buy
            sell_signals = mm_sell | rsi_sell | macd_sell
        elif "ET" in tech_signal_method:
            buy_signals = mm_buy & rsi_buy & macd_buy
            sell_signals = mm_sell & rsi_sell & macd_sell
        elif "Seulement" in tech_signal_method:
            if "MM" in tech_signal_method:
                buy_signals = mm_buy
                sell_signals = mm_sell
            elif "RSI" in tech_signal_method:
                buy_signals = rsi_buy
                sell_signals = rsi_sell
            elif "MACD" in tech_signal_method:
                buy_signals = macd_buy
                sell_signals = macd_sell
        
        df.loc[buy_signals, 'Signal_Technique_Combine'] = 1
        df.loc[sell_signals, 'Signal_Technique_Combine'] = -1
    
    return df
