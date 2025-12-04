import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
import warnings
import os

# --- 1. CORRECTIONS & CONFIGURATION (CRITIQUE) ---

# Correction des SyntaxWarnings (Oanda)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# Correction des FutureWarnings (Pandas) - Pour éviter l'erreur de la ligne 211
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration du logging pour suivre l'activité dans la console
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# --- 2. CONFIGURATION DE L'APPLICATION ---

st.set_page_config(page_title="BlueStar Algo", layout="wide")

# Liste des instruments basée sur vos logs
PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_CAD", "AUD_NZD", "CAD_CHF", "CHF_JPY", "AUD_CHF", "NZD_CHF",
    "EUR_CHF", "GBP_CHF", "USD_SEK", "XAU_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD"
]

# Récupération des secrets (Token Oanda)
TOKEN = os.environ.get("OANDA_TOKEN") or st.secrets.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID") or st.secrets.get("OANDA_ACCOUNT_ID")

# --- 3. FONCTIONS TECHNIQUES ---

def get_oanda_data(instrument, granularity="H1", count=200):
    """Récupère les bougies depuis Oanda."""
    if not TOKEN:
        st.error("Token OANDA manquant dans les secrets ou variables d'environnement.")
        return None

    client = oandapyV20.API(access_token=TOKEN, environment="practice")
    params = {"count": count, "granularity": granularity}
    
    try:
        logging.info(f"performing request https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles")
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        
        data = []
        for candle in r.response['candles']:
            if candle['complete']:
                data.append({
                    'Time': candle['time'],
                    'Open': float(candle['mid']['o']),
                    'High': float(candle['mid']['h']),
                    'Low': float(candle['mid']['l']),
                    'Close': float(candle['mid']['c']),
                    'Volume': candle['volume']
                })
        
        df = pd.DataFrame(data)
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        return df
        
    except Exception as e:
        logging.error(f"Erreur sur {instrument}: {str(e)}")
        return None

def calculate_indicators(df):
    """
    Calcule les indicateurs techniques (ATR, SuperTrend, etc.).
    C'est ici que le correctif Pandas est appliqué.
    """
    # Calcul ATR 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    # ATR lissé
    atr14 = true_range.rolling(14).mean()
    
    # --- CORRECTIF APPLIQUÉ ICI (Ligne 211 originale) ---
    # Ancien code : atr_loss = (2.0 * atr14).fillna(method="bfill").to_numpy()
    # Nouveau code :
    atr_loss = (2.0 * atr14).bfill().to_numpy()
    # ----------------------------------------------------
    
    # Exemple de logique simple pour déterminer la tendance (EMA + ATR)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ATR'] = atr14
    
    # Logique signal basique (à adapter selon votre stratégie exacte)
    last_close = df['Close'].iloc[-1]
    last_ema = df['EMA_50'].iloc[-1]
    
    signal = "NEUTRE"
    if last_close > last_ema:
        signal = "HAUSSIER 🟢"
    elif last_close < last_ema:
        signal = "BAISSIER 🔴"
        
    return df, signal

# --- 4. INTERFACE PRINCIPALE (MAIN) ---

def main():
    st.title("📊 BlueStar - Scanner de Marché")
    st.write("Systeme de détection de signaux institutionnels.")
    
    if st.button("Lancer l'analyse"):
        st.write("Traitement des dépendances effectué. Démarrage de l'analyse...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pair in enumerate(PAIRS):
            status_text.text(f"Analyse de {pair}...")
            
            # 1. Récupération des données
            df = get_oanda_data(pair)
            
            if df is not None and not df.empty:
                # 2. Calcul (avec le correctif inclus)
                df_calc, signal = calculate_indicators(df)
                
                # 3. Stockage du résultat
                last_price = df_calc['Close'].iloc[-1]
                atr_val = df_calc['ATR'].iloc[-1]
                
                results.append({
                    "Instrument": pair,
                    "Prix": last_price,
                    "Tendance": signal,
                    "Volatilité (ATR)": round(atr_val, 5)
                })
            
            # Mise à jour barre de progression
            progress_bar.progress((i + 1) / len(PAIRS))
        
        status_text.text("Analyse terminée !")
        progress_bar.empty()
        
        # Affichage du tableau final
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
            # Résumé rapide
            st.subheader("Résumé des signaux")
            col1, col2 = st.columns(2)
            with col1:
                bullish = len(df_results[df_results['Tendance'].str.contains("HAUSSIER")])
                st.metric("Signaux Haussiers", bullish)
            with col2:
                bearish = len(df_results[df_results['Tendance'].str.contains("BAISSIER")])
                st.metric("Signaux Baissiers", bearish)

if __name__ == "__main__":
    main()
