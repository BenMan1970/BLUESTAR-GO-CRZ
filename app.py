import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import os
import warnings
import logging

# --- 1. CONFIGURATION & NETTOYAGE ---

# Ignorer les warnings spécifiques (SyntaxWarning de Oanda et FutureWarning de Pandas)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

st.set_page_config(page_title="BlueStar Institutional Signals", layout="wide", page_icon="📈")

# --- 2. PARAMÈTRES DE TRADING (Institutionnel) ---

INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

TIMEFRAME = "H1"  # H1 ou H4 sont préférés pour les signaux institutionnels
CANDLE_COUNT = 300

# Paramètres de la stratégie
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # Pour le SuperTrend / Stop Loss

# Récupération des clés API (Assurez-vous qu'elles sont dans les secrets Streamlit ou env)
# Dans Streamlit Cloud, configurez cela dans les "Secrets"
ACCESS_TOKEN = os.environ.get("OANDA_TOKEN") or st.secrets.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID") or st.secrets.get("OANDA_ACCOUNT_ID")

# --- 3. FONCTIONS CŒUR ---

@st.cache_resource
def get_api_client():
    if not ACCESS_TOKEN:
        st.error("Token OANDA manquant !")
        return None
    return oandapyV20.API(access_token=ACCESS_TOKEN, environment="practice") # changez 'practice' en 'live' si besoin

def fetch_data(client, instrument, granularity, count):
    """Récupère les données brutes et retourne un DataFrame propre."""
    params = {"count": count, "granularity": granularity}
    try:
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
        logging.error(f"Erreur fetch {instrument}: {e}")
        return None

def calculate_institutional_indicators(df):
    """
    Calcule les indicateurs techniques avancés.
    Utilise Pandas pur pour éviter les dépendances lourdes.
    """
    # 1. Tendance (EMAs)
    df['EMA_50'] = df['Close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # 2. Momentum (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Volatilité (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(RSI_PERIOD).mean()
    
    # 4. SuperTrend (Logique simplifiée pour stop dynamique)
    # Correction du warning: on utilise .bfill() au lieu de fillna(method='bfill')
    hl2 = (df['High'] + df['Low']) / 2
    df['Basic_UB'] = hl2 + (ATR_MULTIPLIER * df['ATR'])
    df['Basic_LB'] = hl2 - (ATR_MULTIPLIER * df['ATR'])
    
    # Nettoyage des NaN initiaux
    df.bfill(inplace=True)
    
    return df

def generate_signal(df):
    """
    Génère un signal basé sur la CONFLUENCE.
    Un signal institutionnel nécessite plusieurs validations.
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # Condition 1: Tendance de Fond (EMA 200)
    trend_bullish = last_row['Close'] > last_row['EMA_200']
    trend_bearish = last_row['Close'] < last_row['EMA_200']
    
    # Condition 2: Croisement EMA (Golden Cross / Death Cross dynamique)
    ema_bullish = last_row['EMA_50'] > last_row['EMA_200']
    
    # Condition 3: RSI (Pas de surachat/survente extrêmes pour entrer)
    rsi_buy_zone = 40 < last_row['RSI'] < 70
    rsi_sell_zone = 30 < last_row['RSI'] < 60
    
    signal = "NEUTRAL"
    score = 0
    stop_loss = 0.0
    take_profit = 0.0
    
    # Logique d'ACHAT (BUY)
    if trend_bullish and ema_bullish and rsi_buy_zone:
        # On vérifie si le prix est au-dessus de l'EMA 50 (pullback supporté)
        if last_row['Close'] > last_row['EMA_50']:
            signal = "STRONG BUY"
            score = 90
            stop_loss = last_row['Basic_LB'] # Stop Loss basé sur volatilité
            take_profit = last_row['Close'] + (last_row['Close'] - stop_loss) * 1.5 # Ratio 1.5:1
    
    # Logique de VENTE (SELL)
    elif trend_bearish and not ema_bullish and rsi_sell_zone:
        if last_row['Close'] < last_row['EMA_50']:
            signal = "STRONG SELL"
            score = 90
            stop_loss = last_row['Basic_UB']
            take_profit = last_row['Close'] - (stop_loss - last_row['Close']) * 1.5

    return signal, score, stop_loss, take_profit, last_row

# --- 4. INTERFACE UTILISATEUR ---

def main():
    st.title("🏦 BlueStar Institutional Dashboard")
    st.markdown(f"**Environment:** Practice | **Timeframe:** {TIMEFRAME} | **Strategy:** Trend + Momentum + Volatility")
    
    if st.button("🔄 Scanner le Marché"):
        client = get_api_client()
        if not client:
            return

        # Création des colonnes pour l'affichage
        cols = st.columns(3)
        col_idx = 0
        
        status_placeholder = st.empty()
        results = []

        for instrument in INSTRUMENTS:
            status_placeholder.text(f"Analyse de {instrument}...")
            
            df = fetch_data(client, instrument, TIMEFRAME, CANDLE_COUNT)
            
            if df is not None:
                df = calculate_institutional_indicators(df)
                sig, score, sl, tp, row = generate_signal(df)
                
                # On ne garde que les signaux intéressants pour l'affichage prioritaire
                if "BUY" in sig or "SELL" in sig:
                    results.append({
                        "Symbol": instrument,
                        "Signal": sig,
                        "Price": row['Close'],
                        "RSI": round(row['RSI'], 1),
                        "Trend": "Bullish 🟢" if row['Close'] > row['EMA_200'] else "Bearish 🔴",
                        "SL": round(sl, 4) if "JPY" not in instrument else round(sl, 2),
                        "TP": round(tp, 4) if "JPY" not in instrument else round(tp, 2)
                    })
        
        status_placeholder.success("Scan terminé !")
        
        # Affichage des résultats en tableau
        if results:
            st.subheader("🎯 Signaux Détectés")
            res_df = pd.DataFrame(results)
            
            # Styling du dataframe
            def color_signal(val):
                color = 'green' if 'BUY' in val else 'red' if 'SELL' in val else 'grey'
                return f'color: {color}; font-weight: bold'
            
            st.dataframe(res_df.style.applymap(color_signal, subset=['Signal']), use_container_width=True)
            
            # Affichage détaillé sous forme de cartes
            st.subheader("Détails des Opportunités")
            for item in results:
                c_signal = "🟢 ACHAT" if "BUY" in item['Signal'] else "🔴 VENTE"
                with st.expander(f"{item['Symbol']} - {c_signal} (RSI: {item['RSI']})", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prix Entrée", item['Price'])
                    c2.metric("Stop Loss (ATR)", item['SL'], delta=f"{round(item['Price'] - item['SL'], 4)}")
                    c3.metric("Take Profit (1.5R)", item['TP'])
                    st.caption(f"Tendance de fond: {item['Trend']} | Volatilité gérée par ATR.")
        else:
            st.info("Aucun signal fort détecté pour le moment. Le marché est peut-être en consolidation.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info("Le système utilise une EMA 200 pour la tendance de fond et une EMA 50 pour le momentum. L'entrée est validée par le RSI et le Stop Loss est calculé dynamiquement via l'ATR.")
        st.write("---")
        st.write("Developed for BlueStar")

if __name__ == "__main__":
    main()
