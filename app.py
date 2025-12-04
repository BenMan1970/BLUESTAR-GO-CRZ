import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
import warnings
import os

# --- 1. CONFIGURATION & CORRECTIFS (CRITIQUE) ---

# Ignorer les avertissements de syntaxe de la librairie Oanda (pour nettoyer les logs)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# Ignorer les avertissements futurs de Pandas (pour stabiliser l'app)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration du logging (pour voir ce qui se passe dans la console noire)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

st.set_page_config(page_title="BlueStar - Institutional Signals", layout="wide")

# --- 2. PARAMÈTRES ---

# Récupération des identifiants (Secrets Streamlit ou Variables d'environnement)
ACCESS_TOKEN = os.environ.get("OANDA_TOKEN") or st.secrets.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID") or st.secrets.get("OANDA_ACCOUNT_ID")

# Liste des instruments extraite de vos logs
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY", "NZD_JPY",
    "EUR_AUD", "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_CAD", "AUD_NZD", "CAD_CHF", "CHF_JPY", "AUD_CHF", "NZD_CHF",
    "EUR_CHF", "GBP_CHF", "USD_SEK", "XAU_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD"
]

# --- 3. FONCTIONS ---

def get_oanda_data(instrument, granularity="H1", count=200):
    """Connexion à Oanda et récupération des bougies."""
    if not ACCESS_TOKEN:
        st.error("⚠️ Token OANDA manquant. Vérifiez vos secrets Streamlit.")
        return None

    try:
        # Initialisation du client (Practice)
        client = oandapyV20.API(access_token=ACCESS_TOKEN, environment="practice")
        
        params = {
            "count": count,
            "granularity": granularity
        }
        
        # Log pour le suivi (comme dans vos fichiers logs)
        logging.info(f"performing request https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles")
        
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        
        # Transformation en DataFrame
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
        logging.error(f"Erreur lors de la requête pour {instrument}: {e}")
        return None

def process_data(df):
    """
    Traitement des données et calculs techniques.
    C'est ici que le correctif 'bfill' est appliqué.
    """
    # Calcul de l'ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    # ATR sur 14 périodes
    atr14 = true_range.rolling(14).mean()
    
    # --- CORRECTIF MAJEUR (Ligne 211 originale) ---
    # Nous utilisons .bfill() au lieu de .fillna(method='bfill')
    atr_loss = (2.0 * atr14).bfill().to_numpy()
    # ----------------------------------------------
    
    df['ATR'] = atr14
    df['ATR_Loss'] = atr_loss
    
    # Indicateurs de tendance (EMA)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Logique de signal simple (Tendance + Volatilité)
    last_close = df['Close'].iloc[-1]
    last_ema50 = df['EMA_50'].iloc[-1]
    last_ema200 = df['EMA_200'].iloc[-1]
    
    signal = "NEUTRE"
    color = "grey"
    
    # Exemple de logique institutionnelle : Tendance (EMA200) + Momentum (EMA50)
    if last_close > last_ema200 and last_close > last_ema50:
        signal = "ACHAT FORT 🟢"
        color = "green"
    elif last_close < last_ema200 and last_close < last_ema50:
        signal = "VENTE FORTE 🔴"
        color = "red"
        
    return last_close, signal, color, df['ATR'].iloc[-1]

# --- 4. INTERFACE UTILISATEUR (MAIN) ---

def main():
    st.title("🏦 BlueStar - Signaux Institutionnels")
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0068c9;
        color: white;
        font-size: 20px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🔄 SCANNER LE MARCHÉ (H1)"):
        st.write("Démarrage de l'analyse des flux...")
        
        # Conteneur pour les résultats
        results_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        data_results = []

        # Boucle sur les instruments
        for i, instrument in enumerate(INSTRUMENTS):
            status_text.text(f"Analyse en cours : {instrument}...")
            
            # 1. Récupération
            df = get_oanda_data(instrument)
            
            if df is not None and not df.empty:
                # 2. Calcul (avec correctif)
                price, signal, color, atr = process_data(df)
                
                data_results.append({
                    "Paire": instrument,
                    "Prix": price,
                    "Signal": signal,
                    "Volatilité (ATR)": round(atr, 5)
                })
            
            # Mise à jour progression
            progress_bar.progress((i + 1) / len(INSTRUMENTS))

        progress_bar.empty()
        status_text.success("✅ Analyse terminée avec succès !")

        # Affichage des résultats
        if data_results:
            df_res = pd.DataFrame(data_results)
            
            # Affichage en tableau stylisé
            st.subheader("Tableau de Bord")
            st.dataframe(
                df_res.style.applymap(lambda x: 'color: green' if 'ACHAT' in str(x) else ('color: red' if 'VENTE' in str(x) else ''), subset=['Signal']),
                use_container_width=True,
                height=800
            )
            
            # Statistiques rapides
            col1, col2, col3 = st.columns(3)
            nb_buy = len(df_res[df_res['Signal'].str.contains("ACHAT")])
            nb_sell = len(df_res[df_res['Signal'].str.contains("VENTE")])
            
            col1.metric("Opportunités d'Achat", nb_buy)
            col2.metric("Opportunités de Vente", nb_sell)
            col3.metric("Total Analysé", len(INSTRUMENTS))

if __name__ == "__main__":
    main()
   
