import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
st.set_page_config(page_title="Bluestar Hybrid M15", layout="centered", page_icon="⚡")
logging.basicConfig(level=logging.WARNING)

# ==========================================
# CSS FINTECH PRO
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    
    .stApp {
        background-color: #0f1117;
        background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%);
    }
    
    .main .block-container { max-width: 900px; padding-top: 2rem; }

    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 2.5em; text-align: center;
    }
    
    .stButton>button {
        border-radius: 12px; height: 3.5em; font-weight: 700;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(37, 99, 235, 0.4); }

    .streamlit-expanderHeader {
        background-color: #1e293b !important; border: 1px solid #334155;
        border-radius: 10px; color: #f8fafc !important; padding: 1.5rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22; border: 1px solid #334155; border-top: none;
        border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;
    }
    
    .info-box {
        background: #1e293b; border: 1px solid #334155; border-radius: 8px;
        padding: 12px; margin-bottom: 8px;
    }
    
    .badge-fvg { background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    .badge-raw { background: #f59e0b; color: black; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
    
    /* DEBUG STYLE */
    .debug-row { font-size: 0.8em; color: #64748b; border-bottom: 1px solid #334155; padding: 5px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTES & ASSETS
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD", "NAS100_USD"
]

FOREX_PAIRS = [x for x in ASSETS if "_" in x and "USD_" in x or "_USD" in x or "EUR" in x or "JPY" in x or "GBP" in x]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# ==========================================
# MOTEUR API (OANDA) - VERSION DEBUG
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            # Récupération des secrets
            token = st.secrets["OANDA_ACCESS_TOKEN"]
            acc_id = st.secrets["OANDA_ACCOUNT_ID"]
            env = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            
            self.client = oandapyV20.API(access_token=token, environment=env)
            self.account_id = acc_id
            self.req_count = 0
            # Test simple de connexion
            r = instruments.InstrumentsCandles(instrument="EUR_USD", params={"count": 1, "granularity": "M15"})
            self.client.request(r)
        except Exception as e:
            st.error(f"❌ Erreur critique API OANDA: {str(e)}")
            st.error("Vérifiez vos 'Secrets' Streamlit (Token/AccountID).")
            st.stop()

    def get_candles(self, instrument, granularity, count=200):
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            self.req_count += 1
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({
                        'time': c['time'], 'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']), 'low': float(c['mid']['l']),
                        'close': float(c['mid']['c'])
                    })
            df = pd.DataFrame(data)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception as e:
            # En mode debug, on ne crash pas mais on retourne vide
            return pd.DataFrame()

# ==========================================
# INDICATEURS TECHNIQUES
# ==========================================
def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period).mean()

def calculate_hma(series, length=20):
    wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
    wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
    raw_hma = 2 * wma_half - wma_full
    return raw_hma.rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)

def detect_fvg(df):
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull, fvg_bear

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# MOTEUR RAW STRENGTH
# ==========================================
def get_raw_strength_matrix(api):
    scores = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_PAIRS:
        if "XAU" in pair or "US30" in pair or "NAS100" in pair: continue
        df = api.get_candles(pair, "D", count=2)
        if len(df) < 2: continue
        
        open_p = df['open'].iloc[-1]
        close_p = df['close'].iloc[-1]
        pct_change = ((close_p - open_p) / open_p) * 100
        try:
            base, quote = pair.split("_")
            if base in scores: scores[base] += pct_change
            if quote in scores: scores[quote] -= pct_change
        except: continue
    return scores

def get_raw_strength_score(pair, strength_matrix, direction):
    if "XAU" in pair or "US30" in pair or "NAS100" in pair:
        return {'score': 0, 'diff': 0, 'details': 'N/A'}
    try:
        base, quote = pair.split("_")
        s_base = strength_matrix.get(base, 0)
        s_quote = strength_matrix.get(quote, 0)
        diff = s_base - s_quote if direction == "BUY" else s_quote - s_base
        score = 0
        if diff > 2.0: score = 3
        elif diff > 1.0: score = 2
        elif diff > 0.5: score = 1
        return {'score': score, 'diff': diff, 'details': f"{base}({s_base:.1f}) vs {quote}({s_quote:.1f})"}
    except:
        return {'score': 0, 'diff': 0, 'details': 'Error'}

# ==========================================
# MOTEUR GPS
# ==========================================
def analyze_gps(api, symbol):
    bias = "NEUTRAL"
    score = 0
    # Check H4
    df_h4 = api.get_candles(symbol, "H4", count=100)
    if not df_h4.empty:
        sma50 = df_h4['close'].rolling(50).mean().iloc[-1]
        if df_h4['close'].iloc[-1] > sma50: score += 1
        else: score -= 1
    # Check D1
    df_d1 = api.get_candles(symbol, "D", count=100)
    if not df_d1.empty:
        sma50 = df_d1['close'].rolling(50).mean().iloc[-1]
        if df_d1['close'].iloc[-1] > sma50: score += 1
        else: score -= 1
        
    if score >= 1: bias = "BULLISH"
    elif score <= -1: bias = "BEARISH"
    return bias

# ==========================================
# SCANNER HYBRIDE (AVEC DEBUG)
# ==========================================
def run_hybrid_scan(api, min_score=6, debug_mode=False):
    signals = []
    debug_logs = []
    
    st.write("🔄 Calcul des Flux Institutionnels (Raw Strength)...")
    strength_matrix = get_raw_strength_matrix(api)
    st.success("✅ Matrice de flux calculée.")
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, symbol in enumerate(ASSETS):
        progress.progress((i + 1) / len(ASSETS))
        status.text(f"Scanning M15: {symbol}...")
        
        # Récupération données
        df = api.get_candles(symbol, "M15", count=200)
        
        if df.empty:
            debug_logs.append(f"❌ {symbol}: Aucune donnée reçue (Marché fermé ou erreur API)")
            continue
        
        if len(df) < 50:
            debug_logs.append(f"⚠️ {symbol}: Pas assez de bougies ({len(df)})")
            continue

        # Calculs
        df['hma'] = calculate_hma(df['close'], 20)
        df['atr'] = calculate_atr(df, 14)
        fvg_bull, fvg_bear = detect_fvg(df)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # GPS
        gps_bias = analyze_gps(api, symbol)
        
        # BUY Logic
        score_buy = 0
        conf_buy = []
        if curr['hma'] > prev['hma']: score_buy += 2; conf_buy.append("HMA")
        if gps_bias == "BULLISH": score_buy += 3; conf_buy.append("GPS")
        if fvg_bull.iloc[-5:].any(): score_buy += 2; conf_buy.append("FVG")
        raw_buy = get_raw_strength_score(symbol, strength_matrix, "BUY")
        score_buy += raw_buy['score']
        
        # SELL Logic
        score_sell = 0
        conf_sell = []
        if curr['hma'] < prev['hma']: score_sell += 2; conf_sell.append("HMA")
        if gps_bias == "BEARISH": score_sell += 3; conf_sell.append("GPS")
        if fvg_bear.iloc[-5:].any(): score_sell += 2; conf_sell.append("FVG")
        raw_sell = get_raw_strength_score(symbol, strength_matrix, "SELL")
        score_sell += raw_sell['score']
        
        # Selection
        final_signal = None
        current_score = 0
        
        if score_buy > score_sell:
            current_score = score_buy
            direction = "BUY"
            final_conf = conf_buy
            final_raw = raw_buy
            final_fvg = fvg_bull.iloc[-5:].any()
        else:
            current_score = score_sell
            direction = "SELL"
            final_conf = conf_sell
            final_raw = raw_sell
            final_fvg = fvg_bear.iloc[-5:].any()
            
        # Debug info
        debug_logs.append(f"ℹ️ {symbol}: Score {current_score} ({direction}) | Conf: {final_conf}")

        if current_score >= min_score:
            atr_val = curr['atr']
            sl = curr['close'] - (atr_val * 1.5) if direction == "BUY" else curr['close'] + (atr_val * 1.5)
            tp = curr['close'] + (atr_val * 2.0) if direction == "BUY" else curr['close'] - (atr_val * 2.0)
            
            signals.append({
                "symbol": symbol, "type": direction, "price": curr['close'],
                "score": current_score, "confluences": final_conf,
                "raw_details": final_raw['details'], "has_fvg": final_fvg,
                "sl": sl, "tp": tp, "time": curr['time']
            })
            
    status.empty()
    progress.empty()
    return signals, debug_logs

# ==========================================
# UI & EXECUTION
# ==========================================
st.title("🛡️ BlueStar Hybrid M15")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Mode: <b>HYBRID</b> (Sniper + Institutional)</p>", unsafe_allow_html=True)

# Barre de contrôle
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    min_score = st.slider("Score Min", 0, 10, 5, help="Mettez 0 pour tester si le marché est fermé")
with c2:
    debug_mode = st.checkbox("Mode Debug", value=True, help="Affiche les logs détaillés")
with c3:
    scan_btn = st.button("🚀 SCAN", type="primary", use_container_width=True)

if scan_btn:
    api = OandaClient()
    start = time.time()
    
    with st.spinner("Analyse en cours..."):
        results, logs = run_hybrid_scan(api, min_score, debug_mode)
        
    duration = time.time() - start
    st.markdown("---")
    
    # Affichage des logs DEBUG (utile pour comprendre pourquoi rien ne sort)
    if debug_mode:
        with st.expander("🔍 Logs du Scanner (Pourquoi c'est vide ?)", expanded=True):
            for log in logs:
                if "❌" in log: st.error(log)
                elif "ℹ️" in log and "Score 0" in log: st.caption(log)
                else: st.text(log)
    
    if not results:
        st.warning(f"Aucun signal qualifié (Score >= {min_score}).")
        st.info("💡 Conseil : Si nous sommes le week-end, mettez le 'Score Min' à 0 pour voir les dernières données.")
    else:
        st.success(f"🎯 {len(results)} Opportunités détectées")
        results_sorted = sorted(results, key=lambda x: (x['score'], x['has_fvg']), reverse=True)
        
        for sig in results_sorted:
            is_buy = sig['type'] == "BUY"
            color = "#10b981" if is_buy else "#ef4444"
            bg_grad = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
            icon = "🟢" if is_buy else "🔴"
            
            badges = ""
            if sig['has_fvg']: badges += "<span class='badge-fvg'>🏦 FVG</span> "
            if "GPS" in str(sig['confluences']): badges += "<span class='badge-raw'>🛡️ GPS</span>"
            
            with st.expander(f"{icon} {sig['symbol']}  |  Score: {sig['score']}/10  {badges}", expanded=True):
                st.markdown(f"""
                <div style="background: {bg_grad}; padding: 15px; border-radius: 8px; border: 1px solid {color}; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.5em; font-weight: 900; color: white;">{sig['type']} @ {sig['price']:.5f}</span>
                        <span style="color: rgba(255,255,255,0.8);">{sig['time'].strftime('%H:%M')} UTC</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**SL:** <span style='color: #ef4444'>{sig['sl']:.5f}</span>", unsafe_allow_html=True)
                c2.markdown(f"**TP:** <span style='color: #10b981'>{sig['tp']:.5f}</span>", unsafe_allow_html=True)
                c3.markdown(f"**Flux:** {sig['raw_details']}")
                st.caption(f"Confluences: {', '.join(sig['confluences'])}")
