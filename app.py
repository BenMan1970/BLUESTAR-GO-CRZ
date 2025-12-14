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
# 1. CONFIGURATION & STYLE (FINTECH PRO)
# ==========================================
st.set_page_config(page_title="Bluestar Hybrid", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.WARNING)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    
    /* FOND & STRUCTURE */
    .stApp {
        background-color: #0f1117;
        background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%);
    }
    .main .block-container { max-width: 900px; padding-top: 2rem; }

    /* TITRES */
    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 2.2em; text-align: center; margin-bottom: 0;
    }
    .subtitle { text-align: center; color: #64748b; font-size: 0.9em; margin-bottom: 2rem; }
    
    /* BOUTONS */
    .stButton>button {
        border-radius: 12px; height: 3.5em; font-weight: 700; letter-spacing: 0.5px;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.5);
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
    }

    /* CARTES & INFO */
    .streamlit-expanderHeader {
        background-color: #1e293b !important; border: 1px solid #334155;
        border-radius: 10px; color: #f8fafc !important; padding: 1.2rem;
        transition: background 0.2s;
    }
    .streamlit-expanderHeader:hover { background-color: #263345 !important; }
    
    .streamlit-expanderContent {
        background-color: #161b22; border: 1px solid #334155; border-top: none;
        border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; padding: 20px;
    }
    
    .info-box {
        background: rgba(255,255,255,0.03); border-radius: 8px;
        padding: 10px 15px; margin-top: 10px; border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* BADGES INTELLIGENTS */
    .badge-fvg { 
        background: linear-gradient(45deg, #7c3aed, #8b5cf6); 
        color: white; padding: 3px 8px; border-radius: 6px; 
        font-size: 0.75em; font-weight: 700; border: 1px solid rgba(255,255,255,0.2);
    }
    .badge-gps { 
        background: linear-gradient(45deg, #059669, #10b981); 
        color: white; padding: 3px 8px; border-radius: 6px; 
        font-size: 0.75em; font-weight: 700; border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* SLIDERS */
    div[data-baseweb="slider"] { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINITION DES ACTIFS
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD", "NAS100_USD"
]
FOREX_PAIRS = [x for x in ASSETS if "_" in x and "USD" in x or "EUR" in x or "JPY" in x or "GBP" in x]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# ==========================================
# 3. MOTEUR API
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.client = oandapyV20.API(access_token=st.secrets["OANDA_ACCESS_TOKEN"], environment=st.secrets.get("OANDA_ENVIRONMENT", "practice"))
            self.req_count = 0
        except:
            st.error("⚠️ Erreur de connexion API. Vérifiez vos clés.")
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
                    data.append({'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])})
            df = pd.DataFrame(data)
            if not df.empty: df['time'] = pd.to_datetime(df['time'])
            return df
        except: return pd.DataFrame()

# ==========================================
# 4. CERVEAU DU ROBOT (INDICATEURS)
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
    """Détecte les Fair Value Gaps"""
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull, fvg_bear

# ==========================================
# 5. LOGIQUE HYBRIDE (GPS + RAW FLOW)
# ==========================================
def get_raw_strength_matrix(api):
    scores = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_PAIRS:
        if "XAU" in pair or "US30" in pair or "NAS100" in pair: continue
        df = api.get_candles(pair, "D", count=2)
        if len(df) < 2: continue
        pct = ((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
        try:
            base, quote = pair.split("_")
            if base in scores: scores[base] += pct
            if quote in scores: scores[quote] -= pct
        except: continue
    return scores

def get_raw_strength_score(pair, strength_matrix, direction):
    if "XAU" in pair or "US30" in pair or "NAS100" in pair: return {'score': 0, 'details': 'N/A'}
    try:
        base, quote = pair.split("_")
        s_base, s_quote = strength_matrix.get(base, 0), strength_matrix.get(quote, 0)
        diff = s_base - s_quote if direction == "BUY" else s_quote - s_base
        score = 3 if diff > 2.0 else (2 if diff > 1.0 else (1 if diff > 0.5 else 0))
        return {'score': score, 'details': f"{base}({s_base:.1f}) vs {quote}({s_quote:.1f})"}
    except: return {'score': 0, 'details': 'Err'}

def analyze_gps(api, symbol):
    """Vérification Macro Tendance"""
    score = 0
    for tf in ["H4", "D"]:
        df = api.get_candles(symbol, tf, count=100)
        if not df.empty:
            sma = df['close'].rolling(50).mean().iloc[-1]
            score += 1 if df['close'].iloc[-1] > sma else -1
    return "BULLISH" if score >= 1 else ("BEARISH" if score <= -1 else "NEUTRAL")

# ==========================================
# 6. MOTEUR DE SCAN
# ==========================================
def run_scan(api, min_score):
    signals = []
    
    # 1. Calcul du Flux Global
    status_text = st.empty()
    status_text.caption("🌊 Analyse du flux monétaire institutionnel...")
    matrix = get_raw_strength_matrix(api)
    
    # 2. Scan M15
    bar = st.progress(0)
    for i, symbol in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        status_text.caption(f"🔭 Scanning {symbol} (M15)...")
        
        df = api.get_candles(symbol, "M15", count=200)
        if df.empty or len(df) < 50: continue

        # Calculs
        df['hma'] = calculate_hma(df['close'], 20)
        df['atr'] = calculate_atr(df, 14)
        fvg_bull, fvg_bear = detect_fvg(df)
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        # GPS Bias
        bias = analyze_gps(api, symbol)
        
        # Scoring Logic
        s_buy, s_sell = 0, 0
        conf_buy, conf_sell = [], []
        
        # BUY
        if curr['hma'] > prev['hma']: s_buy += 2; conf_buy.append("HMA")
        if bias == "BULLISH": s_buy += 3; conf_buy.append("GPS")
        if fvg_bull.iloc[-5:].any(): s_buy += 2; conf_buy.append("FVG")
        raw_b = get_raw_strength_score(symbol, matrix, "BUY")
        s_buy += raw_b['score']
        
        # SELL
        if curr['hma'] < prev['hma']: s_sell += 2; conf_sell.append("HMA")
        if bias == "BEARISH": s_sell += 3; conf_sell.append("GPS")
        if fvg_bear.iloc[-5:].any(): s_sell += 2; conf_sell.append("FVG")
        raw_s = get_raw_strength_score(symbol, matrix, "SELL")
        s_sell += raw_s['score']
        
        # Selection
        best_score = max(s_buy, s_sell)
        if best_score >= min_score:
            is_buy = s_buy > s_sell
            direction = "BUY" if is_buy else "SELL"
            
            # Gestion du risque
            atr = curr['atr']
            sl = curr['close'] - (atr*1.5) if is_buy else curr['close'] + (atr*1.5)
            tp = curr['close'] + (atr*2.0) if is_buy else curr['close'] - (atr*2.0)
            
            signals.append({
                "symbol": symbol, "type": direction, "price": curr['close'],
                "score": best_score, "conf": conf_buy if is_buy else conf_sell,
                "raw": raw_b['details'] if is_buy else raw_s['details'],
                "fvg": fvg_bull.iloc[-5:].any() if is_buy else fvg_bear.iloc[-5:].any(),
                "sl": sl, "tp": tp, "time": curr['time']
            })
            
    bar.empty()
    status_text.empty()
    return signals

# ==========================================
# 7. INTERFACE UTILISATEUR
# ==========================================
st.title("Bluestar Hybrid")
st.markdown("<div class='subtitle'>ARCHITECTURE SNIPER + MOTEUR INSTITUTIONNEL</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    min_score = st.slider("Qualité Minimale du Signal", 4, 10, 6, format="%d/10")
with col2:
    st.write("") # Spacer
    scan_btn = st.button("LANCER LE SCAN", type="primary", use_container_width=True)

if scan_btn:
    api = OandaClient()
    start = time.time()
    
    with st.spinner("Synchronisation avec les marchés..."):
        results = run_scan(api, min_score)
    
    st.markdown("---")
    
    if not results:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px;">
            <h3>😴 Aucun signal High-Prob détecté</h3>
            <p style="color: #94a3b8;">Le marché ne présente pas de configuration propre (Score < {min_score}/10).</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Tri : Score > FVG > Alphabétique
        results_sorted = sorted(results, key=lambda x: (x['score'], x['fvg']), reverse=True)
        
        st.success(f"⚡ {len(results)} Opportunités Confirmées")
        
        for sig in results_sorted:
            is_buy = sig['type'] == "BUY"
            color = "#10b981" if is_buy else "#ef4444"
            grad = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
            icon = "🟢" if is_buy else "🔴"
            
            # Badges
            badges_html = ""
            if sig['fvg']: badges_html += "<span class='badge-fvg'>🏦 SMART MONEY</span> "
            if "GPS" in sig['conf']: badges_html += "<span class='badge-gps'>🛡️ GPS SECURE</span>"
            
            with st.expander(f"{icon} {sig['symbol']}  |  Score {sig['score']}/10", expanded=True):
                # Header Card
                st.markdown(f"""
                <div style="background: {grad}; padding: 18px; border-radius: 8px; border-left: 5px solid {color}; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.6em; font-weight: 900; color: white; letter-spacing: 1px;">{sig['type']}</span>
                            <div style="color: rgba(255,255,255,0.9); font-weight: 500;">@ {sig['price']:.5f}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.9em; color: rgba(255,255,255,0.7);">{sig['time'].strftime('%H:%M')} UTC</div>
                            <div style="margin-top: 5px;">{badges_html}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Management
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div style='text-align: center'><small style='color:#64748b'>STOP LOSS</small><br><strong style='color:#ef4444; font-size:1.1em'>{sig['sl']:.5f}</strong></div>", unsafe_allow_html=True)
                c2.markdown(f"<div style='text-align: center'><small style='color:#64748b'>TAKE PROFIT</small><br><strong style='color:#10b981; font-size:1.1em'>{sig['tp']:.5f}</strong></div>", unsafe_allow_html=True)
                c3.markdown(f"<div style='text-align: center'><small style='color:#64748b'>RR RATIO</small><br><strong style='color:#e2e8f0; font-size:1.1em'>1:1.3</strong></div>", unsafe_allow_html=True)
                
                # Technical Details
                st.markdown(f"""
                <div class='info-box'>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9em;">
                        <span>🌊 <b>Flux:</b> {sig['raw']}</span>
                        <span>🧩 <b>Confluences:</b> {', '.join(sig['conf'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
