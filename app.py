import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timezone
import time
import logging

# ==========================================
# 1. CONFIGURATION & STYLE (HUD PRO)
# ==========================================
st.set_page_config(page_title="Bluestar Hybrid HUD", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.WARNING)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    
    .stApp {
        background-color: #0f1117;
        background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%);
    }
    .main .block-container { max-width: 950px; padding-top: 2rem; }

    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 2.2em; text-align: center; margin-bottom: 0;
    }
    
    /* BOUTONS */
    .stButton>button {
        border-radius: 10px; height: 3.5em; font-weight: 700;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(37, 99, 235, 0.5); }

    /* HUD CARD DESIGN */
    .streamlit-expanderHeader {
        background-color: #1e293b !important; border: 1px solid #334155;
        border-radius: 10px; color: #f8fafc !important; padding: 1.2rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22; border: 1px solid #334155; border-top: none;
        border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; padding: 20px;
    }
    
    /* DETAIL BOXES */
    .metric-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-label { font-size: 0.8em; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.1em; font-weight: 700; color: #f1f5f9; }
    
    .score-detail { font-size: 0.85em; color: #64748b; margin-top: 5px; font-style: italic; }
    
    /* BADGES */
    .badge-fvg { background: #7c3aed; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: #059669; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINITIONS
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
# 3. API
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.client = oandapyV20.API(access_token=st.secrets["OANDA_ACCESS_TOKEN"], environment=st.secrets.get("OANDA_ENVIRONMENT", "practice"))
        except: st.error("⚠️ Erreur API Key"); st.stop()

    def get_candles(self, instrument, granularity, count=200):
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
            df = pd.DataFrame(data)
            if not df.empty: df['time'] = pd.to_datetime(df['time'])
            return df
        except: return pd.DataFrame()

# ==========================================
# 4. INDICATEURS (AVEC RSI & PIPS)
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

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_pips(pair, price_diff):
    """Calcule la valeur en pips"""
    if "XAU" in pair or "US30" in pair or "NAS100" in pair: return price_diff # Points pour indices/or
    multiplier = 100 if "JPY" in pair else 10000
    return price_diff * multiplier

def detect_fvg(df):
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull, fvg_bear

# ==========================================
# 5. LOGIQUE HYBRIDE
# ==========================================
def get_raw_strength_matrix(api):
    scores = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_PAIRS:
        if "XAU" in pair or "US30" in pair: continue
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
    if "XAU" in pair or "US30" in pair: return {'score': 0, 'details': 'N/A'}
    try:
        base, quote = pair.split("_")
        s_base, s_quote = strength_matrix.get(base, 0), strength_matrix.get(quote, 0)
        diff = s_base - s_quote if direction == "BUY" else s_quote - s_base
        score = 3 if diff > 2.0 else (2 if diff > 1.0 else (1 if diff > 0.5 else 0))
        return {'score': score, 'details': f"{base}({s_base:.1f}) vs {quote}({s_quote:.1f})"}
    except: return {'score': 0, 'details': 'Err'}

def analyze_gps(api, symbol):
    score = 0
    for tf in ["H4", "D"]:
        df = api.get_candles(symbol, tf, count=100)
        if not df.empty:
            sma = df['close'].rolling(50).mean().iloc[-1]
            score += 1 if df['close'].iloc[-1] > sma else -1
    return "BULLISH" if score >= 1 else ("BEARISH" if score <= -1 else "NEUTRAL")

# ==========================================
# 6. SCANNER & UI
# ==========================================
def run_scan(api, min_score):
    signals = []
    matrix = get_raw_strength_matrix(api)
    bar = st.progress(0)
    
    for i, symbol in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        df = api.get_candles(symbol, "M15", count=200)
        if df.empty or len(df) < 50: continue

        df['hma'] = calculate_hma(df['close'], 20)
        df['atr'] = calculate_atr(df, 14)
        df['rsi'] = calculate_rsi(df['close'], 14)
        fvg_bull, fvg_bear = detect_fvg(df)
        curr, prev = df.iloc[-1], df.iloc[-2]
        bias = analyze_gps(api, symbol)
        
        # Scoring
        s_buy, s_sell = 0, 0
        details_buy, details_sell = [], []
        
        # BUY
        if curr['hma'] > prev['hma']: s_buy += 2; details_buy.append("HMA(+2)")
        if bias == "BULLISH": s_buy += 3; details_buy.append("GPS(+3)")
        if fvg_bull.iloc[-5:].any(): s_buy += 2; details_buy.append("FVG(+2)")
        raw_b = get_raw_strength_score(symbol, matrix, "BUY")
        s_buy += raw_b['score']; 
        if raw_b['score'] > 0: details_buy.append(f"Flux(+{raw_b['score']})")
        
        # SELL
        if curr['hma'] < prev['hma']: s_sell += 2; details_sell.append("HMA(+2)")
        if bias == "BEARISH": s_sell += 3; details_sell.append("GPS(+3)")
        if fvg_bear.iloc[-5:].any(): s_sell += 2; details_sell.append("FVG(+2)")
        raw_s = get_raw_strength_score(symbol, matrix, "SELL")
        s_sell += raw_s['score']
        if raw_s['score'] > 0: details_sell.append(f"Flux(+{raw_s['score']})")
        
        best_score = max(s_buy, s_sell)
        if best_score >= min_score:
            is_buy = s_buy > s_sell
            direction = "BUY" if is_buy else "SELL"
            
            atr = curr['atr']
            sl_dist = atr * 1.5
            tp_dist = atr * 2.0
            
            sl = curr['close'] - sl_dist if is_buy else curr['close'] + sl_dist
            tp = curr['close'] + tp_dist if is_buy else curr['close'] - tp_dist
            
            signals.append({
                "symbol": symbol, "type": direction, "price": curr['close'],
                "score": best_score, "score_breakdown": " | ".join(details_buy if is_buy else details_sell),
                "raw": raw_b['details'] if is_buy else raw_s['details'],
                "fvg": fvg_bull.iloc[-5:].any() if is_buy else fvg_bear.iloc[-5:].any(),
                "sl": sl, "tp": tp, 
                "sl_pips": get_pips(symbol, sl_dist), "tp_pips": get_pips(symbol, tp_dist),
                "rsi": curr['rsi'], "atr": atr, "time": curr['time']
            })
            
    bar.empty()
    return signals

st.title("Bluestar Hybrid HUD")
st.markdown("<div style='text-align: center; color: #64748b; margin-bottom: 2rem;'>ARCHITECTURE M15 INFALLIBLE + DATA FEED</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1: min_score = st.slider("Qualité Signal", 4, 10, 5)
with col2: st.write(""); scan_btn = st.button("SCANNER LE MARCHÉ", type="primary", use_container_width=True)

if scan_btn:
    api = OandaClient()
    with st.spinner("Analyse approfondie M15..."):
        results = run_scan(api, min_score)
    
    st.markdown("---")
    
    if results:
        results_sorted = sorted(results, key=lambda x: (x['score'], x['fvg']), reverse=True)
        st.success(f"🎯 {len(results)} Signaux Détectés")
        
        for sig in results_sorted:
            is_buy = sig['type'] == "BUY"
            color = "#10b981" if is_buy else "#ef4444"
            grad = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
            icon = "🟢" if is_buy else "🔴"
            
            # Badges
            badges_html = ""
            if sig['fvg']: badges_html += "<span class='badge-fvg'>🏦 SMART MONEY</span> "
            if "GPS" in sig['score_breakdown']: badges_html += "<span class='badge-gps'>🛡️ GPS SECURE</span>"
            
            with st.expander(f"{icon} {sig['symbol']}  |  Score {sig['score']}/10", expanded=True):
                # HEADER CARD
                st.markdown(f"""
                <div style="background: {grad}; padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.6em; font-weight: 900; color: white;">{sig['type']} @ {sig['price']:.5f}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: rgba(255,255,255,0.7); font-size: 0.9em;">{sig['time'].strftime('%H:%M')} UTC</div>
                            <div>{badges_html}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # ROW 1: RISK MANAGER (AVEC PIPS)
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"""<div class='metric-box'><div class='metric-label'>Stop Loss</div><div class='metric-value' style='color:#ef4444'>{sig['sl']:.5f}</div><div style='font-size:0.8em; color:#ef4444'>-{int(sig['sl_pips'])} pips</div></div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class='metric-box'><div class='metric-label'>Take Profit</div><div class='metric-value' style='color:#10b981'>{sig['tp']:.5f}</div><div style='font-size:0.8em; color:#10b981'>+{int(sig['tp_pips'])} pips</div></div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class='metric-box'><div class='metric-label'>Ratio R:R</div><div class='metric-value'>1:1.33</div><div style='font-size:0.8em; color:#94a3b8'>Risque Fixe</div></div>""", unsafe_allow_html=True)
                
                st.write("")
                
                # ROW 2: CONTEXTE DU MARCHÉ
                c4, c5, c6 = st.columns(3)
                rsi_col = "#ef4444" if sig['rsi'] > 70 or sig['rsi'] < 30 else "#f1f5f9"
                c4.markdown(f"""<div class='metric-box'><div class='metric-label'>RSI (14)</div><div class='metric-value' style='color:{rsi_col}'>{sig['rsi']:.1f}</div></div>""", unsafe_allow_html=True)
                c5.markdown(f"""<div class='metric-box'><div class='metric-label'>Volatilité (ATR)</div><div class='metric-value'>{sig['atr']:.5f}</div></div>""", unsafe_allow_html=True)
                c6.markdown(f"""<div class='metric-box'><div class='metric-label'>Flux Brut</div><div class='metric-value' style='font-size:0.9em'>{sig['raw']}</div></div>""", unsafe_allow_html=True)
                
                # FOOTER: SCORE BREAKDOWN
                st.markdown(f"<div style='margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; text-align: center;'><span style='color: #94a3b8; font-size: 0.8em; text-transform: uppercase;'>Composition du Score :</span><br><span style='color: white; font-weight: bold;'>{sig['score_breakdown']}</span></div>", unsafe_allow_html=True)

    else:
        st.info(f"Aucun signal ne correspond à vos critères (Score > {min_score}). Le marché est calme.")
