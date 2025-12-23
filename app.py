import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import time
import logging
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION & STYLE "BLUESTAR"
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Ultimate", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.WARNING)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0f1117; background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%); }
    .main .block-container { max-width: 950px; padding-top: 2rem; }
    
    /* TITRE */
    h1 { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.8em; text-align: center; margin-bottom: 0.2em; }
    
    /* BOUTONS */
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em; background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); color: white; border: 1px solid rgba(255,255,255,0.1); transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4); }
    
    /* METRICS & TEXT */
    div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.8rem; }
    .streamlit-expanderHeader { background-color: #1e293b !important; border: 1px solid #334155; border-radius: 10px; color: #f8fafc !important; font-weight: 600; }
    .streamlit-expanderContent { background-color: #161b22; border: 1px solid #334155; border-top: none; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }
    
    /* BARRES DE FORCE */
    .meter-container { width: 100%; background-color: #334155; border-radius: 10px; height: 10px; margin-top: 5px; margin-bottom: 10px; }
    .meter-fill { height: 100%; border-radius: 10px; }
    
    /* BADGES */
    .badge-mtf { background: #374151; color: #e5e7eb; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; border: 1px solid #4b5563; margin-right: 4px; }
    .badge-bull { color: #10b981; border-color: #065f46; background: rgba(16, 185, 129, 0.1); }
    .badge-bear { color: #ef4444; border-color: #991b1b; background: rgba(239, 68, 68, 0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALISATION ROBUSTE (Plus d'erreurs !)
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD"
]

ALL_CROSSES = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]

# Initialisation des variables de session (Safe Init)
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'matrix_cache' not in st.session_state: st.session_state.matrix_cache = None
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = 0

# ==========================================
# 3. CLIENT API
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except: st.error("⚠️ Config API manquante (secrets.toml)"); st.stop()

    def get_candles(self, instrument: str, granularity: str, count: int) -> pd.DataFrame:
        key = f"{instrument}_{granularity}"
        # Cache simple 1 minute
        if key in st.session_state.cache: return st.session_state.cache[key]
        
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({'time': pd.to_datetime(c['time']), 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])})
            df = pd.DataFrame(data)
            if not df.empty: st.session_state.cache[key] = df
            return df
        except: return pd.DataFrame()

# ==========================================
# 4. INDICATEURS TECHNIQUES & MTF
# ==========================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_hma(series, length=20):
    wma = lambda s, l: s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/np.arange(1, l+1).sum(), raw=True)
    wma1 = wma(series, int(length/2))
    wma2 = wma(series, length)
    return wma(2*wma1 - wma2, int(np.sqrt(length)))

def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().iloc[-1]

# --- LOGIQUE MTF INSTITUTIONNELLE ---
def get_trend_score(df, tf_name):
    if df.empty or len(df) < 50: return "Range", 0
    close = df['close']
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else close.rolling(50).mean().iloc[-1]
    curr = close.iloc[-1]
    
    if curr > sma200 and ema50 > sma200: return "Bullish", 1
    if curr < sma200 and ema50 < sma200: return "Bearish", -1
    return "Range", 0

def get_mtf_analysis(api, symbol):
    # On vérifie D1, H4, H1
    trends = {}
    score = 0
    
    # Daily
    df_d = api.get_candles(symbol, "D", 250)
    t_d, s_d = get_trend_score(df_d, "D")
    trends['D'] = t_d
    score += s_d * 3 # Poids lourd
    
    # H4
    df_h4 = api.get_candles(symbol, "H4", 200)
    t_h4, s_h4 = get_trend_score(df_h4, "H4")
    trends['H4'] = t_h4
    score += s_h4 * 2
    
    # H1
    df_h1 = api.get_candles(symbol, "H1", 100)
    t_h1, s_h1 = get_trend_score(df_h1, "H1")
    trends['H1'] = t_h1
    score += s_h1 * 1
    
    return trends, score

# ==========================================
# 5. MOTEUR FONDAMENTAL (CSM + BARCHART)
# ==========================================
class CurrencyStrengthSystem:
    @staticmethod
    def calculate_matrix(api: OandaClient):
        # Utilisation du cache pour éviter la latence
        if st.session_state.matrix_cache: return st.session_state.matrix_cache

        with st.spinner("🔄 Scan du marché (Analyse des 28 paires majeures)..."):
            scores = {c: 0.0 for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']}
            details = {c: [] for c in scores.keys()} 
            
            for pair in ALL_CROSSES:
                # Analyse sur 24h (Daily Candle)
                df = api.get_candles(pair, "D", 2)
                if not df.empty and len(df) >= 2:
                    op = df['open'].iloc[-1]
                    cl = df['close'].iloc[-1]
                    pct = ((cl - op) / op) * 100
                    
                    base, quote = pair.split('_')
                    scores[base] += pct
                    scores[quote] -= pct
                    
                    details[base].append({'vs': quote, 'val': pct})
                    details[quote].append({'vs': base, 'val': -pct})
            
            # Normalisation 0-10
            vals = list(scores.values())
            if not vals: return None
            min_v, max_v = min(vals), max(vals)
            
            final = {}
            for k, v in scores.items():
                norm = ((v - min_v) / (max_v - min_v)) * 10.0 if max_v != min_v else 5.0
                final[k] = norm

            result = {'scores': final, 'details': details}
            st.session_state.matrix_cache = result
            return result

    @staticmethod
    def get_pair_analysis(matrix, base, quote):
        if not matrix: return 5.0, 5.0, 0.0, []
        s_b = matrix['scores'].get(base, 5.0)
        s_q = matrix['scores'].get(quote, 5.0)
        # Tri des détails pour le Market Map
        map_data = sorted(matrix['details'].get(base, []), key=lambda x: x['val'], reverse=True)
        return s_b, s_q, (s_b - s_q), map_data

# ==========================================
# 6. SCANNER UNIFIÉ (Tech + Fund + MTF)
# ==========================================
def run_ultimate_scanner(api, min_score, strict_mode):
    # 1. Calculer la Force Fondamentale (Matrice)
    matrix = CurrencyStrengthSystem.calculate_matrix(api)
    
    signals = []
    pbar = st.progress(0)
    
    for i, symbol in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        
        # A. TECHNIQUE (M5 Sniper)
        df = api.get_candles(symbol, "M5", 100)
        if df.empty or len(df) < 50: continue
        
        rsi = calculate_rsi(df['close']).iloc[-1]
        hma = calculate_hma(df['close']).iloc[-1]
        prev_hma = calculate_hma(df['close']).iloc[-2]
        
        sig_type = None
        if rsi > 50 and hma > prev_hma: sig_type = "BUY"
        elif rsi < 50 and hma < prev_hma: sig_type = "SELL"
        
        if not sig_type: continue # Pas de technique, on passe
        
        # B. FONDAMENTAL (CSM)
        score_fund = 0
        cs_data = {}
        is_forex = symbol in ALL_CROSSES
        
        if is_forex:
            base, quote = symbol.split('_')
            sb, sq, gap, map_d = CurrencyStrengthSystem.get_pair_analysis(matrix, base, quote)
            cs_data = {'sb': sb, 'sq': sq, 'gap': gap, 'map': map_d}
            
            # Logique Site Web : Ecart fort requis
            if sig_type == "BUY":
                if sb >= 6.0 and sq <= 4.0: score_fund = 3 # Parfait
                elif gap > 2.0: score_fund = 2 # Bon
            else:
                if sq >= 6.0 and sb <= 4.0: score_fund = 3
                elif gap < -2.0: score_fund = 2
                
            if strict_mode and score_fund < 2: continue # Rejet strict
        else:
            # Pour Or/Indices
            score_fund = 1
        
        # C. CONTEXTE (MTF)
        # On ne calcule le MTF que si le reste est bon (pour aller vite)
        mtf_trends, mtf_score = get_mtf_analysis(api, symbol)
        
        # Validation MTF
        valid_mtf = False
        if sig_type == "BUY" and mtf_score > 0: valid_mtf = True
        if sig_type == "SELL" and mtf_score < 0: valid_mtf = True
        
        if strict_mode and not valid_mtf: continue
        
        # D. SCORE FINAL
        total_score = 4 + score_fund + (abs(mtf_score)/2) # Tech(4) + Fund(3) + MTF(3)
        
        if total_score >= min_score:
            atr = calculate_atr(df)
            signals.append({
                'symbol': symbol, 'type': sig_type, 'price': df['close'].iloc[-1],
                'score': total_score, 'rsi': rsi, 
                'cs': cs_data, 'mtf': mtf_trends,
                'atr': atr
            })
            
    pbar.empty()
    return signals

# ==========================================
# 7. AFFICHAGE (CARTE)
# ==========================================
def draw_meter(label, val, color):
    width = min(100, max(0, val * 10))
    st.markdown(f"""
    <div style="font-size:0.8em;color:#cbd5e1;display:flex;justify-content:space-between;">
        <b>{label}</b> <span>{val:.1f}/10</span>
    </div>
    <div class="meter-container"><div class="meter-fill" style="width:{width}%;background:{color};"></div></div>
    """, unsafe_allow_html=True)

def display_card(s):
    is_buy = s['type'] == 'BUY'
    color = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    with st.expander(f"{s['symbol']} | {s['type']} | Score: {s['score']:.1f}/10", expanded=True):
        # Header
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border-left:5px solid {color};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.6em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:4px 8px;border-radius:4px;color:white;font-weight:bold;margin-left:10px;">{s['type']}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.3em;font-weight:bold;color:white;">{s['price']:.5f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        
        # Colonne Gauche : FORCE (CSM)
        with c1:
            st.markdown("###### 📊 Force Relative (Site Web)")
            if s['cs']:
                sb = s['cs']['sb']
                sq = s['cs']['sq']
                col_b = "#10b981" if sb >= 6 else "#ef4444" if sb <= 4 else "#94a3b8"
                col_q = "#10b981" if sq >= 6 else "#ef4444" if sq <= 4 else "#94a3b8"
                
                base, quote = s['symbol'].split('_')
                draw_meter(base, sb, col_b)
                draw_meter(quote, sq, col_q)
            else:
                st.info("N/A (Indice)")
        
        # Colonne Droite : MTF (Institutionnel)
        with c2:
            st.markdown("###### 🏛️ Tendance MTF")
            mtf = s['mtf']
            for tf, trend in mtf.items():
                css_class = "badge-mtf badge-bull" if trend == "Bullish" else "badge-mtf badge-bear" if trend == "Bearish" else "badge-mtf"
                st.markdown(f"<span class='{css_class}'>{tf}: {trend}</span>", unsafe_allow_html=True)
            
            st.markdown(f"<div style='margin-top:10px;font-size:0.8em;color:#94a3b8;'>RSI M5: {s['rsi']:.1f}</div>", unsafe_allow_html=True)

        # Bas : MARKET MAP (Barchart)
        if s['cs']:
            st.markdown("---")
            st.markdown("###### 🗺️ Market Map (Preuve de Force)")
            cols = st.columns(6)
            map_data = s['cs']['map'][:6] # Top 6
            for i, item in enumerate(map_data):
                with cols[i]:
                    val = item['val']
                    arrow = "▲" if val > 0 else "▼"
                    c_txt = "#10b981" if val > 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align:center;background:#1e293b;border-radius:4px;padding:4px;">
                        <div style="font-size:0.7em;color:#cbd5e1;">vs {item['vs']}</div>
                        <div style="color:{c_txt};font-weight:bold;">{arrow}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# 8. APP PRINCIPALE
# ==========================================
st.title("💎 Bluestar SNP3 Ultimate")
st.markdown("Le meilleur des deux mondes : Technique M5 + Force Fondamentale + MTF.")

with st.expander("⚙️ Configuration", expanded=True):
    k1, k2 = st.columns(2)
    min_sc = k1.slider("Score Min", 5.0, 10.0, 7.0)
    strict = k2.checkbox("🔥 Mode Sniper Strict", True, help="Exige une Force Fondamentale alignée ET une tendance MTF valide.")

if st.button("🚀 LANCER LE SCAN COMPLET", type="primary"):
    # Reset cache partiel pour rafraichir le scan
    st.session_state.cache = {} 
    
    api = OandaClient()
    results = run_ultimate_scanner(api, min_sc, strict)
    
    if not results:
        st.warning("Aucun setup parfait trouvé. Le marché est peut-être calme ou en range.")
    else:
        st.success(f"{len(results)} Signaux 'Sniper' Détectés")
        # Tri : Les meilleurs scores en premier
        results.sort(key=lambda x: x['score'], reverse=True)
        for sig in results:
            display_card(sig)
