import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import time
import logging
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION & DESIGN ORIGINAL (BLUESTAR)
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 GPS", layout="centered", page_icon="💎")
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

    /* TITRE */
    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    /* BOUTONS */
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* CARTES & EXPANDEUR */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #f8fafc !important;
        padding: 1.5rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22;
        border: 1px solid #334155;
        border-top: none;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        padding: 20px;
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }

    /* BADGES */
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    
    /* RISK BOX */
    .risk-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }

    /* JAUGES (METER) */
    .meter-container { width: 100%; background-color: #334155; border-radius: 10px; height: 8px; margin-top: 5px; }
    .meter-fill { height: 100%; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DONNÉES & CACHE (ROBUSTE)
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

# Initialisation sûre du cache
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'matrix_cache' not in st.session_state: st.session_state.matrix_cache = None

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
        except: st.error("⚠️ Config API manquante"); st.stop()

    def get_candles(self, instrument: str, granularity: str, count: int) -> pd.DataFrame:
        key = f"{instrument}_{granularity}"
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
# 4. LE "GPS" (MTF INSTITUTIONNEL ORIGINEL)
# ==========================================
MTF_WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0, 'H4': 2.5, 'H1': 1.5}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def sma_local(series, length): return series.rolling(window=length).mean()

def calc_institutional_trend_macro(df):
    if len(df) < 50: return 'Range', 0
    close = df['close']
    curr = close.iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1] if len(df) >= 200 else sma_local(close, 50).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    
    if curr > sma200 and ema50 > sma200: return "Bullish", 85
    if curr < sma200 and ema50 < sma200: return "Bearish", 85
    if curr > sma200: return "Bullish", 65
    if curr < sma200: return "Bearish", 65
    return "Range", 40

def calc_institutional_trend_daily(df):
    if len(df) < 200: return 'Range', 0
    close = df['close']
    curr = close.iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    
    if curr > sma200 and ema50 > sma200 and ema21 > ema50 and curr > ema21: return "Bullish", 90
    if curr < sma200 and ema50 < sma200 and ema21 < ema50 and curr < ema21: return "Bearish", 90
    
    if curr < sma200 and ema50 > sma200: return "Retracement Bull", 55
    if curr > sma200 and ema50 < sma200: return "Retracement Bear", 55
    
    if curr > sma200: return "Bullish", 50
    if curr < sma200: return "Bearish", 50
    return "Range", 35

def calc_institutional_trend_4h(df):
    if len(df) < 200: return 'Range', 0
    close = df['close']
    curr = close.iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    
    if curr > sma200 and ema21 > ema50 and ema50 > sma200: return "Bullish", 80
    if curr < sma200 and ema21 < ema50 and ema50 < sma200: return "Bearish", 80
    if curr < sma200 and ema50 > sma200: return "Retracement Bull", 50
    if curr > sma200 and ema50 < sma200: return "Retracement Bear", 50
    if curr > sma200: return "Bullish", 60
    if curr < sma200: return "Bearish", 60
    return "Range", 40

def calc_institutional_trend_intraday(df):
    if len(df) < 50: return 'Range', 0
    close = df['close']
    curr = close.iloc[-1]
    lag = 24
    src_adj = close + (close - close.shift(lag))
    zlema = src_adj.ewm(span=50, adjust=False).mean().iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    
    if curr > zlema and ema21 > ema50: return "Bullish", 75
    if curr < zlema and ema21 < ema50: return "Bearish", 75
    return "Range", 30

def calculate_mtf_score_gps(api, symbol, direction):
    # Récupération des données étendues pour l'analyse MTF
    df_d = api.get_candles(symbol, "D", count=500)
    df_h4 = api.get_candles(symbol, "H4", count=200)
    df_h1 = api.get_candles(symbol, "H1", count=200)
    
    if df_d.empty or df_h4.empty or df_h1.empty:
        return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}}

    d_res = df_d.copy()
    d_res.set_index('time', inplace=True)
    
    try:
        df_m = d_res.resample('ME').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    except:
        df_m = d_res.resample('M').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        
    df_w = d_res.resample('W-FRI').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()

    trends = {}
    scores = {}
    
    trends['M'], scores['M'] = calc_institutional_trend_macro(df_m)
    trends['W'], scores['W'] = calc_institutional_trend_macro(df_w)
    trends['D'], scores['D'] = calc_institutional_trend_daily(df_d)
    trends['H4'], scores['H4'] = calc_institutional_trend_4h(df_h4)
    trends['H1'], scores['H1'] = calc_institutional_trend_intraday(df_h1)
    
    target = 'Bullish' if direction == 'BUY' else 'Bearish'
    retrace_target = 'Retracement Bull' if direction == 'BUY' else 'Retracement Bear'
    
    w_score = 0
    for tf, trend in trends.items():
        weight = MTF_WEIGHTS.get(tf, 1.0)
        if trend == target:
            w_score += weight * (scores[tf] / 100)
        elif trend == retrace_target:
            w_score += weight * 0.3
            
    alignment_pct = (w_score / TOTAL_WEIGHT) * 100 * 2.5
    alignment_pct = min(100, alignment_pct)

    quality = 'C'
    if trends['D'] == target and trends['W'] == target:
        if trends['M'] == target: quality = 'A+' if alignment_pct > 80 else 'A'
        else: quality = 'B+'
    elif trends['D'] == target: quality = 'B'
    elif trends['D'] == retrace_target: quality = 'B-'
    
    final_score = 0
    if quality in ['A+', 'A']: final_score = 3
    elif quality in ['B+', 'B']: final_score = 2
    elif quality == 'B-': final_score = 1
    
    if trends['H4'] == target and final_score < 3:
        final_score += 0.5
        
    final_score = min(3, int(final_score))

    analysis_display = {
        'M': trends['M'], 'W': trends['W'], 'D': trends['D'], 'H4': trends['H4']
    }

    return {
        'score': final_score, 'quality': quality, 'alignment': f"{alignment_pct:.0f}%", 'analysis': analysis_display
    }

# ==========================================
# 5. MOTEUR FONDAMENTAL (FORCE 0-10 & MAP)
# ==========================================
class CurrencyStrengthSystem:
    @staticmethod
    def calculate_matrix(api: OandaClient):
        if st.session_state.matrix_cache: return st.session_state.matrix_cache

        with st.spinner("🔄 Scan du marché en cours..."):
            scores = {c: 0.0 for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']}
            details = {c: [] for c in scores.keys()} 
            
            for pair in ALL_CROSSES:
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
        map_data = sorted(matrix['details'].get(base, []), key=lambda x: x['val'], reverse=True)
        return s_b, s_q, (s_b - s_q), map_data

# ==========================================
# 6. SCANNER UNIFIÉ (GPS + CSM + TECH)
# ==========================================
def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().iloc[-1]

def get_rsi_ohlc4(df, length=7):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def get_colored_hma(df, length=20):
    src = df['close']
    wma = lambda s, l: s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/np.arange(1, l+1).sum(), raw=True)
    wma1 = wma(src, int(length / 2))
    wma2 = wma(src, length)
    hma = wma(2 * wma1 - wma2, int(np.sqrt(length)))
    trend = pd.Series(np.where(hma > hma.shift(1), 1, -1), index=df.index)
    return hma, trend

def detect_fvg(df):
    if len(df) < 5: return False
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull.iloc[-5:].any() or fvg_bear.iloc[-5:].any()

def run_scan(api, min_score, strict_mode):
    # 1. Calcul Matrice (Fondamental)
    matrix = CurrencyStrengthSystem.calculate_matrix(api)
    
    sigs = []
    pbar = st.progress(0)
    
    for i, sym in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        
        try:
            # 2. Données M5 (Sniper Entry)
            df = api.get_candles(sym, "M5", 150)
            if df.empty or len(df) < 50: continue
            
            rsi = get_rsi_ohlc4(df).iloc[-1]
            hma, trend = get_colored_hma(df)
            hma_val = trend.iloc[-1]
            fvg = detect_fvg(df)
            
            typ = None
            if rsi > 50 and hma_val == 1: typ = 'BUY'
            elif rsi < 50 and hma_val == -1: typ = 'SELL'
            
            if typ:
                sc = 0
                sc += 3 # Base technique
                
                # 3. GPS (MTF Analysis) - PRIORITAIRE
                mtf = calculate_mtf_score_gps(api, sym, typ)
                sc += mtf['score']
                
                if strict_mode and mtf['score'] < 1.5: continue # GPS doit valider
                
                # 4. Fondamental (CSM 0-10)
                cs_data = {}
                is_forex = sym in ALL_CROSSES
                valid_fund = False
                
                if is_forex:
                    base, quote = sym.split('_')
                    sb, sq, gap, map_d = CurrencyStrengthSystem.get_pair_analysis(matrix, base, quote)
                    cs_data = {'sb': sb, 'sq': sq, 'gap': gap, 'map': map_d}
                    
                    if typ == 'BUY':
                        if sb >= 5.5 and sq <= 4.5: sc += 3; valid_fund = True
                        elif gap > 0: sc += 1
                    else:
                        if sq >= 5.5 and sb <= 4.5: sc += 3; valid_fund = True
                        elif gap < 0: sc += 1
                        
                    if strict_mode and not valid_fund: continue
                else:
                    # Or/Indices : Pas de CSM
                    sc += 2
                    
                if fvg: sc += 1
                
                # 5. Risk Calculation
                atr = calculate_atr(df)
                price = df['close'].iloc[-1]
                sl_dist = atr * 2.0
                tp_dist = atr * 3.0
                sl = price - sl_dist if typ == 'BUY' else price + sl_dist
                tp = price + tp_dist if typ == 'BUY' else price - tp_dist
                
                if sc >= min_score:
                    sigs.append({
                        'symbol': sym, 'type': typ, 'price': price, 'score': sc,
                        'quality': mtf['quality'], 'atr': atr, 'mtf': mtf, 
                        'cs': cs_data, 'fvg': fvg, 'rsi': rsi, 'sl': sl, 'tp': tp, 
                        'time': df['time'].iloc[-1]
                    })
        except: continue
    pbar.empty()
    return sigs

# ==========================================
# 7. AFFICHAGE (DESIGN ORIGINAL + WIDGETS)
# ==========================================
def draw_mini_meter(label, val, color):
    # Jauge compacte pour insertion dans la carte
    w = min(100, max(0, val*10))
    st.markdown(f"""
    <div style="margin-bottom:2px;font-size:0.75em;color:#cbd5e1;display:flex;justify-content:space-between;">
        <span>{label}</span><span>{val:.1f}</span>
    </div>
    <div style="width:100%;background:#334155;height:6px;border-radius:4px;">
        <div style="width:{w}%;background:{color};height:100%;border-radius:4px;"></div>
    </div>
    """, unsafe_allow_html=True)

def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score']
    if sc >= 10: label = "💎 LEGENDARY"
    elif sc >= 8: label = "⭐ EXCELLENT"
    elif sc >= 6: label = "✅ BON"
    else: label = "⚠️ MOYEN"

    # En-tête (Carte Bleue)
    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
                <span style="font-size:0.8em;margin-left:10px;color:#cbd5e1;">M5 Entry</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Badges GPS
        badges = []
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        q_col = "#10b981" if s['quality'] in ['A+', 'A'] else "#f59e0b"
        badges.append(f"<span class='badge-gps' style='background:{q_col}'>🛡️ GPS {s['quality']}</span>")
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Total", f"{sc:.1f}/10", delta=label, delta_color="off")
        c2.metric("Alignement GPS", s['mtf']['alignment'])
        
        # Affichage RSI
        c3.metric("RSI M5", f"{s['rsi']:.1f}")
        c4.metric("ATR M5", f"{s['atr']:.4f}")
        
        # --- PARTIE FONDAMENTALE (INTEGRÉE) ---
        if s['cs']:
            st.markdown("---")
            f1, f2 = st.columns([1, 2])
            
            with f1:
                st.markdown("**Force Relative**")
                base, quote = s['symbol'].split('_')
                sb = s['cs']['sb']; sq = s['cs']['sq']
                cb = "#10b981" if sb >= 5.5 else "#ef4444"; cq = "#10b981" if sq >= 5.5 else "#ef4444"
                draw_mini_meter(base, sb, cb)
                draw_mini_meter(quote, sq, cq)
                
            with f2:
                st.markdown("**Market Map**")
                # Grille 6 items
                cols = st.columns(6)
                for i, item in enumerate(s['cs']['map'][:6]):
                    with cols[i]:
                        val = item['val']
                        arrow = "▲" if val > 0 else "▼"
                        cl = "#10b981" if val > 0 else "#ef4444"
                        st.markdown(f"<div style='text-align:center;font-size:0.7em;color:#94a3b8;'>{item['vs']}</div><div style='text-align:center;color:{cl};font-weight:bold;'>{arrow}</div>", unsafe_allow_html=True)

        # --- RISK MANAGER ---
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        sl_pip = int(abs(s['price']-s['sl'])*10000) if "JPY" not in s['symbol'] else int(abs(s['price']-s['sl'])*100)
        tp_pip = int(abs(s['price']-s['tp'])*10000) if "JPY" not in s['symbol'] else int(abs(s['price']-s['tp'])*100)
        
        r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>STOP LOSS</div><div style='color:#ef4444;font-weight:bold;font-size:1.1em'>{s['sl']:.5f}</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>TAKE PROFIT</div><div style='color:#10b981;font-weight:bold;font-size:1.1em'>{s['tp']:.5f}</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>RISK:REWARD</div><div style='color:white;font-weight:bold;font-size:1.1em'>1:1.5</div></div>", unsafe_allow_html=True)

# ==========================================
# 8. APP PRINCIPALE
# ==========================================
st.title("💎 Bluestar SNP3 GPS")
st.markdown("Scanner : GPS Institutionnel + Force Fondamentale")

with st.expander("⚙️ Paramètres", expanded=True):
    k1, k2 = st.columns(2)
    min_sc = k1.slider("Score Minimum", 5.0, 10.0, 7.0)
    strict = k2.checkbox("🔥 Mode Sniper (Strict)", True, help="Requiert un GPS > B et une Force Fondamentale alignée")

if st.button("🚀 LANCER LE SCAN", type="primary"):
    st.session_state.cache = {} 
    api = OandaClient()
    results = run_scan(api, min_sc, strict)
    
    if not results:
        st.warning("Aucune opportunité Sniper trouvée.")
    else:
        st.success(f"{len(results)} Signaux Validés")
        results.sort(key=lambda x: x['score'], reverse=True)
        for sig in results:
            display_sig(sig)
