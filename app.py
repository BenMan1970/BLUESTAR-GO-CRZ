import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION & STYLE (STRICTEMENT ORIGINAL)
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 GPS", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.INFO)

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
        font-weight: 900;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
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
    
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }

    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-vol { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    
    .risk-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }
    
    .timestamp-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        color: #93c5fd;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLIENT API (AVEC VOLUME)
# ==========================================
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'signal_history' not in st.session_state: st.session_state.signal_history = {}

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except:
            st.error("⚠️ Configuration API manquante")
            st.stop()

    def get_candles(self, instrument, granularity, count):
        key = f"{instrument}_{granularity}"
        if key in st.session_state.cache:
            ts, data = st.session_state.cache[key]
            if (datetime.now() - ts).total_seconds() < 60: return data

        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({
                        'time': pd.to_datetime(c['time']),
                        'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']), 'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })
            df = pd.DataFrame(data)
            if not df.empty:
                st.session_state.cache[key] = (datetime.now(), df)
            return df
        except:
            return pd.DataFrame()

ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD"
]

# ==========================================
# 3. GPS INSTITUTIONNEL
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
    close, curr = df['close'], df['close'].iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1]
    ema50, ema21 = ema(close, 50).iloc[-1], ema(close, 21).iloc[-1]
    if curr > sma200 and ema50 > sma200 and ema21 > ema50: return "Bullish", 90
    if curr < sma200 and ema50 < sma200 and ema21 < ema50: return "Bearish", 90
    if curr < sma200 and ema50 > sma200: return "Retracement Bull", 55
    if curr > sma200 and ema50 < sma200: return "Retracement Bear", 55
    return "Range", 35

def calc_institutional_trend_4h(df):
    if len(df) < 200: return 'Range', 0
    close, curr = df['close'], df['close'].iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1]
    ema50, ema21 = ema(close, 50).iloc[-1], ema(close, 21).iloc[-1]
    if curr > sma200 and ema21 > ema50 and ema50 > sma200: return "Bullish", 80
    if curr < sma200 and ema21 < ema50 and ema50 < sma200: return "Bearish", 80
    return "Range", 40

def calc_institutional_trend_intraday(df):
    if len(df) < 50: return 'Range', 0
    close = df['close']
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    if close.iloc[-1] > ema50 and ema21 > ema50: return "Bullish", 75
    if close.iloc[-1] < ema50 and ema21 < ema50: return "Bearish", 75
    return "Range", 30

def calculate_mtf_score_gps(api, symbol, direction):
    try:
        df_d = api.get_candles(symbol, "D", 300)
        df_h4 = api.get_candles(symbol, "H4", 200)
        df_h1 = api.get_candles(symbol, "H1", 200)
        if df_d.empty: return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}}

        d_res = df_d.copy().set_index('time')
        try: df_m = d_res.resample('ME').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        except: df_m = d_res.resample('M').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
        df_w = d_res.resample('W-FRI').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()

        trends, scores = {}, {}
        trends['M'], scores['M'] = calc_institutional_trend_macro(df_m)
        trends['W'], scores['W'] = calc_institutional_trend_macro(df_w)
        trends['D'], scores['D'] = calc_institutional_trend_daily(df_d)
        trends['H4'], scores['H4'] = calc_institutional_trend_4h(df_h4)
        trends['H1'], scores['H1'] = calc_institutional_trend_intraday(df_h1)
        
        target = 'Bullish' if direction == 'BUY' else 'Bearish'
        w_score = 0
        for tf, trend in trends.items():
            weight = MTF_WEIGHTS.get(tf, 1.0)
            if trend == target: w_score += weight * (scores[tf] / 100)
                
        alignment_pct = (w_score / TOTAL_WEIGHT) * 100
        quality = 'C'
        if trends['D'] == target and trends['W'] == target: quality = 'A' if trends['M'] == target else 'B+'
        elif trends['D'] == target: quality = 'B'
        
        final_score = 1.0
        if quality == 'A': final_score = 3.0
        elif quality == 'B+': final_score = 2.5
        elif quality == 'B': final_score = 2.0
        
        return {'score': final_score, 'quality': quality, 'alignment': f"{alignment_pct:.0f}%", 'analysis': trends}
    except:
        return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}}

# ==========================================
# 4. INDICATEURS INTELLIGENTS
# ==========================================
class SmartIndicators:
    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi_ohlc4(df, period=7):
        ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        delta = ohlc4.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.ewm(alpha=1/period, adjust=False).mean() / loss.ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_hma_trend(series, length=20):
        wma1 = series.rolling(int(length/2)).mean()
        wma2 = series.rolling(length).mean()
        hma = (2 * wma1 - wma2).rolling(int(np.sqrt(length))).mean()
        if len(hma) < 2: return 0, 0
        return (1 if hma.iloc[-1] > hma.iloc[-2] else -1), hma.iloc[-1]

    @staticmethod
    def detect_institutional_fvg(df, atr):
        if len(df) < 5: return False, None
        curr_close = df['close'].iloc[-1]
        for i in range(1, 4):
            high_A, low_A = df['high'].iloc[-(i+2)], df['low'].iloc[-(i+2)]
            high_C, low_C = df['high'].iloc[-i], df['low'].iloc[-i]
            min_gap = atr * 0.3
            
            if low_C > high_A:
                if (low_C - high_A) > min_gap:
                    if curr_close > high_A: return True, "BULL"
            
            if high_C < low_A:
                if (low_A - high_C) > min_gap:
                    if curr_close < low_A: return True, "BEAR"
        return False, None

    @staticmethod
    def detect_obv_pump(df, length=20, sigma=1.5):
        try:
            change = df['close'].diff()
            vol = df['volume']
            obv_direction = np.where(change > 0, vol, np.where(change < 0, -vol, 0))
            obv = pd.Series(obv_direction).cumsum()
            obv_sma = obv.rolling(window=length).mean()
            obv_std = obv.rolling(window=length).std()
            upper = obv_sma + (sigma * obv_std)
            lower = obv_sma - (sigma * obv_std)
            curr = obv.iloc[-1]
            return (curr > upper.iloc[-1]), (curr < lower.iloc[-1])
        except:
            return False, False

# ==========================================
# 5. MATRICE FONDAMENTALE (MATHS)
# ==========================================
def calculate_math_matrix(api):
    prices, pct_changes = {}, {}
    forex_pairs = [p for p in ASSETS if "_" in p and "XAU" not in p and "US30" not in p]
    for sym in forex_pairs:
        df = api.get_candles(sym, "H1", 50)
        if not df.empty:
            prices[sym] = df['close']
            pct_changes[sym] = ((df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]) * 100
    if not prices: return None, None
    
    df_p = pd.DataFrame(prices).ffill().bfill()
    scores = {}
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    for curr in currencies:
        vals = []
        for opp in currencies:
            if curr == opp: continue
            pair_d, pair_i = f"{curr}_{opp}", f"{opp}_{curr}"
            if pair_d in df_p.columns:
                vals.append(((SmartIndicators.calculate_rsi_ohlc4(pd.DataFrame({'open': df_p[pair_d], 'high': df_p[pair_d], 'low': df_p[pair_d], 'close': df_p[pair_d]})).iloc[-1]-50)/50+1)*5)
            elif pair_i in df_p.columns:
                inv_df = pd.DataFrame({'open': 1/df_p[pair_i], 'high': 1/df_p[pair_i], 'low': 1/df_p[pair_i], 'close': 1/df_p[pair_i]})
                vals.append(((SmartIndicators.calculate_rsi_ohlc4(inv_df).iloc[-1]-50)/50+1)*5)
        scores[curr] = np.mean(vals) if vals else 5.0
    return scores, pct_changes

def get_pair_analysis_math(scores, pct_changes, base, quote):
    s_b, s_q = scores.get(base, 5.0), scores.get(quote, 5.0)
    gap = s_b - s_q
    map_data = []
    for pair, pct in pct_changes.items():
        if base in pair or quote in pair:
            if pair == f"{base}_{quote}": continue
            vs_curr = pair.replace(base, "").replace(quote, "").replace("_", "")
            map_data.append({'vs': vs_curr, 'raw': pct})
    return s_b, s_q, gap, sorted(map_data, key=lambda x: abs(x['raw']), reverse=True)[:6]

# ==========================================
# 6. INSTITUTIONAL GRADE ENHANCEMENTS
# ==========================================
def check_signal_cooldown(symbol, hours=2):
    """Évite les re-signaux trop fréquents sur le même asset"""
    now = datetime.now()
    if symbol in st.session_state.signal_history:
        last_signal = st.session_state.signal_history[symbol]
        if (now - last_signal).total_seconds() < hours * 3600:
            return False
    return True

def detect_correlation_conflict(signals, new_signal):
    """Détecte les conflits de corrélation (ex: EUR/USD BUY + GBP/USD SELL)"""
    CORRELATED_PAIRS = {
        'EUR_USD': ['GBP_USD', 'AUD_USD', 'NZD_USD'],
        'GBP_USD': ['EUR_USD', 'AUD_USD', 'NZD_USD'],
        'USD_JPY': ['USD_CHF', 'USD_CAD'],
        'XAU_USD': ['EUR_USD']
    }
    
    sym = new_signal['symbol']
    typ = new_signal['type']
    
    if sym not in CORRELATED_PAIRS:
        return False
    
    for existing in signals[-5:]:
        if existing['symbol'] in CORRELATED_PAIRS[sym]:
            if existing['type'] != typ:
                return True
    return False

def get_session_multiplier():
    """Bonus selon la session de trading active"""
    utc_hour = datetime.now(timezone.utc).hour
    
    # Tokyo (23-08 UTC) - Paires JPY
    if 23 <= utc_hour or utc_hour < 8:
        return 1.15, ['JPY']
    
    # London (07-16 UTC) - EUR, GBP
    elif 7 <= utc_hour < 16:
        return 1.20, ['EUR', 'GBP']
    
    # New York (12-21 UTC) - USD, CAD
    elif 12 <= utc_hour < 21:
        return 1.15, ['USD', 'CAD']
    
    return 1.0, []

def calculate_adaptive_risk(signal):
    """Risk Management adaptatif selon qualité du signal"""
    quality = signal['quality']
    atr = signal['price'] * (signal['atr_pct'] / 100)
    
    # High Quality = SL serré, TP large
    if quality == 'A':
        sl_mult, tp_mult = 1.2, 4.5
    elif quality == 'B+':
        sl_mult, tp_mult = 1.5, 4.0
    elif quality == 'B':
        sl_mult, tp_mult = 1.8, 3.5
    else:
        sl_mult, tp_mult = 2.2, 3.0
    
    # Ajustement selon FVG + OBV
    if signal['fvg'] and signal['obv_status']:
        sl_mult *= 0.85  # SL plus serré si smart money + volume
        tp_mult *= 1.15  # TP plus large
    
    is_buy = signal['type'] == 'BUY'
    sl = signal['price'] - (atr * sl_mult) if is_buy else signal['price'] + (atr * sl_mult)
    tp = signal['price'] + (atr * tp_mult) if is_buy else signal['price'] - (atr * tp_mult)
    
    rr = tp_mult / sl_mult
    
    return sl, tp, rr, sl_mult, tp_mult

# ==========================================
# 7. SCANNER INSTITUTIONAL GRADE
# ==========================================
def run_scan(api, min_score, strict_mode):
    scores, pct_changes = calculate_math_matrix(api)
    if not scores: return []
    
    signals = []
    bar = st.progress(0)
    session_mult, active_currencies = get_session_multiplier()
    
    for i, sym in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        
        # Cooldown check
        if not check_signal_cooldown(sym):
            continue
        
        try:
            df = api.get_candles(sym, "M5", 150)
            if df.empty: continue
            
            # OHLC4 pour RSI
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            # Indicators
            atr = SmartIndicators.calculate_atr(df)
            rsi = SmartIndicators.calculate_rsi_ohlc4(df).iloc[-1]
            trend_hma, _ = SmartIndicators.get_hma_trend(df['close'])
            fvg, fvg_type = SmartIndicators.detect_institutional_fvg(df, atr)
            obv_pump, obv_dump = SmartIndicators.detect_obv_pump(df)
            price = df['close'].iloc[-1]
            
            # ENTRY LOGIC (INTOUCHABLE)
            signal_type = None
            if trend_hma == 1:
                if rsi < 50:  # RSI sous médiane pour BUY
                    signal_type = "BUY"
            elif trend_hma == -1:
                if rsi > 50:  # RSI sur médiane pour SELL
                    signal_type = "SELL"
            
            if not signal_type: continue
            
            # SCORING MULTIPLICATIF INSTITUTIONNEL
            base_score = 5.0
            multiplier = 1.0
            
            # 1. GPS (Facteur dominant)
            mtf = calculate_mtf_score_gps(api, sym, signal_type)
            if mtf['quality'] == 'A':
                multiplier *= 1.50
            elif mtf['quality'] == 'B+':
                multiplier *= 1.35
            elif mtf['quality'] == 'B':
                multiplier *= 1.20
            else:
                multiplier *= 0.90
            
            # 2. Matrice Fondamentale
            fonda_factor = 1.0
            cs_data = {}
            if "_" in sym and "XAU" not in sym and "US30" not in sym:
                base, quote = sym.split('_')
                sb, sq, gap, map_d = get_pair_analysis_math(scores, pct_changes, base, quote)
                cs_data = {'sb': sb, 'sq': sq, 'gap': gap, 'map': map_d, 'base': base, 'quote': quote}
                
                if signal_type == "BUY":
                    if gap > 1.5: fonda_factor = 1.40
                    elif gap > 0.8: fonda_factor = 1.25
                    elif gap > 0.3: fonda_factor = 1.10
                    elif gap < -1.0: fonda_factor = 0.75
                else:
                    if gap < -1.5: fonda_factor = 1.40
                    elif gap < -0.8: fonda_factor = 1.25
                    elif gap < -0.3: fonda_factor = 1.10
                    elif gap > 1.0: fonda_factor = 0.75
            else:
                fonda_factor = 1.15  # Bonus Or/Indices
            
            multiplier *= fonda_factor
            
            # 3. RSI Sweet Spot (Zone d'entrée optimale)
            if signal_type == "BUY":
                if 30 <= rsi <= 45:
                    multiplier *= 1.25
                elif 45 < rsi <= 50:
                    multiplier *= 1.10
            else:
                if 55 <= rsi <= 70:
                    multiplier *= 1.25
                elif 50 <= rsi < 55:
                    multiplier *= 1.10
            
            # 4. Smart Money Confluence
            if fvg:
                if (signal_type == "BUY" and fvg_type == "BULL") or (signal_type == "SELL" and fvg_type == "BEAR"):
                    multiplier *= 1.30
            
            # 5. Volume Anomaly
            obv_status = None
            if signal_type == "BUY" and obv_pump:
                multiplier *= 1.25
                obv_status = "PUMP"
            elif signal_type == "SELL" and obv_dump:
                multiplier *= 1.25
                obv_status = "DUMP"
            
            # 6. Session Multiplier
            for curr in active_currencies:
                if curr in sym:
                    multiplier *= session_mult
                    break
            
            # 7. Volatility Check (Pénalité ajustée)
            atr_pct = (atr / price) * 100
            if atr_pct < 0.05:
                multiplier *= 0.85
            elif atr_pct < 0.08:
                multiplier *= 0.95
            
            # Score final
            final_score = base_score * multiplier
            final_score = min(10.0, max(0.0, final_score))
            
            # Filtre strict mode
            if strict_mode:
                if final_score < 7.0: continue
                if mtf['quality'] not in ['A', 'B+']: continue
            elif final_score < min_score:
                continue
            
            # Correlation conflict check
            if detect_correlation_conflict(signals, {'symbol': sym, 'type': signal_type}):
                continue
            
            # Temporary risk calc for signal object
            sl_temp = price - (atr*1.8) if signal_type == "BUY" else price + (atr*1.8)
            tp_temp = price + (atr*3.0) if signal_type == "BUY" else price - (atr*3.0)
            
            signal_obj = {
                'symbol': sym, 'type': signal_type, 'price': price,
                'score': final_score, 'quality': mtf['quality'],
                'atr_pct': atr_pct, 'mtf': mtf, 'cs': cs_data,
                'fvg': fvg, 'fvg_type': fvg_type, 'rsi': rsi,
                'obv_status': obv_status,
                'sl': sl_temp, 'tp': tp_temp, 'time': df['time'].iloc[-1],
                'session_boost': session_mult > 1.0
            }
            
            # Adaptive risk management
            sl_adapt, tp_adapt, rr, sl_m, tp_m = calculate_adaptive_risk(signal_obj)
            signal_obj['sl'] = sl_adapt
            signal_obj['tp'] = tp_adapt
            signal_obj['rr'] = rr
            signal_obj['sl_mult'] = sl_m
            signal_obj['tp_mult'] = tp_m
            
            signals.append(signal_obj)
            st.session_state.signal_history[sym] = datetime.now()
            
        except Exception: 
            continue
            
    bar.empty()
    return sorted(signals, key=lambda x: x['score'], reverse=True)

# ==========================================
# 8. AFFICHAGE (100% ORIGINAL)
# ==========================================
def draw_mini_meter(label, val, color):
    w = min(100, max(0, val*10))
    st.markdown(f"""
    <div style="margin-bottom:2px;font-size:0.75em;color:#cbd5e1;display:flex;justify-content:space-between;">
        <span>{label}</span><span>{val:.1f}/10</span>
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
    if sc >= 8.0: label = "💎 LEGENDARY"
    elif sc >= 7.0: label = "✅ BON"
    elif sc >= 6.0: label = "📊 CORRECT"
    else: label = "⚠️ MOYEN"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"<div class='timestamp-box'>📅 Signal: {s['time'].strftime('%d/%m %H:%M UTC')}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
                <span style="font-size:0.8em;margin-left:10px;color:#cbd5e1;">M5 Entry</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.2f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        badges = []
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        badges.append(f"<span class='badge-gps'>🛡️ GPS {s['quality']}</span>")
        badges.append(f"<span class='badge-gps' style='background:#64748b'>RSI {s['rsi']:.1f}</span>")
        if s['obv_status'] == 'PUMP': badges.append("<span class='badge-vol'>⚡ VOL PUMP</span>")
        elif s['obv_status'] == 'DUMP': badges.append("<span class='badge-vol'>⚡ VOL DUMP</span>")
        if s.get('session_boost'): badges.append("<span class='badge-vol' style='background:#6366f1'>🌍 SESSION</span>")
            
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        st.write("")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{sc:.1f}/10", label)
        c2.metric("GPS", s['mtf']['alignment'])
        c3.metric("RSI M5", f"{s['rsi']:.1f}")
        vol_status = "🔴" if s['atr_pct'] < 0.05 else "🟢"
        c4.metric("Volatilité", f"{vol_status} {s['atr_pct']:.2f}%")
        
        if s['cs']:
            st.markdown("---")
            st.markdown("### 📊 Analyse Fondamentale")
            f1, f2 = st.columns([1, 2])
            with f1:
                st.markdown("**Force Devises**")
                base, quote = s['cs']['base'], s['cs']['quote']
                sb, sq, gap = s['cs']['sb'], s['cs']['sq'], s['cs']['gap']
                cb = "#10b981" if sb >= 5.5 else "#ef4444"
                cq = "#10b981" if sq >= 5.5 else "#ef4444"
                draw_mini_meter(base, sb, cb)
                st.write("")
                draw_mini_meter(quote, sq, cq)
                st.write("")
                gap_col = "#10b981" if (is_buy and gap > 0) or (not is_buy and gap < 0) else "#ef4444"
                st.markdown(f"<div style='text-align:center;color:{gap_col};font-weight:bold;font-size:1.2em;'>⬇️ {abs(gap):.2f}</div>", unsafe_allow_html=True)
            with f2:
                st.markdown("**Market Map**")
                cols = st.columns(6)
                for i, item in enumerate(s['cs']['map'][:6]):
                    with cols[i]:
                        arrow = "▲" if item['raw'] > 0 else "▼"
                        cl = "#10b981" if item['raw'] > 0 else "#ef4444"
                        st.markdown(f"<div style='text-align:center;color:{cl};font-size:0.8em;'>{item['vs']}<br>{arrow}<br>{abs(item['raw']):.2f}%</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🛡️ GPS Multi-Timeframe")
        mtf_analysis = s['mtf']['analysis']
        timeframes = ['M', 'W', 'D', 'H4', 'H1']
        cols = st.columns(5)
        for i, tf in enumerate(timeframes):
            with cols[i]:
                trend = mtf_analysis.get(tf, 'N/A')
                color = "#10b981" if "Bull" in trend else "#ef4444" if "Bear" in trend else "#94a3b8"
                icon = "🟢" if "Bull" in trend else "🔴" if "Bear" in trend else "⚪"
                st.markdown(f"<div style='text-align:center;'><div style='font-size:0.8em;color:#94a3b8;'>{tf}</div><div style='font-size:1.5em;'>{icon}</div><div style='font-size:0.6em;color:{color};'>{trend}</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚖️ Risk Management")
        r1, r2, r3 = st.columns(3)
        r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em;'>STOP LOSS</div><div style='color:#ef4444;font-weight:bold;'>{s['sl']:.5f}</div><div style='font-size:0.7em;'>{s['sl_mult']:.1f}x ATR</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em;'>TAKE PROFIT</div><div style='color:#10b981;font-weight:bold;'>{s['tp']:.5f}</div><div style='font-size:0.7em;'>{s['tp_mult']:.1f}x ATR</div></div>", unsafe_allow_html=True)
        r3.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em;'>RISK:REWARD</div><div style='color:#3b82f6;font-weight:bold;'>1:{s['rr']:.2f}</div><div style='font-size:0.7em;'>Adaptatif</div></div>", unsafe_allow_html=True)

def main():
    st.title("💎 BLUESTAR SNP3 GPS")
    
    with st.sidebar:
        st.header("Paramètres")
        min_score = st.slider("Score Min", 5.0, 10.0, 6.0, 0.5)
        strict_mode = st.checkbox("🔥 Mode Sniper", value=True)
    
    if st.button("🚀 LANCER LE SCAN", type="primary"):
        st.session_state.cache = {}
        api = OandaClient()
        
        with st.spinner("🔍 Analyse en cours..."):
            results = run_scan(api, min_score, strict_mode)
            
        if not results:
            st.warning("⚠️ Aucune opportunité détectée.")
        else:
            st.success(f"✅ {len(results)} Signaux détectés")
            for sig in results:
                display_sig(sig)

if __name__ == "__main__":
    main()
