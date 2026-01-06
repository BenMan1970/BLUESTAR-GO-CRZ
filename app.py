import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
import time
from datetime import datetime, timedelta
from scipy import stats 

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Bluestar Ultimate V3.6", layout="centered", page_icon="💎")
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
        font-weight: 900; font-size: 2.8em; text-align: center; margin-bottom: 0.2em;
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .streamlit-expanderHeader {
        background-color: #1e293b !important; border: 1px solid #334155;
        border-radius: 10px; color: #f8fafc !important; padding: 1.5rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22; border: 1px solid #334155;
        border-top: none; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; padding: 20px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }
    .badge { color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; margin: 2px; display: inline-block; }
    .badge-regime { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
    .badge-trend { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .badge-weak { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); }
    .badge-blue { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); }
    .badge-purple { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); }
    .badge-gold { background: linear-gradient(135deg, #ca8a04 0%, #eab308 100%); }
    .badge-midnight { background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%); border: 1px solid #818cf8; }
    .risk-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }
    .timestamp-box {
        background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6;
        padding: 8px 12px; border-radius: 6px; font-size: 0.85em;
        color: #93c5fd; margin: 10px 0; font-family: 'Courier New', monospace;
    }
    .quality-indicator {
        display: inline-block; padding: 3px 8px; border-radius: 4px;
        font-size: 0.7em; font-weight: 700; margin-left: 8px;
    }
    .quality-high { background: #10b981; color: white; }
    .quality-medium { background: #f59e0b; color: white; }
    .quality-low { background: #6b7280; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CLIENT API & SESSION STATE
# ==========================================
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'signal_history' not in st.session_state: st.session_state.signal_history = {}
if 'cs_data' not in st.session_state: st.session_state.cs_data = {'data': None, 'time': None}

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception as e:
            st.error(f"⚠️ Configuration API manquante: {e}")
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
        except Exception as e:
            return pd.DataFrame()

ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD"
]

def get_asset_params(symbol):
    if "XAU" in symbol:
        return {'type': 'COMMODITY', 'atr_threshold': 0.06, 'sl_base': 1.8, 'tp_rr': 2.5}
    if "US30" in symbol or "NAS" in symbol or "SPX" in symbol:
        return {'type': 'INDEX', 'atr_threshold': 0.10, 'sl_base': 2.2, 'tp_rr': 3.0}
    return {'type': 'FOREX', 'atr_threshold': 0.035, 'sl_base': 1.5, 'tp_rr': 2.0}

# ==========================================
# MOTEUR D'INDICATEURS
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(df, period=7):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.ewm(alpha=1/period, adjust=False).mean() / loss.ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_adx(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.iloc[-1]

    @staticmethod
    def detect_structure_zscore(df, lookback=20):
        if len(df) < lookback + 1: return 0
        window = df['close'].iloc[-lookback:]
        z_score = stats.zscore(window)[-1]
        if z_score > 1.5: return 1 
        if z_score < -1.5: return -1 
        return 0 

    @staticmethod
    def detect_smart_fvg(df, atr):
        if len(df) < 4: return False, 0
        curr_close = df['close'].iloc[-1]
        min_gap = atr * 0.5
        high_1 = df['high'].iloc[-3]
        low_1 = df['low'].iloc[-3]
        high_3 = df['high'].iloc[-1]
        low_3 = df['low'].iloc[-1]
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        vol_curr = df['volume'].iloc[-1]
        
        gap_bull = low_3 - high_1
        if gap_bull > min_gap and curr_close > high_1 and vol_curr > vol_mean * 0.8: 
            return True, "BULL"
        gap_bear = low_1 - high_3
        if gap_bear > min_gap and curr_close < low_1 and vol_curr > vol_mean * 0.8:
            return True, "BEAR"
        return False, None

    @staticmethod
    def get_mtf_bias(df_d, df_w):
        def trend_score(df):
            if len(df) < 50: return 0
            sma200 = df['close'].rolling(200).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            price = df['close'].iloc[-1]
            score = 0
            if price > sma200: score += 1
            if ema50 > sma200: score += 1
            dist = abs((price - sma200) / sma200)
            if dist > 0.005: score += 1 
            return score 
        score_d = trend_score(df_d)
        score_w = trend_score(df_w)
        total = score_d + score_w
        if total >= 4: return "STRONG_BULL"
        if total >= 2: return "BULL"
        if total <= -4: return "STRONG_BEAR"
        if total <= -2: return "BEAR"
        return "NEUTRAL"

    @staticmethod
    def get_midnight_open_ny(df):
        try:
            df_utc = df.copy()
            df_utc['time'] = df_utc['time'] - pd.Timedelta(hours=5)
            midnight_candle = df_utc[df_utc['time'].dt.hour == 0]
            if not midnight_candle.empty:
                return midnight_candle.iloc[-1]['open']
            else:
                return None
        except Exception:
            return None

# ==========================================
# CURRENCY STRENGTH (CORRIGÉ & SANS WARNING)
# ==========================================
def get_currency_strength_rsi(api):
    now = datetime.now()
    if st.session_state.cs_data.get('time') and (now - st.session_state.cs_data['time']).total_seconds() < 60:
        return st.session_state.cs_data['data']

    forex_pairs = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
        "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
        "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
    ]
    
    prices = {}
    for pair in forex_pairs:
        try:
            df = api.get_candles(pair, "H1", 100)
            if df is not None and not df.empty:
                prices[pair] = df['close']
        except Exception:
            continue

    if not prices: return None

    # CORRECTIF PANDAS ICI : ffill().bfill() au lieu de fillna(method)
    df_prices = pd.DataFrame(prices).ffill().bfill()
    
    def normalize_score(rsi_value):
        return ((rsi_value - 50) / 50 + 1) * 5

    def calculate_rsi_series(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))

    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    final_scores = {}

    for curr in currencies:
        total_score = 0.0
        count = 0
        opponents = [c for c in currencies if c != curr]
        
        for opp in opponents:
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            
            rsi_val = None
            if pair_direct in df_prices.columns:
                rsi_series = calculate_rsi_series(df_prices[pair_direct])
                rsi_val = rsi_series.iloc[-1]
            elif pair_inverse in df_prices.columns:
                # Inversion mathématique pour contrepartie
                inverted_price = 1 / df_prices[pair_inverse]
                rsi_series = calculate_rsi_series(inverted_price)
                rsi_val = rsi_series.iloc[-1]
            
            if rsi_val is not None:
                total_score += normalize_score(rsi_val)
                count += 1
        
        if count > 0:
            final_scores[curr] = total_score / count
        else:
            final_scores[curr] = 5.0

    st.session_state.cs_data = {'data': final_scores, 'time': now}
    return final_scores

# ==========================================
# FILTRE CORRÉLATION
# ==========================================
def check_dynamic_correlation_conflict(new_signal, existing_signals, cs_scores):
    if not existing_signals: return False
    new_sym = new_signal['symbol']
    new_type = new_signal['type']
    if "_" not in new_sym: return False
    base, quote = new_sym.split('_')
    
    CORRELATION_MAP = {
        'EUR_USD':  { 'GBP_USD': 0.9, 'AUD_USD': 0.85, 'NZD_USD': 0.8, 'USD_CHF': -0.9 },
        'GBP_USD':  { 'EUR_USD': 0.9, 'EUR_GBP': -0.8, 'AUD_USD': 0.8 },
        'USD_JPY':  { 'EUR_JPY': 0.9, 'GBP_JPY': 0.9, 'USD_CHF': 0.7 },
        'AUD_USD':  { 'EUR_USD': 0.85, 'GBP_USD': 0.8, 'NZD_USD': 0.9 },
        'XAU_USD':  { 'USD_CHF': -0.7, 'EUR_USD': 0.6 } 
    }
    
    for existing in existing_signals:
        ex_sym = existing['symbol']
        ex_type = existing['type']
        if new_sym == ex_sym: return True 
        if new_sym in CORRELATION_MAP and ex_sym in CORRELATION_MAP[new_sym]:
            corr = CORRELATION_MAP[new_sym][ex_sym]
            if corr > 0.8 and new_type != ex_type: return True 
            if corr < -0.8 and new_type == ex_type: return True 

        shared_currency = None
        if base in ex_sym or quote in ex_sym:
            shared_currency = base if (base in ex_sym) else quote
            if cs_scores and shared_currency in cs_scores:
                cs_val = cs_scores[shared_currency]
                # Si devise faible et on veut acheter -> Danger
                if cs_val < 4.0 and new_type == "BUY" and base == shared_currency: return True
                # Si devise forte et on veut vendre -> Danger
                if cs_val > 6.0 and new_type == "SELL" and base == shared_currency: return True
    return False

# ==========================================
# PROBABILITÉ
# ==========================================
def calculate_signal_probability(df_m5, df_h4, df_d, df_w, symbol, direction):
    prob_factors = []
    weights = []
    details = {}
    
    params = get_asset_params(symbol)
    atr = QuantEngine.calculate_atr(df_m5)
    atr_pct = (atr / df_m5['close'].iloc[-1]) * 100
    
    if atr_pct < params['atr_threshold'] * 0.5:
        return 0, {}, atr_pct 
    
    vol_score = min(atr_pct / params['atr_threshold'], 2.0)
    details['vol_score'] = vol_score
    vol_conf = min(vol_score, 1.2) / 1.2 
    
    rsi_serie = QuantEngine.calculate_rsi(df_m5)
    rsi_val = rsi_serie.iloc[-1]
    rsi_mom = rsi_val - rsi_serie.iloc[-2]
    
    rsi_prob = 0
    if direction == "BUY":
        if rsi_serie.iloc[-2] < 50 and rsi_val >= 50: rsi_prob = 0.85
        elif rsi_val > 50 and rsi_mom > 0: rsi_prob = 0.60
    else:
        if rsi_serie.iloc[-2] > 50 and rsi_val <= 50: rsi_prob = 0.85
        elif rsi_val < 50 and rsi_mom < 0: rsi_prob = 0.60
            
    prob_factors.append(rsi_prob)
    weights.append(0.30)
    details['rsi_mom'] = abs(rsi_mom)
    
    z_score_struc = QuantEngine.detect_structure_zscore(df_h4, 20)
    struc_score = 0
    if direction == "BUY":
        if z_score_struc == 1: struc_score = 0.9
        elif z_score_struc == 0: struc_score = 0.5
    else:
        if z_score_struc == -1: struc_score = 0.9
        elif z_score_struc == 0: struc_score = 0.5
    prob_factors.append(struc_score)
    weights.append(0.20)
    details['structure_z'] = z_score_struc
    
    mtf_bias = QuantEngine.get_mtf_bias(df_d, df_w)
    mtf_score = 0.5
    if direction == "BUY":
        if mtf_bias == "STRONG_BULL": mtf_score = 0.95
        elif mtf_bias == "BULL": mtf_score = 0.80
        elif mtf_bias == "NEUTRAL": mtf_score = 0.50
        else: mtf_score = 0.10
    else:
        if mtf_bias == "STRONG_BEAR": mtf_score = 0.95
        elif mtf_bias == "BEAR": mtf_score = 0.80
        elif mtf_bias == "NEUTRAL": mtf_score = 0.50
        else: mtf_score = 0.10
    prob_factors.append(mtf_score)
    weights.append(0.20)
    details['mtf_bias'] = mtf_bias
    
    midnight_price = QuantEngine.get_midnight_open_ny(df_m5)
    midnight_score = 0.5 
    curr_price = df_m5['close'].iloc[-1]
    details['midnight_val'] = midnight_price
    details['midnight_status'] = "UNKNOWN"
    
    if midnight_price:
        if direction == "BUY":
            if curr_price <= midnight_price: 
                midnight_score = 1.0
                details['midnight_status'] = "OPTIMAL (Discount)"
            else:
                midnight_score = 0.5
                details['midnight_status'] = "STANDARD (Premium)"
        else:
            if curr_price >= midnight_price:
                midnight_score = 1.0
                details['midnight_status'] = "OPTIMAL (Premium)"
            else:
                midnight_score = 0.5
                details['midnight_status'] = "STANDARD (Discount)"
    
    prob_factors.append(midnight_score)
    weights.append(0.15)

    fvg_active, fvg_type = QuantEngine.detect_smart_fvg(df_m5, atr)
    details['fvg_align'] = fvg_active
    adx_val = QuantEngine.calculate_adx(df_h4)
    details['adx_val'] = adx_val
    if adx_val < 18: return 0, details, atr_pct 
    
    extra_score = 0.5
    if adx_val > 22: extra_score += 0.2
    if fvg_active and ((direction=="BUY" and fvg_type=="BULL") or (direction=="SELL" and fvg_type=="BEAR")):
        extra_score += 0.3
    prob_factors.append(min(extra_score, 1.0))
    weights.append(0.15)

    total_weight = sum(weights)
    weighted_prob = sum(p * w for p, w in zip(prob_factors, weights)) / total_weight
    final_score = max(0, weighted_prob * vol_conf)
    
    return final_score, details, atr_pct

# ==========================================
# UTILITAIRE
# ==========================================
def format_time_ago(detection_time):
    now = datetime.now()
    diff = now - detection_time
    minutes = int(diff.total_seconds() / 60)
    
    if minutes < 1: return "À l'instant"
    elif minutes < 60: return f"Il y a {minutes} min"
    else: return f"Il y a {minutes // 60}h {minutes % 60}m"

# ==========================================
# SCANNER
# ==========================================
def run_scan_v36(api, min_prob, strict_mode):
    cs_scores = get_currency_strength_rsi(api)
    signals = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sym in enumerate(ASSETS):
        progress_bar.progress((i+1)/len(ASSETS))
        status_text.markdown(f"⏳ Analyse: **{sym}** ({i+1}/{len(ASSETS)})")
        
        if sym in st.session_state.signal_history:
            if (datetime.now() - st.session_state.signal_history[sym]).total_seconds() < 300: 
                time.sleep(0.05) 
                continue
        
        try:
            df_d_raw = api.get_candles(sym, "D", 300)
            time.sleep(0.1)
            df_m5 = api.get_candles(sym, "M5", 288)
            time.sleep(0.1)
            df_h4 = api.get_candles(sym, "H4", 100)
            
            if df_m5.empty or df_h4.empty or df_d_raw.empty: continue
            
            df_d = df_d_raw.iloc[-100:].copy()
            df_w = df_d_raw.set_index('time').resample('W-FRI').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna().reset_index()
            
            rsi_serie = QuantEngine.calculate_rsi(df_m5)
            if len(rsi_serie) < 3: continue
            
            rsi_mom = rsi_serie.iloc[-1] - rsi_serie.iloc[-2]
            scan_direction = None
            
            # Logique de signal : croisement précis du RSI 50
            if rsi_serie.iloc[-2] < 50 and rsi_serie.iloc[-1] >= 50 and rsi_mom > 0.5:
                scan_direction = "BUY"
            elif rsi_serie.iloc[-2] > 50 and rsi_serie.iloc[-1] <= 50 and rsi_mom < -0.5:
                scan_direction = "SELL"
            
            if not scan_direction: continue
            
            prob, details, atr_pct = calculate_signal_probability(
                df_m5, df_h4, df_d, df_w, sym, scan_direction
            )
            
            if prob < min_prob: continue
            if strict_mode and details['mtf_bias'] == "NEUTRAL": continue
            
            temp_signal_obj = {'symbol': sym, 'type': scan_direction}
            if check_dynamic_correlation_conflict(temp_signal_obj, signals, cs_scores): continue
            
            cs_aligned = False
            if "_" in sym:
                base, quote = sym.split('_')
                if cs_scores and base in cs_scores and quote in cs_scores:
                    gap = cs_scores.get(base, 0) - cs_scores.get(quote, 0)
                    if scan_direction == "BUY" and gap > 0: cs_aligned = True
                    elif scan_direction == "SELL" and gap < 0: cs_aligned = True
            elif "XAU" in sym or "US30" in sym:
                cs_aligned = True 

            price = df_m5['close'].iloc[-1]
            atr = QuantEngine.calculate_atr(df_m5)
            params = get_asset_params(sym)
            
            sl_mult = params['sl_base']
            if details['structure_z'] != 0: sl_mult -= 0.2
            sl = price - (atr * sl_mult) if scan_direction == "BUY" else price + (atr * sl_mult)
            tp = price + (atr * params['tp_rr']) if scan_direction == "BUY" else price - (atr * params['tp_rr'])
            
            signals.append({
                'symbol': sym,
                'type': scan_direction,
                'price': price,
                'prob': prob,
                'score_display': prob * 10,
                'details': details,
                'atr_pct': atr_pct,
                'detection_time': datetime.now(),
                'sl': sl,
                'tp': tp,
                'rr': params['tp_rr'],
                'cs_aligned': cs_aligned
            })
            
            st.session_state.signal_history[sym] = datetime.now()
            
        except Exception as e:
            time.sleep(1)
            continue
            
    progress_bar.empty()
    status_text.empty()
    return sorted(signals, key=lambda x: x['prob'], reverse=True)

# ==========================================
# AFFICHAGE
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score_display']
    mid_status = s['details']['midnight_status']
    
    time_ago_str = format_time_ago(s['detection_time'])
    
    if sc >= 8.0: label, q_badge = "💎 INSTITUTIONAL", "quality-high"
    elif sc >= 7.0: label, q_badge = "⭐ ALGORITHMIC", "quality-high"
    elif sc >= 6.0: label, q_badge = "✅ STRATEGIC", "quality-medium"
    else: label, q_badge = "📊 TACTICAL", "quality-medium"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"<div class='timestamp-box'>⚡ SIGNAL DÉTECTÉ : {time_ago_str}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;
                            color:white;margin-left:10px;">{s['type']}</span>
                <span class="quality-indicator {q_badge}">{int(s['prob']*100)}% CONF</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.3f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        badges = []
        if "OPTIMAL" in mid_status:
            badges.append(f"<span class='badge badge-midnight'>🌑 {mid_status}</span>")
        else:
            badges.append(f"<span class='badge' style='background:#475569;border:1px solid #64748b'>🌗 {mid_status}</span>")
            
        adx = s['details'].get('adx_val', 0)
        if adx >= 25: badges.append(f"<span class='badge badge-trend'>ADX FORT ({int(adx)})</span>")
        
        badges.append(f"<span class='badge badge-regime'>{s['details']['mtf_bias']}</span>")
        
        if s['cs_aligned']: badges.append("<span class='badge badge-blue'>CS ALIGNÉ</span>")
        if s['details']['fvg_align']: badges.append("<span class='badge badge-gold'>FVG ACTIF</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        mid_price_disp = s['details']['midnight_val']
        c1.metric("Midnight", f"{mid_price_disp:.5f}" if mid_price_disp else "N/A")
        c2.metric("RSI Mom.", f"{s['details']['rsi_mom']:.1f}")
        c3.metric("Z-Score", f"{s['details']['structure_z']:.1f}")

        st.write("")
        r1, r2 = st.columns(2)
        r1.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>STOP LOSS</div>
            <div style='color:#ef4444;font-weight:bold;font-size:1.2em;'>{s['sl']:.5f}</div>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>TAKE PROFIT (1:{s['rr']})</div>
            <div style='color:#10b981;font-weight:bold;font-size:1.2em;'>{s['tp']:.5f}</div>
        </div>""", unsafe_allow_html=True)

# ==========================================
# MAIN
# ==========================================
def main():
    st.title("💎 BLUESTAR ULTIMATE V3.6")
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.9em;'>Relative Time | CS Corrected | 30 Assets</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration V3.6")
        strict_mode = st.checkbox("🔥 Mode Strict", value=False)
        min_prob_display = st.slider("Confiance Min (%)", 50, 95, 70, 5)
        
    if st.button("🔍 Lancer le Scan (30 Actifs)", type="primary"):
        st.session_state.cache = {}
        api = OandaClient()
        
        with st.spinner("Analyse du marché en cours..."):
            results = run_scan_v36(api, min_prob_display/100.0, strict_mode)
        
        if not results:
            st.warning("Aucun signal détecté pour le moment (Marché calme).")
        else:
            st.success(f"{len(results)} Opportunités détectées")
            for sig in results:
                display_sig(sig)

if __name__ == "__main__":
    main()
