import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import logging
import time
import csv
import os
from datetime import datetime, timedelta
from scipy import stats 
from scipy.signal import argrelextrema 
import pytz
import warnings

# ==========================================
# CONFIGURATION & STYLE (THEME BLEU V4.6)
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Bluestar Ultimate V4.6", layout="centered", page_icon="🛡️")

LOG_FILE = "bluestar_v4_gps_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "symbol", "direction", "price", "score", "spread_pips", 
            "atr_pct", "mtf_grade", "hma_slope", "sl", "tp", "session"
        ])

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
        font-weight: 900; font-size: 2.5em; text-align: center; margin-bottom: 0.2em;
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
    .badge { color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; margin: 2px; display: inline-block; }
    .badge-session { background: linear-gradient(135deg, #db2777 0%, #ec4899 100%); }
    .badge-blue { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); }
    .timestamp-box {
        background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6; 
        padding: 8px 12px; border-radius: 6px; font-size: 0.85em;
        color: #93c5fd; margin: 10px 0; font-family: 'Courier New', monospace;
    }
    .inst-label { color:#94a3b8; font-size:0.8em; text-transform: uppercase; letter-spacing: 1px; }
    .inst-val { font-size: 1.1em; font-weight: 700; color: #f1f5f9; }
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
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception as e:
            st.error(f"⚠️ Configuration API manquante: {e}")
            st.stop()

    def get_candles(self, instrument, granularity, count):
        key = f"{instrument}_{granularity}_{count}" 
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
            logging.warning(f"Erreur API get_candles {instrument}: {e}")
            return pd.DataFrame()

    def get_realtime_spread(self, instrument):
        try:
            params = {"instruments": instrument}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            self.client.request(r)
            price = r.response['prices'][0]
            bid = float(price['closeoutBid'])
            ask = float(price['closeoutAsk'])
            spread_raw = ask - bid
            
            if "JPY" in instrument: pip_mult = 100
            elif ("XAU" in instrument or "XAG" in instrument or "XPT" in instrument): pip_mult = 100
            else: pip_mult = 10000
            
            return spread_raw, spread_raw * pip_mult
        except Exception:
            return 0, 0

ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "XAG_USD", "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR"
]

def get_asset_params(symbol):
    if any(idx in symbol for idx in ["US30", "NAS100", "SPX500", "DE30"]):
        return {'type': 'INDEX', 'atr_threshold': 0.10, 'sl_base': 2.0, 'tp_rr': 3.0}
    if any(met in symbol for met in ["XAU", "XPT", "XAG"]):
        return {'type': 'COMMODITY', 'atr_threshold': 0.06, 'sl_base': 1.8, 'tp_rr': 2.5}
    return {'type': 'FOREX', 'atr_threshold': 0.035, 'sl_base': 1.5, 'tp_rr': 2.0}

# ==========================================
# MOTEUR D'INDICATEURS
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_atr(df, period=14):
        if len(df) < period + 1: return 0
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(df, period=7):
        if len(df) < period + 1: return pd.Series([50])
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_adx(df, period=14):
        if len(df) < period * 2: return 0
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
        try:
            z_score = stats.zscore(window)[-1]
            if z_score > 1.5: return 1 
            if z_score < -1.5: return -1 
        except: return 0
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
        if gap_bull > min_gap and curr_close > high_1 and vol_curr > vol_mean * 0.8: return True, "BULL"
        gap_bear = low_1 - high_3
        if gap_bear > min_gap and curr_close < low_1 and vol_curr > vol_mean * 0.8: return True, "BEAR"
        return False, None

    @staticmethod
    def get_institutional_grade(df_d, df_w):
        def analyze_tf(df):
            if len(df) < 50: return "C", "NEUTRAL", 0
            close = df['close']
            price = close.iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else close.rolling(50).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            ema21 = close.ewm(span=21).mean().iloc[-1]
            above_sma = price > sma200
            ema50_above_sma = ema50 > sma200
            ema21_above_50 = ema21 > ema50
            price_above_21 = price > ema21
            
            if above_sma and ema50_above_sma and ema21_above_50 and price_above_21: return "A+", "BULLISH", 100
            if not above_sma and not ema50_above_sma and not ema21_above_50 and not price_above_21: return "A+", "BEARISH", 100
            if above_sma and ema50_above_sma: return "A", "BULLISH", 85
            if not above_sma and not ema50_above_sma: return "A", "BEARISH", 85
            if not above_sma and ema50_above_sma: return "B", "RETRACEMENT_BULL", 70
            if above_sma and not ema50_above_sma: return "B", "RETRACEMENT_BEAR", 70
            return "C", "NEUTRAL", 50

        grade_d, trend_d, score_d = analyze_tf(df_d)
        grade_w, trend_w, score_w = analyze_tf(df_w)
        if grade_w == "C": return "C", "NEUTRAL", 0
        final_score = (score_d * 0.6) + (score_w * 0.4)
        
        if final_score >= 95: final_grade = "A+"
        elif final_score >= 85: final_grade = "A"
        elif final_score >= 70: final_grade = "B"
        else: final_grade = "C"
        return final_grade, trend_d, final_score

    @staticmethod
    def get_midnight_open_ny(df):
        try:
            ny_tz = pytz.timezone('America/New_York')
            df_ny = df.copy()
            df_ny['time'] = pd.to_datetime(df_ny['time'], utc=True).dt.tz_convert(ny_tz)
            midnight_candle = df_ny[df_ny['time'].dt.hour == 0]
            if not midnight_candle.empty: return midnight_candle.iloc[-1]['open']
            else: return None
        except Exception: return None

    @staticmethod
    def calculate_hma(series, period=20):
        if len(series) < period: return pd.Series([])
        half = int(period / 2)
        sqrt_p = int(np.sqrt(period))
        weights_half = np.arange(1, half + 1)
        weights_full = np.arange(1, period + 1)
        weights_sqrt = np.arange(1, sqrt_p + 1)
        wma_half = series.rolling(half).apply(lambda x: np.dot(x, weights_half) / weights_half.sum(), raw=True)
        wma_full = series.rolling(period).apply(lambda x: np.dot(x, weights_full) / weights_full.sum(), raw=True)
        diff = 2 * wma_half - wma_full
        hma = diff.rolling(sqrt_p).apply(lambda x: np.dot(x, weights_sqrt) / weights_sqrt.sum(), raw=True)
        return hma

    @staticmethod
    def hma_slope(hma_series, lookback=5, min_slope=0):
        if len(hma_series) < lookback + 1: return 0
        slope = (hma_series.iloc[-1] - hma_series.iloc[-1 - lookback]) / hma_series.iloc[-1]
        if slope > min_slope: return 1
        elif slope < -min_slope: return -1
        return 0

    @staticmethod
    def weekly_range_position(price, df_w):
        if df_w.empty: return 0.5
        high = df_w['high'].iloc[-1]
        low = df_w['low'].iloc[-1]
        if high == low: return 0.5
        return (price - low) / (high - low)

    @staticmethod
    def check_session_killzone(current_dt_utc, force_open=False):
        if force_open: return "24/7_FORCED"
        hour = current_dt_utc.hour
        if 7 <= hour < 13: return "LDN_SESSION"
        if 13 <= hour < 16: return "NY_OVERLAP"
        if 16 <= hour < 22: return "NY_SESSION"
        return None

# ==========================================
# CURRENCY STRENGTH
# ==========================================
def get_currency_strength_rsi(api):
    now = datetime.now()
    if st.session_state.cs_data.get('time') and (now - st.session_state.cs_data['time']).total_seconds() < 900:
        return st.session_state.cs_data['data']

    forex_pairs = [p for p in ASSETS if "_" in p and "XAU" not in p and "XAG" not in p and "US30" not in p]
    prices = {}
    for pair in forex_pairs[:15]: 
        try:
            df = api.get_candles(pair, "H1", 50)
            if df is not None and not df.empty: prices[pair] = df['close']
            time.sleep(0.01) 
        except Exception: continue

    if not prices: return None
    df_prices = pd.DataFrame(prices).ffill().bfill()
    
    def normalize_score(rsi_value): return ((rsi_value - 50) / 50 + 1) * 5

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
        total_score = 0.0; count = 0
        opponents = [c for c in currencies if c != curr]
        for opp in opponents:
            pair_direct = f"{curr}_{opp}"
            pair_inverse = f"{opp}_{curr}"
            rsi_val = None
            if pair_direct in df_prices.columns:
                rsi_series = calculate_rsi_series(df_prices[pair_direct])
                if not rsi_series.empty: rsi_val = rsi_series.iloc[-1]
            elif pair_inverse in df_prices.columns:
                inverted_price = 1 / df_prices[pair_inverse]
                rsi_series = calculate_rsi_series(inverted_price)
                if not rsi_series.empty: rsi_val = rsi_series.iloc[-1]
            if rsi_val is not None:
                total_score += normalize_score(rsi_val)
                count +=1
        
        if count > 0: final_scores[curr] = total_score / count
        else: final_scores[curr] = 5.0

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
        'EUR_USD':  { 'GBP_USD': 0.9, 'AUD_USD': 0.85, 'USD_CHF': -0.9 },
        'GBP_USD':  { 'EUR_USD': 0.9, 'EUR_GBP': -0.8 },
        'USD_JPY':  { 'USD_CHF': 0.7 }
    }
    
    for existing in existing_signals:
        ex_sym = existing['symbol']
        ex_type = existing['type']
        if new_sym == ex_sym: return True 
        if new_sym in CORRELATION_MAP and ex_sym in CORRELATION_MAP[new_sym]:
            corr = CORRELATION_MAP[new_sym][ex_sym]
            if corr > 0.8 and new_type != ex_type: return True 
            if corr < -0.8 and new_type == ex_type: return True 
    return False

# ==========================================
# PROBABILITÉ (V4.5 Logic)
# ==========================================
def calculate_signal_probability(df_m5, df_h4, df_d, df_w, symbol, direction, current_time_utc, spread_pips, force_open=False):
    prob_factors = []
    weights = []
    details = {}
    rejection_reason = None
    
    params = get_asset_params(symbol)
    atr = QuantEngine.calculate_atr(df_m5)
    atr_pct = (atr / df_m5['close'].iloc[-1]) * 100
    
    if atr_pct < params['atr_threshold'] * 0.3:
        return 0, {}, atr_pct, "Low Volatility (ATR)"
    
    session = QuantEngine.check_session_killzone(current_time_utc, force_open)
    details['session'] = session if session else "OFF-HOURS"
    
    if session is None:
        return 0, {}, atr_pct, "Off Session"
    
    spread_limit = 4.0 if params['type'] == 'FOREX' else 40.0
    if spread_pips > spread_limit:
        return 0, {}, atr_pct, f"Spread High ({spread_pips:.1f})"
    
    vol_score = min(atr_pct / params['atr_threshold'], 2.0)
    details['vol_score'] = vol_score
    
    rsi_serie = QuantEngine.calculate_rsi(df_m5)
    rsi_val = rsi_serie.iloc[-1]
    
    rsi_prob = 0.5
    if direction == "BUY":
        if 45 <= rsi_val <= 70: rsi_prob = 1.0
        elif rsi_val > 70: rsi_prob = 0.4
        else: rsi_prob = 0.2
    else:
        if 30 <= rsi_val <= 55: rsi_prob = 1.0
        elif rsi_val < 30: rsi_prob = 0.4
        else: rsi_prob = 0.2
            
    prob_factors.append(rsi_prob)
    weights.append(0.25)
    details['rsi_mom'] = rsi_val

    hma_series = QuantEngine.calculate_hma(df_m5['close'], 20)
    hma_dir = QuantEngine.hma_slope(hma_series)
    details['hma_slope'] = hma_dir
    hma_score = 1.0 if (direction == "BUY" and hma_dir > 0) or (direction == "SELL" and hma_dir < 0) else 0.2
    prob_factors.append(hma_score)
    weights.append(0.15)
    
    z_score_struc = QuantEngine.detect_structure_zscore(df_h4, 20)
    struc_score = 0.5
    if direction == "BUY":
        if z_score_struc <= 1.0: struc_score = 1.0
    else:
        if z_score_struc >= -1.0: struc_score = 1.0
    prob_factors.append(struc_score)
    weights.append(0.15)
    details['structure_z'] = z_score_struc
    
    inst_grade, inst_trend, inst_raw_score = QuantEngine.get_institutional_grade(df_d, df_w)
    details['mtf_grade'] = inst_grade
    
    mtf_score = 0.5
    if inst_grade == "A+": mtf_score = 1.0
    elif inst_grade == "A": mtf_score = 0.9
    elif inst_grade == "B": mtf_score = 0.7
    else: mtf_score = 0.3

    is_buy = (direction == "BUY")
    if is_buy and "BULL" in inst_trend: mtf_score *= 1.1
    elif not is_buy and "BEAR" in inst_trend: mtf_score *= 1.1
    else: mtf_score *= 0.5
    
    prob_factors.append(min(mtf_score, 1.0))
    weights.append(0.25)
    
    midnight_price = QuantEngine.get_midnight_open_ny(df_m5)
    midnight_score = 0.5 
    curr_price = df_m5['close'].iloc[-1]
    details['midnight_val'] = midnight_price
    
    if midnight_price:
        if is_buy and curr_price < midnight_price: midnight_score = 1.0 # Discount
        elif not is_buy and curr_price > midnight_price: midnight_score = 1.0 # Premium
    
    prob_factors.append(midnight_score)
    weights.append(0.10)

    fvg_active, fvg_type = QuantEngine.detect_smart_fvg(df_m5, atr)
    details['fvg_align'] = fvg_active
    adx_val = QuantEngine.calculate_adx(df_h4)
    details['adx_val'] = adx_val
    
    bonus = 0
    if adx_val > 20: bonus += 0.1
    elif adx_val < 15: bonus -= 0.1
    
    if fvg_active and ((is_buy and fvg_type=="BULL") or (not is_buy and fvg_type=="BEAR")):
        bonus += 0.15
    
    prob_factors.append(0.5 + bonus)
    weights.append(0.10)

    total_weight = sum(weights)
    weighted_prob = sum(p * w for p, w in zip(prob_factors, weights)) / total_weight
    final_score = max(0, min(1.0, weighted_prob))
    
    return final_score, details, atr_pct, None

def format_time_ago(detection_time):
    now = datetime.now()
    diff = now - detection_time
    minutes = int(diff.total_seconds() / 60)
    if minutes < 1: return "À l'instant"
    return f"Il y a {minutes} min"

# ==========================================
# SCANNER
# ==========================================
def run_scan_v40_blue(api, min_prob, strict_mode, current_time_utc, force_open=False):
    cs_scores = get_currency_strength_rsi(api)
    signals = []
    rejected_log = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sym in enumerate(ASSETS):
        progress_bar.progress((i+1)/len(ASSETS))
        status_text.markdown(f"⏳ Analyse: **{sym}** ({i+1}/{len(ASSETS)})")
        
        try:
            df_d_raw = api.get_candles(sym, "D", 250)
            time.sleep(0.02)
            df_m5 = api.get_candles(sym, "M5", 200)
            time.sleep(0.02)
            df_h4 = api.get_candles(sym, "H4", 100)
            
            if df_m5.empty or df_h4.empty or df_d_raw.empty: 
                rejected_log.append(f"{sym}: Données vides")
                continue
            
            df_d = df_d_raw.iloc[-100:].copy()
            df_w = df_d_raw.set_index('time').resample('W-FRI').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna().reset_index()
            
            _, spread_pips = api.get_realtime_spread(sym)
            
            rsi_serie = QuantEngine.calculate_rsi(df_m5)
            curr_rsi = rsi_serie.iloc[-1]
            
            scan_direction = None
            if 40 < curr_rsi < 75: scan_direction = "BUY"
            elif 25 < curr_rsi < 60: scan_direction = "SELL"
            
            if not scan_direction:
                rejected_log.append(f"{sym}: RSI Neutre/Extreme ({curr_rsi:.1f})")
                continue
            
            prob, details, atr_pct, reject_reason = calculate_signal_probability(
                df_m5, df_h4, df_d, df_w, sym, scan_direction, current_time_utc, spread_pips, force_open
            )
            
            if reject_reason:
                rejected_log.append(f"{sym}: {reject_reason}")
                continue
                
            if prob < min_prob: 
                rejected_log.append(f"{sym}: Score Faible ({prob:.2f})")
                continue
                
            if strict_mode and details['mtf_grade'] in ["C", "NEUTRAL"]: 
                rejected_log.append(f"{sym}: Strict Mode (Grade C)")
                continue
            
            temp_signal_obj = {'symbol': sym, 'type': scan_direction}
            if check_dynamic_correlation_conflict(temp_signal_obj, signals, cs_scores): 
                rejected_log.append(f"{sym}: Corrélation")
                continue
            
            cs_aligned = False
            if "_" in sym:
                base, quote = sym.split('_')
                if cs_scores and base in cs_scores and quote in cs_scores:
                    gap = cs_scores.get(base, 0) - cs_scores.get(quote, 0)
                    if scan_direction == "BUY" and gap > 0: cs_aligned = True
                    elif scan_direction == "SELL" and gap < 0: cs_aligned = True
            elif "XAU" in sym or "US30" in sym: cs_aligned = True 

            price = df_m5['close'].iloc[-1]
            atr = QuantEngine.calculate_atr(df_m5)
            params = get_asset_params(sym)
            
            sl = price - (atr * params['sl_base']) if scan_direction == "BUY" else price + (atr * params['sl_base'])
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
                'sl': sl, 'tp': tp, 'rr': params['tp_rr'],
                'cs_aligned': cs_aligned, 'spread': spread_pips
            })
            
        except Exception as e:
            logging.warning(f"Erreur scan {sym}: {e}")
            continue
            
    progress_bar.empty()
    status_text.empty()
    return sorted(signals, key=lambda x: x['prob'], reverse=True), rejected_log

# ==========================================
# AFFICHAGE INSTITUTIONNEL
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score_display']
    mtf_grade = s['details']['mtf_grade']
    
    grade_style = "background: #4b5563; color:white;"
    if mtf_grade == "A+": grade_style = "background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%); color:black;"
    elif mtf_grade == "A": grade_style = "background: linear-gradient(135deg, #a3e635 0%, #4ade80 100%); color:black;"
    elif mtf_grade == "B": grade_style = "background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%); color:white;"

    label = "TACTICAL"
    if sc >= 8.0: label = "INSTITUTIONAL 💎"
    elif sc >= 7.0: label = "STRATEGIC ⭐"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">SPR: {s['spread']:.1f} | ATR: {s['atr_pct']:.3f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        badges = [f"<span class='badge' style='{grade_style}'>🏛️ {mtf_grade}</span>"]
        if s['details']['session']: badges.append(f"<span class='badge badge-session'>⚡ {s['details']['session']}</span>")
        if s['cs_aligned']: badges.append("<span class='badge badge-blue'>CS ALIGNÉ</span>")
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- BLOC INSTITUTIONNEL ---
        k1, k2, k3 = st.columns(3)
        
        # 1. Midnight Open / Premium-Discount
        with k1:
            st.markdown("<div class='inst-label'>MIDNIGHT OPEN</div>", unsafe_allow_html=True)
            mid = s['details']['midnight_val']
            if mid:
                mid_status = "PREMIUM 🔴" if s['price'] > mid else "DISCOUNT 🟢"
                st.markdown(f"<div class='inst-val'>{mid_status}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.8em;color:#64748b'>{mid:.5f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='inst-val'>--</div>", unsafe_allow_html=True)
        
        # 2. Risk Reward
        with k2:
            st.markdown("<div class='inst-label'>RISK / REWARD</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='inst-val' style='color:#f59e0b'>1 : {s['rr']}</div>", unsafe_allow_html=True)
            
        # 3. Structure H4
        with k3:
            st.markdown("<div class='inst-label'>STRUCTURE H4</div>", unsafe_allow_html=True)
            z = s['details']['structure_z']
            struct_txt = "EXPANSION 🌊" if abs(z) > 1.0 else "CONSOLIDATION 🧱"
            st.markdown(f"<div class='inst-val'>{struct_txt}</div>", unsafe_allow_html=True)
        
        st.write("")
        
        ex1, ex2 = st.columns(2)
        ex1.info(f"🛑 **SL:** {s['sl']:.5f}")
        ex2.success(f"🎯 **TP:** {s['tp']:.5f}")

# ==========================================
# MAIN
# ==========================================
def main():
    st.title("🛡️ BLUESTAR ULTIMATE V4.6")
    st.markdown("<p style='text-align:center;color:#94a3b8;'>Zone Detection & Institutional Context</p>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"🕒 **Heure UTC Scanner:** {datetime.now(pytz.utc).strftime('%H:%M')}")

    with st.sidebar:
        st.header("⚙️ Configuration")
        col_st, col_fo = st.columns(2)
        strict_mode = col_st.checkbox("🔥 Strict", value=False)
        force_open = col_fo.checkbox("🔓 24/7", value=False)
        min_prob_display = st.slider("Confiance Min (%)", 40, 90, 60, 5)
        
    if st.button("🔍 Lancer le Scan", type="primary"):
        api = OandaClient()
        current_sim_time = datetime.now(pytz.utc)
        
        with st.spinner("Analyse approfondie (Zone Detection + Institutional)..."):
            results, logs = run_scan_v40_blue(api, min_prob_display/100.0, strict_mode, current_sim_time, force_open)
        
        if not results:
            st.warning("Aucun signal haute probabilité.")
            with st.expander("Voir pourquoi (Logs de rejet)"):
                st.write(logs)
        else:
            st.success(f"{len(results)} Opportunités détectées")
            for sig in results:
                display_sig(sig)
            
            if logs:
                with st.expander("Logs des actifs rejetés"):
                    st.write(logs)

if __name__ == "__main__":
    main()
   
