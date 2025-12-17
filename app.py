import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import requests
from datetime import datetime, time as dtime, timezone, timedelta
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# 1. CONFIGURATION & STYLE (DESIGN BLEU ORIGINAL)
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Hybrid Pro", layout="centered", page_icon="💎")
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

    /* LE TITRE BLEU ORIGINAL */
    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    /* LE BOUTON BLEU ORIGINAL */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        font-weight: 700;
        font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        transition: all 0.2s ease;
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

    .risk-box {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; display: inline-block; margin: 2px; }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; display: inline-block; margin: 2px; }
    .badge-sr { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; display: inline-block; margin: 2px; }
    .badge-news { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; display: inline-block; margin: 2px; }

    hr { margin: 1.5em 0; border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DONNÉES & CONSTANTES
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "XPT_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]

# Cache système
if 'cache' not in st.session_state:
    st.session_state.cache = {}
    st.session_state.cache_time = {}
    st.session_state.currency_strength_cache = None
    st.session_state.currency_strength_time = 0

CACHE_DURATION = 30
CURRENCY_STRENGTH_CACHE_DURATION = 300

# ==========================================
# 3. CLIENT API ROBUSTE
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
            self.last_request_time = time.time()
        except KeyError:
            st.error("⚠️ Clé manquante dans secrets.toml")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur API: {str(e)}")
            st.stop()

    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()

    def get_candles(self, instrument: str, granularity: str, count: int = 150) -> pd.DataFrame:
        cache_key = f"{instrument}_{granularity}"
        if cache_key in st.session_state.cache:
            if time.time() - st.session_state.cache_time.get(cache_key, 0) < CACHE_DURATION:
                return st.session_state.cache[cache_key].copy()
        
        self._rate_limit()
        params = {"count": count, "granularity": granularity, "price": "M"}
        
        for attempt in range(2):
            try:
                r = instruments.InstrumentsCandles(instrument=instrument, params=params)
                self.client.request(r)
                if 'candles' not in r.response: return pd.DataFrame()
                
                data = []
                for c in r.response['candles']:
                    if c['complete']:
                        try:
                            data.append({
                                'time': c['time'],
                                'open': float(c['mid']['o']),
                                'high': float(c['mid']['h']),
                                'low': float(c['mid']['l']),
                                'close': float(c['mid']['c']),
                                'volume': int(c['volume'])
                            })
                        except: continue
                
                if not data: return pd.DataFrame()
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                
                if len(df) < 50: return pd.DataFrame()
                
                st.session_state.cache[cache_key] = df.copy()
                st.session_state.cache_time[cache_key] = time.time()
                return df
            except:
                if attempt < 1: time.sleep(0.5)
                else: return pd.DataFrame()
        return pd.DataFrame()

# ==========================================
# 4. MODULE NEWS & SESSIONS (PROFESSIONNEL)
# ==========================================
class NewsManager:
    """Gère les filtres de temps et de nouvelles économiques"""
    
    @staticmethod
    def is_session_open() -> bool:
        """Filtre de liquidité M5: Londres et New York uniquement."""
        now = datetime.now(timezone.utc).time()
        # 07:00 UTC à 20:00 UTC (Couvre Londres et majorité de NY)
        start_time = dtime(7, 0)
        end_time = dtime(20, 0)
        return start_time <= now <= end_time

    @staticmethod
    def check_impact_news(symbol: str) -> tuple[bool, str]:
        """Zones de Danger fixes pour les news USD (NFP/CPI/FOMC)"""
        now = datetime.now(timezone.utc)
        
        # Liste simplifiée des heures "Danger Zone" USD (UTC)
        danger_zones = [
            (dtime(12, 25), dtime(12, 45)), # US Open data / CPI / NFP early
            (dtime(13, 25), dtime(13, 45)), # US Open data standard
            (dtime(18, 55), dtime(19, 15))  # FOMC / Minutes
        ]
        
        if "USD" in symbol or "XAU" in symbol or "US30" in symbol:
            current_t = now.time()
            for start, end in danger_zones:
                if start <= current_t <= end:
                    return True, "Danger Zone (News USD)"
        
        return False, ""

# ==========================================
# 5. INDICATEURS TECHNIQUES
# ==========================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_zlema(series, length):
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def get_rsi_ohlc4(df, length=7):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def get_colored_hma(df, length=20):
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    hma = calculate_wma(2 * wma1 - wma2, int(np.sqrt(length)))
    trend = pd.Series(np.where(hma > hma.shift(1), 1, -1), index=df.index)
    return hma, trend

def detect_fvg(df):
    if len(df) < 5: return False, False
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull.iloc[-5:].any(), fvg_bear.iloc[-5:].any()

def get_nearest_sr(df, current_price):
    if len(df) < 20: return {'sup': None, 'res': None, 'dist_sup': 999, 'dist_res': 999}
    h = df['high']
    l = df['low']
    is_res = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
    is_sup = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))
    res_levels = df[is_res]['high'].values
    sup_levels = df[is_sup]['low'].values
    nr = res_levels[res_levels > current_price].min() if any(res_levels > current_price) else None
    ns = sup_levels[sup_levels < current_price].max() if any(sup_levels < current_price) else None
    dr = ((nr - current_price) / current_price * 100) if nr else 999
    ds = ((current_price - ns) / current_price * 100) if ns else 999
    return {'sup': ns, 'res': nr, 'dist_sup': ds, 'dist_res': dr}

def get_pips(pair, diff):
    if any(x in pair for x in ["XAU", "US30", "NAS100", "SPX500", "XPT"]): return abs(diff)
    return abs(diff * (100 if "JPY" in pair else 10000))

# ==========================================
# 6. MOTEUR MARKET MAP PRO (FUNDAMENTAL)
# ==========================================
def calculate_currency_strength(api: OandaClient, lookback_days: int = 1) -> Dict[str, float]:
    cache_age = time.time() - st.session_state.currency_strength_time
    if st.session_state.currency_strength_cache and cache_age < CURRENCY_STRENGTH_CACHE_DURATION:
        if sum(abs(x) for x in st.session_state.currency_strength_cache.values()) > 0.001:
            return st.session_state.currency_strength_cache

    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        env = st.secrets.get("OANDA_ENVIRONMENT", "practice")
        base_url = "https://api-fxtrade.oanda.com" if env == "live" else "https://api-fxpractice.oanda.com"
    except: return {}

    pairs = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
        'AUD_JPY', 'AUD_CAD', 'AUD_NZD', 'AUD_CHF',
        'CAD_JPY', 'CAD_CHF', 'NZD_JPY', 'NZD_CHF', 'CHF_JPY'
    ]
    
    forex_data = {}
    headers = {"Authorization": f"Bearer {token}"}
    
    for inst in pairs:
        try:
            url = f"{base_url}/v3/instruments/{inst}/candles?count={lookback_days+5}&granularity=D"
            resp = requests.get(url, headers=headers, timeout=2)
            if resp.status_code == 200:
                c = resp.json().get('candles', [])
                if c and len(c) > lookback_days:
                    now = float(c[-1]['mid']['c'])
                    past = float(c[-(lookback_days+1)]['mid']['c'])
                    forex_data[inst] = (now - past) / past * 100
        except: continue

    data = {}
    for s, p in forex_data.items():
        b, q = s.split('_')
        if b not in data: data[b] = []
        if q not in data: data[q] = []
        data[b].append({'pct': p, 'other': q})
        data[q].append({'pct': -p, 'other': b})
    
    scores = {}
    for c, items in data.items():
        sc = 0
        w_sum = 0
        for i in items:
            w = 2.0 if i['other'] in ['USD', 'EUR', 'JPY'] else 1.0
            sc += (i['pct'] * w)
            w_sum += w
        scores[c] = sc / w_sum if w_sum > 0 else 0
        
    if sum(abs(v) for v in scores.values()) > 0.001:
        st.session_state.currency_strength_cache = scores
        st.session_state.currency_strength_time = time.time()
    return scores

def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    if symbol in FOREX_PAIRS:
        parts = symbol.split('_')
        base, quote = parts[0], parts[1]
        scores = calculate_currency_strength(api)
        
        if base not in scores: 
            return {'score': 0, 'details': 'N/A', 'base_score': 0, 'label': 'Neutre'}
        
        b_score = scores[base]
        q_score = scores[quote]
        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {k: i+1 for i, (k, v) in enumerate(sorted_s)}
        b_rank, q_rank = ranks.get(base, 8), ranks.get(quote, 8)
        
        score = 0
        label = "Neutre"
        if direction == 'BUY':
            if b_rank <= 3 and q_rank >= 6: score = 2; label = "Excellent"
            elif b_score > q_score: score = 1; label = "Validé"
            else: label = "Divergence"
        else:
            if q_rank <= 3 and b_rank >= 6: score = 2; label = "Excellent"
            elif q_score > b_score: score = 1; label = "Validé"
            else: label = "Divergence"
        return {'score': score, 'details': label, 'base_score': b_score, 'quote_score': q_score, 'rank_info': f"{base}(#{b_rank}) vs {quote}(#{q_rank})"}
    else:
        try:
            df = api.get_candles(symbol, "D", count=2)
            if df.empty: return {'score': 0, 'details': 'N/A', 'base_score': 0, 'label': 'Neutre'}
            chg = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100
            score = 0
            label = "Neutre"
            if direction == 'BUY':
                if chg > 0.3: score = 2; label = "Impulsion"
                elif chg > 0: score = 1; label = "Positif"
                else: label = "Contre-tendance"
            else:
                if chg < -0.3: score = 2; label = "Chute"
                elif chg < 0: score = 1; label = "Négatif"
                else: label = "Contre-tendance"
            return {'score': score, 'details': label, 'base_score': chg, 'rank_info': f"D1: {chg:+.2f}%"}
        except:
            return {'score': 0, 'details': 'Err', 'base_score': 0, 'label': 'Err'}

# ==========================================
# 7. ANALYSE MTF INSTITUTIONNELLE (COMPLET)
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
    
    # Resampling Macro
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
        'M': trends['M'], 
        'W': trends['W'], 
        'D': trends['D'],
        'H4': trends['H4']
    }

    return {
        'score': final_score, 
        'quality': quality, 
        'alignment': f"{alignment_pct:.0f}%", 
        'analysis': analysis_display
    }

def calculate_risk(price, atr, direction, pair, sl_m, tp_m):
    sl_dist = atr * sl_m
    tp_dist = atr * tp_m
    sl = price - sl_dist if direction == 'BUY' else price + sl_dist
    tp = price + tp_dist if direction == 'BUY' else price - tp_dist
    return {'sl': sl, 'tp': tp, 'sl_pips': get_pips(pair, sl_dist), 'tp_pips': get_pips(pair, tp_dist), 'rr': tp_m/sl_m}

# ==========================================
# 8. LOGIQUE SCANNER (OPTIMISÉE M5 + NEWS)
# ==========================================
def run_scan(api, min_score, risk, sl_m, tp_m, use_session_filter):
    sigs = []
    
    # 1. Calcul silencieux de la force
    calculate_currency_strength(api)
    
    # 2. Vérification Session
    session_open = NewsManager.is_session_open()
    if use_session_filter and not session_open:
        st.warning("⚠️ Session de liquidité fermée (Hors Londres/NY). Le scan M5 peut être erratique.")
    
    pbar = st.progress(0)
    for i, sym in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        
        # 3. Filtre News Immédiat
        is_news, news_msg = NewsManager.check_impact_news(sym)
        if is_news: 
            # On continue pour afficher l'alerte
            pass
        
        try:
            # 4. TARGET M5 (SNIPER)
            df = api.get_candles(sym, "M5", count=150)
            if df.empty or len(df) < 50: continue
            
            p = df['close'].iloc[-1]
            atr = calculate_atr(df).iloc[-1]
            rsi = get_rsi_ohlc4(df).iloc[-1]
            hma, trend = get_colored_hma(df)
            hma_val = trend.iloc[-1]
            fvg_b, fvg_s = detect_fvg(df)
            
            typ = None
            if rsi > 50 and hma_val == 1: typ = 'BUY'
            elif rsi < 50 and hma_val == -1: typ = 'SELL'
            
            if typ:
                sc = 0
                sc += 3 # Base technique
                
                # Context MTF Institutionnel
                mtf = calculate_mtf_score_gps(api, sym, typ)
                cs = calculate_currency_strength_score(api, sym, typ)
                sc += mtf['score'] + cs['score']
                if (typ == 'BUY' and fvg_b) or (typ == 'SELL' and fvg_s): sc += 1
                
                # S/R Veto
                sr = get_nearest_sr(api.get_candles(sym, "D", 200), p)
                warn = ""
                badge = ""
                
                if typ == 'BUY':
                    if sr['dist_res'] < 0.25: sc -= 2; warn = "Résistance proche"
                    elif sr['dist_sup'] < 0.4: badge = "Rebond Support"
                else:
                    if sr['dist_sup'] < 0.25: sc -= 2; warn = "Support proche"
                    elif sr['dist_res'] < 0.4: badge = "Rejet Résistance"
                
                # Penalité News / Session
                news_badge = ""
                if is_news:
                    sc -= 10 # Veto News
                    warn = f"⛔ NEWS IMMINENTE : {news_msg}"
                    news_badge = "NEWS DANGER"
                elif use_session_filter and not session_open:
                    sc -= 2 # Penalité hors session
                    warn = "Hors Session Liquide"
                
                if sc >= min_score:
                    rm = calculate_risk(p, atr, typ, sym, sl_m, tp_m) if risk else None
                    sigs.append({
                        'symbol': sym, 'type': typ, 'price': p, 'score': sc,
                        'quality': mtf['quality'], 'atr': atr, 'warn': warn, 
                        'badge': badge, 'news_badge': news_badge,
                        'rsi': rsi, 'mtf': mtf, 'cs': cs, 'fvg': (fvg_b if typ=='BUY' else fvg_s),
                        'rm': rm, 'time': df['time'].iloc[-1]
                    })
        except: continue
    pbar.empty()
    return sigs

# ==========================================
# 9. AFFICHAGE DES SIGNAUX (DESIGN BLEU)
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    ago = int((datetime.now(timezone.utc) - s['time'].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()/60)
    
    sc = s['score']
    if sc >= 10: label = "💎 LEGENDARY"
    elif sc >= 8: label = "⭐ EXCELLENT"
    elif sc >= 6: label = "✅ BON"
    else: label = "⚠️ MOYEN"

    header_txt = f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc}/10]"
    
    with st.expander(header_txt, expanded=True):
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
                <span style="font-size:0.8em;margin-left:10px;color:#cbd5e1;">M5 Entry</span>
            </div>
            <div style="text-align:right;">
                <div style="color:#cbd5e1;font-size:0.9em;">il y a {ago} min</div>
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        badges = []
        if s['news_badge']: badges.append(f"<span class='badge-news'>{s['news_badge']}</span>")
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        q_badge_color = "#10b981" if s['quality'] in ['A+', 'A'] else "#f59e0b" if 'B' in s['quality'] else "#94a3b8"
        badges.append(f"<span class='badge-gps' style='background:{q_badge_color}'>🛡️ GPS {s['quality']}</span>")
        if s['badge']: badges.append(f"<span class='badge-sr'>{s['badge']}</span>")
        
        if badges: st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        if s['warn']: st.warning(f"⚠️ {s['warn']}")
        
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Total", f"{sc}/10", delta=label, delta_color="off")
        c2.metric("Alignement MTF", s['mtf']['alignment'])
        
        f_val = s['cs']['base_score']
        f_str = f"{f_val:+.2f}%" if abs(f_val) > 0.001 else "0.00%"
        score_mmp = s['cs']['score']
        label_mmp = s['cs']['details']
        delta_col = "normal" if score_mmp > 0 else "off"
        
        c3.metric("Momentum", f"{score_mmp}/2", delta=label_mmp, delta_color=delta_col)
        c4.metric("ATR M5", f"{s['atr']:.4f}")
        
        if s['rm']:
            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            sl_pip = int(s['rm']['sl_pips']) if "XAU" not in s['symbol'] and "US30" not in s['symbol'] else f"{s['rm']['sl_pips']:.1f}"
            tp_pip = int(s['rm']['tp_pips']) if "XAU" not in s['symbol'] and "US30" not in s['symbol'] else f"{s['rm']['tp_pips']:.1f}"
            
            r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>STOP LOSS</div><div style='color:#ef4444;font-weight:bold;font-size:1.1em'>{s['rm']['sl']:.5f}</div><div style='color:#ef4444;font-size:0.8em'>-{sl_pip} pips</div></div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>TAKE PROFIT</div><div style='color:#10b981;font-weight:bold;font-size:1.1em'>{s['rm']['tp']:.5f}</div><div style='color:#10b981;font-size:0.8em'>+{tp_pip} pips</div></div>", unsafe_allow_html=True)
            r3.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em'>R:R</div><div style='color:white;font-weight:bold;font-size:1.1em'>1:{s['rm']['rr']:.2f}</div><div style='color:#94a3b8;font-size:0.8em'>Fixe</div></div>", unsafe_allow_html=True)
            
        st.markdown("---")
        k1, k2 = st.columns(2)
        mtf_str = " | ".join([f"{k}:{v}" for k,v in s['mtf']['analysis'].items()])
        k1.info(f"**Technique**\nRSI : {s['rsi']:.1f}\nStructure : {mtf_str}")
        k2.info(f"**Fondamental**\nForce : {f_str}\n{s['cs']['rank_info']}")

# ==========================================
# 10. MAIN APP
# ==========================================
st.title("💎 Bluestar SNP3 Hybrid Pro")
with st.expander("⚙️ Paramètres & Sécurité"):
    c1, c2 = st.columns(2)
    risk = c1.checkbox("Risk Manager Auto", True)
    session_filt = c1.checkbox("Session Filter (Londres/NY uniquement)", True)
    # Paramètres M5 adaptés
    sl = c2.slider("SL Multiplier (xATR)", 1.5, 4.0, 2.0)
    tp = c2.slider("TP Multiplier (xATR)", 2.0, 6.0, 3.0)

co1, co2 = st.columns([3,1])
min_sc = co1.slider("Sensibilité du signal (Score Min)", 4, 10, 7)

if co2.button("🧹 Reset"):
    st.session_state.cache = {}
    st.session_state.currency_strength_cache = None
    st.toast("Cache vidé")

if st.button("🚀 LANCER LE SCANNER", type="primary"):
    api = OandaClient()
    results = run_scan(api, min_sc, risk, sl, tp, session_filt)
    
    st.success(f"Scan M5 terminé : {len(results)} opportunités trouvées")
    
    # Tri intelligent : Score > Qualité MTF > Time
    for s in sorted(results, key=lambda x: (x['score'], x['quality']), reverse=True):
        display_sig(s)
