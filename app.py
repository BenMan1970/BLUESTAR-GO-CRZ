import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, List
from scipy.signal import find_peaks  # NOUVEAU POUR S/R

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Hybrid Pro", layout="centered", page_icon="💎")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ==========================================
# CSS MODERNE "FINTECH PRO+"
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background-color: #0f1117;
        background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%);
    }
    
    .main .block-container {
        max-width: 950px;
        padding-top: 2rem;
    }

    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    h2, h3 {
        color: #e2e8f0;
        font-weight: 700;
    }

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
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #f8fafc !important;
        padding: 1.5rem;
        transition: all 0.3s;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #64748b;
        background-color: #263345 !important;
    }

    .streamlit-expanderContent {
        background-color: #161b22;
        border: 1px solid #334155;
        border-top: none;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        padding: 20px;
    }
    
    .streamlit-expanderHeader p {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: #f1f5f9;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .info-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    .risk-box {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .badge-fvg {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 700;
        display: inline-block;
        margin: 2px;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3);
    }
    
    .badge-gps {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 700;
        display: inline-block;
        margin: 2px;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3);
    }

    .badge-sr {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75em;
        font-weight: 700;
        display: inline-block;
        margin: 2px;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }

    .stAlert {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }

    hr {
        margin: 1.5em 0;
        border-color: #334155;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        background: #0f1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LISTE DES ACTIFS
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CHF_JPY",
    "XAU_USD", "XPT_USD",
    "US30_USD", "NAS100_USD", "SPX500_USD"
]

FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CHF_JPY"
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
# API CLIENT ROBUSTE
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
            self.request_count = 0
            self.last_request_time = time.time()
        except KeyError as e:
            st.error(f"⚠️ Clé manquante dans les secrets: {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur d'initialisation API: {str(e)}")
            st.stop()

    def _rate_limit(self):
        """Gestion du rate limiting (max 20 req/sec pour OANDA)"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def get_candles(self, instrument: str, granularity: str, count: int = 150) -> pd.DataFrame:
        """Récupération des données avec cache"""
        
        cache_key = f"{instrument}_{granularity}"
        if cache_key in st.session_state.cache:
            cache_age = time.time() - st.session_state.cache_time.get(cache_key, 0)
            if cache_age < CACHE_DURATION:
                return st.session_state.cache[cache_key].copy()
        
        self._rate_limit()
        
        params = {"count": count, "granularity": granularity, "price": "M"}
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                r = instruments.InstrumentsCandles(instrument=instrument, params=params)
                self.client.request(r)
                
                if 'candles' not in r.response:
                    return pd.DataFrame()
                
                data = []
                for candle in r.response['candles']:
                    if candle['complete']:
                        try:
                            data.append({
                                'time': candle['time'],
                                'open': float(candle['mid']['o']),
                                'high': float(candle['mid']['h']),
                                'low': float(candle['mid']['l']),
                                'close': float(candle['mid']['c']),
                                'volume': int(candle['volume'])
                            })
                        except (KeyError, ValueError):
                            continue
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                
                if len(df) < 50:
                    return pd.DataFrame()
                
                st.session_state.cache[cache_key] = df.copy()
                st.session_state.cache_time[cache_key] = time.time()
                
                return df
                
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.3)
                    continue
                return pd.DataFrame()
        
        return pd.DataFrame()

# ==========================================
# INDICATEURS TECHNIQUES
# ==========================================

def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    """WMA optimisé"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan, 
        raw=True
    )

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA standard"""
    return series.ewm(span=length, adjust=False).mean()

def calculate_sma(series: pd.Series, length: int) -> pd.Series:
    """SMA standard"""
    return series.rolling(window=length).mean()

def calculate_zlema(series: pd.Series, length: int) -> pd.Series:
    """ZLEMA - Zero Lag EMA"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR - Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """ADX - Average Directional Index"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_s)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx, plus_di, minus_di

def get_rsi_ohlc4(df: pd.DataFrame, length: int = 7) -> pd.Series:
    """RSI sur OHLC4"""
    if len(df) < length + 10:
        return pd.Series(index=df.index, dtype=float)
    
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def get_colored_hma(df: pd.DataFrame, length: int = 20) -> tuple:
    """HMA coloré"""
    if len(df) < length + 10:
        return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=int)
    
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    raw_hma = 2 * wma1 - wma2
    hma = calculate_wma(raw_hma, int(np.round(np.sqrt(length))))
    
    hma_prev = hma.shift(1)
    trend_array = np.where(hma > hma_prev, 1, -1)
    trend_series = pd.Series(trend_array, index=df.index)
    
    return hma, trend_series

# ==========================================
# NEW: SUPPORT & RESISTANCE ENGINE
# ==========================================

def get_nearest_sr(df: pd.DataFrame, current_price: float, timeframe: str = 'D') -> Dict:
    """
    Identifie le Support/Résistance le plus proche
    Basé sur les pivots (Fractales)
    """
    if df.empty or len(df) < 50:
        return {'sup': None, 'res': None, 'dist_sup': 999, 'dist_res': 999}
    
    # Paramètres de distance adaptatifs
    distance = 10 if timeframe == 'W' else 5
    
    # Trouver pivots
    r_indices, _ = find_peaks(df['high'], distance=distance)
    s_indices, _ = find_peaks(-df['low'], distance=distance)
    
    res_levels = df.iloc[r_indices]['high'].values
    sup_levels = df.iloc[s_indices]['low'].values
    
    # Filtrer les niveaux pertinents (proches du prix actuel)
    # On ne garde que ceux à +/- 5% pour optimiser
    relevant_res = res_levels[res_levels > current_price]
    relevant_sup = sup_levels[sup_levels < current_price]
    
    nearest_res = relevant_res.min() if len(relevant_res) > 0 else None
    nearest_sup = relevant_sup.max() if len(relevant_sup) > 0 else None
    
    # Calculer distances en %
    dist_res = ((nearest_res - current_price) / current_price * 100) if nearest_res else 999
    dist_sup = ((current_price - nearest_sup) / current_price * 100) if nearest_sup else 999
    
    return {
        'sup': nearest_sup,
        'res': nearest_res,
        'dist_sup': dist_sup,
        'dist_res': dist_res
    }

# ==========================================
# FVG DETECTION
# ==========================================

def detect_fvg(df: pd.DataFrame) -> tuple:
    """Détection Fair Value Gap"""
    if len(df) < 5:
        return False, False
    
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    
    has_bull = fvg_bull.iloc[-5:].any()
    has_bear = fvg_bear.iloc[-5:].any()
    
    return has_bull, has_bear

# ==========================================
# PIPS CALCULATOR
# ==========================================

def get_pips(pair: str, price_diff: float) -> float:
    """Calcule la valeur en pips"""
    if "XAU" in pair or "US30" in pair or "NAS100" in pair or "SPX500" in pair or "XPT" in pair:
        return abs(price_diff)
    
    multiplier = 100 if "JPY" in pair else 10000
    return abs(price_diff * multiplier)

# ==========================================
# CURRENCY STRENGTH ENGINE
# ==========================================

def calculate_currency_strength(api: OandaClient, lookback_days: int = 1) -> Dict[str, float]:
    """Calcule le score de force pour chaque devise"""
    cache_age = time.time() - st.session_state.currency_strength_time
    if st.session_state.currency_strength_cache and cache_age < CURRENCY_STRENGTH_CACHE_DURATION:
        return st.session_state.currency_strength_cache
    
    forex_data = {}
    for pair in FOREX_PAIRS:
        try:
            df = api.get_candles(pair, "D", count=lookback_days + 5)
            if df is not None and len(df) > lookback_days:
                now = df['close'].iloc[-1]
                past = df['close'].shift(lookback_days).iloc[-1]
                pct = (now - past) / past * 100
                forex_data[pair] = pct
        except:
            continue
    
    data = {}
    for symbol, pct in forex_data.items():
        parts = symbol.split('_')
        if len(parts) != 2:
            continue
        base, quote = parts[0], parts[1]
        
        if base not in data:
            data[base] = []
        if quote not in data:
            data[quote] = []
        
        data[base].append({'pct': pct, 'other': quote})
        data[quote].append({'pct': -pct, 'other': base})
    
    currency_scores = {}
    for curr, items in data.items():
        score = 0
        valid_items = 0
        for item in items:
            opponent = item['other']
            val = item['pct']
            weight = 2.0 if opponent in ['USD', 'EUR', 'JPY'] else 1.0
            score += (val * weight)
            valid_items += weight
        final_score = score / valid_items if valid_items > 0 else 0
        currency_scores[curr] = final_score
    
    st.session_state.currency_strength_cache = currency_scores
    st.session_state.currency_strength_time = time.time()
    return currency_scores

def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    """Score Currency Strength Hybride"""
    
    # === CAS 1 : FOREX ===
    if symbol in FOREX_PAIRS:
        parts = symbol.split('_')
        if len(parts) != 2:
            return {'score': 0, 'details': 'Format invalide', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
        
        base, quote = parts[0], parts[1]
        try:
            strength_scores = calculate_currency_strength(api)
        except:
            return {'score': 0, 'details': 'Erreur calcul', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
        
        if base not in strength_scores or quote not in strength_scores:
            return {'score': 0, 'details': 'Données manquantes', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
        
        base_score = strength_scores[base]
        quote_score = strength_scores[quote]
        
        sorted_currencies = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
        base_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == base)
        quote_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == quote)
        total_currencies = len(sorted_currencies)
        
        score = 0
        details = []
        
        if direction == 'BUY':
            if base_rank <= 3 and quote_rank >= total_currencies - 2:
                score = 2
                details.append(f"✅ {base} TOP3 (#{base_rank}) & {quote} BOTTOM3 (#{quote_rank})")
            elif base_score > quote_score:
                score = 1
                details.append(f"📊 {base} > {quote} (Δ: {base_score - quote_score:+.2f}%)")
            else:
                score = 0
                details.append(f"⚠️ Divergence : {quote} plus fort que {base}")
        else:  # SELL
            if quote_rank <= 3 and base_rank >= total_currencies - 2:
                score = 2
                details.append(f"✅ {quote} TOP3 (#{quote_rank}) & {base} BOTTOM3 (#{base_rank})")
            elif quote_score > base_score:
                score = 1
                details.append(f"📊 {quote} > {base} (Δ: {quote_score - base_score:+.2f}%)")
            else:
                score = 0
                details.append(f"⚠️ Divergence : {base} plus fort que {quote}")
        
        rank_info = f"{base}:#{base_rank} vs {quote}:#{quote_rank}"
        return {'score': score, 'details': ' | '.join(details), 'base_score': base_score, 'quote_score': quote_score, 'rank_info': rank_info}

    # === CAS 2 : INDICES/OR ===
    else:
        try:
            df_d1 = api.get_candles(symbol, "D", count=2)
            if df_d1.empty:
                 return {'score': 0, 'details': 'Pas de data D1', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
            
            open_price = df_d1['open'].iloc[-1]
            curr_price = df_d1['close'].iloc[-1]
            perf_pct = ((curr_price - open_price) / open_price) * 100
            
            score = 0
            details = []
            THRESHOLD_STRONG = 0.30 
            
            if direction == 'BUY':
                if perf_pct > THRESHOLD_STRONG:
                    score = 2
                    details.append(f"🚀 Grosse impulsion Haussière (+{perf_pct:.2f}%)")
                elif perf_pct > 0:
                    score = 1
                    details.append(f"📈 Journée Verte (+{perf_pct:.2f}%)")
                else:
                    score = 0
                    details.append(f"⚠️ Contre-tendance (Journée Rouge: {perf_pct:.2f}%)")
            else: # SELL
                if perf_pct < -THRESHOLD_STRONG:
                    score = 2
                    details.append(f"☄️ Grosse chute Baissière ({perf_pct:.2f}%)")
                elif perf_pct < 0:
                    score = 1
                    details.append(f"📉 Journée Rouge ({perf_pct:.2f}%)")
                else:
                    score = 0
                    details.append(f"⚠️ Contre-tendance (Journée Verte: +{perf_pct:.2f}%)")
            
            return {'score': score, 'details': ' | '.join(details), 'base_score': perf_pct, 'quote_score': 0, 'rank_info': f"Daily: {perf_pct:+.2f}%"}
        except Exception as e:
            return {'score': 0, 'details': f'Err: {str(e)}', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}

# ==========================================
# MTF GPS LOGIC
# ==========================================

def analyze_timeframe_gps(df: pd.DataFrame, timeframe: str) -> Dict:
    """Analyse GPS d'un timeframe"""
    if df.empty or len(df) < 50:
        return {'trend': 'Neutral', 'score': 0, 'details': 'Données insuffisantes', 'atr': 0}
    
    close = df['close']
    curr_price = close.iloc[-1]
    atr_val = calculate_atr(df, 14).iloc[-1]
    
    if timeframe in ['H4', 'D1']:
        sma50 = calculate_sma(close, 50)
        sma200 = calculate_sma(close, 200)
        curr_sma50 = sma50.iloc[-1] if len(df) >= 50 else curr_price
        has_200 = len(df) >= 200
        curr_sma200 = sma200.iloc[-1] if has_200 else curr_sma50
        
        if has_200:
            if curr_price > curr_sma200:
                trend = "Bullish"
                score = 60
                if curr_price > curr_sma50: score += 20
                if curr_sma50 > curr_sma200: score += 20
                details = f"Prix > SMA200 ({curr_sma200:.5f})"
            else:
                trend = "Bearish"
                score = 60
                if curr_price < curr_sma50: score += 20
                if curr_sma50 < curr_sma200: score += 20
                details = f"Prix < SMA200 ({curr_sma200:.5f})"
        else:
            if curr_price > curr_sma50:
                trend = "Bullish"
                score = 50
                details = f"Prix > SMA50 ({curr_sma50:.5f})"
            else:
                trend = "Bearish"
                score = 50
                details = f"Prix < SMA50 ({curr_sma50:.5f})"
    else:
        zlema_val = calculate_zlema(close, 50)
        baseline = calculate_sma(close, 200)
        adx_val, _, _ = calculate_adx(df, 14)
        curr_zlema = zlema_val.iloc[-1]
        curr_adx = adx_val.iloc[-1]
        has_base = len(df) >= 200
        curr_base = baseline.iloc[-1] if has_base else curr_zlema
        
        trend = "Range"
        score = curr_adx
        if curr_price > curr_zlema:
            if has_base and curr_price > curr_base:
                trend = "Bullish"
                details = f"Prix > ZLEMA & Baseline (ADX: {curr_adx:.1f})"
            elif has_base and curr_price < curr_base:
                trend = "Retracement"
                details = f"Hausse sous Baseline (ADX: {curr_adx:.1f})"
            else:
                trend = "Bullish"
                details = f"Prix > ZLEMA (ADX: {curr_adx:.1f})"
        elif curr_price < curr_zlema:
            if has_base and curr_price < curr_base:
                trend = "Bearish"
                details = f"Prix < ZLEMA & Baseline (ADX: {curr_adx:.1f})"
            elif has_base and curr_price > curr_base:
                trend = "Retracement"
                details = f"Baisse au-dessus Baseline (ADX: {curr_adx:.1f})"
            else:
                trend = "Bearish"
                details = f"Prix < ZLEMA (ADX: {curr_adx:.1f})"
        else:
            details = f"Range (ADX: {curr_adx:.1f})"
        
        if curr_adx < 20 and trend == "Retracement":
            trend = "Range"
            details = f"ADX faible ({curr_adx:.1f})"
    
    return {'trend': trend, 'score': min(100, score), 'details': details, 'atr': atr_val}

# ==========================================
# SCORING SYSTEM (BASE)
# ==========================================

def calculate_rsi_score(rsi_series: pd.Series, direction: str) -> Dict:
    if len(rsi_series) < 3:
        return {'score': 0, 'details': 'Données insuffisantes'}
    curr_rsi = rsi_series.iloc[-1]
    prev_rsi = rsi_series.iloc[-2]
    score = 0
    details = []
    
    if direction == 'BUY':
        if prev_rsi < 50 and curr_rsi > 50:
            score = 3
            details.append("✅ Croisement haussier confirmé")
        elif 45 < curr_rsi < 50 and curr_rsi > prev_rsi:
            score = 2
            details.append("⚡ Approche haussière (momentum +)")
        elif curr_rsi < 50 and curr_rsi > prev_rsi:
            score = 1
            details.append("📊 Zone basse, momentum positif")
    else:  # SELL
        if prev_rsi > 50 and curr_rsi < 50:
            score = 3
            details.append("✅ Croisement baissier confirmé")
        elif 50 < curr_rsi < 55 and curr_rsi < prev_rsi:
            score = 2
            details.append("⚡ Approche baissière (momentum -)")
        elif curr_rsi > 50 and curr_rsi < prev_rsi:
            score = 1
            details.append("📊 Zone haute, momentum négatif")
    return {'score': score, 'value': curr_rsi, 'details': ' | '.join(details) if details else 'Pas de signal'}

def calculate_hma_score(hma_trend: pd.Series, direction: str) -> Dict:
    if len(hma_trend) < 2:
        return {'score': 0, 'details': 'Données insuffisantes'}
    curr = hma_trend.iloc[-1]
    prev = hma_trend.iloc[-2]
    score = 0
    details = []
    
    if direction == 'BUY':
        if prev == -1 and curr == 1:
            score = 2
            details.append("✅ Changement VERT")
        elif curr == 1:
            score = 1
            details.append("📈 Déjà VERT")
    else:  # SELL
        if prev == 1 and curr == -1:
            score = 2
            details.append("✅ Changement ROUGE")
        elif curr == -1:
            score = 1
            details.append("📉 Déjà ROUGE")
    return {'score': score, 'color': 'VERT' if curr == 1 else 'ROUGE', 'details': ' | '.join(details) if details else 'Neutre'}

def calculate_mtf_score_gps(api: OandaClient, symbol: str, direction: str) -> Dict:
    timeframes = {'D1': 'D', 'H4': 'H4', 'H1': 'H1'}
    analysis = {}
    for tf_name, tf_code in timeframes.items():
        df = api.get_candles(symbol, tf_code, count=300)
        if df.empty or len(df) < 50:
            analysis[tf_name] = {'trend': 'Neutral', 'score': 0, 'details': 'N/A', 'atr': 0}
        else:
            analysis[tf_name] = analyze_timeframe_gps(df, tf_name)
    
    score = 0
    details = []
    expected = 'Bullish' if direction == 'BUY' else 'Bearish'
    weights = {'D1': 2.0, 'H4': 1.0, 'H1': 0.5}
    aligned_weight = 0
    for tf in ['D1', 'H4', 'H1']:
        if analysis[tf]['trend'] == expected:
            aligned_weight += weights[tf]
    total_weight = sum(weights.values())
    alignment_pct = (aligned_weight / total_weight) * 100
    
    if alignment_pct >= 85:
        score = 3
        details.append("✅ Alignement FORT")
    elif alignment_pct >= 57:
        score = 2
        details.append("⚡ Alignement MOYEN")
    elif alignment_pct >= 28:
        score = 1
        details.append("📊 Alignement FAIBLE")
    
    quality = 'C'
    if analysis['D1']['trend'] == analysis['H4']['trend']:
        quality = 'B'
    if analysis['D1']['trend'] == analysis['H4']['trend'] == analysis['H1']['trend']:
        quality = 'A'
    if quality == 'A' and analysis['D1']['score'] > 70:
        quality = 'A+'
    
    return {'score': score, 'quality': quality, 'analysis': analysis, 'alignment': f"{alignment_pct:.0f}%", 'details': ' | '.join(details) if details else 'Pas d\'alignement'}

def calculate_risk_management(price: float, atr: float, direction: str, pair: str, 
                              sl_multiplier: float = 1.5, tp_multiplier: float = 2.0) -> Dict:
    sl_distance = atr * sl_multiplier
    tp_distance = atr * tp_multiplier
    if direction == 'BUY':
        sl = price - sl_distance
        tp = price + tp_distance
    else:  # SELL
        sl = price + sl_distance
        tp = price - tp_distance
    sl_pips = get_pips(pair, sl_distance)
    tp_pips = get_pips(pair, tp_distance)
    rr_ratio = tp_multiplier / sl_multiplier
    return {'sl': sl, 'tp': tp, 'sl_pips': sl_pips, 'tp_pips': tp_pips, 'rr_ratio': rr_ratio}

# ==========================================
# SCANNER HYBRIDE ULTIME + S/R OVERLAY
# ==========================================

def run_hybrid_scan(api: OandaClient, min_score: int = 4, 
                   enable_risk_manager: bool = True,
                   sl_atr_mult: float = 1.5,
                   tp_atr_mult: float = 2.0) -> List[Dict]:
    """Scanner Hybride avec Overlay Support/Résistance"""
    signals = []
    skipped = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(ASSETS)
    
    status_text.markdown("🔄 **Synchronisation des devises...**")
    try:
        calculate_currency_strength(api)
        status_text.text("✅ Currency Strength OK")
    except:
        status_text.text("⚠️ Erreur Currency Strength")
    
    time.sleep(1)
    
    for i, symbol in enumerate(ASSETS):
        progress_bar.progress((i + 1) / total)
        status_text.markdown(f"🔍 Scan: **{symbol}** ... ({i+1}/{total})")
        
        try:
            df_m15 = api.get_candles(symbol, "M15", count=150)
            if df_m15.empty or len(df_m15) < 50:
                skipped += 1
                continue

            rsi_series = get_rsi_ohlc4(df_m15)
            if rsi_series.empty: continue
            
            hma, hma_trend = get_colored_hma(df_m15)
            if hma_trend.empty: continue
            
            adx_series, _, _ = calculate_adx(df_m15, 14)
            current_adx = adx_series.iloc[-1]
            has_fvg_bull, has_fvg_bear = detect_fvg(df_m15)
            
            current_price = df_m15['close'].iloc[-1]
            atr_m15 = calculate_atr(df_m15, 14).iloc[-1]
            signal_time_utc = df_m15['time'].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            
            # === OVERLAY S/R (Chargement Lazy) ===
            # On ne charge les S/R que si on a un potentiel signal pour économiser API
            sr_context = None
            
            # === Test BUY ===
            rsi_buy = calculate_rsi_score(rsi_series, 'BUY')
            if rsi_buy['score'] > 0:
                hma_buy = calculate_hma_score(hma_trend, 'BUY')
                mtf_buy = calculate_mtf_score_gps(api, symbol, 'BUY')
                cs_buy = calculate_currency_strength_score(api, symbol, 'BUY')
                
                # TWEAK 1: HMA PRICE CHECK
                current_hma_val = hma.iloc[-1]
                if current_price < current_hma_val:
                     hma_buy['score'] = max(0, hma_buy['score'] - 1)
                
                # Bonus FVG
                raw_fvg_bonus = 2 if has_fvg_bull else 0
                fvg_bonus = raw_fvg_bonus if mtf_buy['score'] > 0 else 0
                
                total_score = rsi_buy['score'] + hma_buy['score'] + mtf_buy['score'] + cs_buy['score'] + fvg_bonus
                
                # Penalties
                if cs_buy['score'] == 0 and symbol in FOREX_PAIRS: total_score -= 2
                if current_adx < 20: total_score -= 1
                
                # --- S/R SAFETY CHECK (VETO) ---
                sr_warning = ""
                sr_badge = ""
                
                # On ne lance l'analyse S/R que si le score est prometteur
                if total_score >= min_score - 2:
                    df_d1_sr = api.get_candles(symbol, "D", 250)
                    sr_data = get_nearest_sr(df_d1_sr, current_price, 'D')
                    
                    # Danger: Résistance proche (< 0.25%)
                    if sr_data['dist_res'] < 0.25:
                        total_score -= 2 # PENALITE
                        sr_warning = f"⚠️ DANGER: Résistance Daily à {sr_data['res']:.5f} ({sr_data['dist_res']:.2f}%)"
                    
                    # Bonus: Rebond Support (< 0.3%)
                    elif sr_data['dist_sup'] < 0.30:
                        sr_badge = "🛡️ REBOND SUPPORT D1"
                
                if total_score >= min_score:
                    risk_data = {}
                    if enable_risk_manager:
                        risk_data = calculate_risk_management(current_price, atr_m15, 'BUY', symbol, sl_atr_mult, tp_atr_mult)
                    
                    quality = "MOYEN"
                    quality_color = "#94a3b8"
                    if total_score >= 10: quality, quality_color = "LEGENDARY", "#fbbf24"
                    elif total_score >= 8: quality, quality_color = "EXCELLENT", "#10b981"
                    elif total_score >= 6: quality, quality_color = "FORT", "#34d399"
                    
                    warning = ""
                    if cs_buy['score'] == 0: warning = "⚠️ Divergence/Contre-tendance"
                    if current_adx < 20: warning += " | ⚠️ Range"
                    if sr_warning: warning += f" | {sr_warning}"
                    
                    # Downgrade visuel si warning majeur
                    if "DANGER" in warning: 
                        quality, quality_color = "RISQUE", "#f59e0b"

                    signals.append({
                        "symbol": symbol, "type": "BUY", "price": current_price, "atr_m15": atr_m15,
                        "total_score": total_score, "quality": quality, "quality_color": quality_color,
                        "warning": warning, "sr_badge": sr_badge,
                        "rsi": rsi_buy, "hma": hma_buy, "mtf": mtf_buy, "currency_strength": cs_buy,
                        "has_fvg": has_fvg_bull, "fvg_bonus": fvg_bonus, "risk_management": risk_data,
                        "timestamp_utc": signal_time_utc
                    })
            
            # === Test SELL ===
            rsi_sell = calculate_rsi_score(rsi_series, 'SELL')
            if rsi_sell['score'] > 0:
                hma_sell = calculate_hma_score(hma_trend, 'SELL')
                mtf_sell = calculate_mtf_score_gps(api, symbol, 'SELL')
                cs_sell = calculate_currency_strength_score(api, symbol, 'SELL')
                
                current_hma_val = hma.iloc[-1]
                if current_price > current_hma_val:
                     hma_sell['score'] = max(0, hma_sell['score'] - 1)
                
                raw_fvg_bonus = 2 if has_fvg_bear else 0
                fvg_bonus = raw_fvg_bonus if mtf_sell['score'] > 0 else 0
                
                total_score = rsi_sell['score'] + hma_sell['score'] + mtf_sell['score'] + cs_sell['score'] + fvg_bonus
                
                if cs_sell['score'] == 0 and symbol in FOREX_PAIRS: total_score -= 2
                if current_adx < 20: total_score -= 1
                
                # --- S/R SAFETY CHECK (VETO) ---
                sr_warning = ""
                sr_badge = ""
                
                if total_score >= min_score - 2:
                    df_d1_sr = api.get_candles(symbol, "D", 250)
                    sr_data = get_nearest_sr(df_d1_sr, current_price, 'D')
                    
                    # Danger: Support proche (pour un Sell)
                    if sr_data['dist_sup'] < 0.25:
                        total_score -= 2
                        sr_warning = f"⚠️ DANGER: Support Daily à {sr_data['sup']:.5f} ({sr_data['dist_sup']:.2f}%)"
                    
                    # Bonus: Rejet Résistance
                    elif sr_data['dist_res'] < 0.30:
                        sr_badge = "🛡️ REJET RESISTANCE D1"

                if total_score >= min_score:
                    risk_data = {}
                    if enable_risk_manager:
                        risk_data = calculate_risk_management(current_price, atr_m15, 'SELL', symbol, sl_atr_mult, tp_atr_mult)
                    
                    quality = "MOYEN"
                    quality_color = "#94a3b8"
                    if total_score >= 10: quality, quality_color = "LEGENDARY", "#fbbf24"
                    elif total_score >= 8: quality, quality_color = "EXCELLENT", "#ec4899"
                    elif total_score >= 6: quality, quality_color = "FORT", "#f472b6"
                    
                    warning = ""
                    if cs_sell['score'] == 0: warning = "⚠️ Divergence/Contre-tendance"
                    if current_adx < 20: warning += " | ⚠️ Range"
                    if sr_warning: warning += f" | {sr_warning}"
                    
                    if "DANGER" in warning: 
                        quality, quality_color = "RISQUE", "#f59e0b"
                    
                    signals.append({
                        "symbol": symbol, "type": "SELL", "price": current_price, "atr_m15": atr_m15,
                        "total_score": total_score, "quality": quality, "quality_color": quality_color,
                        "warning": warning, "sr_badge": sr_badge,
                        "rsi": rsi_sell, "hma": hma_sell, "mtf": mtf_sell, "currency_strength": cs_sell,
                        "has_fvg": has_fvg_bear, "fvg_bonus": fvg_bonus, "risk_management": risk_data,
                        "timestamp_utc": signal_time_utc
                    })
        
        except Exception:
            skipped += 1
            continue
    
    progress_bar.empty()
    status_text.empty()
    if skipped > 0: st.caption(f"ℹ️ {skipped} actifs ignorés")
    return signals

# ==========================================
# AFFICHAGE UI
# ==========================================

def display_hybrid_signal(sig: Dict, show_risk: bool = True):
    is_buy = sig['type'] == 'BUY'
    
    if is_buy:
        color_theme = "#10b981"
        bg_gradient = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)"
        arrow = "🟢 BUY"
        border = "2px solid #059669"
    else:
        color_theme = "#ef4444"
        bg_gradient = "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
        arrow = "🔴 SELL"
        border = "2px solid #dc2626"

    signal_utc = sig['timestamp_utc']
    elapsed_sec = int((datetime.now(timezone.utc) - signal_utc).total_seconds())
    fresh_txt = f"{elapsed_sec//60} min ago" if elapsed_sec >= 60 else f"{elapsed_sec} sec ago"
    
    # Badges
    badges_html = ""
    if sig.get('has_fvg') and sig.get('fvg_bonus', 0) > 0:
        badges_html += "<span class='badge-fvg'>🦅 SMART MONEY</span> "
    if sig['mtf']['quality'] in ['A+', 'A']:
        badges_html += "<span class='badge-gps'>🛡️ GPS SECURE</span> "
    if sig.get('sr_badge'):
        badges_html += f"<span class='badge-sr'>{sig['sr_badge']}</span>"
    
    max_score = 12 if sig.get('fvg_bonus', 0) > 0 else 10
    
    header_title = f"{sig['symbol']}  |  {arrow}  |  Score {sig['total_score']}/{max_score}  [{sig['quality']}]"
    
    with st.expander(header_title, expanded=True):
        st.markdown(f"""
        <div style="background: {bg_gradient}; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: {border}; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 1.8em; font-weight: 900; color: white;">{sig['symbol']}</span>
                <span style="background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 4px; font-weight: bold; margin-left: 10px; color: white;">{sig['type']}</span>
            </div>
            <div style="text-align: right;">
                <div style="color: rgba(255,255,255,0.8); font-size: 0.9em;">{fresh_txt}</div>
                <div style="font-size: 1.4em; font-weight: bold; color: white;">{sig['price']:.5f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if badges_html:
            st.markdown(f"<div style='text-align: center; margin-bottom: 15px;'>{badges_html}</div>", unsafe_allow_html=True)

        if sig.get('warning'):
            st.warning(f"{sig['warning']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Total", f"{sig['total_score']}/{max_score}")
        c2.metric("Qualité GPS", sig['mtf']['quality'])
        cs_text = f"{sig['currency_strength']['score']}/2" if sig['symbol'] in FOREX_PAIRS else f"{sig['currency_strength']['score']}/2 (Mom)"
        c3.metric("Force/Mom.", cs_text)
        c4.metric("ATR M15", f"{sig['atr_m15']:.4f}" if sig['atr_m15'] < 1 else f"{sig['atr_m15']:.2f}")

        if show_risk and sig.get('risk_management'):
            st.divider()
            st.markdown("##### 🎯 Risk Management (ATR-Based)")
            rm = sig['risk_management']
            r1, r2, r3 = st.columns(3)
            sl_pips_str = f"{int(rm['sl_pips'])} pips" if "XAU" not in sig['symbol'] and "US30" not in sig['symbol'] else f"{rm['sl_pips']:.1f} pts"
            tp_pips_str = f"{int(rm['tp_pips'])} pips" if "XAU" not in sig['symbol'] and "US30" not in sig['symbol'] else f"{rm['tp_pips']:.1f} pts"
            
            r1.markdown(f"""<div class='risk-box'><div style='color: #94a3b8; font-size: 0.8em;'>STOP LOSS</div><div style='font-size: 1.2em; font-weight: bold; color: #ef4444;'>{rm['sl']:.5f}</div><div style='font-size: 0.85em; color: #ef4444;'>-{sl_pips_str}</div></div>""", unsafe_allow_html=True)
            r2.markdown(f"""<div class='risk-box'><div style='color: #94a3b8; font-size: 0.8em;'>TAKE PROFIT</div><div style='font-size: 1.2em; font-weight: bold; color: #10b981;'>{rm['tp']:.5f}</div><div style='font-size: 0.85em; color: #10b981;'>+{tp_pips_str}</div></div>""", unsafe_allow_html=True)
            r3.markdown(f"""<div class='risk-box'><div style='color: #94a3b8; font-size: 0.8em;'>RATIO R:R</div><div style='font-size: 1.2em; font-weight: bold; color: #f1f5f9;'>1:{rm['rr_ratio']:.2f}</div><div style='font-size: 0.85em; color: #94a3b8;'>Risque Fixe</div></div>""", unsafe_allow_html=True)

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 🛠️ Technique")
            rsi_val = sig['rsi']['value']
            rsi_col = "#10b981" if (is_buy and rsi_val > 50) or (not is_buy and rsi_val < 50) else "#94a3b8"
            st.markdown(f"""<div class="info-box"><div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">RSI (7)</span><span style="font-weight: bold; color: {rsi_col};">{rsi_val:.1f}</span></div><div style="font-size: 0.85em; margin-top: 5px;">{sig['rsi']['details']}</div></div>""", unsafe_allow_html=True)
            hma_col = "#10b981" if sig['hma']['color'] == 'VERT' else "#ef4444"
            st.markdown(f"""<div class="info-box"><div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">HMA Trend</span><span style="font-weight: bold; color: {hma_col};">{sig['hma']['color']}</span></div><div style="font-size: 0.85em; margin-top: 5px;">{sig['hma']['details']}</div></div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown("##### 🌍 Macro & Force")
            st.markdown(f"""<div class="info-box"><div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Alignement MTF</span><span style="font-weight: bold; color: white;">{sig['mtf']['alignment']}</span></div><div style="font-size: 0.85em; margin-top: 5px; color: #cbd5e1;">{sig['mtf']['details']}</div></div>""", unsafe_allow_html=True)
            base_score = sig['currency_strength']['base_score']
            strength_val = f"{base_score:.1f}%" if sig['symbol'] in FOREX_PAIRS else f"{base_score:+.2f}%"
            st.markdown(f"""<div class="info-box"><div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Force/Momentum</span><span style="font-weight: bold; color: white;">{strength_val}</span></div><div style="font-size: 0.85em; margin-top: 5px;">{sig['currency_strength']['rank_info']}</div></div>""", unsafe_allow_html=True)

        st.markdown("##### 📅 Analyse Multi-Timeframe")
        mtf_cols = st.columns(3)
        for i, tf in enumerate(['D1', 'H4', 'H1']):
            data = sig['mtf']['analysis'][tf]
            badge_col = "#34d399" if data['trend'] == 'Bullish' else "#f87171" if data['trend'] == 'Bearish' else "#cbd5e1"
            with mtf_cols[i]:
                st.markdown(f"""<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; text-align: center;"><div style="color: #94a3b8; font-size: 0.8em; font-weight: bold; margin-bottom: 5px;">{tf}</div><div style="color: {badge_col}; font-weight: bold; font-size: 0.9em;">{data['trend']}</div></div>""", unsafe_allow_html=True)
        
        breakdown_parts = [f"RSI({sig['rsi']['score']})", f"HMA({sig['hma']['score']})", f"GPS({sig['mtf']['score']})", f"Force({sig['currency_strength']['score']})"]
        if sig.get('fvg_bonus', 0) > 0: breakdown_parts.append(f"FVG(+{sig['fvg_bonus']})")
        st.markdown(f"""<div style='margin-top: 15px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 6px; text-align: center;'><span style='color: #94a3b8; font-size: 0.8em;'>Composition du Score :</span><br><span style='color: white; font-weight: bold; font-size: 0.95em;'>{' + '.join(breakdown_parts)} = {sig['total_score']}</span></div>""", unsafe_allow_html=True)

# ==========================================
# MAIN
# ==========================================
st.title("💎 Bluestar SNP3 Hybrid Pro")
st.markdown(f"""<div style="text-align: center; color: #94a3b8; margin-bottom: 30px;">Scanner MTF GPS + Currency Strength + Smart Money + <span style="color: #f59e0b;">S/R Protection</span></div>""", unsafe_allow_html=True)

with st.expander("⚙️ Paramètres Avancés", expanded=False):
    col_set1, col_set2 = st.columns(2)
    with col_set1: enable_risk_mgr = st.checkbox("🎯 Risk Manager (SL/TP Auto)", value=True)
    with col_set2:
        if enable_risk_mgr:
            sl_mult = st.slider("SL Multiplier (× ATR)", 1.0, 3.0, 1.5, 0.1)
            tp_mult = st.slider("TP Multiplier (× ATR)", 1.5, 4.0, 2.0, 0.1)
        else: sl_mult, tp_mult = 1.5, 2.0

col1, col2 = st.columns([3, 1])
with col1: min_score = st.slider("Sensibilité du signal (Score Min)", 4, 12, 6)
with col2:
    st.write("")
    st.write("")
    clear_cache = st.button("🧹 Reset")

if clear_cache:
    st.session_state.cache = {}
    st.session_state.cache_time = {}
    st.session_state.currency_strength_cache = None
    st.session_state.currency_strength_time = 0
    st.toast("Cache vidé !", icon="🧹")

if st.button("🚀 LANCER LE SCANNER", type="primary"):
    api = OandaClient()
    start_time = time.time()
    with st.spinner("Analyse approfondie (Tech + Macro + S/R)..."):
        results = run_hybrid_scan(api, min_score=min_score, enable_risk_manager=enable_risk_mgr, sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)
    duration = time.time() - start_time
    
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signaux", len(results))
    m2.metric("Temps", f"{duration:.1f}s")
    legendary_count = sum(1 for s in results if s['quality'] == 'LEGENDARY')
    m4.metric("🏆 Legendary", legendary_count)
    st.markdown("---")
    
    if not results: st.info(f"Aucun signal détecté score >= {min_score}.")
    else:
        results_sorted = sorted(results, key=lambda x: (x['total_score'], x['mtf']['quality'], x.get('has_fvg', False)), reverse=True)
        for sig in results_sorted: display_hybrid_signal(sig, show_risk=enable_risk_mgr)
