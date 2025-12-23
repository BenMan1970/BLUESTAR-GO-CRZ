import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import time
import logging
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
import re

# ==========================================
# 1. CONFIGURATION & SESSION
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 GPS", layout="centered", page_icon="💎")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation du cache avec expiration
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'cache_timestamps' not in st.session_state:
    st.session_state.cache_timestamps = {}
if 'matrix_cache' not in st.session_state:
    st.session_state.matrix_cache = None
if 'matrix_timestamp' not in st.session_state:
    st.session_state.matrix_timestamp = None

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
# 2. DONNÉES
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

# Configuration des timeframes avec expiration du cache
CACHE_EXPIRY = {
    'M5': 300,    # 5 minutes
    'H1': 3600,   # 1 heure
    'H4': 14400,  # 4 heures
    'D': 86400,   # 24 heures
    'W': 604800,  # 7 jours
    'M': 2592000  # 30 jours
}

# ✅ CONFIGURATION CORRIGÉE DU SCORING
SCORING_CONFIG = {
    'gps_weight': 0.40,
    'fundamental_weight': 0.35,
    'technical_weight': 0.25,
    'min_gps_quality': 'C',
    'min_fundamental_gap': 0.5,
    
    # ✅ RSI - Zones claires et strictes
    'rsi_overbought': 70,       # Surachat
    'rsi_oversold': 30,         # Survente
    'rsi_optimal_min': 45,      # Zone neutre basse
    'rsi_optimal_max': 55,      # Zone neutre haute
    'rsi_acceptable_min': 35,   # Acceptable pour BUY
    'rsi_acceptable_max': 65,   # Acceptable pour SELL
}

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
            logger.info("Client OANDA initialisé avec succès")
        except Exception as e:
            logger.error(f"Erreur d'initialisation OANDA: {e}")
            st.error("⚠️ Configuration API manquante")
            st.stop()

    def get_candles(self, instrument: str, granularity: str, count: int) -> pd.DataFrame:
        """Récupération des chandeliers avec cache intelligent"""
        key = f"{instrument}_{granularity}"
        now = datetime.now(timezone.utc)
        
        # Vérifier le cache avec expiration
        if key in st.session_state.cache:
            cache_time = st.session_state.cache_timestamps.get(key)
            if cache_time:
                expiry = CACHE_EXPIRY.get(granularity, 300)
                if (now - cache_time).total_seconds() < expiry:
                    logger.debug(f"Cache hit: {key}")
                    return st.session_state.cache[key]
        
        # Récupération depuis l'API
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({
                        'time': pd.to_datetime(c['time']),
                        'open': float(c['mid']['o']),
                        'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']),
                        'close': float(c['mid']['c'])
                    })
            
            df = pd.DataFrame(data)
            if not df.empty:
                st.session_state.cache[key] = df
                st.session_state.cache_timestamps[key] = now
                logger.debug(f"Cache mis à jour: {key}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur API pour {instrument}/{granularity}: {e}")
            return pd.DataFrame()

# ==========================================
# 4. GPS MTF INSTITUTIONNEL
# ==========================================
MTF_WEIGHTS = {'M': 5.0, 'W': 4.0, 'D': 4.0, 'H4': 2.5, 'H1': 1.5}
TOTAL_WEIGHT = sum(MTF_WEIGHTS.values())

def ema(series, length):
    """Moyenne mobile exponentielle"""
    return series.ewm(span=length, adjust=False).mean()

def sma_local(series, length):
    """Moyenne mobile simple"""
    return series.rolling(window=length).mean()

def calc_institutional_trend_macro(df):
    """Analyse macro (Mensuel/Hebdo)"""
    if len(df) < 50:
        return 'Range', 0
    
    close = df['close']
    curr = close.iloc[-1]
    
    sma200 = sma_local(close, 200).iloc[-1] if len(df) >= 200 else sma_local(close, 50).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    
    if curr > sma200 and ema50 > sma200:
        return "Bullish", 85
    if curr < sma200 and ema50 < sma200:
        return "Bearish", 85
    if curr > sma200:
        return "Bullish", 65
    if curr < sma200:
        return "Bearish", 65
    
    return "Range", 40

def calc_institutional_trend_daily(df):
    """Analyse Daily"""
    if len(df) < 200:
        return 'Range', 0
    
    close = df['close']
    curr = close.iloc[-1]
    
    sma200 = sma_local(close, 200).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    
    # Tendance forte
    if curr > sma200 and ema50 > sma200 and ema21 > ema50 and curr > ema21:
        return "Bullish", 90
    if curr < sma200 and ema50 < sma200 and ema21 < ema50 and curr < ema21:
        return "Bearish", 90
    
    # Retracement
    if curr < sma200 and ema50 > sma200:
        return "Retracement Bull", 55
    if curr > sma200 and ema50 < sma200:
        return "Retracement Bear", 55
    
    if curr > sma200:
        return "Bullish", 50
    if curr < sma200:
        return "Bearish", 50
    
    return "Range", 35

def calc_institutional_trend_4h(df):
    """Analyse H4"""
    if len(df) < 200:
        return 'Range', 0
    
    close = df['close']
    curr = close.iloc[-1]
    
    sma200 = sma_local(close, 200).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    ema21 = ema(close, 21).iloc[-1]
    
    if curr > sma200 and ema21 > ema50 and ema50 > sma200:
        return "Bullish", 80
    if curr < sma200 and ema21 < ema50 and ema50 < sma200:
        return "Bearish", 80
    
    if curr < sma200 and ema50 > sma200:
        return "Retracement Bull", 50
    if curr > sma200 and ema50 < sma200:
        return "Retracement Bear", 50
    
    if curr > sma200:
        return "Bullish", 60
    if curr < sma200:
        return "Bearish", 60
    
    return "Range", 40

def calc_institutional_trend_intraday(df):
    """Analyse H1"""
    if len(df) < 50:
        return 'Range', 0
    
    close = df['close']
    curr = close.iloc[-1]
    
    lag = 24
    src_adj = close + (close - close.shift(lag))
    zlema = src_adj.ewm(span=50, adjust=False).mean().iloc[-1]
    
    ema21 = ema(close, 21).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    
    if curr > zlema and ema21 > ema50:
        return "Bullish", 75
    if curr < zlema and ema21 < ema50:
        return "Bearish", 75
    
    return "Range", 30

def calculate_mtf_score_gps(api, symbol, direction):
    """Calcul du score GPS multi-timeframe"""
    try:
        df_d = api.get_candles(symbol, "D", count=500)
        df_h4 = api.get_candles(symbol, "H4", count=200)
        df_h1 = api.get_candles(symbol, "H1", count=200)
        
        if df_d.empty or df_h4.empty or df_h1.empty:
            logger.warning(f"Données MTF incomplètes pour {symbol}")
            return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}, 'confidence': 0}

        d_res = df_d.copy()
        d_res.set_index('time', inplace=True)
        
        try:
            df_m = d_res.resample('ME').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna()
        except:
            df_m = d_res.resample('M').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna()
            
        df_w = d_res.resample('W-FRI').agg({
            'open':'first', 'high':'max', 'low':'min', 'close':'last'
        }).dropna()

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
        perfect_alignment = 0
        
        for tf, trend in trends.items():
            weight = MTF_WEIGHTS.get(tf, 1.0)
            if trend == target:
                w_score += weight * (scores[tf] / 100)
                perfect_alignment += weight
            elif trend == retrace_target:
                w_score += weight * 0.3
                
        alignment_pct = (w_score / TOTAL_WEIGHT) * 100
        confidence = (perfect_alignment / TOTAL_WEIGHT) * 100
        
        quality = 'C'
        if trends['D'] == target and trends['W'] == target:
            if trends['M'] == target:
                quality = 'A+' if alignment_pct > 80 else 'A'
            else:
                quality = 'B+'
        elif trends['D'] == target:
            quality = 'B'
        elif trends['D'] == retrace_target:
            quality = 'B-'
        
        final_score = 0
        if quality in ['A+', 'A']:
            final_score = 3
        elif quality in ['B+', 'B']:
            final_score = 2
        elif quality == 'B-':
            final_score = 1
        
        if trends['H4'] == target and final_score < 3:
            final_score += 0.5
            
        final_score = min(3, final_score)
        
        return {
            'score': final_score,
            'quality': quality,
            'alignment': f"{alignment_pct:.0f}%",
            'confidence': confidence,
            'analysis': trends,
            'timeframe_scores': scores
        }
        
    except Exception as e:
        logger.error(f"Erreur calcul GPS pour {symbol}: {e}")
        return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}, 'confidence': 0}

# ==========================================
# 5. SYSTÈME DE FORCE DES DEVISES
# ==========================================
class CurrencyStrengthSystem:
    @staticmethod
    def scrape_currencystrengthmeter():
        """Scraping depuis currencystrengthmeter.org"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            url = "https://currencystrengthmeter.org/"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scores = {}
            all_text = soup.get_text()
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
            
            for currency in currencies:
                patterns = [
                    rf'{currency}[\s:=-]+(\d+\.?\d*)',
                    rf'{currency.lower()}[\s:=-]+(\d+\.?\d*)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, all_text, re.IGNORECASE)
                    if match:
                        value = float(match.group(1))
                        if value > 10:
                            value = value / 10
                        scores[currency] = value
                        break
            
            if len(scores) >= 4:
                logger.info(f"✅ CurrencyStrengthMeter: {len(scores)} devises")
                return scores, 'currencystrengthmeter'
            
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Erreur scraping CurrencyStrengthMeter: {e}")
            return None, None
    
    @staticmethod
    def scrape_barchart():
        """Scraping depuis barchart.com"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }
            
            url = "https://www.barchart.com/forex/performance"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scores = {}
            all_text = soup.get_text()
            
            forex_pairs = [
                ('EUR', 'USD'), ('GBP', 'USD'), ('USD', 'JPY'), 
                ('USD', 'CHF'), ('AUD', 'USD'), ('USD', 'CAD'), 
                ('NZD', 'USD')
            ]
            
            for base, quote in forex_pairs:
                pair_pattern = rf'{base}/{quote}.*?([+-]?\d+\.?\d*)%'
                match = re.search(pair_pattern, all_text)
                if match:
                    pct = float(match.group(1))
                    scores[base] = scores.get(base, 0) + pct
                    scores[quote] = scores.get(quote, 0) - pct
            
            if len(scores) >= 4:
                vals = list(scores.values())
                min_v, max_v = min(vals), max(vals)
                if max_v != min_v:
                    for k in scores:
                        scores[k] = ((scores[k] - min_v) / (max_v - min_v)) * 10.0
                
                logger.info(f"✅ Barchart: {len(scores)} devises")
                return scores, 'barchart'
            
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Erreur scraping Barchart: {e}")
            return None, None
    
    @staticmethod
    def calculate_matrix(api: OandaClient):
        """Calcul de la matrice avec scraping multi-sources"""
        now = datetime.now(timezone.utc)
        
        if st.session_state.matrix_cache and st.session_state.matrix_timestamp:
            age = (now - st.session_state.matrix_timestamp).total_seconds()
            if age < 900:
                logger.debug("Cache matrice valide")
                return st.session_state.matrix_cache

        with st.spinner("🔄 Force des devises..."):
            all_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
            final_scores = {c: 5.0 for c in all_currencies}
            sources_used = []
            
            scores1, source1 = CurrencyStrengthSystem.scrape_currencystrengthmeter()
            if scores1:
                sources_used.append(source1)
                for curr in all_currencies:
                    if curr in scores1:
                        final_scores[curr] = scores1[curr]
            
            scores2, source2 = CurrencyStrengthSystem.scrape_barchart()
            if scores2:
                sources_used.append(source2)
                if scores1:
                    for curr in all_currencies:
                        if curr in scores1 and curr in scores2:
                            final_scores[curr] = (scores1[curr] + scores2[curr]) / 2
                        elif curr in scores2:
                            final_scores[curr] = scores2[curr]
                else:
                    for curr in all_currencies:
                        if curr in scores2:
                            final_scores[curr] = scores2[curr]
            
            if not sources_used:
                logger.warning("⚠️ Scraping échoué, calcul manuel")
                return CurrencyStrengthSystem.calculate_matrix_fallback(api)
            
            details = {c: [] for c in all_currencies}
            for base in all_currencies:
                for quote in all_currencies:
                    if base != quote:
                        gap = final_scores[base] - final_scores[quote]
                        details[base].append({'vs': quote, 'val': gap})
            
            result = {
                'scores': final_scores,
                'details': details,
                'timestamp': now,
                'sources': sources_used
            }
            
            st.session_state.matrix_cache = result
            st.session_state.matrix_timestamp = now
            
            logger.info(f"✅ Matrice: {', '.join(sources_used)}")
            return result
    
    @staticmethod
    def calculate_matrix_fallback(api: OandaClient):
        """Calcul manuel en fallback"""
        logger.info("🔧 Fallback manuel")
        
        scores = {c: 0.0 for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']}
        details = {c: [] for c in scores.keys()}
        count = 0
        
        for pair in ALL_CROSSES:
            try:
                df = api.get_candles(pair, "D", 5)
                if not df.empty and len(df) >= 2:
                    op = df['open'].iloc[-1]
                    cl = df['close'].iloc[-1]
                    pct = ((cl - op) / op) * 100
                    
                    base, quote = pair.split('_')
                    scores[base] += pct
                    scores[quote] -= pct
                    
                    details[base].append({'vs': quote, 'val': pct})
                    details[quote].append({'vs': base, 'val': -pct})
                    count += 1
            except:
                continue
        
        if count < 20:
            logger.error(f"❌ Fallback échoué: {count}/28 paires")
            return None
        
        vals = list(scores.values())
        if not vals or all(v == 0 for v in vals):
            return None
            
        min_v, max_v = min(vals), max(vals)
        final = {}
        for k, v in scores.items():
            if max_v != min_v:
                norm = ((v - min_v) / (max_v - min_v)) * 10.0
            else:
                norm = 5.0
            final[k] = norm

        now = datetime.now(timezone.utc)
        result = {
            'scores': final,
            'details': details,
            'timestamp': now,
            'sources': ['manual_fallback'],
            'pairs_analyzed': count
        }
        
        st.session_state.matrix_cache = result
        st.session_state.matrix_timestamp = now
        
        logger.info(f"✅ Fallback: {count} paires")
        return result

    @staticmethod
    def get_pair_analysis(matrix, base, quote):
        """Analyse de force relative"""
        if not matrix:
            return 5.0, 5.0, 0.0, []
            
        s_b = matrix['scores'].get(base, 5.0)
        s_q = matrix['scores'].get(quote, 5.0)
        gap = s_b - s_q
        
        map_data = sorted(
            matrix['details'].get(base, []),
            key=lambda x: x['val'],
            reverse=True
        )
        
        return s_b, s_q, gap, map_data

# ==========================================
# 6. INDICATEURS TECHNIQUES M5
# ==========================================
def calculate_atr(df, period=14):
    """ATR"""
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([
            h - l,
            abs(h - c.shift(1)),
            abs(l - c.shift(1))
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
    except Exception as e:
        logger.error(f"Erreur ATR: {e}")
        return 0

def get_rsi_ohlc4(df, length=7):
    """RSI OHLC/4"""
    try:
        ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        delta = ohlc4.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)
    except Exception as e:
        logger.error(f"Erreur RSI: {e}")
        return pd.Series([50] * len(df), index=df.index)

def get_colored_hma(df, length=20):
    """Hull Moving Average"""
    try:
        src = df['close']
        
        def wma(s, l):
            weights = np.arange(1, l+1)
            return s.rolling(l).apply(
                lambda x: np.dot(x, weights) / weights.sum(),
                raw=True
            )
        
        wma1 = wma(src, int(length / 2))
        wma2 = wma(src, length)
        hma = wma(2 * wma1 - wma2, int(np.sqrt(length)))
        
        trend = pd.Series(
            np.where(hma > hma.shift(1), 1, -1),
            index=df.index
        )
        
        return hma, trend
    except Exception as e:
        logger.error(f"Erreur HMA: {e}")
        return df['close'], pd.Series([0] * len(df), index=df.index)

def detect_fvg(df):
    """Fair Value Gaps"""
    try:
        if len(df) < 5:
            return False, None
        
        fvg_bull = (df['low'] > df['high'].shift(2))
        fvg_bear = (df['high'] < df['low'].shift(2))
        
        recent_bull = fvg_bull.iloc[-5:].any()
        recent_bear = fvg_bear.iloc[-5:].any()
        
        fvg_type = None
        if recent_bull:
            fvg_type = 'BULL'
        elif recent_bear:
            fvg_type = 'BEAR'
        
        return (recent_bull or recent_bear), fvg_type
    except Exception as e:
        logger.error(f"Erreur FVG: {e}")
        return False, None

def check_volatility_filter(df, threshold=0.5):
    """Filtre de volatilité"""
    try:
        atr = calculate_atr(df)
        price = df['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        return 0.2 <= atr_pct <= 3.0, atr_pct
    except Exception as e:
        logger.error(f"Erreur volatilité: {e}")
        return True, 0

# ✅ ANALYSE RSI CORRIGÉE
def analyze_rsi_signal(rsi_value, hma_direction):
    """
    Analyse RSI stricte et cohérente
    Returns: (signal_type, rsi_quality, score_bonus)
    """
    config = SCORING_CONFIG
    signal_type = None
    rsi_quality = 'invalid'
    score_bonus = 0
    
    # SIGNAUX BUY
    if hma_direction == 1:
        if config['rsi_optimal_min'] <= rsi_value <= config['rsi_optimal_max']:
            signal_type = 'BUY'
            rsi_quality = 'optimal'
            score_bonus = 0.5
        elif rsi_value < config['rsi_oversold']:
            signal_type = 'BUY'
            rsi_quality = 'oversold'
            score_bonus = 0.3
        elif config['rsi_acceptable_min'] <= rsi_value < config['rsi_optimal_min']:
            signal_type = 'BUY'
            rsi_quality = 'acceptable'
            score_bonus = 0.2
        elif config['rsi_optimal_max'] < rsi_value <= config['rsi_acceptable_max']:
            signal_type = 'BUY'
            rsi_quality = 'weak'
            score_bonus = 0
        else:
            signal_type = None
            rsi_quality = 'rejected_high'
    
    # SIGNAUX SELL
    elif hma_direction == -1:
        if config['rsi_optimal_min'] <= rsi_value <= config['rsi_optimal_max']:
            signal_type = 'SELL'
            rsi_quality = 'optimal'
            score_bonus = 0.5
        elif rsi_value > config['rsi_overbought']:
            signal_type = 'SELL'
            rsi_quality = 'overbought'
            score_bonus = 0.3
        elif config['rsi_optimal_max'] < rsi_value <= config['rsi_acceptable_max']:
            signal_type = 'SELL'
            rsi_quality = 'acceptable'
            score_bonus = 0.2
        elif config['rsi_acceptable_min'] <= rsi_value < config['rsi_optimal_min']:
            signal_type = 'SELL'
            rsi_quality = 'weak'
            score_bonus = 0
        else:
            signal_type = None
            rsi_quality = 'rejected_low'
    
    return signal_type, rsi_quality, score_bonus

# ==========================================
# 7. SCANNER CORRIGÉ
# ==========================================
def run_scan(api, min_score, strict_mode):
    """Scanner avec logique RSI fixée"""
    logger.info(f"Scan - Score min: {min_score}, Strict: {strict_mode}")
    
    matrix = CurrencyStrengthSystem.calculate_matrix(api)
    if not matrix:
        st.error("❌ Impossible de calculer la matrice")
        return [], {}
    
    signals = []
    debug_info = {'total': 0, 'filtered': {}, 'near_misses': []}  # ✅ Ajout near_misses
    pbar = st.progress(0)
    scan_start = datetime.now(timezone.utc)
    
    for i, sym in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        debug_info['total'] += 1
        
        try:
            df = api.get_candles(sym, "M5", 150)
            if df.empty or len(df) < 50:
                debug_info['filtered']['data'] = debug_info['filtered'].get('data', 0) + 1
                continue
            
            signal_time = df['time'].iloc[-1]
            
            # Indicateurs
            rsi = get_rsi_ohlc4(df).iloc[-1]
            hma, trend = get_colored_hma(df)
            hma_val = trend.iloc[-1]
            fvg_present, fvg_type = detect_fvg(df)
            vol_ok, atr_pct = check_volatility_filter(df)
            
            if not vol_ok and strict_mode:
                debug_info['filtered']['volatility'] = debug_info['filtered'].get('volatility', 0) + 1
                continue
            
            # ✅ ANALYSE RSI CORRIGÉE
            signal_type, rsi_quality, rsi_bonus = analyze_rsi_signal(rsi, hma_val)
            
            if not signal_type:
                debug_info['filtered']['rsi_invalid'] = debug_info['filtered'].get('rsi_invalid', 0) + 1
                logger.debug(f"{sym} - RSI rejeté: {rsi:.1f} ({rsi_quality})")
                continue
            
            if strict_mode and rsi_quality == 'weak':
                debug_info['filtered']['rsi_weak'] = debug_info['filtered'].get('rsi_weak', 0) + 1
                logger.debug(f"{sym} - RSI faible en strict: {rsi:.1f}")
                continue
            
            # GPS
            mtf = calculate_mtf_score_gps(api, sym, signal_type)
            
            if strict_mode and mtf['score'] < 1.0:
                debug_info['filtered']['gps_weak'] = debug_info['filtered'].get('gps_weak', 0) + 1
                continue
            
            # Fondamental
            cs_data = {}
            fundamental_score = 0
            is_forex = sym in ALL_CROSSES
            
            if is_forex:
                base, quote = sym.split('_')
                sb, sq, gap, map_d = CurrencyStrengthSystem.get_pair_analysis(matrix, base, quote)
                cs_data = {
                    'sb': sb, 'sq': sq, 'gap': gap,
                    'map': map_d, 'base': base, 'quote': quote
                }
                
                if signal_type == 'BUY':
                    if sb >= 6.5 and sq <= 3.5 and gap >= 3.0:
                        fundamental_score = 3.0
                    elif sb >= 5.5 and sq <= 4.5 and gap >= 1.0:
                        fundamental_score = 2.0
                    elif sb >= 5.0 and gap >= 0.5:
                        fundamental_score = 1.5
                    elif gap > 0:
                        fundamental_score = 1.0
                else:
                    if sq >= 6.5 and sb <= 3.5 and gap <= -3.0:
                        fundamental_score = 3.0
                    elif sq >= 5.5 and sb <= 4.5 and gap <= -1.0:
                        fundamental_score = 2.0
                    elif sq >= 5.0 and gap <= -0.5:
                        fundamental_score = 1.5
                    elif gap < 0:
                        fundamental_score = 1.0
                
                if strict_mode and fundamental_score < 1.0:
                    debug_info['filtered']['fundamental'] = debug_info['filtered'].get('fundamental', 0) + 1
                    continue
            else:
                fundamental_score = 2.0
            
            # ✅ Score technique avec bonus RSI corrigé
            technical_score = 2.5 + rsi_bonus
            
            # Bonus FVG
            if fvg_present:
                if (signal_type == 'BUY' and fvg_type == 'BULL') or \
                   (signal_type == 'SELL' and fvg_type == 'BEAR'):
                    technical_score += 0.5
            
            # Score final pondéré
            gps_weighted = mtf['score'] * (SCORING_CONFIG['gps_weight'] / 0.3) * 10
            fund_weighted = fundamental_score * (SCORING_CONFIG['fundamental_weight'] / 0.3) * 10
            tech_weighted = technical_score * (SCORING_CONFIG['technical_weight'] / 0.25) * 10
            
            final_score = (gps_weighted + fund_weighted + tech_weighted) / 10
            final_score = min(10.0, final_score)
            
            if final_score < min_score:
                debug_info['filtered']['score_low'] = debug_info['filtered'].get('score_low', 0) + 1
                # ✅ Capturer les "presque signaux" pour debug
                if final_score >= min_score - 1.0:  # À moins de 1 point du seuil
                    debug_info['near_misses'].append({
                        'symbol': sym,
                        'type': signal_type,
                        'score': final_score,
                        'rsi': rsi,
                        'rsi_quality': rsi_quality,
                        'gps_quality': mtf['quality'],
                        'gap': cs_data.get('gap', 0) if cs_data else 0
                    })
                continue
            
            # Risk Management
            atr = calculate_atr(df)
            price = df['close'].iloc[-1]
            sl_dist = atr * 1.8
            tp_dist = atr * 3.0
            
            if signal_type == 'BUY':
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist
            
            rr_ratio = tp_dist / sl_dist
            
            signals.append({
                'symbol': sym,
                'type': signal_type,
                'price': price,
                'score': final_score,
                'quality': mtf['quality'],
                'atr': atr,
                'atr_pct': atr_pct,
                'mtf': mtf,
                'cs': cs_data,
                'fundamental_score': fundamental_score,
                'technical_score': technical_score,
                'fvg': fvg_present,
                'fvg_type': fvg_type,
                'rsi': rsi,
                'rsi_quality': rsi_quality,
                'sl': sl,
                'tp': tp,
                'rr': rr_ratio,
                'time': signal_time,
                'scan_time': scan_start
            })
            
            logger.info(f"✅ {sym} {signal_type} @ {price:.5f} (Score: {final_score:.1f}, RSI: {rsi:.1f}/{rsi_quality})")
            
        except Exception as e:
            debug_info['filtered']['error'] = debug_info['filtered'].get('error', 0) + 1
            logger.error(f"Erreur {sym}: {e}")
    
    pbar.empty()
    logger.info(f"📊 Scan: {len(signals)} signaux")
    return signals, debug_info

# ==========================================
# 8. AFFICHAGE CORRIGÉ
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

def format_timestamp(dt):
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except:
        return "N/A"

def display_sig(s):
    """Affichage avec badges RSI corrigés"""
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score']
    if sc >= 9.0:
        label = "💎 LEGENDARY"
    elif sc >= 8.0:
        label = "⭐ EXCELLENT"
    elif sc >= 7.0:
        label = "✅ BON"
    elif sc >= 6.0:
        label = "📊 CORRECT"
    else:
        label = "⚠️ MOYEN"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        
        st.markdown(f"""
        <div class="timestamp-box">
            📅 Signal: {format_timestamp(s['time'])} • Scan: {format_timestamp(s['scan_time'])}
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # ✅ BADGES CORRIGÉS
        badges = []
        
        if s['fvg']:
            fvg_match = (is_buy and s['fvg_type'] == 'BULL') or (not is_buy and s['fvg_type'] == 'BEAR')
            if fvg_match:
                badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
            else:
                badges.append("<span class='badge-fvg' style='opacity:0.7'>🦅 FVG</span>")
        
        q_col = "#10b981" if s['quality'] in ['A+', 'A'] else "#f59e0b" if s['quality'] in ['B+', 'B'] else "#ef4444"
        badges.append(f"<span class='badge-gps' style='background:{q_col}'>🛡️ GPS {s['quality']}</span>")
        
        # ✅ BADGE RSI AVEC VALEUR
        rsi_val = s['rsi']
        rsi_qual = s['rsi_quality']
        
        if rsi_qual == 'optimal':
            badges.append(f"<span class='badge-gps' style='background:#10b981'>📊 RSI OPTIMAL ({rsi_val:.1f})</span>")
        elif rsi_qual == 'oversold':
            badges.append(f"<span class='badge-gps' style='background:#3b82f6'>📊 SURVENTE ({rsi_val:.1f})</span>")
        elif rsi_qual == 'overbought':
            badges.append(f"<span class='badge-gps' style='background:#8b5cf6'>📊 SURACHAT ({rsi_val:.1f})</span>")
        elif rsi_qual == 'acceptable':
            badges.append(f"<span class='badge-gps' style='background:#f59e0b'>📊 RSI OK ({rsi_val:.1f})</span>")
        elif rsi_qual == 'weak':
            badges.append(f"<span style='background:#64748b;color:white;padding:4px 10px;border-radius:6px;font-size:0.75em;'>📊 RSI FAIBLE ({rsi_val:.1f})</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        st.write("")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{sc:.1f}/10", label)
        c2.metric("GPS", s['mtf']['alignment'])
        c3.metric("RSI M5", f"{s['rsi']:.1f}")
        c4.metric("Confiance", f"{s['mtf'].get('confidence', 0):.0f}%")
        
        if s['cs']:
            st.markdown("---")
            st.markdown("### 📊 Analyse Fondamentale")
            
            f1, f2 = st.columns([1, 2])
            
            with f1:
                st.markdown("**Force Devises**")
                base, quote = s['cs']['base'], s['cs']['quote']
                sb, sq, gap = s['cs']['sb'], s['cs']['sq'], s['cs']['gap']
                
                cb = "#10b981" if sb >= 6.5 else "#f59e0b" if sb >= 5.5 else "#ef4444"
                cq = "#10b981" if sq >= 6.5 else "#f59e0b" if sq >= 5.5 else "#ef4444"
                
                draw_mini_meter(base, sb, cb)
                st.write("")
                draw_mini_meter(quote, sq, cq)
                st.write("")
                
                gap_color = "#10b981" if gap > 0 else "#ef4444"
                gap_arrow = "⬆️" if gap > 0 else "⬇️"
                st.markdown(f"""
                <div style='text-align:center;margin-top:10px;'>
                    <div style='color:#94a3b8;font-size:0.8em;'>Écart</div>
                    <div style='color:{gap_color};font-size:1.3em;font-weight:bold;'>
                        {gap_arrow} {abs(gap):.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with f2:
                st.markdown("**Market Map**")
                cols = st.columns(6)
                for i, item in enumerate(s['cs']['map'][:6]):
                    with cols[i]:
                        val = item['val']
                        arrow = "▲" if val > 0 else "▼"
                        cl = "#10b981" if val > 0 else "#ef4444"
                        st.markdown(f"""
                        <div style='text-align:center;'>
                            <div style='font-size:0.7em;color:#94a3b8;'>{item['vs']}</div>
                            <div style='color:{cl};font-size:1.2em;'>{arrow}</div>
                            <div style='font-size:0.7em;color:{cl};'>{val:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🛡️ GPS Multi-Timeframe")
        
        mtf_analysis = s['mtf']['analysis']
        timeframes = ['M', 'W', 'D', 'H4', 'H1']
        tf_labels = {'M': 'Mensuel', 'W': 'Hebdo', 'D': 'Daily', 'H4': '4H', 'H1': '1H'}
        
        cols = st.columns(5)
        for i, tf in enumerate(timeframes):
            with cols[i]:
                trend = mtf_analysis.get(tf, 'N/A')
                
                if 'Bullish' in trend:
                    icon = "🟢"
                    color = "#10b981"
                elif 'Bearish' in trend:
                    icon = "🔴"
                    color = "#ef4444"
                elif 'Retracement' in trend:
                    icon = "🟡"
                    color = "#f59e0b"
                else:
                    icon = "⚪"
                    color = "#94a3b8"
                
                st.markdown(f"""
                <div style='text-align:center;background:rgba(255,255,255,0.03);padding:10px;border-radius:8px;'>
                    <div style='font-size:0.75em;color:#94a3b8;'>{tf_labels[tf]}</div>
                    <div style='font-size:1.5em;margin:5px 0;'>{icon}</div>
                    <div style='font-size:0.7em;color:{color};font-weight:600;'>{trend}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚖️ Risk Management")
        
        r1, r2, r3 = st.columns(3)
        
        r1.markdown(f"""
        <div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;margin-bottom:5px;'>STOP LOSS</div>
            <div style='color:#ef4444;font-weight:bold;font-size:1.2em;'>{s['sl']:.5f}</div>
            <div style='color:#94a3b8;font-size:0.7em;margin-top:5px;'>1.8x ATR</div>
        </div>
        """, unsafe_allow_html=True)
        
        r2.markdown(f"""
        <div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;margin-bottom:5px;'>TAKE PROFIT</div>
            <div style='color:#10b981;font-weight:bold;font-size:1.2em;'>{s['tp']:.5f}</div>
            <div style='color:#94a3b8;font-size:0.7em;margin-top:5px;'>3x ATR</div>
        </div>
        """, unsafe_allow_html=True)
        
        r3.markdown(f"""
        <div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;margin-bottom:5px;'>RISK:REWARD</div>
            <div style='color:#3b82f6;font-weight:bold;font-size:1.2em;'>1:{s['rr']:.2f}</div>
            <div style='color:#94a3b8;font-size:0.7em;margin-top:5px;'>Ratio optimal</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Position: 1-2% du capital")

# ==========================================
# 9. APPLICATION PRINCIPALE
# ==========================================
def main():
    st.title("💎 Bluestar SNP3 GPS")
    st.markdown("**Scanner Institutionnel**: GPS MTF + Force Fondamentale + Sniper M5")
    
    # ✅ Info box avec recommandations
    st.info("🎯 **Recommandations**: Score Min 6.0 + Mode Sniper OFF pour commencer. "
            "Activez le mode strict une fois familiarisé avec les signaux.")
    
    with st.expander("⚙️ Paramètres", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider("Score Min", 5.0, 10.0, 6.0, 0.5,
                help="Score minimum pour afficher un signal (6.0 recommandé)")
        
        with col2:
            strict_mode = st.checkbox("🔥 Mode Sniper", False,
                help="GPS ≥ 1.0 + Force alignée + Rejette RSI faible (désactivé par défaut)")
        
        with st.expander("ℹ️ Scoring", expanded=False):
            st.markdown("""
            **Score (0-10)**:
            - 🛡️ **GPS (40%)**: M/W/D/H4/H1
            - 📊 **Fondamental (35%)**: Force devises
            - 📈 **Technique (25%)**: RSI + HMA + FVG
            
            **Qualités GPS**: A+/A (3/3) | B+/B (2/3) | B-/C (1/3)
            
            **RSI (corrigé)**:
            - 🟢 **Optimal**: 45-55 (zone neutre)
            - 🔵 **Survente**: <30 (rebond)
            - 🟣 **Surachat**: >70 (retournement)
            - 🟡 **Acceptable**: 35-45 (BUY) / 55-65 (SELL)
            - ⚪ **Faible**: limites acceptables
            - 🔴 **Rejeté**: hors zones
            
            **Mode Sniper**: Rejette RSI faible + GPS < 1.0 + Gap < 0.5
            """)
    
    if st.button("🚀 LANCER LE SCAN", type="primary"):
        st.session_state.cache_timestamps = {}
        st.session_state.matrix_timestamp = None
        
        try:
            api = OandaClient()
            
            with st.spinner("🔍 Analyse..."):
                results, debug_info = run_scan(api, min_score, strict_mode)
            
            if not results:
                st.warning("⚠️ Aucune opportunité")
                
                with st.expander("🔍 Filtrage", expanded=True):
                    st.markdown(f"**Assets**: {debug_info['total']}")
                    st.markdown("**Raisons**:")
                    for reason, count in sorted(debug_info['filtered'].items(), key=lambda x: x[1], reverse=True):
                        labels = {
                            'data': 'Données insuffisantes',
                            'volatility': 'Volatilité anormale',
                            'rsi_invalid': 'RSI rejeté (hors zones)',
                            'rsi_weak': 'RSI faible (mode strict)',
                            'gps_weak': 'GPS trop faible',
                            'fundamental': 'Fondamental faible',
                            'score_low': 'Score < minimum',
                            'error': 'Erreurs techniques'
                        }
                        label = labels.get(reason, reason)
                        st.write(f"- **{label}**: {count}")
                
                st.info("💡 Réduisez le score ou désactivez le mode strict")
            else:
                st.success(f"✅ {len(results)} Signal{'s' if len(results) > 1 else ''}")
                
                results.sort(key=lambda x: x['score'], reverse=True)
                
                with st.expander("📊 Statistiques", expanded=False):
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    
                    avg_score = np.mean([s['score'] for s in results])
                    buy_sig = sum(1 for s in results if s['type'] == 'BUY')
                    sell_sig = len(results) - buy_sig
                    avg_gps = np.mean([s['mtf']['score'] for s in results])
                    
                    sc1.metric("Score Moy", f"{avg_score:.1f}/10")
                    sc2.metric("BUY", buy_sig)
                    sc3.metric("SELL", sell_sig)
                    sc4.metric("GPS Moy", f"{avg_gps:.1f}/3")
                    
                    if st.session_state.matrix_cache:
                        sources = st.session_state.matrix_cache.get('sources', [])
                        if sources:
                            st.success(f"📡 Sources: {', '.join(sources)}")
                
                for sig in results:
                    display_sig(sig)
                    
        except Exception as e:
            logger.error(f"Erreur scan: {e}")
            st.error(f"❌ Erreur: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#64748b;font-size:0.85em;padding:20px 0;'>
        💎 Bluestar SNP3 GPS v2.1 CORRIGÉ<br>
        <span style='font-size:0.75em;'>RSI Strict • GPS MTF • Smart Money • Force Fondamentale</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
