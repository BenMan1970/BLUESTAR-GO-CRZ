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

# Configuration du scoring amélioré
SCORING_CONFIG = {
    'gps_weight': 0.40,      # 40% GPS MTF
    'fundamental_weight': 0.35,  # 35% Force devises
    'technical_weight': 0.25,    # 25% Technique M5
    'min_gps_quality': 'C',      # Qualité GPS minimale (abaissé de B à C)
    'min_fundamental_gap': 0.5,  # Gap minimal de force (abaissé de 1.0 à 0.5)
    'rsi_overbought': 75,        # Zone de surachat (augmenté de 70 à 75)
    'rsi_oversold': 25           # Zone de survente (abaissé de 30 à 25)
}

# ==========================================
# 3. CLIENT API AMÉLIORÉ
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
# 4. GPS MTF INSTITUTIONNEL (AMÉLIORÉ)
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
    """Analyse macro (Mensuel/Hebdo) - Tendance de fond"""
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
    """Analyse Daily - Tendance primaire"""
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
    
    # Retracement (opportunité)
    if curr < sma200 and ema50 > sma200:
        return "Retracement Bull", 55
    if curr > sma200 and ema50 < sma200:
        return "Retracement Bear", 55
    
    # Tendance moyenne
    if curr > sma200:
        return "Bullish", 50
    if curr < sma200:
        return "Bearish", 50
    
    return "Range", 35

def calc_institutional_trend_4h(df):
    """Analyse H4 - Tendance intermédiaire"""
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
    """Analyse H1 - Tendance court terme"""
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
    """Calcul du score GPS multi-timeframe amélioré"""
    try:
        # Récupération des données
        df_d = api.get_candles(symbol, "D", count=500)
        df_h4 = api.get_candles(symbol, "H4", count=200)
        df_h1 = api.get_candles(symbol, "H1", count=200)
        
        if df_d.empty or df_h4.empty or df_h1.empty:
            logger.warning(f"Données MTF incomplètes pour {symbol}")
            return {'score': 0, 'quality': 'N/A', 'alignment': '0%', 'analysis': {}, 'confidence': 0}

        d_res = df_d.copy()
        d_res.set_index('time', inplace=True)
        
        # Création des timeframes supérieurs
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

        # Analyse de chaque timeframe
        trends = {}
        scores = {}
        
        trends['M'], scores['M'] = calc_institutional_trend_macro(df_m)
        trends['W'], scores['W'] = calc_institutional_trend_macro(df_w)
        trends['D'], scores['D'] = calc_institutional_trend_daily(df_d)
        trends['H4'], scores['H4'] = calc_institutional_trend_4h(df_h4)
        trends['H1'], scores['H1'] = calc_institutional_trend_intraday(df_h1)
        
        # Calcul du score pondéré
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
                w_score += weight * 0.3  # Bonus réduit pour retracement
                
        # Calcul de l'alignement en pourcentage
        alignment_pct = (w_score / TOTAL_WEIGHT) * 100
        confidence = (perfect_alignment / TOTAL_WEIGHT) * 100
        
        # Attribution de la qualité GPS
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
        
        # Score final normalisé (0-3)
        final_score = 0
        if quality in ['A+', 'A']:
            final_score = 3
        elif quality in ['B+', 'B']:
            final_score = 2
        elif quality == 'B-':
            final_score = 1
        
        # Bonus si H4 aligné
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
# 5. SYSTÈME DE FORCE DES DEVISES (WEB SCRAPING)
# ==========================================
import requests
from bs4 import BeautifulSoup
import re

class CurrencyStrengthSystem:
    @staticmethod
    def scrape_currencystrengthmeter():
        """Scraping depuis currencystrengthmeter.org"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            url = "https://currencystrengthmeter.org/"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scores = {}
            
            # Méthode 1: Recherche dans tous les éléments textuels
            all_text = soup.get_text()
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
            
            for currency in currencies:
                # Patterns possibles: "USD: 7.5", "USD 7.5", "USD - 7.5", etc.
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
                logger.info(f"✅ CurrencyStrengthMeter: {len(scores)} devises scrapées")
                return scores, 'currencystrengthmeter'
            
            logger.warning("⚠️ CurrencyStrengthMeter: données insuffisantes")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Erreur scraping CurrencyStrengthMeter: {e}")
            return None, None
    
    @staticmethod
    def scrape_barchart():
        """Scraping depuis barchart.com/forex/market-map"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            url = "https://www.barchart.com/forex/performance"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            scores = {}
            
            # Recherche dans le texte complet
            all_text = soup.get_text()
            
            # Extraction des paires forex et leurs variations
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
            
            # Normalisation 0-10
            if len(scores) >= 4:
                vals = list(scores.values())
                min_v, max_v = min(vals), max(vals)
                if max_v != min_v:
                    for k in scores:
                        scores[k] = ((scores[k] - min_v) / (max_v - min_v)) * 10.0
                
                logger.info(f"✅ Barchart: {len(scores)} devises scrapées")
                return scores, 'barchart'
            
            logger.warning("⚠️ Barchart: données insuffisantes")
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Erreur scraping Barchart: {e}")
            return None, None
    
    @staticmethod
    def calculate_matrix(api: OandaClient):
        """Calcul de la matrice de force avec scraping multi-sources"""
        now = datetime.now(timezone.utc)
        
        # Vérifier le cache (15 minutes)
        if st.session_state.matrix_cache and st.session_state.matrix_timestamp:
            age = (now - st.session_state.matrix_timestamp).total_seconds()
            if age < 900:  # 15 minutes
                logger.debug("Utilisation du cache de la matrice")
                return st.session_state.matrix_cache

        with st.spinner("🔄 Récupération de la force des devises..."):
            all_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']
            final_scores = {c: 5.0 for c in all_currencies}  # Valeur par défaut
            sources_used = []
            
            # Tentative 1: CurrencyStrengthMeter
            scores1, source1 = CurrencyStrengthSystem.scrape_currencystrengthmeter()
            if scores1:
                sources_used.append(source1)
                for curr in all_currencies:
                    if curr in scores1:
                        final_scores[curr] = scores1[curr]
                logger.info("✅ Données de CurrencyStrengthMeter utilisées")
            
            # Tentative 2: Barchart (si première source échoue ou pour moyenne)
            scores2, source2 = CurrencyStrengthSystem.scrape_barchart()
            if scores2:
                sources_used.append(source2)
                if scores1:  # Moyenne des deux sources
                    for curr in all_currencies:
                        if curr in scores1 and curr in scores2:
                            final_scores[curr] = (scores1[curr] + scores2[curr]) / 2
                        elif curr in scores2:
                            final_scores[curr] = scores2[curr]
                else:  # Utiliser uniquement Barchart
                    for curr in all_currencies:
                        if curr in scores2:
                            final_scores[curr] = scores2[curr]
                logger.info("✅ Données de Barchart utilisées")
            
            # Si aucune source ne fonctionne, fallback sur calcul manuel
            if not sources_used:
                logger.warning("⚠️ Scraping échoué, utilisation du calcul manuel")
                return CurrencyStrengthSystem.calculate_matrix_fallback(api)
            
            # Génération des détails (market map)
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
            
            # Mise en cache
            st.session_state.matrix_cache = result
            st.session_state.matrix_timestamp = now
            
            logger.info(f"✅ Matrice générée depuis: {', '.join(sources_used)}")
            return result
    
    @staticmethod
    def calculate_matrix_fallback(api: OandaClient):
        """Calcul manuel en fallback si scraping échoue"""
        logger.info("🔧 Fallback: calcul manuel de la force des devises")
        
        scores = {c: 0.0 for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']}
        details = {c: [] for c in scores.keys()}
        count = 0
        failed = 0
        
        for pair in ALL_CROSSES:
            try:
                # Récupérer 5 bougies Daily au lieu de 2 pour plus de fiabilité
                df = api.get_candles(pair, "D", 5)
                if not df.empty and len(df) >= 2:
                    # Utiliser la dernière bougie complète
                    op = df['open'].iloc[-1]
                    cl = df['close'].iloc[-1]
                    
                    # Calcul du % de variation
                    pct = ((cl - op) / op) * 100
                    
                    base, quote = pair.split('_')
                    scores[base] += pct
                    scores[quote] -= pct
                    
                    details[base].append({'vs': quote, 'val': pct})
                    details[quote].append({'vs': base, 'val': -pct})
                    count += 1
                else:
                    failed += 1
                    logger.debug(f"Données insuffisantes pour {pair}")
                    
            except Exception as e:
                failed += 1
                logger.debug(f"Erreur pour {pair}: {e}")
                continue
        
        logger.info(f"📊 Fallback: {count} paires analysées, {failed} échecs")
        
        # Validation: au moins 20 paires (70% de 28 paires)
        if count < 20:
            logger.error(f"❌ Fallback échoué: seulement {count}/28 paires valides")
            return None
        
        # Normalisation 0-10
        vals = list(scores.values())
        if not vals or all(v == 0 for v in vals):
            logger.error("❌ Toutes les valeurs sont nulles")
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
        
        logger.info(f"✅ Fallback réussi: {count} paires analysées")
        return result

    @staticmethod
    def get_pair_analysis(matrix, base, quote):
        """Analyse de la force relative d'une paire"""
        if not matrix:
            return 5.0, 5.0, 0.0, []
            
        s_b = matrix['scores'].get(base, 5.0)
        s_q = matrix['scores'].get(quote, 5.0)
        gap = s_b - s_q
        
        # Tri de la market map par performance
        map_data = sorted(
            matrix['details'].get(base, []),
            key=lambda x: x['val'],
            reverse=True
        )
        
        return s_b, s_q, gap, map_data

# ==========================================
# 6. INDICATEURS TECHNIQUES M5 (OPTIMISÉS)
# ==========================================
def calculate_atr(df, period=14):
    """Calcul de l'ATR (Average True Range)"""
    try:
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([
            h - l,
            abs(h - c.shift(1)),
            abs(l - c.shift(1))
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
    except Exception as e:
        logger.error(f"Erreur calcul ATR: {e}")
        return 0

def get_rsi_ohlc4(df, length=7):
    """RSI basé sur OHLC/4 pour une meilleure représentation"""
    try:
        ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        delta = ohlc4.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)
    except Exception as e:
        logger.error(f"Erreur calcul RSI: {e}")
        return pd.Series([50] * len(df), index=df.index)

def get_colored_hma(df, length=20):
    """Hull Moving Average avec détection de tendance"""
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
        logger.error(f"Erreur calcul HMA: {e}")
        return df['close'], pd.Series([0] * len(df), index=df.index)

def detect_fvg(df):
    """Détection des Fair Value Gaps (zones institutionnelles)"""
    try:
        if len(df) < 5:
            return False, None
        
        # FVG haussier : low actuel > high il y a 2 bougies
        fvg_bull = (df['low'] > df['high'].shift(2))
        
        # FVG baissier : high actuel < low il y a 2 bougies
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
        logger.error(f"Erreur détection FVG: {e}")
        return False, None

def check_volatility_filter(df, threshold=0.5):
    """Filtre de volatilité pour éviter les périodes trop calmes ou agitées"""
    try:
        atr = calculate_atr(df)
        price = df['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        
        # ATR entre 0.2% et 3% considéré comme acceptable (élargi de 0.3-2%)
        return 0.2 <= atr_pct <= 3.0, atr_pct
    except Exception as e:
        logger.error(f"Erreur filtre volatilité: {e}")
        return True, 0

# ==========================================
# 7. SCANNER UNIFIÉ (AMÉLIORÉ)
# ==========================================
def run_scan(api, min_score, strict_mode):
    """Scanner principal avec logique améliorée"""
    logger.info(f"Démarrage scan - Score min: {min_score}, Mode strict: {strict_mode}")
    
    # 1. Calcul de la matrice fondamentale
    matrix = CurrencyStrengthSystem.calculate_matrix(api)
    if not matrix:
        st.error("❌ Impossible de calculer la force des devises")
        return []
    
    signals = []
    debug_info = {'total': 0, 'filtered': {}}
    pbar = st.progress(0)
    scan_start = datetime.now(timezone.utc)
    
    for i, sym in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        debug_info['total'] += 1
        
        try:
            # 2. Données M5 pour l'entrée
            df = api.get_candles(sym, "M5", 150)
            if df.empty or len(df) < 50:
                debug_info['filtered']['data_insufficient'] = debug_info['filtered'].get('data_insufficient', 0) + 1
                logger.debug(f"Données insuffisantes pour {sym}")
                continue
            
            # Horodatage du signal
            signal_time = df['time'].iloc[-1]
            
            # 3. Indicateurs techniques M5
            rsi = get_rsi_ohlc4(df).iloc[-1]
            hma, trend = get_colored_hma(df)
            hma_val = trend.iloc[-1]
            fvg_present, fvg_type = detect_fvg(df)
            vol_ok, atr_pct = check_volatility_filter(df)
            
            # Filtre volatilité (plus permissif)
            if not vol_ok and strict_mode:
                debug_info['filtered']['volatility'] = debug_info['filtered'].get('volatility', 0) + 1
                logger.debug(f"{sym} filtré: volatilité anormale ({atr_pct:.2f}%)")
                continue
            
            # 4. Détection du signal M5 avec RSI amélioré
            signal_type = None
            rsi_quality = 'normal'
            
            if hma_val == 1:  # HMA haussier
                if 25 < rsi < 75:  # Zone élargie
                    signal_type = 'BUY'
                    if 40 < rsi < 60:
                        rsi_quality = 'optimal'
                elif rsi <= 25:  # Survente
                    signal_type = 'BUY'
                    rsi_quality = 'oversold'
                    
            elif hma_val == -1:  # HMA baissier
                if 25 < rsi < 75:  # Zone élargie
                    signal_type = 'SELL'
                    if 40 < rsi < 60:
                        rsi_quality = 'optimal'
                elif rsi >= 75:  # Surachat
                    signal_type = 'SELL'
                    rsi_quality = 'overbought'
            
            if not signal_type:
                debug_info['filtered']['no_signal_m5'] = debug_info['filtered'].get('no_signal_m5', 0) + 1
                logger.debug(f"{sym} - Pas de signal M5 (RSI:{rsi:.1f}, HMA:{hma_val})")
                continue
            
            # 5. Analyse GPS MTF (critique)
            mtf = calculate_mtf_score_gps(api, sym, signal_type)
            
            # Filtre GPS en mode strict
            if strict_mode and mtf['score'] < 1.0:  # Abaissé de 1.5 à 1.0
                debug_info['filtered']['gps_weak'] = debug_info['filtered'].get('gps_weak', 0) + 1
                logger.debug(f"{sym} filtré: GPS insuffisant ({mtf['quality']})")
                continue
            
            # 6. Analyse fondamentale (CSM)
            cs_data = {}
            fundamental_score = 0
            is_forex = sym in ALL_CROSSES
            
            if is_forex:
                base, quote = sym.split('_')
                sb, sq, gap, map_d = CurrencyStrengthSystem.get_pair_analysis(matrix, base, quote)
                cs_data = {
                    'sb': sb,
                    'sq': sq,
                    'gap': gap,
                    'map': map_d,
                    'base': base,
                    'quote': quote
                }
                
                # Scoring fondamental amélioré (plus permissif)
                if signal_type == 'BUY':
                    if sb >= 6.5 and sq <= 3.5 and gap >= 3.0:
                        fundamental_score = 3.0  # Alignement parfait
                    elif sb >= 5.5 and sq <= 4.5 and gap >= 1.0:
                        fundamental_score = 2.0  # Bon alignement
                    elif sb >= 5.0 and gap >= 0.5:  # Critère assoupli
                        fundamental_score = 1.5  # Alignement faible
                    elif gap > 0:
                        fundamental_score = 1.0  # Minimal
                else:  # SELL
                    if sq >= 6.5 and sb <= 3.5 and gap <= -3.0:
                        fundamental_score = 3.0
                    elif sq >= 5.5 and sb <= 4.5 and gap <= -1.0:
                        fundamental_score = 2.0
                    elif sq >= 5.0 and gap <= -0.5:  # Critère assoupli
                        fundamental_score = 1.5
                    elif gap < 0:
                        fundamental_score = 1.0
                
                # Filtre strict fondamental (assoupli)
                if strict_mode and fundamental_score < 1.0:  # Abaissé de 1.5 à 1.0
                    debug_info['filtered']['fundamental_weak'] = debug_info['filtered'].get('fundamental_weak', 0) + 1
                    logger.debug(f"{sym} filtré: fondamental faible (gap={gap:.2f})")
                    continue
            else:
                # Or/Indices : scoring par défaut
                fundamental_score = 2.0
            
            # 7. Bonus technique
            technical_score = 2.5  # Base
            if rsi_quality == 'optimal':
                technical_score += 0.5
            elif rsi_quality in ['oversold', 'overbought']:
                technical_score += 0.3
            
            # Bonus FVG aligné
            if fvg_present:
                if (signal_type == 'BUY' and fvg_type == 'BULL') or \
                   (signal_type == 'SELL' and fvg_type == 'BEAR'):
                    technical_score += 1.0
            
            # 8. Calcul du score final pondéré
            gps_weighted = mtf['score'] * (SCORING_CONFIG['gps_weight'] / 0.3) * 10
            fund_weighted = fundamental_score * (SCORING_CONFIG['fundamental_weight'] / 0.3) * 10
            tech_weighted = technical_score * (SCORING_CONFIG['technical_weight'] / 0.25) * 10
            
            final_score = (gps_weighted + fund_weighted + tech_weighted) / 10
            final_score = min(10.0, final_score)
            
            # Filtre score minimum
            if final_score < min_score:
                debug_info['filtered']['score_low'] = debug_info['filtered'].get('score_low', 0) + 1
                logger.debug(f"{sym} - Score insuffisant: {final_score:.1f}/{min_score}")
                continue
            
            # 9. Calcul du Risk Management
            atr = calculate_atr(df)
            price = df['close'].iloc[-1]
            
            # Stop Loss : 1.8x ATR
            sl_dist = atr * 1.8
            # Take Profit : 3x ATR
            tp_dist = atr * 3.0
            
            if signal_type == 'BUY':
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist
            
            rr_ratio = tp_dist / sl_dist
            
            # 10. Ajout du signal validé
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
            
            logger.info(f"✅ Signal validé: {sym} {signal_type} @ {price:.5f} (Score: {final_score:.1f})")
            
        except Exception as e:
            debug_info['filtered']['error'] = debug_info['filtered'].get('error', 0) + 1
            logger.error(f"Erreur lors de l'analyse de {sym}: {e}")
            continue
    
    pbar.empty()
    
    # Log des statistiques de filtrage
    logger.info(f"📊 Scan terminé: {len(signals)} signaux trouvés")
    logger.info(f"📊 Assets analysés: {debug_info['total']}")
    for reason, count in debug_info['filtered'].items():
        logger.info(f"   - Filtré ({reason}): {count}")
    
    return signals, debug_info

# ==========================================
# 8. AFFICHAGE (DESIGN CONSERVÉ + AMÉLIORATIONS)
# ==========================================
def draw_mini_meter(label, val, color):
    """Jauge compacte de force"""
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
    """Formatage élégant de l'horodatage"""
    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except:
        return "N/A"

def display_sig(s):
    """Affichage d'un signal avec le design original"""
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    # Label de qualité
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

    # En-tête du signal
    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        
        # Horodatage
        st.markdown(f"""
        <div class="timestamp-box">
            📅 Signal détecté le {format_timestamp(s['time'])} • Scan effectué à {format_timestamp(s['scan_time'])}
        </div>
        """, unsafe_allow_html=True)
        
        # Carte principale
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
        
        # Badges
        badges = []
        
        # Badge FVG
        if s['fvg']:
            fvg_match = (is_buy and s['fvg_type'] == 'BULL') or (not is_buy and s['fvg_type'] == 'BEAR')
            if fvg_match:
                badges.append("<span class='badge-fvg'>🦅 SMART MONEY ALIGNED</span>")
            else:
                badges.append("<span class='badge-fvg' style='opacity:0.7'>🦅 FVG DÉTECTÉ</span>")
        
        # Badge GPS
        q_col = "#10b981" if s['quality'] in ['A+', 'A'] else "#f59e0b" if s['quality'] in ['B+', 'B'] else "#ef4444"
        badges.append(f"<span class='badge-gps' style='background:{q_col}'>🛡️ GPS {s['quality']}</span>")
        
        # Badge RSI
        if s['rsi_quality'] == 'optimal':
            badges.append("<span class='badge-gps' style='background:#3b82f6'>📊 RSI OPTIMAL</span>")
        elif s['rsi_quality'] in ['oversold', 'overbought']:
            badges.append("<span class='badge-gps' style='background:#8b5cf6'>📊 RSI EXTRÊME</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.write("")
        
        # Métriques principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Total", f"{sc:.1f}/10", delta=label, delta_color="off")
        c2.metric("Alignement GPS", s['mtf']['alignment'])
        c3.metric("RSI M5", f"{s['rsi']:.1f}")
        c4.metric("Confiance GPS", f"{s['mtf'].get('confidence', 0):.0f}%")
        
        # --- ANALYSE FONDAMENTALE (si disponible) ---
        if s['cs']:
            st.markdown("---")
            st.markdown("### 📊 Analyse Fondamentale")
            
            f1, f2 = st.columns([1, 2])
            
            with f1:
                st.markdown("**Force Relative des Devises**")
                base = s['cs']['base']
                quote = s['cs']['quote']
                sb = s['cs']['sb']
                sq = s['cs']['sq']
                gap = s['cs']['gap']
                
                # Couleurs selon la force
                cb = "#10b981" if sb >= 6.5 else "#f59e0b" if sb >= 5.5 else "#ef4444"
                cq = "#10b981" if sq >= 6.5 else "#f59e0b" if sq >= 5.5 else "#ef4444"
                
                draw_mini_meter(base, sb, cb)
                st.write("")
                draw_mini_meter(quote, sq, cq)
                st.write("")
                
                # Gap avec indicateur directionnel
                gap_color = "#10b981" if gap > 0 else "#ef4444"
                gap_arrow = "⬆️" if gap > 0 else "⬇️"
                st.markdown(f"""
                <div style='text-align:center;margin-top:10px;'>
                    <div style='color:#94a3b8;font-size:0.8em;'>Écart de Force</div>
                    <div style='color:{gap_color};font-size:1.3em;font-weight:bold;'>
                        {gap_arrow} {abs(gap):.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with f2:
                st.markdown("**Market Map - Performance vs Devises**")
                cols = st.columns(6)
                for i, item in enumerate(s['cs']['map'][:6]):
                    with cols[i]:
                        val = item['val']
                        arrow = "▲" if val > 0 else "▼"
                        cl = "#10b981" if val > 0 else "#ef4444"
                        st.markdown(f"""
                        <div style='text-align:center;'>
                            <div style='font-size:0.7em;color:#94a3b8;margin-bottom:4px;'>{item['vs']}</div>
                            <div style='color:{cl};font-size:1.2em;font-weight:bold;'>{arrow}</div>
                            <div style='font-size:0.7em;color:{cl};'>{val:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # --- ANALYSE GPS DÉTAILLÉE ---
        st.markdown("---")
        st.markdown("### 🛡️ Analyse GPS Multi-Timeframe")
        
        mtf_analysis = s['mtf']['analysis']
        timeframes = ['M', 'W', 'D', 'H4', 'H1']
        tf_labels = {'M': 'Mensuel', 'W': 'Hebdo', 'D': 'Daily', 'H4': '4H', 'H1': '1H'}
        
        cols = st.columns(5)
        for i, tf in enumerate(timeframes):
            with cols[i]:
                trend = mtf_analysis.get(tf, 'N/A')
                
                # Icône selon la tendance
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
        
        # --- RISK MANAGER ---
        st.markdown("---")
        st.markdown("### ⚖️ Gestion du Risque")
        
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
        
        # Note informative
        st.info("💡 **Conseil**: Ajustez la taille de position selon votre capital et tolérance au risque (recommandé: 1-2% du capital par trade)")

# ==========================================
# 9. APPLICATION PRINCIPALE
# ==========================================
def main():
    st.title("💎 Bluestar SNP3 GPS")
    st.markdown("**Scanner Institutionnel**: GPS Multi-Timeframe + Force Fondamentale + Entrée Sniper M5")
    
    # Paramètres de scan
    with st.expander("⚙️ Paramètres de Scan", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider(
                "Score Minimum",
                min_value=5.0,
                max_value=10.0,
                value=7.0,
                step=0.5,
                help="Niveau de qualité minimum pour afficher un signal"
            )
        
        with col2:
            strict_mode = st.checkbox(
                "🔥 Mode Sniper (Strict)",
                value=True,
                help="Filtre strict: GPS ≥ B ET Force fondamentale alignée (recommandé)"
            )
        
        # Informations sur le scoring
        with st.expander("ℹ️ Comprendre le Scoring", expanded=False):
            st.markdown("""
            **Composition du Score (0-10)**:
            - 🛡️ **GPS Multi-Timeframe (40%)** : Analyse M/W/D/H4/H1
            - 📊 **Force Fondamentale (35%)** : Analyse relative des devises
            - 📈 **Technique M5 (25%)** : RSI + HMA + FVG
            
            **Qualités GPS**:
            - **A+/A** : Alignement parfait M/W/D (Score GPS: 3/3)
            - **B+/B** : Tendance confirmée W/D (Score GPS: 2/3)
            - **B-** : Retracement possible (Score GPS: 1/3)
            
            **Mode Sniper**: Exige GPS ≥ B ET gap fondamental ≥ 1.0
            """)
    
    # Bouton de scan
    if st.button("🚀 LANCER LE SCAN", type="primary"):
        # Clear cache pour forcer un refresh
        st.session_state.cache_timestamps = {}
        st.session_state.matrix_timestamp = None
        
        try:
            # Initialisation du client API
            api = OandaClient()
            
            # Exécution du scan
            with st.spinner("🔍 Analyse en cours..."):
                results, debug_info = run_scan(api, min_score, strict_mode)
            
            # Affichage des résultats
            if not results:
                st.warning("⚠️ Aucune opportunité trouvée avec les critères actuels.")
                
                # Affichage des statistiques de filtrage
                with st.expander("🔍 Détails du Filtrage", expanded=True):
                    st.markdown(f"**Total d'assets analysés**: {debug_info['total']}")
                    st.markdown("**Raisons de filtrage**:")
                    for reason, count in sorted(debug_info['filtered'].items(), key=lambda x: x[1], reverse=True):
                        reason_labels = {
                            'data_insufficient': 'Données insuffisantes',
                            'volatility': 'Volatilité anormale',
                            'no_signal_m5': 'Pas de signal M5 (RSI/HMA)',
                            'gps_weak': 'GPS trop faible',
                            'fundamental_weak': 'Fondamental trop faible',
                            'score_low': 'Score final < minimum',
                            'error': 'Erreurs techniques'
                        }
                        label = reason_labels.get(reason, reason)
                        st.write(f"- **{label}**: {count} assets")
                
                st.info("💡 Essayez de réduire le score minimum ou désactiver le mode strict.")
            else:
                st.success(f"✅ {len(results)} Signal{'s' if len(results) > 1 else ''} Validé{'s' if len(results) > 1 else ''}")
                
                # Tri par score décroissant
                results.sort(key=lambda x: x['score'], reverse=True)
                
                # Statistiques rapides
                with st.expander("📊 Statistiques du Scan", expanded=False):
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    avg_score = np.mean([s['score'] for s in results])
                    buy_signals = sum(1 for s in results if s['type'] == 'BUY')
                    sell_signals = len(results) - buy_signals
                    avg_gps = np.mean([s['mtf']['score'] for s in results])
                    
                    stats_col1.metric("Score Moyen", f"{avg_score:.1f}/10")
                    stats_col2.metric("Signaux BUY", buy_signals)
                    stats_col3.metric("Signaux SELL", sell_signals)
                    stats_col4.metric("GPS Moyen", f"{avg_gps:.1f}/3")
                    
                    # Afficher les sources de données fondamentales
                    if api and st.session_state.matrix_cache:
                        sources = st.session_state.matrix_cache.get('sources', [])
                        if sources:
                            st.success(f"📡 Sources fondamentales: {', '.join(sources)}")
                        else:
                            st.info("📡 Sources fondamentales: Calcul manuel")
                
                # Affichage de chaque signal
                for sig in results:
                    display_sig(sig)
                    
        except Exception as e:
            logger.error(f"Erreur lors du scan: {e}")
            st.error(f"❌ Erreur lors du scan: {str(e)}")
            st.info("Vérifiez votre connexion API et réessayez.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#64748b;font-size:0.85em;padding:20px 0;'>
        💎 Bluestar SNP3 GPS v2.0 | Scanner Institutionnel Multi-Stratégies<br>
        <span style='font-size:0.75em;'>Entrée M5 • GPS MTF • Force Fondamentale • Smart Money Detection</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
