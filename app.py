import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import requests  # <--- AJOUTÉ POUR LE CALCUL DE FORCE
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, List

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
    
    h2, h3 { color: #e2e8f0; font-weight: 700; }

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
    
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }

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
    
    .badge-fvg, .badge-gps, .badge-sr {
        color: white; padding: 4px 10px; border-radius: 6px;
        font-size: 0.75em; font-weight: 700; display: inline-block; margin: 2px;
    }
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); box-shadow: 0 2px 8px rgba(124, 58, 237, 0.3); }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3); }
    .badge-sr { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3); }

    .stAlert { background-color: #1e293b; color: #e2e8f0; border: 1px solid #334155; }
    hr { margin: 1.5em 0; border-color: #334155; }
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
CURRENCY_STRENGTH_CACHE_DURATION = 300  # 5 minutes

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
            self.last_request_time = time.time()
        except KeyError as e:
            st.error(f"⚠️ Clé manquante dans secrets.toml: {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Erreur d'initialisation API: {str(e)}")
            st.stop()

    def _rate_limit(self):
        """Gestion du rate limiting"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < 0.1: # Légère augmentation du délai de sécurité
            time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()

    def get_candles(self, instrument: str, granularity: str, count: int = 150) -> pd.DataFrame:
        """Récupération des données avec cache"""
        cache_key = f"{instrument}_{granularity}"
        if cache_key in st.session_state.cache:
            cache_age = time.time() - st.session_state.cache_time.get(cache_key, 0)
            if cache_age < CACHE_DURATION:
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
            except Exception:
                if attempt < 1: time.sleep(0.5)
                else: return pd.DataFrame()
        return pd.DataFrame()

# ==========================================
# INDICATEURS TECHNIQUES
# ==========================================
def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def calculate_zlema(series: pd.Series, length: int) -> pd.Series:
    lag = int((length - 1) / 2)
    src_adj = series + (series - series.shift(lag))
    return src_adj.ewm(span=length, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    high, low, close = df['high'], df['low'], df['close']
    tr = calculate_atr(df, 1).ewm(span=period, adjust=False).mean() # Approximate TR smooth
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di

def get_rsi_ohlc4(df: pd.DataFrame, length: int = 7) -> pd.Series:
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def get_colored_hma(df: pd.DataFrame, length: int = 20) -> tuple:
    src = df['close']
    wma1 = calculate_wma(src, int(length / 2))
    wma2 = calculate_wma(src, length)
    hma = calculate_wma(2 * wma1 - wma2, int(np.sqrt(length)))
    trend = pd.Series(np.where(hma > hma.shift(1), 1, -1), index=df.index)
    return hma, trend

# ==========================================
# SUPPORT & RESISTANCE / FVG
# ==========================================
def get_nearest_sr(df: pd.DataFrame, current_price: float, timeframe: str = 'D') -> Dict:
    if len(df) < 20: return {'sup': None, 'res': None, 'dist_sup': 999, 'dist_res': 999}
    
    # Fractales Bill Williams (Optimisé)
    highs = df['high']
    lows = df['low']
    is_res = (highs > highs.shift(1)) & (highs > highs.shift(2)) & (highs > highs.shift(-1)) & (highs > highs.shift(-2))
    is_sup = (lows < lows.shift(1)) & (lows < lows.shift(2)) & (lows < lows.shift(-1)) & (lows < lows.shift(-2))
    
    res_levels = df[is_res]['high'].values
    sup_levels = df[is_sup]['low'].values
    
    relevant_res = res_levels[res_levels > current_price]
    relevant_sup = sup_levels[sup_levels < current_price]
    
    nearest_res = relevant_res.min() if len(relevant_res) > 0 else None
    nearest_sup = relevant_sup.max() if len(relevant_sup) > 0 else None
    
    dist_res = ((nearest_res - current_price) / current_price * 100) if nearest_res else 999
    dist_sup = ((current_price - nearest_sup) / current_price * 100) if nearest_sup else 999
    
    return {'sup': nearest_sup, 'res': nearest_res, 'dist_sup': dist_sup, 'dist_res': dist_res}

def detect_fvg(df: pd.DataFrame) -> tuple:
    if len(df) < 5: return False, False
    fvg_bull = (df['low'] > df['high'].shift(2))
    fvg_bear = (df['high'] < df['low'].shift(2))
    return fvg_bull.iloc[-5:].any(), fvg_bear.iloc[-5:].any()

def get_pips(pair: str, price_diff: float) -> float:
    if any(x in pair for x in ["XAU", "US30", "NAS100", "SPX500", "XPT"]): return abs(price_diff)
    return abs(price_diff * (100 if "JPY" in pair else 10000))

# ==========================================
# CURRENCY STRENGTH ENGINE (MOTEUR MARKET MAP PRO DIRECT)
# ==========================================
def calculate_currency_strength(api: OandaClient, lookback_days: int = 1) -> Dict[str, float]:
    """
    MOTEUR ORIGINAL DE MARKET MAP PRO (Portage direct via Requests)
    Remplace la version buggée oandapyV20 pour garantir des données.
    """
    # 1. Gestion du Cache pour ne pas ralentir l'app
    cache_age = time.time() - st.session_state.currency_strength_time
    if st.session_state.currency_strength_cache and cache_age < CURRENCY_STRENGTH_CACHE_DURATION:
        # Check intégrité
        total_strength = sum(abs(x) for x in st.session_state.currency_strength_cache.values())
        if total_strength > 0.001:
            return st.session_state.currency_strength_cache

    # 2. Récupération des secrets (Méthode Market Map Pro)
    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        # On détecte si c'est practice ou live selon l'URL
        env = st.secrets.get("OANDA_ENVIRONMENT", "practice")
        base_url = "https://api-fxtrade.oanda.com" if env == "live" else "https://api-fxpractice.oanda.com"
    except:
        return {} # Pas de clé, pas de calcul

    # 3. Liste des paires EXACTE de Market Map Pro
    pairs_to_scan = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
        'AUD_JPY', 'AUD_CAD', 'AUD_NZD', 'AUD_CHF',
        'CAD_JPY', 'CAD_CHF', 'NZD_JPY', 'NZD_CHF', 'CHF_JPY'
    ]

    forex_data = {}
    headers = {"Authorization": f"Bearer {token}"}
    
    # Scan via requests (plus robuste)
    for instrument in pairs_to_scan:
        try:
            url = f"{base_url}/v3/instruments/{instrument}/candles?count={lookback_days+5}&granularity=D"
            # Timeout court pour enchainer vite
            resp = requests.get(url, headers=headers, timeout=2)
            
            if resp.status_code == 200:
                candles = resp.json().get('candles', [])
                if candles and len(candles) > lookback_days:
                    # Logique de calcul Market Map Pro
                    c_now = float(candles[-1]['mid']['c'])
                    # On prend la bougie d'avant (D-1) pour comparer
                    c_past = float(candles[-(lookback_days+1)]['mid']['c']) 
                    
                    pct = (c_now - c_past) / c_past * 100
                    forex_data[instrument] = pct
        except Exception:
            continue

    # 4. ALGORITHME "SMART WEIGHTED SCORE" (Identique à Market Map Pro)
    data_struct = {}
    for symbol, pct in forex_data.items():
        parts = symbol.split('_')
        if len(parts) != 2: continue
        base, quote = parts[0], parts[1]
        
        if base not in data_struct: data_struct[base] = []
        if quote not in data_struct: data_struct[quote] = []
        
        data_struct[base].append({'pct': pct, 'other': quote})
        data_struct[quote].append({'pct': -pct, 'other': base})
    
    final_scores = {}
    for curr, items in data_struct.items():
        score = 0
        valid_items = 0
        for item in items:
            opponent = item['other']
            val = item['pct']
            
            # PONDÉRATION MAJEURE
            weight = 2.0 if opponent in ['USD', 'EUR', 'JPY'] else 1.0
            score += (val * weight)
            valid_items += weight
            
        final_scores[curr] = score / valid_items if valid_items > 0 else 0
    
    # 5. Sauvegarde Cache seulement si données non nulles
    total_check = sum(abs(v) for v in final_scores.values())
    if total_check > 0.001:
        st.session_state.currency_strength_cache = final_scores
        st.session_state.currency_strength_time = time.time()
        
    return final_scores

def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    # CAS 1: FOREX
    if symbol in FOREX_PAIRS:
        parts = symbol.split('_')
        base, quote = parts[0], parts[1]
        strength_scores = calculate_currency_strength(api)
        
        if base not in strength_scores: 
            return {'score': 0, 'details': 'Data manquante', 'base_score': 0, 'rank_info': 'N/A'}
        
        base_score = strength_scores[base]
        quote_score = strength_scores[quote]
        
        # Ranking
        sorted_curr = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {k: i+1 for i, (k, v) in enumerate(sorted_curr)}
        base_rank, quote_rank = ranks.get(base, 8), ranks.get(quote, 8)
        
        score = 0
        details = []
        
        if direction == 'BUY':
            if base_rank <= 3 and quote_rank >= 6:
                score = 2
                details.append(f"✅ {base}(#{base_rank}) vs {quote}(#{quote_rank})")
            elif base_score > quote_score:
                score = 1
                details.append(f"📊 {base} > {quote}")
            else:
                details.append(f"⚠️ Divergence: {quote} > {base}")
        else: # SELL
            if quote_rank <= 3 and base_rank >= 6:
                score = 2
                details.append(f"✅ {quote}(#{quote_rank}) vs {base}(#{base_rank})")
            elif quote_score > base_score:
                score = 1
                details.append(f"📊 {quote} > {base}")
            else:
                details.append(f"⚠️ Divergence: {base} > {quote}")
                
        return {'score': score, 'details': ' | '.join(details), 'base_score': base_score, 'quote_score': quote_score, 'rank_info': f"{base}:#{base_rank} / {quote}:#{quote_rank}"}

    # CAS 2: INDICES/OR
    else:
        try:
            df = api.get_candles(symbol, "D", count=2)
            if df.empty: return {'score': 0, 'details': 'No Data', 'base_score': 0, 'rank_info': 'N/A'}
            
            change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100
            score = 0
            details = []
            
            if direction == 'BUY':
                if change > 0.3: score = 2; details.append("🚀 Impulsion Haussière")
                elif change > 0: score = 1; details.append("📈 Journée Verte")
                else: details.append("⚠️ Contre-tendance")
            else:
                if change < -0.3: score = 2; details.append("☄️ Chute Baissière")
                elif change < 0: score = 1; details.append("📉 Journée Rouge")
                else: details.append("⚠️ Contre-tendance")
                
            return {'score': score, 'details': ' | '.join(details), 'base_score': change, 'rank_info': f"Daily: {change:+.2f}%"}
        except:
            return {'score': 0, 'details': 'Error', 'base_score': 0, 'rank_info': 'N/A'}

# ==========================================
# MTF & SCORING
# ==========================================
def analyze_timeframe_gps(df: pd.DataFrame, timeframe: str) -> Dict:
    close = df['close']
    price = close.iloc[-1]
    
    if timeframe in ['H4', 'D1']:
        sma50 = calculate_sma(close, 50).iloc[-1]
        sma200 = calculate_sma(close, 200).iloc[-1] if len(df) >= 200 else sma50
        
        trend = "Bullish" if price > sma200 else "Bearish"
        details = f"> SMA200" if trend == "Bullish" else "< SMA200"
        return {'trend': trend, 'score': 100, 'details': details}
    else:
        # Logic H1/M15 plus fine
        zlema = calculate_zlema(close, 50).iloc[-1]
        adx, _, _ = calculate_adx(df, 14)
        curr_adx = adx.iloc[-1]
        
        if price > zlema: trend = "Bullish"
        elif price < zlema: trend = "Bearish"
        else: trend = "Range"
        
        return {'trend': trend, 'score': curr_adx, 'details': f"ADX: {curr_adx:.1f}"}

def calculate_mtf_score_gps(api: OandaClient, symbol: str, direction: str) -> Dict:
    tf_map = {'D1': 'D', 'H4': 'H4', 'H1': 'H1'}
    analysis = {}
    
    expected = 'Bullish' if direction == 'BUY' else 'Bearish'
    matches = 0
    weights = {'D1': 2, 'H4': 1, 'H1': 0.5}
    total_w = 3.5
    curr_w = 0
    
    for name, code in tf_map.items():
        df = api.get_candles(symbol, code, count=250)
        if not df.empty:
            res = analyze_timeframe_gps(df, name)
            analysis[name] = res
            if res['trend'] == expected:
                curr_w += weights[name]
                matches += 1
        else:
            analysis[name] = {'trend': 'Neutral', 'details': 'N/A'}

    pct = (curr_w / total_w) * 100
    score = 3 if pct >= 85 else 2 if pct >= 50 else 1 if pct >= 25 else 0
    
    quality = 'C'
    if analysis['D1']['trend'] == expected and analysis['H4']['trend'] == expected: quality = 'A'
    elif analysis['H4']['trend'] == expected and analysis['H1']['trend'] == expected: quality = 'B'
    
    return {'score': score, 'quality': quality, 'analysis': analysis, 'alignment': f"{pct:.0f}%", 'details': f"Match score: {pct:.0f}%"}

def calculate_risk_management(price: float, atr: float, direction: str, pair: str, sl_mult: float, tp_mult: float) -> Dict:
    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult
    
    if direction == 'BUY':
        sl, tp = price - sl_dist, price + tp_dist
    else:
        sl, tp = price + sl_dist, price - tp_dist
        
    return {
        'sl': sl, 'tp': tp,
        'sl_pips': get_pips(pair, sl_dist),
        'tp_pips': get_pips(pair, tp_dist),
        'rr_ratio': tp_mult / sl_mult
    }

# ==========================================
# SCANNER MAIN
# ==========================================
def run_hybrid_scan(api: OandaClient, min_score: int, enable_risk: bool, sl_m: float, tp_m: float) -> List[Dict]:
    signals = []
    
    # Init Currency Strength
    status = st.empty()
    status.text("🔄 Initialisation Force Devises (Mode Direct HTTP)...")
    
    # CALCUL DE LA FORCE AU DÉMARRAGE
    cs_data = calculate_currency_strength(api)
    
    # --- DIAGNOSTIC VISUEL POUR L'UTILISATEUR ---
    if cs_data:
        with st.expander("✅ Données Market Map chargées (Cliquez pour voir les %)", expanded=False):
            cols = st.columns(7)
            sorted_cs = sorted(cs_data.items(), key=lambda x: x[1], reverse=True)
            for i, (curr, val) in enumerate(sorted_cs[:7]): # Top 7 affichés
                cols[i].markdown(f"**{curr}**: `{val:+.2f}%`")
    else:
        st.error("⚠️ Erreur Market Map: 0 données récupérées via Requests.")
    # --------------------------------------------
    
    bar = st.progress(0)
    for i, symbol in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        status.text(f"Scanning {symbol}...")
        
        try:
            # M15 Data
            df = api.get_candles(symbol, "M15", count=150)
            if df.empty or len(df) < 50: continue
            
            # Indicators
            curr_price = df['close'].iloc[-1]
            atr = calculate_atr(df, 14).iloc[-1]
            rsi = get_rsi_ohlc4(df)
            hma, hma_trend = get_colored_hma(df)
            fvg_bull, fvg_bear = detect_fvg(df)
            
            # Scores Logic
            # BUY
            rsi_val = rsi.iloc[-1]
            rsi_prev = rsi.iloc[-2]
            hma_col = hma_trend.iloc[-1]
            
            # Check BUY
            if rsi_val > 50 and hma_col == 1:
                # Basic Score
                score = 0
                details_rsi = ""
                
                # RSI Logic
                if rsi_prev < 50: score += 3; details_rsi = "Cross UP"
                elif rsi_val > rsi_prev: score += 2; details_rsi = "Trend UP"
                else: score += 1
                
                # HMA Logic
                score += 2 if hma_trend.iloc[-2] == -1 else 1
                
                # MTF & Force
                mtf = calculate_mtf_score_gps(api, symbol, 'BUY')
                cs = calculate_currency_strength_score(api, symbol, 'BUY')
                
                score += mtf['score'] + cs['score']
                if fvg_bull: score += 1
                
                # SR Check if score promising
                sr_badge = ""
                warning = ""
                if score >= min_score - 2:
                    df_d = api.get_candles(symbol, "D", 200)
                    sr = get_nearest_sr(df_d, curr_price)
                    if sr['dist_res'] < 0.25: 
                        score -= 2
                        warning = f"Résistance D1 proche ({sr['dist_res']:.2f}%)"
                    elif sr['dist_sup'] < 0.4:
                        sr_badge = "REBOND SUP"

                if score >= min_score:
                    rm = calculate_risk_management(curr_price, atr, 'BUY', symbol, sl_m, tp_m) if enable_risk else {}
                    signals.append({
                        'symbol': symbol, 'type': 'BUY', 'price': curr_price, 'total_score': score,
                        'quality': mtf['quality'], 'atr': atr, 'warning': warning, 'sr_badge': sr_badge,
                        'rsi': {'value': rsi_val, 'details': details_rsi},
                        'hma': {'color': 'VERT', 'details': 'Bullish'},
                        'mtf': mtf, 'cs': cs, 'fvg': fvg_bull, 'rm': rm,
                        'time': df['time'].iloc[-1]
                    })

            # Check SELL
            elif rsi_val < 50 and hma_col == -1:
                score = 0
                details_rsi = ""
                
                if rsi_prev > 50: score += 3; details_rsi = "Cross DOWN"
                elif rsi_val < rsi_prev: score += 2; details_rsi = "Trend DOWN"
                else: score += 1
                
                score += 2 if hma_trend.iloc[-2] == 1 else 1
                
                mtf = calculate_mtf_score_gps(api, symbol, 'SELL')
                cs = calculate_currency_strength_score(api, symbol, 'SELL')
                
                score += mtf['score'] + cs['score']
                if fvg_bear: score += 1
                
                sr_badge = ""
                warning = ""
                if score >= min_score - 2:
                    df_d = api.get_candles(symbol, "D", 200)
                    sr = get_nearest_sr(df_d, curr_price)
                    if sr['dist_sup'] < 0.25: 
                        score -= 2
                        warning = f"Support D1 proche ({sr['dist_sup']:.2f}%)"
                    elif sr['dist_res'] < 0.4:
                        sr_badge = "REJET RES"

                if score >= min_score:
                    rm = calculate_risk_management(curr_price, atr, 'SELL', symbol, sl_m, tp_m) if enable_risk else {}
                    signals.append({
                        'symbol': symbol, 'type': 'SELL', 'price': curr_price, 'total_score': score,
                        'quality': mtf['quality'], 'atr': atr, 'warning': warning, 'sr_badge': sr_badge,
                        'rsi': {'value': rsi_val, 'details': details_rsi},
                        'hma': {'color': 'ROUGE', 'details': 'Bearish'},
                        'mtf': mtf, 'cs': cs, 'fvg': fvg_bear, 'rm': rm,
                        'time': df['time'].iloc[-1]
                    })
                    
        except Exception: continue
        
    status.empty()
    bar.empty()
    return signals

# ==========================================
# UI & DISPLAY
# ==========================================
def display_signal(sig: Dict):
    is_buy = sig['type'] == 'BUY'
    color = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    ago = int((datetime.now(timezone.utc) - sig['time'].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() / 60)
    
    with st.expander(f"{sig['symbol']} | {sig['type']} | Score {sig['total_score']}/12 [{sig['quality']}]", expanded=True):
        st.markdown(f"""
        <div style="background: {bg}; padding: 15px; border-radius: 8px; border: 2px solid {color}; display: flex; justify-content: space-between; align-items: center;">
            <div><span style="font-size: 1.8em; font-weight: 900; color: white;">{sig['symbol']}</span>
            <span style="background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px; color: white; margin-left: 10px;">{sig['type']}</span></div>
            <div style="text-align: right;"><div style="color: #cbd5e1;">{ago} min ago</div><div style="font-size: 1.4em; font-weight: bold; color: white;">{sig['price']:.5f}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Badges
        badges = []
        if sig['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        if sig['quality'] == 'A': badges.append("<span class='badge-gps'>🛡️ GPS A+</span>")
        if sig['sr_badge']: badges.append(f"<span class='badge-sr'>{sig['sr_badge']}</span>")
        
        if badges: st.markdown(f"<div style='margin-top:10px; text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        if sig['warning']: st.warning(sig['warning'])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", sig['total_score'])
        c2.metric("Qualité", sig['quality'])
        # Affichage Force corrigé (évite le -0.0%)
        force_val = sig['cs']['base_score']
        force_str = f"{force_val:.2f}%" if abs(force_val) > 0.001 else "0.00%"
        c3.metric("Force", force_str)
        c4.metric("ATR", f"{sig['atr']:.4f}")
        
        if sig['rm']:
            st.divider()
            r1, r2, r3 = st.columns(3)
            sl_pips = int(sig['rm']['sl_pips']) if "XAU" not in sig['symbol'] else f"{sig['rm']['sl_pips']:.1f}"
            tp_pips = int(sig['rm']['tp_pips']) if "XAU" not in sig['symbol'] else f"{sig['rm']['tp_pips']:.1f}"
            
            r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>STOP LOSS</div><div style='color:#ef4444;font-weight:bold'>{sig['rm']['sl']:.5f}</div><small>-{sl_pips} pips</small></div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>TAKE PROFIT</div><div style='color:#10b981;font-weight:bold'>{sig['rm']['tp']:.5f}</div><small>+{tp_pips} pips</small></div>", unsafe_allow_html=True)
            r3.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>RATIO</div><div style='color:white;font-weight:bold'>1:{sig['rm']['rr_ratio']:.2f}</div></div>", unsafe_allow_html=True)

        st.divider()
        k1, k2 = st.columns(2)
        k1.info(f"**Technique**: RSI {sig['rsi']['value']:.1f} ({sig['rsi']['details']}) | HMA {sig['hma']['color']}")
        k2.info(f"**Macro**: Alignement {sig['mtf']['alignment']} | {sig['cs']['details']}")

# ==========================================
# APP MAIN
# ==========================================
st.title("💎 Bluestar SNP3 Hybrid Pro")
st.markdown("Scanner MTF GPS + Currency Strength Fix + Smart Money")

with st.expander("⚙️ Paramètres", expanded=False):
    c_risk, c_sets = st.columns(2)
    enable_risk = c_risk.checkbox("Risk Manager Auto", value=True)
    sl_m = c_sets.slider("SL Multiplier", 1.0, 3.0, 1.5)
    tp_m = c_sets.slider("TP Multiplier", 1.5, 5.0, 2.0)

col1, col2 = st.columns([3, 1])
min_score = col1.slider("Score Min", 4, 12, 6)

if col2.button("🧹 Reset Cache"):
    st.session_state.cache = {}
    st.session_state.currency_strength_cache = None
    st.toast("Cache vidé !", icon="🧹")

if st.button("🚀 LANCER LE SCANNER", type="primary"):
    api = OandaClient()
    results = run_hybrid_scan(api, min_score, enable_risk, sl_m, tp_m)
    
    st.success(f"Scan terminé : {len(results)} opportunités trouvées")
    
    # Tri par Score puis Qualité
    results = sorted(results, key=lambda x: (x['total_score'], x['quality']), reverse=True)
    
    for sig in results:
        display_signal(sig)
    
