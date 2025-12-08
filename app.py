"""
BlueStar Cascade - VERSION 3.0 INSTITUTIONAL GRADE ENHANCED - FIXED
🔧 CORRECTIONS MAJEURES :
✅ ATR Percentile calcul corrigé (vrai percentile 0-100)
✅ HMA flip detection fix (gestion NaN)
✅ Paramètres par défaut optimisés (moins restrictifs)
✅ Logging détaillé pour debugging
✅ Validation stricte des types bool
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import hashlib
from functools import wraps
from collections import defaultdict

# OANDA API
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

# PDF Export
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional v3.0", layout="wide", initial_sidebar_state="collapsed")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 1rem !important; max-width: 100% !important;}
    .stMetric {background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.7rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.2rem !important; font-weight: 700;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block;}
    .v30-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block; margin-left: 8px;}
    .fixed-badge {background: linear-gradient(45deg, #ff6b6b, #ffa500); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block; margin-left: 8px;}
    .stDataFrame {font-size: 0.75rem !important;}
    .stDataFrame table {border: none !important;}
    .stDataFrame td, .stDataFrame th {border: none !important;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2)); padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 2px solid rgba(0,255,136,0.3);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    .alert-box {background: rgba(255,200,0,0.1); border-left: 3px solid #ffc800; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .success-box {background: rgba(0,255,136,0.1); border-left: 3px solid #00ff88; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .error-box {background: rgba(255,100,100,0.1); border-left: 3px solid #ff6464; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .session-badge {padding: 2px 6px; border-radius: 10px; font-size: 0.6rem; font-weight: bold;}
    .session-london {background: #ff6b6b; color: white;}
    .session-ny {background: #4ecdc4; color: white;}
    .session-tokyo {background: #ffe66d; color: black;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK","XAU_USD","XPT_USD"
]

MAJOR_PAIRS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD"]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== FOREX SESSIONS ====================
def get_active_session(dt: datetime) -> str:
    """Détecte la session Forex active"""
    hour_utc = dt.astimezone(pytz.UTC).hour
    
    # Tokyo: 00:00-09:00 UTC
    if 0 <= hour_utc < 9:
        return "Tokyo"
    # London: 08:00-17:00 UTC (overlap avec Tokyo 08:00-09:00)
    elif 8 <= hour_utc < 17:
        return "London"
    # NY: 13:00-22:00 UTC (overlap avec London 13:00-17:00)
    elif 13 <= hour_utc < 22:
        return "NY"
    else:
        return "Off-Hours"

def get_session_badge(session: str) -> str:
    """Génère badge HTML pour session"""
    badges = {
        "London": "<span class='session-badge session-london'>LONDON</span>",
        "NY": "<span class='session-badge session-ny'>NY</span>",
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>",
        "Off-Hours": "<span class='session-badge' style='background:#666;color:white;'>OFF</span>"
    }
    return badges.get(session, "")

# ==================== CORRELATION CHECKER ====================
@st.cache_data(ttl=3600)
def calculate_pair_correlation(pair1: str, pair2: str, lookback: int = 100) -> float:
    """Calcule corrélation entre 2 paires"""
    try:
        df1 = get_candles(pair1, "D1", lookback)
        df2 = get_candles(pair2, "D1", lookback)
        
        if len(df1) < 50 or len(df2) < 50:
            return 0.0
        
        returns1 = df1['close'].pct_change().dropna()
        returns2 = df2['close'].pct_change().dropna()
        
        min_len = min(len(returns1), len(returns2))
        if min_len < 20:
            return 0.0
        
        corr = returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:])
        return round(corr, 3)
    except Exception as e:
        logger.warning(f"Correlation error {pair1}-{pair2}: {e}")
        return 0.0

def check_portfolio_correlation(signals: List['Signal'], max_corr: float = 0.7) -> List['Signal']:
    """Filtre signaux trop corrélés"""
    if len(signals) <= 1:
        return signals
    
    filtered = []
    pairs_added = []
    
    # Trier par score décroissant
    sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
    
    for signal in sorted_signals:
        is_correlated = False
        
        for existing_pair in pairs_added:
            corr = calculate_pair_correlation(signal.pair, existing_pair)
            if abs(corr) > max_corr:
                logger.info(f"⚠️ {signal.pair} filtré (corr={corr:.2f} avec {existing_pair})")
                is_correlated = True
                break
        
        if not is_correlated:
            filtered.append(signal)
            pairs_added.append(signal.pair)
    
    return filtered

# ==================== RATE LIMITER ====================
@dataclass
class RateLimiter:
    """Rate limiter avec backoff exponentiel"""
    min_interval: float = 0.12
    max_retries: int = 3
    backoff_factor: float = 2.0
    _last_request: float = field(default=0.0, init=False, repr=False)
    _request_count: int = field(default=0, init=False, repr=False)
    _error_count: int = field(default=0, init=False, repr=False)
    
    def wait(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()
        self._request_count += 1
    
    def backoff(self, attempt: int) -> float:
        return self.min_interval * (self.backoff_factor ** attempt)
    
    def record_error(self) -> None:
        self._error_count += 1
    
    def get_stats(self) -> Dict[str, float]:
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "success_rate": round((1 - self._error_count / max(self._request_count, 1)) * 100, 2)
        }

rate_limiter = RateLimiter()

# ==================== DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "Institutional"
    PREMIUM = "Premium"
    STANDARD = "Standard"

@dataclass
class TradingParams:
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.5
    min_adx_threshold: int = 20
    adx_strong_threshold: int = 25
    min_rr_ratio: float = 1.5
    cascade_required: bool = False
    strict_flip_only: bool = False
    min_score_threshold: int = 55
    min_volatility_percentile: float = 15.0
    max_correlation: float = 0.75
    session_filter: bool = False

@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05
    kelly_fraction: float = 0.25
    min_win_rate: float = 0.50
    max_win_rate: float = 0.75
    max_drawdown_threshold: float = 0.15

@dataclass
class Signal:
    timestamp: datetime
    pair: str
    timeframe: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    score: int
    quality: SignalQuality
    position_size: float
    risk_amount: float
    risk_reward: float
    adx: float
    rsi: float
    atr: float
    atr_percentile: float
    higher_tf_trend: str
    is_live: bool
    is_fresh_flip: bool
    candle_index: int
    is_strict_flip: bool
    session: str
    correlation_risk: float = 0.0

@dataclass
class ScanStats:
    total_pairs: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    signals_found: int = 0
    signals_filtered: int = 0
    scan_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        return API(access_token=token)
    except Exception as e:
        logger.error(f"❌ OANDA Token Error: {e}")
        st.error("⚠️ OANDA Token manquant dans secrets Streamlit")
        st.stop()

client = get_oanda_client()

def retry_on_failure(max_attempts: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except V20Error as e:
                    logger.warning(f"⚠️ API Error (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(rate_limiter.backoff(attempt))
                    else:
                        rate_limiter.record_error()
                        raise
                except Exception as e:
                    logger.error(f"❌ Error in {func.__name__}: {e}")
                    rate_limiter.record_error()
                    raise
            return None
        return wrapper
    return decorator

def get_cache_key(pair: str, tf: str, count: int) -> str:
    return hashlib.md5(f"{pair}_{tf}_{count}".encode()).hexdigest()

@st.cache_data(ttl=30, show_spinner=False)
def _cache_wrapper(cache_key: str, pair: str, tf: str, count: int):
    return fetch_candles_raw(pair, tf, count)

@retry_on_failure(max_attempts=3)
def fetch_candles_raw(pair: str, tf: str, count: int) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran:
        logger.warning(f"⚠️ Timeframe invalide: {tf}")
        return pd.DataFrame()
    
    rate_limiter.wait()
    
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        
        data = []
        for c in req.response.get("candles", []):
            data.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "complete": c.get("complete", False)
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df["time"] = df["time"].dt.tz_localize(None)
        
        logger.debug(f"✅ {pair} {tf}: {len(df)} candles")
        return df
        
    except Exception as e:
        logger.error(f"❌ API Error {pair} {tf}: {e}")
        raise

def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    cache_key = get_cache_key(pair, tf, count)
    return _cache_wrapper(cache_key, pair, tf, count)

# ==================== INDICATEURS OPTIMISÉS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        logger.warning("⚠️ Pas assez de données")
        return df
    
    close = df['close']
    high = df['high']
    low = df['low']

    def wma(series, length):
        if len(series) < length:
            return pd.Series([np.nan] * len(series), index=series.index)
        weights = np.arange(1, length + 1)
        return series.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan,
            raw=True
        )

    # HMA optimisé avec FIX
    try:
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        hma_length = int(np.sqrt(20))
        
        if wma_half.isna().all() or wma_full.isna().all():
            df['hma'] = np.nan
            df['hma_up'] = False
        else:
            df['hma'] = wma(2 * wma_half - wma_full, hma_length)
            
            # 🔧 FIX: Gestion correcte des NaN
            hma_condition = df['hma'] > df['hma'].shift(1)
            df['hma_up'] = hma_condition.where(hma_condition.notna(), False)
            df['hma_up'] = df['hma_up'].astype(bool)
            
    except Exception as e:
        logger.error(f"❌ HMA Error: {e}")
        df['hma'] = np.nan
        df['hma_up'] = False

    # RSI
    try:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.ewm(alpha=1/14, min_periods=14).mean() / down.ewm(alpha=1/14, min_periods=14).mean()
        df['rsi'] = 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"❌ RSI Error: {e}")
        df['rsi'] = np.nan

    # ATR + UT + ADX
    try:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # UT Bot
        xATR = tr.rolling(1).mean()
        nLoss = 2.0 * xATR
        xATRTrailingStop = [0.0] * len(df)
        
        for i in range(1, len(df)):
            prev_stop = xATRTrailingStop[i-1]
            curr_src = close.iloc[i]
            loss = nLoss.iloc[i]
            
            if pd.isna(loss):
                xATRTrailingStop[i] = prev_stop
                continue
            
            if (curr_src > prev_stop) and (close.iloc[i-1] > prev_stop):
                xATRTrailingStop[i] = max(prev_stop, curr_src - loss)
            elif (curr_src < prev_stop) and (close.iloc[i-1] < prev_stop):
                xATRTrailingStop[i] = min(prev_stop, curr_src + loss)
            elif curr_src > prev_stop:
                xATRTrailingStop[i] = curr_src - loss
            else:
                xATRTrailingStop[i] = curr_src + loss
        
        df['ut_state'] = np.where(close > xATRTrailingStop, 1, -1)
        
        # ADX
        atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
        df['atr_val'] = atr14
        
        # 🔧 FIX: ATR Percentile CORRECT (vrai percentile 0-100)
        def calc_percentile(x):
            if len(x) < 50:
                return np.nan
            return pd.Series(x).rank(pct=True).iloc[-1] * 100
        
        df['atr_percentile'] = df['atr_val'].rolling(100, min_periods=50).apply(calc_percentile)
        
    except Exception as e:
        logger.error(f"❌ ADX/UT Error: {e}")
        df['ut_state'] = 0
        df['adx'] = np.nan
        df['atr_val'] = np.nan
        df['atr_percentile'] = np.nan
    
    return df

# ==================== CASCADE MULTI-TF ====================
@st.cache_data(ttl=180, show_spinner=False)
def get_trend_alignment(pair: str, signal_tf: str) -> Tuple[str, float]:
    """Retourne (trend, strength_score)"""
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf:
        return "Neutral", 0.0
    
    try:
        df = get_candles(pair, higher_tf, 100)
        if len(df) < 50:
            return "Neutral", 0.0
        
        df = calculate_indicators(df)
        
        if pd.isna(df['hma'].iloc[-1]) or pd.isna(df['hma'].iloc[-2]):
            return "Neutral", 0.0
        
        close = df['close']
        ema50 = close.ewm(span=50, min_periods=50).mean().iloc[-1]
        ema200 = close.ewm(span=200, min_periods=100).mean().iloc[-1] if len(df) >= 100 else ema50
        hma_current = df['hma'].iloc[-1]
        hma_prev = df['hma'].iloc[-2]
        adx = df['adx'].iloc[-1]
        
        strength = 0.0
        
        # Scoring trend strength
        if close.iloc[-1] > ema50 and hma_current > hma_prev:
            trend = "Bullish"
            strength += 30
            if close.iloc[-1] > ema200:
                strength += 20
            if adx > 25:
                strength += 30
            if df['ut_state'].iloc[-1] == 1:
                strength += 20
        elif close.iloc[-1] < ema50 and hma_current < hma_prev:
            trend = "Bearish"
            strength += 30
            if close.iloc[-1] < ema200:
                strength += 20
            if adx > 25:
                strength += 30
            if df['ut_state'].iloc[-1] == -1:
                strength += 20
        else:
            trend = "Neutral"
            strength = 0
        
        return trend, min(100, strength)
        
    except Exception as e:
        logger.error(f"❌ Cascade Error {pair}: {e}")
        return "Neutral", 0.0

# ==================== RISK MANAGER AMÉLIORÉ ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
        self.current_exposure = 0.0
        self.active_trades = []
    
    def calculate_position_size(self, signal: Signal, win_rate: float = 0.58, 
                               historical_dd: float = 0.0) -> float:
        """Kelly Criterion optimisé avec drawdown protection"""
        win_rate = max(self.config.min_win_rate, min(win_rate, self.config.max_win_rate))
        
        # Kelly standard
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        # Drawdown adjustment
        if historical_dd > self.config.max_drawdown_threshold:
            dd_factor = 1 - (historical_dd - self.config.max_drawdown_threshold) * 2
            kelly *= max(0.3, dd_factor)
        
        # Score adjustment
        score_factor = signal.score / 100
        kelly *= score_factor
        
        if kelly <= 0:
            return 0.0
        
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0:
            return 0.0
        
        size = (self.balance * kelly) / pip_risk
        
        # Portfolio risk limit
        max_position_risk = self.balance * self.config.max_risk_per_trade
        if size * pip_risk > max_position_risk:
            size = max_position_risk / pip_risk
        
        return round(size, 2)

# ==================== ANALYSE OPTIMISÉE ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, 
                params: TradingParams) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100:
            return None
        
        df = calculate_indicators(df)
        
        if mode_live:
            idx = -1
        else:
            idx = -2 if not df.iloc[-1]['complete'] else -1
        
        if abs(idx) > len(df):
            return None
        
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2] if abs(idx-2) <= len(df) else None
        
        # Validation stricte des données
        required_fields = ['hma', 'rsi', 'adx', 'atr_val', 'ut_state', 'atr_percentile']
        for field in required_fields:
            if pd.isna(last[field]):
                logger.debug(f"⚠️ {pair} {tf}: {field} est NaN")
                return None
        
        # Vérification type bool pour hma_up
        if not isinstance(last['hma_up'], (bool, np.bool_)):
            logger.debug(f"⚠️ {pair} {tf}: hma_up invalide (type={type(last['hma_up'])})")
            return None
        
        if not isinstance(prev['hma_up'], (bool, np.bool_)):
            return None
        
        # Filtrage volatilité avec DEBUG
        if last['atr_percentile'] < params.min_volatility_percentile:
            logger.info(f"🚫 {pair} {tf} VOLATILITÉ: {last['atr_percentile']:.1f}% < {params.min_volatility_percentile}%")
            return None
        
        # Détection flip (maintenant safe avec bool strict)
        hma_flip_green = (last['hma_up'] == True) and (prev['hma_up'] == False)
        hma_flip_red = (last['hma_up'] == False) and (prev['hma_up'] == True)
        
        hma_extended_green = False
        hma_extended_red = False
        if prev2 is not None and isinstance(prev2['hma_up'], (bool, np.bool_)):
            hma_extended_green = (last['hma_up'] == True) and (prev['hma_up'] == True) and (prev2['hma_up'] == False)
            hma_extended_red = (last['hma_up'] == False) and (prev['hma_up'] == False) and (prev2['hma_up'] == True)
        
        # Logique signal
        if params.strict_flip_only:
            raw_buy = hma_flip_green and last['rsi'] > 50 and last['ut_state'] == 1
            raw_sell = hma_flip_red and last['rsi'] < 50 and last['ut_state'] == -1
            is_strict = True
        else:
            raw_buy = (hma_flip_green or hma_extended_green) and last['rsi'] > 50 and last['ut_state'] == 1
            raw_sell = (hma_flip_red or hma_extended_red) and last['rsi'] < 50 and last['ut_state'] == -1
            is_strict = hma_flip_green or hma_flip_red
        
        if not (raw_buy or raw_sell):
            logger.debug(f"🚫 {pair} {tf} AUCUN
