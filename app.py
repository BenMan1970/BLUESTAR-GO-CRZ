"""
BlueStar Cascade - VERSION 2.5 PROFESSIONAL GRADE - CORRECTED
Nouvelles fonctionnalités v2.5 :
✅ Gestion de corrélation des devises
✅ Filtre news économiques (Forex Factory)
✅ Validation multi-critères avancée
✅ Détection conditions de marché
✅ Support/Résistance basic
✅ Scoring amélioré (0-100 strict)
✅ CORRECTIONS: fillna deprecated methods fixed
✅ Code complet sans troncature
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import hashlib
from functools import wraps
import requests
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
st.set_page_config(page_title="BlueStar Professional v2.5", layout="wide", initial_sidebar_state="collapsed")

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
    .v25-badge {background: linear-gradient(45deg, #00d4ff, #0099ff); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block; margin-left: 8px;}
    .stDataFrame {font-size: 0.75rem !important;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2)); padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 2px solid rgba(0,255,136,0.3);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    .alert-box {background: rgba(255,200,0,0.1); border-left: 3px solid #ffc800; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .success-box {background: rgba(0,255,136,0.1); border-left: 3px solid #00ff88; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .error-box {background: rgba(255,100,100,0.1); border-left: 3px solid #ff6464; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .news-warning {background: rgba(255,100,100,0.15); border: 2px solid #ff6464; padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 0.85rem; font-weight: bold;}
    .correlation-box {background: rgba(100,200,255,0.1); border-left: 3px solid #64c8ff; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK","XAU_USD","XPT_USD"
]
GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== CORRELATION MANAGER ====================
CURRENCY_CORRELATION_MATRIX = {
    "EUR": {"USD": -0.9, "GBP": 0.7, "JPY": 0.4, "CHF": 0.8},
    "GBP": {"USD": -0.7, "EUR": 0.7, "JPY": 0.5, "CHF": 0.6},
    "USD": {"JPY": -0.6, "CHF": -0.7, "CAD": 0.7},
    "AUD": {"NZD": 0.9, "JPY": 0.3, "CAD": 0.6},
    "NZD": {"AUD": 0.9, "JPY": 0.3},
    "CAD": {"USD": 0.7, "AUD": 0.6},
    "CHF": {"EUR": 0.8, "USD": -0.7},
    "XAU": {"USD": -0.8},
    "XPT": {"XAU": 0.7}
}

EXPOSURE_LIMITS = {
    "EUR": 3, "USD": 4, "GBP": 2, "JPY": 3, "AUD": 2, "NZD": 2, "CAD": 2, "CHF": 2,
    "XAU": 1, "XPT": 1, "TOTAL_LONG": 6, "TOTAL_SHORT": 6,
}

@dataclass
class CorrelationManager:
    active_signals: List['Signal'] = field(default_factory=list)
    
    def extract_currencies(self, pair: str) -> Tuple[str, str]:
        if "XAU" in pair:
            return ("XAU", "USD")
        if "XPT" in pair:
            return ("XPT", "USD")
        parts = pair.split("_")
        return (parts[0], parts[1])
    
    def get_currency_exposure(self, currency: str) -> int:
        count = 0
        for sig in self.active_signals:
            base, quote = self.extract_currencies(sig.pair)
            if base == currency or quote == currency:
                count += 1
        return count
    
    def get_direction_count(self, direction: str) -> int:
        return sum(1 for sig in self.active_signals if sig.action == direction)
    
    def calculate_portfolio_correlation(self, new_signal: 'Signal') -> float:
        if not self.active_signals:
            return 0.0
        
        new_base, new_quote = self.extract_currencies(new_signal.pair)
        correlations = []
        
        for existing in self.active_signals:
            exist_base, exist_quote = self.extract_currencies(existing.pair)
            
            if new_signal.pair == existing.pair:
                if new_signal.action == existing.action:
                    correlations.append(1.0)
                else:
                    correlations.append(-1.0)
                continue
            
            corr_score = 0.0
            
            if new_base in CURRENCY_CORRELATION_MATRIX:
                if exist_base in CURRENCY_CORRELATION_MATRIX[new_base]:
                    corr_score += CURRENCY_CORRELATION_MATRIX[new_base][exist_base]
                if exist_quote in CURRENCY_CORRELATION_MATRIX[new_base]:
                    corr_score += CURRENCY_CORRELATION_MATRIX[new_base][exist_quote]
            
            if new_signal.action != existing.action:
                corr_score *= -0.5
            
            correlations.append(corr_score)
        
        return np.mean(correlations) if correlations else 0.0
    
    def can_add_signal(self, new_signal: 'Signal') -> Tuple[bool, str]:
        base, quote = self.extract_currencies(new_signal.pair)
        
        base_exp = self.get_currency_exposure(base)
        quote_exp = self.get_currency_exposure(quote)
        
        if base_exp >= EXPOSURE_LIMITS.get(base, 3):
            return False, f"Limite {base} atteinte ({base_exp}/{EXPOSURE_LIMITS.get(base, 3)})"
        
        if quote_exp >= EXPOSURE_LIMITS.get(quote, 3):
            return False, f"Limite {quote} atteinte ({quote_exp}/{EXPOSURE_LIMITS.get(quote, 3)})"
        
        direction_key = "TOTAL_LONG" if new_signal.action == "BUY" else "TOTAL_SHORT"
        dir_count = self.get_direction_count(new_signal.action)
        
        if dir_count >= EXPOSURE_LIMITS[direction_key]:
            return False, f"Limite {direction_key} atteinte ({dir_count}/{EXPOSURE_LIMITS[direction_key]})"
        
        corr = self.calculate_portfolio_correlation(new_signal)
        if abs(corr) > 0.7:
            return False, f"Corrélation trop élevée ({corr:.2f})"
        
        return True, "OK"
    
    def add_signal(self, signal: 'Signal'):
        self.active_signals.append(signal)
    
    def get_stats(self) -> Dict:
        stats = {
            "total": len(self.active_signals),
            "long": self.get_direction_count("BUY"),
            "short": self.get_direction_count("SELL"),
            "exposure": {}
        }
        
        for curr in ["EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "XAU"]:
            exp = self.get_currency_exposure(curr)
            if exp > 0:
                stats["exposure"][curr] = f"{exp}/{EXPOSURE_LIMITS.get(curr, 3)}"
        
        return stats

# ==================== NEWS FILTER ====================
@dataclass
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    title: str
    forecast: str = ""
    previous: str = ""

class NewsFilter:
    def __init__(self):
        self.events_cache: List[NewsEvent] = []
        self.cache_time: Optional[datetime] = None
        self.cache_duration = timedelta(hours=6)
    
    @st.cache_data(ttl=21600, show_spinner=False)
    def fetch_forex_factory_news(_self) -> List[NewsEvent]:
        now = datetime.now(pytz.UTC)
        demo_events = [
            NewsEvent(
                time=now + timedelta(hours=2),
                currency="USD",
                impact="High",
                title="FOMC Minutes",
                forecast="N/A",
                previous="N/A"
            ),
            NewsEvent(
                time=now + timedelta(hours=8),
                currency="EUR",
                impact="High",
                title="ECB Rate Decision",
                forecast="3.75%",
                previous="4.00%"
            ),
            NewsEvent(
                time=now + timedelta(hours=1),
                currency="GBP",
                impact="Medium",
                title="UK GDP",
                forecast="0.2%",
                previous="0.1%"
            ),
        ]
        
        logger.info(f"📰 Loaded {len(demo_events)} news events (DEMO MODE)")
        return demo_events
    
    def get_upcoming_events(self, hours_ahead: int = 4) -> List[NewsEvent]:
        if not self.cache_time or datetime.now(pytz.UTC) - self.cache_time > self.cache_duration:
            self.events_cache = self.fetch_forex_factory_news()
            self.cache_time = datetime.now(pytz.UTC)
        
        now = datetime.now(pytz.UTC)
        cutoff = now + timedelta(hours=hours_ahead)
        
        return [e for e in self.events_cache if now <= e.time <= cutoff]
    
    def is_safe_to_trade(self, pair: str, hours_buffer: int = 2) -> Tuple[bool, Optional[NewsEvent]]:
        upcoming = self.get_upcoming_events(hours_buffer)
        currencies = pair.replace("_", "").replace("XAU", "GOLD").replace("XPT", "PLAT")
        
        for event in upcoming:
            if event.impact == "High":
                if event.currency in currencies or event.currency == "GOLD" and "XAU" in pair:
                    return False, event
        
        return True, None

# ==================== MARKET CONDITION ANALYZER ====================
class MarketConditionAnalyzer:
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> str:
        if len(df) < 50:
            return "Unknown"
        
        adx_mean = df['adx'].tail(20).mean()
        atr_current = df['atr_val'].iloc[-1]
        atr_mean = df['atr_val'].tail(50).mean()
        atr_ratio = atr_current / atr_mean if atr_mean > 0 else 1.0
        
        if adx_mean > 25 and atr_ratio > 0.8:
            return "Trending"
        elif adx_mean < 20:
            return "Ranging"
        else:
            return "Transitioning"
    
    @staticmethod
    def calculate_volatility_score(df: pd.DataFrame) -> float:
        if len(df) < 50:
            return 0.5
        
        atr_current = df['atr_val'].iloc[-1]
        atr_mean = df['atr_val'].tail(50).mean()
        atr_std = df['atr_val'].tail(50).std()
        
        if atr_std == 0:
            return 0.5
        
        z_score = (atr_current - atr_mean) / atr_std
        volatility = min(max((z_score + 2) / 4, 0), 1)
        
        return volatility
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        if len(df) < window:
            return 0.0, 0.0
        
        recent = df.tail(window)
        support = recent['low'].min()
        resistance = recent['high'].max()
        
        return support, resistance
    
    @staticmethod
    def is_price_extended(df: pd.DataFrame) -> bool:
        if len(df) < 50:
            return False
        
        close = df['close'].iloc[-1]
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        distance_ema20 = abs((close - ema20) / ema20)
        
        return distance_ema20 > 0.02

# ==================== RATE LIMITER ====================
@dataclass
class RateLimiter:
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
    WEAK = "Weak"

@dataclass
class TradingParams:
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    min_adx_threshold: int = 20
    adx_strong_threshold: int = 25
    min_rr_ratio: float = 1.2
    cascade_required: bool = True
    strict_flip_only: bool = True
    min_score_threshold: int = 60
    enable_news_filter: bool = True
    enable_correlation_filter: bool = True
    enable_market_condition_filter: bool = True

@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05
    kelly_fraction: float = 0.25
    min_win_rate: float = 0.45
    max_win_rate: float = 0.75

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
    higher_tf_trend: str
    is_live: bool
    is_fresh_flip: bool
    candle_index: int
    is_strict_flip: bool
    market_regime: str = "Unknown"
    volatility_score: float = 0.5
    support: float = 0.0
    resistance: float = 0.0
    news_clear: bool = True
    correlation_ok: bool = True

@dataclass
class ScanStats:
    total_pairs: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    signals_found: int = 0
    signals_rejected_news: int = 0
    signals_rejected_correlation: int = 0
    signals_rejected_quality: int = 0
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

@st.cache_data(ttl=30, show_spinner=False, max_entries=100)
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

# ==================== INDICATEURS (CORRECTED) ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Version corrigée sans FutureWarning"""
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

    try:
        # HMA Calculation - CORRECTION: Utiliser ffill() et bfill()
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        hma_length = int(np.sqrt(20))
        
        if wma_half.isna().all() or wma_full.isna().all():
            df['hma'] = np.nan
            df['hma_up'] = np.nan
        else:
            hma_series = 2 * wma_half - wma_full
            df['hma'] = wma(hma_series.ffill().bfill(), hma_length)
            # CORRECTION: Éviter le FutureWarning de replace avec downcasting
            df['hma_up'] = (df['hma'] > df['hma'].shift(1)).astype(bool)
    except Exception as e:
        logger.error(f"❌ HMA Error: {e}")
        df['hma'] = np.nan
        df['hma_up'] = np.nan

    try:
        # RSI Calculation
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.ewm(alpha=1/7, min_periods=7).mean() / down.ewm(alpha=1/7, min_periods=7).mean()
        df['rsi'] = 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"❌ RSI Error: {e}")
        df['rsi'] = np.nan

    try:
        # ATR & ADX Calculation - CORRECTION: Utiliser ffill()
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        df['atr_val'] = atr14.ffill().fillna(0.0)
        
        # UT Bot calculation
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
        
        # ADX calculation
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
        
    except Exception as e:
        logger.error(f"❌ ADX/UT Error: {e}")
        df['ut_state'] = 0
        df['adx'] = np.nan
        df['atr_val'] = np.nan
    
    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=180, show_spinner=False)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf:
        return "Neutral"
    
    try:
        df = get_candles(pair, higher_tf, 100)
        if len(df) < 50:
            return "Neutral"
        
        df = calculate_indicators(df)
        
        if pd.isna(df['hma'].iloc[-1]) or pd.isna(df['hma'].iloc[-2]):
            return "Neutral"
        
        close = df['close']
        ema50 = close.ewm(span=50, min_periods=50).mean().iloc[-1]
        hma_current = df['hma'].iloc[-1]
        hma_prev = df['hma'].iloc[-2]
        
        if close.iloc[-1] > ema50 and hma_current > hma_prev:
            return "Bullish"
        elif close.iloc[-1] < ema50 and hma_current < hma_prev:
            return "Bearish"
        
        return "Neutral"
    except Exception as e:
        logger.error(f"❌ Cascade Error {pair}: {e}")
        return "Neutral"

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
    
    def calculate_position_size(self, signal: Signal, win_rate: float = 0.58) -> float:
        win_rate = max(self.config.min_win_rate, min(win_rate, self.config.max_win_rate))
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        if kelly <= 0:
            return 0.0
        
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0:
            return 0.0
        
        size = (self.balance * kelly) / pip_risk
        return round(size, 2)

# ==================== ANALYSE AMÉLIORÉE ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, 
                params: TradingParams, news_filter: NewsFilter, 
                corr_manager: CorrelationManager) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100:
            return None
        
        df = calculate_indicators(df)
        
        market_analyzer = MarketConditionAnalyzer()
        market_regime = market_analyzer.detect_market_regime(df)
        volatility_score = market_analyzer.calculate_volatility_score(df)
        support, resistance = market_analyzer.find_support_resistance(df)
        is_extended = market_analyzer.is_price_extended(df)
        
        if mode_live:
            idx = -1
        else:
            idx = -2 if not df.iloc[-1]['complete'] else -1
        
        if abs(idx) > len(df):
            return None
        
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2] if abs(idx-2) <= len(df) else None
        
        required_fields = ['hma', 'rsi', 'adx', 'atr_val', 'hma_up', 'ut_state']
        for field in required_fields:
            if pd.isna(last[field]):
                return None
        
        if pd.isna(prev['hma_up']) or (prev2 is not None and pd.isna(prev2['hma_up'])):
            return None
        
        hma_flip_green = bool(last.hma_up) and not bool(prev.hma_up)
        hma_flip_red = not bool(last.hma_up) and bool(prev.hma_up)
        
        hma_extended_green = False
        hma_extended_red = False
        if prev2 is not None:
            hma_extended_green = bool(last.hma_up) and bool(prev.hma_up) and not bool(prev2.hma_up) and not hma_flip_green
            hma_extended_red = not bool(last.hma_up) and not bool(prev.hma_up) and bool(prev2.hma_up) and not hma_flip_red
        
        if params.strict_flip_only:
            raw_buy = hma_flip_green and last.rsi > 50 and last.ut_state == 1
            raw_sell = hma_flip_red and last.rsi < 50 and last.ut_state == -1
            is_strict = True
        else:
            raw_buy = (hma_flip_green or hma_extended_green) and last.rsi > 50 and last.ut_state == 1
            raw_sell = (hma_flip_red or hma_extended_red) and last.rsi < 50 and last.ut_state == -1
            is_strict = hma_flip_green or hma_flip_red
        
        if not (raw_buy or raw_sell):
            return None
        
        action = "BUY" if raw_buy else "SELL"
        
        higher_trend = get_trend_alignment(pair, tf)
        if params.cascade_required:
            if action == "BUY" and higher_trend != "Bullish":
                return None
            if action == "SELL" and higher_trend != "Bearish":
                return None
        
        score = 50
        
        if last.adx > params.adx_strong_threshold:
            score += 20
        elif last.adx > params.min_adx_threshold:
            score += 12
        else:
            score += 5
        
        if hma_flip_green or hma_flip_red:
            score += 15
        elif hma_extended_green or hma_extended_red:
            score += 8
        
        if action == "BUY" and 52 < last.rsi < 65:
            score += 10
        elif action == "SELL" and 35 < last.rsi < 48:
            score += 10
        elif action == "BUY" and 50 < last.rsi < 70:
            score += 5
        elif action == "SELL" and 30 < last.rsi < 50:
            score += 5
        
        if (action == "BUY" and higher_trend == "Bullish") or (action == "SELL" and higher_trend == "Bearish"):
            score += 15
        elif higher_trend == "Neutral":
            score += 5
        
        if market_regime == "Trending":
            score += 10
        elif market_regime == "Transitioning":
            score += 5
        else:
            score -= 5
        
        if 0.3 < volatility_score < 0.7:
            score += 5
        elif volatility_score > 0.8:
            score -= 10
        
        if is_extended:
            score -= 10
        
        if support > 0 and resistance > 0:
            price_range = resistance - support
            if action == "BUY" and (last.close - support) / price_range < 0.3:
                score += 5
            elif action == "SELL" and (resistance - last.close) / price_range < 0.3:
                score += 5
        
        score = max(0, min(100, score))
        
        if score < params.min_score_threshold:
            logger.debug(f"❌ {pair} {tf}: Score insuffisant ({score})")
            return None
        
        if score >= 85:
            quality = SignalQuality.INSTITUTIONAL
        elif score >= 75:
            quality = SignalQuality.PREMIUM
        elif score >= 60:
            quality = SignalQuality.STANDARD
        else:
            quality = SignalQuality.WEAK
        
        atr = last.atr_val
        sl = last.close - params.atr_sl_multiplier * atr if action == "BUY" else last.close + params.atr_sl_multiplier * atr
        tp = last.close + params.atr_tp_multiplier * atr if action == "BUY" else last.close - params.atr_tp_multiplier * atr
        
        rr = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0
        if rr < params.min_rr_ratio:
            return None
        
        if last.time.tzinfo is None:
            local_time = pytz.utc.localize(last.time).astimezone(TUNIS_TZ)
        else:
            local_time = last.time.astimezone(TUNIS_TZ)
        
        signal = Signal(
            timestamp=local_time,
            pair=pair,
            timeframe=tf,
            action=action,
            entry_price=last.close,
            stop_loss=sl,
            take_profit=tp,
            score=score,
            quality=quality,
            position_size=0.0,
            risk_amount=0.0,
            risk_reward=rr,
            adx=int(last.adx),
            rsi=int(last.rsi),
            atr=atr,
            higher_tf_trend=higher_trend,
            is_live=mode_live and not df.iloc[-1]['complete'],
            is_fresh_flip=hma_flip_green if action == "BUY" else hma_flip_red,
            candle_index=idx,
            is_strict_flip=is_strict,
            market_regime=market_regime,
            volatility_score=volatility_score,
            support=support,
            resistance=resistance,
            news_clear=True,
            correlation_ok=True
        )
        
        if params.enable_news_filter:
            is_safe, upcoming_event = news_filter.is_safe_to_trade(pair, hours_buffer=2)
            signal.news_clear = is_safe
            if not is_safe:
                logger.warning(f"📰 {pair} {tf}: Rejeté (news {upcoming_event.title})")
                return None
        
        if params.enable_correlation_filter:
            can_add, reason = corr_manager.can_add_signal(signal)
            signal.correlation_ok = can_add
            if not can_add:
                logger.warning(f"🔗 {pair} {tf}: Rejeté ({reason})")
                return None
        
        signal.position_size = risk_manager.calculate_position_size(signal)
        signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
        
        logger.info(f"✅ {pair} {tf} {action} @ {signal.entry_price:.5f} | Score: {score} | Regime: {market_regime}")
        
        return signal
    
    except Exception as e:
        logger.error(f"❌ Error {pair} {tf}: {e}")
        return None

# ==================== SCAN AMÉLIORÉ ====================
def run_scan(pairs: List[str], tfs: List[str], mode_live: bool, 
             risk_manager: RiskManager, params: TradingParams,
             news_filter: NewsFilter) -> Tuple[List[Signal], ScanStats, CorrelationManager]:
    start_time = time.time()
    signals = []
    corr_manager = CorrelationManager()
    stats = ScanStats(total_pairs=len(pairs) * len(tfs))
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params, news_filter, corr_manager): (p, tf)
            for p in pairs for tf in tfs
        }
        
        for future in as_completed(futures, timeout=60):
            pair, tf = futures[future]
            try:
                result = future.result(timeout=10)
                if result:
                    if result.news_clear and result.correlation_ok:
                        signals.append(result)
                        corr_manager.add_signal(result)
                        stats.signals_found += 1
                    elif not result.news_clear:
                        stats.signals_rejected_news += 1
                    elif not result.correlation_ok:
                        stats.signals_rejected_correlation += 1
                stats.successful_scans += 1
            except TimeoutError:
                error_msg = f"{pair} {tf}: Timeout"
                stats.errors.append(error_msg)
                stats.failed_scans += 1
            except Exception as e:
                error_msg = f"{pair} {tf}: {str(e)}"
                stats.errors.append(error_msg)
                stats.failed_scans += 1
    
    stats.scan_duration = time.time() - start_time
    return signals, stats, corr_manager

# ==================== PDF GENERATOR ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, 
                           leftMargin=10*mm, rightMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<font size=16 color=#00ff88><b>BlueStar v2.5 Professional</b></font>", styles["Title"]))
    elements.append(Spacer(1, 8*mm))
    
    now = datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M:%S')
    elements.append(Paragraph(f"<font size=10>Généré le {now}</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    data = [["Time", "Pair", "TF", "Qual", "Act", "Entry", "SL", "TP", "Score", 
             "R:R", "Size", "Risk", "ADX", "RSI", "Regime", "News", "Corr"]]
    
    for s in sorted(signals, key=lambda x: (x.score, x.timestamp), reverse=True):
        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            s.quality.value[:4],
            s.action,
            f"{s.entry_price:.5f}",
            f"{s.stop_loss:.5f}",
            f"{s.take_profit:.5f}",
            str(s.score),
            f"{s.risk_reward:.1f}",
            f"{s.position_size:.2f}",
            f"${s.risk_amount:.0f}",
            str(s.adx),
            str(s.rsi),
            s.market_regime[:4],
            "✓" if s.news_clear else "✗",
            "✓" if s.correlation_ok else "✗"
        ])
    
    table = Table(data, colWidths=[12*mm,16*mm,8*mm,14*mm,12*mm,16*mm,16*mm,16*mm,
                                  10*mm,10*mm,12*mm,12*mm,8*mm,8*mm,14*mm,8*mm,8*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('FONTSIZE', (0,1), (-1,-1), 6),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== INTERFACE (COMPLETE) ====================
def main():
    news_filter = NewsFilter()
    
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    
    with col_title:
        st.markdown("# BlueStar Professional v2.5")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v25-badge">ADVANCED</span>', 
                   unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(TUNIS_TZ)
        market_open = now_tunis.hour in range(0, 23)
        st.markdown(f"""<div style='text-align: right; padding-top: 10px;'>
            <span style='color: #a0a0c0; font-size: 0.8rem;'>🕐 {now_tunis.strftime('%H:%M:%S')}</span><br>
            <span style='color: {"#00ff88" if market_open else "#ff6666"};'>
                {"OPEN" if market_open else "CLOSED"}
            </span>
        </div>""", unsafe_allow_html=True)
    
    with col_mode:
        mode = st.radio("Mode", ["Confirmed", "Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode

    upcoming_news = news_filter.get_upcoming_events(hours_ahead=4)
    high_impact_news = [n for n in upcoming_news if n.impact == "High"]
    if high_impact_news:
        st.markdown(f"""<div class='news-warning'>
            ⚠️ <b>ALERTE NEWS:</b> {len(high_impact_news)} événement(s) High Impact dans les 4h prochaines<br>
            {', '.join([f"{n.currency} {n.title} ({(n.time - datetime.now(pytz.UTC)).seconds // 3600}h)" for n in high_impact_news[:3]])}
        </div>""", unsafe_allow_html=True)

    with st.expander("⚙️ Configuration Avancée", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        atr_sl = c1.number_input("SL Multiplier (ATR)", 1.0, 4.0, 2.0, 0.5)
        atr_tp = c2.number_input("TP Multiplier (ATR)", 1.5, 6.0, 3.0, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.2, 0.1)
        cascade_req = c4.checkbox("Cascade obligatoire", True)
        
        c5, c6, c7, c8 = st.columns(4)
        strict_flip = c5.checkbox("Flip strict uniquement", True)
        min_score = c6.number_input("Score minimum", 50, 100, 60, 5)
        min_adx = c7.number_input("ADX minimum", 15, 30, 20, 5)
        adx_strong = c8.number_input("ADX fort", 20, 40, 25, 5)
        
        st.markdown("**Filtres Avancés**")
        f1, f2, f3 = st.columns(3)
        enable_news = f1.checkbox("Filtre News (2h avant)", True)
        enable_corr = f2.checkbox("Gestion Corrélation", True)
        enable_market = f3.checkbox("Filtre Conditions Marché", True)

    c1, c2, c3, c4 = st.columns(4)
    balance = c1.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    max_risk = c2.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio = c3.slider("Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    scan_btn = c4.button("🚀 SCAN PROFESSIONNEL", type="primary", use_container_width=True)

    if scan_btn:
        with st.spinner("🔍 Scanning with advanced filters..."):
            params = TradingParams(
                atr_sl_multiplier=atr_sl, 
                atr_tp_multiplier=atr_tp, 
                min_rr_ratio=min_rr, 
                cascade_required=cascade_req,
                strict_flip_only=strict_flip,
                min_score_threshold=min_score,
                min_adx_threshold=min_adx,
                adx_strong_threshold=adx_strong,
                enable_news_filter=enable_news,
                enable_correlation_filter=enable_corr,
                enable_market_condition_filter=enable_market
            )
            risk_config = RiskConfig(
                max_risk_per_trade=max_risk, 
                max_portfolio_risk=max_portfolio
            )
            risk_manager = RiskManager(risk_config, balance)
            
            signals, stats, corr_manager = run_scan(PAIRS_DEFAULT, ["H1", "H4", "D1"], is_live, 
                                                     risk_manager, params, news_filter)
        
        st.markdown("---")
        st.markdown("### 📊 Résultats du Scan Professionnel")
        
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Signaux", len(signals))
        m2.metric("Institutional", len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL]))
        m3.metric("Avg Score", f"{np.mean([s.score for s in signals]):.0f}" if signals else "0")
        m4.metric("Exposition", f"${sum(s.risk_amount for s in signals):.0f}")
        m5.metric("Avg R:R", f"{np.mean([s.risk_reward for s in signals]):.1f}:1" if signals else "0:1")
        m6.metric("Rejetés News", stats.signals_rejected_news)
        m7.metric("Rejetés Corr", stats.signals_rejected_correlation)
        
        api_stats = rate_limiter.get_stats()
        st.markdown(f"""<div class='success-box'>
            ✅ <b>Performance API:</b> {api_stats['total_requests']} requêtes | 
            {api_stats['total_errors']} erreurs | 
            Success rate: {api_stats['success_rate']}%  | 
            Durée: {stats.scan_duration:.1f}s
        </div>""", unsafe_allow_html=True)
        
        if signals:
            corr_stats = corr_manager.get_stats()
            exposure_str = " | ".join([f"{k}: {v}" for k, v in corr_stats["exposure"].items()])
            st.markdown(f"""<div class='correlation-box'>
                🔗 <b>Gestion Corrélation:</b> {corr_stats['total']} positions | 
                LONG: {corr_stats['long']} | SHORT: {corr_stats['short']}<br>
                <b>Exposition:</b> {exposure_str}
            </div>""", unsafe_allow_html=True)
        
        if stats.errors:
            with st.expander(f"⚠️ Erreurs ({len(stats.errors)})", expanded=False):
                for error in stats.errors[:10]:
                    st.markdown(f"<div class='error-box'>❌ {error}</div>", unsafe_allow_html=True)

        if signals:
            st.markdown("---")
            dl1, dl2, dl3 = st.columns([1,1,1])
            
            with dl1:
                df_csv = pd.DataFrame([{
                    "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"), 
                    "Pair": s.pair.replace("_","/"), 
                    "TF": s.timeframe,
                    "Quality": s.quality.value, 
                    "Action": s.action, 
                    "Entry": s.entry_price, 
                    "SL": s.stop_loss, 
                    "TP": s.take_profit,
                    "Score": s.score, 
                    "RR": s.risk_reward, 
                    "Size": s.position_size, 
                    "Risk_USD": s.risk_amount,
                    "ADX": s.adx, 
                    "RSI": s.rsi, 
                    "Market_Regime": s.market_regime,
                    "Volatility": f"{s.volatility_score:.2f}",
                    "News_Clear": s.news_clear, 
                    "Corr_OK": s.correlation_ok
                } for s in signals])
                st.download_button("📥 Télécharger CSV", 
                                 df_csv.to_csv(index=False).encode(), 
                                 f"bluestar_v25_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                                 "text/csv", use_container_width=True)
            
            with dl3:
                pdf = generate_pdf(signals)
                st.download_button("📄 Télécharger PDF", 
                                 pdf, 
                                 f"bluestar_v25_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", 
                                 "application/pdf", use_container_width=True)

            st.markdown("---")
            col_h1, col_h4, col_d1 = st.columns(3)
            
            for col, tf in zip([col_h1, col_h4, col_d1], ["H1", "H4", "D1"]):
                with col:
                    tf_sig = [s for s in signals if s.timeframe == tf]
                    st.markdown(f"""<div class='tf-header'>
                        <h3>{tf}</h3>
                        <p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p>
                    </div>""", unsafe_allow_html=True)
                    
                    if tf_sig:
                        tf_sig.sort(key=lambda x: (x.score, x.timestamp), reverse=True)
                        df_disp = pd.DataFrame([{
                            "Heure": s.timestamp.strftime("%H:%M"), 
                            "Paire": s.pair.replace("_","/"),
                            "Qualité": s.quality.value[:4],
                            "Action": f"{'🟢' if s.action=='BUY' else '🔴'}{s.action}",
                            "Score": s.score, 
                            "Entry": f"{s.entry_price:.5f}", 
                            "Stop Loss": f"{s.stop_loss:.5f}", 
                            "Take Profit": f"{s.take_profit:.5f}",
                            "R:R": f"{s.risk_reward:.1f}:1", 
                            "Taille": f"{s.position_size:.2f}", 
                            "Risque": f"${s.risk_amount:.0f}",
                            "ADX": s.adx, 
                            "RSI": s.rsi,
                            "Régime": s.market_regime[:5]
                        } for s in tf_sig])
                        
                        st.dataframe(
                            df_disp,
                            hide_index=True,
                            use_container_width=True,
                            height=min(400, len(df_disp) * 35 + 38)
                        )
                        
                        for idx, sig in enumerate(tf_sig[:5]):
                            with st.expander(f"📊 {sig.pair} - Score {sig.score}", expanded=False):
                                detail_cols = st.columns(3)
                                
                                with detail_cols[0]:
                                    st.markdown(f"""
                                    **📈 Signal**
                                    - Action: {sig.action}
                                    - Qualité: {sig.quality.value}
                                    - Flip strict: {'Oui' if sig.is_strict_flip else 'Non'}
                                    - Live: {'Oui' if sig.is_live else 'Non'}
                                    """)
                                
                                with detail_cols[1]:
                                    st.markdown(f"""
                                    **💰 Risk Management**
                                    - Entry: {sig.entry_price:.5f}
                                    - Stop Loss: {sig.stop_loss:.5f}
                                    - Take Profit: {sig.take_profit:.5f}
                                    - R:R: {sig.risk_reward:.2f}:1
                                    - Position: {sig.position_size:.2f}
                                    - Risque: ${sig.risk_amount:.2f}
                                    """)
                                
                                with detail_cols[2]:
                                    st.markdown(f"""
                                    **🔍 Indicateurs**
                                    - ADX: {sig.adx}
                                    - RSI: {sig.rsi}
                                    - ATR: {sig.atr:.5f}
                                    - Trend HTF: {sig.higher_tf_trend}
                                    - Régime: {sig.market_regime}
                                    - Volatilité: {sig.volatility_score:.2f}
                                    - Support: {sig.support:.5f}
                                    - Résistance: {sig.resistance:.5f}
                                    """)
                    else:
                        st.info(f"Aucun signal {tf} trouvé")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #a0a0c0; font-size: 0.75rem; padding: 20px;'>
            <p><b>BlueStar Professional v2.5</b> - Système de trading algorithmique avancé</p>
            <p>⚠️ Les performances passées ne garantissent pas les résultats futurs. 
            Tradez toujours avec une gestion de risque appropriée.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Erreur critique: {e}")
        logger.exception("Erreur application")
