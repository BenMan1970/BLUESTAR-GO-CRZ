"""
BlueStar Cascade - VERSION 3.0 INSTITUTIONAL GRADE ENHANCED - FIXED
CORRECTIONS MAJEURES :
ATR Percentile calcul corrigé (vrai percentile 0-100)
HMA flip detection fix (gestion NaN + bool strict)
Paramètres par défaut optimisés (moins restrictifs)
Logging détaillé pour debugging
Validation stricte des types bool
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
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

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== FOREX SESSIONS ====================
def get_active_session(dt: datetime) -> str:
    hour_utc = dt.astimezone(pytz.UTC).hour
    if 0 <= hour_utc < 9:
        return "Tokyo"
    elif 8 <= hour_utc < 17:
        return "London"
    elif 13 <= hour_utc < 22:
        return "NY"
    else:
        return "Off-Hours"

def get_session_badge(session: str) -> str:
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
    if len(signals) <= 1:
        return signals
    filtered = []
    pairs_added = []
    sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
    for signal in sorted_signals:
        is_correlated = False
        for existing_pair in pairs_added:
            corr = calculate_pair_correlation(signal.pair, existing_pair)
            if abs(corr) > max_corr:
                logger.info(f"{signal.pair} filtré (corr={corr:.2f} avec {existing_pair})")
                is_correlated = True
                break
        if not is_correlated:
            filtered.append(signal)
            pairs_added.append(signal.pair)
    return filtered

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
        logger.error(f"OANDA Token Error: {e}")
        st.error("OANDA Token manquant dans secrets Streamlit")
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
                    logger.warning(f"API Error (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(rate_limiter.backoff(attempt))
                    else:
                        rate_limiter.record_error()
                        raise
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
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
        logger.warning(f"Timeframe invalide: {tf}")
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
        logger.debug(f"{pair} {tf}: {len(df)} candles")
        return df
    except Exception as e:
        logger.error(f"API Error {pair} {tf}: {e}")
        raise

def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    cache_key = get_cache_key(pair, tf, count)
    return _cache_wrapper(cache_key, pair, tf, count)

# ==================== INDICATEURS OPTIMISÉS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df

    close = df['close']
    high = df['high']
    low = df['low']

    def wma(series, length):
        if len(series) < length:
            return pd.Series([np.nan] * len(series), index=series.index)
        weights = np.arange(1, length + 1)
        return series.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan, raw=True
        )

    # HMA avec FIX NaN + bool strict
    try:
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        hma_length = int(np.sqrt(20))
        raw_hma = 2 * wma_half - wma_full
        df['hma'] = wma(raw_hma, hma_length)

        hma_up = df['hma'] > df['hma'].shift(1)
        df['hma_up'] = hma_up.fillna(False).astype(bool)
    except Exception as e:
        logger.error(f"HMA Error: {e}")
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
        df['rsi'] = np.nan

    # ATR + UT Bot + ADX
    try:
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        df['atr_val'] = atr14

        # UT Bot Trailing Stop
        nLoss = 2.0 * atr14
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
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()

        # ATR Percentile FIXÉ (vrai percentile 0-100)
        def calc_percentile(col):
            return col.rank(pct=True) * 100
        df['atr_percentile'] = df['atr_val'].rolling(100, min_periods=50).apply(
            lambda x: calc_percentile(pd.Series(x)).iloc[-1] if len(x) >= 50 else np.nan, raw=True
        )
    except Exception as e:
        logger.error(f"ADX/UT Error: {e}")
        df['ut_state'] = 0
        df['adx'] = np.nan
        df['atr_val'] = np.nan
        df['atr_percentile'] = np.nan

    return df

# ==================== CASCADE MULTI-TF ====================
@st.cache_data(ttl=180, show_spinner=False)
def get_trend_alignment(pair: str, signal_tf: str) -> Tuple[str, float]:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf:
        return "Neutral", 0.0

    try:
        df = get_candles(pair, higher_tf, 100)
        if len(df) < 50:
            return "Neutral", 0.0
        df = calculate_indicators(df)
        if df['hma'].isna().iloc[-1] or df['hma'].isna().iloc[-2]:
            return "Neutral", 0.0

        close_val = df['close'].iloc[-1]
        ema50 = df['close'].ewm(span=50, min_periods=50).mean().iloc[-1]
        ema200 = df['close'].ewm(span=200, min_periods=100).mean().iloc[-1] if len(df) >= 100 else ema50
        hma_up = df['hma'].iloc[-1] > df['hma'].iloc[-2]
        adx_val = df['adx'].iloc[-1]

        strength = 0.0
        if close_val > ema50 and hma_up:
            trend = "Bullish"
            strength += 30
            if close_val > ema200: strength += 20
            if adx_val > 25: strength += 30
            if df['ut_state'].iloc[-1] == 1: strength += 20
        elif close_val < ema50 and not hma_up:
            trend = "Bearish"
            strength += 30
            if close_val < ema200: strength += 20
            if adx_val > 25: strength += 30
            if df['ut_state'].iloc[-1] == -1: strength += 20
        else:
            trend = "Neutral"
            strength = 0
        return trend, min(100, strength)
    except Exception as e:
        logger.error(f"Cascade Error {pair}: {e}")
        return "Neutral", 0.0

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance

    def calculate_position_size(self, signal: Signal, win_rate: float = 0.58) -> float:
        win_rate = max(self.config.min_win_rate, min(win_rate, self.config.max_win_rate))
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        kelly *= (signal.score / 100)

        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0:
            return 0.0

        size = (self.balance * kelly) / pip_risk
        max_position_risk = self.balance * self.config.max_risk_per_trade
        if size * pip_risk > max_position_risk:
            size = max_position_risk / pip_risk
        return round(size, 2)

# ==================== ANALYSE PAIRE ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100:
            return None
        df = calculate_indicators(df)

        idx = -1 if mode_live and df.iloc[-1]['complete'] else -2
        if abs(idx) >= len(df):
            return None

        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2] if abs(idx-2) < len(df) else None

        # Validation stricte
        for field in ['hma_up', 'rsi', 'adx', 'atr_val', 'ut_state', 'atr_percentile']:
            if pd.isna(last[field]):
                return None
        if not isinstance(last['hma_up'], (bool, np.bool_)):
            return None

        # Volatilité
        if last['atr_percentile'] < params.min_volatility_percentile:
            return None

        # Flip detection
        hma_flip_green = last['hma_up'] and not prev['hma_up']
        hma_flip_red = not last['hma_up'] and prev['hma_up']
        hma_extended_green = prev2 is not None and last['hma_up'] and prev['hma_up'] and not prev2['hma_up']
        hma_extended_red = prev2 is not None and not last['hma_up'] and not prev['hma_up'] and prev2['hma_up']

        if params.strict_flip_only:
            raw_buy = hma_flip_green and last['rsi'] > 50 and last['ut_state'] == 1
            raw_sell = hma_flip_red and last['rsi'] < 50 and last['ut_state'] == -1
            is_strict = True
        else:
            raw_buy = (hma_flip_green or hma_extended_green) and last['rsi'] > 50 and last['ut_state'] == 1
            raw_sell = (hma_flip_red or hma_extended_red) and last['rsi'] < 50 and last['ut_state'] == -1
            is_strict = hma_flip_green or hma_flip_red

        if not (raw_buy or raw_sell):
            return None

        action = "BUY" if raw_buy else "SELL"

        # Cascade
        higher_trend, trend_strength = get_trend_alignment(pair, tf)
        if params.cascade_required:
            if (action == "BUY" and higher_trend != "Bullish") or (action == "SELL" and higher_trend != "Bearish"):
                return None

        # Scoring
        score = 50
        if last['adx'] > params.adx_strong_threshold:
            score += 30
        elif last['adx'] > params.min_adx_threshold:
            score += 15
        if hma_flip_green or hma_flip_red:
            score += 20
        elif hma_extended_green or hma_extended_red:
            score += 10
        if action == "BUY" and 50 < last['rsi'] < 70:
            score += 15 if last['rsi'] < 60 else 10
        elif action == "SELL" and 30 < last['rsi'] < 50:
            score += 15 if last['rsi'] > 40 else 10
        score += int(trend_strength * 0.25)
        if last['atr_percentile'] > 50: score += 10
        elif last['atr_percentile'] > 30: score += 5
        score = max(params.min_score_threshold, min(100, score))
        if score < params.min_score_threshold:
            return None

        # Session
        t = last['time']
        local_time = t.astimezone(TUNIS_TZ) if t.tzinfo else pytz.utc.localize(t).astimezone(TUNIS_TZ)
        session = get_active_session(local_time)
        if params.session_filter and session == "Off-Hours":
            return None

        # SL/TP
        atr = last['atr_val']
        sl = last['close'] - params.atr_sl_multiplier * atr if action == "BUY" else last['close'] + params.atr_sl_multiplier * atr
        tp = last['close'] + params.atr_tp_multiplier * atr if action == "BUY" else last['close'] - params.atr_tp_multiplier * atr
        rr = abs(tp - last['close']) / abs(last['close'] - sl) if abs(last['close'] - sl) > 0 else 0
        if rr < params.min_rr_ratio:
            return None

        quality = (SignalQuality.INSTITUTIONAL if score >= 85 else
                  SignalQuality.PREMIUM if score >= 75 else SignalQuality.STANDARD)

        signal = Signal(
            timestamp=local_time, pair=pair, timeframe=tf, action=action,
            entry_price=last['close'], stop_loss=sl, take_profit=tp,
            score=score, quality=quality, position_size=0.0, risk_amount=0.0,
            risk_reward=rr, adx=int(last['adx']), rsi=int(last['rsi']),
            atr=atr, atr_percentile=last['atr_percentile'],
            higher_tf_trend=higher_trend, is_live=mode_live and idx == -1 and not df.iloc[-1]['complete'],
            is_fresh_flip=hma_flip_green if action == "BUY" else hma_flip_red,
            candle_index=idx, is_strict_flip=is_strict, session=session
        )

        signal.position_size = risk_manager.calculate_position_size(signal)
        signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size

        return signal
    except Exception as e:
        logger.error(f"Error {pair} {tf}: {e}")
        return None

# ==================== SCAN ====================
def run_scan(pairs: List[str], tfs: List[str], mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Tuple[List[Signal], ScanStats]:
    start_time = time.time()
    signals = []
    stats = ScanStats(total_pairs=len(pairs) * len(tfs))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params): (p, tf) for p in pairs for tf in tfs}
        for future in as_completed(futures, timeout=90):
            pair, tf = futures[future]
            try:
                result = future.result(timeout=15)
                if result:
                    signals.append(result)
                    stats.signals_found += 1
                stats.successful_scans += 1
            except FutureTimeoutError:
                stats.errors.append(f"{pair} {tf}: Timeout")
                stats.failed_scans += 1
            except Exception as e:
                stats.errors.append(f"{pair} {tf}: {str(e)}")
                stats.failed_scans += 1

    original = len(signals)
    if params.max_correlation < 1.0:
        signals = check_portfolio_correlation(signals, params.max_correlation)
        stats.signals_filtered = original - len(signals)

    stats.scan_duration = time.time() - start_time
    return signals, stats

# ==================== PDF ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, leftMargin=10*mm, rightMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<font size=16 color=#00ff88><b>BlueStar v3.0 FIXED</b></font>", styles["Title"]))
    elements.append(Spacer(1, 8*mm))
    elements.append(Paragraph(f"<font size=10>Généré le {datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M:%S')}</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))

    data = [["Heure", "Paire", "TF", "Qualité", "Action", "Entry", "SL", "TP", "Score", "R:R", "Size", "Risk", "ADX", "RSI", "Trend", "Session", "ATR%"]]
    for s in sorted(signals, key=lambda x: (x.score, x.timestamp), reverse=True):
        data.append([
            s.timestamp.strftime("%H:%M"), s.pair.replace("_", "/"), s.timeframe, s.quality.value[:4], s.action,
            f"{s.entry_price:.5f}", f"{s.stop_loss:.5f}", f"{s.take_profit:.5f}", str(s.score), f"{s.risk_reward:.1f}",
            f"{s.position_size:.2f}", f"${s.risk_amount:.0f}", str(s.adx), str(s.rsi), s.higher_tf_trend[:4], s.session[:3], f"{s.atr_percentile:.0f}"
        ])

    table = Table(data, colWidths=[14*mm,18*mm,10*mm,16*mm,14*mm,18*mm,18*mm,18*mm,10*mm,10*mm,14*mm,14*mm,10*mm,10*mm,16*mm,12*mm,12*mm])
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

# ==================== INTERFACE ====================
def main():
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    with col_title:
        st.markdown("# BlueStar Enhanced v3.0")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v30-badge">v3.0 ADVANCED</span><span class="fixed-badge">FIXED</span>', unsafe_allow_html=True)

    with col_time:
        now_tunis = datetime.now(TUNIS_TZ)
        session = get_active_session(now_tunis)
        st.markdown(f"""<div style='text-align: right; padding-top: 10px;'>
            <span style='color: #a0a0c0; font-size: 0.8rem;'>Tunis {now_tunis.strftime('%H:%M:%S')}</span><br>
            <span style='color: {"#00ff88" if session != "Off-Hours" else "#ff6666"};'>
                {"OPEN" if session != "Off-Hours" else "CLOSED"}
            </span> {get_session_badge(session)}
        </div>""", unsafe_allow_html=True)

    with col_mode:
        mode = st.radio("Mode", ["Confirmed", "Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode

    with st.expander("Configuration Avancée", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        atr_sl = c1.number_input("SL Multiplier (ATR)", 1.0, 4.0, 2.0, 0.5)
        atr_tp = c2.number_input("TP Multiplier (ATR)", 1.5, 6.0, 3.5, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.5, 0.1)
        cascade_req = c4.checkbox("Cascade obligatoire", False)

        c5, c6, c7, c8 = st.columns(4)
        strict_flip = c5.checkbox("Flip strict uniquement", False)
        min_score = c6.number_input("Score minimum", 50, 100, 55, 5)
        min_adx = c7.number_input("ADX minimum", 15, 35, 20, 1)
        adx_strong = c8.number_input("ADX fort", 20, 45, 25, 1)

        st.markdown("**Filtres Avancés v3.0 FIXED**")
        c9, c10, c11, _ = st.columns(4)
        min_vol_pct = c9.slider("Min Volatilité %ile", 0, 70, 15, 5)
        max_corr = c10.slider("Max Corrélation", 0.5, 1.0, 0.75, 0.05)
        session_filter = c11.checkbox("Filtrer off-hours", False)

    c1, c2, c3, c4 = st.columns(4)
    balance = c1.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    max_risk = c2.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio = c3.slider("Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    scan_btn = c4.button("SCAN v3.0 FIXED", type="primary", use_container_width=True)

    if scan_btn:
        params = TradingParams(
            atr_sl_multiplier=atr_sl, atr_tp_multiplier=atr_tp, min_rr_ratio=min_rr,
            cascade_required=cascade_req, strict_flip_only=strict_flip,
            min_score_threshold=min_score, min_adx_threshold=min_adx,
            adx_strong_threshold=adx_strong, min_volatility_percentile=min_vol_pct,
            max_correlation=max_corr, session_filter=session_filter
        )
        risk_config = RiskConfig(max_risk_per_trade=max_risk, max_portfolio_risk=max_portfolio)
        risk_manager = RiskManager(risk_config, balance)

        with st.spinner("Scan en cours sur 80+ paires (H1/H4/D1)..."):
            signals, stats = run_scan(PAIRS_DEFAULT, ["H1", "H4", "D1"], is_live, risk_manager, params)

        st.success(f"Scan terminé en {stats.scan_duration:.1f}s !")

        st.markdown("---")
        st.markdown("### Résultats du Scan v3.0 FIXED")

        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Signaux", len(signals))
        m2.metric("Institutional", len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL]))
        m3.metric("Filtrés", stats.signals_filtered)
        avg_score = np.mean([s.score for s in signals]) if signals else 0
        m4.metric("Avg Score", f"{avg_score:.0f}")
        total_risk = sum(s.risk_amount for s in signals)
        m5.metric("Exposition", f"${total_risk:.0f}")
        avg_rr = np.mean([s.risk_reward for s in signals]) if signals else 0
        m6.metric("Avg R:R", f"{avg_rr:.1f}:1")
        m7.metric("Durée", f"{stats.scan_duration:.1f}s")

        api_stats = rate_limiter.get_stats()
        st.markdown(f"<div class='success-box'>Performance API: {api_stats['total_requests']} req | {api_stats['total_errors']} err | Success: {api_stats['success_rate']}%</div>", unsafe_allow_html=True)

        if stats.signals_filtered > 0:
            st.markdown(f"<div class='alert-box'>Correlation Filter: {stats.signals_filtered} signaux filtrés</div>", unsafe_allow_html=True)

        if signals:
            col1, _, col3 = st.columns([1, 1, 1])
            with col1:
                csv = pd.DataFrame([{k: getattr(s, k) if hasattr(s, k) else v for k, v in {
                    "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"), "Pair": s.pair.replace("_","/"), "TF": s.timeframe,
                    "Quality": s.quality.value, "Action": s.action, "Entry": s.entry_price, "SL": s.stop_loss,
                    "TP": s.take_profit, "Score": s.score, "RR": s.risk_reward, "Size": s.position_size,
                    "Risk_USD": s.risk_amount, "ADX": s.adx, "RSI": s.rsi, "ATR%": round(s.atr_percentile,1),
                    "Trend": s.higher_tf_trend, "Session": s.session
                }.items()} for s in signals]).to_csv(index=False).encode()
                st.download_button("Télécharger CSV", csv, f"bluestar_v30_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

            with col3:
                pdf = generate_pdf(signals)
                st.download_button("Télécharger PDF", pdf, f"bluestar_v30_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")

            st.markdown("---")
            for tf in ["H1", "H4", "D1"]:
                tf_sig = [s for s in signals if s.timeframe == tf]
                if tf_sig:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df_disp = pd.DataFrame([{
                        "Heure": s.timestamp.strftime("%H:%M"), "Paire": s.pair.replace("_","/"), "Qualité": s.quality.value[:4],
                        "Action": f"{'BUY' if s.action=='BUY' else 'SELL'}", "Score": s.score, "Entry": f"{s.entry_price:.5f}",
                        "SL": f"{s.stop_loss:.5f}", "TP": f"{s.take_profit:.5f}", "R:R": f"{s.risk_reward:.1f}:1",
                        "Taille": f"{s.position_size:.2f}", "Risque": f"${s.risk_amount:.0f}", "ADX": s.adx, "RSI": s.rsi,
                        "ATR%": f"{s.atr_percentile:.0f}", "Trend": s.higher_tf_trend[:4], "Session": s.session[:3]
                    } for s in sorted(tf_sig, key=lambda x: x.score, reverse=True)])
                    st.dataframe(df_disp, use_container_width=True, hide_index=True)
        else:
            st.warning("Aucun signal détecté.")
            st.info(f"""
            **Paramètres actuels :**
            - Volatilité ≥ {min_vol_pct}%ile
            - Score ≥ {min_score}
            - Cascade : {'Oui' if cascade_req else 'Non'}
            - Flip strict : {'Oui' if strict_flip else 'Non'}
            """)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.7rem; padding: 15px;'>"
                "<b>BlueStar Cascade Enhanced v3.0 FIXED</b> | Institutional Grade | "
                "ATR Percentile Fix | HMA Flip Fix | Optimized Defaults | Debug Logging"
                "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
