# bluestar_v2_5.py
"""
BlueStar Cascade - VERSION 2.5 INSTITUTIONAL OPTIMIZED
Changements logiques (HMA flip robust, RSI dynamique, cascade momentum,
scoring pondéré, R:R dynamique, filtre bougie, double validation).
Aucun changement visuel.
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
st.set_page_config(page_title="BlueStar Institutional v2.5", layout="wide", initial_sidebar_state="collapsed")

# Logging amélioré
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
    .v24-badge {background: linear-gradient(45deg, #ff6b6b, #ee5a6f); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block; margin-left: 8px;}
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

# ==================== RATE LIMITER ROBUSTE ====================
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
    atr_tp_multiplier: float = 3.0
    min_adx_threshold: int = 20
    adx_strong_threshold: int = 25
    min_rr_ratio: float = 1.2
    cascade_required: bool = True
    strict_flip_only: bool = True
    min_score_threshold: int = 50

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

@dataclass
class ScanStats:
    total_pairs: int = 0
    successful_scans: int = 0
    failed_scans: int = 0
    signals_found: int = 0
    scan_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

# ==================== OANDA API AVEC RETRY ====================
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

# ==================== INDICATEURS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        logger.warning("Pas assez de données")
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
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        hma_length = int(np.sqrt(20))
        
        if wma_half.isna().all() or wma_full.isna().all():
            df['hma'] = np.nan
            df['hma_up'] = np.nan
        else:
            df['hma'] = wma(2 * wma_half - wma_full, hma_length)
            df['hma_up'] = (df['hma'] > df['hma'].shift(1)).astype(float)
            df['hma_up'] = df['hma_up'].replace(0, False).replace(1, True)
    except Exception as e:
        logger.error(f"HMA Error: {e}")
        df['hma'] = np.nan
        df['hma_up'] = np.nan

    try:
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        rs = up.ewm(alpha=1/7, min_periods=7).mean() / down.ewm(alpha=1/7, min_periods=7).mean()
        df['rsi'] = 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"RSI Error: {e}")
        df['rsi'] = np.nan

    try:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
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
        
        atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
        df['atr_val'] = atr14
    except Exception as e:
        logger.error(f"ADX/UT Error: {e}")
        df['ut_state'] = 0
        df['adx'] = np.nan
        df['atr_val'] = np.nan
    
    return df

# ==================== CASCADE (v2.5) ====================
@st.cache_data(ttl=180, show_spinner=False)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf:
        return "Neutral"
    
    try:
        df = get_candles(pair, higher_tf, 150)
        if len(df) < 60:
            return "Neutral"
        
        df = calculate_indicators(df)
        
        if pd.isna(df['hma'].iloc[-1]) or pd.isna(df['adx'].iloc[-1]):
            return "Neutral"
        
        close = df['close']
        ema50 = close.ewm(span=50, min_periods=50).mean().iloc[-1]
        hma_current = df['hma'].iloc[-1]
        hma_prev = df['hma'].iloc[-2]
        hma_up = hma_current > hma_prev
        adx_momentum = df['adx'].iloc[-1] > 18  # require momentum on higher TF
        
        if close.iloc[-1] > ema50 and hma_up and adx_momentum:
            return "Bullish"
        elif close.iloc[-1] < ema50 and not hma_up and adx_momentum:
            return "Bearish"
        
        return "Neutral"
    except Exception as e:
        logger.error(f"Cascade Error {pair}: {e}")
        return "Neutral"

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
    
    def calculate_position_size(self, signal: Signal, win_rate: float = 0.58) -> float:
        win_rate = max(self.config.min_win_rate, min(win_rate, self.config.max_win_rate))
        if signal.risk_reward <= 0:
            return 0.0
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        if kelly <= 0:
            return 0.0
        
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0:
            return 0.0
        
        size = (self.balance * kelly) / pip_risk
        return round(size, 2)

# ==================== ANALYSE (v2.5) ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Optional[Signal]:
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
        
        required_fields = ['hma', 'rsi', 'adx', 'atr_val', 'hma_up', 'ut_state']
        for field in required_fields:
            if pd.isna(last[field]):
                return None
        
        if pd.isna(prev['hma_up']) or (prev2 is not None and pd.isna(prev2['hma_up'])):
            return None
        
        # === v2.5 HMA Flip Logic (Robuste) ===
        hma_slope = last.hma - prev.hma
        hma_slope_prev = prev.hma - (prev2.hma if prev2 is not None else prev.hma)
        MIN_HMA_SLOPE = 0.15 * last.atr_val if last.atr_val and last.atr_val > 0 else 0.0001

        hma_flip_green = (
            bool(last.hma_up)
            and not bool(prev.hma_up)
            and hma_slope > MIN_HMA_SLOPE
            and hma_slope_prev < MIN_HMA_SLOPE
        )

        hma_flip_red = (
            not bool(last.hma_up)
            and bool(prev.hma_up)
            and hma_slope < -MIN_HMA_SLOPE
            and hma_slope_prev > -MIN_HMA_SLOPE
        )

        hma_extended_green = (
            bool(last.hma_up)
            and bool(prev.hma_up)
            and (prev2 is not None and not bool(prev2.hma_up))
            and hma_slope > MIN_HMA_SLOPE
        )

        hma_extended_red = (
            not bool(last.hma_up)
            and not bool(prev.hma_up)
            and (prev2 is not None and bool(prev2.hma_up))
            and hma_slope < -MIN_HMA_SLOPE
        )

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

        # === v2.5 Cascade requirement (with momentum on higher TF) ===
        higher_trend = get_trend_alignment(pair, tf)
        if params.cascade_required:
            if action == "BUY" and higher_trend != "Bullish":
                return None
            if action == "SELL" and higher_trend != "Bearish":
                return None

        # === v2.5 Candle Quality Filter ===
        candle_range = last.high - last.low
        if candle_range == 0:
            return None
        body_ratio = abs(last.close - last.open) / candle_range
        if body_ratio < 0.35:
            return None

        # === v2.5 Scoring ===
        score = 50

        # ADX momentum
        if last.adx > params.adx_strong_threshold:
            score += 20
        elif last.adx > params.min_adx_threshold:
            score += 10
        else:
            score -= 5

        # Flip quality
        if hma_flip_green or hma_flip_red:
            score += 20
        elif hma_extended_green or hma_extended_red:
            score += 8

        # RSI optimal bands (stricter)
        if action == "BUY" and 55 < last.rsi < 62:
            score += 10
        elif action == "BUY" and 52 < last.rsi <= 55:
            score += 5

        if action == "SELL" and 38 < last.rsi < 45:
            score += 10
        elif action == "SELL" and 45 <= last.rsi < 48:
            score += 5

        # Cascade confirmed
        if higher_trend in ["Bullish", "Bearish"]:
            score += 10

        score = int(min(100, max(score, params.min_score_threshold)))

        if score < params.min_score_threshold:
            return None

        quality = (SignalQuality.INSTITUTIONAL if score >= 90 
                  else SignalQuality.PREMIUM if score >= 80 
                  else SignalQuality.STANDARD)

        atr = last.atr_val
        sl = last.close - params.atr_sl_multiplier * atr if action == "BUY" else last.close + params.atr_sl_multiplier * atr
        tp = last.close + params.atr_tp_multiplier * atr if action == "BUY" else last.close - params.atr_tp_multiplier * atr

        rr = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0

        # === v2.5 Dynamic R:R filter ===
        volatility_factor = 1 + (last.atr_val / last.close) * 200 if last.atr_val and last.close else 1.0
        dynamic_rr = params.min_rr_ratio * volatility_factor

        if rr < dynamic_rr:
            return None

        # === v2.5 RSI band stricter check (post-score) ===
        if action == "BUY" and not (52 < last.rsi < 68):
            return None
        if action == "SELL" and not (32 < last.rsi < 48):
            return None

        # === v2.5 Directional double validation (UT_State + HMA) ===
        if action == "BUY":
            if not (last.ut_state == 1 and last.hma > prev.hma):
                return None
        if action == "SELL":
            if not (last.ut_state == -1 and last.hma < prev.hma):
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
            is_strict_flip=is_strict
        )

        signal.position_size = risk_manager.calculate_position_size(signal)
        signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size

        logger.info(f"{pair} {tf} {action} @ {signal.entry_price:.5f} | Score: {score}")
        return signal

    except Exception as e:
        logger.error(f"Error {pair} {tf}: {e}")
        return None

# ==================== SCAN ====================
def run_scan(pairs: List[str], tfs: List[str], mode_live: bool, 
             risk_manager: RiskManager, params: TradingParams) -> Tuple[List[Signal], ScanStats]:
    start_time = time.time()
    signals = []
    stats = ScanStats(total_pairs=len(pairs) * len(tfs))
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params): (p, tf)
            for p in pairs for tf in tfs
        }
        
        for future in as_completed(futures, timeout=60):
            pair, tf = futures[future]
            try:
                result = future.result(timeout=10)
                if result:
                    signals.append(result)
                    stats.signals_found += 1
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
    return signals, stats

# ==================== PDF ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, 
                           leftMargin=10*mm, rightMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<font size=16 color=#00ff88><b>BlueStar v2.5</b></font>", styles["Title"]))
    elements.append(Spacer(1, 8*mm))
    
    now = datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M:%S')
    elements.append(Paragraph(f"<font size=10>Généré le {now}</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    data = [["Heure", "Paire", "TF", "Qualité", "Action", "Entry", "SL", "TP", "Score", 
             "R:R", "Size", "Risk", "ADX", "RSI", "Trend", "Flip", "Live"]]
    
    for s in sorted(signals, key=lambda x: (x.score, x.timestamp), reverse=True):
        flip_type = "Strict" if s.is_strict_flip else "Ext"
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
            s.higher_tf_trend[:4],
            flip_type,
            "Y" if s.is_live else "N"
        ])
    
    table = Table(data, colWidths=[14*mm,18*mm,10*mm,16*mm,14*mm,18*mm,18*mm,18*mm,
                                  10*mm,10*mm,14*mm,14*mm,10*mm,10*mm,16*mm,12*mm,10*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('FONTSIZE', (0,1), (-1,-1), 7),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== INTERFACE (UNCHANGED VISUALLY) ====================
def main():
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    
    with col_title:
        st.markdown("# BlueStar Enhanced v2.5")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v24-badge">OPTIMIZED</span>', 
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

    with st.expander("⚙️ Configuration Avancée", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        atr_sl = c1.number_input("SL Multiplier (ATR)", 1.0, 4.0, 2.0, 0.5)
        atr_tp = c2.number_input("TP Multiplier (ATR)", 1.5, 6.0, 3.0, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.2, 0.1)
        cascade_req = c4.checkbox("Cascade obligatoire", True)
        
        c5, c6, c7, c8 = st.columns(4)
        strict_flip = c5.checkbox("Flip strict uniquement", True)
        min_score = c6.number_input("Score minimum", 50, 100, 50, 5)
        min_adx = c7.number_input("ADX minimum", 15, 30, 20, 5)
        adx_strong = c8.number_input("ADX fort", 20, 40, 25, 5)

    c1, c2, c3, c4 = st.columns(4)
    balance = c1.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    max_risk = c2.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio = c3.slider("Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    scan_btn = c4.button("🚀 SCAN", type="primary", width="stretch")

    if scan_btn:
        with st.spinner("🔍 Scanning markets..."):
            params = TradingParams(
                atr_sl_multiplier=atr_sl, 
                atr_tp_multiplier=atr_tp, 
                min_rr_ratio=min_rr, 
                cascade_required=cascade_req,
                strict_flip_only=strict_flip,
                min_score_threshold=min_score,
                min_adx_threshold=min_adx,
                adx_strong_threshold=adx_strong
            )
            risk_config = RiskConfig(
                max_risk_per_trade=max_risk, 
                max_portfolio_risk=max_portfolio
            )
            risk_manager = RiskManager(risk_config, balance)
            
            signals, stats = run_scan(PAIRS_DEFAULT, ["H1", "H4", "D1"], is_live, 
                                     risk_manager, params)
        
        st.markdown("---")
        st.markdown("### 📊 Résultats du Scan")
        
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Signaux", len(signals))
        m2.metric("Institutional", len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL]))
        m3.metric("Avg Score", f"{np.mean([s.score for s in signals]):.0f}" if signals else "0")
        m4.metric("Exposition", f"${sum(s.risk_amount for s in signals):.0f}")
        m5.metric("Avg R:R", f"{np.mean([s.risk_reward for s in signals]):.1f}:1" if signals else "0:1")
        m6.metric("Durée", f"{stats.scan_duration:.1f}s")
        
        api_stats = rate_limiter.get_stats()
        st.markdown(f"""<div class='success-box'>
            ✅ <b>Performance API:</b> {api_stats['total_requests']} requêtes | 
            {api_stats['total_errors']} erreurs | 
            Success rate: {api_stats['success_rate']}%
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
                    "Higher_TF": s.higher_tf_trend, 
                    "Fresh_Flip": s.is_fresh_flip, 
                    "Live": s.is_live
                } for s in signals])
                st.download_button("📥 Télécharger CSV", 
                                 df_csv.to_csv(index=False).encode(), 
                                 f"bluestar_v25_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                                 "text/csv", width="stretch")
            
            with dl3:
                pdf = generate_pdf(signals)
                st.download_button("📄 Télécharger PDF", 
                                 pdf, 
                                 f"bluestar_v25_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", 
                                 "application/pdf", width="stretch")

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
                            "Trend": s.higher_tf_trend[:4]
                        } for s in tf_sig])
                        st.dataframe(df_disp, width="stretch", hide_index=True, height=400)
                    else:
                        st.info("Aucun signal")

        else:
            st.warning("⚠️ Aucun signal détecté avec les paramètres actuels")

    st.markdown("---")
    st.markdown("""<div style='text-align: center; color: #666; font-size: 0.7rem; padding: 15px;'>
        <b>BlueStar Cascade Enhanced v2.5</b> | Institutional Grade | 
        Rate Limiter ✅ | Retry Logic ✅ | Timeout ✅ | HMA Validation ✅ | Score >= 50 ✅
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
