"""
BlueStar Institutional v3.0
Professional Grade Algorithm
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
from reportlab.lib.pagesizes import A4, landscape
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
    
    /* STYLE METRIQUE */
    .stMetric {background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.7rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.2rem !important; font-weight: 700;}
    
    /* BADGES */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block;}
    .v30-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block; margin-left: 8px;}
    
    /* SUPPRESSION QUADRILLAGE ET BORDURES */
    .stDataFrame {font-size: 0.75rem !important;}
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stDataFrame"] div[role="grid"] {border: none !important;}
    [data-testid="stDataFrame"] div[role="row"] {border: none !important; background-color: transparent !important;}
    [data-testid="stDataFrame"] div[role="columnheader"] {background-color: rgba(255,255,255,0.05) !important; border-bottom: 1px solid rgba(255,255,255,0.1) !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    /* HEADERS ET TEXTES */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,255,0.1)); padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 1px solid rgba(0,255,136,0.2);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    
    /* STATUS BOXES */
    .alert-box {background: rgba(255,200,0,0.1); border-left: 3px solid #ffc800; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .success-box {background: rgba(0,255,136,0.1); border-left: 3px solid #00ff88; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    
    /* SESSION BADGES */
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

GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D", "W": "W"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== UTILITAIRES SESSIONS & CORRELATIONS ====================
def get_active_session(dt: datetime) -> str:
    hour_utc = dt.astimezone(pytz.UTC).hour
    if 0 <= hour_utc < 9: return "Tokyo"
    elif 8 <= hour_utc < 17: return "London"
    elif 13 <= hour_utc < 22: return "NY"
    else: return "Off-Hours"

def get_session_badge(session: str) -> str:
    badges = {
        "London": "<span class='session-badge session-london'>LONDON</span>",
        "NY": "<span class='session-badge session-ny'>NY</span>",
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>",
        "Off-Hours": "<span class='session-badge' style='background:#666;color:white;'>OFF</span>"
    }
    return badges.get(session, "")

@st.cache_data(ttl=3600)
def calculate_pair_correlation(pair1: str, pair2: str, lookback: int = 100) -> float:
    try:
        df1 = get_candles(pair1, "D1", lookback)
        df2 = get_candles(pair2, "D1", lookback)
        if len(df1) < 50 or len(df2) < 50: return 0.0
        
        returns1 = df1['close'].pct_change().dropna()
        returns2 = df2['close'].pct_change().dropna()
        min_len = min(len(returns1), len(returns2))
        
        if min_len < 20: return 0.0
        return round(returns1.iloc[-min_len:].corr(returns2.iloc[-min_len:]), 3)
    except:
        return 0.0

def check_portfolio_correlation(signals: List['Signal'], max_corr: float = 0.7) -> List['Signal']:
    if len(signals) <= 1: return signals
    filtered, pairs_added = [], []
    sorted_signals = sorted(signals, key=lambda x: x.score, reverse=True)
    
    for signal in sorted_signals:
        is_correlated = False
        for existing_pair in pairs_added:
            if abs(calculate_pair_correlation(signal.pair, existing_pair)) > max_corr:
                is_correlated = True
                break
        if not is_correlated:
            filtered.append(signal)
            pairs_added.append(signal.pair)
    return filtered

# ==================== RATE LIMITER & API ====================
@dataclass
class RateLimiter:
    min_interval: float = 0.12
    max_retries: int = 3
    backoff_factor: float = 2.0
    _last_request: float = field(default=0.0, init=False)
    _request_count: int = field(default=0, init=False)
    _error_count: int = field(default=0, init=False)
    
    def wait(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval: time.sleep(self.min_interval - elapsed)
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

@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except Exception as e:
        st.error("⚠️ OANDA Token Error")
        st.stop()

client = get_oanda_client()

def retry_on_failure(max_attempts: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except V20Error:
                    if attempt < max_attempts - 1: time.sleep(rate_limiter.backoff(attempt))
                    else: 
                        rate_limiter.record_error()
                        raise
                except Exception:
                    rate_limiter.record_error()
                    raise
            return None
        return wrapper
    return decorator

@st.cache_data(ttl=30, show_spinner=False)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran: return pd.DataFrame()
    
    rate_limiter.wait()
    try:
        req = InstrumentsCandles(instrument=pair, params={"granularity": gran, "count": count, "price": "M"})
        client.request(req)
        
        data = [{
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "complete": c.get("complete", False)
        } for c in req.response.get("candles", [])]
        
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

# ==================== DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "Institutional"
    PREMIUM = "Premium"
    STANDARD = "Standard"

@dataclass
class TradingParams:
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_adx_threshold: int
    adx_strong_threshold: int
    min_rr_ratio: float
    cascade_required: bool
    strict_flip_only: bool
    min_score_threshold: int
    min_volatility_percentile: float
    max_correlation: float
    session_filter: bool

@dataclass
class RiskConfig:
    max_risk_per_trade: float
    max_portfolio_risk: float

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
    session: str

@dataclass
class ScanStats:
    total_pairs: int = 0
    successful_scans: int = 0
    signals_found: int = 0
    signals_filtered: int = 0
    scan_duration: float = 0.0

# ==================== INDICATORS & LOGIC ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    
    close, high, low = df['close'], df['high'], df['low']

    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    try:
        # HMA
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
        hma_cond = df['hma'] > df['hma'].shift(1)
        df['hma_up'] = hma_cond.where(hma_cond.notna(), False).astype(bool)

        # RSI
        delta = close.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.ewm(alpha=1/14).mean() / down.ewm(alpha=1/14).mean()
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR & ADX
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        df['atr_val'] = tr.ewm(alpha=1/14).mean()
        
        plus_dm, minus_dm = high.diff().clip(lower=0), -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / df['atr_val'])
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / df['atr_val'])
        df['adx'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()

        # UT Bot Logic
        nLoss = 2.0 * tr.rolling(1).mean()
        xATRTrailingStop = [0.0] * len(df)
        for i in range(1, len(df)):
            prev = xATRTrailingStop[i-1]
            curr = close.iloc[i]
            loss = nLoss.iloc[i] if not pd.isna(nLoss.iloc[i]) else 0
            if curr > prev and close.iloc[i-1] > prev: xATRTrailingStop[i] = max(prev, curr - loss)
            elif curr < prev and close.iloc[i-1] < prev: xATRTrailingStop[i] = min(prev, curr + loss)
            elif curr > prev: xATRTrailingStop[i] = curr - loss
            else: xATRTrailingStop[i] = curr + loss
        df['ut_state'] = np.where(close > xATRTrailingStop, 1, -1)
        
        # Percentile
        df['atr_percentile'] = df['atr_val'].rolling(100, min_periods=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
        
    except Exception:
        pass
    
    return df

@st.cache_data(ttl=180, show_spinner=False)
def get_trend_alignment(pair: str, signal_tf: str) -> Tuple[str, float]:
    map_higher = {"M15": "H1", "H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf: return "Neutral", 0.0
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50: return "Neutral", 0.0
    
    df = calculate_indicators(df)
    close = df['close']
    ema50 = close.ewm(span=50).mean().iloc[-1]
    
    strength = 0.0
    trend = "Neutral"
    
    if close.iloc[-1] > ema50 and df['hma'].iloc[-1] > df['hma'].iloc[-2]:
        trend = "Bullish"
        strength = 50 + (30 if df['adx'].iloc[-1] > 25 else 0)
    elif close.iloc[-1] < ema50 and df['hma'].iloc[-1] < df['hma'].iloc[-2]:
        trend = "Bearish"
        strength = 50 + (30 if df['adx'].iloc[-1] > 25 else 0)
        
    return trend, min(100, strength)

class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
    
    def calculate_position_size(self, signal: Signal) -> float:
        risk_per_trade = self.balance * self.config.max_risk_per_trade
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0: return 0.0
        
        # Kelly simplified adjustment
        size = risk_per_trade / pip_risk
        score_adj = signal.score / 100
        return round(size * score_adj, 2)

def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100: return None
        df = calculate_indicators(df)
        
        idx = -1 if mode_live else (-2 if not df.iloc[-1]['complete'] else -1)
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        if pd.isna(last['hma']) or pd.isna(last['atr_percentile']): return None
        if last['atr_percentile'] < params.min_volatility_percentile: return None
        
        hma_flip_green = (last['hma_up'] == True) and (prev['hma_up'] == False)
        hma_flip_red = (last['hma_up'] == False) and (prev['hma_up'] == True)
        
        raw_buy = hma_flip_green and last['rsi'] > 50 and last['ut_state'] == 1
        raw_sell = hma_flip_red and last['rsi'] < 50 and last['ut_state'] == -1
        
        if not (raw_buy or raw_sell): return None
        action = "BUY" if raw_buy else "SELL"
        
        higher_trend, trend_strength = get_trend_alignment(pair, tf)
        if params.cascade_required:
            if (action == "BUY" and higher_trend != "Bullish") or (action == "SELL" and higher_trend != "Bearish"):
                return None
        
        # Scoring
        score = 50
        if last['adx'] > params.adx_strong_threshold: score += 30
        elif last['adx'] > params.min_adx_threshold: score += 15
        if hma_flip_green or hma_flip_red: score += 20
        score += int(trend_strength * 0.25)
        
        score = max(params.min_score_threshold, min(100, score))
        if score < params.min_score_threshold: return None
        
        local_time = pytz.utc.localize(last['time']).astimezone(TUNIS_TZ) if last['time'].tzinfo is None else last['time'].astimezone(TUNIS_TZ)
        session = get_active_session(local_time)
        if params.session_filter and session == "Off-Hours": return None
        
        quality = SignalQuality.INSTITUTIONAL if score >= 85 else (SignalQuality.PREMIUM if score >= 75 else SignalQuality.STANDARD)
        atr = last['atr_val']
        sl = last['close'] - params.atr_sl_multiplier * atr if action == "BUY" else last['close'] + params.atr_sl_multiplier * atr
        tp = last['close'] + params.atr_tp_multiplier * atr if action == "BUY" else last['close'] - params.atr_tp_multiplier * atr
        
        rr = abs(tp - last['close']) / abs(last['close'] - sl) if abs(last['close'] - sl) > 0 else 0
        if rr < params.min_rr_ratio: return None
        
        signal = Signal(
            timestamp=local_time, pair=pair, timeframe=tf, action=action,
            entry_price=last['close'], stop_loss=sl, take_profit=tp,
            score=score, quality=quality, position_size=0.0, risk_amount=0.0,
            risk_reward=rr, adx=int(last['adx']), rsi=int(last['rsi']),
            atr=atr, atr_percentile=last['atr_percentile'], higher_tf_trend=higher_trend,
            is_live=mode_live, is_fresh_flip=True, candle_index=idx, session=session
        )
        
        signal.position_size = risk_manager.calculate_position_size(signal)
        signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
        return signal
        
    except Exception:
        return None

def run_scan(pairs: List[str], tfs: List[str], mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Tuple[List[Signal], ScanStats]:
    start_time = time.time()
    signals = []
    stats = ScanStats(total_pairs=len(pairs) * len(tfs))
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params): (p, tf) for p in pairs for tf in tfs}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res: signals.append(res)
                stats.successful_scans += 1
            except:
                pass
    
    stats.signals_found = len(signals)
    if params.max_correlation < 1.0:
        signals = check_portfolio_correlation(signals, params.max_correlation)
        stats.signals_filtered = stats.signals_found - len(signals)
    
    stats.scan_duration = time.time() - start_time
    return signals, stats

# ==================== PDF GENERATOR (LANDSCAPE & BIGGER) ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    # Utilisation de Landscape (Format Paysage)
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=10*mm, bottomMargin=10*mm, leftMargin=5*mm, rightMargin=5*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    # Titre plus grand
    elements.append(Paragraph("<font size=20 color=#00ff88><b>BlueStar v3.0 Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 10*mm))
    
    # Headers
    data = [["Heure", "Paire", "TF", "Qualité", "Action", "Entry", "SL", "TP", "Score", "R:R", "Size", "Trend", "Session"]]
    
    for s in sorted(signals, key=lambda x: (x.score, x.timestamp), reverse=True):
        # Couleurs conditionnelles pour Action
        act_color = "#00ff88" if s.action == "BUY" else "#ff6b6b"
        action_text = f"<font color={act_color}><b>{s.action}</b></font>"
        
        data.append([
            s.timestamp.strftime("%H:%M"), 
            s.pair.replace("_", "/"), 
            s.timeframe,
            s.quality.value[:4], 
            Paragraph(action_text, styles["Normal"]), # Utilisation de Paragraph pour la couleur
            f"{s.entry_price:.5f}", 
            f"{s.stop_loss:.5f}",
            f"{s.take_profit:.5f}", 
            str(s.score), 
            f"{s.risk_reward:.1f}",
            f"{s.position_size:.2f}", 
            s.higher_tf_trend[:4], 
            s.session[:3]
        ])
    
    # Largeur des colonnes ajustée pour le mode Paysage (~280mm total)
    col_widths = [18*mm, 22*mm, 15*mm, 22*mm, 20*mm, 25*mm, 25*mm, 25*mm, 15*mm, 15*mm, 25*mm, 20*mm, 20*mm]
    
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10), # Header Font plus grand
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.white),
        ('FONTSIZE', (0,1), (-1,-1), 9), # Body Font plus grand (était 6)
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN UI ====================
def main():
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    
    with col_title:
        st.markdown("# BlueStar Institutional")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v30-badge">v3.0</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(TUNIS_TZ)
        market_open = now_tunis.hour in range(0, 23)
        session = get_active_session(now_tunis)
        st.markdown(f"""<div style='text-align: right; padding-top: 10px;'>
            <span style='color: #a0a0c0; font-size: 0.8rem;'>🕐 {now_tunis.strftime('%H:%M:%S')}</span><br>
            <span style='color: {"#00ff88" if market_open else "#ff6666"};'>{"OPEN" if market_open else "CLOSED"}</span> {get_session_badge(session)}
        </div>""", unsafe_allow_html=True)
    
    with col_mode:
        mode = st.radio("Mode", ["Confirmed", "Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode

    with st.expander("⚙️ Configuration", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        atr_sl = c1.number_input("SL Multiplier", 1.0, 4.0, 2.0, 0.5)
        atr_tp = c2.number_input("TP Multiplier", 1.5, 6.0, 3.5, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.5, 0.1)
        cascade_req = c4.checkbox("Cascade obligatoire", False)
        
        c5, c6, c7, c8 = st.columns(4)
        strict_flip = c5.checkbox("Strict Flip", False)
        min_score = c6.number_input("Score Min", 50, 100, 55, 5)
        min_adx = c7.number_input("ADX Min", 15, 35, 20, 1)
        
        c9, c10, c11 = st.columns(3)
        min_vol_pct = c9.slider("Min Volatility %", 0, 70, 15, 5)
        max_corr = c10.slider("Max Correlation", 0.5, 1.0, 0.75, 0.05)
        session_filter = c11.checkbox("Session Filter", False)

    c1, c2, c3, c4 = st.columns(4)
    balance = c1.number_input("Capital ($)", 1000, 1000000, 10000, 1000)
    max_risk = c2.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    scan_btn = c4.button("🚀 SCAN MARKET", type="primary", use_container_width=True)

    # === GESTION DU SESSION STATE POUR LA PERSISTANCE ===
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
    if "scan_stats" not in st.session_state:
        st.session_state.scan_stats = None

    # Si on clique sur le bouton SCAN, on met à jour le state
    if scan_btn:
        with st.spinner("Analyzing Market Structure..."):
            params = TradingParams(
                atr_sl_multiplier=atr_sl, atr_tp_multiplier=atr_tp, min_rr_ratio=min_rr,
                cascade_required=cascade_req, strict_flip_only=strict_flip, min_score_threshold=min_score,
                min_adx_threshold=min_adx, adx_strong_threshold=25, min_volatility_percentile=min_vol_pct,
                max_correlation=max_corr, session_filter=session_filter
            )
            risk_manager = RiskManager(RiskConfig(max_risk, 0.05), balance)
            signals, stats = run_scan(PAIRS_DEFAULT, ["M15", "H1", "H4", "D1"], is_live, risk_manager, params)
            
            # Sauvegarde dans la session
            st.session_state.scan_results = signals
            st.session_state.scan_stats = stats
    
    # === AFFICHAGE DES RÉSULTATS (Depuis le State) ===
    if st.session_state.scan_results is not None:
        signals = st.session_state.scan_results
        stats = st.session_state.scan_stats

        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signaux", len(signals))
        m2.metric("Institutional", len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL]))
        m3.metric("Filtrés", stats.signals_filtered)
        m4.metric("Exposition", f"${sum(s.risk_amount for s in signals):.0f}")
        m5.metric("Temps", f"{stats.scan_duration:.1f}s")
        
        if signals:
            st.markdown("---")
            d1, d2 = st.columns(2)
            with d1:
                df_exp = pd.DataFrame([vars(s) for s in signals])
                st.download_button("📥 CSV", df_exp.astype(str).to_csv(index=False).encode(), f"bluestar_{datetime.now().strftime('%H%M')}.csv", "text/csv")
            with d2:
                # Le clic ici recharge la page, mais réaffiche les données grâce au session_state
                st.download_button("📄 PDF (Landscape)", generate_pdf(signals), f"report_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")

            st.markdown("---")
            cols = st.columns(4)
            for col, tf in zip(cols, ["M15", "H1", "H4", "D1"]):
                with col:
                    tf_sig = sorted([s for s in signals if s.timeframe == tf], key=lambda x: x.score, reverse=True)
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal(s)</p></div>", unsafe_allow_html=True)
                    
                    if tf_sig:
                        df_disp = pd.DataFrame([{
                            "Pair": s.pair.replace("_","/"), "Act": s.action,
                            "Score": s.score, "Entry": s.entry_price, "SL": s.stop_loss,
                            "TP": s.take_profit
                        } for s in tf_sig])
                        st.dataframe(df_disp, use_container_width=True, hide_index=True)
                    else:
                        st.info("No signal")
        else:
            if scan_btn: # Afficher warning seulement si on vient de cliquer
                st.warning("Aucun signal détecté avec les paramètres actuels.")

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.7rem;'>BlueStar Institutional v3.0 | Proprietary Algorithm</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
