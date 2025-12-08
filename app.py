"""
BlueStar Cascade - VERSION 3.0 INSTITUTIONAL GRADE ENHANCED
+ M15 ajouté | Visuel ancien restauré | Aucun quadrillage | Bouton "SCAN"
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
import time
import hashlib
from functools import wraps

# OANDA
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

# PDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar Institutional v3.0", layout="wide", initial_sidebar_state="collapsed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem;}
    .v30-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; margin-left: 8px;}
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2));
        padding: 12px; border-radius: 10px; text-align: center; margin: 15px 0 10px 0;
        border: 2px solid rgba(0,255,136,0.4);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.4rem;}
    .tf-header p {margin: 5px 0 0 0; color: #a0a0c0; font-size: 0.9rem;}
    .stDataFrame {border: none !important;}
    .stDataFrame table {border: none !important;}
    .stDataFrame td, .stDataFrame th {border: none !important; padding: 6px 8px !important;}
    .stDataFrame thead {display: none;}
    .session-badge {padding: 3px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold;}
    .session-london {background: #ff6b6b; color: white;}
    .session-ny {background: #4ecdc4; color: white;}
    .session-tokyo {background: #ffe66d; color: black;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
                 "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
                 "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
                 "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
                 "EUR_CHF","GBP_CHF","XAU_USD","XPT_USD"]

GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== SESSIONS ====================
def get_active_session(dt: datetime) -> str:
    h = dt.astimezone(pytz.UTC).hour
    if 0 <= h < 9: return "Tokyo"
    elif 8 <= h < 17: return "London"
    elif 13 <= h < 22: return "NY"
    else: return "Off-Hours"

def get_session_badge(s: str) -> str:
    return {
        "London": "<span class='session-badge session-london'>LONDON</span>",
        "NY": "<span class='session-badge session-ny'>NY</span>",
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>",
    }.get(s, "<span class='session-badge' style='background:#666;color:white;'>OFF</span>")

# ==================== DATACLASSES & PARAMS ====================
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
    adx: int
    rsi: int
    atr_percentile: float
    higher_tf_trend: str
    session: str

# ==================== OANDA CLIENT ====================
@st.cache_resource
def get_oanda_client():
    token = st.secrets["OANDA_ACCESS_TOKEN"]
    return API(access_token=token)

client = get_oanda_client()

# ==================== RATE LIMITER & CACHING ====================
@dataclass
class RateLimiter:
    min_interval: float = 0.12
    _last: float = field(default_factory=time.time)
    def wait(self):
        sleep = self.min_interval - (time.time() - self._last)
        if sleep > 0: time.sleep(sleep)
        self._last = time.time()

rate_limiter = RateLimiter()

def get_cache_key(pair, tf, count): return hashlib.md5(f"{pair}_{tf}_{count}".encode()).hexdigest()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_candles(_key, pair, tf, count): return fetch_candles(pair, tf, count)

def fetch_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    rate_limiter.wait()
    params = {"granularity": GRANULARITY_MAP[tf], "count": count, "price": "M"}
    req = InstrumentsCandles(instrument=pair, params=params)
    client.request(req)
    data = []
    for c in req.response.get("candles", []):
        data.append({
            "time": pd.to_datetime(c["time"]).tz_localize(None),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "complete": c.get("complete", False)
        })
    return pd.DataFrame(data)

def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    key = get_cache_key(pair, tf, count)
    return _cached_candles(key, pair, tf, count)

# ==================== INDICATORS (FIXED) ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    c, h, l = df['close'], df['high'], df['low']

    # HMA
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
    half = wma(c, 10)
    full = wma(c, 20)
    raw = 2*half - full
    df['hma'] = wma(raw, int(np.sqrt(20)))
    df['hma_up'] = (df['hma'] > df['hma'].shift(1)).fillna(False).astype(bool)

    # RSI
    delta = c.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14).mean()
    roll_down = down.ewm(alpha=1/14).mean()
    df['rsi'] = 100 - (100 / (1 + roll_up/roll_down))

    # ATR + ADX + UT Bot
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14).mean()
    df['atr_val'] = atr
    df['atr_percentile'] = atr.rolling(100, min_periods=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=True)

    # UT Bot
    nLoss = 2 * atr
    stop = [0.0] * len(df)
    for i in range(1, len(df)):
        prev, curr = stop[i-1], c.iloc[i]
        loss = nLoss.iloc[i]
        if curr > prev and c.iloc[i-1] > prev:
            stop[i] = max(prev, curr - loss)
        elif curr < prev and c.iloc[i-1] < prev:
            stop[i] = min(prev, curr + loss)
        elif curr > prev: stop[i] = curr - loss
        else: stop[i] = curr + loss
    df['ut_state'] = np.where(c > stop, 1, -1)

    # ADX
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ==================== CASCADE & ANALYSIS ====================
@st.cache_data(ttl=180)
def get_trend_alignment(pair: str, tf: str) -> Tuple[str, float]:
    higher = {"M15": "H1", "H1": "H4", "H4": "D1"}.get(tf, "D1")
    df = get_candles(pair, higher, 100)
    if len(df) < 50: return "Neutral", 0
    df = calculate_indicators(df)
    close = df['close'].iloc[-1]
    hma_up = df['hma'].iloc[-1] > df['hma'].iloc[-2]
    adx = df['adx'].iloc[-1]
    strength = 50
    if close > df['close'].ewm(span=50).mean().iloc[-1] and hma_up:
        strength += 30 + (30 if adx > 25 else 0)
        return "Bullish", min(100, strength)
    elif close < df['close'].ewm(span=50).mean().iloc[-1] and not hma_up:
        strength += 30 + (30 if adx > 25 else 0)
        return "Bearish", min(100, strength)
    return "Neutral", 30

class RiskManager:
    def __init__(self, cfg: RiskConfig, balance: float):
        self.cfg = cfg
        self.balance = balance
    def calc_size(self, signal: Signal) -> float:
        kelly = max(0, (0.6 * signal.risk_reward - 0.4) / signal.risk_reward) * self.cfg.kelly_fraction * (signal.score/100)
        size = (self.balance * kelly) / abs(signal.entry_price - signal.stop_loss)
        max_risk = self.balance * self.cfg.max_risk_per_trade
        if size * abs(signal.entry_price - signal.stop_loss) > max_risk:
            size = max_risk / abs(signal.entry_price - signal.stop_loss)
        return round(size, 2)

def analyze_pair(pair: str, tf: str, live: bool, rm: RiskManager, p: TradingParams) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100: return None
        df = calculate_indicators(df)
        idx = -1 if live and df.iloc[-1]['complete'] else -2
        last = df.iloc[idx]; prev = df.iloc[idx-1]

        if any(pd.isna(last[f]) for f in ['hma_up','rsi','adx','atr_val','ut_state','atr_percentile']): return None
        if last['atr_percentile'] < p.min_volatility_percentile: return None

        flip_up = last['hma_up'] and not prev['hma_up']
        flip_down = not last['hma_up'] and prev['hma_up']
        buy = (flip_up or (idx > -3 and df.iloc[idx-2:idx].hma_up.all() and not df.iloc[idx-3]['hma_up'])) and last['rsi'] > 50 and last['ut_state'] == 1
        sell = (flip_down or (idx > -3 and (~df.iloc[idx-2:idx].hma_up).all() and df.iloc[idx-3]['hma_up'])) and last['rsi'] < 50 and last['ut_state'] == -1

        if not (buy or sell): return None
        action = "BUY" if buy else "SELL"

        trend, _ = get_trend_alignment(pair, tf)
        if p.cascade_required and ((action=="BUY" and trend!="Bullish") or (action=="SELL" and trend!="Bearish")): return None

        score = 50
        score += 30 if last['adx'] > p.adx_strong_threshold else 15 if last['adx'] > p.min_adx_threshold else 0
        score += 20 if flip_up or flip_down else 10
        score += 15 if (50 < last['rsi'] < 60 and buy) or (40 < last['rsi'] < 50 and sell) else 8
        score += 10 if last['atr_percentile'] > 50 else 5 if last['atr_percentile'] > 30 else 0
        if score < p.min_score_threshold: return None

        atr = last['atr_val']
        sl = last['close'] - p.atr_sl_multiplier * atr if action == "BUY" else last['close'] + p.atr_sl_multiplier * atr
        tp = last['close'] + p.atr_tp_multiplier * atr if action == "BUY" else last['close'] - p.atr_tp_multiplier * atr
        rr = abs(tp - last['close']) / abs(sl - last['close'])
        if rr < p.min_rr_ratio: return None

        t = last['time'].astimezone(TUNIS_TZ) if last['time'].tzinfo else pytz.utc.localize(last['time']).astimezone(TUNIS_TZ)
        session = get_active_session(t)
        if p.session_filter and session == "Off-Hours": return None

        quality = SignalQuality.INSTITUTIONAL if score >= 85 else SignalQuality.PREMIUM if score >= 75 else SignalQuality.STANDARD

        sig = Signal(t, pair, tf, action, last['close'], sl, tp, score, quality, 0, 0, rr,
                     int(last['adx']), int(last['rsi']), last['atr_percentile'], trend, session)
        sig.position_size = rm.calc_size(sig)
        sig.risk_amount = abs(sig.entry_price - sig.stop_loss) * sig.position_size
        return sig
    except: return None

# ==================== SCAN ====================
def run_scan(live: bool, params: TradingParams, balance: float):
    rm = RiskManager(RiskConfig(), balance)
    signals = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(analyze_pair, pair, tf, live, rm, params) 
                  for pair in PAIRS_DEFAULT for tf in ["M15", "H1", "H4", "D1"]]
        for f in as_completed(futures, timeout=90):
            r = f.result()
            if r: signals.append(r)
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== MAIN ====================
def main():
    col1, col2, col3 = st.columns([3,2,2])
    with col1:
        st.markdown("# BlueStar Enhanced v3.0")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v30-badge">v3.0</span>', unsafe_allow_html=True)
    with col2:
        now = datetime.now(TUNIS_TZ)
        st.markdown(f"**Tunis** {now.strftime('%H:%M:%S')} {get_session_badge(get_active_session(now))}", unsafe_allow_html=True)
    with col3:
        mode = st.radio("Mode", ["Confirmed", "Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode

    with st.expander("Configuration", expanded=False):
        c1,c2,c3,c4 = st.columns(4)
        sl = c1.number_input("SL × ATR", 1.0, 4.0, 2.0, 0.5)
        tp = c2.number_input("TP × ATR", 2.0, 6.0, 3.5, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.5, 0.1)
        cascade = c4.checkbox("Cascade obligatoire")

        c5,c6,c7,c8 = st.columns(4)
        strict = c5.checkbox("Flip strict uniquement")
        score_min = c6.number_input("Score min", 50, 100, 55, 5)
        vol_min = c7.slider("Volatilité %ile min", 0, 70, 15, 5)
        corr_max = c8.slider("Corrélation max", 0.5, 1.0, 0.75, 0.05)

    bal_col, risk_col, _, btn_col = st.columns([2,2,3,2])
    balance = bal_col.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    risk_col.slider("Risk/Trade (%)", 0.1, 3.0, 1.0, 0.1)
    scan = btn_col.button("SCAN", type="primary", use_container_width=True)

    if scan:
        params = TradingParams(
            atr_sl_multiplier=sl, atr_tp_multiplier=tp, min_rr_ratio=min_rr,
            cascade_required=cascade, strict_flip_only=strict,
            min_score_threshold=score_min, min_volatility_percentile=vol_min,
            max_correlation=corr_max
        )
        with st.spinner("Scan M15 • H1 • H4 • D1 en cours..."):
            signals = run_scan(is_live, params, balance)

        if signals:
            for tf in ["M15", "H1", "H4", "D1"]:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sigs)} signal{'s' if len(tf_sigs)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df_show = pd.DataFrame([{
                        "Heure": s.timestamp.strftime("%H:%M"),
                        "Paire": s.pair.replace("_","/"),
                        "Action": f"{'BUY' if s.action=='BUY' else 'SELL'}",
                        "Entry": f"{s.entry_price:.5f}",
                        "SL": f"{s.stop_loss:.5f}",
                        "TP": f"{s.take_profit:.5f}",
                        "Score": s.score,
                        "R:R": f"{s.risk_reward:.1f}",
                        "Taille": f"{s.position_size:.2f}",
                        "Risk $": f"{s.risk_amount:.0f}",
                        "Quality": s.quality.value[:5],
                        "Trend HTF": s.higher_tf_trend,
                        "Session": s.session
                    } for s in sorted(tf_sigs, key=lambda x: x.score, reverse=True)])
                    st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.warning("Aucun signal trouvé avec ces paramètres.")

if __name__ == "__main__":
    main()
