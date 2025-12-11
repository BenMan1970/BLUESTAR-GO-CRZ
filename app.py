"""
BlueStar Institutional v6.2 Fusion - Enhanced Performance Engine v3.0
Raw Strength Logic v6.2 + Pro Architecture v3.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import wraps

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
st.set_page_config(page_title="BlueStar Institutional v6.2", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .force-strong { color: #00ff88; font-weight: bold; }
    .force-good { color: #88ff00; font-weight: bold; }
    .force-medium { color: #ffff88; font-weight: bold; }
    .force-weak { color: #ff8888; font-weight: bold; }
    .session-london { background: linear-gradient(90deg, #00ff88, #008844); padding: 4px 8px; border-radius: 12px; color: white; font-weight: bold; }
    .session-ny { background: linear-gradient(90deg, #ffaa00, #aa6600); padding: 4px 8px; border-radius: 12px; color: white; font-weight: bold; }
    .session-tokyo { background: linear-gradient(90deg, #8888ff, #4444aa); padding: 4px 8px; border-radius: 12px; color: white; font-weight: bold; }
    .session-off { background: linear-gradient(90deg, #666666, #333333); padding: 4px 8px; border-radius: 12px; color: #ccc; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

FOREX_28_PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY"
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== RATE LIMITER ====================
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
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()
        self._request_count += 1

    def backoff(self, attempt: int) -> float:
        return self.min_interval * (self.backoff_factor ** attempt)

    def record_error(self) -> None:
        self._error_count += 1

rate_limiter = RateLimiter()

# ==================== API ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except Exception as e:
        st.error("⚠️ OANDA Token Error - Vérifiez st.secrets['OANDA_ACCESS_TOKEN']")
        st.stop()

client = get_oanda_client()

@st.cache_data(ttl=30, show_spinner=False)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran: 
        return pd.DataFrame()
    
    rate_limiter.wait()
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity": gran, "count": count, "price": "M"})
        client.request(r)
        data = [{
            'time': c['time'], 
            'open': float(c['mid']['o']), 
            'high': float(c['mid']['h']), 
            'low': float(c['mid']['l']), 
            'close': float(c['mid']['c'])
        } for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

# ==================== UTILITIES ====================
def get_active_session(dt: datetime) -> str:
    hour_utc = dt.astimezone(pytz.UTC).hour
    if 0 <= hour_utc < 9: return "Tokyo"
    elif 8 <= hour_utc < 17: return "London"
    elif 13 <= hour_utc < 22: return "NY"
    else: return "Off-Hours"

def get_session_badge(session: str) -> str:
    badges = {
        "London": "LONDON",
        "NY": "NY",
        "Tokyo": "TOKYO",
        "Off-Hours": "OFF"
    }
    return badges.get(session, "")

def get_session_class(session: str) -> str:
    classes = {
        "London": "session-london",
        "NY": "session-ny",
        "Tokyo": "session-tokyo",
        "Off-Hours": "session-off"
    }
    return classes.get(session, "")

# ==================== CLASSES ====================
@dataclass
class TradingParams:
    atr_sl: float
    atr_tp: float
    use_fvg: bool
    strict_flip: bool

@dataclass
class Signal:
    timestamp: datetime
    pair: str
    timeframe: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    raw_strength_diff: float
    confluences: List[str]
    session: str

# ==================== RAW STRENGTH OPTIMIZED ====================
@st.cache_data(ttl=60, show_spinner=False)
def calculate_raw_strength() -> Dict[str, float]:
    raw_strengths = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_28_PAIRS:
        df = get_candles(pair, "D", count=2)
        if len(df) < 1: continue
        candle = df.iloc[-1]
        open_p, close_p = candle['open'], candle['close']
        if open_p == 0: continue
        pct = ((close_p - open_p) / open_p) * 100
        base, quote = pair.split("_")
        if base in raw_strengths: raw_strengths[base] += pct
        if quote in raw_strengths: raw_strengths[quote] -= pct
    return {c: round(v, 2) for c, v in raw_strengths.items()}

# ==================== ANALYSIS LOGIC ====================
def analyze_market(pair: str, tf: str, params: TradingParams, raw_data: Dict[str, float]) -> Optional[Signal]:
    df = get_candles(pair, tf, count=250)
    if len(df) < 100: 
        return None

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)
    
    df['hma'] = hma(df['close'], 20)
    h52, l52 = df['high'].rolling(52).max(), df['low'].rolling(52).min()
    df['ssb'] = ((h52 + l52) / 2).shift(26)
    
    fvg_bull = any((df['low'] > df['high'].shift(2)).iloc[-5:])
    fvg_bear = any((df['high'] < df['low'].shift(2)).iloc[-5:])
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    hma_up = curr['hma'] > prev['hma']
    hma_down = curr['hma'] < prev['hma']
    flip_up = hma_up and (prev['hma'] < df.iloc[-3]['hma'])
    flip_down = hma_down and (prev['hma'] > df.iloc[-3]['hma'])
    
    buy_sig = flip_up if params.strict_flip else hma_up
    sell_sig = flip_down if params.strict_flip else hma_down
    
    trend_bull = curr['close'] > curr['ema200']
    trend_bear = curr['close'] < curr['ema200']
    
    action = None
    conf = []
    
    if buy_sig and trend_bull:
        action = "BUY"
        if curr['close'] > curr['ssb']: conf.append("Cloud")
        if fvg_bull: conf.append("FVG")
        conf.append("Trend")
    elif sell_sig and trend_bear:
        action = "SELL"
        if curr['close'] < curr['ssb']: conf.append("Cloud")
        if fvg_bear: conf.append("FVG")
        conf.append("Trend")
    
    if not action: 
        return None
    if params.use_fvg and "FVG" not in conf: 
        return None

    # Raw Strength Diff
    raw_diff = 0.0
    if "_" in pair and "US30" not in pair and "NAS100" not in pair and "XAU" not in pair:
        try:
            base, quote = pair.split("_")
            s_base = raw_data.get(base, 0.0)
            s_quote = raw_data.get(quote, 0.0)
            if action == "BUY":
                raw_diff = s_base - s_quote
            else:
                raw_diff = s_quote - s_base
        except:
            pass
    
    if raw_diff < -1.0 and ("US30" not in pair and "XAU" not in pair):
        return None

    atr = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr * params.atr_sl) if action == "BUY" else curr['close'] + (atr * params.atr_sl)
    tp = curr['close'] + (atr * params.atr_tp) if action == "BUY" else curr['close'] - (atr * params.atr_tp)
    
    local_time = pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ) if curr['time'].tzinfo is None else curr['time'].astimezone(TUNIS_TZ)
    
    return Signal(
        timestamp=local_time,
        pair=pair,
        timeframe=tf,
        action=action,
        entry_price=curr['close'],
        stop_loss=sl,
        take_profit=tp,
        raw_strength_diff=raw_diff,
        confluences=conf,
        session=get_active_session(local_time)
    )

# ==================== SCAN ENGINE ====================
def run_scan_raw_strength(pairs: List[str], tfs: List[str], params: TradingParams) -> List[Signal]:
    raw_data = calculate_raw_strength()
    signals: List[Signal] = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(analyze_market, p, tf, params, raw_data): (p, tf) 
                  for p in pairs for tf in tfs}
        
        for future in as_completed(futures, timeout=60):
            try:
                res = future.result(timeout=12)
                if res:
                    signals.append(res)
            except TimeoutError:
                continue
            except Exception:
                continue
    
    return signals

# ==================== UI HELPERS ====================
def smart_format(pair: str, price: float) -> str:
    if "JPY" in pair:
        return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair:
        return f"{price:.1f}"
    elif "XAU" in pair:
        return f"{price:.2f}"
    else:
        return f"{price:.5f}"

def get_force_class(value: float) -> str:
    if value >= 1.0: return "force-strong"
    elif value >= 0.5: return "force-good"
    elif value >= 0.0: return "force-medium"
    else: return "force-weak"

# ==================== MAIN ====================
def main():
    # Header
    col_title, col_time = st.columns([3, 2])
    with col_title:
        st.markdown("# 🚀 BlueStar Institutional")
        st.markdown('<h2 style="color: #00ff88; font-weight: bold;">INSTITUTIONAL v6.2 Enhanced</h2>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(TUNIS_TZ)
        market_open = now_tunis.hour in range(0, 23)
        session = get_active_session(now_tunis)
        st.markdown(f"""
        <div class="session-{session.lower()}">
            **{get_session_badge(session)} SESSION**<br>
            {now_tunis.strftime('%H:%M Tunis')}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Marché**: {'🟢 OUVERT' if market_open else '🔴 FERMÉ'}")

    # Sidebar Controls
    with st.sidebar:
        st.subheader("⚙️ Paramètres Institutionnels")
        atr_sl = st.slider("ATR Stop Loss (x)", 1.0, 4.0, 2.0, 0.5)
        atr_tp = st.slider("ATR Take Profit (x)", 2.0, 6.0, 3.0, 0.5)
        use_fvg = st.checkbox("✅ Filtre FVG obligatoire", value=True)
        strict_flip = st.checkbox("🔒 Strict HMA Flip only", value=False)
        
        st.markdown("---")
        if st.button("🔍 **SCANNER LE MARCHÉ**", type="primary", use_container_width=True):
            st.cache_data.clear()  # Force refresh
            st.rerun()
        
        st.markdown("---")
        st.info(f"**Scan Targets**: {len(SCAN_TARGETS)} paires\n**Timeframes**: {', '.join(TIMEFRAMES)}")

    params = TradingParams(atr_sl=atr_sl, atr_tp=atr_tp, use_fvg=use_fvg, strict_flip=strict_flip)
    
    # Raw Strength Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 Raw Currency Strength")
        raw_strength = calculate_raw_strength()
        strength_df = pd.DataFrame(list(raw_strength.items()), columns=['Currency', 'Strength'])
        strength_df = strength_df.sort_values('Strength', key=abs, ascending=False)
        st.dataframe(strength_df, use_container_width=True)
    
    # Signals
    tf_sigs: List[Signal] = []
    if st.button("🔍 **SCANNER LE MARCHÉ**", type="primary", use_container_width=True):
        with st.spinner(f"Analyse institutionnelle en cours... ({len(SCAN_TARGETS)*len(TIMEFRAMES)} combinaisons)"):
            tf_sigs = run_scan_raw_strength(SCAN_TARGETS, TIMEFRAMES, params)
            st.success(f"✅ Scan terminé: {len(tf_sigs)} signal(s) institutionnel(s)")

    st.markdown(f"## 📊 {len(tf_sigs)} Signal(s) Institutionnel(s)")
    
    if not tf_sigs:
        st.info("🔍 Aucun signal institutionnel valide pour le moment.\n💡 Essayez d'ajuster les paramètres ou attendez la prochaine session.")
        st.stop()

    # Signals Table
    data = []
    for s in sorted(tf_sigs, key=lambda x: x.raw_strength_diff, reverse=True):
        data.append({
            "Heure": s.timestamp.strftime("%H:%M"),
            "Paire": s.pair.replace("_", "/"),
            "TF": s.timeframe,
            "Action": f"🟢 {s.action}" if s.action == "BUY" else f"🔴 {s.action}",
            "Entry": smart_format(s.pair, s.entry_price),
            "SL": smart_format(s.pair, s.stop_loss),
            "TP": smart_format(s.pair, s.take_profit),
            "Force": f'<span class="{get_force_class(s.raw_strength_diff)}">{s.raw_strength_diff:.2f}</span>',
            "Confluences": ", ".join(s.confluences),
            "Session": f'<span class="session-{s.session.lower()}">{get_session_badge(s.session)}</span>'
        })
    
    df_view = pd.DataFrame(data)
    st.markdown(df_view.to_html(escape=False), unsafe_allow_html=True)

    # PDF Export
    if st.button("📄 Exporter PDF", use_container_width=True):
        # Simple PDF generation (you can enhance this)
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
        # Add your PDF logic here...
        st.download_button("Télécharger Rapport", buffer.getvalue(), "bluestar-signals.pdf", "application/pdf")

if __name__ == "__main__":
    main()
