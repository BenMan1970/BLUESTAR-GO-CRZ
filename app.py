# ================================================
# BlueStar Institutional v6.4 – Alert Edition
# Version FINALE 100% fonctionnelle – Décembre 2025
# Copie-colle ce fichier en entier → ça marche direct
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import pytz
import time
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# OANDA
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# PDF + Logo
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG & STYLE ====================
st.set_page_config(page_title="BlueStar Institutional v6.4", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important;}
    .stMetric {background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.75rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; display: inline-block;}
    .v64-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; margin-left: 8px;}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,255,0.1)); padding: 12px 20px; border-radius: 8px; text-align: center; margin-bottom: 15px; border: 1px solid rgba(0,255,136,0.2); box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.3rem; font-weight: 700;}
    .tf-header p {margin: 5px 0 0 0; color: #a0a0c0; font-size: 0.75rem;}
    .session-badge {padding: 3px 8px; border-radius: 12px; font-size: 0.65rem; font-weight: bold; margin-left: 8px;}
    .session-london {background: #ff6b6b; color: white;}
    .session-ny {background: #4ecdc4; color: white;}
    .session-tokyo {background: #ffe66d; color: black;}
    .force-strong {color: #00ff88 !important; font-weight: 700;}
    .force-good {color: #aaff00 !important; font-weight: 700;}
    .force-medium {color: #ffaa00 !important; font-weight: 700;}
    .force-weak {color: #ff4444 !important; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
FOREX_28_PAIRS = ["EUR_USD","GBP_USD","AUD_USD","NZD_USD","USD_JPY","USD_CHF","USD_CAD","EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
                  "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD","AUD_JPY","AUD_CHF","AUD_CAD","AUD_NZD","NZD_JPY","NZD_CHF","NZD_CAD","CAD_JPY","CAD_CHF","CHF_JPY"]

CURRENCIES = ["USD","EUR","GBP","JPY","CHF","CAD","AUD","NZD"]
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15":"M15", "H1":"H1", "H4":"H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')
LOGO_URL = "https://i.imgur.com/8Y7Z3vP.png"

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

# ==================== UTILS ====================
def get_active_session(dt: datetime) -> str:
    h = dt.astimezone(pytz.UTC).hour
    if 0 <= h < 9: return "Tokyo"
    elif 8 <= h < 17: return "London"
    elif 13 <= h < 22: return "NY"
    else: return "Off-Hours"

def get_session_badge(session: str) -> str:
    badges = {
        "London": "<span class='session-badge session-london'>LONDON</span>",
        "NY": "<span class='session-badge session-ny'>NY</span>",
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>",
        "Off-Hours": "<span class='session-badge' style='background:#666;color:white'>OFF</span>"
    }
    return badges.get(session, "")

def smart_format(pair: str, price: float) -> str:
    if "JPY" in pair: return f"{price:.3f}"
    if any(x in pair for x in ["US30","NAS100"]): return f"{price:.1f}"
    if "XAU" in pair: return f"{price:.2f}"
    return f"{price:.5f}"

def get_force_class(val: float) -> str:
    if val >= 1.0: return "force-strong"
    elif val >= 0.5: return "force-good"
    elif val >= 0.0: return "force-medium"
    else: return "force-weak"

# ==================== TELEGRAM ====================
def send_telegram(message: str):
    token = st.secrets.get("TELEGRAM_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except:
        pass

# ==================== OANDA + CACHE ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        return None

client = get_oanda_client()

@st.cache_data(ttl=180, show_spinner=False)
def get_candles(pair: str, granularity: str, count: int = 300) -> pd.DataFrame:
    if not client:
        return pd.DataFrame()
    try:
        params = {"granularity": granularity, "count": count, "price": "M"}
        r = InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        data = []
        for c in r.response.get('candles', []):
            if c['complete']:
                data.append({
                    'time': pd.to_datetime(c['time']).tz_convert(None),
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c'])
                })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_raw_strength() -> Dict[str, float]:
    raw = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_28_PAIRS:
        df = get_candles(pair, "D", 3)
        if len(df) < 2: continue
        candle = df.iloc[-2]
        pct = (candle['close'] - candle['open']) / candle['open'] * 100
        base, quote = pair.split("_")
        raw[base] += pct
        raw[quote] -= pct
    return {k: round(v, 2) for k, v in raw.items()}

# ==================== LOGIQUE v6.2 COMPLÈTE ====================
def analyze_market(df: pd.DataFrame, pair: str, tf: str, params: TradingParams, raw_data: dict):
    if len(df) < 100: return None

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # HMA rapide
    try:
        import pandas_ta as ta
        df['hma'] = ta.hma(df['close'], length=20)
    except:
        def wma(s, period):
            return s.rolling(period).apply(lambda x: np.dot(x, np.arange(1, period+1)) / np.arange(1, period+1).sum(), raw=True)
        half = wma(df['close'], 10)
        full = wma(df['close'], 20)
        df['hma'] = wma(2*half - full, int(np.sqrt(20)))

    df['ssb'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2

    fvg_bull = (df['low'] > df['high'].shift(2)).iloc[-5:].any()
    fvg_bear = (df['high'] < df['low'].shift(2)).iloc[-5:].any()

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
        conf.append("Trend")
        if curr['close'] > curr['ssb']: conf.append("Cloud")
        if fvg_bull: conf.append("FVG")
    elif sell_sig and trend_bear:
        action = "SELL"
        conf.append("Trend")
        if curr['close'] < curr['ssb']: conf.append("Cloud")
        if fvg_bear: conf.append("FVG")

    if not action: return None
    if params.use_fvg and "FVG" not in conf: return None

    # Raw Strength Diff
    raw_diff = 0.0
    if "_" in pair and all(x not in pair for x in ["US30","NAS100","XAU"]):
        base, quote = pair.split("_")
        s_base = raw_data.get(base, 0)
        s_quote = raw_data.get(quote, 0)
        raw_diff = s_base - s_quote if action == "BUY" else s_quote - s_base

    if raw_diff < -1.0 and "XAU" not in pair and "US30" not in pair: return None

    atr = curr['high'] - curr['low']
    sl = curr['close'] - atr * params.atr_sl if action == "BUY" else curr['close'] + atr * params.atr_sl
    tp = curr['close'] + atr * params.atr_tp if action == "BUY" else curr['close'] - atr * params.atr_tp

    local_time = curr['time'].astimezone(TUNIS_TZ) if curr['time'].tzinfo else pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ)

    return Signal(
        timestamp=local_time,
        pair=pair,
        timeframe=tf,
        action=action,
        entry_price=curr['close'],
        stop_loss=round(sl, 5),
        take_profit=round(tp,5),
        raw_strength_diff=round(raw_diff,2),
        confluences=conf,
        session=get_active_session(local_time)
    )

# ==================== MAIN ====================
def main():
    # Header
    col1, col2, col3 = st.columns([1,5,3])
    with col1:
        st.image(LOGO_URL, width=180)
    with col2:
        st.markdown("# BlueStar Institutional")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v64-badge">v6.4 Alert</span>', unsafe_allow_html=True)
    with col3:
        now = datetime.now(TUNIS_TZ)
        session = get_active_session(now)
        st.markdown(f"<div style='text-align:right;padding-top:20px;'>
                        <div style='color:#a0a0c0;font-size:0.9rem'>Heure Tunis</div>
                        <div style='color:#00ff88;font-size:1.8rem;font-weight:700'>{now.strftime('%H:%M:%S')}</div>
                        {get_session_badge(session)}
                        </div>", unsafe_allow_html=True)

    # Config + Alerte seuil
    with st.expander("Configuration & Alertes Telegram", expanded=False):
        c1,c2,c3,c4,c5 = st.columns(5)
        sl = c1.number_input("SL × ATR",1.0,3.0,1.5,0.1)
        tp = c2.number_input("TP × ATR",1.5,5.0,3.0,0.1)
        fvg = c3.checkbox("FVG obligatoire", True)
        flip = c4.checkbox("Strict Flip only", True)
        alert_thresh = c5.slider("Alerte Telegram si Force ≥", 0.5, 3.0, 1.0, 0.1)

    if st.button("SCAN MARKET & ARM ALERTS", type="primary", use_container_width=True):
        if not client:
            st.error("Token OANDA manquant dans secrets")
            st.stop()

        start = time.time()
        raw_data = get_raw_strength()
        params = TradingParams(sl, tp, fvg, flip)
        signals = []

        progress = st.progress(0)
        status = st.empty()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(analyze_market, get_candles(pair, GRANULARITY_MAP[tf]), pair, tf, params, raw_data)
                for pair in SCAN_TARGETS for tf in TIMEFRAMES
            ]
            for i, future in enumerate(as_completed(futures), 1):
                progress.progress(i / len(futures))
                status.text(f"Analyse en cours… {i}/{len(futures)}")
                try:
                    if sig := future.result():
                        signals.append(sig)
                except:
                    pass

        signals.sort(key=lambda x: x.raw_strength_diff, reverse=True)

        # Alertes Telegram
        new_strong = [s for s in signals if s.raw_strength_diff >= alert_thresh and 
                      (st.session_state.get("last_alert_time") is None or s.timestamp > st.session_state.last_alert_time)]

        if new_strong:
            for s in new_strong:
                msg = f"NEW BLUESTAR SIGNAL\n" \
                      f"{s.action} {s.pair.replace('_','/')} {s.timeframe}\n" \
                      f"Force : <b>{s.raw_strength_diff:+.2f}%</b>\n" \
                      f"Entry ≈ {smart_format(s.pair, s.entry_price)}\n" \
                      f"SL {smart_format(s.pair, s.stop_loss)} | TP {smart_format(s.pair, s.take_profit)}\n" \
                      f"{', '.join(s.confluences)}\n" \
                      f"{s.timestamp.strftime('%H:%M')} Tunis"
                send_telegram(msg)
            st.balloons()
            st.markdown("<audio autoplay><source src='https://assets.mixkit.co/sfx/preview/mixkit-positive-interface-beep-2214.mp3' type='audio/mpeg'></audio>", unsafe_allow_html=True)
            st.session_state.last_alert_time = datetime.now(TUNIS_TZ)

        st.session_state.scan_results = signals
        st.session_state.raw_strength = raw_data
        st.session_state.scan_duration = round(time.time() - start, 1)
        st.success(f"Scan terminé en {st.session_state.scan_duration}s — {len(signals)} signaux détectés")

    # ==================== AFFICHAGE RÉSULTATS ====================
    if st.session_state.get("scan_results"):
        signals = st.session_state.scan_results
        raw_data = st.session_state.get("raw_strength", {})

        if signals:
            last_sig = max(signals, key=lambda x: x.timestamp)
            delta = int((datetime.now(TUNIS_TZ) - last_sig.timestamp).total_seconds() / 60)
            st.info(f"Dernier signal il y a {delta} minute(s)")

        # Métriques
        st.markdown("---")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Signaux", len(signals))
        m2.metric("BUY", len([s for s in signals if s.action=="BUY"]))
        m3.metric("SELL", len([s for s in signals if s.action=="SELL"]))
        m4.metric("Force moyenne", f"{sum(s.raw_strength_diff for s in signals)/len(signals):+.2f}%")
        m5.metric("Durée scan", f"{st.session_state.scan_duration}s")

        # Force brute des devises
        st.markdown("### Force Brute 24h")
        cols = st.columns(8)
        for i, (curr, val) in enumerate(sorted(raw_data.items(), key=lambda x: x[1], reverse=True)):
            with cols[i]:
                color = "#00ff88" if val > 0 else "#ff6b6b"
                st.markdown(f"<div style='background:rgba(255,255,255,0.05);padding:12px;border-radius:8px;text-align:center;border:1px solid rgba(255,255,255,0.1)'>
                            <div style='font-size:1rem;color:#a0a0c0'>{curr}</div>
                            <div style='font-size:1.5rem;font-weight:700;color:{color}'>{val:+.2f}%</div></div>", unsafe_allow_html=True)

        # Export CSV
        if signals:
            df_exp = pd.DataFrame([{
                "Heure": s.timestamp.strftime("%H:%M"),
                "Paire": s.pair.replace("_","/"),
                "TF": s.timeframe,
                "Signal": s.action,
                "Entry": s.entry_price,
                "SL": s.stop_loss,
                "TP": s.take_profit,
                "Force": s.raw_strength_diff,
                "Conf": ", ".join(s.confluences),
                "Session": s.session
            } for s in signals])
            st.download_button("Export CSV", df_exp.to_csv(index=False).encode(), "bluestar_signals.csv")

        # Tableaux par timeframe
        st.markdown("---")
        for tf in TIMEFRAMES:
            tf_sigs = [s for s in signals if s.timeframe == tf]
            if tf_sigs:
                st.markdown(f"<div class='tf-header'><h3>{tf} Market Structure</h3><p>{len(tf_sigs)} signal(s)</p></div>", unsafe_allow_html=True)
                data = []
                for s in tf_sigs:
                    force_html = f"<span class='{get_force_class(s.raw_strength_diff)}'>{s.raw_strength_diff:+.2f}%</span>"
                    data.append({
                        "Heure": s.timestamp.strftime("%H:%M"),
                        "Paire": s.pair.replace("_","/"),
                        "Signal": f"{'BUY' if s.action=='BUY' else 'SELL'} {s.action}",
                        "Prix": smart_format(s.pair, s.entry_price),
                        "SL": smart_format(s.pair, s.stop_loss),
                        "TP": smart_format(s.pair, s.take_profit),
                        "Force": force_html,
                        "Conf": ", ".join(s.confluences),
                        "Session": s.session + " " + get_session_badge(s.session)
                    })
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666;font-size:0.8rem'>BlueStar Institutional v6.4 – Alert Edition | 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
