"""
BlueStar Cascade v3.0 INSTITUTIONAL - VERSION FINALE (comme tu la veux)
→ M15 ajouté
→ Bouton SCAN
→ Seulement signaux INSTITUTIONAL
→ PDF + CSV
→ Aucun quadrillage
→ Configuration complète conservée
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from functools import wraps

from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar Institutional v3.0", layout="wide", initial_sidebar_state="collapsed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem;}
    .v30-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; margin-left: 8px;}
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2));
        padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px;
        border: 2px solid rgba(0,255,136,0.3);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    .stDataFrame {font-size: 0.75rem !important;}
    .stDataFrame table, .stDataFrame td, .stDataFrame th {border: none !important;}
    thead tr th:first-child, tbody th {display: none;}
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

# M15 ajouté ici
TFS = ["M15", "H1", "H4", "D1"]
GRANULARITY_MAP = {"M15":"M15", "H1":"H1", "H4":"H4", "D1":"D"}
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
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>"
    }.get(s, "<span class='session-badge' style='background:#666;color:white;'>OFF</span>")

# ==================== OANDA + CACHE ====================
@st.cache_resource
def get_client(): return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
client = get_client()

@st.cache_data(ttl=45, show_spinner=False)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    try:
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
    except: return pd.DataFrame()

# ==================== INDICATORS (FIXED) ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 100: return df
    c, h, l = df['close'], df['high'], df['low']

    # HMA
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
    df['hma'] = wma(2*wma(c,10) - wma(c,20), int(np.sqrt(20)))
    df['hma_up'] = (df['hma'] > df['hma'].shift(1)).fillna(False).astype(bool)

    # RSI
    delta = c.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.ewm(alpha=1/14).mean() / down.ewm(alpha=1/14).mean())

    # ATR + Percentile
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14).mean()
    df['atr_pct'] = df['atr'].rolling(100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=True
    )

    # UT Bot
    loss = 2 * df['atr']
    stop = [0.0] * len(df)
    for i in range(1, len(df)):
        prev, cur = stop[i-1], c.iloc[i]
        if cur > prev and c.iloc[i-1] > prev:
            stop[i] = max(prev, cur - loss.iloc[i])
        elif cur < prev and c.iloc[i-1] < prev:
            stop[i] = min(prev, cur + loss.iloc[i])
        elif cur > prev: stop[i] = cur - loss.iloc[i]
        else: stop[i] = cur + loss.iloc[i]
    df['ut'] = np.where(c > stop, 1, -1)

    # ADX
    plus_dm = h.diff().clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
    atr14 = df['atr']
    plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / atr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=300)
def higher_trend(pair: str, tf: str) -> str:
    higher = {"M15":"H1", "H1":"H4", "H4":"D1"}.get(tf, "D1")
    df = get_candles(pair, higher, 100)
    if len(df) < 50: return "Neutral"
    df = calculate_indicators(df)
    close = df['close'].iloc[-1]
    hma_up = df['hma'].iloc[-1] > df['hma'].iloc[-2]
    adx = df['adx'].iloc[-1]
    if close > df['close'].ewm(span=50).mean().iloc[-1] and hma_up and adx > 25:
        return "Bullish"
    if close < df['close'].ewm(span=50).mean().iloc[-1] and not hma_up and adx > 25:
        return "Bearish"
    return "Neutral"

# ==================== ANALYSE (INSTITUTIONAL ONLY) ====================
@st.fragment
def analyze_pair(pair: str, tf: str, params) -> Optional[dict]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100: return None
        df = calculate_indicators(df)
        last = df.iloc[-2]
        prev = df.iloc[-3]

        if last['atr_pct'] < params['vol']: return None
        if last['adx'] < 25: return None

        flip_up = last['hma_up'] and not prev['hma_up']
        flip_down = not last['hma_up'] and prev['hma_up']

        if params['strict'] and not (flip_up or flip_down): return None

        if flip_up and last['rsi'] > 55 and last['ut'] == 1:
            action = "BUY"
        elif flip_down and last['rsi'] < 45 and last['ut'] == -1:
            action = "SELL"
        else:
            return None

        trend = higher_trend(pair, tf)
        if params['cascade'] and ((action == "BUY" and trend != "Bullish") or (action == "SELL" and trend != "Bearish")):
            return None

        atr = last['atr']
        sl = round(last['close'] - params['sl'] * atr, 5) if action == "BUY" else round(last['close'] + params['sl'] * atr, 5)
        tp = round(last['close'] + params['tp'] * atr, 5) if action == "BUY" else round(last['close'] - params['tp'] * atr, 5)
        rr = round(abs(tp - last['close']) / abs(sl - last['close']), 1)

        t = last['time'].astimezone(TUNIS_TZ) if last['time'].tzinfo else pytz.utc.localize(last['time']).astimezone(TUNIS_TZ)
        session = get_active_session(t)

        return {
            "pair": pair.replace("_", "/"),
            "tf": tf,
            "action": action,
            "entry": round(last['close'], 5),
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "trend": trend,
            "session": session,
            "time": t.strftime("%H:%M")
        }
    except: return None

# ==================== SCAN ====================
def run_scan(params):
    signals = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(analyze_pair, p, tf, params) for p in PAIRS_DEFAULT for tf in TFS]
        for f in as_completed(futures):
            r = f.result()
            if r: signals.append(r)
    return signals

# ==================== PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, leftMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<font size=18 color=#00ff88><b>BlueStar Institutional Signals</b></font>", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12*mm))

    data = [["Paire", "TF", "Action", "Entry", "SL", "TP", "R:R", "Trend", "Session", "Heure"]]
    for s in signals:
        data.append([s["pair"], s["tf"], s["action"], s["entry"], s["sl"], s["tp"], s["rr"], s["trend"], s["session"], s["time"]])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN ====================
def main():
    col1, col2 = st.columns([3, 4])
    with col1:
        st.markdown("# BlueStar Institutional v3.0")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v30-badge">v3.0</span>', unsafe_allow_html=True)
    with col2:
        now = datetime.now(TUNIS_TZ)
        session = get_active_session(now)
        st.markdown(f"**{now.strftime('%H:%M:%S')} Tunis** {get_session_badge(session)}", unsafe_allow_html=True)

    with st.expander("Configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        sl = c1.slider("SL × ATR", 1.0, 3.5, 2.0, 0.1)
        tp = c2.slider("TP × ATR", 2.5, 5.5, 3.5, 0.1)
        cascade = c3.checkbox("Cascade obligatoire", True)
        strict = c4.checkbox("Flip strict", True)
        vol = st.slider("Volatilité min (%ile)", 10, 50, 20, 5)

    if st.button("SCAN", type="primary", use_container_width=True):
        params = {"sl": sl, "tp": tp, "cascade": cascade, "strict": strict, "vol": vol}
        with st.spinner("Scan institutionnel M15 • H1 • H4 • D1..."):
            signals = run_scan(params)

        if signals:
            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                csv = pd.DataFrame(signals).to_csv(index=False).encode()
                st.download_button("CSV", csv, "bluestar_institutional.csv", "text/csv")
            with col_dl2:
                pdf = generate_pdf(signals)
                st.download_button("PDF", pdf, "BlueStar_Institutional.pdf", "application/pdf")

            for tf in TFS:
                tf_sig = [s for s in signals if s["tf"] == tf]
                if tf_sig:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df_show = pd.DataFrame(tf_sig)[["pair", "action", "entry", "sl", "tp", "rr", "trend", "session", "time"]]
                    df_show.columns = ["Paire", "Action", "Entry", "SL", "TP", "R:R", "Trend", "Session", "Heure"]
                    st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun signal institutionnel détecté pour le moment.")

if __name__ == "__main__":
    main()
