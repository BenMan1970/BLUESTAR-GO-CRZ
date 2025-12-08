"""
BlueStar Institutional v3.0 FINAL - SEULEMENT LES SIGNAUX INSTITUTIONNELS
+ Configuration complète
+ M15 ajouté
+ PDF + CSV
+ Visuel ancien restauré (sans bordures)
+ Bouton SCAN
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
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="collapsed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 5px 14px; border-radius: 20px; font-weight: bold; font-size: 0.8rem;}
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.25), rgba(0,200,255,0.25));
        padding: 14px; border-radius: 12px; text-align: center; margin: 20px 0 10px 0;
        border: 2px solid rgba(0,255,136,0.5);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.6rem;}
    .tf-header p {margin: 6px 0 0; color: #a0a0c0; font-size: 1rem;}
    .stDataFrame, .stDataFrame table, .stDataFrame td, .stDataFrame th {border: none !important;}
    .stDataFrame thead {display: none;}
    .session-badge {padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.75rem;}
    .session-london {background: #ff4444; color: white;}
    .session-ny {background: #00cccc; color: white;}
    .session-tokyo {background: #ffdd00; color: black;}
</style>
""", unsafe_allow_html=True)

PAIRS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
         "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
         "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
         "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","XAU_USD"]

TFS = ["M15", "H1", "H4", "D1"]
GRANULARITY_MAP = {"M15":"M15", "H1":"H1", "H4":"H4", "D1":"D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== PARAMÈTRES ====================
@dataclass
class Params:
    atr_sl: float = 2.0
    atr_tp: float = 3.5
    min_rr: float = 1.5
    cascade: bool = True
    strict_flip: bool = True
    min_vol: float = 20.0
    min_adx: int = 25

# ==================== OANDA + CACHE ====================
@st.cache_resource
def get_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])

client = get_client()

@st.cache_data(ttl=60, show_spinner=False)
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
    except:
        return pd.DataFrame()

# ==================== INDICATORS ====================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 100: return df
    c, h, l = df['close'], df['high'], df['low']

    # HMA
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/sum(np.arange(1,n+1)), raw=True)
    df['hma'] = wma(2*wma(c,10) - wma(c,20), int(np.sqrt(20)))
    df['hma_up'] = (df['hma'] > df['hma'].shift(1)).fillna(False).astype(bool)

    # RSI
    delta = c.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.ewm(alpha=1/14).mean()/down.ewm(alpha=1/14).mean())

    # ATR + Percentile
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14).mean()
    df['atr_pct'] = df['atr'].rolling(100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100 if len(x)>=50 else np.nan
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
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / df['atr']
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / df['atr']
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=300)
def higher_trend(pair: str, tf: str) -> str:
    higher = {"M15":"H1", "H1":"H4", "H4":"D1"}.get(tf, "D1")
    df = get_candles(pair, higher, 100)
    if len(df) < 50: return "Neutral"
    df = calc_indicators(df)
    close = df['close'].iloc[-1]
    hma_up = df['hma'].iloc[-1] > df['hma'].iloc[-2]
    adx = df['adx'].iloc[-1]
    if close > df['close'].ewm(span=50).mean().iloc[-1] and hma_up and adx > 25:
        return "Bullish"
    if close < df['close'].ewm(span=50).mean().iloc[-1] and not hma_up and adx > 25:
        return "Bearish"
    return "Neutral"

# ==================== SIGNAL ====================
@dataclass
class Signal:
    pair: str
    tf: str
    action: str
    entry: float
    sl: float
    tp: float
    rr: float
    trend: str
    session: str
    time: datetime

def analyze(pair: str, tf: str, params: Params) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100: return None
        df = calc_indicators(df)
        last = df.iloc[-2]  # bougie confirmée
        prev = df.iloc[-3]

        if pd.isna(last['atr_pct']) or last['atr_pct'] < params.min_vol: return None
        if last['adx'] < params.min_adx: return None

        flip_up = last['hma_up'] and not prev['hma_up']
        flip_down = not last['hma_up'] and prev['hma_up']

        if params.strict_flip:
            if not (flip_up or flip_down): return None
        else:
            if not (flip_up or flip_down or (last['hma_up'] and prev['hma_up'] and not df.iloc[-4]['hma_up']) or
                    (not last['hma_up'] and not prev['hma_up'] and df.iloc[-4]['hma_up'])): return None

        if flip_up and last['rsi'] > 55 and last['ut'] == 1:
            action = "BUY"
        elif flip_down and last['rsi'] < 45 and last['ut'] == -1:
            action = "SELL"
        else:
            return None

        trend = higher_trend(pair, tf)
        if params.cascade and ((action == "BUY" and trend != "Bullish") or (action == "SELL" and trend != "Bearish")):
            return None

        atr = last['atr']
        sl = round(last['close'] - params.atr_sl * atr, 5) if action == "BUY" else round(last['close'] + params.atr_sl * atr, 5)
        tp = round(last['close'] + params.atr_tp * atr, 5) if action == "BUY" else round(last['close'] - params.atr_tp * atr, 5)
        rr = round(abs(tp - last['close']) / abs(sl - last['close']), 1)
        if rr < params.min_rr: return None

        t = last['time'].astimezone(TUNIS_TZ) if last['time'].tzinfo else pytz.utc.localize(last['time']).astimezone(TUNIS_TZ)
        session = "London" if 8 <= t.hour < 17 else "NY" if 13 <= t.hour < 22 else "Tokyo" if t.hour < 9 else "Off"

        return Signal(pair.replace("_", "/"), tf, action, round(last['close'], 5), sl, tp, rr, trend, session, t)
    except:
        return None

# ==================== SCAN ====================
def run_scan(params: Params) -> List[Signal]:
    signals = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(analyze, pair, tf, params) for pair in PAIRS for tf in TFS]
        for future in as_completed(futures):
            result = future.result()
            if result:
                signals.append(result)
    return signals

# ==================== PDF ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, leftMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<font size=18 color=#00ff88><b>BlueStar Institutional Signals</b></font>", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M')} (Tunis)", styles["Normal"]))
    elements.append(Spacer(1, 12*mm))

    data = [["Paire", "TF", "Action", "Entry", "SL", "TP", "R:R", "Trend", "Session", "Heure"]]
    for s in sorted(signals, key=lambda x: (TFS.index(x.tf), -x.time.timestamp())):
        data.append([s.pair, s.tf, s.action, s.entry, s.sl, s.tp, s.rr, s.trend, s.session, s.time.strftime("%H:%M")])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN ====================
def main():
    col1, col2 = st.columns([3, 4])
    with col1:
        st.markdown("# BlueStar Institutional")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL ONLY</span>', unsafe_allow_html=True)
    with col2:
        now = datetime.now(TUNIS_TZ)
        session = "London" if 8 <= now.hour < 17 else "NY" if 13 <= now.hour < 22 else "Tokyo" if now.hour < 9 else "Off"
        badge = f"<span class='session-badge session-{'london' if session=='London' else 'ny' if session=='NY' else 'tokyo'}'>{session}</span>"
        st.markdown(f"**{now.strftime('%H:%M:%S')} Tunis** {badge}", unsafe_allow_html=True)

    with st.expander("Configuration Institutionnelle", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        sl = c1.slider("SL × ATR", 1.0, 3.0, 2.0, 0.1)
        tp = c2.slider("TP × ATR", 2.5, 5.0, 3.5, 0.1)
        rr = c3.slider("Min R:R", 1.3, 2.5, 1.6, 0.1)
        cascade = c4.checkbox("Cascade HTF obligatoire", True)
        strict = st.checkbox("Flip strict uniquement", True)
        vol = st.slider("Volatilité min (%ile)", 10, 50, 20, 5)

    if st.button("SCAN", type="primary", use_container_width=True):
        params = Params(sl, tp, rr, cascade, strict, vol, 25)
        with st.spinner("Scan institutionnel en cours..."):
            signals = run_scan(params)

        if signals:
            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                csv = pd.DataFrame([vars(s) for s in signals]).to_csv(index=False).encode()
                st.download_button("CSV", csv, "bluestar_institutional.csv", "text/csv")
            with col_dl2:
                pdf = generate_pdf(signals)
                st.download_button("PDF", pdf, "BlueStar_Institutional.pdf", "application/pdf")

            for tf in TFS:
                tf_sig = [s for s in signals if s.tf == tf]
                if tf_sig:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df_show = pd.DataFrame([{
                        "Paire": s.pair, "Action": s.action, "Entry": s.entry,
                        "SL": s.sl, "TP": s.tp, "R:R": s.rr,
                        "Trend": s.trend, "Session": s.session, "Heure": s.time.strftime("%H:%M")
                    } for s in tf_sig])
                    st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun signal institutionnel détecté actuellement.")

if __name__ == "__main__":
    main()
