"""
BlueStar Institutional v3.0 - VERSION FINALE
→ Seulement signaux INSTITUTIONAL
→ Visuel propre + M15 + PDF/CSV
→ Aucun détail inutile (pas de score, risk $, taille…)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib

# OANDA
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# PDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide")
logging.basicConfig(level=logging.INFO)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem;}
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.25), rgba(0,200,255,0.25));
        padding: 14px; border-radius: 12px; text-align: center; margin: 20px 0 10px 0;
        border: 2px solid rgba(0,255,136,0.5);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.5rem;}
    .tf-header p {margin: 5px 0 0; color: #a0a0c0; font-size: 0.9rem;}
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

# ==================== OANDA ====================
@st.cache_resource
def get_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
client = get_client()

@st.cache_data(ttl=60, show_spinner=False)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    params = {"granularity": GRANULARITY_MAP[tf], "count": count, "price": "M"}
    req = InstrumentsCandles(instrument=pair, params=params)
    client.request(req)
    data = []
    for c in req.response.get("candles", []):
        data.append({
            "time": pd.to_datetime(c["time"]).tz_localize(None),
            "close": float(c["mid"]["c"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "complete": c.get("complete", False)
        })
    df = pd.DataFrame(data)
    return df.iloc[-100:] if len(df) >= 100 else df

# ==================== INDICATORS ====================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    c, h, l = df['close'], df['high'], df['low']

    # HMA
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
    df['hma'] = wma(2*wma(c,10) - wma(c,20), int(np.sqrt(20)))
    df['hma_up'] = (df['hma'] > df['hma'].shift(1)).fillna(False).astype(bool)

    # RSI + ATR + ADX + UT
    delta = c.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - (100 / (1 + up.ewm(alpha=1/14).mean() / down.ewm(alpha=1/14).mean()))

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14).mean()
    df['atr'] = atr
    df['atr_pct'] = atr.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100 if len(x)>=50 else np.nan)

    # UT Bot
    loss = 2 * atr
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
    plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=180)
def higher_trend(pair: str, tf: str) -> str:
    higher_tf = {"M15":"H1", "H1":"H4", "H4":"D1"}.get(tf, "D1")
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50: return "Neutral"
    df = calc_indicators(df)
    close = df['close'].iloc[-1]
    hma_up = df['hma'].iloc[-1] > df['hma'].iloc[-2]
    if close > df['close'].ewm(span=50).mean().iloc[-1] and hma_up and df['adx'].iloc[-1] > 25:
        return "Bullish"
    if close < df['close'].ewm(span=50).mean().iloc[-1] and not hma_up and df['adx'].iloc[-1] > 25:
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

def analyze(pair: str, tf: str) -> Optional[Signal]:
    try:
        df = get_candles(pair, tf, 300)
        if len(df) < 100: return None
        df = calc_indicators(df)
        last = df.iloc[-2]  # bougie confirmée

        # Conditions strictes Institutional
        if last['atr_pct'] < 20: return None
        if last['adx'] < 25: return None
        if not last['hma_up'] and df.iloc[-3]['hma_up']:  # flip vert récent
            if last['rsi'] <= 55 or last['ut'] != 1: return None
            action = "BUY"
        elif last['hma_up'] and not df.iloc[-3]['hma_up']:  # flip rouge récent
            if last['rsi'] >= 45 or last['ut'] != -1: return None
            action = "SELL"
        else:
            return None

        trend = higher_trend(pair, tf)
        if trend == "Neutral": return None
        if action == "BUY" and trend != "Bullish": return None
        if action == "SELL" and trend != "Bearish": return None

        atr = last['atr']
        sl = last['close'] - 2*atr if action == "BUY" else last['close'] + 2*atr
        tp = last['close'] + 3.5*atr if action == "BUY" else last['close'] - 3.5*atr
        rr = round(abs(tp - last['close']) / abs(sl - last['close']), 1)

        t = last['time'].astimezone(TUNIS_TZ) if last['time'].tzinfo else pytz.utc.localize(last['time']).astimezone(TUNIS_TZ)
        session = "London" if 8 <= t.hour < 17 else "NY" if 13 <= t.hour < 22 else "Tokyo" if t.hour < 9 else "Off"

        return Signal(pair, tf, action, round(last['close'], 5), round(sl, 5), round(tp, 5), rr, trend, session, t)
    except:
        return None

# ==================== SCAN ====================
def run_scan() -> List[Signal]:
    signals = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(analyze, p, tf) for p in PAIRS for tf in TFS]
        for f in as_completed(futures):
            r = f.result()
            if r: signals.append(r)
    return signals

# ==================== PDF ====================
def make_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, leftMargin=15*mm, rightMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<font size=18 color=#00ff88><b>BlueStar Institutional Signals</b></font>", styles["Title"]))
    elements.append(Paragraph(f"<font size=10>Généré le {datetime.now(TUNIS_TZ).strftime('%d/%m/%Y %H:%M')} (Tunis)</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))

    data = [["Paire", "TF", "Action", "Entry", "SL", "TP", "R:R", "Trend HTF", "Session", "Heure"]]
    for s in sorted(signals, key=lambda x: (TFS.index(x.tf), x.time), reverse=True):
        data.append([
            s.pair.replace("_", "/"), s.tf, s.action,
            str(s.entry), str(s.sl), str(s.tp),
            str(s.rr), s.trend, s.session, s.time.strftime("%H:%M")
        ])

    table = Table(data, colWidths=[45,30,40,50,50,50,35,55,45,40])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN ====================
def main():
    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("# BlueStar Institutional")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL GRADE ONLY</span>', unsafe_allow_html=True)
    with col2:
        now = datetime.now(TUNIS_TZ)
        session = "London" if 8 <= now.hour < 17 else "NY" if 13 <= now.hour < 22 else "Tokyo" if now.hour < 9 else "Off"
        badge = "<span class='session-badge session-london'>LONDON</span>" if session=="London" else \
                "<span class='session-badge session-ny'>NY</span>" if session=="NY" else \
                "<span class='session-badge session-tokyo'>TOKYO</span>" if session=="Tokyo" else "OFF"
        st.markdown(f"**{now.strftime('%H:%M:%S')} Tunis** {badge}", unsafe_allow_html=True)

    scan = st.button("SCAN", type="primary", use_container_width=True)

    if scan:
        with st.spinner("Recherche signaux institutionnels sur M15 • H1 • H4 • D1..."):
            signals = run_scan()

        if signals:
            # Téléchargements
            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                csv = pd.DataFrame([{
                    "Paire": s.pair.replace("_","/"), "TF": s.tf, "Action": s.action,
                    "Entry": s.entry, "SL": s.sl, "TP": s.tp, "R:R": s.rr,
                    "Trend HTF": s.trend, "Session": s.session, "Heure": s.time.strftime("%H:%M")
                } for s in signals]).to_csv(index=False).encode()
                st.download_button("CSV", csv, "bluestar_institutional.csv", "text/csv")

            with col_dl2:
                pdf = make_pdf(signals)
                st.download_button("PDF", pdf, "BlueStar_Institutional.pdf", "application/pdf")

            # Affichage par timeframe
            for tf in TFS:
                tf_signals = [s for s in signals if s.tf == tf]
                if tf_signals:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_signals)} signal{'s' if len(tf_signals)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df_disp = pd.DataFrame([{
                        "Paire": s.pair.replace("_","/"),
                        "Action": f"BUY" if s.action=="BUY" else "SELL",
                        "Entry": s.entry,
                        "SL": s.sl,
                        "TP": s.tp,
                        "R:R": s.rr,
                        "Trend HTF": s.trend,
                        "Session": s.session,
                        "Heure": s.time.strftime("%H:%M")
                    } for s in sorted(tf_signals, key=lambda x: x.time, reverse=True)])
                    st.dataframe(df_disp, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun signal institutionnel détecté pour le moment.")

if __name__ == "__main__":
    main()
