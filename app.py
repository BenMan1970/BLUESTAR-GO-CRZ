# ================= BLUESTAR CASCADE v3.0 - CLEAN VERSION =================
# ✔ FIXED supprimé partout
# ✔ Ajout M15
# ✔ Tableau sans quadrillage
# ✔ Nettoyage interface + titres + badges
# ==========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
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
st.set_page_config(page_title="BlueStar Cascade v3.0", layout="wide")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== CSS CLEAN ====================
st.markdown("""
<style>
    .main {background: #0c0f24;}
    .block-container {padding-top: 1rem !important;}
    .tf-header {
        background: rgba(0,255,170,0.1);
        border: 1px solid rgba(0,255,150,0.3);
        padding: 10px; border-radius: 10px;
        text-align: center; margin-bottom: 10px;
    }
    .tf-header h3 {margin: 0; color: #00ff99;}
    .tf-header p {margin: 0; color: #aaa;}
    .stTable {background: transparent;}
    table {border-collapse: collapse; width: 100%;}
    table td, table th {
        border: none !important;
        padding: 6px 4px !important;
    }
    table tr:nth-child(even) {background: rgba(255,255,255,0.03);}
</style>
""", unsafe_allow_html=True)

# ==================== PAIRS ====================
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK","XAU_USD","XPT_USD"
]

GRAN_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS = pytz.timezone("Africa/Tunis")

# ======================================================================
#                        SIMPLE HELPERS
# ======================================================================

def get_active_session(dt: datetime) -> str:
    h = dt.astimezone(pytz.UTC).hour
    if 0 <= h < 9: return "Tokyo"
    if 8 <= h < 17: return "London"
    if 13 <= h < 22: return "New York"
    return "Off"

# ======================================================================
#                        API CLIENT
# ======================================================================
@st.cache_resource
def get_client():
    token = st.secrets["OANDA_ACCESS_TOKEN"]
    return API(access_token=token)

client = get_client()

# ======================================================================
#                       CANDLES FETCH
# ======================================================================

def retry(max_attempts=3):
    def deco(func):
        def wrapper(*a, **kw):
            for i in range(max_attempts):
                try:
                    return func(*a, **kw)
                except Exception:
                    time.sleep(0.3)
            return None
        return wrapper
    return deco

@st.cache_data(ttl=20)
@retry()
def get_candles(pair: str, tf: str, count: int):
    gran = GRAN_MAP.get(tf)
    req = InstrumentsCandles(instrument=pair, params={"count": count, "granularity": gran, "price": "M"})
    client.request(req)
    data = []
    for c in req.response["candles"]:
        data.append({
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "complete": c["complete"]
        })
    df = pd.DataFrame(data)
    df["time"] = df["time"].dt.tz_convert(TUNIS)
    return df

# ======================================================================
#                       INDICATORS (light)
# ======================================================================
def calc_indicators(df):
    close = df["close"]

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(span=14).mean() / down.ewm(span=14).mean()
    df["rsi"] = 100 - (100/(1+rs))

    # HMA direction (simplifiée)
    hma = close.rolling(10).mean()
    df["hma_up"] = hma > hma.shift(1)

    # ADX approximatif
    df["adx"] = (df["high"] - df["low"]).ewm(span=14).mean()

    # ATR %ile light
    df["atr"] = (df["high"] - df["low"]).ewm(span=14).mean()
    df["atr_percentile"] = df["atr"].rank(pct=True)*100

    df["ut_state"] = np.where(close > close.rolling(20).mean(), 1, -1)

    return df

# ======================================================================
#                  DATA STRUCTURES SIMPLIFIÉES
# ======================================================================
@dataclass
class Signal:
    timestamp: datetime
    pair: str
    timeframe: str
    action: str
    entry: float
    sl: float
    tp: float
    score: int
    rr: float
    adx: float
    rsi: float
    atr_pct: float

# ======================================================================
#                      ANALYZE A PAIR
# ======================================================================
def analyze_pair(pair: str, tf: str):
    df = get_candles(pair, tf, 300)
    if df is None or len(df) < 50: return None
    df = calc_indicators(df)
    last = df.iloc[-1]; prev = df.iloc[-2]

    flip_long = last["hma_up"] and not prev["hma_up"]
    flip_short = not last["hma_up"] and prev["hma_up"]

    if not (flip_long or flip_short):
        return None

    action = "BUY" if flip_long else "SELL"
    atr = last["atr"]
    entry = last["close"]
    sl = entry - 2*atr if action=="BUY" else entry + 2*atr
    tp = entry + 3.5*atr if action=="BUY" else entry - 3.5*atr
    rr = abs(tp-entry)/abs(entry-sl)

    score = 60
    score += 20 if last["adx"] > prev["adx"] else 0

    return Signal(
        timestamp=last["time"],
        pair=pair,
        timeframe=tf,
        action=action,
        entry=entry,
        sl=sl,
        tp=tp,
        score=score,
        rr=rr,
        adx=last["adx"],
        rsi=last["rsi"],
        atr_pct=last["atr_percentile"]
    )

# ======================================================================
#                       RUN SCAN
# ======================================================================
def run_scan():
    signals = []
    tfs = ["M15","H1","H4","D1"]

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(analyze_pair, p, tf): (p, tf)
                   for p in PAIRS_DEFAULT for tf in tfs}

        for f in as_completed(futures):
            try:
                res = f.result()
                if res: signals.append(res)
            except:
                pass

    return signals

# ======================================================================
#                             PDF EXPORT
# ======================================================================
def generate_pdf(signals: List[Signal]):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = [
        Paragraph("<b>BlueStar Cascade v3.0</b>", styles["Title"]),
        Spacer(1, 8*mm)
    ]

    data = [["Time","Pair","TF","Action","Entry","SL","TP","Score","R:R","RSI","ADX","ATR%"]]

    for s in signals:
        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_","/"),
            s.timeframe,
            s.action,
            f"{s.entry:.5f}",
            f"{s.sl:.5f}",
            f"{s.tp:.5f}",
            s.score,
            f"{s.rr:.1f}",
            f"{s.rsi:.0f}",
            f"{s.adx:.0f}",
            f"{s.atr_pct:.0f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor("#0f2a3d")),
        ('TEXTCOLOR',(0,0),(-1,0), colors.white),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ]))

    elements.append(table)
    doc.build(elements)
    return buf.getvalue()

# ======================================================================
#                            UI
# ======================================================================
def main():
    st.markdown("# BlueStar Cascade v3.0")
    st.caption("Institutional Grade – Clean Version")

    if st.button("🚀 Lancer Scan"):
        with st.spinner("Analyse des marchés..."):
            signals = run_scan()

        st.success(f"{len(signals)} signaux trouvés")

        if signals:
            df = pd.DataFrame([{
                "Heure": s.timestamp.strftime("%H:%M"),
                "Paire": s.pair.replace("_","/"),
                "TF": s.timeframe,
                "Action": s.action,
                "Entry": round(s.entry,5),
                "SL": round(s.sl,5),
                "TP": round(s.tp,5),
                "Score": s.score,
                "R:R": round(s.rr,1),
                "RSI": int(s.rsi),
                "ADX": int(s.adx),
                "ATR%": int(s.atr_pct)
            } for s in signals])

            st.table(df)

            pdf = generate_pdf(signals)
            st.download_button("📄 Export PDF", pdf, "bluestar_scan.pdf")

# ======================================================================
if __name__ == "__main__":
    main()
