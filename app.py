# BlueStar Cascade v3.0 — Clean & Organized by Timeframes (M15 / H1 / H4 / D1)
# - "FIXED" removed everywhere
# - M15 added to scan
# - Results organized in columns by timeframe (visual artifact preserved)
# - Tables shown without quadrillage (CSS + st.table)
# - Cleaned, slightly simplified logic but compatible with original architecture
# Paste this file in your Streamlit app folder and run with `streamlit run filename.py`.

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import hashlib

# OANDA client
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error

# PDF export
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ----------------------- Basic config -----------------------
st.set_page_config(page_title="BlueStar Cascade v3.0", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bluestar")

TUNIS_TZ = pytz.timezone("Africa/Tunis")

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK","XAU_USD","XPT_USD"
]

GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}

# ----------------------- CSS (remove gridlines, TF headers) -----------------------
st.markdown(
    """
    <style>
      /* Streamlit tables use <table> elements — remove borders */
      table {border-collapse: collapse; width: 100%;}
      table th, table td {border: none !important; padding: 6px 8px !important;}
      table tr:nth-child(even) {background: rgba(255,255,255,0.02);}
      .tf-header {
          background: linear-gradient(135deg, rgba(0,200,150,0.06), rgba(0,120,200,0.03));
          border: 1px solid rgba(255,255,255,0.03);
          padding: 8px;
          border-radius: 10px;
          text-align: center;
          margin-bottom: 8px;
      }
      .tf-header h4 {margin: 0; color: #00ff99;}
      .tf-header p {margin: 0; color: #aab;}
      .small-metric {font-size:0.85rem; color:#cdd;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------- OANDA client -----------------------
@st.cache_resource
def get_oanda_client():
    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        return API(access_token=token)
    except Exception as e:
        logger.error("OANDA token missing or invalid in Streamlit secrets.")
        st.error("OANDA token manquant dans les secrets - configuration requise.")
        st.stop()

client = get_oanda_client()

# ----------------------- Utilities & caching -----------------------
def retry_on_error(max_attempts: int = 3, wait: float = 0.2):
    def deco(func):
        @wraps(func)
        def wrapper(*a, **kw):
            last_exc = None
            for i in range(max_attempts):
                try:
                    return func(*a, **kw)
                except Exception as e:
                    last_exc = e
                    time.sleep(wait * (i + 1))
            logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {last_exc}")
            raise last_exc
        return wrapper
    return deco

def cache_key(pair: str, tf: str, count: int) -> str:
    return hashlib.md5(f"{pair}_{tf}_{count}".encode()).hexdigest()

@st.cache_data(ttl=30)
@retry_on_error(max_attempts=3)
def fetch_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    """Fetch candles from OANDA and return a tz-aware dataframe in TUNIS timezone."""
    gran = GRANULARITY_MAP.get(tf)
    if gran is None:
        return pd.DataFrame()
    params = {"granularity": gran, "count": count, "price": "M"}
    req = InstrumentsCandles(instrument=pair, params=params)
    client.request(req)
    raw = []
    for c in req.response.get("candles", []):
        raw.append({
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "complete": c.get("complete", False)
        })
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    # convert to Tunisia timezone for consistent display
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(TUNIS_TZ).dt.tz_localize(None)
    return df

# ----------------------- Indicators (stable & light) -----------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA as proxy HMA (lightweight)
    df["ema10"] = close.ewm(span=10, adjust=False).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["hma_up"] = df["ema10"] > df["ema20"]

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, min_periods=14).mean()
    ma_down = down.ewm(alpha=1/14, min_periods=14).mean()
    rs = ma_up / ma_down
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR (simple)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, min_periods=14).mean()
    # ATR percentile
    if len(df) >= 50:
        df["atr_pct"] = df["atr"].rolling(100, min_periods=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100)
    else:
        df["atr_pct"] = np.nan

    # ADX proxy (smoothed true range)
    df["adx"] = df["atr"].ewm(alpha=1/14, min_periods=14).mean()

    # UT state (simple trend detector)
    df["ut_state"] = np.where(close > close.rolling(20).mean(), 1, -1)
    return df

# ----------------------- Data structures -----------------------
@dataclass
class Signal:
    timestamp: datetime
    pair: str
    timeframe: str
    action: str
    entry: float
    stop_loss: float
    take_profit: float
    score: int
    risk_reward: float
    adx: float
    rsi: float
    atr_pct: float
    session: str

# ----------------------- Session detection -----------------------
def get_active_session(dt: datetime) -> str:
    hour_utc = dt.astimezone(pytz.UTC).hour
    if 0 <= hour_utc < 9:
        return "Tokyo"
    if 8 <= hour_utc < 17:
        return "London"
    if 13 <= hour_utc < 22:
        return "NY"
    return "Off-Hours"

def session_badge_html(session: str) -> str:
    color_map = {"London": "#ff6b6b", "NY": "#4ecdc4", "Tokyo": "#ffe66d", "Off-Hours": "#777"}
    color = color_map.get(session, "#777")
    return f"<span style='background:{color};color:#fff;padding:4px 8px;border-radius:8px;font-size:0.8rem'>{session}</span>"

# ----------------------- Analysis core -----------------------
def analyze_pair(pair: str, tf: str, params: dict) -> Optional[Signal]:
    try:
        df = fetch_candles(pair, tf, count=300)
        if df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # strict boolean check for HMA direction
        if not isinstance(last["hma_up"], (bool, np.bool_)) or not isinstance(prev["hma_up"], (bool, np.bool_)):
            return None

        # detect flips
        flip_long = (last["hma_up"] is True) and (prev["hma_up"] is False)
        flip_short = (last["hma_up"] is False) and (prev["hma_up"] is True)

        if not (flip_long or flip_short):
            return None

        action = "BUY" if flip_long else "SELL"

        # volatility filter
        if pd.notna(last["atr_pct"]) and last["atr_pct"] < params.get("min_vol_pct", 10):
            return None

        # basic scoring
        score = 50
        if last["adx"] > params.get("adx_strong", 25):
            score += 25
        elif last["adx"] > params.get("adx_min", 20):
            score += 10

        if flip_long or flip_short:
            score += 15

        # RSI contribution
        if action == "BUY":
            score += 10 if last["rsi"] > 50 else 0
        else:
            score += 10 if last["rsi"] < 50 else 0

        score = min(100, int(score))

        # entry / SL / TP using ATR multipliers
        atr = last["atr"] if pd.notna(last["atr"]) else 0.0
        entry = float(last["close"])
        sl = entry - params.get("atr_sl_mult", 2.0) * atr if action == "BUY" else entry + params.get("atr_sl_mult", 2.0) * atr
        tp = entry + params.get("atr_tp_mult", 3.5) * atr if action == "BUY" else entry - params.get("atr_tp_mult", 3.5) * atr

        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
        if rr < params.get("min_rr", 1.2):
            return None

        # session/time
        local_time = last["time"] if isinstance(last["time"], datetime) else datetime.now()
        session = get_active_session(local_time)

        return Signal(
            timestamp=local_time,
            pair=pair,
            timeframe=tf,
            action=action,
            entry=entry,
            stop_loss=sl,
            take_profit=tp,
            score=score,
            risk_reward=round(rr, 2),
            adx=float(last["adx"]) if pd.notna(last["adx"]) else 0.0,
            rsi=float(last["rsi"]) if pd.notna(last["rsi"]) else 0.0,
            atr_pct=float(last["atr_pct"]) if pd.notna(last["atr_pct"]) else 0.0,
            session=session
        )
    except V20Error as e:
        logger.warning(f"OANDA API error for {pair} {tf}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error analyzing {pair} {tf}: {e}")
        return None

# ----------------------- Run scan -----------------------
def run_scan(pairs: List[str], tfs: List[str], params: dict, max_workers: int = 6) -> List[Signal]:
    signals: List[Signal] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_pair, p, tf, params): (p, tf) for p in pairs for tf in tfs}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    signals.append(res)
            except Exception:
                # ignore single failure
                pass
    # sort by score desc, time desc
    signals.sort(key=lambda s: (s.timeframe, -s.score, s.timestamp), reverse=False)
    return signals

# ----------------------- PDF generator -----------------------
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=10*mm, rightMargin=10*mm,
                            topMargin=10*mm, bottomMargin=10*mm)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("BlueStar Cascade v3.0", styles["Title"]))
    elements.append(Spacer(1, 6*mm))

    data = [["Time", "Pair", "TF", "Qual", "Action", "Entry", "SL", "TP", "Score", "R:R", "RSI", "ADX", "ATR%"]]
    for s in sorted(signals, key=lambda x: (-x.score, x.timestamp)):
        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            "STD",
            s.action,
            f"{s.entry:.5f}",
            f"{s.stop_loss:.5f}",
            f"{s.take_profit:.5f}",
            str(s.score),
            f"{s.risk_reward:.2f}",
            f"{int(s.rsi)}",
            f"{int(s.adx)}",
            f"{int(s.atr_pct)}"
        ])
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0f2a3d")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ----------------------- UI -----------------------
def main():
    st.markdown("# BlueStar Cascade v3.0")
    st.markdown("Clean interface — organized by timeframe (M15 / H1 / H4 / D1)")

    # Top controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.caption("Pairs: default major + crosses (editable below)")
    with col2:
        balance = st.number_input("Balance (USD)", min_value=100.0, value=10000.0, step=100.0)
    with col3:
        risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    with col4:
        scan_button = st.button("🚀 SCAN v3.0", type="primary")

    with st.expander("Options avancées", expanded=False):
        c1, c2, c3 = st.columns(3)
        atr_sl_mult = c1.number_input("SL ATR multiplier", 1.0, 4.0, 2.0, 0.1)
        atr_tp_mult = c2.number_input("TP ATR multiplier", 1.0, 6.0, 3.5, 0.1)
        min_vol_pct = c3.slider("Min volatility percentile", 0, 50, 15, 5)
        adx_min = st.number_input("ADX min (proxy)", 5, 40, 20, 1)
        adx_strong = st.number_input("ADX strong (proxy)", 10, 60, 25, 1)
        min_rr = st.number_input("Min R:R", 1.0, 5.0, 1.2, 0.1)

    # Pair picker
    pairs_input = st.multiselect("Pairs to scan", PAIRS_DEFAULT, default=PAIRS_DEFAULT[:18], help="Choisir les paires à scanner")

    # When scan clicked
    if scan_button:
        tfs = ["M15", "H1", "H4", "D1"]
        params = {
            "atr_sl_mult": atr_sl_mult,
            "atr_tp_mult": atr_tp_mult,
            "min_vol_pct": min_vol_pct,
            "adx_min": adx_min,
            "adx_strong": adx_strong,
            "min_rr": min_rr
        }
        with st.spinner("🔍 Scanning markets..."):
            signals = run_scan(pairs_input or PAIRS_DEFAULT, tfs, params)

        # Metrics
        st.markdown("---")
        cols = st.columns(6)
        cols[0].metric("Signaux", len(signals))
        inst_count = sum(1 for s in signals if s.score >= 85)
        cols[1].metric("Institutional (≥85)", inst_count)
        cols[2].metric("Avg Score", f"{int(np.mean([s.score for s in signals]) ) if signals else 0}")
        cols[3].metric("Avg R:R", f"{round(np.mean([s.risk_reward for s in signals]),2) if signals else 0}")
        cols[4].metric("Balance", f"${int(balance)}")
        cols[5].metric("Risk/trade", f"{risk_pct:.2f}%")

        # Downloads
        if signals:
            df_export = pd.DataFrame([{
                "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Pair": s.pair.replace("_", "/"),
                "TF": s.timeframe,
                "Action": s.action,
                "Entry": s.entry,
                "SL": s.stop_loss,
                "TP": s.take_profit,
                "Score": s.score,
                "R:R": s.risk_reward,
                "RSI": int(s.rsi),
                "ADX": int(s.adx),
                "ATR%": int(s.atr_pct),
                "Session": s.session
            } for s in signals])
            csv_bytes = df_export.to_csv(index=False).encode()
            st.download_button("📥 Télécharger CSV", csv_bytes, "bluestar_scan_v3.csv", "text/csv")
            pdf_bytes = generate_pdf(signals)
            st.download_button("📄 Télécharger PDF", pdf_bytes, "bluestar_scan_v3.pdf", "application/pdf")

        # Layout by timeframe: M15 / H1 / H4 / D1
        st.markdown("---")
        st.markdown("### Résultats par timeframe")
        col_M15, col_H1, col_H4, col_D1 = st.columns(4)

        tf_columns = {
            "M15": col_M15,
            "H1": col_H1,
            "H4": col_H4,
            "D1": col_D1
        }

        # For each timeframe, build display dataframe and show header + table
        for tf, column in tf_columns.items():
            with column:
                tf_signals = [s for s in signals if s.timeframe == tf]
                st.markdown(f"<div class='tf-header'><h4>{tf}</h4><p class='small-metric'>{len(tf_signals)} signal{'s' if len(tf_signals)>1 else ''}</p></div>", unsafe_allow_html=True)
                if tf_signals:
                    # sort top by score desc then time desc
                    tf_signals.sort(key=lambda x: (-x.score, x.timestamp))
                    df_disp = pd.DataFrame([{
                        "Heure": s.timestamp.strftime("%H:%M"),
                        "Paire": s.pair.replace("_","/"),
                        "Qual": "INST" if s.score>=85 else ("PREM" if s.score>=75 else "STD"),
                        "Action": ("🟢 " + s.action) if s.action=="BUY" else ("🔴 " + s.action),
                        "Score": s.score,
                        "Entry": f"{s.entry:.5f}",
                        "SL": f"{s.stop_loss:.5f}",
                        "TP": f"{s.take_profit:.5f}",
                        "R:R": f"{s.risk_reward:.2f}:1",
                        "ATR%": int(s.atr_pct),
                        "RSI": int(s.rsi),
                        "ADX": int(s.adx),
                        "Sess": s.session[:3]
                    } for s in tf_signals])
                    # present as a plain table (no gridlines due to CSS above)
                    st.table(df_disp)
                else:
                    st.info("Aucun signal")

    # Footer
    st.markdown("---")
    now = datetime.now(TUNIS_TZ).strftime("%d/%m/%Y %H:%M:%S")
    st.markdown(f"<div style='text-align:center;color:#888;font-size:0.8rem'>BlueStar Cascade v3.0 — {now} — Clean UI — Organized by timeframe</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
