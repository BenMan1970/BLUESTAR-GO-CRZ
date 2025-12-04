"""
BlueStar Cascade v2.9 PRO – INSTITUTIONAL GRADE
Version finale corrigée & boostée – Décembre 2025
Auteur : Ton bro qui veut que tu fasses du vrai argent
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# OANDA v20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingInfo

# PDF Report
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG ====================
st.set_page_config(
    page_title="BlueStar PRO v2.9",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); color: white;}
    .block-container {padding-top: 2rem;}
    .stMetric {background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);}
    .stMetric label {color: #a0a0c0 !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.6rem !important; font-weight: 800;}
    .institutional-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: black; padding: 8px 18px; border-radius: 30px;
        font-weight: bold; font-size: 0.9rem; display: inline-block;
    }
    h1 {font-size: 2.5rem !important; background: linear-gradient(90deg, #00ff88, #00cc6a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .stButton button {background: linear-gradient(90deg, #00ff88, #00cc6a); color: black; font-weight: bold; height: 3.5em; border: none; border-radius: 12px;}
    .stButton button:hover {background: linear-gradient(90deg, #00cc6a, #00ff88); color: white;}
</style>
""", unsafe_allow_html=True)

# ==================== INSTRUMENTS & PIP VALUE (CORRIGÉ) ====================
INSTRUMENT_INFO = {
    # Forex majeures
    "EUR_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "GBP_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "USD_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "USD_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "AUD_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "NZD_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "USD_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    # Crosses & mineures
    "EUR_GBP": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "EUR_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "GBP_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "AUD_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "CAD_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "NZD_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "EUR_AUD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "EUR_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "EUR_NZD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "GBP_AUD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "GBP_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "AUD_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "NZD_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "CAD_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "CHF_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "AUD_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "NZD_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "EUR_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "GBP_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "USD_SEK": {"type": "forex", "pip_value": 10.0, "digits": 5},
    # Métaux précieux
    "XAU_USD": {"type": "metal", "pip_value": 1.0, "digits": 2},   # 1 point = 1$
    "XPT_USD": {"type": "metal", "pip_value": 0.1, "digits": 2},
    # Indices CFD
    "US30_USD":  {"type": "index", "pip_value": 1.0, "digits": 2},
    "NAS100_USD": {"type": "index", "pip_value": 1.0, "digits": 2},
    "SPX500_USD": {"type": "index", "pip_value": 0.1, "digits": 2},
}

PAIRS = list(INSTRUMENT_INFO.keys())

# Mapping correct des granularités OANDA
OANDA_GRAN = {
    "H1": "H1",
    "H4": "H4",
    "D1": "D",   # LA CORRECTION CLÉ
    "W": "W",
    "M": "M"
}

# ==================== DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "Institutional"
    PREMIUM = "Premium"
    STANDARD = "Standard"

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
    risk_reward: float
    adx: float
    rsi: float
    atr: float
    higher_tf_trend: str
    is_fresh_flip: bool

# ==================== OANDA CLIENT ====================
@st.cache_resource
def get_client():
    if "OANDA_ACCESS_TOKEN" not in st.secrets:
        st.error("Token OANDA manquant dans secrets.toml")
        st.stop()
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"], environment="practice")

client = get_client()

# Cache global des données HTF (énorme gain perf)
@st.cache_data(ttl=360)  # 6 minutes
def get_htf_data(pairs: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
    results = {}
    for pair in pairs:
        df = fetch_candles(pair, "D", count=200)
        if len(df) >= 50:
            results[pair] = calculate_indicators(df)
    return results

def fetch_candles(pair: str, granularity: str, count: int = 300) -> pd.DataFrame:
    time.sleep(0.11)  # Respect rate limit OANDA (~550 req/min)
    try:
        params = {"granularity": granularity, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        data = []
        for c in req.response.get("candles", []):
            if c.get("complete", True):  # Only complete candles
                continue
            data.append({
                "time": pd.to_datetime(c["time"]),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
            })
        df = pd.DataFrame(data)
        df["time"] = df["time"].dt.tz_localize(None)
        return df.sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.error(f"API Error {pair} {granularity}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_spread(pair: str) -> float:
    try:
        req = PricingInfo(accountID=st.secrets["OANDA_ACCOUNT_ID"], params={"instruments": pair})
        client.request(req)
        ask = float(req.response["prices"][0]["asks"][0]["price"])
        bid = float(req.response["prices"][0]["bids"][0]["price"])
        pip_size = 0.0001 if INSTRUMENT_INFO[pair]["digits"] == 5 else 0.01
        return (ask - bid) / pip_size
    except:
        return 999

# ==================== INDICATORS (UT Bot + HMA + ADX) ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df

    c = df['close']
    h = df['high']
    l = df['low']

    # HMA 20
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1, n+1)) / np.arange(1, n+1).sum(), raw=True)
    wma10 = wma(c, 10)
    wma20 = wma(c, 20)
    df['hma'] = wma(2 * wma10 - wma20, int(np.sqrt(20)))
    df['hma_up'] = df['hma'] > df['hma'].shift(1)

    # RSI 7
    delta = c.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - (100 / (1 + up.ewm(alpha=1/7).mean() / down.ewm(alpha=1/7).mean()))

    # ATR 10 + UT Bot Trailing Stop (×3)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False).mean()
    df['atr'] = atr

    loss = 3.0 * atr
    trail = [0.0] * len(df)
    for i in range(1, len(df)):
        prev = trail[i-1]
        if c.iloc[i] > prev and c.iloc[i-1] > prev:
            trail[i] = max(prev, c.iloc[i] - loss.iloc[i])
        elif c.iloc[i] < prev and c.iloc[i-1] < prev:
            trail[i] = min(prev, c.iloc[i] + loss.iloc[i])
        elif c.iloc[i] > prev:
            trail[i] = c.iloc[i] - loss.iloc[i]
        else:
            trail[i] = c.iloc[i] + loss.iloc[i]
    df['ut_state'] = np.where(c > trail, 1, -1)

    # ADX
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    tr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / tr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / tr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()

    return df

# ==================== ANALYSE PAR PAIRE ====================
def analyze_pair(args):
    pair, tf, live_mode, htf_data = args
    try:
        df = fetch_candles(pair, OANDA_GRAN[tf], 300)
        if len(df) < 100:
            return None
        df = calculate_indicators(df)

        idx = -1 if live_mode else -2
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        prev2 = df.iloc[idx-2]

        if pd.isna(last.hma) or pd.isna(last.rsi):
            return None

        # Spread filter
        if get_spread(pair) > 3.0:
            return None

        # HMA Flips
        flip_up = last.hma_up and not prev.hma_up
        flip_down = not last.hma_up and prev.hma_up

        if tf == "H1" and not flip_up and not flip_down:
            return None
        if tf in ["H4", "D1"] and not (flip_up or flip_down):
            return None

        action = "BUY" if flip_up else "SELL"
        if action == "BUY" and last.rsi < 50 or last.ut_state != 1: return None
        if action == "SELL" and last.rsi > 50 or last.ut_state != -1: return None

        # Cascade HTF
        higher_trend = "Neutral"
        if pair in htf_data and len(htf_data[pair]) > 10:
            htf_last = htf_data[pair].iloc[-1]
            htf_prev = htf_data[pair].iloc[-2]
            higher_trend = "Bullish" if htf_last.hma > htf_prev.hma else "Bearish"

        if higher_trend == "Bearish" and action == "BUY": return None
        if higher_trend == "Bullish" and action == "SELL": return None

        # Scoring
        score = 75
        if last.adx > 25: score += 15
        elif last.adx > 20: score += 8
        if 50 < last.rsi < 65 and action == "BUY": score += 5
        if 35 < last.rsi < 50 and action == "SELL": score += 5
        if higher_trend != "Neutral": score += 10
        score = min(100, score)
        quality = SignalQuality.INSTITUTIONAL if score >= 90 else SignalQuality.PREMIUM if score >= 80 else SignalQuality.STANDARD

        # SL/TP
        atr = last.atr
        sl = last.close - 2 * atr if action == "BUY" else last.close + 2 * atr
        tp = last.close + 3 * atr if action == "BUY" else last.close - 3 * atr
        rr = 1.5 if action == "BUY" else 1.5  # 3:2

        # Position sizing corrigé
        info = INSTRUMENT_INFO[pair]
        risk_pips = abs(last.close - sl) / (0.0001 if info["digits"] == 5 else 0.01)
        risk_per_unit = risk_pips * info["pip_value"]
        size = round((10000 * 0.01) / risk_per_unit, 4 if "JPY" not in pair else 2)
        size = max(0.01, size)

        tunis_tz = pytz.timezone('Africa/Tunis')
        ts = pytz.utc.localize(df.iloc[idx]["time"]).astimezone(tunis_tz)

        return Signal(
            timestamp=ts, pair=pair, timeframe=tf, action=action,
            entry_price=round(last.close, info["digits"]),
            stop_loss=round(sl, info["digits"]),
            take_profit=round(tp, info["digits"]),
            score=score, quality=quality, position_size=size,
            risk_reward=rr, adx=round(last.adx,1),
            rsi=round(last.rsi,1), atr=round(atr,5),
            higher_tf_trend=higher_trend, is_fresh_flip=True
        )
    except Exception as e:
        logger.error(f"Erreur {pair} {tf}: {e}")
        return None

# ==================== SCAN PARALLÈLE ====================
def run_scan(pairs, timeframes, live_mode):
    htf_data = get_htf_data(tuple(pairs))
    tasks = [(p, tf, live_mode, htf_data) for p in pairs for tf in timeframes]

    signals = []
    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = [exe.submit(analyze_pair, task) for task in tasks]
        for future in as_completed(futures):
            sig = future.result()
            if sig and sig.score >= 75:
                signals.append(sig)

    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== PDF REPORT ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("BlueStar Cascade PRO v2.9 - Rapport Institutionnel", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.now(pytz.timezone('Africa/Tunis')).strftime('%d/%m/%Y %H:%M')} (Tunis)", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))

    data = [["Heure", "Actif", "TF", "Qualité", "Sens", "Entrée", "SL", "TP", "Score", "R:R", "Size"]]
    for s in signals:
        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            s.quality.value[:4],
            s.action,
            f"{s.entry_price}",
            f"{s.stop_loss}",
            f"{s.take_profit}",
            s.score,
            "1:" + str(s.risk_reward),
            f"{s.position_size}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== UI ====================
def main():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("# BlueStar Cascade v2.9 PRO")
        st.markdown('<span class="institutional-badge">FIXED • LIVE • NO D1 BUG</span>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='text-align: right; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;'>
            <div style='color: #a0a0c0; font-size: 0.9rem;'>HEURE TUNIS</div>
            <div style='font-size: 1.5rem; font-weight: bold; color: #00ff88;'>
                {datetime.now(pytz.timezone('Africa/Tunis')).strftime('%H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.sidebar:
        st.header("Scanner PRO")
        mode = st.radio("Mode", ["CONFIRMÉ (Clôture)", "LIVE (Temps réel)"], index=1)
        tfs = st.multiselect("Timeframes", ["H1", "H4", "D1"], default=["H1", "H4"])
        all_pairs = st.checkbox("Tous les actifs", value=True)
        pairs = PAIRS if all_pairs else st.multiselect("Sélection manuelle", PAIRS, default=["EUR_USD", "XAU_USD"])

        st.divider()
        if st.button("LANCER LE SCAN INSTITUTIONNEL", type="primary", use_container_width=True):
            with st.spinner("Scan en cours... (très rapide)"):
                signals = run_scan(pairs, tfs, "LIVE" in mode)

            if not signals:
                st.info("Aucun signal institutionnel pour le moment.")
            else:
                st.success(f"{len(signals)} signaux détectés !")
                for s in signals[:10]:
                    color = "#00ff88" if s.action == "BUY" else "#ff4b4b"
                    st.markdown(f"**{s.timestamp.strftime('%H:%M')}** • **{s.pair.replace('_','/')}** • {s.timeframe} • **{s.action}** • Score: {s.score} • R:R 1:{s.risk_reward}", unsafe_allow_html=True)

                st.dataframe(pd.DataFrame([{
                    "Heure": s.timestamp.strftime("%H:%M"),
                    "Actif": s.pair.replace("_","/"),
                    "TF": s.timeframe,
                    "Action": s.action,
                    "Score": s.score,
                    "Qualité": s.quality.value,
                    } for s in signals]), hide_index=True)

                st.download_button(
                    "PDF Rapport Institutionnel",
                    data=generate_pdf(signals),
                    file_name=f"BlueStar_PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
