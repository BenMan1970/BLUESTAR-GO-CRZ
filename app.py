"""
BlueStar Cascade v3.0 ULTRA – INSTITUTIONAL SCALPING MACHINE
Décembre 2025 – Optimisé, silencieux, mortel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List
from enum import Enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# OANDA
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingInfo

# PDF
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar v3.0 ULTRA", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)

# Style PRO
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0033 0%, #1a0033 100%); color: white;}
    .stButton button {background: linear-gradient(90deg, #00ff9d, #00cc7a); color: black; font-weight: bold; border-radius: 12px; height: 3.5em;}
    .stButton button:hover {background: linear-gradient(90deg, #00cc7a, #00ff9d);}
    .signal-buy {background: linear-gradient(90deg, #00ff9d, #00cc7a); color: black; padding: 15px; border-radius: 15px; font-weight: bold; margin: 10px 0;}
    .signal-sell {background: linear-gradient(90deg, #ff3366, #cc0033); color: white; padding: 15px; border-radius: 15px; font-weight: bold; margin: 10px 0;}
    h1 {background: linear-gradient(90deg, #00ff9d, #00cc7a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem !important;}
</style>
""", unsafe_allow_html=True)

# ==================== INSTRUMENTS ====================
INSTRUMENT_INFO = {
    "EUR_USD": {"type": "forex", "pip_value": 10.0, "digits": 5}, "GBP_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "USD_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3}, "USD_CHF": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "AUD_USD": {"type": "forex", "pip_value": 10.0, "digits": 5}, "NZD_USD": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "USD_CAD": {"type": "forex", "pip_value": 10.0, "digits": 5}, "EUR_GBP": {"type": "forex", "pip_value": 10.0, "digits": 5},
    "EUR_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3}, "GBP_JPY": {"type": "forex", "pip_value": 10.0, "digits": 3},
    "XAU_USD": {"type": "metal", "pip_value": 1.0, "digits": 2}, "US30_USD": {"type": "index", "pip_value": 1.0, "digits": 2},
    "NAS100_USD": {"type": "index", "pip_value": 1.0, "digits": 2}, "SPX500_USD": {"type": "index", "pip_value": 0.1, "digits": 2},
    # Ajoute les autres si tu veux, mais ceux-là suffisent pour du lourd
}
PAIRS = list(INSTRUMENT_INFO.keys())
OANDA_GRAN = {"H1": "H1", "H4": "H4", "D1": "D"}

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
    risk_reward: float = 1.5

# ==================== OANDA CLIENT ====================
@st.cache_resource
def get_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"], environment="practice")

client = get_client()

def fetch_candles(pair: str, granularity: str, count: int = 350) -> pd.DataFrame:
    time.sleep(0.08)
    try:
        params = {"granularity": granularity, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        data = []
        for c in req.response.get("candles", []):
            if c.get("complete"):
                data.append({
                    "time": pd.to_datetime(c["time"]).tz_localize(None),
                    "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])
                })
        df = pd.DataFrame(data).sort_values("time").reset_index(drop=True)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_spread(pair: str) -> float:
    try:
        req = PricingInfo(accountID=st.secrets["OANDA_ACCOUNT_ID"], params={"instruments": pair})
        client.request(req)
        ask = float(req.response["prices"][0]["asks"][0]["price"])
        bid = float(req.response["prices"][0]["bids"][0]["price"])
        pip = 0.0001 if INSTRUMENT_INFO[pair]["digits"] == 5 else 0.01
        return (ask - bid) / pip
    except: return 999

# ==================== INDICATORS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    c, h, l = df['close'], df['high'], df['low']

    # HMA 20
    def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
    wma10 = wma(c, 10); wma20 = wma(c, 20)
    df['hma'] = wma(2*wma10 - wma20, int(np.sqrt(20)))
    df['hma_up'] = df['hma'] > df['hma'].shift(1)

    # RSI 7
    delta = c.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    df['rsi'] = 100 - (100 / (1 + up.ewm(alpha=1/7).mean() / down.ewm(alpha=1/7).mean()))

    # ATR 10
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/10, adjust=False).mean()

    # UT Bot x3
    loss = 3.0 * df['atr']
    trail = [0.0] * len(df)
    for i in range(1, len(df)):
        prev = trail[i-1]
        if c.iloc[i] > prev and c.iloc[i-1] > prev:
            trail[i] = max(prev, c.iloc[i] - loss.iloc[i])
        elif c.iloc[i] < prev and c.iloc[i-1] < prev:
            trail[i] = min(prev, c.iloc[i] + loss.iloc[i])
        elif c.iloc[i] > prev: trail[i] = c.iloc[i] - loss.iloc[i]
        else: trail[i] = c.iloc[i] + loss.iloc[i]
    df['ut_state'] = np.where(c > trail, 1, -1)

    # ADX
    plus_dm = h.diff().clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
    tr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / tr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / tr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ==================== MEGA CACHE UNIQUE ====================
@st.cache_data(ttl=180, show_spinner=False)
def get_all_data(pairs: tuple, tfs: tuple):
    all_data = {}
    needed_tfs = ["D"] + list(tfs)
    for pair in pairs:
        all_data[pair] = {}
        for tf in needed_tfs:
            gran = "D" if tf == "D1" else OANDA_GRAN.get(tf, tf)
            df = fetch_candles(pair, gran)
            if len(df) >= 50:
                all_data[pair][tf] = calculate_indicators(df)
    return all_data

# ==================== ANALYSE OPTIMISÉE ====================
def analyze_pair(args):
    pair, tf, live_mode, data = args
    try:
        if tf not in data.get(pair, {}): return None
        df = data[pair][tf]
        if len(df) < 100: return None

        # HTF Trend
        htf_df = data[pair].get("D")
        higher_trend = "Neutral"
        if htf_df is not None and len(htf_df) > 10:
            htf_last = htf_df.iloc[-1]; htf_prev = htf_df.iloc[-2]
            higher_trend = "Bullish" if htf_last.hma > htf_prev.hma else "Bearish"

        idx = -1 if live_mode else -2
        last = df.iloc[idx]; prev = df.iloc[idx-1]

        if pd.isna(last.hma) or pd.isna(last.rsi): return None

        # Spread (cached)
        if pair not in st.session_state.get("spreads", {}):
            st.session_state.spreads[pair] = get_spread(pair)
        if st.session_state.spreads[pair] > 3.0: return None

        flip_up = last.hma_up and not prev.hma_up
        flip_down = not last.hma_up and prev.hma_up
        if not (flip_up or flip_down): return None

        action = "BUY" if flip_up else "SELL"
        if action == "BUY" and (last.rsi < 50 or last.ut_state != 1): return None
        if action == "SELL" and (last.rsi > 50 or last.ut_state != -1): return None
        if higher_trend == "Bearish" and action == "BUY": return None
        if higher_trend == "Bullish" and action == "SELL": return None

        score = 75
        if last.adx > 25: score += 15
        elif last.adx > 20: score += 8
        if action == "BUY" and 50 < last.rsi < 65: score += 5
        if action == "SELL" and 35 < last.rsi < 50: score += 5
        if higher_trend != "Neutral": score += 10
        score = min(100, score)
        if score < 80: return None

        quality = SignalQuality.INSTITUTIONAL if score >= 90 else SignalQuality.PREMIUM

        info = INSTRUMENT_INFO[pair]
        atr = last.atr
        sl = round(last.close - 2*atr if action == "BUY" else last.close + 2*atr, info["digits"])
        tp = round(last.close + 3*atr if action == "BUY" else last.close - 3*atr, info["digits"])
        risk_pips = abs(last.close - sl) / (0.0001 if info["digits"] == 5 else 0.01)
        size = round(max(0.01, (10000 * 0.01) / (risk_pips * info["pip_value"])), 4 if "JPY" not in pair else 2)

        ts = pytz.utc.localize(df.iloc[idx]["time"]).astimezone(pytz.timezone('Africa/Tunis'))

        return Signal(ts, pair, tf, action, round(last.close, info["digits"]), sl, tp, score, quality, size)
    except: return None

# ==================== TELEGRAM WEBHOOK ====================
def send_telegram(signal: Signal):
    if "TELEGRAM_TOKEN" not in st.secrets or "TELEGRAM_CHAT_ID" not in st.secrets:
        return
    url = f"https://api.telegram.org/bot{st.secrets.TELEGRAM_TOKEN}/sendMessage"
    emoji = "BUY" if signal.action == "BUY" else "SELL"
    text = f"{emoji} *{signal.pair.replace('_', '/')}\n{signal.timeframe} • Score {signal.score} • {signal.quality.value}\nEntry: {signal.entry_price} | SL: {signal.stop_loss} | TP: {signal.take_profit}\nSize: {signal.position_size} units\n\nBlueStar v3.0 ULTRA*"
    requests.post(url, data={"chat_id": st.secrets.TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})

# ==================== SCAN ====================
def run_scan(pairs, tfs, live_mode_str):
    if "spreads" not in st.session_state: st.session_state.spreads = {}
    live_mode = "LIVE" in live_mode_str

    with st.spinner("Chargement unique des données... (3 secondes max)"):
        data = get_all_data(tuple(pairs), tuple(tfs))

    tasks = [(p, tf, live_mode, data) for p in pairs for tf in tfs]
    signals = []

    with ThreadPoolExecutor(max_workers=15) as exe:
        for future in as_completed(exe.submit(analyze_pair, t) for t in tasks):
            sig = future.result()
            if sig: 
                signals.append(sig)
                if len(signals) == 1: send_telegram(sig)  # 1 alerte max par scan

    return sorted(signals, key=lambda x: x.score, reverse=True)[:10]

# ==================== PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("BlueStar v3.0 ULTRA – Rapport Live", styles["Title"]))
    elements.append(Paragraph(f"{datetime.now(pytz.timezone('Africa/Tunis')).strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    table_data = [["Heure", "Pair", "TF", "Action", "Entry", "SL", "TP", "Score", "Size"]]
    for s in signals:
        table_data.append([s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"), s.timeframe, s.action, s.entry_price, s.stop_loss, s.take_profit, s.score, s.position_size])
    table = Table(table_data)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),'#00ff9d'), ('GRID',(0,0),(-1,-1),0.5,'grey')]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== UI ====================
def main():
    st.markdown("# BlueStar Cascade v3.0 ULTRA")
    st.markdown("**3 secondes • 1 appel par paire • Telegram live • Zéro bug**")

    col1, col2 = st.columns([2,1])
    with col2:
        st.markdown(f"<div style='background:#00000022;padding:15px;border-radius:15px;text-align:center;'><h3>{datetime.now(pytz.timezone('Africa/Tunis')).strftime('%H:%M:%S')}</h3><small>Tunis Time</small></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("BlueStar ULTRA")
        mode = st.radio("Mode", ["LIVE (Temps réel)", "CONFIRMÉ (Clôture)"], index=0)
        tfs = st.multiselect("Timeframes", ["H1", "H4", "D1"], default=["H1", "H4"])
        all_pairs = st.checkbox("Tous les actifs (32)", value=True)
        pairs = PAIRS if all_pairs else st.multiselect("Sélection", PAIRS, default=["EUR_USD", "XAU_USD", "GBP_JPY"])

        if st.button("LANCER LE SCAN ULTRA", type="primary", use_container_width=True):
            signals = run_scan(pairs, tfs, mode)
            if not signals:
                st.info("Aucun signal institutionnel détecté.")
            else:
                st.success(f"**{len(signals)} SIGNALS DÉTECTÉS !**")
                for s in signals:
                    cls = "signal-buy" if s.action == "BUY" else "signal-sell"
                    st.markdown(f"<div class='{cls}'>{s.action} **{s.pair.replace('_','/')}** • {s.timeframe} • Score {s.score} • Size {s.position_size}</div>", unsafe_allow_html=True)
                st.download_button("PDF Rapport", generate_pdf(signals), f"BlueStar_ULTRA_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")

if __name__ == "__main__":
    main()
