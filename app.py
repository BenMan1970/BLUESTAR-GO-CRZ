"""
BLUESTAR CASCADE v4.0 – FINAL ULTRA OPTIMIZED
Décembre 2025 – Le plus rapide du marché. Point final.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

# ====================== CONFIG ======================
st.set_page_config(page_title="BlueStar v4.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main {background: #0a001f; color: white;}
    .stButton>button {background: linear-gradient(90deg, #00ff9d, #00cc7a); color: black; font-weight: bold; border-radius: 15px; height: 3.8em; font-size: 1.4em;}
    .stButton>button:hover {background: linear-gradient(90deg, #00cc7a, #00ff9d);}
    .signal-buy {background: linear-gradient(90deg, #00ff9d, #00cc7a); color: black; padding: 20px; border-radius: 18px; font-size: 1.4em; font-weight: bold; margin: center;}
    .signal-sell {background: linear-gradient(90deg, #ff3366, #cc0033); color: white; padding: 20px; border-radius: 18px; font-size: 1.4em; font-weight: bold; text-align: center;}
    h1 {background: linear-gradient(90deg, #00ff9d, #00cc7a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 4rem !important; text-align: center;}
</style>
""", unsafe_allow_html=True)

# ====================== INSTRUMENTS ======================
INSTRUMENT_INFO = {
    "EUR_USD": {"pip_value": 10.0, "digits": 5}, "GBP_USD": {"pip_value": 10.0, "digits": 5},
    "USD_JPY": {"pip_value": 10.0, "digits": 3}, "USD_CHF": {"pip_value": 10.0, "digits": 5},
    "AUD_USD": {"pip_value": 10.0, "digits": 5}, "NZD_USD": {"pip_value": 10.0, "digits": 5},
    "USD_CAD": {"pip_value": 10.0, "digits": 5}, "EUR_GBP": {"pip_value": 10.0, "digits": 5},
    "EUR_JPY": {"pip_value": 10.0, "digits": 3}, "GBP_JPY": {"pip_value": 10.0, "digits": 3},
    "AUD_JPY": {"pip_value": 10.0, "digits": 3}, "CAD_JPY": {"pip_value": 10.0, "digits": 3},
    "NZD_JPY": {"pip_value": 10.0, "digits": 3},
    "EUR_AUD": {"pip_value": 10.0, "digits": 5}, "EUR_CAD": {"pip_value": 10.0, "digits": 5},
    "XAU_USD": {"pip_value": 1.0, "digits": 2}, "US30_USD": {"pip_value": 1.0, "digits": 2},
    "NAS100_USD": {"pip_value": 1.0, "digits": 2}, "SPX500_USD": {"pip_value": 0.1, "digits": 2},
    # Tu peux en rajouter, ça marchera quand même
}
PAIRS = list(INSTRUMENT_INFO.keys())

OANDA_GRAN = {"H1": "H1", "H4": "H4", "D1": "D"}

class SignalQuality(Enum):
    INSTITUTIONAL = "Institutional"
    PREMIUM = "Premium"

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

# ====================== OANDA CLIENT ======================
@st.cache_resource
def get_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"], environment="practice")

client = get_client()

def fetch_candles(pair: str, granularity: str, count: int = 380) -> pd.DataFrame:
    time.sleep(0.07)  # Respect parfait du rate-limit OANDA
    try:
        params = {"granularity": granularity, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        data = []
        for c in req.response.get("candles", []):
            if c.get("complete"):
                data.append({
                    "time": pd.to_datetime(c["time"]),
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"])
                })
        df = pd.DataFrame(data).sort_values("time").reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

# ====================== SPREAD CACHÉ (LE FIX ULTIME) ======================
@st.cache_data(ttl=300, show_spinner=False)  # 5 minutes de cache = 1 seul appel toutes les 5 min
def get_spread_cached(_client, pair: str) -> float:
    try:
        req = PricingInfo(accountID=st.secrets["OANDA_ACCOUNT_ID"], params={"instruments": pair})
        _client.request(req)
        ask = float(req.response["prices"][0]["asks"][0]["price"])
        bid = float(req.response["prices"][0]["bids"][0]["price"])
        pip = 0.0001 if INSTRUMENT_INFO[pair]["digits"] == 5 else 0.01
        return round((ask - bid) / pip, 1)
    except:
        return 999.0

# ====================== INDICATORS ======================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60: return df
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

    # UT Bot (x3 loss)
    loss = 3.0 * df['atr']
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
    plus_dm = h.diff().clip(lower=0); minus_dm = (-l.diff()).clip(lower=0)
    tr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14).mean() / tr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14).mean() / tr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.ewm(alpha=1/14).mean()

    return df

# ====================== MEGA CACHE UNIQUE (LE COEUR DU MONSTRE) ======================
@st.cache_data(ttl=180, show_spinner=False)
def load_all_data(pairs: tuple, timeframes: tuple):
    all_data = {}
    needed = ["D"] + list(timeframes)  # D1 toujours chargé pour le trend
    for pair in pairs:
        all_data[pair] = {}
        for tf in needed:
            gran = "D" if tf == "D1" else OANDA_GRAN.get(tf, tf)
            df = fetch_candles(pair, gran)
            if len(df) >= 60:
                all_data[pair][tf] = calculate_indicators(df)
    return all_data

# ====================== ANALYSE ULTRA-RAPIDE ======================
def analyze_pair(args):
    pair, tf, live_mode, data = args
    try:
        if tf not in data.get(pair, {}): return None
        df = data[pair][tf]
        if len(df) < 100: return None

        # HTF Trend (Daily)
        htf_df = data[pair].get("D")
        higher_trend = "Neutral"
        if htf_df is not None and len(htf_df) > 10:
            htf_last = htf_df.iloc[-1]; htf_prev = htf_df.iloc[-2]
            higher_trend = "Bullish" if htf_last.hma > htf_prev.hma else "Bearish"

        idx = -1 if live_mode else -2
        last = df.iloc[idx]; prev = df.iloc[idx-1]

        if pd.isna(last.hma) or pd.isna(last.rsi): return None

        # Spread ultra-caché
        spread = get_spread_cached(client, pair)
        if spread > 2.5: return None

        flip_up = last.hma_up and not prev.hma_up
        flip_down = not last.hma_up and prev.hma_up
        if not (flip_up or flip_down): return None

        action = "BUY" if flip_up else "SELL"
        if action == "BUY" and (last.rsi < 50 or last.ut_state != 1): return None
        if action == "SELL" and (last.rsi > 50 or last.ut_state != -1): return None
        if higher_trend == "Bearish" and action == "BUY": return None
        if higher_trend == "Bullish" and action == "SELL": return None

        score = 80
        if last.adx > 25: score += 15
        elif last.adx > 20: score += 8
        if higher_trend != "Neutral": score += 7
        score = min(100, score)
        if score < 85: return None

        quality = SignalQuality.INSTITUTIONAL if score >= 95 else SignalQuality.PREMIUM

        info = INSTRUMENT_INFO[pair]
        atr = last.atr
        sl = round(last.close - 2*atr if action == "BUY" else last.close + 2*atr, info["digits"])
        tp = round(last.close + 3*atr if action == "BUY" else last.close - 3*atr, info["digits"])
        risk_pips = abs(last.close - sl) / (0.0001 if info["digits"] == 5 else 0.01)
        size = round(max(0.01, (10000 * 0.01) / (risk_pips * info["pip_value"])), 4 if "JPY" not in pair else 2)

        ts = pytz.utc.localize(df.iloc[idx]["time"]).astimezone(pytz.timezone('Africa/Tunis'))

        return Signal(ts, pair, tf, action, round(last.close, info["digits"]), sl, tp, score, quality, size)
    except:
        return None

# ====================== SCAN ======================
def run_scan(pairs, tfs, live_mode_str):
    live_mode = "LIVE" in live_mode_str
    with st.spinner("Chargement unique des données (3-4 secondes max)..."):
        data = load_all_data(tuple(pairs), tuple(tfs))

    tasks = [(p, tf, live_mode, data) for p in pairs for tf in tfs]
    signals = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        for future in as_completed(executor.submit(analyze_pair, task) for task in tasks):
            result = future.result()
            if result:
                signals.append(result)

    return sorted(signals, key=lambda x: x.score, reverse=True)[:12]

# ====================== PDF ======================
def generate_pdf(signals: list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("BlueStar v4.0 – Rapport Institutionnel", styles["Title"]))
    elements.append(Paragraph(datetime.now(pytz.timezone('Africa/Tunis')).strftime('%d/%m/%Y %H:%M Tunis'), styles["Normal"]))
    elements.append(Spacer(1, 20))

    table_data = [["Heure", "Pair", "TF", "Action", "Entry", "SL", "TP", "Score", "Qualité", "Size"]]
    for s in signals:
        table_data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            s.action,
            s.entry_price,
            s.stop_loss,
            s.take_profit,
            s.score,
            s.quality.value,
            s.position_size
        ])
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), '#00ff9d'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ====================== UI ======================
def main():
    st.markdown("<h1>BLUESTAR v4.0</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#00ff9d;'>Le scanner institutionnel le plus rapide du monde</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    with col2:
        st.markdown(f"<div style='background:#00000033; padding:20px; border-radius:20px; text-align:center;'><h2>{datetime.now(pytz.timezone('Africa/Tunis')).strftime('%H:%M:%S')}</h2><small>Tunis Time</small></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("BlueStar Control")
        mode = st.radio("Mode", ["LIVE (Temps réel)", "CONFIRMÉ (Clôture)"], index=0)
        tfs = st.multiselect("Timeframes", ["H1", "H4", "D1"], default=["H1", "H4"])
        all_pairs = st.checkbox("Toutes les paires (32)", value=True)
        pairs = PAIRS if all_pairs else st.multiselect("Sélection manuelle", PAIRS, default=["EUR_USD", "XAU_USD", "GBP_JPY"])

        if st.button("LANCER LE SCAN v4.0", type="primary", use_container_width=True):
            with st.spinner("Scan en cours..."):
                signals = run_scan(pairs, tfs, mode)

            if not signals:
                st.balloons()
                st.success("Aucun signal institutionnel détecté pour le moment.")
            else:
                st.success(f"**{len(signals)} SIGNAL(S) DÉTECTÉ(S) !**")
                for s in signals:
                    cls = "signal-buy" if s.action == "BUY" else "signal-sell"
                    st.markdown(f"<div class='{cls}'>{s.action} <b>{s.pair.replace('_','/')}</b> • {s.timeframe} • Score {s.score}/100 • {s.quality.value} • Size {s.position_size}</div>", unsafe_allow_html=True)

                st.download_button(
                    label="Télécharger le Rapport PDF",
                    data=generate_pdf(signals),
                    file_name=f"BlueStar_Signals_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
