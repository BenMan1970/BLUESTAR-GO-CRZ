import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz

# ==================== WEBHOOK TRADINGVIEW (0 lag) ====================
from fastapi import FastAPI, Request
import threading
import uvicorn

webapp = FastAPI()

@webapp.post("/webhook")
async def live_signal(request: Request):
    try:
        data = await request.json()
        sig = data.get("signal")
        if sig in ["BUY", "SELL"]:
            if not hasattr(st.session_state, "live_signals"):
                st.session_state.live_signals = []
            st.session_state.live_signals.insert(0, data)
            st.session_state.live_signals = st.session_state.live_signals[:20]
            st.rerun()
    except:
        pass

threading.Thread(target=lambda: uvicorn.run(webapp, host="0.0.0.0", port=8000), daemon=True).start()
# ====================================================================

st.set_page_config(page_title="Forex Scanner PRO", layout="wide")
st.title("Forex Multi-Timeframe Scanner PRO + Signaux Live TradingView")

# ==================== BANDEAU SIGNAUX EN TEMPS RÉEL ====================
if hasattr(st.session_state, "live_signals") and st.session_state.live_signals:
    st.markdown("### SIGNAUX EN DIRECT (identiques à TradingView)")
    cols = st.columns(min(6, len(st.session_state.live_signals)))
    for i, sig in enumerate(st.session_state.live_signals[:6]):
        with cols[i]:
            color = "green" if sig["signal"] == "BUY" else "red"
            emoji = "ACHAT" if sig["signal"] == "BUY" else "VENTE"
            pair = sig.get("pair", "EURUSD").replace("_", "/")
            st.markdown(f"**<span style='color:{color}'>{emoji} {pair} {sig.get('tf','')}</span>**", 
                        unsafe_allow_html=True)
            st.caption(f"{sig.get('price','')} | {sig.get('time','')[11:16]}")
    st.markdown("---")
# ====================================================================

PAIRS_DEFAULT = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
                 "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
                 "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
                 "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ==================== OANDA CLIENT ====================
@st.cache_resource
def get_oanda_client() -> Tuple[API, str]:
    token = st.secrets["OANDA_ACCESS_TOKEN"]
    account_id = st.secrets["OANDA_ACCOUNT_ID"]
    return API(access_token=token), account_id

try:
    client, ACCOUNT_ID = get_oanda_client()
except Exception as e:
    st.error(f"Erreur OANDA : {e}")
    st.stop()

# ==================== FONCTIONS CANDLES & INDICATEURS ====================
@st.cache_data(ttl=30)
def get_candles(pair: str, tf: str, count: int = 200, include_incomplete: bool = False) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran: return pd.DataFrame()
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        candles = req.response.get("candles", [])
        records = []
        for c in candles:
            if not include_incomplete and not c.get("complete", True):
                continue
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c.get("volume", 0))
            })
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except:
        return pd.DataFrame()

def wma(s, length): 
    w = np.arange(1, length + 1)
    return s.rolling(length).apply(lambda x: np.dot(x, w) / w.sum() if len(x)==length else np.nan, raw=True)

def hma(s, length=20):
    return wma(2 * wma(s, length//2) - wma(s, length), int(np.sqrt(length)))

def rsi(s, length=7):
    delta = s.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length).mean()
    roll_down = down.ewm(alpha=1/length).mean()
    rs = roll_up / roll_down
    return 100 - 100/(1+rs)

def atr(df, length=14):
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(), (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length).mean()

@st.cache_data(ttl=120)
def check_mtf_trend(pair: str, tf: str) -> Dict:
    higher = {"H1":"H4", "H4":"D1", "D1":"W"}.get(tf)
    if not higher: return {"trend":"neutral","strength":0}
    df = get_candles(pair, higher, 100)
    if len(df)<50: return {"trend":"neutral","strength":0}
    ema20 = df.close.ewm(span=20).mean().iloc[-1]
    ema50 = df.close.ewm(span=50).mean().iloc[-1]
    price = df.close.iloc[-1]
    dist = abs(ema20-ema50)/ema50*100
    if ema20>ema50 and price>ema20:
        return {"trend":"bullish", "strength":min(dist*10,100)}
    if ema20<ema50 and price<ema20:
        return {"trend":"bearish", "strength":min(dist*10,100)}
    return {"trend":"neutral","strength":0}

# ==================== ANAL
