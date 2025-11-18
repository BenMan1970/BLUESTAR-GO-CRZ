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

# ==================== ANALYSE PAIRE ====================
def analyze_pair(pair: str, tf: str, count: int, back: int = 3) -> Optional[Dict]:
    df = get_candles(pair, tf, count, include_incomplete=True)
    if len(df)<30: return None
    df["hma20"] = hma(df.close)
    df["rsi7"]  = rsi(df.close)
    df["atr14"]  = atr(df)
    df["hma_up"] = df.hma20 > df.hma20.shift(1)

    last = df.iloc[-1]
    recent = df.tail(back)

    hma_bull = any((recent.hma_up.iloc[i] and not recent.hma_up.iloc[i-1]) for i in range(1,len(recent)))
    hma_bear = any((not recent.hma_up.iloc[i] and recent.hma_up.iloc[i-1]) for i in range(1,len(recent)))

    mtf = check_mtf_trend(pair, tf)
    buy  = (hma_bull or last.hma_up) and last.rsi7>50 and mtf["trend"]=="bullish"
    sell = (hma_bear or not last.hma_up) and last.rsi7<50 and mtf["trend"]=="bearish"
    if not (buy and not sell): return None

    atrv = last.atr14
    sl = last.close - 2*atrv if buy else last.close + 2*atrv
    tp = last.close + 3*atrv if buy else last.close - 3*atrv
    conf = round((abs(last.rsi7-50)/50*100*0.4 + mtf["strength"]*0.6),1)

    return {
        "Instrument": pair.replace("_","/"),
        "TF": tf,
        "Signal": "ACHAT" if buy else "VENTE",
        "Confiance": conf,
        "Prix": round(last.close,5),
        "SL": round(sl,5),
        "TP": round(tp,5),
        "R:R": f"1:{round(abs(tp-last.close)/abs(last.close-sl),1)}",
        "RSI": round(last.rsi7,1),
        "Tendance": mtf["trend"].upper(),
        "Force": f"{mtf['strength']}%",
        "Heure": last.time.strftime("%H:%M"),
        "_confidence_val": conf,
        "_time_raw": last.time
    }

# ==================== SCAN PARALLÈLE ====================
def scan_parallel(pairs, tfs, count, workers=5, back=3):
    results = []
    tasks = [(p,tf) for p in pairs for tf in tfs]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyze_pair, p, tf, count, back): (p,tf) for p,tf in tasks}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
    return results

# ==================== INTERFACE (ton code original) ====================
st.sidebar.header("Configuration du Scanner")

with st.sidebar.expander("Paires", expanded=False):
    selected_pairs = st.multiselect("Désélectionner pour retirer", PAIRS_DEFAULT, default=PAIRS_DEFAULT)

selected_tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], default=["H1","H4","D1"])
candles_count = st.sidebar.selectbox("Nombre de bougies", [100,150,200], index=2)
min_conf = st.sidebar.slider("Confiance min (%)", 0,100,20)
freshness = st.sidebar.selectbox("Fraîcheur signaux",["Dernière bougie","2 dernières","3 dernières","Toutes"], index=1)
auto = st.sidebar.checkbox("Auto-refresh toutes les 5 min")
scan_btn = st.sidebar.button("LANCER LE SCAN", type="primary", use_container_width=True)

if scan_btn or auto:
    freshness_map = {"Dernière bougie":1,"2 dernières":2,"3 dernières":3,"Toutes":999}
    back = freshness_map[freshness]
    with st.spinner("Scan en cours…"):
        results = scan_parallel(selected_pairs or PAIRS_DEFAULT, selected_tfs, candles_count, max_workers=6, back=back)
        results = [r for r in results if r["_confidence_val"] >= min_conf]
    
    if results:
        for tf in ["H1","H4","D1"]:
            tf_res = [r for r in results if r["TF"]==tf]
            if tf_res:
                tf_res.sort(key=lambda x: x["_confidence_val"], reverse=True)
                st.markdown(f"### {tf} — {len(tf_res)} signal(s)")
                df_show = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tf_res])
                st.dataframe(df_show.style.apply(lambda row: ['background: rgba(0,255,0,0.15)' if 'ACHAT' in row.Signal else 'background: rgba(255,0,0,0.15)' if 'VENTE' in row.Signal else '' for _ in row], axis=1),
                             use_container_width=True, hide_index=True)
    else:
        st.info("Aucun signal avec les critères actuels")

else:
    st.info("Configure et clique sur LANCER LE SCAN")
