import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz

# ==================== WEBHOOK SANS UVICORN (compatible Streamlit Cloud) ====================
# Méthode 2025 officielle : on utilise st.experimental_get_query_params + rerun
# Le webhook TradingView pointe vers : https://ton-app.streamlit.app/?signal=BUY&pair=EUR_USD&tf=H1&price=1.0850

def process_webhook():
    params = st.query_params
    sig = params.get("signal", [None])[0]
    if sig in ["BUY", "SELL"]:
        data = {
            "signal": sig,
            "pair": params.get("pair", ["EUR_USD"])[0],
            "tf": params.get("tf", [""])[0],
            "price": params.get("price", [""])[0],
            "time": params.get("time", [datetime.now().strftime("%Y-%m-%d %H:%M")])[0]
        }
        if not hasattr(st.session_state, "live_signals"):
            st.session_state.live_signals = []
        # Évite les doublons
        if not st.session_state.live_signals or st.session_state.live_signals[0] != data:
            st.session_state.live_signals.insert(0, data)
            st.session_state.live_signals = st.session_state.live_signals[:20]

process_webhook()  # Appel au démarrage et à chaque refresh
# =====================================================================================

st.set_page_config(page_title="Forex Scanner PRO", layout="wide")
st.title("Forex Multi-Timeframe Scanner PRO + Signaux Live TradingView")

# ==================== BANDEAU SIGNAUX EN DIRECT ====================
if hasattr(st.session_state, "live_signals") and st.session_state.live_signals:
    st.markdown("### SIGNAUX EN DIRECT (0 lag – via TradingView)")
    cols = st.columns(min(6, len(st.session_state.live_signals)))
    for i, sig in enumerate(st.session_state.live_signals[:6]):
        with cols[i]:
            color = "green" if sig["signal"] == "BUY" else "red"
            emoji = "ACHAT" if sig["signal"] == "BUY" else "VENTE"
            pair = sig["pair"].replace("_", "/")
            st.markdown(f"**<span style='color:{color}'>{emoji} {pair} {sig['tf']}</span>**", unsafe_allow_html=True)
            st.caption(f"{sig['price']} | {sig['time'][11:16]}")
    st.markdown("---")
# ====================================================================

PAIRS_DEFAULT = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
                 "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
                 "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
                 "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"]

GRANULARITY_MAP = {"H1":"H1", "H4":"H4", "D1":"D", "W":"W"}

@st.cache_resource
def get_oanda_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"]), st.secrets["OANDA_ACCOUNT_ID"]

try:
    client, _ = get_oanda_client()
except Exception as e:
    st.error(f"Connexion OANDA échouée : {e}")
    st.stop()

@st.cache_data(ttl=30)
def get_candles(pair: str, tf: str, count: int = 200, include_incomplete: bool = False):
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

def hma(s, length=20):
    return s.rolling(window=length).apply(
        lambda x: 2 * pd.Series(x).rolling(int(length/2)).mean().iloc[-1] - pd.Series(x).rolling(length).mean().iloc[-1], raw=False
    ).rolling(int(np.sqrt(length))).mean()

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
def check_mtf_trend(pair, tf):
    higher = {"H1":"H4", "H4":"D1", "D1":"W"}.get(tf)
    if not higher: return {"trend":"neutral", "strength":0}
    df = get_candles(pair, higher, 100)
    if len(df)<40: return {"trend":"neutral", "strength":0}
    ema20 = df.close.ewm(span=20).mean().iloc[-1]
    ema50 = df.close.ewm(span=50).mean().iloc[-1]
    price = df.close.iloc[-1]
    dist = abs(ema20-ema50)/ema50*100
    if ema20>ema50 and price>ema20:
        return {"trend":"bullish","strength":min(dist*10,100)}
    if ema20<ema50 and price<ema20:
        return {"trend":"bearish","strength":min(dist*10,100)}
    return {"trend":"neutral","strength":0}

def analyze_pair(pair, tf, count=200, back=3):
    df = get_candles(pair, tf, count, include_incomplete=True)
    if len(df)<30: return None
    df["hma20"] = hma(df.close)
    df["rsi7"] = rsi(df.close)
    df["atr14"] = atr(df)
    df["hma_up"] = df.hma20 > df.hma20.shift(1)

    last = df.iloc[-1]
    recent = df.tail(back)

    hma_bull = any((recent.hma_up.iloc[i] and not recent.hma_up.iloc[i-1]) for i in range(1,len(recent)))
    hma_bear = any((not recent.hma_up.iloc[i] and recent.hma_up.iloc[i-1]) for i in range(1,len(recent)))

    mtf = check_mtf_trend(pair, tf)
    buy  = (hma_bull or last.hma_up) and last.rsi7>50 and mtf["trend"]=="bullish"
    sell = (hma_bear or not last.hma_up) and last.rsi7<50 and mtf["trend"]=="bearish"
    if not (buy or sell): return None

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
        "R:R": f"1:{round(1.5,1)}",
        "RSI": round(last.rsi7,1),
        "Heure": last.time.strftime("%H:%M"),
        "_confidence_val": conf
    }

# ==================== SCAN CORRIGÉ (max_workers remis) ====================
def scan_parallel(pairs, tfs, count, max_workers=6, back=3):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_pair, p, tf, count, back): (p,tf) for p in pairs for tf in tfs}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
    return results
# ====================================================================

# ==================== INTERFACE ====================
st.sidebar.header("Configuration")
selected_pairs = st.sidebar.multiselect("Paires", PAIRS_DEFAULT, default=PAIRS_DEFAULT, key="pairs")
selected_tfs = st.sidebar.multiselect("TF", ["H1","H4","D1"], default=["H1","H4","D1"])
min_conf = st.sidebar.slider("Confiance min",0,100,20)
scan_btn = st.sidebar.button("LANCER SCAN", type="primary")

if scan_btn or st.sidebar.checkbox("Auto-refresh 5min"):
    with st.spinner("Scan en cours…"):
        results = scan_parallel(selected_pairs or PAIRS_DEFAULT, selected_tfs, 200, max_workers=6)
        results = [r for r["_confidence_val"] >= min_conf for r in results]

    if results:
        for tf in ["H1","H4","D1"]:
            tf_res = [r for r in results if r["TF"]==tf]
            if tf_res:
                tf_res.sort(key=lambda x: x["_confidence_val"], reverse=True)
                st.markdown(f"### {tf} – {len(tf_res)} signaux")
                df_show = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tf_res])
                st.dataframe(df_show.style.apply(lambda row: ['background: rgba(0,255,0,0.15)' if 'ACHAT' in row.Signal else 'background: rgba(255,0,0,0.15)'], axis=1),
                             use_container_width=True, hide_index=True)
    else:
        st.info("Aucun signal pour l’instant")
else:
    st.info("Clique sur LANCER SCAN")
