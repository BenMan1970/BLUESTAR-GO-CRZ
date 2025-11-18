# app.py → Version ULTIME (tout ce que tu aimais + encore mieux)
import streamlit as st
import pandas as pd
import numpy as np
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz

# ==================== WEBHOOK 0 LAG ====================
def process_webhook():
    params = st.query_params
    sig = params.get("signal", [None])[0]
    if sig in ["BUY", "SELL"]:
        data = {
            "signal": sig,
            "pair": params.get("pair", ["EUR_USD"])[0].replace("_", "/"),
            "tf": params.get("tf", [""])[0],
            "price": params.get("price", [""])[0],
            "time": params.get("time", [datetime.now().strftime("%H:%M")])[0]
        }
        if not hasattr(st.session_state, "live_signals"):
            st.session_state.live_signals = []
        if not st.session_state.live_signals or st.session_state.live_signals[0] != data:
            st.session_state.live_signals.insert(0, data)
            st.session_state.live_signals = st.session_state.live_signals[:20]

process_webhook()

st.set_page_config(page_title="Forex Scanner PRO", layout="wide")
st.title("Forex Scanner PRO + Signaux Live TradingView")

# ==================== BANDEAU SIGNAUX LIVE ====================
if hasattr(st.session_state, "live_signals") and st.session_state.live_signals:
    st.markdown("### SIGNAUX EN DIRECT (0 lag TradingView)")
    cols = st.columns(min(6, len(st.session_state.live_signals)))
    for i, sig in enumerate(st.session_state.live_signals[:6]):
        with cols[i]:
            color = "green" if sig["signal"] == "BUY" else "red"
            emoji = "ACHAT" if sig["signal"] == "BUY" else "VENTE"
            st.markdown(f"**<span style='color:{color}'>{emoji} {sig['pair']} {sig['tf']}</span>**", unsafe_allow_html=True)
            st.caption(f"{sig['price']} | {sig['time']}")
    st.markdown("---")

# ==================== CONFIG ====================
PAIRS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
         "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
         "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
         "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"]

client = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])

@st.cache_data(ttl=30)
def get_candles(pair, tf, count=250):
    gran = {"H1":"H1", "H4":"H4", "D1":"D"}.get(tf, tf)
    req = InstrumentsCandles(instrument=pair, params={"count": count, "granularity": gran, "price": "M"})
    client.request(req)
    data = []
    for c in req.response.get("candles", []):
        data.append({
            "time": c["time"],
            "o": float(c["mid"]["o"]),
            "h": float(c["mid"]["h"]),
            "l": float(c["mid"]["l"]),
            "c": float(c["mid"]["c"]),
            "complete": c.get("complete", False)
        })
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    return df

def hma(s, length=20):
    return (2 * s.rolling(length//2).mean() - s.rolling(length).mean()).rolling(int(np.sqrt(length))).mean()

def rsi(s, length=7):
    d = s.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    return 100 - 100/(1 + up.ewm(alpha=1/length).mean()/down.ewm(alpha=1/length).mean())

def atr(df, length=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length).mean()

@st.cache_data(ttl=120)
def mtf_trend(pair, tf):
    higher = {"H1":"H4", "H4":"D1", "D1":"W"}.get(tf, "W")
    df = get_candles(pair, higher, 100)
    if len(df) < 40: return "neutral", 0
    ema20 = df.c.ewm(span=20).mean().iloc[-1]
    ema50 = df.c.ewm(span=50).mean().iloc[-1]
    price = df.c.iloc[-1]
    dist = abs(ema20-ema50)/ema50*100
    if ema20 > ema50 and price > ema20: return "bullish", min(dist*10,100)
    if ema20 < ema50 and price < ema20: return "bearish", min(dist*10,100)
    return "neutral", 0

def analyze(pair, tf):
    df = get_candles(pair, tf, 250)
    if len(df) < 50: return None
    df["hma"] = hma(df.c)
    df["rsi"] = rsi(df.c)
    df["atr"] = atr(df)
    df["up"] = df.hma > df.hma.shift(1)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Détection du changement sur les 2 dernières bougies
    hma_became_bull = df["up"].iloc[-1] and not df["up"].iloc[-2]
    hma_became_bear = not df["up"].iloc[-1] and df["up"].iloc[-2]

    trend, strength = mtf_trend(pair, tf)

    buy  = (hma_became_bull or last.up) and last.rsi > 50 and trend == "bullish"
    sell = (hma_became_bear or not last.up) and last.rsi < 50 and trend == "bearish"

    if not (buy or sell): return None

    sl = last.c - 2*last.atr if buy else last.c + 2*last.atr
    tp = last.c + 3*last.atr if buy else last.c - 3*last.atr
    conf = round((abs(last.rsi-50)/50*100*0.4 + strength*0.6), 1)

    # Étoile jaune si signal sur bougie en cours
    star = "NEW" if not last.complete else ""

    return {
        "Instrument": pair.replace("_","/"),
        "TF": tf,
        "Signal": "ACHAT" if buy else "VENTE",
        "Confiance": conf,
        "Prix": round(last.c, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": "1:1.5",
        "RSI": round(last.rsi, 1),
        "Heure": last.time.strftime("%H:%M"),
        "Étoile": star,
        "_conf": conf,
        "_time": last.time
    }

def scan(pairs, tfs):
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(analyze, p, tf) for p in pairs for tf in tfs]):
            r = f.result()
            if r: results.append(r)
    return results

# ==================== SIDEBAR ====================
st.sidebar.header("Configuration")
pairs = st.sidebar.multiselect("Paires", PAIRS, default=PAIRS[:12])
tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], default=["H1","H4","D1"])
min_conf = st.sidebar.slider("Confiance min (%)", 0, 100, 15)
scan_btn = st.sidebar.button("LANCER LE SCAN", type="primary", use_container_width=True)

if scan прекрасно_btn:
    with st.spinner("Scan ultra-rapide…"):
        results = scan(pairs, tfs)
        results = [r for r in results if r["_conf"] >= min_conf]

    if results:
        # === TOP 5 + ÉTOILE JAUNE ===
        top5 = sorted(results, key=lambda x: x["_conf"], reverse=True)[:5]
        if top5:
            st.markdown("### TOP 5 SIGNAUX LES PLUS FORTS")
            for sig in top5:
                color = "🟢" if sig["Signal"] == "ACHAT" else "🔴"
                star = " NEW" if sig["Étoile"] else ""
                st.markdown(f"{color} **{sig['Instrument']} {sig['TF']}** → {sig['Signal']} | Confiance {sig['Confiance']}% | {sig['Prix']} {star}")

        # === AFFICHAGE PAR TF ===
        for tf in ["H1", "H4", "D1"]:
            tf_data = [r for r in results if r["TF"] == tf]
            if tf_data:
                tf_data.sort(key=lambda x: x["_conf"], reverse=True)
                st.markdown(f"### {tf} – {len(tf_data)} signal(s)")

                df_show = pd.DataFrame([{k:v for k,v in r.items() if k != "_conf" and k != "_time"} for r in tf_data])

                def color_row(row):
                    bg = "#d4edda" if "ACHAT" in row.Signal else "#f8d7da"  # vert clair / rouge clair
                    if row.Étoile: bg = "#fff3cd"  # jaune clair pour les signaux frais
                    return [f"background-color: {bg}; color: black;"] * len(row)

                st.dataframe(
                    df_show.style.apply(color_row, axis=1)
                    .set_properties(**{'text-align': 'center', 'font-size': '15px'}),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.success("Aucun signal pour le moment")
else:
    st.info("Clique sur **LANCER LE SCAN**")
