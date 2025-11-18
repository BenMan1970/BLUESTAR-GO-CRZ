import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from typing import Tuple, List, Dict, Optional
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

st.set_page_config(page_title="Forex Scanner PRO (Live)", layout="wide")

# ----------------------
# Configuration
# ----------------------
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ----------------------
# OANDA API & Auth
# ----------------------
@st.cache_resource
def get_oanda_client() -> Tuple[API, str]:
    try:
        account_id = st.secrets.get("OANDA_ACCOUNT_ID")
        token = st.secrets.get("OANDA_ACCESS_TOKEN")
        if not account_id or not token:
            st.error("⚠️ Secrets OANDA manquants (.streamlit/secrets.toml)")
            st.stop()
        client = API(access_token=token)
        return client, account_id
    except Exception as e:
        st.error(f"Erreur critique API : {e}")
        st.stop()

client, ACCOUNT_ID = get_oanda_client()

# ----------------------
# Core Data Functions (MODIFIÉ POUR TEMPS RÉEL)
# ----------------------
@st.cache_data(ttl=10) # TTL réduit à 10s pour avoir les données fraiches
def get_candles(pair: str, tf: str, count: int = 150) -> pd.DataFrame:
    """
    Récupère les bougies OANDA.
    IMPORTANT : Inclut la dernière bougie non terminée (complete=False) pour le 0 Lag.
    """
    gran = GRANULARITY_MAP.get(tf)
    if gran is None: return pd.DataFrame()
    
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        candles = req.response.get("candles", [])
        
        records = []
        for c in candles:
            # ON PREND TOUT, même si c'est incomplet
            try:
                records.append({
                    "time": c["time"],
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "complete": c.get("complete", False), # Info cruciale pour savoir si c'est LIVE
                    "volume": int(c.get("volume", 0))
                })
            except (KeyError, ValueError):
                continue
        
        if not records: return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        return pd.DataFrame()

# ----------------------
# Indicateurs Mathématiques (Optimisés Numpy)
# ----------------------
def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    def weighted_mean(x):
        return np.dot(x, weights) / weights.sum() if len(x) == length else np.nan
    return series.rolling(length).apply(weighted_mean, raw=True)

def hma(series: pd.Series, length: int = 20) -> pd.Series:
    half = max(1, int(length / 2))
    sqrt_l = max(1, int(np.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrt_l)

def rsi(series: pd.Series, length: int = 7) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

@st.cache_data(ttl=60) # Tendance MTF peut être cachée un peu plus longtemps
def check_mtf_trend(pair: str, tf: str) -> Dict[str, any]:
    higher_map = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = higher_map.get(tf)
    
    if not higher_tf: return {"trend": "neutral", "strength": 0}
    
    # Pour la tendance de fond, on veut des bougies confirmées généralement, 
    # mais ici on garde la logique standard pour la rapidité.
    df = get_candles(pair, higher_tf, count=100)
    if df.empty or len(df) < 50: return {"trend": "neutral", "strength": 0}
    
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    
    dist = abs((ema20 - ema50) / ema50) * 100
    strength = min(dist * 10, 100)
    
    if ema20 > ema50 and price > ema20: return {"trend": "bullish", "strength": round(strength, 1)}
    if ema20 < ema50 and price < ema20: return {"trend": "bearish", "strength": round(strength, 1)}
    
    return {"trend": "neutral", "strength": 0}

# ----------------------
# Analyseur "0 Lag"
# ----------------------
def analyze_pair(pair: str, tf: str, candles_count: int) -> Optional[Dict]:
    # On charge les bougies (y compris l'incomplète)
    df = get_candles(pair, tf, count=candles_count)
    if df.empty or len(df) < 30: return None
    
    df = df.sort_values("time").reset_index(drop=True)
    
    # Calculs
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    df["atr14"] = atr(df, 14)
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)
    
    # --- ANALYSE SUR LA DERNIERE BOUGIE (LIVE) ---
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Statut LIVE
    is_live = not last["complete"]
    live_tag = "⚡" if is_live else "⭐" # Eclair = En cours, Etoile = Clôturé récent
    
    # Logique HMA (Changement de couleur ou continuité)
    hma_bull = last["hma_up"]
    hma_bear = not last["hma_up"]
    
    hma_flip_bull = hma_bull and not prev["hma_up"]
    hma_flip_bear = hma_bear and prev["hma_up"]
    
    # Logique RSI
    rsi_val = last["rsi7"]
    rsi_buy = rsi_val > 50
    rsi_sell = rsi_val < 50
    
    # Tendance MTF
    mtf = check_mtf_trend(pair, tf)
    
    # Condition d'Entrée
    # Signal valide si : (HMA vient de changer OU HMA est déjà dans le sens) ET RSI confirme ET MTF confirme
    signal_buy = (hma_flip_bull or hma_bull) and rsi_buy and mtf["trend"] == "bullish"
    signal_sell = (hma_flip_bear or hma_bear) and rsi_sell and mtf["trend"] == "bearish"
    
    final_signal = ""
    conf = 0
    
    if signal_buy:
        final_signal = f"ACHAT {live_tag}"
        strength_rsi = (rsi_val - 50) / 50 * 100
        conf = (strength_rsi * 0.4) + (mtf["strength"] * 0.6)
        # Bonus si c'est un retournement frais
        if hma_flip_bull: conf *= 1.1 

    elif signal_sell:
        final_signal = f"VENTE {live_tag}"
        strength_rsi = (50 - rsi_val) / 50 * 100
        conf = (strength_rsi * 0.4) + (mtf["strength"] * 0.6)
        # Bonus si c'est un retournement frais
        if hma_flip_bear: conf *= 1.1
        
    if not final_signal: return None
    
    # Niveaux
    price = last["close"]
    atr_val = last["atr14"]
    
    if "ACHAT" in final_signal:
        sl = price - (2.0 * atr_val)
        tp = price + (3.0 * atr_val)
    else:
        sl = price + (2.0 * atr_val)
        tp = price - (3.0 * atr_val)
        
    rr = abs(tp - price) / abs(price - sl) if abs(price-sl) > 0 else 0
    
    return {
        "Instrument": pair,
        "TF": tf,
        "Signal": final_signal,
        "Confiance": min(round(conf, 1), 100),
        "Prix": round(price, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": f"1:{round(rr, 1)}",
        "RSI": round(rsi_val, 1),
        "Heure": last["time"].strftime("%H:%M"),
        "_sort_conf": conf,
        "_is_live": is_live
    }

def scan_parallel(pairs, tfs, max_workers=8):
    results = []
    # On fixe count=120 pour aller vite
    tasks = [(p, tf, 120) for p in pairs for tf in tfs]
    
    prog = st.progress(0)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_pair, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futures)):
            prog.progress((i + 1) / len(tasks))
            try:
                res = f.result()
                if res: results.append(res)
            except: pass
    prog.empty()
    return results

# ----------------------
# Interface & Session Market
# ----------------------
st.title("⚡ Forex Scanner Pro • LIVE 0-LAG")
st.markdown("**Mode Temps Réel activé :** Les signaux avec ⚡ changent tick par tick (comme TradingView).")

# Sidebar
st.sidebar.header("Paramètres Scan")
sel_pairs = st.sidebar.multiselect("Paires", PAIRS_DEFAULT, PAIRS_DEFAULT)
sel_tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], ["H1","H4"])
min_conf = st.sidebar.slider("Confiance Min (%)", 0, 100, 20)
auto_refresh = st.sidebar.checkbox("Auto-refresh (2 min)", value=False)

# Session Info
tz = pytz.timezone('Africa/Tunis')
now = datetime.now(tz)
hour = now.hour
session_name = "Calme"
session_color = "blue"

if 9 <= hour < 18: session_name, session_color = "Londres", "green"
if 14 <= hour < 23: session_name, session_color = "New York", "green"
if 14 <= hour < 18: session_name, session_color = "OVERLAP (Volatilité Max)", "red"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Paires", len(sel_pairs))
col2.metric("Mode", "⚡ LIVE (Incomplet)")
col3.metric("Timeframes", len(sel_tfs))
col4.markdown(f"### <span style='color:{session_color}'>{session_name}</span> ({now.strftime('%H:%M')})", unsafe_allow_html=True)

st.markdown("---")

# Execution
if st.sidebar.button("SCANNER MAINTENANT", type="primary") or auto_refresh:
    
    if auto_refresh:
        time.sleep(1) # Petit délai pour stabilité
        st.toast("Scan automatique lancé...", icon="🔄")
        
    if auto_refresh:
        # Compte à rebours discret
        with st.empty():
            for i in range(120, 0, -1):
                if i % 10 == 0: st.caption(f"Prochain scan dans {i}s")
                time.sleep(1)
            st.rerun()

    start = time.time()
    data = scan_parallel(sel_pairs, sel_tfs)
    
    # Filtrage
    filtered = [d for d in data if d["_sort_conf"] >= min_conf]
    filtered.sort(key=lambda x: x["_sort_conf"], reverse=True)
    
    st.success(f"Scan terminé en {round(time.time()-start, 2)}s • {len(filtered)} signaux trouvés")
    
    if filtered:
        # TOP 5
        cols = st.columns(5)
        for i, item in enumerate(filtered[:5]):
            with cols[i]:
                color = "green" if "ACHAT" in item["Signal"] else "red"
                st.markdown(f":{color}[**{item['Signal']}**]")
                st.metric(item['Instrument'], item['Prix'], f"{item['TF']} • {item['Confiance']}%")
                st.caption(f"SL {item['SL']} | TP {item['TP']}")
        
        st.markdown("---")
        
        # TABLEAU DETAILLÉ
        df_res = pd.DataFrame([{k:v for k,v in d.items() if not k.startswith("_")} for d in filtered])
        
        def color_df(row):
            bg = ""
            if "ACHAT" in row["Signal"]: bg = "background-color: #d4edda; color: black"
            elif "VENTE" in row["Signal"]: bg = "background-color: #f8d7da; color: black"
            
            # Bordure jaune si c'est du LIVE (⚡)
            if "⚡" in row["Signal"]: 
                return [f"{bg}; border-left: 5px solid orange"] * len(row)
            return [bg] * len(row)

        st.dataframe(
            df_res.style.apply(color_df, axis=1),
            use_container_width=True, 
            hide_index=True,
            height=600
        )
    else:
        st.info("Aucun signal détecté. Essaie de baisser la confiance min ou d'ajouter des timeframes.")

else:
    st.info("👋 Prêt. Clique sur **SCANNER MAINTENANT**.")

# Légende
with st.expander("ℹ️ Légende des symboles"):
    st.write("**⚡ (Éclair)** : Signal en cours de formation (Bougie LIVE). Il correspond exactement au prix actuel. Attention, il peut disparaître si le prix se retourne avant la fin de l'heure.")
    st.write("**⭐ (Étoile)** : Signal confirmé sur une bougie clôturée.")
