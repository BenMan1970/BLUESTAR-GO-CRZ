import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration de la page (Style Code 1)
st.set_page_config(page_title="Forex Scanner PRO", layout="wide")

# ==================== CONFIGURATION & API ====================
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

# Mapping des granularités OANDA
GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

@st.cache_resource
def get_oanda_client():
    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        return API(access_token=token)
    except Exception as e:
        st.error(f"Erreur de connexion API : {e}")
        st.stop()

client = get_oanda_client()

# ==================== FONCTIONS DATA (Moteur Code 2) ====================
@st.cache_data(ttl=10) # Cache court pour réactivité
def get_candles(pair, tf, count=150):
    gran = GRANULARITY_MAP.get(tf)
    if not gran: return pd.DataFrame()
    
    try:
        # On récupère TOUJOURS la dernière bougie (même incomplète) pour avoir le choix après
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        
        data = []
        for c in req.response.get("candles", []):
            data.append({
                "time": c["time"],
                "o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]),
                "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"]),
                "complete": c.get("complete", False)
            })
            
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
        return df
    except:
        return pd.DataFrame()

# Indicateurs optimisés (Numpy)
def hma(s, length=20):
    weights = np.arange(1, length + 1)
    def wma(x, w): return np.dot(x, w) / w.sum()
    
    wma1 = s.rolling(length//2).apply(lambda x: wma(x, weights[:length//2]), raw=True)
    wma2 = s.rolling(length).apply(lambda x: wma(x, weights), raw=True)
    diff = 2 * wma1 - wma2
    sqrt_l = int(np.sqrt(length))
    return diff.rolling(sqrt_l).apply(lambda x: wma(x, weights[:sqrt_l]), raw=True)

def rsi(s, length=7):
    d = s.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/length).mean()
    roll_down = down.ewm(alpha=1/length).mean()
    rs = roll_up / roll_down
    return 100 - 100/(1 + rs)

def atr(df, length=14):
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length).mean()

@st.cache_data(ttl=60)
def mtf_trend(pair, tf):
    higher = {"H1":"H4", "H4":"D1", "D1":"W"}.get(tf, "W")
    df = get_candles(pair, higher, 100)
    if len(df) < 40: return "neutral", 0
    
    # Pour la tendance de fond, on préfère travailler sur du clôturé
    if not df.iloc[-1]["complete"]: df = df[:-1]
    
    ema20 = df.c.ewm(span=20).mean().iloc[-1]
    ema50 = df.c.ewm(span=50).mean().iloc[-1]
    price = df.c.iloc[-1]
    
    dist = abs(ema20-ema50)/ema50*100
    if ema20 > ema50 and price > ema20: return "bullish", min(dist*10,100)
    if ema20 < ema50 and price < ema20: return "bearish", min(dist*10,100)
    return "neutral", 0

# ==================== ANALYSE (Hybride) ====================
def analyze(pair, tf, mode_live):
    df = get_candles(pair, tf, 150)
    if len(df) < 50: return None

    # Calculs sur tout l'historique
    df["hma"] = hma(df.c)
    df["rsi"] = rsi(df.c)
    df["atr"] = atr(df)
    df["up"] = df.hma > df.hma.shift(1)

    # SELECTION DE LA BOUGIE SELON LE MODE
    if mode_live:
        # Mode 0-LAG : On prend la dernière ligne, qu'elle soit finie ou non
        last = df.iloc[-1]
        prev = df.iloc[-2]
        tag = "⚡" if not last.complete else "" # Petit éclair si live
    else:
        # Mode CONFIRMÉ : Si la dernière n'est pas finie, on l'ignore et on prend celle d'avant
        if not df.iloc[-1]["complete"]:
            last = df.iloc[-2]
            prev = df.iloc[-3]
        else:
            last = df.iloc[-1]
            prev = df.iloc[-2]
        tag = ""

    # Logique de signal
    trend, strength = mtf_trend(pair, tf)
    
    hma_flip_bull = last.up and not prev.up
    hma_flip_bear = not last.up and prev.up
    
    # Conditions strictes
    buy  = (hma_flip_bull or last.up) and last.rsi > 50 and trend == "bullish"
    sell = (hma_flip_bear or not last.up) and last.rsi < 50 and trend == "bearish"

    if not (buy or sell): return None

    # Gestion SL/TP
    sl = last.c - 2*last.atr if buy else last.c + 2*last.atr
    tp = last.c + 3*last.atr if buy else last.c - 3*last.atr
    
    # Calcul confiance
    dist_rsi = abs(last.rsi - 50)
    conf = (dist_rsi/50 * 40) + (strength * 0.6)
    if (buy and hma_flip_bull) or (sell and hma_flip_bear): conf += 10 # Bonus si nouveau signal
    
    signal_txt = f"{'ACHAT' if buy else 'VENTE'} {tag}"
    if (buy and hma_flip_bull) or (sell and hma_flip_bear):
        signal_txt = f"NEW {signal_txt}"

    return {
        "Instrument": pair.replace("_","/"),
        "TF": tf,
        "Signal": signal_txt,
        "Confiance": min(round(conf, 0), 100),
        "Prix": round(last.c, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": "1:1.5",
        "RSI": round(last.rsi, 1),
        "Heure": last.time.strftime("%H:%M"),
        "_conf": conf,
        "_raw_sig": "ACHAT" if buy else "VENTE"
    }

def scan(pairs, tfs, mode_live):
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        # On envoie le paramètre mode_live à l'analyse
        futures = [ex.submit(analyze, p, tf, mode_live) for p in pairs for tf in tfs]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r: results.append(r)
            except: pass
    return results

# ==================== INTERFACE (Design Code 1) ====================
st.title("Forex Multi-Timeframe Signal Scanner Pro")
st.write("Scanner hybride : Choisissez entre sécurité (Clôture) ou vitesse (0-Lag)")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
with st.sidebar.expander("Filtrer les paires", expanded=False):
    selected_pairs = st.multiselect("Paires :", PAIRS_DEFAULT, default=PAIRS_DEFAULT)

# LE BOUTON MAGIQUE pour choisir le mode
scan_mode = st.sidebar.radio(
    "Mode de détection :",
    ["Signaux Confirmés (Clôture)", "Temps Réel (0-Lag ⚡)"],
    index=0,
    help="Clôture = Signal validé à la fin de la bougie (Sûr).\nTemps Réel = Signal immédiat (Rapide mais peut changer)."
)
is_live_mode = "Temps Réel" in scan_mode

selected_tfs = st.sidebar.multiselect("Timeframes :", ["H1","H4","D1"], default=["H1","H4","D1"])
min_confidence = st.sidebar.slider("Confiance minimale (%) :", 0, 100, 20)
auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)")
scan_button = st.sidebar.button("LANCER LE SCAN", type="primary", use_container_width=True)

# --- SESSION MARKET (Info utile) ---
tz = pytz.timezone('Africa/Tunis')
now = datetime.now(tz)
sessions = {
    "Londres": (9, 18, "green"), "New York": (14, 23, "green"), 
    "Overlap": (14, 18, "red"), "Tokyo": (1, 10, "orange")
}
active_sess = [k for k,v in sessions.items() if v[0] <= now.hour < v[1]]
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Paires", len(selected_pairs))
c2.metric("Mode", "LIVE ⚡" if is_live_mode else "CONFIRMÉ ⭐")
c3.metric("Heure Tunis", now.strftime("%H:%M"))
c4.markdown(f"**Sessions :** {', '.join(active_sess) if active_sess else 'Calme'}")

# --- EXECUTION DU SCAN ---
if scan_button or auto_refresh:
    if auto_refresh and not scan_button:
        with st.spinner(f"Attente refresh..."):
            time.sleep(1) 

    with st.spinner("Analyse du marché en cours..."):
        # On passe le mode choisi à la fonction de scan
        results = scan(selected_pairs, selected_tfs, is_live_mode)
        results = [r for r in results if r["_conf"] >= min_confidence]

    if results:
        # 1. TOP 5 (Comme Code 1)
        st.subheader("🏆 Top 5 Signaux par Confiance")
        top5 = sorted(results, key=lambda x: x["_conf"], reverse=True)[:5]
        cols = st.columns(5)
        for i, r in enumerate(top5):
            with cols[i]:
                color = "green" if r["_raw_sig"] == "ACHAT" else "red"
                st.markdown(f":{color}[**{r['Signal']}**]")
                st.metric(f"{r['Instrument']}", f"{r['Prix']}", f"{r['TF']} • {int(r['Confiance'])}%")
                st.caption(f"SL {r['SL']} | TP {r['TP']}")

        # 2. TABLEAUX SEPARÉS PAR TIMEFRAME (Design Code 1 que tu aimais)
        st.markdown("---")
        
        # Fonction de style (Couleurs pastels Code 1)
        def style_df(row):
            if "ACHAT" in row["Signal"]: 
                return ["background-color: #d4edda; color: black"] * len(row) # Vert pastel
            elif "VENTE" in row["Signal"]: 
                return ["background-color: #f8d7da; color: black"] * len(row) # Rouge pastel
            return [""] * len(row)

        # On boucle sur l'ordre logique H1 -> H4 -> D1
        for tf in ["H1", "H4", "D1"]:
            if tf not in selected_tfs: continue
            
            tf_data = [r for r in results if r["TF"] == tf]
            if tf_data:
                st.markdown(f"### 📅 Signaux {tf} ({len(tf_data)})")
                
                # On prépare le dataframe propre pour l'affichage
                df_show = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tf_data])
                
                # On trie par confiance
                df_show = df_show.sort_values("Confiance", ascending=False)
                
                st.dataframe(
                    df_show.style.apply(style_df, axis=1).format({"Prix": "{:.5f}", "SL": "{:.5f}", "TP": "{:.5f}"}),
                    use_container_width=True, 
                    hide_index=True
                )
        
        # Export CSV
        df_all = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in results])
        csv = df_all.to_csv(index=False).encode()
        st.download_button("📥 Télécharger CSV", csv, "signaux.csv", "text/csv")

    else:
        st.warning("Aucun signal trouvé avec ces critères. Essayez de baisser la confiance min.")

    if auto_refresh:
        time.sleep(300)
        st.rerun()

else:
    st.info("👈 Configurez vos paramètres et cliquez sur **LANCER LE SCAN**")
