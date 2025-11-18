import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== CONFIG "QUANT" ====================
st.set_page_config(page_title="Hedge Fund FX Scanner", layout="wide", initial_sidebar_state="expanded")

# CSS Pro sombre et épuré
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;}
    .stDataFrame {border: 1px solid #333;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ==================== API & DATA ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        st.error("Manque le Token OANDA dans les secrets.")
        st.stop()

client = get_oanda_client()

@st.cache_data(ttl=15)
def get_candles_quant(pair, tf, count=200):
    gran = GRANULARITY_MAP.get(tf)
    if not gran: return pd.DataFrame()
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        
        data = []
        for c in req.response.get("candles", []):
            data.append({
                "time": c["time"],
                "o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]),
                "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"]),
                "vol": int(c["volume"]),
                "complete": c.get("complete", False)
            })
        df = pd.DataFrame(data)
        if not df.empty: df["time"] = pd.to_datetime(df["time"])
        return df
    except: return pd.DataFrame()

# ==================== MATHÉMATIQUES FINANCIÈRES (NUMPY) ====================
def calculate_indicators(df):
    # 1. HMA (Hull Moving Average) - Réactivité
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    close = df.c
    df["hma20"] = wma(2 * wma(close, 10) - wma(close, 20), int(np.sqrt(20)))
    
    # 2. RSI 7 (Momentum)
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7).mean() / down.ewm(alpha=1/7).mean()
    df["rsi"] = 100 - 100/(1+rs)

    # 3. ATR 14 (Volatilité)
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14).mean()
    
    # 4. ADX 14 (Force de Tendance - INSTITUTIONNEL)
    plus_dm = df.h.diff().clip(lower=0)
    minus_dm = -df.l.diff().clip(upper=0)
    tr_smooth = tr.ewm(alpha=1/14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["adx"] = dx.ewm(alpha=1/14).mean()

    # 5. EMA 200 (Biais Long Terme)
    df["ema200"] = close.ewm(span=200).mean()

    return df

@st.cache_data(ttl=60)
def get_macro_bias(pair):
    """Check D1 Trend pour filtrer les faux signaux H1/H4"""
    df = get_candles_quant(pair, "D1", 100)
    if len(df) < 50: return "Neutral"
    df = calculate_indicators(df)
    last = df.iloc[-1]
    # Si Prix > EMA200 et HMA monte = Bias Bullish
    if last.c > last.ema200 and last.hma20 > df.iloc[-2].hma20: return "Bullish"
    if last.c < last.ema200 and last.hma20 < df.iloc[-2].hma20: return "Bearish"
    return "Neutral"

# ==================== LOGIQUE DE TRADING "QUANT" ====================
def analyze_pair(pair, tf, mode_live):
    df = get_candles_quant(pair, tf, 250)
    if len(df) < 200: return None
    
    df = calculate_indicators(df)
    
    # Gestion Live vs Clôture
    if mode_live:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        is_live_signal = not last.complete
    else:
        # Si la dernière n'est pas finie, on recule d'un cran
        idx = -2 if not df.iloc[-1].complete else -1
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        is_live_signal = False

    # --- 1. ANALYSE TECHNIQUE ---
    hma_bull = last.hma20 > df.iloc[-3].hma20 # Pente HMA
    hma_bear = last.hma20 < df.iloc[-3].hma20
    
    # Cross RSI
    rsi_buy = last.rsi > 50
    rsi_sell = last.rsi < 50
    
    # --- 2. FILTRES INSTITUTIONNELS ---
    # A. Filtre ADX (Le plus important) : On évite le "Chop"
    is_trending = last.adx > 20 
    trend_strength = "Fort" if last.adx > 30 else ("Moyen" if last.adx > 20 else "Faible")
    
    # B. Biais Macro (D1)
    macro = get_macro_bias(pair)
    
    # C. Alignement EMA200 (On n'achète pas sous la 200 en général, sauf retournement violent)
    above_ema200 = last.c > last.ema200
    
    # --- 3. DÉCISION ---
    # Signal Achat : HMA Monte + RSI > 50 + (Idéalement ADX ok ou Macro ok)
    signal_buy = hma_bull and rsi_buy
    signal_sell = hma_bear and rsi_sell
    
    if not (signal_buy or signal_sell): return None

    # --- 4. SCORING QUANT (0 à 100) ---
    score = 0
    # Base technique (40 pts)
    score += 40 
    
    # Alignement Macro D1 (30 pts)
    if signal_buy and macro == "Bullish": score += 30
    elif signal_sell and macro == "Bearish": score += 30
    
    # Qualité de la tendance ADX (20 pts)
    if last.adx > 25: score += 20
    elif last.adx > 20: score += 10
    else: score -= 10 # Pénalité si marché plat
    
    # Position / EMA 200 (10 pts)
    if signal_buy and above_ema200: score += 10
    if signal_sell and not above_ema200: score += 10

    # RSI Extrême (Bonus/Malus)
    if signal_buy and last.rsi > 75: score -= 10 # Attention surachat
    if signal_sell and last.rsi < 25: score -= 10 # Attention survente
    
    final_score = max(0, min(100, score))

    # Gestion SL/TP Dynamique
    sl_dist = 2.0 * last.atr
    tp_dist = 3.0 * last.atr # Risk Reward 1:1.5
    
    sl = last.c - sl_dist if signal_buy else last.c + sl_dist
    tp = last.c + tp_dist if signal_buy else last.c - tp_dist

    tag = "⚡" if is_live_signal else ""
    
    return {
        "Instrument": pair.replace("_", "/"),
        "TF": tf,
        "Action": "ACHAT" if signal_buy else "VENTE",
        "Tag": tag,
        "Score": final_score,
        "Prix": last.c,
        "SL": sl, "TP": tp,
        "ADX": f"{int(last.adx)} ({trend_strength})",
        "Macro": macro,
        "RSI": int(last.rsi),
        "Time": last.time
    }

def run_scan(pairs, tfs, mode_live):
    res = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(analyze_pair, p, tf, mode_live) for p in pairs for tf in tfs]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r: res.append(r)
            except: pass
    return res

# ==================== INTERFACE ====================
st.title("🛡️ Hedge Fund FX Scanner • Quant Edition")
st.markdown("Algorithme Institutionnel : **ADX Filter + Macro Bias + Dynamic Risk**")

# Sidebar
st.sidebar.header("Paramètres Stratégie")
mode = st.sidebar.radio("Type de Signal", ["Sécurisé (Clôture)", "Aggressif (0-Lag ⚡)"], index=0)
is_live = "Aggressif" in mode

st.sidebar.subheader("Filtres")
pairs_sel = st.sidebar.multiselect("Univers", PAIRS_DEFAULT, PAIRS_DEFAULT)
tfs_sel = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], ["H1","H4"])
min_score = st.sidebar.slider("Quant Score Min", 0, 100, 60, help="Score < 50 = Signal faible ou contre-tendance.")

if st.sidebar.button("🔍 LANCER L'ANALYSE QUANT", type="primary"):
    
    # Dashboard Info Marché
    tz = pytz.timezone('Africa/Tunis')
    now = datetime.now(tz)
    
    with st.spinner("Calcul des métriques institutionnelles..."):
        results = run_scan(pairs_sel, tfs_sel, is_live)
        # Filtre par score
        results = [r for r in results if r["Score"] >= min_score]
        results.sort(key=lambda x: x["Score"], reverse=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paires Analysées", len(pairs_sel))
    c2.metric("Signaux Qualifiés", len(results))
    c3.metric("Filtre ADX", "Activé (>20)")
    c4.metric("Session", now.strftime("%H:%M"))

    if results:
        st.markdown("### 🔥 Top Opportunités (Score > 80)")
        top_tier = [r for r in results if r["Score"] >= 80]
        
        if top_tier:
            cols = st.columns(min(4, len(top_tier)))
            for i, r in enumerate(top_tier[:4]):
                with cols[i]:
                    color = "green" if r["Action"] == "ACHAT" else "red"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color:{color}; margin:0">{r['Action']} {r['Tag']}</h3>
                        <h4 style="margin:0">{r['Instrument']}</h4>
                        <p style="font-size:0.9em; color:#888">{r['TF']} • Score: <b>{r['Score']}/100</b></p>
                        <hr style="border-color:#444">
                        <p style="font-size:0.8em">TP: {r['TP']:.5f}<br>SL: {r['SL']:.5f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Aucune opportunité 'Premium' (Score > 80) détectée. Le marché est peut-être indécis.")

        st.markdown("---")
        
        # TABLEAU COMPLET
        df_disp = pd.DataFrame(results)
        
        # Mise en forme pour affichage
        df_show = df_disp[["Instrument", "TF", "Action", "Tag", "Score", "Prix", "SL", "TP", "ADX", "Macro", "RSI"]]
        
        def highlight(row):
            styles = [''] * len(row)
            if row["Action"] == "ACHAT":
                base = "background-color: rgba(40, 167, 69, 0.2);"
            else:
                base = "background-color: rgba(220, 53, 69, 0.2);"
            
            # Si Score très élevé, on met en gras/plus visible
            if row["Score"] >= 80:
                base += " font-weight: bold; border-left: 4px solid gold;"
            
            return [base] * len(row)

        st.dataframe(
            df_show.style.apply(highlight, axis=1).format({
                "Prix": "{:.5f}", "SL": "{:.5f}", "TP": "{:.5f}"
            }),
            use_container_width=True,
            height=800,
            hide_index=True
        )
        
    else:
        st.warning("Aucun signal ne respecte vos critères de risque (Score insuffisant).")
        st.caption("Conseil : Si l'ADX est faible partout, restez à l'écart.")

else:
    st.info("Prêt pour l'analyse. Cliquez sur le bouton dans la barre latérale.")

with st.expander("📚 Comprendre le Score Quant"):
    st.markdown("""
    Le **Score (0-100)** n'est pas juste technique, il est probabiliste :
    - **Base Technique (40%)** : HMA + RSI valident la direction.
    - **Biais Macro (30%)** : Le D1 confirme-t-il le trade H1/H4 ? (Alignement des planètes).
    - **Force ADX (20%)** : Y a-t-il une vraie tendance ou juste du bruit ? (ADX > 25 est idéal).
    - **Zone Institutionnelle (10%)** : Sommes-nous du bon côté de l'EMA 200 ?
    
    *Un score > 80 est une opportunité statistiquement très forte.*
    """)
