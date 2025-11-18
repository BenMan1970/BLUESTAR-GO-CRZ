import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
import base64
from fpdf import FPDF
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== CONFIGURATION & STYLE ====================
st.set_page_config(page_title="Hedge Fund FX Scanner", layout="wide")

# CSS : Cache les index, ajuste la police et force la hauteur auto (pas de scroll)
st.markdown("""
<style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .stDataFrame {font-size: 0.9rem;}
    .stDataFrame div[data-testid="stDataFrame"] > div {
        height: auto !important; 
    }
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
        st.error("Token OANDA manquant dans les secrets.")
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

# ==================== INDICATEURS INSTITUTIONNELS ====================
def calculate_indicators(df):
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    close = df.c
    # HMA 20
    df["hma20"] = wma(2 * wma(close, 10) - wma(close, 20), int(np.sqrt(20)))
    
    # RSI 7
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7).mean() / down.ewm(alpha=1/7).mean()
    df["rsi"] = 100 - 100/(1+rs)

    # ATR 14 & ADX
    tr = pd.concat([df.h-df.l, (df.h-df.c.shift()).abs(), (df.l-df.c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14).mean()
    
    plus_dm = df.h.diff().clip(lower=0)
    minus_dm = -df.l.diff().clip(upper=0)
    tr_smooth = tr.ewm(alpha=1/14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / tr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["adx"] = dx.ewm(alpha=1/14).mean()

    # EMA 200
    df["ema200"] = close.ewm(span=200).mean()
    return df

@st.cache_data(ttl=60)
def get_macro_bias(pair, current_tf):
    target_tf = "W" if current_tf == "D1" else "D1"
    df = get_candles_quant(pair, target_tf, 100)
    if len(df) < 50: return "Neutral"
    df = calculate_indicators(df)
    last = df.iloc[-1]
    if last.c > last.ema200: return "Bullish"
    if last.c < last.ema200: return "Bearish"
    return "Neutral"

# ==================== ANALYSE ====================
def analyze_pair(pair, tf, mode_live):
    df = get_candles_quant(pair, tf, 250)
    if len(df) < 200: return None
    df = calculate_indicators(df)
    
    if mode_live:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        is_live_signal = not last.complete
    else:
        idx = -2 if not df.iloc[-1].complete else -1
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        is_live_signal = False

    # Logique technique
    hma_bull = last.hma20 > df.iloc[-3].hma20
    hma_bear = last.hma20 < df.iloc[-3].hma20
    rsi_buy = last.rsi > 50
    rsi_sell = last.rsi < 50
    
    signal_buy = hma_bull and rsi_buy
    signal_sell = hma_bear and rsi_sell
    
    if not (signal_buy or signal_sell): return None

    # Filtres Hedge Fund
    macro = get_macro_bias(pair, tf)
    above_ema200 = last.c > last.ema200
    
    # Scoring
    score = 40
    if signal_buy and macro == "Bullish": score += 30
    elif signal_sell and macro == "Bearish": score += 30
    
    if last.adx > 25: score += 20
    elif last.adx > 20: score += 10
    else: score -= 10 
    
    if signal_buy and above_ema200: score += 10
    if signal_sell and not above_ema200: score += 10

    final_score = max(0, min(100, score))
    
    # SL/TP
    sl = last.c - 2.0 * last.atr if signal_buy else last.c + 2.0 * last.atr
    tp = last.c + 3.0 * last.atr if signal_buy else last.c - 3.0 * last.atr

    tag = "⚡" if is_live_signal else ""
    action = "ACHAT" if signal_buy else "VENTE"
    fmt = "%H:%M" if tf != "D1" else "%Y-%m-%d"
    
    return {
        "Heure": last.time.strftime(fmt),
        "Instrument": pair.replace("_", "/"),
        "TF": tf,
        "Action": f"{action} {tag}",
        "Score": final_score,
        "Prix": last.c,
        "SL": sl, "TP": tp,
        "ADX": int(last.adx),
        "Bias": macro,
        "RSI": int(last.rsi),
        "_raw_action": action
    }

def run_scan(pairs, tfs, mode_live):
    res = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(analyze_pair, p, tf, mode_live) for p in pairs for tf in tfs]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r: res.append(r)
            except: pass
    return res

# ==================== GENERATEUR PDF ====================
def create_pdf(results):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Hedge Fund Scanner Report', 0, 1, 'C')
            self.ln(5)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    headers = ["Heure", "Instrument", "TF", "Action", "Score", "Prix", "SL", "TP"]
    col_widths = [25, 25, 15, 25, 15, 25, 25, 25]
    
    pdf.set_fill_color(200, 200, 200)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 10, h, 1, 0, 'C', 1)
    pdf.ln()
    
    for row in results:
        pdf.set_font("Arial", size=9)
        if "ACHAT" in row["_raw_action"]:
            pdf.set_text_color(0, 100, 0)
        else:
            pdf.set_text_color(150, 0, 0)
        data = [str(row["Heure"]), row["Instrument"], row["TF"], row["Action"].replace("⚡",""), 
                str(row["Score"]), str(row["Prix"]), str(row["SL"]), str(row["TP"])]
        for i, datum in enumerate(data):
            pdf.cell(col_widths[i], 10, datum, 1, 0, 'C')
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# ==================== INTERFACE ====================
st.title("🛡️ Hedge Fund FX Scanner • Quant Edition")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
scan_mode = st.sidebar.radio("Mode", ["Sécurisé (Clôture)", "Aggressif (0-Lag ⚡)"], index=0)
is_live = "Aggressif" in scan_mode

selected_tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], default=["H1","H4","D1"])
min_score = st.sidebar.slider("Score Quant Min", 0, 100, 60)
scan_btn = st.sidebar.button("SCANNER LE MARCHÉ", type="primary", use_container_width=True)

# --- EXECUTION ---
if scan_btn:
    with st.spinner("Analyse algorithmique en cours..."):
        results = run_scan(PAIRS_DEFAULT, selected_tfs, is_live)
        results = [r for r in results if r["Score"] >= min_score]
        
    if results:
        # Stats basiques
        c1, c2, c3 = st.columns(3)
        c1.metric("Signaux", len(results))
        c2.metric("Meilleur Score", max(r["Score"] for r in results) if results else 0)
        c3.metric("Mode", "LIVE" if is_live else "CONFIRMÉ")
        
        st.markdown("---")

        # ==========================================
        # 🏆 RETOUR DU TOP 5 (La section manquante)
        # ==========================================
        st.subheader("🏆 Top 5 Meilleures Opportunités")
        
        # Tri global par score décroissant
        top5 = sorted(results, key=lambda x: x["Score"], reverse=True)[:5]
        
        cols = st.columns(5)
        for i, row in enumerate(top5):
            with cols[i]:
                color = "green" if "ACHAT" in row["_raw_action"] else "red"
                # Affichage convivial style Code 1
                st.markdown(f":{color}[**{row['Action']}**]")
                st.metric(
                    label=row['Instrument'],
                    value=row['Prix'],
                    delta=f"{row['TF']} • Score {row['Score']}/100"
                )
                st.caption(f"TP: {row['TP']:.5f}")
        
        st.markdown("---")
        # ==========================================

        # Fonction de style
        def style_quant(row):
            base = "color: black;"
            if "ACHAT" in row["Action"]: 
                base += "background-color: #d4edda;" # Vert pastel
            else: 
                base += "background-color: #f8d7da;" # Rouge pastel
            if row["Score"] >= 80: 
                base += "font-weight: bold; border-left: 4px solid gold;"
            return [base] * len(row)

        # Affichage par Timeframe
        tf_order = ["H1", "H4", "D1"]
        for tf in tf_order:
            if tf not in selected_tfs: continue
            subset = [r for r in results if r["TF"] == tf]
            if subset:
                st.subheader(f"🕒 Timeframe {tf} ({len(subset)})")
                subset.sort(key=lambda x: x["Score"], reverse=True)
                
                df_show = pd.DataFrame(subset)
                cols_to_show = ["Heure", "Instrument", "Action", "Score", "Prix", "SL", "TP", "ADX", "Bias"]
                
                # Hauteur dynamique pour éviter le scroll
                height_dynamic = (len(df_show) + 1) * 35 + 3
                
                st.dataframe(
                    df_show[cols_to_show].style.apply(style_quant, axis=1).format({
                        "Prix": "{:.5f}", "SL": "{:.5f}", "TP": "{:.5f}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=height_dynamic
                )
                st.markdown(" ")

        # Export PDF & CSV
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        
        df_all = pd.DataFrame(results)
        csv = df_all.to_csv(index=False).encode()
        col_dl1.download_button("📥 Télécharger CSV", csv, "quant_signals.csv", "text/csv", use_container_width=True)
        
        try:
            pdf_bytes = create_pdf(results)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="quant_report.pdf" style="text-decoration:none; color:white; background-color:#FF4B4B; padding:10px 20px; border-radius:5px; display:block; text-align:center;">📄 Télécharger Rapport PDF</a>'
            col_dl2.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            col_dl2.error(f"Ajoutez 'fpdf' dans requirements.txt pour le PDF. Erreur: {e}")

    else:
        st.warning("Aucun signal ne correspond à vos critères.")
