import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from fpdf import FPDF
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== CONFIGURATION & STYLE ====================
st.set_page_config(page_title="BlueStar HedgeFund Pro", layout="wide")

st.markdown("""
<style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .stDataFrame {font-size: 0.9rem;}
    .stDataFrame div[data-testid="stDataFrame"] > div {height: auto !important;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ==================== API OANDA ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: st.error("Token manquant"); st.stop()

client = get_oanda_client()

@st.cache_data(ttl=15)
def get_candles(pair, tf, count=300):
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
                "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
                "complete": c.get("complete", False)
            })
        df = pd.DataFrame(data)
        if not df.empty: df["time"] = pd.to_datetime(df["time"])
        return df
    except: return pd.DataFrame()

# ==================== INDICATEURS FUSIONNÉS ====================

def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    
    # --- 1. LOGIQUE BLUESTAR (TRIGGER) ---
    # HMA 20
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    hma_period = 20
    half = int(hma_period / 2)
    sqrt_l = int(np.sqrt(hma_period))
    wma_half = wma(close, half)
    wma_full = wma(close, hma_period)
    df['hma'] = wma(2 * wma_half - wma_full, sqrt_l)
    df['hma_up'] = df['hma'] > df['hma'].shift(1)
    
    # RSI 7
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.ewm(alpha=1/7).mean()/down.ewm(alpha=1/7).mean())
    
    # UT BOT (Sensitivity 2.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    xATR = tr.rolling(1).mean() # Period 1
    nLoss = 2.0 * xATR
    
    xATRTrailingStop = [0.0] * len(df)
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        curr_src = close.iloc[i]
        prev_src = close.iloc[i-1]
        loss = nLoss.iloc[i]
        if (curr_src > prev_stop) and (prev_src > prev_stop):
            xATRTrailingStop[i] = max(prev_stop, curr_src - loss)
        elif (curr_src < prev_stop) and (prev_src < prev_stop):
            xATRTrailingStop[i] = min(prev_stop, curr_src + loss)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - loss
        else:
            xATRTrailingStop[i] = curr_src + loss
    
    df['ut_state'] = np.where(close > xATRTrailingStop, 1, -1)
    
    # --- 2. LOGIQUE HEDGE FUND (FILTRES) ---
    # ADX 14
    atr14 = tr.ewm(alpha=1/14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()
    
    # EMA 200 (Institutionnel)
    df['ema200'] = close.ewm(span=200).mean()
    
    # ATR pour SL/TP
    df['atr_val'] = atr14
    
    return df

# --- FONCTION MACRO BIAS (Hedge Fund) ---
@st.cache_data(ttl=60)
def get_macro_bias(pair):
    # On regarde toujours le D1 pour la macro trend
    df = get_candles(pair, "D1", 100)
    if len(df) < 50: return "Neutral"
    
    close = df['close']
    ema200 = close.ewm(span=200).mean().iloc[-1]
    # HMA D1 pour la direction court terme du daily
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    hma = wma(2 * wma_half - wma_full, int(np.sqrt(20))).iloc[-1]
    hma_prev = wma(2 * wma_half - wma_full, int(np.sqrt(20))).iloc[-2]
    
    price = close.iloc[-1]
    
    if price > ema200 and hma > hma_prev: return "Bullish"
    if price < ema200 and hma < hma_prev: return "Bearish"
    return "Neutral"

# ==================== ANALYSE CORE ====================
def analyze_pair(pair, tf, mode_live):
    df = get_candles(pair, tf, 300)
    if len(df) < 100: return None
    
    df = calculate_indicators(df)
    
    if mode_live:
        idx = -1
        is_live_signal = not df.iloc[-1]['complete']
    else:
        idx = -2 if not df.iloc[-1]['complete'] else -1
        is_live_signal = False
        
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    # --- 1. DÉTECTION DU SIGNAL (BLUESTAR LOGIC) ---
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    
    rsi_ok_buy = last.rsi > 50
    rsi_ok_sell = last.rsi < 50
    
    ut_bull = last.ut_state == 1
    ut_bear = last.ut_state == -1
    
    raw_buy = (hma_flip_green or (last.hma_up and not prev2.hma_up)) and rsi_ok_buy and ut_bull
    raw_sell = (hma_flip_red or (not last.hma_up and prev2.hma_up)) and rsi_ok_sell and ut_bear
    
    if not (raw_buy or raw_sell): return None
    
    action = "ACHAT" if raw_buy else "VENTE"
    
    # --- 2. FILTRAGE HEDGE FUND (QUANT LOGIC) ---
    score = 50 # Base score
    
    # A. Filtre ADX (Force de la tendance)
    adx_val = last.adx
    if adx_val > 25: score += 20 # Autoroute
    elif adx_val > 20: score += 10 # Correct
    else: score -= 20 # Range/Chop (Dangereux)
    
    # B. Filtre Macro Bias (Tendance D1)
    macro = get_macro_bias(pair)
    if action == "ACHAT" and macro == "Bullish": score += 20
    elif action == "VENTE" and macro == "Bearish": score += 20
    elif macro == "Neutral": score += 0
    else: score -= 15 # Contre-tendance Daily
    
    # C. Filtre EMA 200 (Zone Institutionnelle)
    above_ema = last.close > last.ema200
    if (action == "ACHAT" and above_ema) or (action == "VENTE" and not above_ema):
        score += 10
        
    # D. Bonus Fresh Signal
    is_fresh = (action == "ACHAT" and hma_flip_green) or (action == "VENTE" and hma_flip_red)
    if is_fresh: score += 10
    
    # Note finale
    final_score = max(0, min(100, score))
    
    # SL/TP
    atr = last.atr_val
    if action == "ACHAT":
        sl = last.close - 2.0 * atr
        tp = last.close + 3.0 * atr
    else:
        sl = last.close + 2.0 * atr
        tp = last.close - 3.0 * atr
        
    tag = "⚡" if is_live_signal else ""
    fmt = "%H:%M" if tf != "D1" else "%Y-%m-%d"
    
    return {
        "Heure": last.time.strftime(fmt),
        "Instrument": pair.replace("_", "/"),
        "TF": tf,
        "Action": f"{action} {tag}",
        "IsFresh": is_fresh,
        "Score": final_score,
        "Prix": last.close,
        "SL": sl, "TP": tp,
        "ADX": int(adx_val),
        "Macro": macro,
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

# ==================== PDF ====================
def create_pdf(results):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'BlueStar HedgeFund Report', 0, 1, 'C')
            self.ln(5)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    headers = ["Heure", "Instrument", "TF", "Action", "Score", "ADX", "Macro", "TP"]
    col_widths = [20, 25, 15, 30, 15, 15, 25, 25]
    pdf.set_fill_color(200, 200, 200)
    for i, h in enumerate(headers): pdf.cell(col_widths[i], 10, h, 1, 0, 'C', 1)
    pdf.ln()
    for row in results:
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(0, 100, 0) if "ACHAT" in row["_raw_action"] else pdf.set_text_color(150, 0, 0)
        data = [str(row["Heure"]), row["Instrument"], row["TF"], row["Action"].replace("⚡",""), 
                str(row["Score"]), str(row["ADX"]), row["Macro"], str(round(row["TP"],4))]
        for i, d in enumerate(data): pdf.cell(col_widths[i], 10, d, 1, 0, 'C')
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# ==================== INTERFACE ====================
st.title("🚀 BlueStar x Hedge Fund Ultimate")
st.markdown("Déclencheur : **BlueStar (HMA+RSI+UTBot)**  |  Filtres : **Institutionnels (ADX+D1+EMA200)**")

st.sidebar.header("Paramètres")
scan_mode = st.sidebar.radio("Mode", ["Signaux Confirmés", "Temps Réel (⚡)"], index=0)
is_live = "Temps Réel" in scan_mode
tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], ["H1","H4","D1"])
min_score = st.sidebar.slider("Score Min (Qualité)", 0, 100, 50, help="Filtre les signaux faibles (ADX bas ou Contre-tendance)")
scan_btn = st.sidebar.button("LANCER LE SCAN", type="primary", use_container_width=True)

if scan_btn:
    with st.spinner("Fusion BlueStar & Données Institutionnelles..."):
        results = run_scan(PAIRS_DEFAULT, tfs, is_live)
        results = [r for r in results if r["Score"] >= min_score]
        
    if results:
        # TOP 5
        st.subheader("🏆 Top 5 Opportunités (Validées HF)")
        top5 = sorted(results, key=lambda x: x["Score"], reverse=True)[:5]
        cols = st.columns(5)
        for i, r in enumerate(top5):
            with cols[i]:
                color = "green" if "ACHAT" in r["_raw_action"] else "red"
                st.markdown(f":{color}[**{r['Action']}**]")
                st.metric(r['Instrument'], r['Prix'], f"Score {r['Score']}/100")
                st.caption(f"ADX: {r['ADX']} | Bias: {r['Macro']}")
        
        st.markdown("---")
        
        # TABLES
        def style_hf(row):
            base = "color: black;"
            if "ACHAT" in row["Action"]: base += "background-color: #d4edda;"
            else: base += "background-color: #f8d7da;"
            if row["Score"] >= 80: base += "font-weight: bold; border-left: 5px solid gold;"
            return [base] * len(row)

        for tf in ["H1","H4","D1"]:
            if tf not in tfs: continue
            subset = [r for r in results if r["TF"] == tf]
            if subset:
                st.subheader(f"Timeframe {tf} ({len(subset)})")
                subset.sort(key=lambda x: x["Score"], reverse=True)
                df_s = pd.DataFrame(subset)
                h_dyn = (len(df_s)+1)*35+3
                
                st.dataframe(
                    df_s[["Heure","Instrument","Action","Score","Prix","SL","TP","ADX","Macro"]].style.apply(style_hf, axis=1).format({"Prix":"{:.5f}","SL":"{:.5f}","TP":"{:.5f}"}),
                    use_container_width=True, hide_index=True, height=h_dyn
                )

        # EXPORT
        c1, c2 = st.columns(2)
        try:
            pdf = create_pdf(results)
            b64 = base64.b64encode(pdf).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="hf_report.pdf" style="text-decoration:none; color:white; background-color:#FF4B4B; padding:10px; border-radius:5px; display:block; text-align:center">📄 PDF Rapport</a>'
            c2.markdown(href, unsafe_allow_html=True)
        except: pass
        
        csv = pd.DataFrame(results).to_csv(index=False).encode()
        c1.download_button("📥 CSV Data", csv, "hf_data.csv", "text/csv", use_container_width=True)

    else: st.warning("Aucun signal ne passe les filtres Hedge Fund (Essayez de baisser le Score Min).")
