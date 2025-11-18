import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import pytz # Indispensable pour l'heure de Tunis
from fpdf import FPDF
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Cascade", layout="wide")

# CSS pour cacher les erreurs de contexte et styliser
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

# ==================== API ====================
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
                "time": c["time"], # Format UTC brut
                "open": float(c["mid"]["o"]), "h": float(c["mid"]["h"]),
                "l": float(c["mid"]["l"]), "close": float(c["mid"]["c"]),
                "complete": c.get("complete", False)
            })
        df = pd.DataFrame(data)
        if not df.empty: 
            # Conversion propre des dates
            df["time"] = pd.to_datetime(df["time"])
        return df
    except: return pd.DataFrame()

# ==================== INDICATEURS ====================
def calculate_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']
    
    # HMA 20
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
    df['hma_up'] = df['hma'] > df['hma'].shift(1)
    
    # RSI 7
    delta = close.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - 100/(1 + up.ewm(alpha=1/7).mean()/down.ewm(alpha=1/7).mean())
    
    # UT BOT
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    xATR = tr.rolling(1).mean()
    nLoss = 2.0 * xATR
    
    xATRTrailingStop = [0.0] * len(df)
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        curr_src = close.iloc[i]
        loss = nLoss.iloc[i]
        if (curr_src > prev_stop) and (close.iloc[i-1] > prev_stop):
            xATRTrailingStop[i] = max(prev_stop, curr_src - loss)
        elif (curr_src < prev_stop) and (close.iloc[i-1] < prev_stop):
            xATRTrailingStop[i] = min(prev_stop, curr_src + loss)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - loss
        else:
            xATRTrailingStop[i] = curr_src + loss
    
    df['ut_state'] = np.where(close > xATRTrailingStop, 1, -1)
    
    # ADX
    atr14 = tr.ewm(alpha=1/14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()
    
    df['atr_val'] = atr14
    return df

# ==================== LOGIQUE CASCADE ====================
@st.cache_data(ttl=60)
def get_trend_alignment(pair, signal_tf):
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf: return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50: return "Neutral"
    
    close = df['close']
    ema50 = close.ewm(span=50).mean().iloc[-1]
    
    # HMA TF Supérieur
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    hma = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
    
    hma_now = hma.iloc[-1]
    hma_prev = hma.iloc[-2]
    price = close.iloc[-1]
    
    if price > ema50 and hma_now > hma_prev: return "Bullish"
    if price < ema50 and hma_now < hma_prev: return "Bearish"
    return "Neutral"

# ==================== ANALYSE & TIMEZONE FIX ====================
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
    
    # Signal BlueStar
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
    
    # Cascade Check
    higher_trend = get_trend_alignment(pair, tf)
    if action == "ACHAT" and higher_trend != "Bullish": return None
    if action == "VENTE" and higher_trend != "Bearish": return None
    
    # Scoring
    score = 70
    if last.adx > 25: score += 15
    elif last.adx > 20: score += 10
    is_fresh = (action == "ACHAT" and hma_flip_green) or (action == "VENTE" and hma_flip_red)
    if is_fresh: score += 15
    score = min(100, score)
    
    # SL/TP
    atr = last.atr_val
    if action == "ACHAT":
        sl = last.close - 2.0 * atr
        tp = last.close + 3.0 * atr
    else:
        sl = last.close + 2.0 * atr
        tp = last.close - 3.0 * atr
        
    tag = "⚡" if is_live_signal else ""
    
    # --- FIX TIMEZONE (UTC -> TUNIS) ---
    utc_time = last.time # C'est du datetime UTC
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time) # On s'assure qu'il sait qu'il est UTC
    
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = utc_time.astimezone(tunis_tz)
    
    # Formatage
    fmt = "%H:%M" if tf != "D1" else "%Y-%m-%d"
    time_str = local_time.strftime(fmt)
    
    higher_tf_name = {"H1": "H4", "H4": "D1", "D1": "W"}.get(tf, "")
    
    return {
        "Heure": time_str,
        "Instrument": pair.replace("_", "/"),
        "TF": tf,
        "Action": f"{action} {tag}",
        "IsFresh": is_fresh,
        "Score": score,
        "Prix": last.close,
        "SL": sl, "TP": tp,
        "ADX": int(last.adx),
        "Validation": f"✅ {higher_tf_name} {higher_trend}",
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
            self.cell(0, 10, 'BlueStar Cascade Report', 0, 1, 'C')
            self.ln(5)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    headers = ["Heure", "Instrument", "TF", "Action", "Score", "Validation", "TP"]
    col_widths = [20, 25, 15, 30, 15, 40, 25]
    pdf.set_fill_color(200, 200, 200)
    for i, h in enumerate(headers): pdf.cell(col_widths[i], 10, h, 1, 0, 'C', 1)
    pdf.ln()
    for row in results:
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(0, 100, 0) if "ACHAT" in row["_raw_action"] else pdf.set_text_color(150, 0, 0)
        data = [str(row["Heure"]), row["Instrument"], row["TF"], row["Action"].replace("⚡",""), 
                str(row["Score"]), row["Validation"], str(round(row["TP"],4))]
        for i, d in enumerate(data): pdf.cell(col_widths[i], 10, d, 1, 0, 'C')
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# ==================== INTERFACE ====================
st.title("💎 BlueStar Cascade (Alignement Strict)")

# Affichage de l'heure locale pour vérification
now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
st.caption(f"Heure Serveur (Tunis) : {now_tunis.strftime('%H:%M:%S')}")

st.sidebar.header("Configuration")
scan_mode = st.sidebar.radio("Mode", ["Signaux Confirmés", "Temps Réel (⚡)"], index=0)
is_live = "Temps Réel" in scan_mode
tfs = st.sidebar.multiselect("Timeframes", ["H1","H4","D1"], ["H1","H4","D1"])
scan_btn = st.sidebar.button("LANCER LE SCAN", type="primary", use_container_width=True)

if scan_btn:
    with st.spinner("Analyse en cours..."):
        results = run_scan(PAIRS_DEFAULT, tfs, is_live)
        
    if results:
        st.subheader("🏆 Top Opportunités")
        top5 = sorted(results, key=lambda x: x["Score"], reverse=True)[:5]
        cols = st.columns(5)
        for i, r in enumerate(top5):
            with cols[i]:
                color = "green" if "ACHAT" in r["_raw_action"] else "red"
                st.markdown(f":{color}[**{r['Action']}**]")
                st.metric(r['Instrument'], r['Prix'], f"Score {r['Score']}")
                st.caption(r['Validation'])
        
        st.markdown("---")
        
        def style_cascade(row):
            base = "color: black;"
            if "ACHAT" in row["Action"]: base += "background-color: #d4edda;"
            else: base += "background-color: #f8d7da;"
            if row["Score"] >= 85: base += "font-weight: bold; border-left: 5px solid gold;"
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
                    df_s[["Heure","Instrument","Action","Score","Prix","SL","TP","ADX","Validation"]].style.apply(style_cascade, axis=1).format({"Prix":"{:.5f}","SL":"{:.5f}","TP":"{:.5f}"}),
                    use_container_width=True, hide_index=True, height=h_dyn
                )

        c1, c2 = st.columns(2)
        try:
            pdf = create_pdf(results)
            b64 = base64.b64encode(pdf).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="cascade_report.pdf" style="text-decoration:none; color:white; background-color:#FF4B4B; padding:10px; border-radius:5px; display:block; text-align:center">📄 PDF Rapport</a>'
            c2.markdown(href, unsafe_allow_html=True)
        except: pass
        
        csv = pd.DataFrame(results).to_csv(index=False).encode()
        c1.download_button("📥 CSV Data", csv, "cascade_data.csv", "text/csv", use_container_width=True)

    else: st.warning("Aucun signal parfaitement aligné détecté.")
