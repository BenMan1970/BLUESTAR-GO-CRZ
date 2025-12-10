"""
BlueStar Institutional v5.1 (Stable CSM)
Fixes the API Freeze by fetching CSM data sequentially.
Replaces Score column with CSM Strength Differential.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# OANDA API
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# PDF Export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar v5.1", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
    /* DASHBOARD CSM */
    .csm-box {background: rgba(255,255,255,0.05); border-radius: 8px; padding: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.1);}
    .csm-val {font-size: 1.4rem; font-weight: bold;}
    
    /* TABLEAUX */
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; margin-top: 20px; margin-bottom: 10px;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem;}
    
    /* BADGES */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem;}
</style>
""", unsafe_allow_html=True)

# Listes
ALL_MAJOR_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"] # Liste réduite pour stabilité
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# ==================== DATA CLASSES ====================
@dataclass
class TradingParams:
    atr_sl: float
    atr_tp: float
    use_fvg: bool
    strict_flip: bool

@dataclass
class Signal:
    timestamp: datetime
    pair: str
    timeframe: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confluences: List[str]
    csm_diff: float  # La valeur CSM (Force)

# ==================== API ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles(pair, tf, count=300):
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":count, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

# ==================== 1. CSM CALCULATOR (SÉQUENTIEL = SAFE) ====================
def calculate_csm_safe():
    """Calcule la force des devises une par une pour ne pas bloquer l'API"""
    scores = {c: 0.0 for c in CURRENCIES}
    
    # On utilise une liste réduite mais représentative pour aller vite et ne pas crash
    pairs_to_check = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD"]
    
    for pair in pairs_to_check:
        # Pause de sécurité
        time.sleep(0.1)
        df = get_candles(pair, "H1", count=10) # Besoin de peu de bougies
        if len(df) < 2: continue
        
        open_p = df.iloc[-1]['open']
        close_p = df.iloc[-1]['close']
        pct = ((close_p - open_p) / open_p) * 100
        
        base, quote = pair.split("_")
        
        # Poids simplifié
        if pct > 0: # Base forte, Quote faible
            scores[base] += abs(pct) * 10
            scores[quote] -= abs(pct) * 10
        else: # Base faible, Quote forte
            scores[base] -= abs(pct) * 10
            scores[quote] += abs(pct) * 10
            
    # Normalisation 0-10
    final = {}
    vals = list(scores.values())
    if not vals: return {c: 5.0 for c in CURRENCIES}
    
    mn, mx = min(vals), max(vals)
    for c, s in scores.items():
        if mx - mn == 0: norm = 5.0
        else: norm = ((s - mn) / (mx - mn)) * 10
        final[c] = round(norm, 1)
        
    return final

# ==================== 2. ANALYSE TECHNIQUE ====================
def analyze_market(df, pair, tf, params, csm_data):
    if len(df) < 200: return None
    
    # --- INDICATEURS ---
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)
    df['hma'] = hma(df['close'], 20)
    
    h52, l52 = df['high'].rolling(52).max(), df['low'].rolling(52).min()
    df['ssb'] = ((h52 + l52) / 2).shift(26)
    
    fvg_bull = any((df['low'] > df['high'].shift(2)).iloc[-5:])
    fvg_bear = any((df['high'] < df['low'].shift(2)).iloc[-5:])

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- LOGIQUE ---
    hma_up = curr['hma'] > prev['hma']
    hma_down = curr['hma'] < prev['hma']
    
    flip_up = hma_up and (prev['hma'] < df.iloc[-3]['hma'])
    flip_down = hma_down and (prev['hma'] > df.iloc[-3]['hma'])
    
    buy_sig = flip_up if params.strict_flip else hma_up
    sell_sig = flip_down if params.strict_flip else hma_down
    
    trend_bull = curr['close'] > curr['ema200']
    trend_bear = curr['close'] < curr['ema200']

    action = None
    conf = []
    
    if buy_sig and trend_bull:
        action = "BUY"
        if curr['close'] > curr['ssb']: conf.append("Cloud")
        if fvg_bull: conf.append("FVG")
        conf.append("Trend")
    elif sell_sig and trend_bear:
        action = "SELL"
        if curr['close'] < curr['ssb']: conf.append("Cloud")
        if fvg_bear: conf.append("FVG")
        conf.append("Trend")
        
    if not action: return None
    if params.use_fvg and "FVG" not in conf: return None

    # --- CSM CALC ---
    try:
        base, quote = pair.split("_")
        s_base = csm_data.get(base, 5.0)
        s_quote = csm_data.get(quote, 5.0)
    except:
        s_base, s_quote = 5.0, 5.0
        
    # Différentiel de force : Si je BUY, je veux Base > Quote
    if action == "BUY": diff = s_base - s_quote
    else: diff = s_quote - s_base
    
    # On filtre les trades stupides (contre la force)
    if diff < -1.5: return None # Trop dangereux

    atr = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr * params.atr_sl) if action == "BUY" else curr['close'] + (atr * params.atr_sl)
    tp = curr['close'] + (atr * params.atr_tp) if action == "BUY" else curr['close'] - (atr * params.atr_tp)
    
    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        confluences=conf, csm_diff=diff
    )

def smart_format(pair, price):
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

# ==================== MAIN ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v5.1</span>", unsafe_allow_html=True)
    with c2: 
        if st.button("Reset"):
            st.session_state.clear()
            st.rerun()

    if 'scan_results' not in st.session_state: st.session_state.scan_results = None
    if 'csm_data' not in st.session_state: st.session_state.csm_data = None

    with st.expander("⚙️ Configuration", expanded=False):
        c1, c2 = st.columns(2)
        sl = c1.number_input("SL xATR", 1.0, 3.0, 1.5)
        tp = c1.number_input("TP xATR", 1.0, 5.0, 3.0)
        fvg = c2.checkbox("FVG Required", True)
        flip = c2.checkbox("Strict Flip", True)

    if st.button("🚀 SCANNER (CSM + TECHNIQUE)", type="primary", use_container_width=True):
        if not client: st.error("No API Token")
        else:
            # 1. CSM Phase (Séquentiel pour éviter crash)
            with st.spinner("Analyse des Forces (Currency Strength)..."):
                csm = calculate_csm_safe()
                st.session_state.csm_data = csm
            
            # 2. Scan Phase (Barre progression)
            progress = st.progress(0)
            status = st.empty()
            
            params = TradingParams(sl, tp, fvg, flip)
            signals = []
            total = len(SCAN_TARGETS) * len(TIMEFRAMES)
            done = 0
            
            # ThreadPool limité à 4
            with ThreadPoolExecutor(max_workers=4) as exc:
                futures = {exc.submit(lambda p,t: (get_candles(p,t), p, t), p, tf): (p,tf) for p in SCAN_TARGETS for tf in TIMEFRAMES}
                
                for f in as_completed(futures):
                    # Petite pause pour laisser respirer l'API
                    time.sleep(0.05)
                    done += 1
                    progress.progress(done/total)
                    status.text(f"Scanning... {int((done/total)*100)}%")
                    try:
                        df, p, tf = f.result()
                        if not df.empty:
                            s = analyze_market(df, p, tf, params, csm)
                            if s: signals.append(s)
                    except: pass
            
            status.empty()
            progress.empty()
            st.session_state.scan_results = sorted(signals, key=lambda x: x.csm_diff, reverse=True)

    # --- DISPLAY CSM ---
    if st.session_state.csm_data:
        csm = st.session_state.csm_data
        st.markdown("#### 📊 Force des Devises (0-10)")
        cols = st.columns(8)
        sorted_csm = sorted(csm.items(), key=lambda x: x[1], reverse=True)
        for i, (curr, val) in enumerate(sorted_csm):
            col_txt = "#00ff88" if val >= 7 else ("#ff4b4b" if val <= 3 else "#fff")
            cols[i].markdown(f"<div class='csm-box'><div style='color:#aaa'>{curr}</div><div class='csm-val' style='color:{col_txt}'>{val}</div></div>", unsafe_allow_html=True)
        st.markdown("---")

    # --- DISPLAY SIGNALS ---
    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        if signals:
            # PDF GEN
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=10*mm)
            elems = [Paragraph("<b>BlueStar v5.1 Report</b>", getSampleStyleSheet()['Title']), Spacer(1, 10*mm)]
            
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if not tf_sigs: continue
                elems.append(Paragraph(f"<b>{tf}</b>", getSampleStyleSheet()['Normal']))
                data = [["Time", "Pair", "Action", "Price", "Force Diff", "Conf"]]
                for s in tf_sigs:
                    c = colors.green if s.action == "BUY" else colors.red
                    data.append([s.timestamp.strftime("%H:%M"), s.pair, s.action, smart_format(s.pair, s.entry_price), f"{s.csm_diff:+.1f}", ", ".join(s.confluences)])
                t = Table(data)
                t.setStyle(TableStyle([('TEXTCOLOR',(0,0),(-1,0),colors.white), ('BACKGROUND',(0,0),(-1,0),colors.black)]))
                elems.append(t); elems.append(Spacer(1, 5*mm))
            
            doc.build(elems)
            st.download_button("📄 PDF Report", buf.getvalue(), "report.pdf", "application/pdf")
        
        if not signals: st.info("Aucun signal. Les devises sont peut-être neutres.")
        else:
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3></div>", unsafe_allow_html=True)
                    data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        # Nouvelle colonne FORCE = CSM Diff
                        data.append({
                            "Heure": s.timestamp.strftime("%H:%M"), "Paire": s.pair.replace("_","/"),
                            "Signal": f"{icon} {s.action}", "Prix": smart_format(s.pair, s.entry_price),
                            "Force (CSM)": f"{s.csm_diff:+.1f}",  # C'est ici que ça change !
                            "Confirmations": ", ".join(s.confluences)
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
