"""
BlueStar Institutional v5.2 (Corrected Logic)
- CSM calculated on DAILY timeframe (matches websites logic)
- UI restored to focus on the Signals Table
- CSM integrates into the 'Score' column quietly
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
st.set_page_config(page_title="BlueStar v5.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
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
    .csm-badge {background: #333; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; border: 1px solid #555;}
</style>
""", unsafe_allow_html=True)

# Paires Principales pour le CSM (Le "Panier")
MAJORS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# Paires à Scanner (Trading)
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

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
    score: int          # Score Technique (0-100)
    csm_val: float      # Score CSM (-10 à +10)
    confluences: List[str]

# ==================== API ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles_safe(pair, tf, count=250):
    time.sleep(0.05) # Anti-ban
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":count, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

# ==================== 1. CSM LOGIC (CORRIGÉE : DAILY TREND) ====================
def calculate_csm_daily():
    """
    Calcule la force relative basée sur la bougie DAILY actuelle.
    Simule la logique de currencystrengthmeter.org
    """
    scores = {c: 0.0 for c in CURRENCIES}
    
    # On scanne les 7 majeurs en Daily
    for pair in MAJORS:
        df = get_candles_safe(pair, "D1", count=2) # On a juste besoin de la bougie du jour
        if len(df) < 1: continue
        
        # Variation du jour en %
        open_p = df.iloc[-1]['open']
        close_p = df.iloc[-1]['close']
        change = ((close_p - open_p) / open_p) * 100
        
        base, quote = pair.split("_")
        
        # Pondération : 1% de mouvement = 10 points de force
        points = change * 10
        
        # Distribution des points
        # Ex: EURUSD monte (+). EUR gagne, USD perd.
        scores[base] += points
        scores[quote] -= points
        
    # Normalisation pour avoir du 0 à 10 (approx)
    # On ajoute 5.0 à tout le monde pour centrer autour de 5
    final_scores = {}
    for c, s in scores.items():
        # Clamp entre 0 et 10
        val = 5.0 + s
        val = max(0.0, min(10.0, val))
        final_scores[c] = round(val, 1)
        
    return final_scores

# ==================== 2. ANALYSE TECHNIQUE (BLUESTAR CORE) ====================
def analyze_market(df, pair, tf, params, csm_data):
    if len(df) < 100: return None
    
    # Indicateurs
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
    
    # Logique Signal
    hma_up = curr['hma'] > prev['hma']
    hma_down = curr['hma'] < prev['hma']
    
    # Strict Flip check
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

    # Score Technique de base (pour le tri)
    tech_score = 60
    if "Cloud" in conf: tech_score += 15
    if "FVG" in conf: tech_score += 15
    
    # --- CSM INTEGRATION ---
    try:
        base, quote = pair.split("_")
        s_base = csm_data.get(base, 5.0)
        s_quote = csm_data.get(quote, 5.0)
    except:
        s_base, s_quote = 5.0, 5.0 # Neutre pour or/indices
        
    # Calcul du Différentiel pour affichage
    if action == "BUY":
        csm_diff = s_base - s_quote # On veut positif
    else:
        csm_diff = s_quote - s_base # On veut positif
        
    # Filtre CSM (Si on trade contre une force majeure > 2.0, on ignore)
    if csm_diff < -2.0: return None

    atr = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr * params.atr_sl) if action == "BUY" else curr['close'] + (atr * params.atr_sl)
    tp = curr['close'] + (atr * params.atr_tp) if action == "BUY" else curr['close'] - (atr * params.atr_tp)
    
    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=tech_score, csm_val=csm_diff, confluences=conf
    )

def smart_format(pair, price):
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

# ==================== MAIN ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v5.2</span>", unsafe_allow_html=True)
    with c2: 
        if st.button("Reset"):
            st.session_state.clear()
            st.rerun()

    if 'scan_results' not in st.session_state: st.session_state.scan_results = None

    with st.expander("⚙️ Configuration", expanded=False):
        c1, c2 = st.columns(2)
        sl = c1.number_input("SL xATR", 1.0, 3.0, 1.5)
        tp = c1.number_input("TP xATR", 1.0, 5.0, 3.0)
        fvg = c2.checkbox("FVG Required", True)
        flip = c2.checkbox("Strict Flip", True)

    if st.button("🚀 SCANNER MARCHÉ", type="primary", use_container_width=True):
        if not client: st.error("Token API Manquant")
        else:
            # 1. Calcul CSM Daily (Rapide et en background)
            # On ne l'affiche pas en gros, on l'utilise pour le calcul
            with st.spinner("Calibrage des forces Daily..."):
                csm_data = calculate_csm_daily()
            
            # 2. Scan Technique
            progress = st.progress(0)
            status = st.empty()
            
            params = TradingParams(sl, tp, fvg, flip)
            signals = []
            total = len(SCAN_TARGETS) * len(TIMEFRAMES)
            done = 0
            
            with ThreadPoolExecutor(max_workers=4) as exc:
                futures = {exc.submit(lambda p,t: (get_candles_safe(p,t), p, t), p, tf): (p,tf) for p in SCAN_TARGETS for tf in TIMEFRAMES}
                
                for f in as_completed(futures):
                    done += 1
                    progress.progress(done/total)
                    status.text(f"Analyse Technique... {int((done/total)*100)}%")
                    try:
                        df, p, tf = f.result()
                        if not df.empty:
                            s = analyze_market(df, p, tf, params, csm_data)
                            if s: signals.append(s)
                    except: pass
            
            status.empty()
            progress.empty()
            st.session_state.scan_results = sorted(signals, key=lambda x: x.csm_val, reverse=True) # Tri par force CSM

    # --- RESULTATS ---
    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        if signals:
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=10*mm)
            elems = [Paragraph("<b>BlueStar Report</b>", getSampleStyleSheet()['Title']), Spacer(1, 10*mm)]
            
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if not tf_sigs: continue
                elems.append(Paragraph(f"<b>{tf} Structure</b>", getSampleStyleSheet()['Normal']))
                data = [["Time", "Pair", "Action", "Price", "Force", "Conf"]]
                for s in tf_sigs:
                    c = colors.green if s.action == "BUY" else colors.red
                    data.append([
                        s.timestamp.strftime("%H:%M"), s.pair, s.action, 
                        smart_format(s.pair, s.entry_price), 
                        f"{s.csm_val:+.1f}", # Affiche la force CSM
                        ", ".join(s.confluences)
                    ])
                t = Table(data)
                t.setStyle(TableStyle([('TEXTCOLOR',(0,0),(-1,0),colors.white), ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1a1f3a"))]))
                elems.append(t); elems.append(Spacer(1, 5*mm))
            
            doc.build(elems)
            st.download_button("📄 PDF Report", buf.getvalue(), "report.pdf", "application/pdf")

        if not signals: st.info("Aucun signal technique valide aligné avec la force des devises.")
        else:
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3></div>", unsafe_allow_html=True)
                    data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        
                        # Affichage de la FORCE CSM
                        force_str = f"{s.csm_val:+.1f}"
                        
                        data.append({
                            "Time": s.timestamp.strftime("%H:%M"), 
                            "Pair": s.pair.replace("_","/"),
                            "Signal": f"{icon} {s.action}", 
                            "Price": smart_format(s.pair, s.entry_price),
                            "SL": smart_format(s.pair, s.stop_loss), 
                            "TP": smart_format(s.pair, s.take_profit),
                            "CSM Force": force_str,  # VOILA LA COLONNE MODIFIÉE
                            "Confluences": ", ".join(s.confluences)
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
