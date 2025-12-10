"""
BlueStar Institutional v5.0 (Currency Strength Edition)
Real-Time Currency Strength Meter (CSM) integrated into Scoring Logic
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
from oandapyV20.exceptions import V20Error

# PDF Export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar v5.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
    /* CSM BARS */
    .csm-container {background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1);}
    .csm-label {color: #fff; font-weight: bold; font-size: 0.9rem; width: 40px; display: inline-block;}
    .csm-bar-bg {background: #333; width: 100%; height: 8px; border-radius: 4px; display: inline-block; width: 150px;}
    .csm-value {color: #a0a0c0; font-size: 0.8rem; margin-left: 10px;}
    
    /* BADGES */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
    .strength-badge {background: linear-gradient(45deg, #00d2ff, #3a7bd5); color: white; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; margin-left: 5px;}

    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; margin-top: 20px; margin-bottom: 10px;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

# Listes complètes pour le calcul du CSM
ALL_MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CAD", "USD_CHF",
    "EUR_GBP", "EUR_AUD", "EUR_NZD", "EUR_JPY", "EUR_CAD", "EUR_CHF",
    "GBP_JPY", "GBP_AUD", "GBP_NZD", "GBP_CAD", "GBP_CHF",
    "AUD_JPY", "AUD_NZD", "AUD_CAD", "AUD_CHF",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "CAD_JPY", "CAD_CHF", "CHF_JPY"
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

# Paires à trader (Scan cible)
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== CLASSES ====================
@dataclass
class TradingParams:
    atr_sl: float
    atr_tp: float
    min_score: int
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
    score: int
    confluences: List[str]
    strength_diff: float # La différence de force (ex: EUR 8.0 - USD 2.0 = 6.0)

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles_safe(pair, tf, count=500):
    """Récupération sécurisée avec micro-pause"""
    time.sleep(0.05) 
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":count, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

# ==================== MOTEUR CSM (CURRENCY STRENGTH) ====================
def calculate_csm() -> Dict[str, float]:
    """Calcule la force de 0 à 10 pour chaque devise sur base du H1"""
    scores = {c: 0.0 for c in CURRENCIES}
    counts = {c: 0 for c in CURRENCIES}
    
    # On récupère la dernière bougie H1 pour toutes les paires majeures
    # Pour voir qui pousse qui en temps réel.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_candles_safe, p, "H1", 2): p for p in ALL_MAJOR_PAIRS}
        
        for f in as_completed(futures):
            pair = futures[f]
            df = f.result()
            if len(df) < 2: continue
            
            # % Change de la dernière bougie close
            open_p = df.iloc[-1]['open']
            close_p = df.iloc[-1]['close']
            if open_p == 0: continue
            
            pct_change = ((close_p - open_p) / open_p) * 100
            
            base, quote = pair.split("_")
            
            # Si Base monte (ex: EURUSD monte), Base gagne, Quote perd
            # Si Base baisse (ex: EURUSD baisse), Base perd, Quote gagne
            
            # On pondère un peu pour que les scores soient lisibles (mult par 10)
            score_delta = pct_change * 10 
            
            if base in scores:
                scores[base] += score_delta
                counts[base] += 1
            if quote in scores:
                scores[quote] -= score_delta
                counts[quote] += 1

    # Normalisation 0 - 10
    final_scores = {}
    
    # On trouve min et max pour normaliser
    vals = list(scores.values())
    if not vals: return {c: 5.0 for c in CURRENCIES}
    
    min_val, max_val = min(vals), max(vals)
    
    for curr, raw_score in scores.items():
        if max_val - min_val == 0:
            norm = 5.0
        else:
            # Formule: (X - Min) / (Max - Min) * 10
            norm = ((raw_score - min_val) / (max_val - min_val)) * 10
        final_scores[curr] = round(norm, 1)
        
    return final_scores

# ==================== ANALYSE TECHNIQUE + CSM ====================
def analyze_market(df: pd.DataFrame, pair: str, tf: str, params: TradingParams, csm: Dict[str, float]) -> Optional[Signal]:
    if len(df) < 200: return None
    
    # --- 1. INDICATEURS ---
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)
    df['hma'] = hma(df['close'], 20)
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    h52, l52 = df['high'].rolling(52).max(), df['low'].rolling(52).min()
    df['ssb'] = ((h52 + l52) / 2).shift(26)

    # FVG
    fvg_bull = any((df['low'] > df['high'].shift(2)).iloc[-5:])
    fvg_bear = any((df['high'] < df['low'].shift(2)).iloc[-5:])

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    timestamp_local = pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ)

    # --- 2. LOGIQUE TECHNIQUE ---
    # Trigger HMA
    hma_rising = curr['hma'] > prev['hma']
    hma_falling = curr['hma'] < prev['hma']
    
    hma_flip_up = hma_rising and (prev['hma'] < df.iloc[-3]['hma'])
    hma_flip_down = hma_falling and (prev['hma'] > df.iloc[-3]['hma'])
    
    is_buy = hma_flip_up if params.strict_flip else hma_rising
    is_sell = hma_flip_down if params.strict_flip else hma_falling
    
    # Trend Filter
    bull_trend = curr['close'] > curr['ema200']
    bear_trend = curr['close'] < curr['ema200']

    action = None
    confluences = []

    if is_buy and bull_trend:
        action = "BUY"
        if curr['close'] > curr['ssb']: confluences.append("Cloud")
        if fvg_bull: confluences.append("FVG")
        confluences.append("Trend")
    elif is_sell and bear_trend:
        action = "SELL"
        if curr['close'] < curr['ssb']: confluences.append("Cloud")
        if fvg_bear: confluences.append("FVG")
        confluences.append("Trend")

    if not action: return None
    if params.use_fvg and "FVG" not in confluences: return None

    # --- 3. CALCUL DU SCORE AVEC CSM (LA CLÉ) ---
    # Récupération de la force des deux devises
    try:
        base_curr, quote_curr = pair.split("_")
        base_strength = csm.get(base_curr, 5.0)
        quote_strength = csm.get(quote_curr, 5.0)
    except:
        base_strength, quote_strength = 5.0, 5.0

    strength_diff = 0.0
    csm_score_bonus = 0

    if action == "BUY":
        # Pour acheter EURUSD, il faut EUR fort et USD faible
        strength_diff = base_strength - quote_strength
    else: # SELL
        # Pour vendre EURUSD, il faut EUR faible et USD fort
        strength_diff = quote_strength - base_strength
    
    # Calcul du Bonus CSM
    # Si diff > 2.0 (Ecart significatif) -> Bonus
    # Si diff > 4.0 (Ecart énorme) -> Gros Bonus
    # Si diff < 0 (Contre le CSM) -> Malus (Trade risqué)
    
    if strength_diff > 4.0: csm_score_bonus = 40 # Confirmed by CSM Meter
    elif strength_diff > 2.0: csm_score_bonus = 20
    elif strength_diff > 0.5: csm_score_bonus = 10
    elif strength_diff < -1.0: csm_score_bonus = -30 # Contradiction CSM
    
    # Score de base technique
    tech_score = 50
    if "Cloud" in confluences: tech_score += 10
    if "FVG" in confluences: tech_score += 10
    
    final_score = tech_score + csm_score_bonus
    final_score = max(10, min(99, int(final_score))) # Clamp 10-99

    if final_score < params.min_score: return None

    atr = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr * params.atr_sl) if action == "BUY" else curr['close'] + (atr * params.atr_sl)
    tp = curr['close'] + (atr * params.atr_tp) if action == "BUY" else curr['close'] - (atr * params.atr_tp)

    # Ajout du tag CSM si valide
    if strength_diff > 2.0: confluences.insert(0, "CSM+")

    return Signal(
        timestamp=timestamp_local, pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=final_score, confluences=confluences, strength_diff=strength_diff
    )

def smart_format(pair: str, price: float) -> str:
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

# ==================== MAIN UI ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v5.0</span> <span class='strength-badge'>CSM EDITION</span>", unsafe_allow_html=True)
    with c2: 
        if st.button("Clear Cache"): 
            st.session_state.scan_results = None
            st.session_state.csm_data = None
            st.rerun()

    # State Init
    if 'scan_results' not in st.session_state: st.session_state.scan_results = None
    if 'csm_data' not in st.session_state: st.session_state.csm_data = None

    # Config
    with st.expander("⚙️ Configuration", expanded=False):
        c1, c2 = st.columns(2)
        sl = c1.number_input("SL xATR", 1.0, 3.0, 1.5)
        tp = c1.number_input("TP xATR", 1.0, 5.0, 3.0)
        sc = c2.slider("Min Score", 50, 95, 70)
        fvg = c2.checkbox("Smart Money (FVG)", True)
        flip = c2.checkbox("Strict Flip", True)

    # --- SCAN LOGIC ---
    if st.button("🚀 SCANNER AVEC CSM", type="primary", use_container_width=True):
        if not client: st.error("API Token Manquant")
        else:
            # 1. Calcul du CSM (Barre de chargement)
            with st.spinner("Analyse des Forces des Devises (CSM)..."):
                csm = calculate_csm()
                st.session_state.csm_data = csm
            
            # 2. Scan Technique
            signals = []
            progress = st.progress(0)
            status = st.empty()
            
            params = TradingParams(sl, tp, sc, fvg, flip)
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
                            s = analyze_market(df, p, tf, params, csm)
                            if s: signals.append(s)
                    except: pass
            
            progress.empty()
            status.empty()
            st.session_state.scan_results = sorted(signals, key=lambda x: x.score, reverse=True)

    # --- AFFICHAGE DASHBOARD CSM ---
    if st.session_state.csm_data:
        csm = st.session_state.csm_data
        st.markdown("#### 📊 Currency Strength Meter (Real-Time)")
        
        # Tri des devises par force
        sorted_csm = sorted(csm.items(), key=lambda x: x[1], reverse=True)
        
        # Affichage en colonnes
        cols = st.columns(8)
        for i, (curr, score) in enumerate(sorted_csm):
            color = "#00ff88" if score > 7 else ("#ff4b4b" if score < 3 else "#a0a0c0")
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center; background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;">
                    <div style="font-weight:bold; color:#fff;">{curr}</div>
                    <div style="font-size:1.5rem; font-weight:bold; color:{color};">{score}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")

    # --- AFFICHAGE RESULTATS ---
    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        # PDF Button
        if signals:
            pdf_buf = BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=landscape(A4), topMargin=10*mm, leftMargin=10*mm, rightMargin=10*mm)
            elems = []
            styles = getSampleStyleSheet()
            
            elems.append(Paragraph("<font size=18 color='#00ff88'><b>BlueStar CSM Report</b></font>", styles["Title"]))
            elems.append(Spacer(1, 10*mm))
            
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if not tf_sigs: continue
                elems.append(Paragraph(f"<font size=12 color='white' backcolor='#00ff88'>&nbsp;<b>{tf}</b>&nbsp;</font>", styles["Normal"]))
                elems.append(Spacer(1, 3*mm))
                data = [["HEURE", "PAIRE", "ACTION", "PRIX", "SCORE", "CONFIRMATIONS"]]
                for s in tf_sigs:
                    col = "#00ff88" if s.action == "BUY" else "#ff6b6b"
                    data.append([
                        s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"),
                        Paragraph(f"<font color='{col}'><b>{s.action}</b></font>", styles["Normal"]),
                        smart_format(s.pair, s.entry_price), str(s.score),
                        Paragraph(f"<font size=8>{', '.join(s.confluences)}</font>", styles["Normal"])
                    ])
                t = Table(data, colWidths=[25*mm, 35*mm, 25*mm, 35*mm, 20*mm, 130*mm])
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1a1f3a")), ('TEXTCOLOR',(0,0),(-1,0),colors.white), ('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
                elems.append(t)
                elems.append(Spacer(1, 5*mm))
            
            doc.build(elems)
            st.download_button("📄 Télécharger PDF", pdf_buf.getvalue(), "BlueStar_CSM.pdf", "application/pdf")

        if not signals: st.info("Aucun signal. Vérifiez si les devises sont en range (Scores CSM proches de 5.0).")
        else:
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3></div>", unsafe_allow_html=True)
                    data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        data.append({
                            "Time": s.timestamp.strftime("%H:%M"), "Pair": s.pair.replace("_","/"),
                            "Signal": f"{icon} {s.action}", "Price": smart_format(s.pair, s.entry_price),
                            "Score": s.score, "Diff Force": f"{s.strength_diff:+.1f}",
                            "Confluences": ", ".join(s.confluences)
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
