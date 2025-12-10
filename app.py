"""
BlueStar Institutional v4.1 (Stable Engine)
Fixes the "Freeze" issue by adding Rate Limiting and Progress Bar
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
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
st.set_page_config(page_title="BlueStar v4.1", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; margin-top: 20px; margin-bottom: 10px;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem;}
    /* Style de la barre de progression */
    .stProgress > div > div > div > div { background-color: #00ff88; }
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== DATA CLASSES ====================
@dataclass
class TradingParams:
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_score: int
    use_fvg: bool
    strict_flip: bool
    session_filter: bool

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

# ==================== UTILS ====================
def is_active_session(dt: datetime) -> bool:
    hour = dt.hour
    if 8 <= hour < 22: return True
    return False

def smart_format(pair: str, price: float) -> str:
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair or "SPX" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

# ==================== API & ENGINE (CORRIGÉ) ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles_safe(pair, tf):
    """Récupère les bougies avec sécurité anti-crash"""
    # 1. Petite pause pour ne pas spammer l'API
    time.sleep(0.15) 
    
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":500, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except V20Error:
        return pd.DataFrame() # Erreur API connue
    except Exception:
        return pd.DataFrame() # Autre erreur

def analyze_market(df: pd.DataFrame, pair: str, tf: str, params: TradingParams) -> Optional[Signal]:
    if len(df) < 200: return None
    
    # --- INDICATORS ---
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)
    df['hma'] = hma(df['close'], 20)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    h52 = df['high'].rolling(52).max(); l52 = df['low'].rolling(52).min()
    df['ssb'] = ((h52 + l52) / 2).shift(26)

    # --- LOGIC ---
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    timestamp_local = pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ)

    if params.session_filter and not is_active_session(timestamp_local): return None

    hma_rising = curr['hma'] > prev['hma']
    hma_falling = curr['hma'] < prev['hma']
    
    # Strict Flip Logic
    hma_flip_up = hma_rising and (prev['hma'] < df.iloc[-3]['hma'])
    hma_flip_down = hma_falling and (prev['hma'] > df.iloc[-3]['hma'])
    
    is_buy = hma_flip_up if params.strict_flip else hma_rising
    is_sell = hma_flip_down if params.strict_flip else hma_falling

    bull_trend = curr['close'] > curr['ema200']
    bear_trend = curr['close'] < curr['ema200']

    fvg_bull = any((df['low'] > df['high'].shift(2)).iloc[-5:])
    fvg_bear = any((df['high'] < df['low'].shift(2)).iloc[-5:])

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

    # Scoring
    score = 60
    if "Cloud" in confluences: score += 10
    if "FVG" in confluences: score += 10
    
    rsi = curr['rsi']
    if action == "BUY":
        if 40 <= rsi <= 65: score += 10
        elif rsi > 70: score -= 15
    else:
        if 35 <= rsi <= 60: score += 10
        elif rsi < 30: score -= 15
        
    score = max(50, min(99, int(score)))

    if params.use_fvg and "FVG" not in confluences: return None
    if score < params.min_score: return None

    atr_val = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr_val * params.atr_sl_multiplier) if action == "BUY" else curr['close'] + (atr_val * params.atr_sl_multiplier)
    tp = curr['close'] + (atr_val * params.atr_tp_multiplier) if action == "BUY" else curr['close'] - (atr_val * params.atr_tp_multiplier)

    return Signal(
        timestamp=timestamp_local, pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=score, confluences=confluences
    )

def scan_with_progress(pairs, tfs, params):
    signals = []
    total_tasks = len(pairs) * len(tfs)
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Max Workers réduit à 4 pour éviter le freeze OANDA
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(lambda p,t: (get_candles_safe(p,t), p, t), p, tf): (p,tf) for p in pairs for tf in tfs}
        
        completed = 0
        for f in as_completed(futures):
            try:
                completed += 1
                # Mise à jour progression
                progress = completed / total_tasks
                progress_bar.progress(progress)
                status_text.text(f"Scanning... {int(progress*100)}%")
                
                df, p, tf = f.result()
                if not df.empty:
                    s = analyze_market(df, p, tf, params)
                    if s: signals.append(s)
            except: pass
            
    status_text.empty()
    progress_bar.empty()
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=10*mm, leftMargin=10*mm, rightMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<font size=20 color='#00ff88'><b>BlueStar Institutional Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph(f"Scan Time: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    for tf in ["M15", "H1", "H4"]:
        tf_sigs = [s for s in signals if s.timeframe == tf]
        if not tf_sigs: continue
        
        elements.append(Paragraph(f"<font size=14 color='white' backcolor='#00ff88'>&nbsp;<b>{tf} STRUCTURE</b>&nbsp;</font>", styles["Normal"]))
        elements.append(Spacer(1, 5*mm))
        
        data = [["HEURE", "PAIRE", "ACTION", "PRIX", "SCORE", "CONFIRMATIONS"]]
        for s in tf_sigs:
            col = "#00ff88" if s.action == "BUY" else "#ff6b6b"
            p_str = smart_format(s.pair, s.entry_price)
            data.append([
                s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"),
                Paragraph(f"<font color='{col}'><b>{s.action}</b></font>", styles["Normal"]),
                p_str, f"{s.score}", 
                Paragraph(f"<font size=9>{', '.join(s.confluences)}</font>", styles["Normal"])
            ])
            
        t = Table(data, colWidths=[25*mm, 35*mm, 25*mm, 35*mm, 20*mm, 130*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#555")),
            ('ROWBACKGROUNDS', (1,0), (-1,-1), [colors.white, colors.HexColor("#f0f0f0")]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10*mm))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== MAIN ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v4.1</span>", unsafe_allow_html=True)
    with c2: 
        if st.button("Clear Cache"): 
            st.session_state.scan_results = None
            st.rerun()

    if 'scan_results' not in st.session_state: st.session_state.scan_results = None

    with st.expander("⚙️ Configuration Avancée", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sl = st.number_input("SL xATR", 1.0, 3.0, 1.5)
            tp = st.number_input("TP xATR", 1.0, 5.0, 3.0)
            sc = st.slider("Score Min", 50, 95, 65)
        with c2:
            st.markdown("**Filtres**")
            use_fvg = st.checkbox("Smart Money (FVG)", value=True)
            strict_flip = st.checkbox("Strict HMA Flip", value=True)
            sess_filt = st.checkbox("Session Filter", value=False)

    if st.button("SCAN MARKET (SAFE MODE)", type="primary", use_container_width=True):
        if not client: st.error("No API Token")
        else:
            with st.spinner("Initializing Scan Engine..."):
                params = TradingParams(sl, tp, sc, use_fvg, strict_flip, sess_filt)
                # Utilisation de la nouvelle fonction avec progression
                st.session_state.scan_results = scan_with_progress(PAIRS_DEFAULT, TIMEFRAMES, params)

    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        st.markdown("###")
        if signals:
            st.download_button("📄 Télécharger PDF", generate_pdf(signals), f"BlueStar_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")
        
        if not signals: st.info("No Signals found.")
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
                            "Signal": f"{icon} {s.action}", 
                            "Price": smart_format(s.pair, s.entry_price),
                            "SL": smart_format(s.pair, s.stop_loss), "TP": smart_format(s.pair, s.take_profit),
                            "Score": s.score, "Confluences": ", ".join(s.confluences)
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
