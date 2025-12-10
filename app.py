"""
BlueStar Institutional v3.4 (Nuanced Scoring & Organized PDF)
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
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

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar v3.4", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
    .tf-badge {background: #00ff88; color: #000; padding: 2px 8px; border-radius: 4px; font-weight: bold;}
    
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; margin-top: 15px; margin-bottom: 5px;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

@dataclass
class TradingParams:
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_score: int
    use_fvg: bool

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

# ==================== LOGIC ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles(pair, tf):
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":500, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

def analyze_intraday(df: pd.DataFrame, pair: str, tf: str, params: TradingParams) -> Optional[Signal]:
    if len(df) < 200: return None
    
    # Indics
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # HMA
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)
    df['hma'] = hma(df['close'], 20)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Cloud proxy
    h52 = df['high'].rolling(52).max(); l52 = df['low'].rolling(52).min()
    df['ssb'] = ((h52 + l52) / 2).shift(26) # Kumo B projected back

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Logic
    action = None
    bull_trend = curr['close'] > curr['ema200']
    bear_trend = curr['close'] < curr['ema200']
    hma_green = curr['hma'] > prev['hma']
    hma_red = curr['hma'] < prev['hma']

    # FVG
    fvg_bull = any((df['low'] > df['high'].shift(2)).iloc[-5:])
    fvg_bear = any((df['high'] < df['low'].shift(2)).iloc[-5:])

    confluences = []
    
    if hma_green and bull_trend:
        action = "BUY"
        if curr['close'] > curr['ssb']: confluences.append("Cloud")
        if fvg_bull: confluences.append("FVG")
        if curr['close'] > curr['ema200']: confluences.append("Trend")
            
    elif hma_red and bear_trend:
        action = "SELL"
        if curr['close'] < curr['ssb']: confluences.append("Cloud")
        if fvg_bear: confluences.append("FVG")
        if curr['close'] < curr['ema200']: confluences.append("Trend")

    if not action: return None

    # --- NOUVEAU SCORING NUANCÉ ---
    # Base starts at 60 (Passable)
    score = 60
    
    # 1. Structure Quality (+10 per major confluence)
    if "Cloud" in confluences: score += 10
    if "FVG" in confluences: score += 10
    if "Trend" in confluences: score += 10
    
    # 2. RSI Health Check (Momentum)
    rsi = curr['rsi']
    if action == "BUY":
        if 40 <= rsi <= 65: score += 5  # Perfect buy zone
        elif rsi > 70: score -= 10      # Overbought danger
    else: # SELL
        if 35 <= rsi <= 60: score += 5  # Perfect sell zone
        elif rsi < 30: score -= 10      # Oversold danger
        
    # 3. Volatility Check (Body size relative to ATR)
    atr = (curr['high'] - curr['low'])
    avg_atr = df['high'].diff().abs().rolling(14).mean().iloc[-1]
    if atr > avg_atr * 1.2: score += 4 # Strong impulsive candle
    
    score = max(50, min(99, int(score))) # Cap between 50 and 99

    if params.use_fvg and "FVG" not in confluences: return None
    if score < params.min_score: return None

    atr_val = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr_val * params.atr_sl_multiplier) if action == "BUY" else curr['close'] + (atr_val * params.atr_sl_multiplier)
    tp = curr['close'] + (atr_val * params.atr_tp_multiplier) if action == "BUY" else curr['close'] - (atr_val * params.atr_tp_multiplier)

    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=score, confluences=confluences
    )

def scan_thread(pairs, tfs, params):
    signals = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(lambda p,t: (get_candles(p,t), p, t), p, tf): (p,tf) for p in pairs for tf in tfs}
        for f in as_completed(futures):
            try:
                df, p, tf = f.result()
                if not df.empty:
                    s = analyze_intraday(df, p, tf, params)
                    if s: signals.append(s)
            except: pass
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== ORGANIZED PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    elements.append(Paragraph("<font size=20 color='#00ff88'><b>BlueStar Intraday Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph(f"Scan Time: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    # Loop through Timeframes to organize PDF
    for tf in ["M15", "H1", "H4"]:
        tf_sigs = [s for s in signals if s.timeframe == tf]
        if not tf_sigs: continue
        
        # Header for Section
        elements.append(Paragraph(f"<font size=14 color='white' backcolor='#00ff88'>&nbsp;<b>{tf} STRUCTURE</b>&nbsp;</font>", styles["Normal"]))
        elements.append(Spacer(1, 3*mm))
        
        data = [["HEURE", "PAIRE", "ACTION", "PRIX", "SCORE", "CONFIRMATIONS"]]
        for s in tf_sigs:
            col = "#00ff88" if s.action == "BUY" else "#ff6b6b"
            conf_str = ", ".join(s.confluences)
            data.append([
                s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"),
                Paragraph(f"<font color='{col}'><b>{s.action}</b></font>", styles["Normal"]),
                f"{s.entry_price:.5f}", f"{s.score}", 
                Paragraph(f"<font size=9>{conf_str}</font>", styles["Normal"])
            ])
            
        t = Table(data, colWidths=[25*mm, 35*mm, 25*mm, 35*mm, 20*mm, 100*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")), # Header row
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#555")),
            ('ROWBACKGROUNDS', (1,0), (-1,-1), [colors.white, colors.HexColor("#f0f0f0")]) # Striped rows
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10*mm))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== MAIN ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v3.4</span>", unsafe_allow_html=True)
    with c2: 
        if st.button("Clear Cache"): 
            st.session_state.scan_results = None
            st.rerun()

    if 'scan_results' not in st.session_state: st.session_state.scan_results = None

    with st.expander("Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        sl = c1.number_input("SL xATR", 1.0, 3.0, 1.5)
        tp = c2.number_input("TP xATR", 1.0, 5.0, 3.0)
        sc = c3.slider("Score Min", 50, 95, 65)

    if st.button("SCAN MARKET", type="primary", use_container_width=True):
        if not client: st.error("No API Token")
        else:
            with st.spinner("Analyzing..."):
                params = TradingParams(sl, tp, sc, True)
                st.session_state.scan_results = scan_thread(PAIRS_DEFAULT, TIMEFRAMES, params)

    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        # PDF Button
        st.markdown("###")
        if signals:
            st.download_button("📥 DOWNLOAD ORGANIZED PDF", generate_pdf(signals), f"BlueStar_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")
        
        # Display Grouped
        if not signals: st.info("No Signals")
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
                            "Signal": f"{icon} {s.action}", "Price": f"{s.entry_price:.5f}",
                            "SL": f"{s.stop_loss:.5f}", "TP": f"{s.take_profit:.5f}",
                            "Score": s.score, "Confluences": ", ".join(s.confluences)
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
   
