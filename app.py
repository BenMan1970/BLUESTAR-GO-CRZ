"""
BlueStar Institutional v6.1 (Visual Strength Edition)
- Raw Strength Logic (24h sum)
- Visual Heatmap for Strength (Red -> Orange -> Green)
- Simplified UI labels
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
st.set_page_config(page_title="BlueStar v6.1", layout="wide", initial_sidebar_state="collapsed")

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
</style>
""", unsafe_allow_html=True)

FOREX_28_PAIRS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "NZD_JPY", "NZD_CHF", "NZD_CAD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY"
]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
SCAN_TARGETS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD","EUR_JPY","GBP_JPY","XAU_USD","US30_USD","NAS100_USD"]
TIMEFRAMES = ["M15", "H1", "H4"]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== CLASSES ====================
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
    raw_strength_diff: float
    confluences: List[str]

# ==================== API ====================
@st.cache_resource
def get_oanda_client():
    try: return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except: return None

client = get_oanda_client()

def get_candles_safe(pair, tf, count=250):
    time.sleep(0.05)
    try:
        r = InstrumentsCandles(instrument=pair, params={"granularity":GRANULARITY_MAP.get(tf,"H1"), "count":count, "price":"M"})
        client.request(r)
        data = [{'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])} for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty: df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except: return pd.DataFrame()

# ==================== LOGIC: RAW STRENGTH ====================
def calculate_raw_strength():
    raw_strengths = {c: 0.0 for c in CURRENCIES}
    for pair in FOREX_28_PAIRS:
        time.sleep(0.05)
        df = get_candles_safe(pair, "D", count=2)
        if len(df) < 1: continue
        
        candle = df.iloc[-1]
        open_p, close_p = candle['open'], candle['close']
        if open_p == 0: continue
        
        pct = ((close_p - open_p) / open_p) * 100
        base, quote = pair.split("_")
        
        if base in raw_strengths: raw_strengths[base] += pct
        if quote in raw_strengths: raw_strengths[quote] -= pct
            
    return {c: round(v, 2) for c, v in raw_strengths.items()}

# ==================== LOGIC: ANALYSIS ====================
def analyze_market(df, pair, tf, params, raw_data):
    if len(df) < 100: return None
    
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

    # Raw Diff
    raw_diff = 0.0
    if "_" in pair and "US30" not in pair and "NAS100" not in pair and "XAU" not in pair:
        try:
            base, quote = pair.split("_")
            s_base = raw_data.get(base, 0.0)
            s_quote = raw_data.get(quote, 0.0)
            if action == "BUY": raw_diff = s_base - s_quote
            else: raw_diff = s_quote - s_base
        except: pass
        
    if raw_diff < -1.0 and ("US30" not in pair and "XAU" not in pair): return None

    atr = (curr['high'] - curr['low'])
    sl = curr['close'] - (atr * params.atr_sl) if action == "BUY" else curr['close'] + (atr * params.atr_sl)
    tp = curr['close'] + (atr * params.atr_tp) if action == "BUY" else curr['close'] - (atr * params.atr_tp)
    
    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        raw_strength_diff=raw_diff, confluences=conf
    )

def smart_format(pair, price):
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

# ==================== HELPERS COLORS ====================
def get_color_hex(value):
    if value >= 1.0: return "#00ff00" # Bright Green
    elif value >= 0.5: return "#aaff00" # Lime
    elif value >= 0.0: return "#ffaa00" # Orange
    else: return "#ff4444" # Red

def style_dataframe(df):
    def color_force_col(val):
        try:
            v = float(val.strip('%').strip('+'))
            color = get_color_hex(v)
            return f'color: {color}; font-weight: bold;'
        except: return ''
        
    return df.style.map(color_force_col, subset=['Force'])

# ==================== MAIN ====================
def main():
    c1, c2 = st.columns([3,1])
    with c1: st.markdown("### BlueStar Institutional <span class='institutional-badge'>v6.1</span>", unsafe_allow_html=True)
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

    if st.button("SCAN", type="primary", use_container_width=True):
        if not client: st.error("Token API Manquant")
        else:
            with st.spinner("Analyse Raw Strength (24h)..."):
                raw_data = calculate_raw_strength()
            
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
                    status.text(f"Scanning... {int((done/total)*100)}%")
                    try:
                        df, p, tf = f.result()
                        if not df.empty:
                            s = analyze_market(df, p, tf, params, raw_data)
                            if s: signals.append(s)
                    except: pass
            
            status.empty()
            progress.empty()
            st.session_state.scan_results = sorted(signals, key=lambda x: x.raw_strength_diff, reverse=True)

    if st.session_state.scan_results:
        signals = st.session_state.scan_results
        
        if signals:
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=10*mm)
            elems = [Paragraph("<b>BlueStar v6 Report</b>", getSampleStyleSheet()['Title']), Spacer(1, 10*mm)]
            
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if not tf_sigs: continue
                elems.append(Paragraph(f"<b>{tf} Structure</b>", getSampleStyleSheet()['Normal']))
                data = [["Time", "Pair", "Action", "Price", "Force", "Conf"]]
                for s in tf_sigs:
                    act_col = colors.green if s.action == "BUY" else colors.red
                    
                    # Color Logic for PDF
                    force_val = s.raw_strength_diff
                    force_col = colors.HexColor(get_color_hex(force_val))
                    
                    data.append([
                        s.timestamp.strftime("%H:%M"), s.pair, 
                        Paragraph(f"<font color='{act_col}'><b>{s.action}</b></font>", getSampleStyleSheet()['Normal']),
                        smart_format(s.pair, s.entry_price), 
                        Paragraph(f"<font color='{force_col}'><b>{s.raw_strength_diff:+.2f}%</b></font>", getSampleStyleSheet()['Normal']),
                        ", ".join(s.confluences)
                    ])
                t = Table(data)
                t.setStyle(TableStyle([('TEXTCOLOR',(0,0),(-1,0),colors.white), ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1a1f3a"))]))
                elems.append(t); elems.append(Spacer(1, 5*mm))
            
            doc.build(elems)
            st.download_button("📄 PDF Report", buf.getvalue(), "bluestar_v6_report.pdf", "application/pdf")

        if not signals: st.info("Aucun signal aligné avec la Force Brute.")
        else:
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3></div>", unsafe_allow_html=True)
                    data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        data.append({
                            "Heure": s.timestamp.strftime("%H:%M"), 
                            "Paire": s.pair.replace("_","/"),
                            "Signal": f"{icon} {s.action}", 
                            "Prix": smart_format(s.pair, s.entry_price),
                            "SL": smart_format(s.pair, s.stop_loss), 
                            "TP": smart_format(s.pair, s.take_profit),
                            "Force": f"{s.raw_strength_diff:+.2f}%",
                            "Confirmations": ", ".join(s.confluences)
                        })
                    
                    # Application du style Pandas avec couleurs
                    df_view = pd.DataFrame(data)
                    st.dataframe(style_dataframe(df_view), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
