"""
BlueStar Institutional v6.2 Enhanced
- Raw Strength Logic from v6.2
- Premium Visual Style from v3.0
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

# PDF Export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional v6.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 1rem !important; max-width: 100% !important;}
    
    /* STYLE METRIQUE */
    .stMetric {background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.75rem !important; font-weight: 500 !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700;}
    
    /* BADGES */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; display: inline-block;}
    .v62-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; display: inline-block; margin-left: 8px;}
    
    /* SUPPRESSION QUADRILLAGE ET BORDURES */
    .stDataFrame {font-size: 0.75rem !important;}
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stDataFrame"] div[role="grid"] {border: none !important;}
    [data-testid="stDataFrame"] div[role="row"] {border: none !important; background-color: transparent !important;}
    [data-testid="stDataFrame"] div[role="columnheader"] {background-color: rgba(255,255,255,0.05) !important; border-bottom: 1px solid rgba(255,255,255,0.1) !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    /* HEADERS ET TEXTES */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,255,0.1)); padding: 12px 20px; border-radius: 8px; text-align: center; margin-bottom: 15px; border: 1px solid rgba(0,255,136,0.2); box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.3rem; font-weight: 700;}
    .tf-header p {margin: 5px 0 0 0; color: #a0a0c0; font-size: 0.75rem;}
    h1 {font-size: 2rem !important; margin-bottom: 0.5rem !important; font-weight: 700 !important;}
    
    /* STATUS BOXES */
    .alert-box {background: rgba(255,200,0,0.1); border-left: 4px solid #ffc800; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 0.85rem;}
    .success-box {background: rgba(0,255,136,0.1); border-left: 4px solid #00ff88; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 0.85rem;}
    .info-box {background: rgba(0,200,255,0.1); border-left: 4px solid #00ccff; padding: 12px; border-radius: 6px; margin: 10px 0; font-size: 0.85rem;}
    
    /* SESSION BADGES */
    .session-badge {padding: 3px 8px; border-radius: 12px; font-size: 0.65rem; font-weight: bold; margin-left: 8px;}
    .session-london {background: #ff6b6b; color: white;}
    .session-ny {background: #4ecdc4; color: white;}
    .session-tokyo {background: #ffe66d; color: black;}
    
    /* FORCE COLORS IN TABLE */
    .force-strong {color: #00ff00 !important; font-weight: 700;}
    .force-good {color: #aaff00 !important; font-weight: 700;}
    .force-medium {color: #ffaa00 !important; font-weight: 700;}
    .force-weak {color: #ff4444 !important; font-weight: 700;}
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

# ==================== UTILITIES ====================
def get_active_session(dt: datetime) -> str:
    hour_utc = dt.astimezone(pytz.UTC).hour
    if 0 <= hour_utc < 9: return "Tokyo"
    elif 8 <= hour_utc < 17: return "London"
    elif 13 <= hour_utc < 22: return "NY"
    else: return "Off-Hours"

def get_session_badge(session: str) -> str:
    badges = {
        "London": "<span class='session-badge session-london'>LONDON</span>",
        "NY": "<span class='session-badge session-ny'>NY</span>",
        "Tokyo": "<span class='session-badge session-tokyo'>TOKYO</span>",
        "Off-Hours": "<span class='session-badge' style='background:#666;color:white;'>OFF</span>"
    }
    return badges.get(session, "")

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
    session: str

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
    
    local_time = pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ) if curr['time'].tzinfo is None else curr['time'].astimezone(TUNIS_TZ)
    
    return Signal(
        timestamp=local_time,
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        raw_strength_diff=raw_diff, confluences=conf,
        session=get_active_session(local_time)
    )

def smart_format(pair, price):
    if "JPY" in pair: return f"{price:.3f}"
    elif "US30" in pair or "NAS100" in pair: return f"{price:.1f}"
    elif "XAU" in pair: return f"{price:.2f}"
    else: return f"{price:.5f}"

def get_force_class(value):
    if value >= 1.0: return "force-strong"
    elif value >= 0.5: return "force-good"
    elif value >= 0.0: return "force-medium"
    else: return "force-weak"

# ==================== MAIN ====================
def main():
    col_title, col_time = st.columns([3, 2])
    
    with col_title:
        st.markdown("# BlueStar Institutional")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL</span><span class="v62-badge">v6.2 Enhanced</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(TUNIS_TZ)
        market_open = now_tunis.hour in range(0, 23)
        session = get_active_session(now_tunis)
        st.markdown(f"""<div style='text-align: right; padding-top: 10px;'>
            <span style='color: #a0a0c0; font-size: 0.85rem;'>🕐 {now_tunis.strftime('%H:%M:%S')}</span><br>
            <span style='color: {"#00ff88" if market_open else "#ff6666"}; font-weight: 700;'>{"MARKET OPEN" if market_open else "CLOSED"}</span> {get_session_badge(session)}
        </div>""", unsafe_allow_html=True)

    if 'scan_results' not in st.session_state: 
        st.session_state.scan_results = None
        st.session_state.raw_strength = None
        st.session_state.scan_duration = 0

    with st.expander("⚙️ Configuration Avancée", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        sl = c1.number_input("SL Multiplier (xATR)", 1.0, 3.0, 1.5, 0.1)
        tp = c2.number_input("TP Multiplier (xATR)", 1.5, 5.0, 3.0, 0.1)
        fvg = c3.checkbox("FVG Required", True)
        flip = c4.checkbox("Strict Flip Only", True)

    if st.button("🚀 SCAN MARKET", type="primary", use_container_width=True):
        if not client: 
            st.error("⚠️ Token API Manquant - Vérifiez vos secrets Streamlit")
        else:
            start_time = time.time()
            
            with st.spinner("📊 Calcul de la Force Brute (24h)..."):
                raw_data = calculate_raw_strength()
                st.session_state.raw_strength = raw_data
            
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
                    status.text(f"🔍 Scanning Market Structure... {int((done/total)*100)}%")
                    try:
                        df, p, tf = f.result()
                        if not df.empty:
                            s = analyze_market(df, p, tf, params, raw_data)
                            if s: signals.append(s)
                    except: pass
            
            status.empty()
            progress.empty()
            
            st.session_state.scan_results = sorted(signals, key=lambda x: x.raw_strength_diff, reverse=True)
            st.session_state.scan_duration = time.time() - start_time
            
            if signals:
                st.success(f"✅ Scan terminé - {len(signals)} signaux institutionnels détectés")
            else:
                st.info("ℹ️ Aucun signal aligné avec la Force Brute détecté")

    # ==================== DISPLAY RESULTS ====================
    if st.session_state.scan_results is not None:
        signals = st.session_state.scan_results
        
        # METRICS ROW
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        total_signals = len(signals)
        buy_signals = len([s for s in signals if s.action == "BUY"])
        sell_signals = len([s for s in signals if s.action == "SELL"])
        avg_force = sum(s.raw_strength_diff for s in signals) / len(signals) if signals else 0
        
        m1.metric("Total Signaux", total_signals)
        m2.metric("BUY Signals", buy_signals)
        m3.metric("SELL Signals", sell_signals)
        m4.metric("Force Moyenne", f"{avg_force:+.2f}%")
        m5.metric("Durée Scan", f"{st.session_state.scan_duration:.1f}s")
        
        # CURRENCY STRENGTH DISPLAY
        if st.session_state.raw_strength:
            st.markdown("---")
            st.markdown("### 📈 Force Brute des Devises (24h)")
            
            raw_data = st.session_state.raw_strength
            sorted_currencies = sorted(raw_data.items(), key=lambda x: x[1], reverse=True)
            
            cols = st.columns(8)
            for idx, (curr, strength) in enumerate(sorted_currencies):
                with cols[idx]:
                    color = "#00ff88" if strength > 0 else "#ff6b6b"
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                        <div style='font-size: 1.1rem; font-weight: 700; color: #a0a0c0;'>{curr}</div>
                        <div style='font-size: 1.3rem; font-weight: 700; color: {color};'>{strength:+.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        if signals:
            st.markdown("---")
            
            # DOWNLOAD BUTTONS
            d1, d2 = st.columns(2)
            with d1:
                df_exp = pd.DataFrame([{
                    'Time': s.timestamp.strftime('%H:%M'),
                    'Pair': s.pair,
                    'TF': s.timeframe,
                    'Action': s.action,
                    'Entry': s.entry_price,
                    'SL': s.stop_loss,
                    'TP': s.take_profit,
                    'Force': s.raw_strength_diff,
                    'Confirmations': ', '.join(s.confluences),
                    'Session': s.session
                } for s in signals])
                st.download_button("📥 Export CSV", df_exp.to_csv(index=False).encode(), f"bluestar_{datetime.now().strftime('%H%M')}.csv", "text/csv")
            
            with d2:
                # PDF Generation
                buf = BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=10*mm)
                elems = [Paragraph("<b>BlueStar v6.2 Enhanced Report</b>", getSampleStyleSheet()['Title']), Spacer(1, 10*mm)]
                
                for tf in TIMEFRAMES:
                    tf_sigs = [s for s in signals if s.timeframe == tf]
                    if not tf_sigs: continue
                    elems.append(Paragraph(f"<b>{tf} Structure</b>", getSampleStyleSheet()['Normal']))
                    data = [["Time", "Pair", "Action", "Price", "SL", "TP", "Force", "Conf"]]
                    for s in tf_sigs:
                        data.append([
                            s.timestamp.strftime("%H:%M"), s.pair, s.action,
                            smart_format(s.pair, s.entry_price),
                            smart_format(s.pair, s.stop_loss),
                            smart_format(s.pair, s.take_profit),
                            f"{s.raw_strength_diff:+.2f}%",
                            ", ".join(s.confluences)
                        ])
                    t = Table(data)
                    t.setStyle(TableStyle([
                        ('TEXTCOLOR',(0,0),(-1,0),colors.white), 
                        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1a1f3a")),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333"))
                    ]))
                    elems.append(t)
                    elems.append(Spacer(1, 5*mm))
                
                doc.build(elems)
                st.download_button("📄 Export PDF", buf.getvalue(), "bluestar_v62_report.pdf", "application/pdf")

            # SIGNALS DISPLAY BY TIMEFRAME
            st.markdown("---")
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"""<div class='tf-header'>
                        <h3>{tf} Market Structure</h3>
                        <p>{len(tf_sigs)} Signal(s) Institutionnel(s)</p>
                    </div>""", unsafe_allow_html=True)
                    
                    data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        force_class = get_force_class(s.raw_strength_diff)
                        
                        data.append({
                            "Heure": s.timestamp.strftime("%H:%M"), 
                            "Paire": s.pair.replace("_","/"),
                            "Signal": f"{icon} {s.action}", 
                            "Prix": smart_format(s.pair, s.entry_price),
                            "SL": smart_format(s.pair, s.stop_loss), 
                            "TP": smart_format(s.pair, s.take_profit),
                            "Force": f"{s.raw_strength_diff:+.2f}%",
                            "Confirmations": ", ".join(s.confluences),
                            "Session": s.session
                        })
                    
                    df_view = pd.DataFrame(data)
                    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 0.75rem;'>BlueStar Institutional v6.2 Enhanced | Professional Grade Algorithm</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
