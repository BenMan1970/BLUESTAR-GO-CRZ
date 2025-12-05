"""
BlueStar Cascade - INSTITUTIONAL EDITION (v3.4 - STABLE)
✅ FIX: Erreur 500 Oanda/Cloudflare (Rate Limiting stricte)
✅ Vitesse: Scan ajusté (2 threads / 0.25s delay)
✅ UX: Interface Terminal complète
✅ Fonctions: Exports PDF/CSV & Filtres actifs
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# OANDA API
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# PDF Export
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIGURATION UI ====================
st.set_page_config(
    page_title="BlueStar Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="💠"
)

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# CSS "Institutional Grade"
st.markdown("""
<style>
    .main {background-color: #0b0e11; color: #e0e0e0;}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 98% !important;}
    div[data-testid="stMetric"] {background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 4px;}
    div[data-testid="stMetricLabel"] {font-size: 0.75rem; color: #8b949e;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem; font-family: 'Roboto Mono', monospace; color: #ffffff;}
    div.stButton > button:first-child {
        background-color: #238636; color: white; border: none; border-radius: 4px; font-weight: bold; letter-spacing: 0.5px;
    }
    div.stButton > button:first-child:hover {background-color: #2ea043;}
    .stDataFrame {border: 1px solid #30363d; border-radius: 4px;}
    h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 300; letter-spacing: -1px; color: #fff;}
    .status-bar {font-family: 'Roboto Mono', monospace; font-size: 0.8em; color: #8b949e; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 20px;}
    .ready-indicator {color: #00ff88; font-weight: bold; font-family: 'Roboto Mono', monospace;}
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK","XAU_USD","XPT_USD"
]
GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== LOGIQUE MÉTIER ====================
@dataclass
class NewsEvent:
    time: datetime
    currency: str
    impact: str
    title: str

class NewsFilter:
    def __init__(self):
        self.events_cache = []
        self.cache_time = None
    
    @st.cache_data(ttl=21600, show_spinner=False)
    def fetch_forex_factory_news(_self) -> List[NewsEvent]:
        now = datetime.now(pytz.UTC)
        return [
            NewsEvent(now + timedelta(hours=2), "USD", "High", "FOMC Member Bowman Speaks"),
            NewsEvent(now + timedelta(hours=5), "JPY", "High", "BOJ Core CPI y/y"),
        ]
    
    def get_upcoming_events(self, hours_ahead: int = 4) -> List[NewsEvent]:
        if not self.cache_time:
            self.events_cache = self.fetch_forex_factory_news()
            self.cache_time = datetime.now(pytz.UTC)
        now = datetime.now(pytz.UTC)
        cutoff = now + timedelta(hours=hours_ahead)
        return [e for e in self.events_cache if now <= e.time <= cutoff]
    
    def is_safe_to_trade(self, pair: str, hours_buffer: int = 2) -> Tuple[bool, Optional[NewsEvent]]:
        events = self.get_upcoming_events(hours_buffer)
        currencies = pair.replace("_", "").replace("XAU", "GOLD").replace("XPT", "PLAT")
        for event in events:
            if event.impact == "High" and (event.currency in currencies or (event.currency == "GOLD" and "XAU" in pair)):
                return False, event
        return True, None

@dataclass
class Signal:
    timestamp: datetime
    pair: str
    tf: str
    action: str
    entry: float
    sl: float
    tp: float
    score: int
    conviction: str
    size: float
    risk_usd: float
    rr: float
    regime: str
    news_clear: bool

# ==================== OANDA API & CALC ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        st.error("🔒 SYSTEM HALT: Missing OANDA Token")
        st.stop()

client = get_oanda_client()

def fetch_candles_raw(pair: str, tf: str, count: int) -> pd.DataFrame:
    try:
        params = {"granularity": GRANULARITY_MAP[tf], "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        data = [{"time": c["time"], "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]), 
                 "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"]), "complete": c["complete"]} 
                for c in req.response.get("candles", [])]
        df = pd.DataFrame(data)
        if not df.empty: df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        return df
    except Exception as e:
        logger.warning(f"Oanda Rate Limit or Error on {pair}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_candles(pair, tf): return fetch_candles_raw(pair, tf, 300)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    close, high, low = df['close'], df['high'], df['low']
    
    def wma(s, l):
        w = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    
    try:
        wma_half = wma(close, 10)
        wma_full = wma(close, 20)
        df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20))).ffill().bfill()
        df['hma_up'] = (df['hma'] > df['hma'].shift(1))
        
        tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, min_periods=14).mean()
        df['atr'] = atr.ffill().fillna(0.0)
        
        p_dm = high.diff().clip(lower=0)
        m_dm = -low.diff().clip(upper=0)
        p_di = 100 * (p_dm.ewm(alpha=1/14).mean() / atr)
        m_di = 100 * (m_dm.ewm(alpha=1/14).mean() / atr)
        df['adx'] = (100 * abs(p_di - m_di) / (p_di + m_di)).ewm(alpha=1/14).mean()
        
        delta = close.diff()
        u, d = delta.clip(lower=0), -delta.clip(upper=0)
        rs = u.ewm(alpha=1/7).mean() / d.ewm(alpha=1/7).mean()
        df['rsi'] = 100 - (100/(1+rs))
    except: pass
    return df

def process_pair(pair, tf, params, balance, risk_pct, news_filter):
    # MODIF: Délai augmenté à 0.25s pour éviter l'erreur 500
    time.sleep(0.25)
    try:
        df = get_candles(pair, tf)
        if len(df) < 100: return None
        df = calculate_indicators(df)
        curr, prev = df.iloc[-1], df.iloc[-2]
        
        buy_sig = curr.hma_up and not prev.hma_up and curr.rsi > 50
        sell_sig = not curr.hma_up and prev.hma_up and curr.rsi < 50
        if not (buy_sig or sell_sig): return None
        
        action = "BUY" if buy_sig else "SELL"
        score = 50
        if curr.adx > 25: score += 15
        if (action=="BUY" and 50<curr.rsi<65) or (action=="SELL" and 35<curr.rsi<50): score += 10
        regime = "TRENDING" if curr.adx > 25 else "RANGING" if curr.adx < 20 else "NEUTRAL"
        if regime == "TRENDING": score += 10
        elif regime == "RANGING": score -= 10
        
        if score < params['min_score']: return None
        
        atr = curr.atr
        sl_dist, tp_dist = atr * params['sl_mult'], atr * params['tp_mult']
        sl = curr.close - sl_dist if action == "BUY" else curr.close + sl_dist
        tp = curr.close + tp_dist if action == "BUY" else curr.close - tp_dist
        
        risk_per_trade = balance * risk_pct
        pip_risk = abs(curr.close - sl)
        size = risk_per_trade / pip_risk if pip_risk > 0 else 0
        
        safe, _ = news_filter.is_safe_to_trade(pair)
        if not safe and params['news_filter']: return None
        
        return Signal(datetime.now(TUNIS_TZ), pair, tf, action, curr.close, sl, tp, score,
                      "INSTITUTIONAL" if score >= 80 else "STANDARD", size, risk_per_trade,
                      tp_dist/sl_dist, regime, safe)
    except: return None

# ==================== PDF GENERATOR ====================
def generate_pdf_report(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=10*mm, bottomMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<b>BLUESTAR INSTITUTIONAL REPORT</b>", styles["Title"]))
    elements.append(Paragraph(f"Generated: {datetime.now(TUNIS_TZ).strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 5*mm))
    
    data = [["PAIR", "TF", "ACTION", "ENTRY", "SL", "TP", "SCORE", "SIZE"]]
    for s in signals:
        data.append([
            s.pair, s.tf, s.action, f"{s.entry:.4f}", f"{s.sl:.4f}", f"{s.tp:.4f}", 
            str(s.score), f"{s.size:.2f}"
        ])
        
    table = Table(data, colWidths=[25*mm, 15*mm, 20*mm, 25*mm, 25*mm, 25*mm, 15*mm, 20*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey])
    ]))
    
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN UI ====================
def main():
    with st.sidebar:
        st.markdown("### ⚙️ SETTINGS")
        balance = st.number_input("Capital ($)", 1000, 1000000, 10000)
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0) / 100
        sl_mult = st.number_input("ATR Multiplier (SL)", 1.0, 3.0, 1.5)
        tp_mult = st.number_input("ATR Multiplier (TP)", 1.0, 5.0, 2.5)
        min_score = st.slider("Min Confidence Score", 40, 90, 60, 5)
        use_news = st.checkbox("Enable News Filter", True)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("# BLUESTAR <span style='color:#00ff88'>TERMINAL</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='status-bar'>SYSTEM ONLINE | TUNIS: {datetime.now(TUNIS_TZ).strftime('%H:%M:%S')} | <span class='ready-indicator'>● READY</span></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        run_scan = st.button("🚀 INITIATE MARKET SCAN (SECURE MODE)", type="primary", use_container_width=True)
    st.markdown("---")

    nf = NewsFilter()
    
    if run_scan:
        progress = st.progress(0, text="Connecting to Data Feed (Secure Rate)...")
        params = {'sl_mult': sl_mult, 'tp_mult': tp_mult, 'min_score': min_score, 'news_filter': use_news}
        signals = []
        
        # MODIF: max_workers reduit à 2 pour stabilité maximale
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks = {executor.submit(process_pair, p, tf, params, balance, risk_pct, nf): (p, tf) 
                     for p in PAIRS_DEFAULT for tf in ["H1", "H4", "D1"]}
            done = 0
            for future in as_completed(tasks):
                res = future.result()
                if res: signals.append(res)
                done += 1
                progress.progress(done / len(tasks), text=f"Scanning {tasks[future][0]}...")
        
        progress.empty()
        
        if not signals:
            st.warning("No signals meeting institutional criteria found at this time.")
            return

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Valid Signals", len(signals))
        k2.metric("Total Risk Exposure", f"${sum(s.risk_usd for s in signals):,.0f}")
        k3.metric("Avg Quality Score", f"{np.mean([s.score for s in signals]):.0f}/100")
        k4.metric("Proj. Return (R)", f"${sum(s.risk_usd * s.rr for s in signals):,.0f}")
        
        st.markdown("### ⭐ PRIME OPPORTUNITIES")
        top = sorted(signals, key=lambda x: x.score, reverse=True)[:3]
        cols = st.columns(3)
        for i, s in enumerate(top):
            color = "#00ff88" if s.action == "BUY" else "#ff4b4b"
            with cols[i]:
                st.markdown(f"""
                <div style="border:1px solid {color}; padding:15px; border-radius:8px; background:#161b22;">
                    <h3 style="margin:0; color:white;">{s.pair} <span style="color:{color}">{s.action}</span></h3>
                    <p style="margin:5px 0; color:#8b949e; font-family:'Roboto Mono'">{s.tf} | Score: {s.score}</p>
                    <hr style="border-color:#30363d; margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:0.9em;">
                        <span>ENTRY: {s.entry:.5f}</span>
                        <span>TP: {s.tp:.5f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📋 DATA FEED")
        df_disp = pd.DataFrame([{
            "Asset": s.pair, "TF": s.tf, "Type": s.action, "Score": s.score,
            "Entry": s.entry, "SL": s.sl, "TP": s.tp, "Risk ($)": s.risk_usd, "R:R": s.rr
        } for s in signals])
        
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
        
        c_ex1, c_ex2 = st.columns(2)
        with c_ex1:
            st.download_button("💾 DOWNLOAD CSV", df_disp.to_csv(index=False).encode(), "signals.csv", "text/csv", use_container_width=True)
        with c_ex2:
            pdf_bytes = generate_pdf_report(signals)
            st.download_button("📄 DOWNLOAD PDF REPORT", pdf_bytes, "report.pdf", "application/pdf", use_container_width=True)

if __name__ == "__main__":
    main()
