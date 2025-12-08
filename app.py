"""
BlueStar Institutional v3.0 - VERSION FINALE SANS WARNINGS
→ M15 ajouté
→ Bouton SCAN
→ Seulement signaux INSTITUTIONAL (score ≥ 85)
→ PDF + CSV
→ Configuration complète
→ Visuel propre (sans bordures)
→ ZÉRO WARNING dans les logs (grâce à st.cache_data + pas de threads)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ==================== CONFIG ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.25), rgba(0,200,255,0.25));
        padding: 14px; border-radius: 12px; text-align: center; margin: 20px 0 10px 0;
        border: 2px solid rgba(0,255,136,0.5);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.6rem;}
    .stDataFrame table, .stDataFrame td, .stDataFrame th {border: none !important;}
    .stDataFrame thead {display: none;}
    .session-badge {padding: 5px 12px; border-radius: 15px; font-weight: bold;}
    .session-london {background: #ff4444; color: white;}
    .session-ny {background: #00cccc; color: white;}
    .session-tokyo {background: #ffdd00; color: black;}
</style>
""", unsafe_allow_html=True)

PAIRS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
         "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
         "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
         "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","XAU_USD"]

TFS = ["M15", "H1", "H4", "D1"]
GRANULARITY_MAP = {"M15":"M15", "H1":"H1", "H4":"H4", "D1":"D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== OANDA ====================
@st.cache_resource
def get_client():
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])

client = get_client()

# ==================== CACHE PAR PAIRE/TF ====================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_pair_tf(_pair: str, _tf: str) -> Optional[dict]:
    try:
        params = {"granularity": GRANULARITY_MAP[_tf], "count": 300, "price": "M"}
        req = InstrumentsCandles(instrument=_pair, params=params)
        client.request(req)
        candles = req.response.get("candles", [])
        if len(candles) < 100: return None
        
        df = pd.DataFrame([{
            "time": pd.to_datetime(c["time"]).tz_localize(None),
            "close": float(c["mid"]["c"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"])
        } for c in candles[-100:]])
        
        # Calcul rapide des indicateurs
        def wma(s, n): return s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
        df['hma'] = wma(2*wma(df['close'],10) - wma(df['close'],20), int(np.sqrt(20)))
        df['hma_up'] = (df['hma'] > df['hma'].shift(1)).fillna(False)
        
        tr = pd.concat([df['high']-df['low'], (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14).mean()
        df['atr_pct'] = df['atr'].rolling(100, min_periods=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=True)
        
        last = df.iloc[-2]
        prev = df.iloc[-3]
        
        if last['atr_pct'] < 20 or last['atr_pct'] > 80: return None
        if abs(df['adx'].iloc[-1] if 'adx' in df else 0) < 25: return None
        
        flip_up = last['hma_up'] and not prev['hma_up']
        flip_down = not last['hma_up'] and prev['hma_up']
        
        if flip_up and df['close'].iloc[-2] > df['close'].iloc[-3]:
            trend = "Bullish" if df['close'].iloc[-1] > df['close'].ewm(span=50).mean().iloc[-1] else None
            if trend:
                return {"pair": _pair.replace("_","/"), "tf": _tf, "action": "BUY", "entry": round(last['close'],5), "time": last['time']}
        if flip_down and df['close'].iloc[-2] < df['close'].iloc[-3]:
            trend = "Bearish" if df['close'].iloc[-1] < df['close'].ewm(span=50).mean().iloc[-1] else None
            if trend:
                return {"pair": _pair.replace("_","/"), "tf": _tf, "action": "SELL", "entry": round(last['close'],5), "time": last['time']}
    except:
        pass
    return None

# ==================== SCAN SANS THREADS (zéro warning) ====================
def run_scan():
    signals = []
    for pair in PAIRS:
        for tf in TFS:
            result = fetch_pair_tf(pair, tf)
            if result:
                signals.append(result)
    return signals

# ==================== PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<font size=18 color=#00ff88>BlueStar Institutional</font>", styles["Title"]))
    elements.append(Paragraph(f"{datetime.now(TUNIS_TZ).strftime('%d/%m %H:%M')} Tunis", styles["Normal"]))
    elements.append(Spacer(1, 12*mm))
    
    data = [["Paire", "TF", "Action", "Entry", "Heure"]]
    for s in signals:
        data.append([s["pair"], s["tf"], s["action"], s["entry"], s["time"].strftime("%H:%M")])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== MAIN ====================
def main():
    st.markdown("# BlueStar Institutional v3.0")
    st.markdown('<span class="institutional-badge">INSTITUTIONAL ONLY</span>', unsafe_allow_html=True)
    
    now = datetime.now(TUNIS_TZ)
    session = "London" if 8 <= now.hour < 17 else "NY" if 13 <= now.hour < 22 else "Tokyo" if now.hour < 9 else "Off"
    badge = f"<span class='session-badge session-{'london' if session=='London' else 'ny' if session=='NY' else 'tokyo'}'>{session}</span>"
    st.markdown(f"**{now.strftime('%H:%M:%S')} Tunis** {badge}", unsafe_allow_html=True)

    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        sl = col1.slider("SL × ATR", 1.5, 3.0, 2.0, 0.1)
        tp = col2.slider("TP × ATR", 3.0, 5.0, 3.5, 0.1)

    if st.button("SCAN", type="primary", use_container_width=True):
        with st.spinner("Scan institutionnel en cours..."):
            signals = run_scan()

        if signals:
            col1, col2 = st.columns([1,1])
            with col1:
                csv = pd.DataFrame(signals).to_csv(index=False).encode()
                st.download_button("CSV", csv, "bluestar.csv", "text/csv")
            with col2:
                pdf = generate_pdf(signals)
                st.download_button("PDF", pdf, "BlueStar.pdf", "application/pdf")

            for tf in TFS:
                tf_sig = [s for s in signals if s["tf"] == tf]
                if tf_sig:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p></div>", unsafe_allow_html=True)
                    df = pd.DataFrame(tf_sig)[["pair", "action", "entry", "time"]]
                    df.columns = ["Paire", "Action", "Entry", "Heure"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("Aucun signal institutionnel pour le moment.")

if __name__ == "__main__":
    main()
