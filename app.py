"""
BlueStar Institutional v3.3 (Intraday Edition)
Dynamic Scoring, No Red Bars, M15/H1/H4 Focus
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

# ==================== STYLE & CONFIG ====================
st.set_page_config(page_title="BlueStar Intraday v3.3", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
    /* BADGES HEADER */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
    .intraday-badge {background: linear-gradient(45deg, #ff0055, #ff5599); color: white; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; margin-left: 8px;}
    
    /* TABLEAUX CLEAN */
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    /* TIME FRAMES */
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; 
        margin-top: 15px;
        margin-bottom: 5px;
        border-radius: 0 10px 10px 0;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem; letter-spacing: 1px;}
</style>
""", unsafe_allow_html=True)

# ==================== PARAMÈTRES ====================
# Liste réduite pour focus Intraday rapide
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","USD_CAD",
    "EUR_JPY","GBP_JPY","XAU_USD","US30_USD", "NAS100_USD"
]

# On garde seulement l'Intraday
TIMEFRAMES = ["M15", "H1", "H4"]

GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== DATA STRUCTURES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "DIAMOND"
    PREMIUM = "GOLD"
    STANDARD = "SILVER"

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
    quality: SignalQuality
    confluences: List[str]

# ==================== LOGIQUE ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        return None

client = get_oanda_client()

def get_candles(pair: str, tf: str) -> pd.DataFrame:
    # On demande 500 bougies pour être sûr que l'EMA 200 se calcule bien sur le M15
    count = 500 
    gran = GRANULARITY_MAP.get(tf, "H1")
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        r = InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        data = [{
            'time': c['time'],
            'open': float(c['mid']['o']),
            'high': float(c['mid']['h']),
            'low': float(c['mid']['l']),
            'close': float(c['mid']['c'])
        } for c in r.response['candles'] if c['complete']]
        df = pd.DataFrame(data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

def analyze_intraday(df: pd.DataFrame, pair: str, tf: str, params: TradingParams) -> Optional[Signal]:
    if len(df) < 200: return None
    
    # --- INDICATEURS ---
    # EMA 200 (Trend)
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # HMA (Trigger)
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)

    df['hma'] = hma(df['close'], 20)
    
    # Ichimoku (Cloud uniquement pour filtre)
    h9 = df['high'].rolling(9).max(); l9 = df['low'].rolling(9).min()
    h26 = df['high'].rolling(26).max(); l26 = df['low'].rolling(26).min()
    h52 = df['high'].rolling(52).max(); l52 = df['low'].rolling(52).min()
    df['ssa'] = ((h9+l9)/2 + (h26+l26)/2) / 2
    df['ssb'] = (h52 + l52) / 2
    # Shift du cloud (Attention: dans Pandas on ne shift pas les valeurs futures vers le présent pour l'analyse current, on regarde les valeurs décalées)
    # Pour simplifier en intraday : On regarde si le prix est au dessus du "Kumo actuel" (projeté il y a 26 périodes)
    df['kumo_top'] = df[['ssa', 'ssb']].max(axis=1).shift(26)
    df['kumo_bottom'] = df[['ssa', 'ssb']].min(axis=1).shift(26)
    
    # ATR & ADX & RSI (Pour le scoring dynamique)
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    change = df['close'].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = -change.clip(upper=0).rolling(14).mean()
    rs = gain/loss
    df['rsi'] = 100 - (100/(1+rs))
    
    # ADX Simplifié
    df['adx'] = (abs(df['high'] - df['low']) / df['close']).rolling(14).mean() * 1000 # Juste un proxy de volatilité

    # FVG
    df['fvg_bull'] = (df['low'] > df['high'].shift(2))
    df['fvg_bear'] = (df['high'] < df['low'].shift(2))

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    # --- LOGIQUE ---
    action = None
    
    # 1. Filtre Trend
    bullish = curr['close'] > curr['ema200']
    bearish = curr['close'] < curr['ema200']
    
    # 2. Trigger HMA (Retournement)
    # On cherche un retournement récent (sur les 3 dernières bougies)
    hma_turned_green = (curr['hma'] > prev['hma']) and (prev['hma'] < df.iloc[-3]['hma'])
    hma_turned_red = (curr['hma'] < prev['hma']) and (prev['hma'] > df.iloc[-3]['hma'])
    
    # Si pas de retournement immédiat, on accepte la continuation si c'est "fresh"
    hma_green = curr['hma'] > prev['hma']
    hma_red = curr['hma'] < prev['hma']
    
    score = 0
    confluences = []
    
    if hma_green and bullish:
        action = "BUY"
        if curr['close'] > curr['kumo_top']: 
            score += 20; confluences.append("Cloud")
        if any(df['fvg_bull'].iloc[-5:]): # FVG dans les 5 dernières bougies
            score += 20; confluences.append("FVG")
        if curr['close'] > curr['ema200']:
            score += 20; confluences.append("Trend")
            
    elif hma_red and bearish:
        action = "SELL"
        if curr['close'] < curr['kumo_bottom']: 
            score += 20; confluences.append("Cloud")
        if any(df['fvg_bear'].iloc[-5:]):
            score += 20; confluences.append("FVG")
        if curr['close'] < curr['ema200']:
            score += 20; confluences.append("Trend")
            
    if not action: return None
    
    # --- SCORING DYNAMIQUE (Pour éviter le 85 partout) ---
    base_score = 60 # Départ
    score += base_score
    
    # Bonus RSI (Pas de surachat/survente excessif)
    rsi = curr['rsi']
    if action == "BUY":
        if 40 <= rsi <= 65: score += 5  # Zone saine pour acheter
        if rsi > 70: score -= 5         # Attention surachat
    else:
        if 35 <= rsi <= 60: score += 5  # Zone saine pour vendre
        if rsi < 30: score -= 5         # Attention survente
        
    # Bonus Volatilité
    if curr['adx'] > 1.5: score += 5    # Le marché bouge bien
    
    # Plafond
    score = min(99, score)
    
    # Filtres finaux
    if params.use_fvg and "FVG" not in confluences: return None
    if score < params.min_score: return None

    atr = curr['atr']
    sl = curr['close'] - (atr * params.atr_sl_multiplier) if action == "BUY" else curr['close'] + (atr * params.atr_sl_multiplier)
    tp = curr['close'] + (atr * params.atr_tp_multiplier) if action == "BUY" else curr['close'] - (atr * params.atr_tp_multiplier)
    
    qual = SignalQuality.INSTITUTIONAL if score >= 85 else SignalQuality.PREMIUM

    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=int(score), quality=qual, confluences=confluences
    )

def scan_thread(pairs, tfs, params):
    signals = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for p in pairs:
            for tf in tfs:
                futures.append(executor.submit(lambda p, t: (get_candles(p,t), p, t), p, tf))
        for f in as_completed(futures):
            try:
                df, p, tf = f.result()
                if not df.empty:
                    sig = analyze_intraday(df, p, tf, params)
                    if sig: signals.append(sig)
            except: pass
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== PDF ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<font size=20 color='#00ff88'><b>BlueStar Intraday Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 10*mm))
    
    data = [["HEURE", "PAIRE", "TF", "ACTION", "PRIX", "SCORE", "CONFIRMATIONS"]]
    for s in signals:
        col = "#00ff88" if s.action == "BUY" else "#ff6b6b"
        conf_str = ", ".join(s.confluences)
        data.append([
            s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"), s.timeframe,
            Paragraph(f"<font color='{col}'><b>{s.action}</b></font>", styles["Normal"]),
            f"{s.entry_price:.5f}", f"{s.score}/100", conf_str
        ])
    
    t = Table(data, colWidths=[25*mm, 30*mm, 20*mm, 25*mm, 30*mm, 20*mm, 100*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#444")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== UI ====================
def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <h1 style='margin-bottom:0;'>BlueStar Intraday</h1>
            <div>
                <span class='institutional-badge'>INSTITUTIONAL</span>
                <span class='intraday-badge'>INTRADAY v3.3</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("🧹 Reset"):
            st.session_state.scan_results = None
            st.rerun()

    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None

    with st.expander("⚙️ Paramètres de Scan", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        sl = c1.number_input("SL xATR", 1.0, 3.0, 1.5)
        tp = c2.number_input("TP xATR", 1.0, 6.0, 3.0)
        sc = c3.slider("Min Score", 50, 95, 70)
        fvg = c4.checkbox("FVG Required", False)

    if st.button("🔎 SCANNER INTRADAY (M15 / H1 / H4)", type="primary", use_container_width=True):
        if not client:
            st.error("API Token Manquant")
        else:
            with st.spinner("Analyse Intraday... (Calcul EMA200 & SMC)"):
                params = TradingParams(sl, tp, sc, fvg)
                st.session_state.scan_results = scan_thread(PAIRS_DEFAULT, TIMEFRAMES, params)

    if st.session_state.scan_results is not None:
        signals = st.session_state.scan_results
        
        # Bouton PDF isolé
        st.markdown("###")
        if signals:
            st.download_button("📄 Télécharger PDF", generate_pdf(signals), f"Scan_{datetime.now().strftime('%H%M')}.pdf", "application/pdf")
        
        # Affichage
        if not signals:
            st.warning("Aucun signal trouvé. Le marché est peut-être calme ou contre-tendance.")
        else:
            for tf in TIMEFRAMES:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3></div>", unsafe_allow_html=True)
                    
                    data_display = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        data_display.append({
                            "Heure": s.timestamp.strftime("%H:%M"),
                            "Paire": s.pair.replace("_", "/"),
                            "Signal": f"{icon} {s.action}",
                            "Prix": f"{s.entry_price:.5f}",
                            "SL": f"{s.stop_loss:.5f}",
                            "TP": f"{s.take_profit:.5f}",
                            "Score": s.score, # Juste le chiffre brut
                            "Confirmations": " + ".join(s.confluences)
                        })
                    
                    # Plus de barre rouge bizarre, juste un tableau propre
                    st.dataframe(
                        pd.DataFrame(data_display),
                        use_container_width=True,
                        hide_index=True
                    )

if __name__ == "__main__":
    main()
