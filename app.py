"""
BlueStar Institutional v3.2 (Visual Artifact + Smart Logic + State Fix)
Professional Grade Algorithm with Smart Money Concepts & Persistent State
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

# ==================== CONFIGURATION & VISUAL STYLE (ARTEFACT RESTORED) ====================
st.set_page_config(page_title="BlueStar Institutional v3.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    
    /* STYLE METRIQUE & BADGES (Le look "Artefact") */
    .stMetric {background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1);}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.75rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700; text-shadow: 0 0 10px rgba(0,255,136,0.3);}
    
    /* BADGES HEADER */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; display: inline-block; box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
    .v3-badge {background: linear-gradient(45deg, #00ff88, #00ccff); color: white; padding: 4px 12px; border-radius: 15px; font-weight: 800; font-size: 0.7rem; display: inline-block; margin-left: 8px; box-shadow: 0 0 10px rgba(0, 255, 136, 0.4);}
    
    /* ZONE TIMEFRAME */
    .tf-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,0,0,0) 100%); 
        border-left: 4px solid #00ff88;
        padding: 8px 15px; 
        margin-top: 20px;
        margin-bottom: 10px;
        border-radius: 0 10px 10px 0;
    }
    .tf-header h3 {margin: 0; color: #fff; font-size: 1.1rem; letter-spacing: 1px;}
    
    /* TABLEAUX */
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    /* BADGES CONFLUENCE */
    .conf-tag {background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #ccc; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; margin-right: 4px;}
    .smc-tag {background: rgba(157, 0, 255, 0.2); border: 1px solid #9d00ff; color: #d08fff; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_JPY","GBP_JPY","AUD_JPY","XAU_USD","US30_USD", "NAS100_USD"
]
GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== LOGIQUE MÉTIER (SMART MONEY v3.1) ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "DIAMOND"
    PREMIUM = "GOLD"
    STANDARD = "SILVER"

@dataclass
class TradingParams:
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_rr: float
    min_score: int
    use_ema_filter: bool
    detect_fvg: bool

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
    rr_ratio: float

# --- OANDA API ---
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        return None

client = get_oanda_client()

def get_candles(pair: str, tf: str, count: int = 250) -> pd.DataFrame:
    if not client: return pd.DataFrame()
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

# --- INDICATEURS & ANALYSE ---
def analyze_market_structure(df: pd.DataFrame, pair: str, tf: str, params: TradingParams) -> Optional[Signal]:
    if len(df) < 200: return None
    
    # Indicateurs Vectorisés
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # HMA
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(int(np.sqrt(length))).apply(lambda x: np.dot(x, np.arange(1, int(np.sqrt(length))+1)) / np.arange(1, int(np.sqrt(length))+1).sum(), raw=True)

    df['hma'] = hma(df['close'], 20)
    
    # Ichimoku Cloud Check
    high9, low9 = df['high'].rolling(9).max(), df['low'].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26, low26 = df['high'].rolling(26).max(), df['low'].rolling(26).min()
    kijun = (high26 + low26) / 2
    df['ssa'] = ((tenkan + kijun) / 2).shift(26)
    df['ssb'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # ATR & FVG
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
    df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Filtres
    trend_bull = curr['close'] > curr['ema200']
    trend_bear = curr['close'] < curr['ema200']
    if params.use_ema_filter and not (trend_bull or trend_bear): return None

    # Trigger HMA (Changement de pente)
    hma_up = curr['hma'] > prev['hma'] and prev['hma'] < df.iloc[-3]['hma']
    hma_down = curr['hma'] < prev['hma'] and prev['hma'] > df.iloc[-3]['hma']
    
    # Simple validation de direction courante si pas de flip exact
    is_green = curr['hma'] > prev['hma']
    is_red = curr['hma'] < prev['hma']

    action = None
    score = 0
    confluences = []

    if is_green and trend_bull:
        action = "BUY"
        if curr['close'] > max(curr['ssa'], curr['ssb']):
            score += 20; confluences.append("Cloud")
        if any(df['fvg_bull'].iloc[-3:]):
            score += 30; confluences.append("FVG")
        if curr['close'] > curr['ema200']:
            score += 25; confluences.append("Trend")
            
    elif is_red and trend_bear:
        action = "SELL"
        if curr['close'] < min(curr['ssa'], curr['ssb']):
            score += 20; confluences.append("Cloud")
        if any(df['fvg_bear'].iloc[-3:]):
            score += 30; confluences.append("FVG")
        if curr['close'] < curr['ema200']:
            score += 25; confluences.append("Trend")

    if not action: return None
    
    score += 10 # Base score
    if params.detect_fvg and "FVG" not in confluences: return None
    if score < params.min_score: return None

    # Risk Mgmt
    atr = curr['atr']
    sl = curr['close'] - (atr * params.atr_sl_multiplier) if action == "BUY" else curr['close'] + (atr * params.atr_sl_multiplier)
    tp = curr['close'] + (atr * params.atr_tp_multiplier) if action == "BUY" else curr['close'] - (atr * params.atr_tp_multiplier)
    rr = abs(tp - curr['close']) / abs(curr['close'] - sl)

    qual = SignalQuality.INSTITUTIONAL if score >= 80 else SignalQuality.PREMIUM

    return Signal(
        timestamp=pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ),
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=score, quality=qual, confluences=confluences, rr_ratio=rr
    )

def scan_market_thread(pairs, tfs, params):
    signals = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for p in pairs:
            for tf in tfs:
                futures.append(executor.submit(lambda p, t: (get_candles(p,t), p, t), p, tf))
        
        for f in as_completed(futures):
            try:
                df, p, tf = f.result()
                if not df.empty:
                    sig = analyze_market_structure(df, p, tf, params)
                    if sig: signals.append(sig)
            except: pass
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== PDF GENERATOR (LOOK ARTEFACT) ====================
def generate_pdf(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=10*mm, bottomMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    # Titre stylisé
    elements.append(Paragraph("<font size=24 color='#00ff88'><b>BlueStar Institutional Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph(f"<font size=10 color='#a0a0c0'>Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    # Données
    data = [["TIME", "PAIR", "TF", "ACTION", "ENTRY", "SL", "TP", "SCORE", "CONFLUENCES"]]
    
    for s in signals:
        act_color = "#00ff88" if s.action == "BUY" else "#ff6b6b"
        action_txt = f"<font color='{act_color}'><b>{s.action}</b></font>"
        
        # Confluences jolies
        conf_txt = ""
        for c in s.confluences:
            bg = "#9d00ff" if c == "FVG" else "#333333"
            conf_txt += f"[{c}] "

        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            Paragraph(action_txt, styles["Normal"]),
            f"{s.entry_price:.4f}",
            f"{s.stop_loss:.4f}",
            f"{s.take_profit:.4f}",
            f"{s.score}",
            Paragraph(f"<font size=8>{conf_txt}</font>", styles["Normal"])
        ])
        
    # Style Tableau "Cyber"
    table = Table(data, colWidths=[20*mm, 25*mm, 15*mm, 20*mm, 25*mm, 25*mm, 25*mm, 15*mm, 80*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f1429")), # Header Dark Blue
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")), # Header Neon Green
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        
        # Lignes alternées sombres
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333333")),
        
        # Bordure externe
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#00ff88")),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== MAIN UI ====================
def main():
    # En-tête "Artefact"
    col_logo, col_refresh = st.columns([3, 1])
    with col_logo:
        st.markdown("""
            <h1 style='margin-bottom:0; padding-bottom:0'>BlueStar Institutional</h1>
            <div>
                <span class='institutional-badge'>INSTITUTIONAL</span>
                <span class='v3-badge'>v3.2</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col_refresh:
        st.write("") # Spacer
        if st.button("🧹 Clear Cache"):
            st.session_state.scan_results = None
            st.rerun()

    # --- ÉTAT DE SESSION (LE FIX POUR LE PDF) ---
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None

    # --- CONFIG ---
    with st.expander("⚙️ PARAMÈTRES ALGORITHME", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        sl_mult = c1.number_input("SL (xATR)", 1.0, 3.0, 1.5)
        tp_mult = c2.number_input("TP (xATR)", 1.0, 6.0, 3.0)
        min_score = c3.slider("Min Score", 50, 90, 65)
        use_fvg = c4.checkbox("FVG Only", False)

    # --- BOUTON DE SCAN ---
    if st.button("🚀 SCANNER LE MARCHÉ", type="primary", use_container_width=True):
        if not client:
            st.error("API Token Manquant")
        else:
            with st.spinner("Analyse Institutionnelle en cours..."):
                params = TradingParams(sl_mult, tp_mult, 1.5, min_score, True, use_fvg)
                results = scan_market_thread(PAIRS_DEFAULT, ["M15", "H1", "H4", "D1"], params)
                # SAUVEGARDE DANS LE STATE
                st.session_state.scan_results = results
                # Pas de rerun forcé, Streamlit va relire le script et trouver le state rempli en bas

    # --- AFFICHAGE DES RÉSULTATS (Depuis le State) ---
    if st.session_state.scan_results is not None:
        signals = st.session_state.scan_results
        
        st.markdown("---")
        
        # Métriques Globales
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Opportunités", len(signals))
        best = signals[0] if signals else None
        if best:
            k2.metric("Meilleur Signal", best.pair.replace("_","/"))
            k3.metric("Score Max", f"{best.score}/100")
            k4.metric("Qualité", best.quality.value)

        # --- BOUTON PDF (HORS DU BLOC SCAN POUR NE PAS RELANCER) ---
        st.markdown("###")
        c_pdf, c_csv = st.columns(2)
        with c_pdf:
            if signals:
                pdf_file = generate_pdf(signals)
                st.download_button(
                    label="📄 TÉLÉCHARGER RAPPORT PDF",
                    data=pdf_file,
                    file_name=f"BlueStar_Report_{datetime.now().strftime('%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # --- VISUEL TABLEAU PAR TF ---
        if not signals:
            st.info("Aucun signal détecté.")
        else:
            for tf in ["M15", "H1", "H4", "D1"]:
                tf_sigs = [s for s in signals if s.timeframe == tf]
                if tf_sigs:
                    st.markdown(f"<div class='tf-header'><h3>{tf} TIMEFRAME</h3></div>", unsafe_allow_html=True)
                    
                    # Préparation données pour affichage propre
                    disp_data = []
                    for s in tf_sigs:
                        icon = "🟢" if s.action == "BUY" else "🔴"
                        conf_str = " ".join([f"[{c}]" for c in s.confluences])
                        disp_data.append({
                            "Heure": s.timestamp.strftime("%H:%M"),
                            "Paire": s.pair.replace("_", "/"),
                            "Signal": f"{icon} {s.action}",
                            "Prix": f"{s.entry_price:.5f}",
                            "SL": f"{s.stop_loss:.5f}",
                            "TP": f"{s.take_profit:.5f}",
                            "Score": s.score,
                            "Confluences": conf_str
                        })
                    
                    st.dataframe(
                        pd.DataFrame(disp_data), 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn("Force", min_value=0, max_value=100, format="%d"),
                        }
                    )

if __name__ == "__main__":
    main()
