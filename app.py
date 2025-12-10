"""
BlueStar Institutional v3.1 (Enhanced Logic)
Professional Grade Algorithm with Smart Money Concepts
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
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

# ==================== CONFIGURATION & STYLE ====================
st.set_page_config(page_title="BlueStar Institutional v3.1", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 1rem !important;}
    
    /* STYLE CARTE SIGNAL */
    .stMetric {background: rgba(16, 20, 45, 0.8); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.8rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700; text-shadow: 0 0 10px rgba(0,255,136,0.3);}
    
    /* BADGES */
    .institutional-badge {background: linear-gradient(90deg, #ffd700, #ffaa00); color: black; padding: 4px 12px; border-radius: 4px; font-weight: 800; font-size: 0.7rem; box-shadow: 0 2px 5px rgba(255, 215, 0, 0.3); letter-spacing: 1px;}
    .smart-money-badge {background: linear-gradient(90deg, #9d00ff, #bd00ff); color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.6rem; font-weight: bold; margin-left: 5px;}
    
    /* TABLES */
    [data-testid="stDataFrame"] {border: none !important;}
    [data-testid="stHeader"] {background-color: rgba(0,0,0,0) !important;}
    
    /* ZONES DE TIME FRAME */
    .tf-header {
        background: rgba(255, 255, 255, 0.03); 
        border-left: 4px solid #00ff88;
        padding: 10px 15px; 
        border-radius: 0 8px 8px 0; 
        margin-bottom: 15px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .tf-title {font-size: 1.1rem; font-weight: bold; color: #fff;}
    .tf-count {background: rgba(0,255,136,0.2); color: #00ff88; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem;}

    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: #0a0e27;}
    ::-webkit-scrollbar-thumb {background: #333; border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: #555;}
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES & SETTINGS ====================
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_JPY","GBP_JPY","AUD_JPY","XAU_USD","US30_USD", "NAS100_USD"
]

GRANULARITY_MAP = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D"}
TUNIS_TZ = pytz.timezone('Africa/Tunis')

# ==================== CLASSES DE DONNÉES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "💎 INST"  # Trade parfait
    PREMIUM = "✨ PREM"       # Très bon trade
    STANDARD = "⚡ STD"       # Trade standard

@dataclass
class TradingParams:
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_rr: float
    min_score: int
    use_ema_filter: bool  # Force trend direction
    detect_fvg: bool      # Cherche les Fair Value Gaps

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
    confluences: List[str]  # Liste des confirmations (ex: "EMA200", "FVG", "Cloud")
    rr_ratio: float
    adx: float

# ==================== API & OANDA ====================
@st.cache_resource
def get_oanda_client():
    try:
        # Assurez-vous d'avoir .streamlit/secrets.toml avec OANDA_ACCESS_TOKEN
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
        data = []
        for c in r.response['candles']:
            if c['complete']:
                data.append({
                    'time': c['time'],
                    'open': float(c['mid']['o']),
                    'high': float(c['mid']['h']),
                    'low': float(c['mid']['l']),
                    'close': float(c['mid']['c'])
                })
        df = pd.DataFrame(data)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None) # UTC Naive
        return df
    except Exception as e:
        return pd.DataFrame()

# ==================== COEUR ALGORITHMIQUE (NOUVEAU) ====================
def calculate_indicators_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Calcul optimisé vectoriel pour performance maximale"""
    if df.empty: return df
    
    # 1. EMA 200 (Trend Filter)
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. HMA (Hull Moving Average) - Trigger rapide
    def hma(series, length=20):
        wma_half = series.rolling(length//2).apply(lambda x: np.dot(x, np.arange(1, length//2+1)) / np.arange(1, length//2+1).sum(), raw=True)
        wma_full = series.rolling(length).apply(lambda x: np.dot(x, np.arange(1, length+1)) / np.arange(1, length+1).sum(), raw=True)
        diff = 2 * wma_half - wma_full
        sqrt_len = int(np.sqrt(length))
        return diff.rolling(sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=True)

    df['hma'] = hma(df['close'], 20)
    
    # 3. Ichimoku Cloud (Filter) - Juste le Kumo pour la direction
    high9 = df['high'].rolling(9).max()
    low9 = df['low'].rolling(9).min()
    tenkan = (high9 + low9) / 2
    
    high26 = df['high'].rolling(26).max()
    low26 = df['low'].rolling(26).min()
    kijun = (high26 + low26) / 2
    
    df['ssa'] = ((tenkan + kijun) / 2).shift(26)
    df['ssb'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
    
    # 4. ATR & ADX (Volatilité & Force)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # ADX simplifié
    up = df['high'].diff()
    down = -df['low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Utilisation d'EWM pour lisser comme TradingView
    df['adx'] = pd.Series(plus_dm).ewm(alpha=1/14).mean() # Approx rapide pour la démo
    
    # 5. Stochastic RSI (Timing précis)
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    min_rsi = df['rsi'].rolling(14).min()
    max_rsi = df['rsi'].rolling(14).max()
    df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # 6. Fair Value Gap (Smart Money)
    # Bullish FVG: Low of candle 0 > High of candle 2
    # Bearish FVG: High of candle 0 < Low of candle 2
    df['fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1))
    df['fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1))

    return df

def analyze_market_structure(df: pd.DataFrame, pair: str, tf: str, params: TradingParams) -> Optional[Signal]:
    if len(df) < 200: return None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # --- 1. FILTRE TENDANCE MAJEURE (EMA 200) ---
    trend_bullish = curr['close'] > curr['ema200']
    trend_bearish = curr['close'] < curr['ema200']
    
    if params.use_ema_filter:
        if not (trend_bullish or trend_bearish): return None # Marché plat sur l'EMA

    # --- 2. TRIGGER (HMA + STOCH RSI) ---
    # HMA Change de couleur ET StochRSI confirme
    hma_buy = curr['hma'] > curr['hma'] - (prev['hma'] - df.iloc[-3]['hma']) # HMA tourne vers le haut
    hma_sell = curr['hma'] < curr['hma'] - (prev['hma'] - df.iloc[-3]['hma'])
    
    # Validation HMA pure (pente)
    is_hma_green = curr['hma'] > prev['hma']
    is_hma_red = curr['hma'] < prev['hma']
    
    # --- 3. CONFLUENCES CHECKLIST ---
    score = 0
    confluences = []
    
    # Action Determination
    action = None
    
    # Logique d'achat
    if is_hma_green and trend_bullish:
        # Check Cloud
        above_cloud = curr['close'] > max(curr['ssa'], curr['ssb'])
        if above_cloud: 
            score += 20
            confluences.append("☁️ Cloud")
        
        # Check FVG (Récent sur les 3 dernières bougies)
        has_fvg = any(df['fvg_bull'].iloc[-3:])
        if has_fvg: 
            score += 25
            confluences.append("💰 FVG")
            
        # Check Momentum (StochRSI sort de la zone de survente)
        stoch_ok = prev['stoch_k'] < 80 # Pas en surachat extrême au départ
        if stoch_ok: score += 10
        
        # Check Trend Strength
        if curr['close'] > curr['ema200']: 
            score += 25
            confluences.append("📈 Trend")
            
        action = "BUY"

    # Logique de vente
    elif is_hma_red and trend_bearish:
        # Check Cloud
        below_cloud = curr['close'] < min(curr['ssa'], curr['ssb'])
        if below_cloud: 
            score += 20
            confluences.append("☁️ Cloud")
            
        # Check FVG
        has_fvg = any(df['fvg_bear'].iloc[-3:])
        if has_fvg: 
            score += 25
            confluences.append("💰 FVG")
            
        # Check Momentum
        stoch_ok = prev['stoch_k'] > 20
        if stoch_ok: score += 10
        
        # Check Trend Strength
        if curr['close'] < curr['ema200']: 
            score += 25
            confluences.append("📉 Trend")
            
        action = "SELL"
        
    else:
        return None

    # --- SCORE FINAL & FILTRAGE ---
    if params.detect_fvg and "💰 FVG" not in confluences and score < 80:
        return None # Si on veut être strict sur la Smart Money
        
    if score < params.min_score: return None
    
    # Calcul SL/TP
    atr = curr['atr']
    if action == "BUY":
        sl = curr['close'] - (atr * params.atr_sl_multiplier)
        tp = curr['close'] + (atr * params.atr_tp_multiplier)
    else:
        sl = curr['close'] + (atr * params.atr_sl_multiplier)
        tp = curr['close'] - (atr * params.atr_tp_multiplier)
        
    rr = abs(tp - curr['close']) / abs(curr['close'] - sl)
    if rr < params.min_rr: return None
    
    quality = SignalQuality.STANDARD
    if score >= 85: quality = SignalQuality.INSTITUTIONAL
    elif score >= 65: quality = SignalQuality.PREMIUM
    
    # UTC -> Tunis time for display
    local_time = pytz.utc.localize(curr['time']).astimezone(TUNIS_TZ)

    return Signal(
        timestamp=local_time,
        pair=pair, timeframe=tf, action=action,
        entry_price=curr['close'], stop_loss=sl, take_profit=tp,
        score=score, quality=quality, confluences=confluences,
        rr_ratio=rr, adx=curr['adx'] if not pd.isna(curr['adx']) else 0
    )

# ==================== ORCHESTRATION DU SCAN ====================
def scan_market(pairs, tfs, params):
    signals = []
    
    def worker(p, tf):
        df = get_candles(p, tf)
        if df.empty: return None
        df = calculate_indicators_v3(df)
        return analyze_market_structure(df, p, tf, params)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker, p, tf): (p, tf) for p in pairs for tf in tfs}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res: signals.append(res)
            except: pass
            
    return sorted(signals, key=lambda x: x.score, reverse=True)

# ==================== GÉNÉRATION PDF ====================
def create_pdf_report(signals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    elements = []
    styles = getSampleStyleSheet()
    
    title = Paragraph("<b>Rapport BlueStar Institutional v3.1</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 10*mm))
    
    data = [['Heure', 'Paire', 'TF', 'Action', 'Prix', 'Score', 'Confluences']]
    for s in signals:
        conf_str = ", ".join(s.confluences)
        color = colors.green if s.action == "BUY" else colors.red
        data.append([
            s.timestamp.strftime('%H:%M'),
            s.pair, s.timeframe, s.action,
            f"{s.entry_price:.5f}", str(s.score), conf_str
        ])
        
    table = Table(data, colWidths=[20*mm, 30*mm, 15*mm, 20*mm, 30*mm, 15*mm, 120*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== INTERFACE UTILISATEUR ====================
def main():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🔵 BlueStar Institutional v3.1")
        st.markdown("*Algorithme de Confluence Multi-Timeframe & Smart Money*")
    with col2:
        if st.button("🔄 Rafraîchir Données"):
            st.cache_data.clear()
            st.rerun()

    # Sidebar Config
    with st.expander("🛠️ Configuration Avancée & Stratégie", expanded=False):
        c1, c2, c3 = st.columns(3)
        sl_mult = c1.slider("Stop Loss (x ATR)", 1.0, 3.0, 1.5, 0.1)
        tp_mult = c2.slider("Take Profit (x ATR)", 1.0, 5.0, 3.0, 0.1)
        min_score = c3.slider("Score Minimum (Qualité)", 50, 90, 65, 5)
        
        c4, c5 = st.columns(2)
        use_ema = c4.checkbox("Filtre Tendance (EMA 200) - Recommandé", True)
        detect_fvg = c5.checkbox("Focus Smart Money (FVG Obligatoire)", False)

    # Bouton d'action
    run_scan = st.button("🚀 LANCER LE SCAN DE MARCHÉ", type="primary", use_container_width=True)

    if run_scan:
        if not client:
            st.error("⚠️ Clé API OANDA manquante dans .streamlit/secrets.toml")
            st.stop()
            
        with st.spinner("Analyse des structures institutionnelles en cours..."):
            params = TradingParams(sl_mult, tp_mult, 1.5, min_score, use_ema, detect_fvg)
            signals = scan_market(PAIRS_DEFAULT, ["M15", "H1", "H4", "D1"], params)
            
            if not signals:
                st.warning("Aucune opportunité haute probabilité détectée pour le moment. Le marché est peut-être indécis.")
            else:
                # Top Metrics
                best_signal = signals[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Opportunités", len(signals))
                m2.metric("Meilleure Paire", best_signal.pair.replace("_", "/"))
                m3.metric("Direction", best_signal.action, delta=str(best_signal.score), delta_color="normal")
                m4.metric("Qualité", best_signal.quality.value)
                
                st.markdown("---")
                
                # Affichage par Timeframe
                for tf in ["M15", "H1", "H4", "D1"]:
                    tf_sigs = [s for s in signals if s.timeframe == tf]
                    if not tf_sigs: continue
                    
                    st.markdown(f"<div class='tf-header'><span class='tf-title'>{tf} Timeframe</span><span class='tf-count'>{len(tf_sigs)} Signaux</span></div>", unsafe_allow_html=True)
                    
                    for s in tf_sigs:
                        # Carte Signal
                        with st.container():
                            cc1, cc2, cc3, cc4, cc5, cc6 = st.columns([1, 1, 1, 2, 1, 1])
                            
                            cc1.markdown(f"**{s.pair.replace('_', '/')}**")
                            
                            color = "#00ff88" if s.action == "BUY" else "#ff4b4b"
                            cc2.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.1rem'>{s.action}</span>", unsafe_allow_html=True)
                            
                            cc3.markdown(f"Entry: `{s.entry_price:.5f}`")
                            
                            # Confluences visuelles
                            conf_html = ""
                            for conf in s.confluences:
                                conf_html += f"<span class='institutional-badge' style='margin-right:4px;'>{conf}</span>"
                            if "💰 FVG" in s.confluences:
                                conf_html += "<span class='smart-money-badge'>SMC</span>"
                            cc4.markdown(conf_html, unsafe_allow_html=True)
                            
                            cc5.markdown(f"SL: `{s.stop_loss:.5f}`<br>TP: `{s.take_profit:.5f}`", unsafe_allow_html=True)
                            
                            score_color = "#00ff88" if s.score > 80 else "#ffcc00"
                            cc6.markdown(f"<span style='font-size:1.5rem; font-weight:bold; color:{score_color}'>{s.score}%</span>", unsafe_allow_html=True)
                            
                            st.markdown("<hr style='margin: 5px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

                # Export PDF
                st.markdown("### 📤 Export")
                pdf_data = create_pdf_report(signals)
                st.download_button("Télécharger Rapport PDF", pdf_data, "BlueStar_Report.pdf", "application/pdf", use_container_width=True)

if __name__ == "__main__":
    main()
