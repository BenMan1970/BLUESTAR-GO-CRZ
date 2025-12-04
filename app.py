"""
BlueStar Cascade - VERSION 2.4 FINALE
Corrections & Optimisations :
- Bug critique HMA corrigé
- Cache optimisé pour Live/Confirmed
- Rate limiting OANDA stabilisé
- Interface Esthétique complète
"""
import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
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

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# CSS Esthétique
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 100% !important;}
    
    /* Metrics */
    .stMetric {background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.8rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700;}
    
    /* Badges */
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; box-shadow: 0 2px 5px rgba(0,0,0,0.3); display: inline-block;}
    .signal-card {background: rgba(255,255,255,0.03); border-radius: 10px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #333;}
    .buy-border {border-left-color: #00ff88 !important;}
    .sell-border {border-left-color: #ff4b4b !important;}
    
    /* Tables */
    .stDataFrame {font-size: 0.8rem !important;}
    
    /* Titles */
    h1 {font-size: 2rem !important; margin-bottom: 0.5rem !important; background: -webkit-linear-gradient(#eee, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    h2 {color: #00ff88 !important; font-size: 1.3rem !important; margin-top: 1rem !important;}
    
    /* Buttons */
    .stButton button {width: 100%; border-radius: 6px; font-weight: bold; transition: all 0.3s;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {background-color: #0f1429; border-right: 1px solid rgba(255,255,255,0.05);}
</style>
""", unsafe_allow_html=True)

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK",
    "XAU_USD", "XPT_USD"
]
GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D"}

# Rate limiting pour OANDA
LAST_REQUEST_TIME = {"time": 0}
MIN_REQUEST_INTERVAL = 0.15  # Légèrement augmenté pour sécurité

# ==================== DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "Institutional"
    PREMIUM = "Premium"
    STANDARD = "Standard"

@dataclass
class TradingParams:
    atr_sl_multiplier: float = 2.0
    atr_tp_multiplier: float = 3.0
    min_adx_threshold: int = 20
    adx_strong_threshold: int = 25
    min_rr_ratio: float = 1.2
    cascade_required: bool = True
    strict_flip_only: bool = True

@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05
    kelly_fraction: float = 0.25

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
    position_size: float
    risk_amount: float
    risk_reward: float
    adx: float
    rsi: float
    atr: float
    higher_tf_trend: str
    is_live: bool
    is_fresh_flip: bool
    candle_index: int
    is_strict_flip: bool

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try:
        # Vérification silencieuse ou affichage d'erreur propre
        if "OANDA_ACCESS_TOKEN" not in st.secrets:
            st.error("⚠️ Token OANDA manquant dans `.streamlit/secrets.toml`")
            st.stop()
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except Exception as e:
        st.error(f"⚠️ Erreur de connexion OANDA: {e}")
        st.stop()

client = get_oanda_client()

# TTL ajusté à 15s : Bon compromis Live/History
@st.cache_data(ttl=15)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    """Récupération des bougies avec rate limiting"""
    gran = GRANULARITY_MAP.get(tf)
    if not gran:
        return pd.DataFrame()
    
    # Rate limiting basique
    elapsed = time.time() - LAST_REQUEST_TIME["time"]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        LAST_REQUEST_TIME["time"] = time.time()
        
        data = []
        for c in req.response.get("candles", []):
            data.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "complete": c.get("complete", False)
            })
        df = pd.DataFrame(data)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            df["time"] = df["time"].dt.tz_localize(None)
        return df
    except Exception as e:
        logger.error(f"❌ Erreur API {pair} {tf}: {e}")
        return pd.DataFrame()

# ==================== INDICATEURS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    
    close = df['close']
    high = df['high']
    low = df['low']

    # Helper WMA
    def wma(series, length):
        if len(series) < length: return pd.Series([np.nan] * len(series), index=series.index)
        weights = np.arange(1, length + 1)
        return series.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan, raw=True
        )

    # HMA
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    
    if wma_half.isna().all() or wma_full.isna().all():
        df['hma'] = np.nan
        df['hma_up'] = np.nan
    else:
        df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
        df['hma_up'] = df['hma'] > df['hma'].shift(1)

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7, min_periods=7).mean() / down.ewm(alpha=1/7, min_periods=7).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    # UT Bot
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    xATR = tr.rolling(1).mean()
    nLoss = 2.0 * xATR
    xATRTrailingStop = [0.0] * len(df)
    
    # Boucle UT Bot optimisée (un peu lente en python pur mais OK pour <500 bougies)
    # Pour la lisibilité, on garde la boucle explicite ici
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        curr_src = close.iloc[i]
        loss = nLoss.iloc[i]
        
        if (curr_src > prev_stop) and (close.iloc[i-1] > prev_stop):
            xATRTrailingStop[i] = max(prev_stop, curr_src - loss)
        elif (curr_src < prev_stop) and (close.iloc[i-1] < prev_stop):
            xATRTrailingStop[i] = min(prev_stop, curr_src + loss)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - loss
        else:
            xATRTrailingStop[i] = curr_src + loss
    
    df['ut_state'] = np.where(close > xATRTrailingStop, 1, -1)

    # ADX
    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    df['atr_val'] = atr14
    
    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=300) # 5 min pour la cascade (D1/H4 bougent lentement)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf: return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50: return "Neutral"
    
    df = calculate_indicators(df)
    
    if pd.isna(df['hma'].iloc[-1]) or pd.isna(df['hma'].iloc[-2]): return "Neutral"
    
    close = df['close']
    ema50 = close.ewm(span=50, min_periods=50).mean().iloc[-1]
    
    hma_curr = df['hma'].iloc[-1]
    hma_prev = df['hma'].iloc[-2]
    
    if close.iloc[-1] > ema50 and hma_curr > hma_prev: return "Bullish"
    elif close.iloc[-1] < ema50 and hma_curr < hma_prev: return "Bearish"
    return "Neutral"

# ==================== ANALYSE & SCAN ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager, params) -> Optional[Signal]:
    df = get_candles(pair, tf, 300)
    if len(df) < 100: return None
    
    df = calculate_indicators(df)
    
    # Gestion Live vs Confirmed
    if mode_live:
        idx = -1
    else:
        idx = -2 if not df.iloc[-1]['complete'] else -1
    
    if abs(idx) >= len(df): return None
    
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    # Validation technique
    if pd.isna(last.hma_up) or pd.isna(prev.hma_up) or pd.isna(prev2.hma_up): return None
    if pd.isna(last.rsi) or pd.isna(last.adx): return None
    
    # Détection des Flips
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    
    # LOGIQUE HMA CORRIGÉE (Critique point 8)
    hma_extended_green = last.hma_up and prev.hma_up and not prev2.hma_up and not hma_flip_green
    hma_extended_red = not last.hma_up and not prev.hma_up and prev2.hma_up and not hma_flip_red
    
    # Setup Signal
    if params.strict_flip_only:
        raw_buy = hma_flip_green and last.rsi > 50 and last.ut_state == 1
        raw_sell = hma_flip_red and last.rsi < 50 and last.ut_state == -1
        is_strict = True
    else:
        raw_buy = (hma_flip_green or hma_extended_green) and last.rsi > 50 and last.ut_state == 1
        raw_sell = (hma_flip_red or hma_extended_red) and last.rsi < 50 and last.ut_state == -1
        is_strict = hma_flip_green or hma_flip_red
        
    if not (raw_buy or raw_sell): return None
    
    action = "BUY" if raw_buy else "SELL"
    
    # Cascade
    higher_trend = get_trend_alignment(pair, tf)
    if params.cascade_required:
        if action == "BUY" and higher_trend != "Bullish": return None
        if action == "SELL" and higher_trend != "Bearish": return None
    
    # Scoring
    score = 70
    if last.adx > params.adx_strong_threshold: score += 15
    elif last.adx > params.min_adx_threshold: score += 10
    else: score -= 5
    
    if is_strict: score += 15
    else: score += 5
    
    if (action == "BUY" and 50 < last.rsi < 65) or (action == "SELL" and 35 < last.rsi < 50): score += 5
    if (action == "BUY" and higher_trend == "Bullish") or (action == "SELL" and higher_trend == "Bearish"): score += 10
    
    score = max(0, min(100, score)) # Cap corrigé
    
    quality = SignalQuality.INSTITUTIONAL if score >= 90 else SignalQuality.PREMIUM if score >= 80 else SignalQuality.STANDARD
    
    # Risk Management
    atr = last.atr_val
    sl = last.close - params.atr_sl_multiplier * atr if action == "BUY" else last.close + params.atr_sl_multiplier * atr
    tp = last.close + params.atr_tp_multiplier * atr if action == "BUY" else last.close - params.atr_tp_multiplier * atr
    
    dist_sl = abs(last.close - sl)
    rr = abs(tp - last.close) / dist_sl if dist_sl > 0 else 0
    
    if rr < params.min_rr_ratio: return None
    
    # Timezone
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = pytz.utc.localize(last.time).astimezone(tunis_tz)
    
    # Risk Calc (Simplifié pour l'objet Signal)
    pip_risk = dist_sl
    pos_size = 0.0 # Calculé par le manager
    
    sig = Signal(
        timestamp=local_time, pair=pair, timeframe=tf, action=action,
        entry_price=last.close, stop_loss=sl, take_profit=tp,
        score=score, quality=quality, position_size=0.0, risk_amount=0.0,
        risk_reward=rr, adx=int(last.adx), rsi=int(last.rsi), atr=atr,
        higher_tf_trend=higher_trend, is_live=mode_live and not df.iloc[-1]['complete'],
        is_fresh_flip=hma_flip_green if action == "BUY" else hma_flip_red,
        candle_index=idx, is_strict_flip=is_strict
    )
    return sig

class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
    def calculate_position_size(self, signal: Signal) -> float:
        win_rate = 0.58
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0: return 0.0
        size = (self.balance * kelly) / pip_risk
        return round(size, 2)

def run_scan(pairs, tfs, mode_live, risk_manager, params):
    signals = []
    # ThreadPool réduit à 5 pour OANDA (Critique point 6)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params): (p, tf) for p in pairs for tf in tfs}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    res.position_size = risk_manager.calculate_position_size(res)
                    res.risk_amount = abs(res.entry_price - res.stop_loss) * res.position_size
                    signals.append(res)
            except Exception:
                pass
    return signals

def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<b>BlueStar Cascade - Institutional Scan</b>", styles["Title"]))
    elements.append(Spacer(1, 5*mm))
    
    data = [["Time", "Pair", "TF", "Qual", "Dir", "Price", "SL", "TP", "Scr", "R:R"]]
    for s in sorted(signals, key=lambda x: x.score, reverse=True):
        data.append([
            s.timestamp.strftime("%H:%M"), s.pair.replace("_","/"), s.timeframe,
            s.quality.value[:4], s.action, f"{s.entry_price:.4f}",
            f"{s.stop_loss:.4f}", f"{s.take_profit:.4f}", str(s.score), f"{s.risk_reward:.1f}"
        ])
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    elements.append(t)
    doc.build(elements)
    return buffer.getvalue()

# ==================== INTERFACE FINALE ====================
def main():
    # En-tête avec Heure Tunis
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.markdown("# BlueStar Enhanced v2.4")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL GRADE</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
        is_open = 0 <= now_tunis.weekday() <= 4  # Lundi=0, Dimanche=6
        status_color = "#00ff88" if is_open else "#ff4b4b"
        status_text = "MARKET OPEN" if is_open else "MARKET CLOSED"
        
        st.markdown(f"""
        <div style='text-align: right; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px;'>
            <div style='color: #a0a0c0; font-size: 0.8rem;'>TUNIS TIME</div>
            <div style='font-size: 1.2rem; font-weight: bold; color: white;'>{now_tunis.strftime('%H:%M')}</div>
            <div style='color: {status_color}; font-size: 0.7rem; font-weight: bold;'>● {status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("🔍 Paramètres de Scan", expanded=True):
            scan_mode = st.radio("Mode d'analyse", ["CONFIRMED (Clôture)", "LIVE (En cours)"], index=0)
            selected_tfs = st.multiselect("Timeframes", ["H1", "H4", "D1"], default=["H1", "H4"])
            selected_pairs = st.multiselect("Paires", PAIRS_DEFAULT, default=["EUR_USD", "GBP_USD", "XAU_USD", "USD_JPY", "GBP_JPY"])
        
        with st.expander("🛡️ Risk Manager", expanded=False):
            balance = st.number_input("Capital ($)", value=10000, step=1000)
            risk_pct = st.slider("Risque par trade (%)", 0.5, 3.0, 1.0) / 100
        
        with st.expander("📊 Filtres Techniques", expanded=False):
            strict_flip = st.checkbox("Strict Flips Only", value=True, help="Ignore les continuations de tendance")
            cascade_on = st.checkbox("Cascade Tendance Sup.", value=True)
            min_score = st.slider("Score Min.", 0, 100, 70)

        st.markdown("---")
        if st.button("LANCER LE SCANNER", type="primary"):
            run_scan_trigger = True
        else:
            run_scan_trigger = False

    # Logique Principale
    if run_scan_trigger:
        if not selected_pairs or not selected_tfs:
            st.warning("⚠️ Veuillez sélectionner au moins une paire et un timeframe.")
        else:
            mode_live_bool = "LIVE" in scan_mode
            params = TradingParams(strict_flip_only=strict_flip, cascade_required=cascade_on)
            risk_mgr = RiskManager(RiskConfig(max_risk_per_trade=risk_pct), balance)
            
            with st.status("📡 Analyse des marchés institutionnels...", expanded=True) as status:
                st.write("Connexion OANDA établie...")
                signals = run_scan(selected_pairs, selected_tfs, mode_live_bool, risk_mgr, params)
                
                # Filtrage par score min
                signals = [s for s in signals if s.score >= min_score]
                status.update(label=f"✅ Scan terminé ! {len(signals)} opportunités détectées", state="complete", expanded=False)
            
            if not signals:
                st.info("Aucun signal détecté avec les critères actuels.")
            else:
                # Métriques Globales
                col1, col2, col3 = st.columns(3)
                col1.metric("Opportunités", len(signals))
                col2.metric("Meilleur Score", max(s.score for s in signals))
                col3.metric("Paire Top", sorted(signals, key=lambda x: x.score)[-1].pair.replace("_", "/"))
                
                st.markdown("### 📋 Tableau des Signaux")
                
                # Création DataFrame pour affichage propre
                df_res = pd.DataFrame([{
                    "Heure": s.timestamp.strftime("%H:%M"),
                    "Paire": s.pair.replace("_", "/"),
                    "TF": s.timeframe,
                    "Action": s.action,
                    "Prix": s.entry_price,
                    "Score": s.score,
                    "Qualité": s.quality.value,
                    "R:R": f"{s.risk_reward:.2f}",
                    "ADX": s.adx
                } for s in sorted(signals, key=lambda x: x.score, reverse=True)])
                
                # Coloration conditionnelle du tableau
                def color_action(val):
                    color = '#00ff88' if val == 'BUY' else '#ff4b4b'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(
                    df_res.style.map(color_action, subset=['Action']),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export PDF
                pdf_data = generate_pdf(signals)
                st.download_button(
                    label="📥 Télécharger Rapport PDF",
                    data=pdf_data,
                    file_name=f"BlueStar_Scan_{datetime.now().strftime('%H%M')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
