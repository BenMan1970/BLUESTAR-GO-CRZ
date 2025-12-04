"""
BlueStar Cascade - VERSION CORRIGÉE
Corrections majeures :
- Gestion correcte des bougies complètes/incomplètes
- Détection stricte des flips HMA
- Validation des indicateurs
- Logging détaillé pour debugging
- Cache optimisé
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
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="collapsed")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 1rem !important; max-width: 100% !important;}
    .stMetric {background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.7rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.2rem !important; font-weight: 700;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 3px 10px; border-radius: 15px; font-weight: bold; font-size: 0.65rem; display: inline-block;}
    .stDataFrame {font-size: 0.75rem !important;}
    .stDataFrame div[data-testid="stDataFrame"] {height: auto !important;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .tf-header {background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2)); padding: 10px; border-radius: 8px; text-align: center; margin-bottom: 10px; border: 2px solid rgba(0,255,136,0.3);}
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    h2 {font-size: 1.2rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
    .alert-box {background: rgba(255,200,0,0.1); border-left: 3px solid #ffc800; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.8rem;}
    .debug-box {background: rgba(0,200,255,0.1); border-left: 3px solid #00c8ff; padding: 8px; border-radius: 4px; margin: 5px 0; font-size: 0.7rem; font-family: monospace;}
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
    strict_flip_only: bool = False  # NOUVEAU : mode flip strict

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
    candle_index: int  # NOUVEAU : pour tracking
    is_strict_flip: bool  # NOUVEAU : flip strict vs étendu

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except Exception as e:
        logger.error(f"OANDA Token Error: {e}")
        st.error("⚠️ OANDA Token manquant ou invalide dans les secrets Streamlit")
        st.stop()

client = get_oanda_client()

@st.cache_data(ttl=5)  # CORRIGÉ : 5 secondes au lieu de 15
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran:
        logger.warning(f"Timeframe invalide: {tf}")
        return pd.DataFrame()
    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
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
        logger.debug(f"✓ {pair} {tf}: {len(df)} bougies | Last complete: {df.iloc[-1]['complete'] if not df.empty else 'N/A'}")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur API {pair} {tf}: {e}")
        return pd.DataFrame()

# ==================== INDICATEURS CORRIGÉS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcul des indicateurs avec validation renforcée"""
    if len(df) < 50:
        logger.warning("Pas assez de données pour calculer les indicateurs")
        return df
    
    close = df['close']
    high = df['high']
    low = df['low']

    # WMA avec validation
    def wma(series, length):
        if len(series) < length:
            return pd.Series([np.nan] * len(series), index=series.index)
        weights = np.arange(1, length + 1)
        return series.rolling(length, min_periods=length).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == length else np.nan,
            raw=True
        )

    # HMA (Hull Moving Average) - CORRIGÉ
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    hma_length = int(np.sqrt(20))
    
    if wma_half.isna().all() or wma_full.isna().all():
        df['hma'] = np.nan
        df['hma_up'] = False
        logger.warning("HMA: Impossible de calculer (WMA invalides)")
    else:
        df['hma'] = wma(2 * wma_half - wma_full, hma_length)
        df['hma_up'] = df['hma'] > df['hma'].shift(1)
        # Remplir les premiers NaN pour éviter les erreurs
        df['hma_up'].fillna(False, inplace=True)

    # RSI (7 périodes)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7, min_periods=7).mean() / down.ewm(alpha=1/7, min_periods=7).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    # UT Bot (ATR Trailing Stop)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    xATR = tr.rolling(1).mean()
    nLoss = 2.0 * xATR
    xATRTrailingStop = [0.0] * len(df)
    
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

    # ADX (14 périodes)
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
@st.cache_data(ttl=60)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    """Analyse de la tendance sur le timeframe supérieur"""
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf:
        return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50:
        return "Neutral"
    
    df = calculate_indicators(df)
    
    # Vérifier validité des indicateurs
    if pd.isna(df['hma'].iloc[-1]) or pd.isna(df['hma'].iloc[-2]):
        return "Neutral"
    
    close = df['close']
    ema50 = close.ewm(span=50, min_periods=50).mean().iloc[-1]
    hma_current = df['hma'].iloc[-1]
    hma_prev = df['hma'].iloc[-2]
    
    if close.iloc[-1] > ema50 and hma_current > hma_prev:
        return "Bullish"
    elif close.iloc[-1] < ema50 and hma_current < hma_prev:
        return "Bearish"
    
    return "Neutral"

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
    
    def calculate_position_size(self, signal: Signal) -> float:
        """Calcul de la taille de position avec Kelly Criterion"""
        win_rate = 0.58
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0:
            return 0.0
        
        size = (self.balance * kelly) / pip_risk
        return round(size, 2)

# ==================== ANALYSE CORRIGÉE ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, params: TradingParams) -> Optional[Signal]:
    """Analyse d'une paire avec gestion corrigée des bougies"""
    
    df = get_candles(pair, tf, 300)
    if len(df) < 100:
        logger.debug(f"⚠️ {pair} {tf}: Pas assez de données ({len(df)} bougies)")
        return None
    
    df = calculate_indicators(df)
    
    # ========== CORRECTION CRITIQUE : GESTION DES BOUGIES ==========
    if mode_live:
        # Mode LIVE : analyser la bougie en cours (risqué mais temps réel)
        idx = -1
        logger.debug(f"🔴 LIVE MODE: {pair} {tf} - Analyse bougie en cours (idx=-1)")
    else:
        # Mode CONFIRMED : TOUJOURS prendre la dernière bougie COMPLÈTE
        if not df.iloc[-1]['complete']:
            idx = -2  # Avant-dernière bougie (complète)
            logger.debug(f"🟢 CONFIRMED MODE: {pair} {tf} - Dernière bougie incomplète, analyse idx=-2")
        else:
            idx = -1  # Dernière bougie (vient de se fermer)
            logger.debug(f"🟢 CONFIRMED MODE: {pair} {tf} - Dernière bougie complète, analyse idx=-1")
    
    # Vérification de sécurité
    if abs(idx) > len(df) - 2:
        logger.warning(f"⚠️ {pair} {tf}: Index {idx} hors limites (len={len(df)})")
        return None
    
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    # Validation des indicateurs
    if pd.isna(last.hma) or pd.isna(last.rsi) or pd.isna(last.adx) or pd.isna(last.atr_val):
        logger.warning(f"⚠️ {pair} {tf}: Indicateurs invalides à idx={idx}")
        return None
    
    # ========== DÉTECTION DES FLIPS (STRICTE) ==========
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    
    # Détection étendue (optionnelle)
    hma_extended_green = last.hma_up and not prev2.hma_up and not hma_flip_green
    hma_extended_red = not last.hma_up and prev2.hma_up and not hma_flip_red
    
    # Conditions BUY/SELL
    if params.strict_flip_only:
        # Mode strict : SEULEMENT les flips directs
        raw_buy = hma_flip_green and last.rsi > 50 and last.ut_state == 1
        raw_sell = hma_flip_red and last.rsi < 50 and last.ut_state == -1
        is_strict = True
    else:
        # Mode standard : flips directs OU étendus
        raw_buy = (hma_flip_green or hma_extended_green) and last.rsi > 50 and last.ut_state == 1
        raw_sell = (hma_flip_red or hma_extended_red) and last.rsi < 50 and last.ut_state == -1
        is_strict = hma_flip_green or hma_flip_red
    
    if not (raw_buy or raw_sell):
        return None
    
    action = "BUY" if raw_buy else "SELL"
    
    # ========== LOGGING DÉTAILLÉ ==========
    logger.info(f"""
    🎯 SIGNAL DÉTECTÉ : {pair} {tf}
    ├─ Mode: {'🔴 LIVE' if mode_live else '🟢 CONFIRMED'}
    ├─ Bougie analysée: idx={idx} | complete={df.iloc[-1]['complete']}
    ├─ Timestamp: {last.time}
    ├─ HMA: {last.hma:.5f} (up={last.hma_up})
    ├─ HMA prev: {prev.hma:.5f} (up={prev.hma_up})
    ├─ Flip: Strict={hma_flip_green or hma_flip_red} | Extended={hma_extended_green or hma_extended_red}
    ├─ RSI: {last.rsi:.1f} | ADX: {last.adx:.1f} | UT: {last.ut_state}
    └─ Action: {action}
    """)
    
    # Cascade (vérification timeframe supérieur)
    higher_trend = get_trend_alignment(pair, tf)
    if params.cascade_required:
        if action == "BUY" and higher_trend != "Bullish":
            logger.debug(f"❌ {pair} {tf}: BUY rejeté (cascade {higher_trend})")
            return None
        if action == "SELL" and higher_trend != "Bearish":
            logger.debug(f"❌ {pair} {tf}: SELL rejeté (cascade {higher_trend})")
            return None
    
    # Scoring
    score = 70
    
    # ADX
    if last.adx > params.adx_strong_threshold:
        score += 15
    elif last.adx > params.min_adx_threshold:
        score += 10
    else:
        score -= 5
    
    # Type de flip
    if hma_flip_green or hma_flip_red:
        score += 15  # Flip strict = meilleur score
    elif hma_extended_green or hma_extended_red:
        score += 5  # Flip étendu = score moindre
    
    # RSI optimal
    if (action == "BUY" and 50 < last.rsi < 65) or (action == "SELL" and 35 < last.rsi < 50):
        score += 5
    
    # Cascade alignée
    if (action == "BUY" and higher_trend == "Bullish") or (action == "SELL" and higher_trend == "Bearish"):
        score += 10
    
    score = max(50, min(100, score))
    quality = SignalQuality.INSTITUTIONAL if score >= 90 else SignalQuality.PREMIUM if score >= 80 else SignalQuality.STANDARD
    
    # Calcul SL/TP
    atr = last.atr_val
    sl = last.close - params.atr_sl_multiplier * atr if action == "BUY" else last.close + params.atr_sl_multiplier * atr
    tp = last.close + params.atr_tp_multiplier * atr if action == "BUY" else last.close - params.atr_tp_multiplier * atr
    
    rr = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0
    if rr < params.min_rr_ratio:
        logger.debug(f"❌ {pair} {tf}: R:R insuffisant ({rr:.2f})")
        return None
    
    # Timestamp (Tunis)
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = last.time.astimezone(tunis_tz) if last.time.tzinfo else pytz.utc.localize(last.time).astimezone(tunis_tz)
    
    signal = Signal(
        timestamp=local_time,
        pair=pair,
        timeframe=tf,
        action=action,
        entry_price=last.close,
        stop_loss=sl,
        take_profit=tp,
        score=score,
        quality=quality,
        position_size=0.0,
        risk_amount=0.0,
        risk_reward=rr,
        adx=int(last.adx),
        rsi=int(last.rsi),
        atr=atr,
        higher_tf_trend=higher_trend,
        is_live=mode_live and not df.iloc[-1]['complete'],
        is_fresh_flip=hma_flip_green if action == "BUY" else hma_flip_red,
        candle_index=idx,
        is_strict_flip=is_strict
    )
    
    signal.position_size = risk_manager.calculate_position_size(signal)
    signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
    
    logger.info(f"✅ Signal validé: {pair} {tf} {action} @ {signal.entry_price:.5f} | Score: {score} | R:R: {rr:.1f}")
    
    return signal

# ==================== SCAN ====================
def run_scan(pairs, tfs, mode_live, risk_manager, params):
    """Scan multi-threadé avec gestion d'erreurs"""
    signals = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params): (p, tf)
            for p in pairs for tf in tfs
        }
        
        for future in as_completed(futures):
            pair, tf = futures[future]
            try:
                result = future.result()
                if result:
                    signals.append(result)
            except Exception as e:
                error_msg = f"{pair} {tf}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ Erreur: {error_msg}")
    
    if errors:
        logger.warning(f"⚠️ {len(errors)} erreurs pendant le scan")
    
    return signals, errors

# ==================== PDF GENERATOR ====================
def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, leftMargin=10*mm, rightMargin=10*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<font size=16 color=#00ff88><b>BlueStar Cascade - Signaux Institutionnels</b></font>", styles["Title"]))
    elements.append(Spacer(1, 8*mm))
    
    now = datetime.now(pytz.timezone('Africa/Tunis')).strftime('%d/%m/%Y %H:%M:%S')
    elements.append(Paragraph(f"<font size=10 color=#a0a0c0>Généré le {now} (Tunis)</font>", styles["Normal"]))
    elements.append(Spacer(1, 10*mm))
    
    data = [["Heure", "Paire", "TF", "Qualité", "Action", "Entry", "SL", "TP", "Score", "R:R", "Taille", "Risque $", "ADX", "RSI", "Tendance Sup.", "Flip Type", "Live"]]
    
    for s in sorted(signals, key=lambda x: (x.score, x.timestamp), reverse=True):
        flip_type = "Strict" if s.is_strict_flip else "Extended"
        data.append([
            s.timestamp.strftime("%H:%M"),
            s.pair.replace("_", "/"),
            s.timeframe,
            s.quality.value,
            s.action,
            f"{s.entry_price:.5f}",
            f"{s.stop_loss:.5f}",
            f"{s.take_profit:.5f}",
            str(s.score),
            f"{s.risk_reward:.1f}",
            f"{s.position_size:.2f}",
            f"{s.risk_amount:.0f}",
            str(s.adx),
            str(s.rsi),
            s.higher_tf_trend,
            flip_type,
            "Oui" if s.is_live else "Non"
        ])
    
    table = Table(data, colWidths=[14*mm,18*mm,10*mm,20*mm,14*mm,18*mm,18*mm,18*mm,10*mm,10*mm,14*mm,14*mm,10*mm,10*mm,20*mm,16*mm,12*mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a1f3a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#00ff88")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#0f1429")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#333")),
        ('FONTSIZE', (0,1), (-1,-1), 7),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()

# ==================== INTERFACE ====================
def main():
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    
    with col_title:
        st.markdown("# BlueStar Enhanced v2.2")
        st.markdown('<span class="institutional-badge">CORRECTED VERSION</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
        st.markdown(f"<div style='text-align: right; padding-top: 10px;'><span style='color: #a0a0c0; font-size: 0.8rem;'>🕐 {now_tunis.strftime('%H:%M:%S')}</span><br><span style='color: {'#00ff88' if now_tunis.hour in range(0,23) else '#ff6666'};'>{'OPEN' if now_tunis.hour in range(0,23) else 'CLOSED'}</span></div>", unsafe_allow_html=True)
    
    with col_mode:
        mode = st.radio("Mode", ["🟢 Confirmed", "🔴 Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode

    # Configuration
    with st.expander("⚙️ Configuration Avancée", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        atr_sl = c1.number_input("SL Multiplier (ATR)", 1.0, 4.0, 2.0, 0.5)
        atr_tp = c2.number_input("TP Multiplier (ATR)", 1.5, 6.0, 3.0, 0.5)
        min_rr = c3.number_input("Min R:R", 1.0, 3.0, 1.2, 0.1)
        cascade_req = c4.checkbox("Cascade obligatoire", True)
        strict_flip = c5.checkbox("Flip strict uniquement", False)

    # Risk Management
    c1, c2, c3, c4 = st.columns(4)
    balance = c1.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    max_risk = c2.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio = c3.slider("Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    scan_btn = c4.button("🔍 SCAN MARKETS", type="primary", use_container_width=True)

    # Scan
    if scan_btn:
        with st.spinner("🔄 Scanning markets..."):
            params = TradingParams(
                atr_sl_multiplier=atr_sl,
                atr_tp_multiplier=atr_tp,
                min_rr_ratio=min_rr,
                cascade_required=cascade_req,
                strict_flip_only=strict_flip
            )
            risk_manager = RiskManager(
                RiskConfig(max_risk_per_trade=max_risk, max_portfolio_risk=max_portfolio),
                balance
            )
            signals, errors = run_scan(PAIRS_DEFAULT, ["H1", "H4", "D1"], is_live, risk_manager, params)

        # Affichage des erreurs
        if errors:
            with st.expander(f"⚠️ {len(errors)} erreur(s) détectée(s)", expanded=False):
                for err in errors[:10]:  # Max 10 erreurs affichées
                    st.markdown(f'<div class="debug-box">❌ {err}</div>', unsafe_allow_html=True)

        if signals:
            st.markdown("---")
            
            # Métriques
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Signaux", len(signals))
            m2.metric("Institutional", len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL]))
            m3.metric("Flip Strict", len([s for s in signals if s.is_strict_flip]))
            m4.metric("Score Moyen", f"{np.mean([s.score for s in signals]):.0f}")
            m5.metric("Exposition", f"${sum(s.risk_amount for s in signals):.0f}")
            m6.metric("R:R Moyen", f"{np.mean([s.risk_reward for s in signals]):.1f}:1")

            st.markdown("---")
            
            # Export
            dl1, dl2, dl3 = st.columns([1,1,1])
            
            # CSV
            with dl1:
                df_csv = pd.DataFrame([{
                    "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Pair": s.pair.replace("_","/"),
                    "TF": s.timeframe,
                    "Quality": s.quality.value,
                    "Action": s.action,
                    "Entry": s.entry_price,
                    "SL": s.stop_loss,
                    "TP": s.take_profit,
                    "Score": s.score,
                    "R:R": s.risk_reward,
                    "Size": s.position_size,
                    "Risk $": s.risk_amount,
                    "ADX": s.adx,
                    "RSI": s.rsi,
                    "Higher TF": s.higher_tf_trend,
                    "Flip Type": "Strict" if s.is_strict_flip else "Extended",
                    "Live": s.is_live,
                    "Candle Index": s.candle_index
                } for s in signals])
                st.download_button(
                    "📥 Télécharger CSV",
                    df_csv.to_csv(index=False).encode(),
                    f"bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            # PDF
            with dl3:
                pdf = generate_pdf(signals)
                st.download_button(
                    "📄 Télécharger PDF",
                    pdf,
                    f"bluestar_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                    use_container_width=True
                )

            st.markdown("---")
            
            # Affichage par timeframe
            col_h1, col_h4, col_d1 = st.columns(3)
            
            for col, tf in zip([col_h1, col_h4, col_d1], ["H1", "H4", "D1"]):
                with col:
                    tf_sig = [s for s in signals if s.timeframe == tf]
                    st.markdown(f"<div class='tf-header'><h3>{tf}</h3><p>{len(tf_sig)} signal{'s' if len(tf_sig)>1 else ''}</p></div>", unsafe_allow_html=True)
                    
                    if tf_sig:
                        tf_sig.sort(key=lambda x: (x.score, x.timestamp), reverse=True)
                        
                        df_disp = pd.DataFrame([{
                            "Heure": s.timestamp.strftime("%H:%M"),
                            "Paire": s.pair.replace("_","/"),
                            "Qualité": s.quality.value,
                            "Action": s.action,
                            "Type": "🎯" if s.is_strict_flip else "📊",
                            "Score": s.score,
                            "Entry": f"{s.entry_price:.5f}",
                            "SL": f"{s.stop_loss:.5f}",
                            "TP": f"{s.take_profit:.5f}",
                            "R:R": f"{s.risk_reward:.1f}",
                            "Taille": f"{s.position_size:.2f}",
                            "Risque": f"${s.risk_amount:.0f}",
                            "ADX": s.adx,
                            "RSI": s.rsi,
                            "Cascade": s.higher_tf_trend
                        } for s in tf_sig])
                        
                        st.dataframe(df_disp, use_container_width=True, hide_index=True)
                        
                        # Debug info (optionnel)
                        if st.checkbox(f"Debug {tf}", key=f"debug_{tf}"):
                            for s in tf_sig[:3]:  # Max 3 signaux
                                st.markdown(f"""
                                <div class='debug-box'>
                                <b>{s.pair} {s.action}</b><br>
                                Timestamp: {s.timestamp}<br>
                                Candle Index: {s.candle_index}<br>
                                Flip Type: {'Strict' if s.is_strict_flip else 'Extended'}<br>
                                Live: {s.is_live}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun signal")

        else:
            st.warning("🔍 Aucun signal détecté avec les paramètres actuels")
            st.info("💡 Essayez de désactiver le mode 'Flip strict uniquement' ou la cascade obligatoire")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.7rem; padding: 15px;'>
        <b>BlueStar Cascade Enhanced v2.2</b> - Version Corrigée<br>
        ✅ Gestion correcte des bougies | ✅ Détection stricte des flips | ✅ Logging détaillé
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
