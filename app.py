"""
BlueStar Cascade - VERSION 2.7 (ALL ASSETS DEFAULT)
------------------------------------------------------------
État : FONCTIONNEL & COMPLET
Changements v2.7 :
- Ajout Indices : US30, NAS100, SPX500
- Ajout Métaux : XAU, XPT
- Sélection par défaut : TOUT coché
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 100% !important;}
    .stMetric {background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin: 0;}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.8rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.4rem !important; font-weight: 700;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.7rem; display: inline-block;}
    h1 {font-size: 2rem !important; margin-bottom: 0.5rem !important; background: -webkit-linear-gradient(#eee, #999); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .stButton button {width: 100%; border-radius: 6px; font-weight: bold; transition: all 0.3s; background-color: #00ff88; color: #000;}
    .stButton button:hover {background-color: #00cc6a; color: #fff;}
    .stProgress > div > div > div > div {background-color: #00ff88;}
</style>
""", unsafe_allow_html=True)

# ==================== LISTE DES ACTIFS ÉTENDUE ====================
PAIRS_DEFAULT = [
    # FOREX MAJEURS & MINEURS
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK",
    
    # MÉTAUX
    "XAU_USD", "XPT_USD",
    
    # INDICES (Symboles OANDA standards)
    "US30_USD", "NAS100_USD", "SPX500_USD"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D"}

LAST_REQUEST_TIME = {"time": 0}
MIN_REQUEST_INTERVAL = 0.20

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
    is_strict_flip: bool

# ==================== API & DATA ====================
@st.cache_resource
def get_oanda_client():
    if "OANDA_ACCESS_TOKEN" not in st.secrets:
        st.error("⚠️ Token OANDA manquant dans secrets.toml")
        st.stop()
    return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])

client = get_oanda_client()

@st.cache_data(ttl=15)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran: return pd.DataFrame()
    
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
        logger.error(f"Erreur API {pair}: {e}")
        return pd.DataFrame()

# ==================== INDICATEURS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return df
    
    close = df['close']
    high = df['high']
    low = df['low']

    def wma(series, length):
        if len(series) < length: return pd.Series([np.nan] * len(series), index=series.index)
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    
    if wma_half.isna().all() or wma_full.isna().all():
        df['hma'] = np.nan
        df['hma_up'] = False
    else:
        df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
        df['hma_up'] = df['hma'] > df['hma'].shift(1)

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7, min_periods=7).mean() / down.ewm(alpha=1/7, min_periods=7).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
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

    atr14 = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    df['atr_val'] = atr14
    
    return df

# ==================== ANALYSE ====================
@st.cache_data(ttl=300)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    if not higher_tf: return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50: return "Neutral"
    df = calculate_indicators(df)
    
    if pd.isna(df['hma'].iloc[-1]): return "Neutral"
    hma_curr = df['hma'].iloc[-1]
    hma_prev = df['hma'].iloc[-2]
    
    if hma_curr > hma_prev: return "Bullish"
    elif hma_curr < hma_prev: return "Bearish"
    return "Neutral"

def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager, params) -> Optional[Signal]:
    df = get_candles(pair, tf, 300)
    if len(df) < 100: return None
    df = calculate_indicators(df)
    
    if mode_live:
        idx = -1
    else:
        idx = -1 if df.iloc[-1]['complete'] else -2
        
    if abs(idx) >= len(df): return None
    
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    if pd.isna(last.hma) or pd.isna(prev.hma) or pd.isna(prev2.hma): return None
    if pd.isna(last.rsi) or pd.isna(last.adx): return None
    
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    
    hma_extended_green = last.hma_up and prev.hma_up and not prev2.hma_up and not hma_flip_green
    hma_extended_red = not last.hma_up and not prev.hma_up and prev2.hma_up and not hma_flip_red
    
    is_strict = False
    raw_buy = False
    raw_sell = False
    
    if params.strict_flip_only:
        if hma_flip_green and last.rsi > 50 and last.ut_state == 1:
            raw_buy = True; is_strict = True
        elif hma_flip_red and last.rsi < 50 and last.ut_state == -1:
            raw_sell = True; is_strict = True
    else:
        if (hma_flip_green or hma_extended_green) and last.rsi > 50 and last.ut_state == 1:
            raw_buy = True
            is_strict = hma_flip_green
        elif (hma_flip_red or hma_extended_red) and last.rsi < 50 and last.ut_state == -1:
            raw_sell = True
            is_strict = hma_flip_red

    if not (raw_buy or raw_sell): return None
    
    action = "BUY" if raw_buy else "SELL"
    
    higher_trend = get_trend_alignment(pair, tf)
    if params.cascade_required:
        if action == "BUY" and higher_trend == "Bearish": return None
        if action == "SELL" and higher_trend == "Bullish": return None
    
    score = 70
    if last.adx > params.adx_strong_threshold: score += 15
    elif last.adx > params.min_adx_threshold: score += 10
    else: score -= 5
    
    if is_strict: score += 15
    else: score += 5
    
    if action == "BUY" and 50 < last.rsi < 65: score += 5
    if action == "SELL" and 35 < last.rsi < 50: score += 5
    
    if (action == "BUY" and higher_trend == "Bullish") or (action == "SELL" and higher_trend == "Bearish"): score += 10
    
    score = max(0, min(100, score))
    quality = SignalQuality.INSTITUTIONAL if score >= 90 else SignalQuality.PREMIUM if score >= 80 else SignalQuality.STANDARD
    
    atr = last.atr_val
    sl = last.close - params.atr_sl_multiplier * atr if action == "BUY" else last.close + params.atr_sl_multiplier * atr
    tp = last.close + params.atr_tp_multiplier * atr if action == "BUY" else last.close - params.atr_tp_multiplier * atr
    
    rr = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0
    if rr < params.min_rr_ratio: return None
    
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = pytz.utc.localize(last.time).astimezone(tunis_tz)
    
    sig = Signal(
        timestamp=local_time, pair=pair, timeframe=tf, action=action,
        entry_price=last.close, stop_loss=sl, take_profit=tp,
        score=score, quality=quality, position_size=0.0, risk_amount=0.0,
        risk_reward=rr, adx=int(last.adx), rsi=int(last.rsi), atr=atr,
        higher_tf_trend=higher_trend, is_live=mode_live and not df.iloc[-1]['complete'],
        is_fresh_flip=is_strict, is_strict_flip=is_strict
    )
    return sig

class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
        
    def calculate_position_size(self, signal: Signal) -> float:
        win_rate = 0.55
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        if pip_risk <= 0: return 0.0
        size = (self.balance * kelly) / pip_risk 
        return round(size, 2)

# ==================== MOTEUR SÉQUENTIEL ====================
def run_scan_sequential(pairs, tfs, mode_live, risk_manager, params):
    signals = []
    progress_text = "Scan en cours..."
    bar = st.progress(0, text=progress_text)
    
    total_steps = len(pairs) * len(tfs)
    current_step = 0
    
    for pair in pairs:
        for tf in tfs:
            current_step += 1
            pct = current_step / total_steps
            bar.progress(pct, text=f"Analyse {pair} [{tf}]...")
            
            try:
                res = analyze_pair(pair, tf, mode_live, risk_manager, params)
                if res:
                    res.position_size = risk_manager.calculate_position_size(res)
                    res.risk_amount = abs(res.entry_price - res.stop_loss) * res.position_size
                    signals.append(res)
            except Exception as e:
                logger.error(f"Erreur Scan {pair}: {e}")
                continue
                
    bar.empty()
    return signals

def generate_pdf(signals: List[Signal]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("<b>BlueStar Cascade - Institutional Report</b>", styles["Title"]))
    elements.append(Spacer(1, 5*mm))
    
    now = datetime.now(pytz.timezone('Africa/Tunis')).strftime('%d/%m/%Y %H:%M')
    elements.append(Paragraph(f"Généré le: {now} (Tunis Time)", styles["Normal"]))
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
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(t)
    doc.build(elements)
    return buffer.getvalue()

# ==================== INTERFACE ====================
def main():
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.markdown("# BlueStar Enhanced v2.7")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL GRADE • ALL ASSETS</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
        is_open = 0 <= now_tunis.weekday() <= 4
        st.markdown(f"""
        <div style='text-align: right; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px;'>
            <div style='color: #a0a0c0; font-size: 0.8rem;'>TUNIS TIME</div>
            <div style='font-size: 1.2rem; font-weight: bold; color: white;'>{now_tunis.strftime('%H:%M')}</div>
            <div style='color: {'#00ff88' if is_open else '#ff4b4b'}; font-size: 0.7rem;'>{'● MARKET OPEN' if is_open else '● MARKET CLOSED'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("Scan Settings", expanded=True):
            scan_mode = st.radio("Mode", ["CONFIRMED (Clôture)", "LIVE (Temps réel)"], index=0)
            selected_tfs = st.multiselect("Timeframes", ["H1", "H4", "D1"], default=["H1", "H4"])
            
            # Sélection par défaut : TRUE
            all_pairs = st.checkbox("Toutes les paires/actifs", value=True)
            if all_pairs:
                selected_pairs = PAIRS_DEFAULT
                st.caption(f"✅ {len(PAIRS_DEFAULT)} actifs sélectionnés (Indices inclus)")
            else:
                selected_pairs = st.multiselect("Paires", PAIRS_DEFAULT, default=["EUR_USD", "XAU_USD", "US30_USD"])
        
        with st.expander("Risk Manager", expanded=False):
            balance = st.number_input("Capital", value=10000)
            risk_pct = st.slider("Risk %", 0.5, 5.0, 1.0) / 100
        
        with st.expander("Filtres Techniques", expanded=True):
            strict_flip = st.checkbox("Strict Flips Only", value=True)
            cascade_on = st.checkbox("Cascade Tendance", value=True)
            min_score = st.slider("Score Min", 0, 100, 60)

        st.markdown("---")
        launch = st.button("LANCER LE SCANNER", type="primary")

    if launch:
        if not selected_pairs or not selected_tfs:
            st.warning("Selectionnez au moins une paire et un TF.")
        else:
            mode_live_bool = "LIVE" in scan_mode
            params = TradingParams(strict_flip_only=strict_flip, cascade_required=cascade_on)
            risk_mgr = RiskManager(RiskConfig(max_risk_per_trade=risk_pct), balance)
            
            signals = run_scan_sequential(selected_pairs, selected_tfs, mode_live_bool, risk_mgr, params)
            signals = [s for s in signals if s.score >= min_score]
            
            if not signals:
                st.info("Aucun signal détecté. Essayez de décocher 'Strict Flips Only' ou changez de TF.")
            else:
                st.success(f"{len(signals)} opportunités trouvées !")
                
                kpi1, kpi2, kpi3 = st.columns(3)
                best_sig = sorted(signals, key=lambda x: x.score)[-1]
                kpi1.metric("Opportunités", len(signals))
                kpi2.metric("Meilleur Score", best_sig.score)
                kpi3.metric("Top Asset", best_sig.pair.replace("_","/"))
                
                st.markdown("### 📋 Signaux Détectés")
                df_view = pd.DataFrame([{
                    "Heure": s.timestamp.strftime("%H:%M"),
                    "Paire": s.pair.replace("_", "/"),
                    "TF": s.timeframe,
                    "Action": s.action,
                    "Prix": s.entry_price,
                    "Score": s.score,
                    "Type": "Strict" if s.is_strict_flip else "Extend",
                    "R:R": s.risk_reward
                } for s in sorted(signals, key=lambda x: x.score, reverse=True)])
                
                def color_action(val):
                    color = '#00ff88' if val == 'BUY' else '#ff4b4b'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(df_view.style.map(color_action, subset=['Action']), hide_index=True)
                
                pdf = generate_pdf(signals)
                st.download_button("📥 Télécharger PDF", pdf, "BlueStar_Report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
