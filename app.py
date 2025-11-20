"""
BlueStar Cascade - Enhanced Institutional Grade
Vue 3 colonnes : H1 | H4 | D1
Améliorations : Logging, Paramètres configurables, Validation des données
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# OANDA API
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="collapsed")

# Logging amélioré
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# CSS Ultra-compact
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); padding: 1rem !important;}
    .block-container {padding-top: 2rem !important; padding-bottom: 1rem !important; max-width: 100% !important;}
    
    .stMetric {
        background: rgba(255,255,255,0.05); 
        padding: 8px; 
        border-radius: 6px; 
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0;
    }
    .stMetric label {color: #a0a0c0 !important; font-size: 0.7rem !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.2rem !important; font-weight: 700;}
    
    .institutional-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4e); 
        color: black; 
        padding: 3px 10px; 
        border-radius: 15px; 
        font-weight: bold; 
        font-size: 0.65rem;
        display: inline-block;
    }
    
    /* Tableau compact */
    .stDataFrame {font-size: 0.75rem !important;}
    .stDataFrame div[data-testid="stDataFrame"] {height: auto !important;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    /* Headers timeframes */
    .tf-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,200,255,0.2));
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
        border: 2px solid rgba(0,255,136,0.3);
    }
    .tf-header h3 {margin: 0; color: #00ff88; font-size: 1.2rem;}
    .tf-header p {margin: 3px 0; color: #a0a0c0; font-size: 0.7rem;}
    
    /* Compact top bar */
    .top-bar {
        background: rgba(255,255,255,0.03);
        padding: 8px 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    h2 {font-size: 1.2rem !important; margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
    
    /* Alert box */
    .alert-box {
        background: rgba(255,200,0,0.1);
        border-left: 3px solid #ffc800;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Paires Forex + Métaux précieux
PAIRS_DEFAULT = [
    # Forex Majors
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    # Forex Crosses
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK",
    # Métaux précieux
    "XAU_USD",  # Or
    "XPT_USD"   # Platine
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D"}

# ==================== DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "🏦"
    PREMIUM = "⭐"
    STANDARD = "✓"

@dataclass
class TradingParams:
    """Paramètres de trading configurables"""
    atr_sl_multiplier: float = 2.0  # Stop Loss = Entry ± (ATR * multiplier)
    atr_tp_multiplier: float = 3.0  # Take Profit = Entry ± (ATR * multiplier)
    min_adx_threshold: int = 20     # ADX minimum pour signaux de qualité
    adx_strong_threshold: int = 25  # ADX pour bonus de score
    min_rr_ratio: float = 1.2       # Risk:Reward minimum acceptable
    cascade_required: bool = True   # Cascade multi-TF obligatoire?
    
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
    rejection_reason: Optional[str] = None

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except Exception as e:
        logger.error(f"⚠️ OANDA Token Error: {e}")
        st.error("⚠️ OANDA Token manquant")
        st.stop()

client = get_oanda_client()

@st.cache_data(ttl=15)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if not gran:
        logger.warning(f"Granularité invalide: {tf}")
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
            # Validation: détection de prix anormaux
            if validate_price_data(df, pair):
                logger.info(f"✓ {pair} {tf}: {len(df)} candles chargées")
            else:
                logger.warning(f"⚠️ {pair} {tf}: Données suspectes détectées")
        
        return df
    except Exception as e:
        logger.error(f"❌ Erreur API {pair} {tf}: {str(e)[:100]}")
        return pd.DataFrame()

def validate_price_data(df: pd.DataFrame, pair: str) -> bool:
    """Validation basique des données de prix"""
    if df.empty or len(df) < 50:
        return False
    
    # Vérifier les valeurs nulles
    if df[['open', 'high', 'low', 'close']].isnull().any().any():
        logger.warning(f"{pair}: Valeurs nulles détectées")
        return False
    
    # Vérifier cohérence High/Low
    invalid_bars = (df['high'] < df['low']).sum()
    if invalid_bars > 0:
        logger.warning(f"{pair}: {invalid_bars} barres invalides (high < low)")
        return False
    
    # Détection de gaps extrêmes (> 5% entre closes)
    price_changes = df['close'].pct_change().abs()
    extreme_gaps = (price_changes > 0.05).sum()
    if extreme_gaps > 2:
        logger.warning(f"{pair}: {extreme_gaps} gaps extrêmes détectés")
    
    return True

# ==================== INDICATEURS ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close']
    high = df['high']
    low = df['low']
    
    # HMA 20
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    df['hma'] = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
    df['hma_up'] = df['hma'] > df['hma'].shift(1)
    
    # RSI 7
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/7).mean() / down.ewm(alpha=1/7).mean()
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # UT BOT (ATR period = 1 pour réactivité)
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
    
    # ADX avec ATR 14 pour SL/TP
    atr14 = tr.ewm(alpha=1/14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()
    df['atr_val'] = atr14
    
    return df

# ==================== CASCADE ====================
@st.cache_data(ttl=60)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    
    if not higher_tf:
        return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50:
        logger.warning(f"Cascade {pair} {higher_tf}: Données insuffisantes")
        return "Neutral"
    
    close = df['close']
    ema50 = close.ewm(span=50).mean().iloc[-1]
    
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    hma = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
    
    hma_now = hma.iloc[-1]
    hma_prev = hma.iloc[-2]
    price = close.iloc[-1]
    
    if price > ema50 and hma_now > hma_prev:
        return "Bullish"
    elif price < ema50 and hma_now < hma_prev:
        return "Bearish"
    return "Neutral"

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
        self.open_positions = []
    
    def calculate_position_size(self, signal: Signal) -> float:
        risk_amount = self.balance * self.config.max_risk_per_trade
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        
        # Kelly Criterion avec win rate réaliste
        win_rate = 0.58  # Basé sur backtests (à valider)
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        position_size = (self.balance * kelly) / pip_risk if pip_risk > 0 else 0
        
        logger.debug(f"Position size {signal.pair}: Kelly={kelly:.3f}, Size={position_size:.2f}")
        return round(position_size, 2)

# ==================== SIGNAL GENERATOR ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager, 
                 params: TradingParams) -> Optional[Signal]:
    df = get_candles(pair, tf, 300)
    if len(df) < 100:
        logger.debug(f"{pair} {tf}: Données insuffisantes ({len(df)} candles)")
        return None
    
    df = calculate_indicators(df)
    
    if mode_live:
        idx = -1
        is_live_signal = not df.iloc[-1]['complete']
    else:
        idx = -2 if not df.iloc[-1]['complete'] else -1
        is_live_signal = False
    
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    # BlueStar Logic
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    rsi_ok_buy = last.rsi > 50
    rsi_ok_sell = last.rsi < 50
    ut_bull = last.ut_state == 1
    ut_bear = last.ut_state == -1
    
    raw_buy = (hma_flip_green or (last.hma_up and not prev2.hma_up)) and rsi_ok_buy and ut_bull
    raw_sell = (hma_flip_red or (not last.hma_up and prev2.hma_up)) and rsi_ok_sell and ut_bear
    
    if not (raw_buy or raw_sell):
        return None
    
    action = "BUY" if raw_buy else "SELL"
    is_fresh_flip = (action == "BUY" and hma_flip_green) or (action == "SELL" and hma_flip_red)
    
    # Cascade (optionnel selon params)
    higher_trend = get_trend_alignment(pair, tf)
    if params.cascade_required:
        if action == "BUY" and higher_trend != "Bullish":
            logger.info(f"⚠️ Cascade reject: {pair} {tf} BUY vs {higher_trend}")
            return None
        if action == "SELL" and higher_trend != "Bearish":
            logger.info(f"⚠️ Cascade reject: {pair} {tf} SELL vs {higher_trend}")
            return None
    
    # Scoring
    score = 70
    if last.adx > params.adx_strong_threshold:
        score += 15
    elif last.adx > params.min_adx_threshold:
        score += 10
    else:
        # Pénalité légère si ADX faible (mais pas de rejet)
        score -= 5
        
    if is_fresh_flip:
        score += 15
    if action == "BUY" and 50 < last.rsi < 65:
        score += 5
    elif action == "SELL" and 35 < last.rsi < 50:
        score += 5
    
    score = max(50, min(100, score))  # Borné entre 50 et 100
    
    if score >= 90:
        quality = SignalQuality.INSTITUTIONAL
    elif score >= 80:
        quality = SignalQuality.PREMIUM
    else:
        quality = SignalQuality.STANDARD
    
    # SL/TP configurables
    atr = last.atr_val
    if action == "BUY":
        sl = last.close - params.atr_sl_multiplier * atr
        tp = last.close + params.atr_tp_multiplier * atr
    else:
        sl = last.close + params.atr_sl_multiplier * atr
        tp = last.close - params.atr_tp_multiplier * atr
    
    rr_ratio = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0
    
    # Filtre R:R minimum
    if rr_ratio < params.min_rr_ratio:
        logger.info(f"⚠️ R:R reject: {pair} {tf} ({rr_ratio:.2f} < {params.min_rr_ratio})")
        return None
    
    # Timezone
    utc_time = last.time
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = utc_time.astimezone(tunis_tz)
    
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
        risk_reward=rr_ratio,
        adx=int(last.adx),
        rsi=int(last.rsi),
        atr=atr,
        higher_tf_trend=higher_trend,
        is_live=is_live_signal,
        is_fresh_flip=is_fresh_flip
    )
    
    signal.position_size = risk_manager.calculate_position_size(signal)
    signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
    
    logger.info(f"✓ Signal: {pair} {tf} {action} | Score={score} | ADX={int(last.adx)} | R:R={rr_ratio:.1f}")
    
    return signal

# ==================== SCANNER ====================
def run_scan(pairs: List[str], tfs: List[str], mode_live: bool, 
             risk_manager: RiskManager, params: TradingParams) -> List[Signal]:
    signals = []
    total_pairs = len(pairs) * len(tfs)
    logger.info(f"🔍 Scan démarré: {len(pairs)} paires × {len(tfs)} TF = {total_pairs} analyses")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_pair, p, tf, mode_live, risk_manager, params) 
                   for p in pairs for tf in tfs]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    signals.append(result)
            except Exception as e:
                logger.error(f"❌ Erreur analyse: {str(e)[:100]}")
    
    logger.info(f"✅ Scan terminé: {len(signals)} signaux détectés sur {total_pairs} analyses")
    return signals

# ==================== INTERFACE ====================
def main():
    # Top bar compact
    col_title, col_time, col_mode = st.columns([3, 2, 2])
    
    with col_title:
        st.markdown("# 💎 BlueStar Enhanced")
        st.markdown('<span class="institutional-badge">INSTITUTIONAL GRADE</span>', unsafe_allow_html=True)
    
    with col_time:
        now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
        market_open = 0 <= now_tunis.hour < 23
        st.markdown(f"""
        <div style='text-align: right; padding-top: 10px;'>
        <span style='color: #a0a0c0; font-size: 0.8rem;'>🕐 {now_tunis.strftime('%H:%M:%S')}</span><br>
        <span style='color: {"#00ff88" if market_open else "#ff6666"}; font-size: 0.75rem;'>{"🟢 OPEN" if market_open else "🔴 CLOSED"}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_mode:
        mode = st.radio("Mode", ["✅ Confirmed", "⚡ Live"], horizontal=True, label_visibility="collapsed")
        is_live = "Live" in mode
    
    # Config avec expander pour paramètres avancés
    with st.expander("⚙️ Configuration Avancée", expanded=False):
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            atr_sl = st.number_input("SL Multiplier (ATR)", 1.0, 4.0, 2.0, 0.5)
        with col_a2:
            atr_tp = st.number_input("TP Multiplier (ATR)", 1.5, 6.0, 3.0, 0.5)
        with col_a3:
            min_rr = st.number_input("Min R:R", 1.0, 3.0, 1.2, 0.1)
        with col_a4:
            cascade_req = st.checkbox("Cascade obligatoire", value=True)
    
    # Config de base
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        balance = st.number_input("Balance ($)", 1000, 1000000, 10000, 1000)
    with col_c2:
        max_risk = st.slider("Risk/Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    with col_c3:
        max_portfolio = st.slider("Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    with col_c4:
        scan_btn = st.button("🚀 SCAN", type="primary", use_container_width=True)
    
    if scan_btn:
        with st.spinner("🔍 Scanning markets..."):
            # Params
            trading_params = TradingParams(
                atr_sl_multiplier=atr_sl,
                atr_tp_multiplier=atr_tp,
                min_rr_ratio=min_rr,
                cascade_required=cascade_req
            )
            
            risk_config = RiskConfig(max_risk_per_trade=max_risk, max_portfolio_risk=max_portfolio)
            risk_manager = RiskManager(risk_config, balance)
            
            signals = run_scan(PAIRS_DEFAULT, ["H1", "H4", "D1"], is_live, risk_manager, trading_params)
        
        if signals:
            # Métriques globales
            st.markdown("---")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            with col_m1:
                st.metric("Signals", len(signals))
            with col_m2:
                inst = len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL])
                st.metric("Institutional", inst)
            with col_m3:
                st.metric("Avg Score", f"{np.mean([s.score for s in signals]):.0f}")
            with col_m4:
                exposure = sum([s.risk_amount for s in signals])
                st.metric("Exposure", f"${exposure:.0f}")
            with col_m5:
                st.metric("Avg R:R", f"{np.mean([s.risk_reward for s in signals]):.1f}:1")
            
            # === 3 COLONNES : H1 | H4 | D1 ===
            st.markdown("---")
            
            # Bouton de téléchargement PDF
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
            with col_dl2:
                # Préparer données pour export
                export_data = []
                for s in signals:
                    export_data.append({
                        "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "Pair": s.pair.replace('_', '/'),
                        "TF": s.timeframe,
                        "Quality": s.quality.value,
                        "Action": s.action,
                        "Entry": f"{s.entry_price:.5f}",
                        "SL": f"{s.stop_loss:.5f}",
                        "TP": f"{s.take_profit:.5f}",
                        "Score": s.score,
                        "R:R": f"{s.risk_reward:.1f}",
                        "Size": f"{s.position_size:.2f}",
                        "Risk": f"${s.risk_amount:.0f}",
                        "ADX": s.adx,
                        "RSI": s.rsi,
                        "Higher TF": s.higher_tf_trend,
                        "Fresh Flip": "Yes" if s.is_fresh_flip else "No",
                        "Live": "Yes" if s.is_live else "No"
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Télécharger Signaux (CSV)",
                    data=csv,
                    file_name=f"bluestar_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            col_h1, col_h4, col_d1 = st.columns(3)
            
            for col, tf in zip([col_h1, col_h4, col_d1], ["H1", "H4", "D1"]):
                with col:
                    tf_signals = [s for s in signals if s.timeframe == tf]
                    
                    # Header
                    st.markdown(f"""
                    <div class='tf-header'>
                        <h3>{tf}</h3>
                        <p>{len(tf_signals)} signal{'s' if len(tf_signals) > 1 else ''}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if tf_signals:
                        # Trier par score DESC puis timestamp DESC
                        tf_signals.sort(key=lambda x: (x.score, x.timestamp), reverse=True)
                        
                        # DataFrame pour affichage
                        df_clean = pd.DataFrame([{
                            "⏰": s.timestamp.strftime("%H:%M"),
                            "Pair": s.pair.replace('_', '/'),
                            "💪": s.quality.value,
                            "Action": f"{'🟢' if s.action == 'BUY' else '🔴'} {s.action}{'⚡' if s.is_live else ''}{'🔥' if s.is_fresh_flip else ''}",
                            "Score": s.score,
                            "Entry": f"{s.entry_price:.5f}",
                            "SL": f"{s.stop_loss:.5f}",
                            "TP": f"{s.take_profit:.5f}",
                            "R:R": f"{s.risk_reward:.1f}",
                            "Size": f"{s.position_size:.2f}",
                            "Risk": f"${s.risk_amount:.0f}",
                            "ADX": s.adx,
                            "RSI": s.rsi
                        } for s in tf_signals])
                        
                        st.dataframe(
                            df_clean,
                            use_container_width=True, 
                            hide_index=True, 
                            height=min(len(df_clean) * 35 + 38, 600)
                        )
                    else:
                        st.info(f"No {tf} signals")
        
        else:
            st.warning("⚠️ No signals detected with current parameters")
            st.info("💡 Astuce: Essayez de désactiver 'Cascade obligatoire' ou réduire le Min R:R pour plus de signaux")
    
    # Footer avec stats
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.7rem; padding: 15px;'>
    <b>BlueStar Cascade Enhanced</b> | HMA + RSI + UT Bot + ADX | Cascade Multi-TF | Kelly Criterion<br>
    <span style='color: #888;'>v2.0 - Logging amélioré, Paramètres configurables, Validation des données</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
              
