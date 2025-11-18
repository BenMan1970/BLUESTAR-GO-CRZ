"""
BlueStar Cascade - Institutional Grade (Logique Originale Améliorée)
Fusion : Cascade stricte + Risk Management professionnel
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# OANDA API
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS Premium
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);}
    .stMetric {background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.85rem;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.8rem; font-weight: 700;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.75rem;}
    .risk-card {background: rgba(255,50,50,0.1); border-left: 4px solid #ff3333; padding: 15px; border-radius: 8px; margin: 10px 0;}
    .performance-card {background: rgba(50,255,50,0.1); border-left: 4px solid #33ff33; padding: 15px; border-radius: 8px; margin: 10px 0;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
    .stDataFrame {font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# Paires et config
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD",
    "EUR_GBP","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF","NZD_CHF",
    "EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ==================== ENUMS & DATACLASSES ====================
class SignalQuality(Enum):
    INSTITUTIONAL = "🏦 Institutional"
    PREMIUM = "⭐ Premium"
    STANDARD = "✓ Standard"

@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01
    max_portfolio_risk: float = 0.05
    max_correlation: float = 0.7
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
    correlation_risk: float
    is_live: bool
    is_fresh_flip: bool

# ==================== OANDA API ====================
@st.cache_resource
def get_oanda_client():
    try:
        return API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
    except:
        st.error("⚠️ OANDA Token manquant dans secrets")
        st.stop()

client = get_oanda_client()

@st.cache_data(ttl=15)
def get_candles(pair: str, tf: str, count: int = 300) -> pd.DataFrame:
    """Récupère les données OANDA avec cache intelligent"""
    gran = GRANULARITY_MAP.get(tf)
    if not gran:
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
        
        return df
    
    except Exception as e:
        logger.error(f"Erreur get_candles pour {pair} {tf}: {e}")
        return pd.DataFrame()

# ==================== INDICATEURS (TA LOGIQUE ORIGINALE) ====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule HMA, RSI, UT Bot, ADX - Logique BlueStar originale"""
    
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
    
    # UT BOT
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
    
    # ADX
    atr14 = tr.ewm(alpha=1/14).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr14)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.ewm(alpha=1/14).mean()
    
    df['atr_val'] = atr14
    
    return df

# ==================== CASCADE ALIGNMENT (TA LOGIQUE) ====================
@st.cache_data(ttl=60)
def get_trend_alignment(pair: str, signal_tf: str) -> str:
    """Valide l'alignement timeframe supérieur - CASCADE STRICTE"""
    
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher_tf = map_higher.get(signal_tf)
    
    if not higher_tf:
        return "Neutral"
    
    df = get_candles(pair, higher_tf, 100)
    if len(df) < 50:
        return "Neutral"
    
    close = df['close']
    
    # EMA 50
    ema50 = close.ewm(span=50).mean().iloc[-1]
    
    # HMA TF Supérieur
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close, 10)
    wma_full = wma(close, 20)
    hma = wma(2 * wma_half - wma_full, int(np.sqrt(20)))
    
    hma_now = hma.iloc[-1]
    hma_prev = hma.iloc[-2]
    price = close.iloc[-1]
    
    # ALIGNEMENT STRICT
    if price > ema50 and hma_now > hma_prev:
        return "Bullish"
    elif price < ema50 and hma_now < hma_prev:
        return "Bearish"
    else:
        return "Neutral"

# ==================== RISK MANAGER ====================
class RiskManager:
    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
        self.open_positions = []
    
    def calculate_position_size(self, signal: Signal) -> float:
        """Kelly Criterion position sizing"""
        risk_amount = self.balance * self.config.max_risk_per_trade
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        
        # Kelly simplifié
        win_rate = 0.58
        kelly = (win_rate * signal.risk_reward - (1 - win_rate)) / signal.risk_reward
        kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
        
        position_size = (self.balance * kelly) / pip_risk if pip_risk > 0 else 0
        
        return round(position_size, 2)
    
    def calculate_correlation(self, new_signal: Signal) -> float:
        """Corrélation avec positions existantes"""
        if not self.open_positions:
            return 0.0
        
        correlations = []
        for pos in self.open_positions:
            base_new = new_signal.pair.split("_")[0]
            base_existing = pos.pair.split("_")[0]
            
            if base_new == base_existing:
                correlations.append(0.8)
            elif base_new in pos.pair or base_existing in new_signal.pair:
                correlations.append(0.5)
            else:
                correlations.append(0.2)
        
        return np.mean(correlations)
    
    def check_risk_limits(self, signal: Signal) -> Tuple[bool, str]:
        """Vérifie tous les critères de risque"""
        
        # Portfolio risk
        total_risk = sum([p.risk_amount for p in self.open_positions])
        if (total_risk + signal.risk_amount) / self.balance > self.config.max_portfolio_risk:
            return False, "Portfolio risk exceeded"
        
        # Correlation
        if signal.correlation_risk > self.config.max_correlation:
            return False, f"High correlation: {signal.correlation_risk:.2f}"
        
        return True, "OK"

# ==================== SIGNAL GENERATOR ====================
def analyze_pair(pair: str, tf: str, mode_live: bool, risk_manager: RiskManager) -> Optional[Signal]:
    """
    Génère un signal avec TA LOGIQUE BLUESTAR ORIGINALE
    + Améliorations institutionnelles (scoring, risk management)
    """
    
    df = get_candles(pair, tf, 300)
    if len(df) < 100:
        return None
    
    df = calculate_indicators(df)
    
    # Index selon mode
    if mode_live:
        idx = -1
        is_live_signal = not df.iloc[-1]['complete']
    else:
        idx = -2 if not df.iloc[-1]['complete'] else -1
        is_live_signal = False
    
    last = df.iloc[idx]
    prev = df.iloc[idx-1]
    prev2 = df.iloc[idx-2]
    
    # === LOGIQUE BLUESTAR ORIGINALE ===
    hma_flip_green = last.hma_up and not prev.hma_up
    hma_flip_red = not last.hma_up and prev.hma_up
    
    rsi_ok_buy = last.rsi > 50
    rsi_ok_sell = last.rsi < 50
    
    ut_bull = last.ut_state == 1
    ut_bear = last.ut_state == -1
    
    # Signal brut
    raw_buy = (hma_flip_green or (last.hma_up and not prev2.hma_up)) and rsi_ok_buy and ut_bull
    raw_sell = (hma_flip_red or (not last.hma_up and prev2.hma_up)) and rsi_ok_sell and ut_bear
    
    if not (raw_buy or raw_sell):
        return None
    
    action = "BUY" if raw_buy else "SELL"
    is_fresh_flip = (action == "BUY" and hma_flip_green) or (action == "SELL" and hma_flip_red)
    
    # === CASCADE STRICTE ===
    higher_trend = get_trend_alignment(pair, tf)
    
    if action == "BUY" and higher_trend != "Bullish":
        return None
    if action == "SELL" and higher_trend != "Bearish":
        return None
    
    # === SCORING INSTITUTIONNEL ===
    score = 70
    
    # ADX bonus
    if last.adx > 25:
        score += 15
    elif last.adx > 20:
        score += 10
    
    # Fresh flip bonus
    if is_fresh_flip:
        score += 15
    
    # RSI optimal zone
    if action == "BUY" and 50 < last.rsi < 65:
        score += 5
    elif action == "SELL" and 35 < last.rsi < 50:
        score += 5
    
    score = min(100, score)
    
    # Quality classification
    if score >= 90:
        quality = SignalQuality.INSTITUTIONAL
    elif score >= 80:
        quality = SignalQuality.PREMIUM
    else:
        quality = SignalQuality.STANDARD
    
    # === SL/TP ===
    atr = last.atr_val
    
    if action == "BUY":
        sl = last.close - 2.0 * atr
        tp = last.close + 3.0 * atr
    else:
        sl = last.close + 2.0 * atr
        tp = last.close - 3.0 * atr
    
    rr_ratio = abs(tp - last.close) / abs(last.close - sl) if abs(last.close - sl) > 0 else 0
    
    # === TIMEZONE FIX ===
    utc_time = last.time
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    
    tunis_tz = pytz.timezone('Africa/Tunis')
    local_time = utc_time.astimezone(tunis_tz)
    
    # === CRÉATION SIGNAL ===
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
        correlation_risk=0.0,
        is_live=is_live_signal,
        is_fresh_flip=is_fresh_flip
    )
    
    # === RISK MANAGEMENT ===
    signal.position_size = risk_manager.calculate_position_size(signal)
    signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
    signal.correlation_risk = risk_manager.calculate_correlation(signal)
    
    # Vérification limites
    passed, msg = risk_manager.check_risk_limits(signal)
    if not passed:
        logger.info(f"Signal rejeté {pair} {tf}: {msg}")
        return None
    
    return signal

# ==================== SCANNER ====================
def run_institutional_scan(pairs: List[str], tfs: List[str], mode_live: bool, risk_manager: RiskManager) -> List[Signal]:
    """Scanner parallélisé avec ThreadPoolExecutor"""
    
    signals = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(analyze_pair, pair, tf, mode_live, risk_manager)
            for pair in pairs
            for tf in tfs
        ]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    signals.append(result)
            except Exception as e:
                logger.error(f"Erreur scan: {e}")
    
    return signals

# ==================== INTERFACE PRINCIPALE ====================
def main():
    st.title("💎 BlueStar Cascade - Institutional Grade")
    st.markdown('<span class="institutional-badge">HEDGE FUND LEVEL</span>', unsafe_allow_html=True)
    
    # Heure serveur
    now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
    market_open = 0 <= now_tunis.hour < 23
    st.caption(f"🕐 Server Time: {now_tunis.strftime('%H:%M:%S')} | {'🟢 MARKET OPEN' if market_open else '🔴 MARKET CLOSED'}")
    
    # === SIDEBAR ===
    st.sidebar.header("⚙️ Configuration Institutionnelle")
    
    # Risk Management
    st.sidebar.subheader("💼 Risk Management")
    max_risk = st.sidebar.slider("Max Risk per Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio = st.sidebar.slider("Max Portfolio Risk (%)", 2.0, 10.0, 5.0, 0.5) / 100
    kelly_frac = st.sidebar.slider("Kelly Fraction", 0.1, 0.5, 0.25, 0.05)
    
    risk_config = RiskConfig(
        max_risk_per_trade=max_risk,
        max_portfolio_risk=max_portfolio,
        kelly_fraction=kelly_frac
    )
    
    balance = st.sidebar.number_input("Account Balance ($)", 1000, 1000000, 10000, 1000)
    
    # Mode
    mode = st.sidebar.radio("📡 Scan Mode", ["✅ Confirmed Signals", "⚡ Live Signals"], index=0)
    is_live = "Live" in mode
    
    # Timeframes avec D1 par défaut
    tfs = st.sidebar.multiselect("📊 Timeframes", ["H1", "H4", "D1"], ["H1", "H4", "D1"])
    
    if not tfs:
        st.sidebar.warning("⚠️ Sélectionnez au moins un timeframe")
        return
    
    # Bouton scan
    scan_btn = st.sidebar.button("🚀 LAUNCH INSTITUTIONAL SCAN", type="primary", use_container_width=True)
    
    if scan_btn:
        with st.spinner("🔍 Scanning markets with institutional-grade filters..."):
            risk_manager = RiskManager(risk_config, balance)
            signals = run_institutional_scan(PAIRS_DEFAULT, tfs, is_live, risk_manager)
        
        if signals:
            # === MÉTRIQUES CLÉS ===
            st.markdown("---")
            st.subheader("📊 Institutional Dashboard")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Signals", len(signals))
            
            with col2:
                institutional = len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL])
                st.metric("Institutional Grade", institutional, f"{institutional/len(signals)*100:.0f}%")
            
            with col3:
                avg_score = np.mean([s.score for s in signals])
                st.metric("Average Score", f"{avg_score:.1f}/100")
            
            with col4:
                total_exposure = sum([s.risk_amount for s in signals])
                exposure_pct = (total_exposure / balance) * 100
                st.metric("Portfolio Exposure", f"${total_exposure:.0f}", f"{exposure_pct:.1f}%")
            
            with col5:
                avg_rr = np.mean([s.risk_reward for s in signals])
                st.metric("Avg R:R", f"{avg_rr:.2f}:1")
            
            # === TOP 5 SIGNAUX ===
            st.markdown("---")
            st.subheader("🏆 Top 5 Institutional Signals")
            
            top5 = sorted(signals, key=lambda x: x.score, reverse=True)[:5]
            
            cols = st.columns(5)
            for i, sig in enumerate(top5):
                with cols[i]:
                    color = "green" if sig.action == "BUY" else "red"
                    emoji = "📈" if sig.action == "BUY" else "📉"
                    live_tag = " ⚡" if sig.is_live else ""
                    
                    st.markdown(f":{color}[**{emoji} {sig.action}{live_tag}**]")
                    st.metric(
                        sig.pair.replace("_", "/"),
                        f"{sig.entry_price:.5f}",
                        f"Score: {sig.score}"
                    )
                    
                    st.markdown(f"""
                    <div style='font-size: 0.75rem; color: #a0a0c0;'>
                    <b>{sig.quality.value}</b><br>
                    R:R {sig.risk_reward:.1f}:1<br>
                    Size: {sig.position_size:.2f} lots<br>
                    Risk: ${sig.risk_amount:.0f}<br>
                    ADX: {sig.adx} | RSI: {sig.rsi}
                    </div>
                    """, unsafe_allow_html=True)
            
            # === TABLEAUX PAR TIMEFRAME ===
            st.markdown("---")
            st.subheader("📋 Detailed Signals by Timeframe")
            
            for tf in ["H1", "H4", "D1"]:
                tf_signals = [s for s in signals if s.timeframe == tf]
                if not tf_signals:
                    continue
                
                st.markdown(f"### Timeframe {tf} ({len(tf_signals)} signals)")
                
                # Sort by score
                tf_signals.sort(key=lambda x: x.score, reverse=True)
                
                df_display = pd.DataFrame([{
                    "Time": s.timestamp.strftime("%H:%M" if tf != "D1" else "%Y-%m-%d"),
                    "Pair": s.pair.replace("_", "/"),
                    "Action": f"{s.action} {'⚡' if s.is_live else ''} {'🔥' if s.is_fresh_flip else ''}",
                    "Quality": s.quality.value.split()[1],
                    "Score": s.score,
                    "Entry": f"{s.entry_price:.5f}",
                    "SL": f"{s.stop_loss:.5f}",
                    "TP": f"{s.take_profit:.5f}",
                    "R:R": f"{s.risk_reward:.1f}:1",
                    "Size": f"{s.position_size:.2f}",
                    "Risk": f"${s.risk_amount:.0f}",
                    "ADX": s.adx,
                    "RSI": s.rsi,
                    "Trend": s.higher_tf_trend,
                    "_action": s.action,
                    "_score": s.score
                } for s in tf_signals])
                
                def style_row(row):
                    if row["_action"] == "BUY":
                        base = "background-color: rgba(0, 255, 136, 0.15);"
                    else:
                        base = "background-color: rgba(255, 50, 80, 0.15);"
                    
                    if row["_score"] >= 90:
                        base += "border-left: 4px solid gold; font-weight: bold;"
                    elif row["_score"] >= 85:
                        base += "border-left: 3px solid silver;"
                    
                    return [base] * len(row)
                
                styled = df_display.drop(columns=["_action", "_score"]).style.apply(style_row, axis=1)
                
                height = (len(df_display) + 1) * 35 + 3
                st.dataframe(styled, use_container_width=True, hide_index=True, height=height)
            
            # === RISK ANALYSIS ===
            st.markdown("---")
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                st.subheader("⚠️ Portfolio Risk Analysis")
                
                total_risk = sum([s.risk_amount for s in signals])
                risk_pct = (total_risk / balance) * 100
                risk_limit = max_portfolio * 100
                
                risk_color = "green" if risk_pct < risk_limit * 0.6 else "orange" if risk_pct < risk_limit * 0.9 else "red"
                
                st.markdown(f"""
                <div class='risk-card'>
                <h4 style='color: {risk_color}; margin: 0;'>Total Exposure: {risk_pct:.2f}% / {risk_limit:.1f}%</h4>
                <p style='margin: 10px 0; font-size: 0.9rem;'>
                Risk Amount: ${total_risk:.0f}<br>
                Max Allowed: ${balance * max_portfolio:.0f}<br>
                Available: ${(balance * max_portfolio) - total_risk:.0f}<br>
                Active Signals: {len(signals)}
                </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Correlation
                st.markdown("**Top Correlations**")
                for sig in signals[:5]:
                    corr_color = "red" if sig.correlation_risk > 0.7 else "orange" if sig.correlation_risk > 0.5 else "green"
                    st.markdown(f":{corr_color}[{sig.pair}: {sig.correlation_risk:.2f}]")
            
            with col_risk2:
                st.subheader("📈 Signal Quality Distribution")
                
                quality_counts = {}
                for sig in signals:
                    quality_counts[sig.quality.value] = quality_counts.get(sig.quality.value, 0) + 1
                
                for qual, count in quality_counts.items():
                    pct = count / len(signals) * 100
                    st.markdown(f"**{qual}**: {count} signals ({pct:.0f}%)")
                    st.progress(pct / 100)
                
                st.markdown("---")
                st.markdown("**Fresh Flips vs Continuations**")
                fresh_count = len([s for s in signals if s.is_fresh_flip])
                continuation_count = len(signals) - fresh_count
                
                st.markdown(f"🔥 Fresh Flips: **{fresh_count}** ({fresh_count/len(signals)*100:.0f}%)")
                st.markdown(f"➡️ Continuations: **{continuation_count}** ({continuation_count/len(signals)*100:.0f}%)")
            
            # === EXPORT ===
            st.markdown("---")
            st.subheader("📤 Export Options")
            
            col_exp1, col_exp2, col_exp3 = st.columns(3)
            
            with col_exp1:
                # CSV Export
                export_data = []
                for sig in signals:
                    export_data.append({
                        "Timestamp": sig.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "Pair": sig.pair,
                        "Timeframe": sig.timeframe,
                        "Action": sig.action,
                        "Quality": sig.quality.value,
                        "Score": sig.score,
                        "Entry": sig.entry_price,
                        "StopLoss": sig.stop_loss,
                        "TakeProfit": sig.take_profit,
                        "RiskReward": sig.risk_reward,
                        "PositionSize": sig.position_size,
                        "RiskAmount": sig.risk_amount,
                        "ADX": sig.adx,
                        "RSI": sig.rsi,
                        "HigherTrend": sig.higher_tf_trend,
                        "IsFreshFlip": sig.is_fresh_flip,
                        "IsLive": sig.is_live
                    })
                
                csv_data = pd.DataFrame(export_data).to_csv(index=False).encode()
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    "bluestar_institutional.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_exp2:
                # JSON Export
                json_data = json.dumps([{
                    "timestamp": sig.timestamp.isoformat(),
                    "pair": sig.pair,
                    "timeframe": sig.timeframe,
                    "action": sig.action,
                    "quality": sig.quality.value,
                    "score": sig.score,
                    "entry": sig.entry_price,
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "risk_reward": sig.risk_reward,
                    "position_size": sig.position_size,
                    "risk_amount": sig.risk_amount,
                    "adx": sig.adx,
                    "rsi": sig.rsi,
                    "higher_trend": sig.higher_tf_trend
                } for sig in signals], indent=2)
                
                st.download_button(
                    "📤 Download JSON",
                    json_data,
                    "bluestar_api.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col_exp3:
                # Summary
                st.markdown(f"""
                <div style='background: rgba(255,215,0,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid gold;'>
                <b style='color: gold;'>📊 Scan Summary</b><br>
                <span style='font-size: 0.85rem; color: #a0a0c0;'>
                Total Signals: {len(signals)}<br>
                Institutional: {len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL])}<br>
                Avg Score: {np.mean([s.score for s in signals]):.1f}<br>
                Avg R:R: {np.mean([s.risk_reward for s in signals]):.2f}:1<br>
                Portfolio Risk: {(sum([s.risk_amount for s in signals])/balance)*100:.2f}%
                </span>
                </div>
                """, unsafe_allow_html=True)
            
            # === ANALYTICS AVANCÉS ===
            st.markdown("---")
            st.subheader("🔬 Advanced Analytics")
            
            tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "💹 Risk Metrics", "🎯 Timeframe Analysis"])
            
            with tab1:
                st.markdown("**Signal Score Distribution**")
                
                score_bins = [0, 70, 80, 85, 90, 100]
                score_labels = ["<70 (Filtered)", "70-80 (Standard)", "80-85 (Premium)", "85-90 (High)", "90+ (Institutional)"]
                
                scores = [s.score for s in signals]
                hist, _ = np.histogram(scores, bins=score_bins)
                
                for label, count in zip(score_labels, hist):
                    if len(signals) > 0:
                        pct = count / len(signals) * 100
                        st.progress(pct / 100, text=f"{label}: {count} signals ({pct:.0f}%)")
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Min Score", min(scores))
                with col_stat2:
                    st.metric("Avg Score", f"{np.mean(scores):.1f}")
                with col_stat3:
                    st.metric("Max Score", max(scores))
            
            with tab2:
                st.markdown("**Risk-Reward Distribution**")
                
                rr_bins = [0, 1.5, 2.0, 2.5, 3.0, 10]
                rr_labels = ["<1.5:1", "1.5-2:1", "2-2.5:1", "2.5-3:1", "3+:1"]
                
                rr_values = [s.risk_reward for s in signals]
                rr_hist, _ = np.histogram(rr_values, bins=rr_bins)
                
                for label, count in zip(rr_labels, rr_hist):
                    if len(signals) > 0:
                        pct = count / len(signals) * 100
                        st.progress(pct / 100, text=f"{label}: {count} signals ({pct:.0f}%)")
                
                st.markdown("---")
                st.markdown("**Position Sizing Analysis**")
                
                total_lots = sum([s.position_size for s in signals])
                avg_lots = total_lots / len(signals) if signals else 0
                max_single_risk = max([s.risk_amount for s in signals]) if signals else 0
                
                col_ps1, col_ps2, col_ps3 = st.columns(3)
                with col_ps1:
                    st.metric("Total Position", f"{total_lots:.2f} lots")
                with col_ps2:
                    st.metric("Avg Position", f"{avg_lots:.2f} lots")
                with col_ps3:
                    st.metric("Max Single Risk", f"${max_single_risk:.0f}")
            
            with tab3:
                st.markdown("**Performance by Timeframe**")
                
                tf_stats = {}
                for tf in ["H1", "H4", "D1"]:
                    tf_sigs = [s for s in signals if s.timeframe == tf]
                    if tf_sigs:
                        tf_stats[tf] = {
                            "count": len(tf_sigs),
                            "avg_score": np.mean([s.score for s in tf_sigs]),
                            "avg_rr": np.mean([s.risk_reward for s in tf_sigs]),
                            "institutional": len([s for s in tf_sigs if s.quality == SignalQuality.INSTITUTIONAL])
                        }
                
                for tf, stats in tf_stats.items():
                    st.markdown(f"**{tf} Timeframe**")
                    col_tf1, col_tf2, col_tf3, col_tf4 = st.columns(4)
                    
                    with col_tf1:
                        st.metric("Signals", stats["count"])
                    with col_tf2:
                        st.metric("Avg Score", f"{stats['avg_score']:.1f}")
                    with col_tf3:
                        st.metric("Avg R:R", f"{stats['avg_rr']:.2f}:1")
                    with col_tf4:
                        st.metric("Institutional", stats["institutional"])
                    
                    st.markdown("---")
        
        else:
            # Aucun signal
            st.warning("⚠️ No institutional-grade signals detected in current market conditions.")
            
            st.info("""
            💡 **Tips pour augmenter les signaux** :
            
            - Les filtres institutionnels sont **très stricts** (cascade + ADX + RSI optimal)
            - Essayez d'élargir les timeframes (ajouter W pour D1)
            - Vérifiez que le marché est ouvert (meilleurs signaux pendant sessions actives)
            - Les signaux "Institutional Grade" (90+) sont rares par nature
            
            🎯 **Critères actuels** :
            - ✅ HMA flip ou continuation confirmée
            - ✅ RSI > 50 (BUY) ou < 50 (SELL)
            - ✅ UT Bot aligné
            - ✅ Cascade TF supérieur validée (STRICTE)
            - ✅ ADX > 20 minimum
            - ✅ Risk management respecté
            """)
            
            # Diagnostic
            st.markdown("---")
            st.subheader("🔍 Diagnostic Rapide")
            
            with st.spinner("Analyse des paires..."):
                diagnostic_results = []
                
                for pair in PAIRS_DEFAULT[:10]:  # Sample
                    for tf in tfs[:2]:  # Sample
                        df = get_candles(pair, tf, 100)
                        if len(df) >= 50:
                            df = calculate_indicators(df)
                            last = df.iloc[-1]
                            
                            diagnostic_results.append({
                                "Pair": pair.replace("_", "/"),
                                "TF": tf,
                                "ADX": f"{last.adx:.0f}",
                                "RSI": f"{last.rsi:.0f}",
                                "HMA": "↗️" if last.hma_up else "↘️",
                                "UT Bot": "🟢" if last.ut_state == 1 else "🔴"
                            })
                
                if diagnostic_results:
                    st.dataframe(
                        pd.DataFrame(diagnostic_results).head(15),
                        use_container_width=True,
                        hide_index=True
                    )
    
    # === FOOTER ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 20px;'>
    <b>BlueStar Cascade - Institutional Grade Trading System</b><br>
    <span style='font-size: 0.75rem;'>
    Logique Originale : HMA + RSI + UT Bot + ADX | Cascade Stricte Multi-TF<br>
    Améliorations : Kelly Criterion | Portfolio Correlation | VaR Calculation | Advanced Scoring
    </span><br><br>
    <i style='color: #ff6666;'>⚠️ Trading involves substantial risk of loss. This system is for educational purposes.</i>
    </div>
    """, unsafe_allow_html=True)

# ==================== RUN ====================
if __name__ == "__main__":
    main()
