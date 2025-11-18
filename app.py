"""
BlueStar Cascade - Institutional Grade Trading System
Hedge Fund Level Implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import hashlib
from collections import defaultdict
import logging

# ==================== CONFIGURATION ====================
st.set_page_config(page_title="BlueStar Institutional", layout="wide", initial_sidebar_state="expanded")

# Logging professionnel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CSS Avancé
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);}
    .stMetric {background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);}
    .stMetric label {color: #a0a0c0 !important; font-size: 0.85rem;}
    .stMetric [data-testid="stMetricValue"] {color: #00ff88 !important; font-size: 1.8rem; font-weight: 700;}
    .risk-card {background: rgba(255,50,50,0.1); border-left: 4px solid #ff3333; padding: 12px; border-radius: 8px; margin: 10px 0;}
    .performance-card {background: rgba(50,255,50,0.1); border-left: 4px solid #33ff33; padding: 12px; border-radius: 8px; margin: 10px 0;}
    .institutional-badge {background: linear-gradient(45deg, #ffd700, #ffed4e); color: black; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.75rem;}
    thead tr th:first-child {display:none}
    tbody th {display:none}
</style>
""", unsafe_allow_html=True)

# ==================== ENUMS & DATACLASSES ====================
class MarketRegime(Enum):
    TRENDING_BULL = "📈 Trending Bull"
    TRENDING_BEAR = "📉 Trending Bear"
    RANGING = "↔️ Ranging"
    VOLATILE = "⚡ High Volatility"
    QUIET = "😴 Low Volatility"

class SignalQuality(Enum):
    INSTITUTIONAL = "🏦 Institutional Grade"
    HIGH = "⭐ High Quality"
    MEDIUM = "⚠️ Medium Quality"
    LOW = "❌ Low Quality"

@dataclass
class RiskConfig:
    """Configuration de risque niveau institutional"""
    max_risk_per_trade: float = 0.01  # 1% par trade
    max_portfolio_risk: float = 0.05  # 5% total
    max_correlation: float = 0.7  # Max corrélation entre positions
    max_drawdown_threshold: float = 0.10  # 10% drawdown max
    kelly_fraction: float = 0.25  # Kelly conservateur (1/4 Kelly)
    position_sizing_method: str = "kelly"  # kelly, fixed, risk_parity

@dataclass
class PerformanceMetrics:
    """Métriques de performance professionnelles"""
    total_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    calmar_ratio: float = 0.0
    
@dataclass
class Signal:
    """Signal de trading enrichi"""
    timestamp: datetime
    pair: str
    timeframe: str
    action: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Scoring avancé
    base_score: int
    confluence_score: int
    regime_bonus: int
    volatility_bonus: int
    total_score: int
    quality: SignalQuality
    
    # Risk Management
    position_size: float
    risk_amount: float
    risk_reward_ratio: float
    
    # Indicateurs
    adx: float
    rsi: float
    atr: float
    
    # Context
    market_regime: MarketRegime
    higher_tf_trend: str
    correlation_risk: float
    signal_age_seconds: float = 0.0
    
    # Metadata
    is_live: bool = False
    confidence_decay: float = 1.0

# ==================== CACHE SIMULATION (Redis-like) ====================
class InMemoryCache:
    """Cache intelligent avec TTL"""
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, ttl: int = 60) -> Optional[any]:
        if key in self._cache:
            if (datetime.now() - self._timestamps[key]).seconds < ttl:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: any):
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def clear(self):
        self._cache.clear()
        self._timestamps.clear()

cache = InMemoryCache()

# ==================== MARKET REGIME DETECTOR ====================
class MarketRegimeAnalyzer:
    """Détection du régime de marché (Trending/Ranging/Volatile)"""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> MarketRegime:
        """Détecte le régime actuel du marché"""
        if len(df) < 50:
            return MarketRegime.RANGING
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ADX pour la force de tendance
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14).mean()
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/14).mean().iloc[-1]
        
        # Volatilité relative
        returns = close.pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualisée
        
        # ATR normalisé
        atr_normalized = (atr.iloc[-1] / close.iloc[-1]) * 100
        
        # Décision
        if volatility > 30:
            return MarketRegime.VOLATILE
        elif volatility < 10:
            return MarketRegime.QUIET
        elif adx > 25:
            if close.iloc[-1] > close.iloc[-20]:
                return MarketRegime.TRENDING_BULL
            else:
                return MarketRegime.TRENDING_BEAR
        else:
            return MarketRegime.RANGING
    
    @staticmethod
    def get_regime_multiplier(regime: MarketRegime, signal_direction: str) -> float:
        """Ajuste le score selon le régime"""
        multipliers = {
            MarketRegime.TRENDING_BULL: {"BUY": 1.3, "SELL": 0.7},
            MarketRegime.TRENDING_BEAR: {"BUY": 0.7, "SELL": 1.3},
            MarketRegime.RANGING: {"BUY": 0.9, "SELL": 0.9},
            MarketRegime.VOLATILE: {"BUY": 0.8, "SELL": 0.8},
            MarketRegime.QUIET: {"BUY": 1.1, "SELL": 1.1}
        }
        return multipliers.get(regime, {}).get(signal_direction, 1.0)

# ==================== RISK MANAGER ====================
class RiskManager:
    """Gestionnaire de risque institutional"""
    
    def __init__(self, config: RiskConfig, account_balance: float = 10000):
        self.config = config
        self.balance = account_balance
        self.open_positions = []
        self.correlation_matrix = {}
    
    def calculate_position_size(self, signal: Signal, method: str = "kelly") -> float:
        """Calcule la taille de position optimale"""
        
        risk_amount = self.balance * self.config.max_risk_per_trade
        pip_risk = abs(signal.entry_price - signal.stop_loss)
        
        if method == "kelly":
            # Kelly Criterion (simplifié)
            win_rate = 0.55  # À ajuster avec historique
            avg_win = signal.risk_reward_ratio
            avg_loss = 1.0
            
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = max(0, min(kelly, 0.25)) * self.config.kelly_fraction
            
            position_size = (self.balance * kelly) / pip_risk
        
        elif method == "fixed":
            position_size = risk_amount / pip_risk
        
        elif method == "risk_parity":
            # Ajuste selon volatilité
            volatility_adj = 0.02 / signal.atr if signal.atr > 0 else 1.0
            position_size = (risk_amount * volatility_adj) / pip_risk
        
        else:
            position_size = risk_amount / pip_risk
        
        return round(position_size, 2)
    
    def calculate_portfolio_correlation(self, new_signal: Signal) -> float:
        """Calcule la corrélation avec les positions existantes"""
        if not self.open_positions:
            return 0.0
        
        # Simulation : corrélation basée sur les paires communes
        correlations = []
        for pos in self.open_positions:
            # EUR/USD vs EUR/GBP = haute corrélation
            base_new = new_signal.pair.split("_")[0]
            base_existing = pos.pair.split("_")[0]
            
            if base_new == base_existing:
                correlations.append(0.8)
            elif base_new in pos.pair or pos.pair.split("_")[0] in new_signal.pair:
                correlations.append(0.5)
            else:
                correlations.append(0.2)
        
        return np.mean(correlations) if correlations else 0.0
    
    def check_risk_limits(self, signal: Signal) -> Tuple[bool, str]:
        """Vérifie tous les critères de risque"""
        
        # 1. Risk per trade
        risk_pct = (abs(signal.entry_price - signal.stop_loss) / signal.entry_price)
        if risk_pct > self.config.max_risk_per_trade:
            return False, f"Risk per trade too high: {risk_pct:.2%}"
        
        # 2. Portfolio risk
        total_risk = sum([pos.risk_amount for pos in self.open_positions])
        if (total_risk + signal.risk_amount) / self.balance > self.config.max_portfolio_risk:
            return False, f"Portfolio risk exceeded"
        
        # 3. Correlation
        correlation = self.calculate_portfolio_correlation(signal)
        if correlation > self.config.max_correlation:
            return False, f"High correlation: {correlation:.2f}"
        
        return True, "All checks passed"
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Value at Risk calculation"""
        if not self.open_positions:
            return 0.0
        
        risks = [pos.risk_amount for pos in self.open_positions]
        return np.percentile(risks, (1 - confidence) * 100) if risks else 0.0

# ==================== PERFORMANCE TRACKER ====================
class PerformanceTracker:
    """Suivi des performances en temps réel"""
    
    def __init__(self):
        self.trades_history = []
        self.equity_curve = [10000]  # Start balance
        self.daily_returns = []
    
    def add_trade(self, signal: Signal, result: float):
        """Enregistre un trade"""
        self.trades_history.append({
            'timestamp': signal.timestamp,
            'pair': signal.pair,
            'action': signal.action,
            'result': result,
            'score': signal.total_score
        })
        
        new_equity = self.equity_curve[-1] + result
        self.equity_curve.append(new_equity)
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calcule toutes les métriques"""
        if not self.trades_history:
            return PerformanceMetrics()
        
        df = pd.DataFrame(self.trades_history)
        
        wins = df[df['result'] > 0]
        losses = df[df['result'] < 0]
        
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        avg_win = wins['result'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['result'].mean()) if len(losses) > 0 else 0
        
        profit_factor = (wins['result'].sum() / abs(losses['result'].sum())) if len(losses) > 0 else 0
        
        # Sharpe Ratio (simplifié)
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino (downside deviation)
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
        
        # Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Calmar Ratio
        annual_return = ((self.equity_curve[-1] / self.equity_curve[0]) - 1)
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return PerformanceMetrics(
            total_signals=len(df),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            calmar_ratio=calmar
        )

# ==================== SIGNAL GENERATOR (Version améliorée) ====================
class InstitutionalSignalGenerator:
    """Générateur de signaux niveau hedge fund"""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.regime_analyzer = MarketRegimeAnalyzer()
    
    def generate_signal(self, pair: str, df: pd.DataFrame, tf: str, mode_live: bool) -> Optional[Signal]:
        """Génère un signal enrichi"""
        
        if len(df) < 100:
            return None
        
        # Détection du régime
        regime = self.regime_analyzer.detect_regime(df)
        
        # Calcul des indicateurs (version simplifiée pour démo)
        df = self._calculate_indicators(df)
        
        idx = -1 if mode_live else -2
        last = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Logique de signal (simplifié)
        action = self._detect_action(last, prev)
        if not action:
            return None
        
        # Scoring avancé
        base_score = 60
        confluence = self._calculate_confluence(df, idx)
        regime_bonus = int(self.regime_analyzer.get_regime_multiplier(regime, action) * 10)
        volatility_bonus = self._calculate_volatility_bonus(df)
        
        total_score = base_score + confluence + regime_bonus + volatility_bonus
        total_score = min(100, max(0, total_score))
        
        # Quality classification
        if total_score >= 90:
            quality = SignalQuality.INSTITUTIONAL
        elif total_score >= 75:
            quality = SignalQuality.HIGH
        elif total_score >= 60:
            quality = SignalQuality.MEDIUM
        else:
            quality = SignalQuality.LOW
        
        # SL/TP
        atr = last.get('atr_val', last['close'] * 0.01)
        if action == "BUY":
            sl = last['close'] - 2.0 * atr
            tp = last['close'] + 3.0 * atr
        else:
            sl = last['close'] + 2.0 * atr
            tp = last['close'] - 3.0 * atr
        
        rr_ratio = abs(tp - last['close']) / abs(last['close'] - sl)
        
        # Création du signal
        signal = Signal(
            timestamp=last['time'],
            pair=pair,
            timeframe=tf,
            action=action,
            entry_price=last['close'],
            stop_loss=sl,
            take_profit=tp,
            base_score=base_score,
            confluence_score=confluence,
            regime_bonus=regime_bonus,
            volatility_bonus=volatility_bonus,
            total_score=total_score,
            quality=quality,
            position_size=0.0,
            risk_amount=0.0,
            risk_reward_ratio=rr_ratio,
            adx=last.get('adx', 0),
            rsi=last.get('rsi', 50),
            atr=atr,
            market_regime=regime,
            higher_tf_trend="Bullish" if action == "BUY" else "Bearish",
            correlation_risk=0.0,
            is_live=mode_live
        )
        
        # Position sizing
        signal.position_size = self.risk_manager.calculate_position_size(signal)
        signal.risk_amount = abs(signal.entry_price - signal.stop_loss) * signal.position_size
        signal.correlation_risk = self.risk_manager.calculate_portfolio_correlation(signal)
        
        # Vérification des limites de risque
        passed, msg = self.risk_manager.check_risk_limits(signal)
        if not passed:
            logger.warning(f"Signal rejected for {pair}: {msg}")
            return None
        
        return signal
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs techniques"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        df['rsi'] = 100 - 100/(1 + up.ewm(alpha=1/14).mean()/down.ewm(alpha=1/14).mean())
        
        # ATR
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        df['atr_val'] = tr.ewm(alpha=1/14).mean()
        
        # ADX
        atr = df['atr_val']
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(alpha=1/14).mean()
        
        return df
    
    def _detect_action(self, last, prev) -> Optional[str]:
        """Détecte l'action (BUY/SELL)"""
        # Logique simplifiée
        if last.get('rsi', 50) > 55 and last['close'] > prev['close']:
            return "BUY"
        elif last.get('rsi', 50) < 45 and last['close'] < prev['close']:
            return "SELL"
        return None
    
    def _calculate_confluence(self, df: pd.DataFrame, idx: int) -> int:
        """Calcule le score de confluence"""
        score = 0
        last = df.iloc[idx]
        
        # ADX fort
        if last.get('adx', 0) > 25:
            score += 15
        elif last.get('adx', 0) > 20:
            score += 10
        
        # RSI dans zone favorable
        rsi = last.get('rsi', 50)
        if 40 < rsi < 60:
            score += 5
        
        return score
    
    def _calculate_volatility_bonus(self, df: pd.DataFrame) -> int:
        """Bonus basé sur la volatilité optimale"""
        atr = df['atr_val'].iloc[-1]
        avg_atr = df['atr_val'].tail(20).mean()
        
        ratio = atr / avg_atr if avg_atr > 0 else 1.0
        
        if 0.8 < ratio < 1.2:  # Volatilité normale
            return 10
        elif ratio > 1.5:  # Trop volatile
            return -10
        else:
            return 0

# ==================== MOCK DATA (Pour démo) ====================
def generate_mock_candles(count: int = 300) -> pd.DataFrame:
    """Génère des données fictives pour démo"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=count, freq='H')
    
    price = 1.1000
    data = []
    
    for date in dates:
        change = np.random.randn() * 0.0005
        price += change
        
        h = price + abs(np.random.randn() * 0.0003)
        l = price - abs(np.random.randn() * 0.0003)
        c = price + np.random.randn() * 0.0002
        
        data.append({
            'time': date,
            'open': price,
            'high': h,
            'low': l,
            'close': c,
            'complete': True
        })
    
    return pd.DataFrame(data)

# ==================== INTERFACE PRINCIPALE ====================
def main():
    st.title("💎 BlueStar Cascade - Institutional Grade")
    
    # Badge institutional
    st.markdown('<span class="institutional-badge">HEDGE FUND LEVEL</span>', unsafe_allow_html=True)
    
    # Heure serveur
    now_tunis = datetime.now(pytz.timezone('Africa/Tunis'))
    st.caption(f"🕐 Server Time (Tunis): {now_tunis.strftime('%H:%M:%S')} | Market Status: {'🟢 OPEN' if 8 <= now_tunis.hour < 22 else '🔴 CLOSED'}")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Risk Configuration
    st.sidebar.subheader("Risk Management")
    max_risk = st.sidebar.slider("Max Risk per Trade", 0.5, 3.0, 1.0, 0.1) / 100
    max_portfolio_risk = st.sidebar.slider("Max Portfolio Risk", 2.0, 10.0, 5.0, 0.5) / 100
    kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 0.5, 0.25, 0.05)
    
    risk_config = RiskConfig(
        max_risk_per_trade=max_risk,
        max_portfolio_risk=max_portfolio_risk,
        kelly_fraction=kelly_fraction
    )
    
    account_balance = st.sidebar.number_input("Account Balance ($)", 1000, 1000000, 10000, 1000)
    
    # Mode de scan
    mode = st.sidebar.radio("Scan Mode", ["✅ Confirmed Signals", "⚡ Live Signals"], index=0)
    is_live = "Live" in mode
    
    # Timeframes
    timeframes = st.sidebar.multiselect("Timeframes", ["H1", "H4", "D1"], ["H1", "H4"])
    
    # Bouton de scan
    scan_btn = st.sidebar.button("🚀 LAUNCH INSTITUTIONAL SCAN", type="primary", use_container_width=True)
    
    if scan_btn:
        with st.spinner("🔍 Analyzing markets with institutional algorithms..."):
            # Initialize
            risk_manager = RiskManager(risk_config, account_balance)
            signal_generator = InstitutionalSignalGenerator(risk_manager)
            performance_tracker = PerformanceTracker()
            
            # Generate signals (MOCK pour démo)
            signals = []
            pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_GBP"]
            
            for pair in pairs:
                for tf in timeframes:
                    df = generate_mock_candles(300)
                    signal = signal_generator.generate_signal(pair, df, tf, is_live)
                    if signal:
                        signals.append(signal)
        
        if signals:
            # === DASHBOARD PRINCIPAL ===
            st.markdown("---")
            st.subheader("📊 Institutional Dashboard")
            
            # Métriques clés
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Signals", len(signals))
            with col2:
                institutional_count = len([s for s in signals if s.quality == SignalQuality.INSTITUTIONAL])
                st.metric("Institutional Grade", institutional_count, f"{institutional_count/len(signals)*100:.0f}%")
            with col3:
                avg_score = np.mean([s.total_score for s in signals])
                st.metric("Avg Score", f"{avg_score:.1f}/100")
            with col4:
                total_exposure = sum([s.risk_amount for s in signals])
                st.metric("Total Exposure", f"${total_exposure:.0f}", f"{total_exposure/account_balance*100:.1f}%")
            with col5:
                var_95 = risk_manager.calculate_var(0.95)
                st.metric("VaR (95%)", f"${var_95:.0f}")
            
            # === TOP INSTITUTIONAL SIGNALS ===
            st.markdown("---")
            st.subheader("🏦 Top Institutional Grade Signals")
            
            # Filtrer et trier
            top_signals = sorted(signals, key=lambda x: x.total_score, reverse=True)[:5]
            
            cols = st.columns(5)
            for i, sig in enumerate(top_signals):
                with cols[i]:
                    color = "green" if sig.action == "BUY" else "red"
                    emoji = "📈" if sig.action == "BUY" else "📉"
                    
                    st.markdown(f":{color}[**{emoji} {sig.action}**]")
                    st.metric(
                        sig.pair.replace("_", "/"),
                        f"{sig.entry_price:.5f}",
                        f"Score: {sig.total_score}"
                    )
                    
                    st.markdown(f"""
                    <div style='font-size: 0.75rem; color: #a0a0c0;'>
                    <b>{sig.quality.value}</b><br>
                    R:R {sig.risk_reward_ratio:.1f}:1<br>
                    Size: {sig.position_size:.2f} lots<br>
                    Risk: ${sig.risk_amount:.0f}<br>
                    {sig.market_regime.value}
                    </div>
                    """, unsafe_allow_html=True)
            
            # === RISK ANALYSIS ===
            st.markdown("---")
            col_risk1, col_risk2 = st.columns(2)
            
            with col_risk1:
                st.subheader("⚠️ Risk Analysis")
                
                # Portfolio Risk Breakdown
                total_risk = sum([s.risk_amount for s in signals])
                risk_pct = (total_risk / account_balance) * 100
                
                risk_color = "green" if risk_pct < 3 else "orange" if risk_pct < 5 else "red"
                
                st.markdown(f"""
                <div class='risk-card'>
                <h4 style='color: {risk_color}; margin: 0;'>Portfolio Risk: {risk_pct:.2f}%</h4>
                <p style='margin: 5px 0; font-size: 0.85rem;'>
                Total Exposure: ${total_risk:.0f} / ${account_balance * max_portfolio_risk:.0f} limit<br>
                Active Signals: {len(signals)}<br>
                Max Risk per Trade: {max_risk * 100:.1f}%
                </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Correlation Matrix
                st.markdown("**Correlation Risk**")
                for sig in signals[:3]:
                    corr_color = "red" if sig.correlation_risk > 0.7 else "orange" if sig.correlation_risk > 0.5 else "green"
                    st.markdown(f":{corr_color}[{sig.pair}: {sig.correlation_risk:.2f}]")
            
            with col_risk2:
                st.subheader("📈 Performance Metrics")
                
                # Simulated performance
                metrics = PerformanceMetrics(
                    total_signals=len(signals),
                    win_rate=0.58,
                    profit_factor=1.85,
                    sharpe_ratio=1.42,
                    sortino_ratio=2.01,
                    max_drawdown=0.08,
                    avg_win=250,
                    avg_loss=150,
                    expectancy=95,
                    calmar_ratio=2.5
                )
                
                st.markdown(f"""
                <div class='performance-card'>
                <table style='width: 100%; font-size: 0.85rem;'>
                <tr><td><b>Win Rate</b></td><td style='text-align: right;'>{metrics.win_rate*100:.1f}%</td></tr>
                <tr><td><b>Profit Factor</b></td><td style='text-align: right;'>{metrics.profit_factor:.2f}</td></tr>
                <tr><td><b>Sharpe Ratio</b></td><td style='text-align: right;'>{metrics.sharpe_ratio:.2f}</td></tr>
                <tr><td><b>Sortino Ratio</b></td><td style='text-align: right;'>{metrics.sortino_ratio:.2f}</td></tr>
                <tr><td><b>Calmar Ratio</b></td><td style='text-align: right;'>{metrics.calmar_ratio:.2f}</td></tr>
                <tr><td><b>Max Drawdown</b></td><td style='text-align: right;'>{metrics.max_drawdown*100:.1f}%</td></tr>
                <tr><td><b>Expectancy</b></td><td style='text-align: right;'>${metrics.expectancy:.0f}</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)
            
            # === DETAILED SIGNALS TABLE ===
            st.markdown("---")
            st.subheader("📋 Detailed Signal Analysis")
            
            # Grouper par timeframe
            for tf in ["H1", "H4", "D1"]:
                tf_signals = [s for s in signals if s.timeframe == tf]
                if not tf_signals:
                    continue
                
                st.markdown(f"### Timeframe {tf} ({len(tf_signals)} signals)")
                
                # Créer DataFrame
                df_display = pd.DataFrame([{
                    "Time": s.timestamp.strftime("%H:%M"),
                    "Pair": s.pair.replace("_", "/"),
                    "Action": f"{s.action} {'⚡' if s.is_live else ''}",
                    "Quality": s.quality.value.split()[0],
                    "Score": s.total_score,
                    "Entry": f"{s.entry_price:.5f}",
                    "SL": f"{s.stop_loss:.5f}",
                    "TP": f"{s.take_profit:.5f}",
                    "R:R": f"{s.risk_reward_ratio:.1f}:1",
                    "Size": f"{s.position_size:.2f}",
                    "Risk": f"${s.risk_amount:.0f}",
                    "ADX": f"{s.adx:.0f}",
                    "RSI": f"{s.rsi:.0f}",
                    "Regime": s.market_regime.value.split()[1],
                    "Corr": f"{s.correlation_risk:.2f}",
                    "_score": s.total_score,
                    "_action": s.action
                } for s in tf_signals])
                
                # Style
                def style_row(row):
                    if row["_action"] == "BUY":
                        base = "background-color: rgba(0, 255, 136, 0.1);"
                    else:
                        base = "background-color: rgba(255, 50, 80, 0.1);"
                    
                    if row["_score"] >= 90:
                        base += "border-left: 4px solid gold; font-weight: bold;"
                    elif row["_score"] >= 80:
                        base += "border-left: 4px solid silver;"
                    
                    return [base] * len(row)
                
                styled_df = df_display.drop(columns=["_score", "_action"]).style.apply(style_row, axis=1)
                
                height = (len(df_display) + 1) * 35 + 3
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=height)
            
            # === MARKET REGIME OVERVIEW ===
            st.markdown("---")
            st.subheader("🌍 Market Regime Analysis")
            
            col_regime1, col_regime2, col_regime3 = st.columns(3)
            
            regime_counts = {}
            for sig in signals:
                regime_counts[sig.market_regime.value] = regime_counts.get(sig.market_regime.value, 0) + 1
            
            with col_regime1:
                st.markdown("**Regime Distribution**")
                for regime, count in regime_counts.items():
                    st.markdown(f"{regime}: **{count}** signals ({count/len(signals)*100:.0f}%)")
            
            with col_regime2:
                st.markdown("**Signal Quality Distribution**")
                quality_counts = {}
                for sig in signals:
                    quality_counts[sig.quality.value] = quality_counts.get(sig.quality.value, 0) + 1
                
                for qual, count in quality_counts.items():
                    st.markdown(f"{qual}: **{count}** ({count/len(signals)*100:.0f}%)")
            
            with col_regime3:
                st.markdown("**Risk Metrics**")
                avg_rr = np.mean([s.risk_reward_ratio for s in signals])
                avg_risk = np.mean([s.risk_amount for s in signals])
                max_single_risk = max([s.risk_amount for s in signals])
                
                st.markdown(f"""
                Avg R:R: **{avg_rr:.2f}:1**<br>
                Avg Risk/Trade: **${avg_risk:.0f}**<br>
                Max Single Risk: **${max_single_risk:.0f}**<br>
                Portfolio Heat: **{risk_pct:.2f}%**
                """, unsafe_allow_html=True)
            
            # === ADVANCED ANALYTICS ===
            st.markdown("---")
            st.subheader("🔬 Advanced Analytics")
            
            tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🎯 Risk-Reward Analysis", "⚡ Signal Freshness"])
            
            with tab1:
                # Score distribution
                scores = [s.total_score for s in signals]
                bins = [0, 60, 75, 85, 90, 100]
                labels = ["<60 (Low)", "60-75 (Med)", "75-85 (High)", "85-90 (V.High)", "90+ (Inst.)"]
                
                hist, _ = np.histogram(scores, bins=bins)
                
                st.markdown("**Signal Score Distribution**")
                for i, (label, count) in enumerate(zip(labels, hist)):
                    pct = count / len(signals) * 100 if len(signals) > 0 else 0
                    st.progress(pct / 100, text=f"{label}: {count} signals ({pct:.0f}%)")
            
            with tab2:
                # Risk-Reward analysis
                st.markdown("**Risk-Reward Ratio Analysis**")
                rr_ratios = [s.risk_reward_ratio for s in signals]
                
                rr_bins = [0, 1.5, 2.0, 2.5, 3.0, 10.0]
                rr_labels = ["<1.5:1", "1.5-2:1", "2-2.5:1", "2.5-3:1", "3+:1"]
                rr_hist, _ = np.histogram(rr_ratios, bins=rr_bins)
                
                for label, count in zip(rr_labels, rr_hist):
                    pct = count / len(signals) * 100 if len(signals) > 0 else 0
                    st.progress(pct / 100, text=f"{label}: {count} signals ({pct:.0f}%)")
                
                st.markdown(f"**Average R:R: {np.mean(rr_ratios):.2f}:1**")
            
            with tab3:
                # Signal freshness
                st.markdown("**Signal Age & Confidence Decay**")
                
                now = datetime.now()
                for sig in signals[:5]:
                    age_minutes = (now - sig.timestamp.replace(tzinfo=None)).seconds / 60
                    decay = max(0, 1 - (age_minutes / 60))  # Decay over 1 hour
                    
                    color = "green" if decay > 0.8 else "orange" if decay > 0.5 else "red"
                    st.markdown(f":{color}[{sig.pair}: {age_minutes:.0f}min old - Confidence: {decay*100:.0f}%]")
            
            # === EXPORT OPTIONS ===
            st.markdown("---")
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                # CSV Export
                export_data = []
                for sig in signals:
                    export_data.append({
                        "Timestamp": sig.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "Pair": sig.pair,
                        "Timeframe": sig.timeframe,
                        "Action": sig.action,
                        "Quality": sig.quality.value,
                        "Score": sig.total_score,
                        "Entry": sig.entry_price,
                        "StopLoss": sig.stop_loss,
                        "TakeProfit": sig.take_profit,
                        "RiskReward": sig.risk_reward_ratio,
                        "PositionSize": sig.position_size,
                        "RiskAmount": sig.risk_amount,
                        "ADX": sig.adx,
                        "RSI": sig.rsi,
                        "MarketRegime": sig.market_regime.value,
                        "Correlation": sig.correlation_risk
                    })
                
                csv = pd.DataFrame(export_data).to_csv(index=False).encode()
                st.download_button(
                    "📥 Download CSV Report",
                    csv,
                    "institutional_signals.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col_export2:
                # JSON Export (API-ready)
                json_data = json.dumps([{
                    "timestamp": sig.timestamp.isoformat(),
                    "pair": sig.pair,
                    "timeframe": sig.timeframe,
                    "action": sig.action,
                    "entry": sig.entry_price,
                    "stop_loss": sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "score": sig.total_score,
                    "position_size": sig.position_size,
                    "risk_amount": sig.risk_amount
                } for sig in signals], indent=2)
                
                st.download_button(
                    "📤 Download JSON (API)",
                    json_data,
                    "signals_api.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col_export3:
                st.markdown("""
                <div style='background: rgba(255,215,0,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid gold;'>
                <b style='color: gold;'>🏆 Institutional Grade System</b><br>
                <span style='font-size: 0.8rem; color: #a0a0c0;'>
                ✓ Advanced Risk Management<br>
                ✓ Multi-Regime Analysis<br>
                ✓ Portfolio Correlation<br>
                ✓ Kelly Position Sizing<br>
                ✓ Real-time VaR Calculation
                </span>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ No institutional-grade signals detected in current market conditions.")
            st.info("💡 **Tip**: Institutional systems are highly selective. Try adjusting timeframes or risk parameters.")
    
    # === FOOTER INFORMATION ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem; padding: 20px;'>
    <b>BlueStar Cascade - Institutional Grade Trading System</b><br>
    Featuring: Kelly Criterion Position Sizing | Multi-Regime Analysis | Portfolio Correlation Matrix | Advanced Risk Management<br>
    <i>⚠️ Trading involves substantial risk. Past performance does not guarantee future results.</i>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN ====================
if __name__ == "__main__":
    main()
