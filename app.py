import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Ultimate", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.INFO)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    
    * { font-family: 'Roboto', sans-serif; }
    
    .stApp {
        background-color: #0f1117;
        background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%);
    }
    
    .main .block-container { max-width: 950px; padding-top: 2rem; }

    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.8em;
        text-align: center;
        margin-bottom: 0.2em;
    }
    
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 10px;
        color: #f8fafc !important;
        padding: 1.5rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22;
        border: 1px solid #334155;
        border-top: none;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        padding: 20px;
    }
    
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }

    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-vol { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-regime { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    
    .risk-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }
    
    .timestamp-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        color: #93c5fd;
        margin: 10px 0;
    }
    
    .confluence-box {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid #8b5cf6;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .regime-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLIENT API & ASSETS
# ==========================================
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'signal_history' not in st.session_state: st.session_state.signal_history = {}

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except:
            st.error("⚠️ Configuration API manquante")
            st.stop()

    def get_candles(self, instrument, granularity, count):
        key = f"{instrument}_{granularity}"
        if key in st.session_state.cache:
            ts, data = st.session_state.cache[key]
            if (datetime.now() - ts).total_seconds() < 60: return data

        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({
                        'time': pd.to_datetime(c['time']),
                        'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']), 'close': float(c['mid']['c']),
                        'volume': int(c['volume'])
                    })
            df = pd.DataFrame(data)
            if not df.empty:
                st.session_state.cache[key] = (datetime.now(), df)
            return df
        except:
            return pd.DataFrame()

ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD"
]

# ==========================================
# 3. HELPER DYNAMIQUE PAR ACTIF
# ==========================================
def get_asset_params(symbol):
    """Retourne les seuils adaptés à la classe d'actif"""
    if "XAU" in symbol:
        return {'type': 'COMMODITY', 'min_atr': 0.08, 'adx_threshold': 20}
    if "US30" in symbol or "NAS" in symbol or "SPX" in symbol:
        return {'type': 'INDEX', 'min_atr': 0.12, 'adx_threshold': 22}
    
    # Forex par défaut
    return {'type': 'FOREX', 'min_atr': 0.045, 'adx_threshold': 20}

# ==========================================
# 4. MARKET REGIME (COEUR)
# ==========================================
def calculate_adx(df, period=14):
    if len(df) < period + 1: return 0
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up, down = high - high.shift(1), low.shift(1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean().iloc[-1] if len(dx) > 0 else 0

def detect_structure_htf(df):
    if len(df) < 20: return "UNKNOWN", 0
    highs, lows = df['high'].rolling(5).max(), df['low'].rolling(5).min()
    recent_highs, recent_lows = highs.iloc[-15:].values, lows.iloc[-15:].values
    hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
    ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
    total = len(recent_highs) - 1
    if hh_count / total > 0.6: return "BULLISH_STRUCTURE", 80
    elif ll_count / total > 0.6: return "BEARISH_STRUCTURE", 80
    else: return "OSCILLATING", 30

def get_market_regime(api, symbol):
    try:
        df_h4 = api.get_candles(symbol, "H4", 100)
        df_d = api.get_candles(symbol, "D", 50)
        if df_h4.empty or df_d.empty: return "UNKNOWN", {}
        
        # Paramètres dynamiques selon l'actif
        params = get_asset_params(symbol)
        
        atr_h4 = SmartIndicators.calculate_atr(df_h4, 14)
        atr_median = df_h4['close'].rolling(14).apply(lambda x: SmartIndicators.calculate_atr(df_h4.iloc[:len(x)], 14), raw=False).median()
        atr_ratio = atr_h4 / atr_median if atr_median > 0 else 1.0
        
        adx_h4 = calculate_adx(df_h4, 14)
        structure, s_score = detect_structure_htf(df_d)
        
        regime_data = {'atr_ratio': atr_ratio, 'adx_h4': adx_h4, 'structure': structure}
        
        # Logique Regime
        if atr_ratio < 0.75 and adx_h4 < 15: return "RANGE", regime_data
        
        if adx_h4 > params['adx_threshold'] and structure in ["BULLISH_STRUCTURE", "BEARISH_STRUCTURE"]: 
            return "TREND", regime_data
        
        if adx_h4 > 25: return "TREND", regime_data
        
        return "WEAK_TREND", regime_data
    except: return "UNKNOWN", {}

# ==========================================
# 5. GPS & UTILS
# ==========================================
def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def sma_local(series, length): return series.rolling(window=length).mean()

def calc_institutional_trend(df):
    if len(df) < 100: return 'Range'
    close = df['close']
    curr = close.iloc[-1]
    sma200 = sma_local(close, 200).iloc[-1] if len(df) >= 200 else sma_local(close, 50).iloc[-1]
    ema50 = ema(close, 50).iloc[-1]
    if curr > sma200 and ema50 > sma200: return "Bullish"
    if curr < sma200 and ema50 < sma200: return "Bearish"
    return "Range"

def calculate_mtf_gps(api, symbol, direction):
    try:
        df_d = api.get_candles(symbol, "D", 250)
        if df_d.empty: return {'quality': 'N/A', 'analysis': {}}
        d_res = df_d.copy().set_index('time')
        df_w = d_res.resample('W-FRI').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()

        trend_d = calc_institutional_trend(df_d)
        trend_w = calc_institutional_trend(df_w)
        target = 'Bullish' if direction == 'BUY' else 'Bearish'
        
        quality = 'C'
        if trend_d == target and trend_w == target: quality = 'A'
        elif trend_d == target: quality = 'B'
            
        return {'quality': quality, 'analysis': {'D': trend_d, 'W': trend_w}}
    except: return {'quality': 'N/A', 'analysis': {}}

class SmartIndicators:
    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi_ohlc4(df, period=7):
        ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        delta = ohlc4.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.ewm(alpha=1/period, adjust=False).mean() / loss.ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def get_hma_trend(series, length=20):
        wma1 = series.rolling(int(length/2)).mean()
        wma2 = series.rolling(length).mean()
        hma = (2 * wma1 - wma2).rolling(int(np.sqrt(length))).mean()
        if len(hma) < 2: return 0, 0
        return (1 if hma.iloc[-1] > hma.iloc[-2] else -1), hma.iloc[-1]

    @staticmethod
    def detect_fvg(df, atr):
        if len(df) < 5: return False, None
        curr_close = df['close'].iloc[-1]
        for i in range(1, 4):
            high_A, low_A = df['high'].iloc[-(i+2)], df['low'].iloc[-(i+2)]
            high_C, low_C = df['high'].iloc[-i], df['low'].iloc[-i]
            min_gap = atr * 0.3
            if low_C > high_A and (low_C - high_A) > min_gap and curr_close > high_A: return True, "BULL"
            if high_C < low_A and (low_A - high_C) > min_gap and curr_close < low_A: return True, "BEAR"
        return False, None

    @staticmethod
    def detect_obv_pump(df, length=20):
        try:
            change = df['close'].diff()
            vol = df['volume']
            obv_direction = np.where(change > 0, vol, np.where(change < 0, -vol, 0))
            obv = pd.Series(obv_direction).cumsum()
            obv_sma = obv.rolling(window=length).mean()
            return obv.iloc[-1] > obv_sma.iloc[-1], obv.iloc[-1] < obv_sma.iloc[-1]
        except: return False, False

def check_signal_cooldown(symbol, hours=2):
    now = datetime.now()
    if symbol in st.session_state.signal_history:
        last_signal = st.session_state.signal_history[symbol]
        if (now - last_signal).total_seconds() < hours * 3600: return False
    return True

def detect_correlation_conflict(signals, new_signal):
    CORRELATED_PAIRS = {
        'EUR_USD': ['GBP_USD', 'AUD_USD', 'NZD_USD'],
        'GBP_USD': ['EUR_USD', 'AUD_USD', 'NZD_USD'],
        'USD_JPY': ['USD_CHF', 'USD_CAD'],
        'XAU_USD': ['EUR_USD']
    }
    sym = new_signal['symbol']
    typ = new_signal['type']
    if sym not in CORRELATED_PAIRS: return False
    for existing in signals[-5:]:
        if existing['symbol'] in CORRELATED_PAIRS[sym] and existing['type'] != typ: return True
    return False

# ==========================================
# 6. SCANNER ADDITIF (ADAPTATIF & MULTI-MODE)
# ==========================================
def calculate_currency_strength(api):
    pct_changes = {}
    forex_pairs = [p for p in ASSETS if "_" in p and "XAU" not in p and "US30" not in p]
    for sym in forex_pairs:
        df = api.get_candles(sym, "H1", 50)
        if not df.empty and len(df) >= 2:
            pct_changes[sym] = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
    
    if not pct_changes: return None
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    raw_scores = {c: 0.0 for c in currencies}
    for curr in currencies:
        for pair, pct in pct_changes.items():
            if curr in pair:
                base, quote = pair.split('_')
                if base == curr: raw_scores[curr] += pct
                elif quote == curr: raw_scores[curr] -= pct
    return raw_scores

def run_scan(api, min_score, strict_mode):
    cs_scores = calculate_currency_strength(api)
    signals = []
    bar = st.progress(0)
    
    # Session Check (Heure Serveur ou UTC)
    utc_hour = datetime.now(timezone.utc).hour
    session_active = (8 <= utc_hour < 21)
    
    for i, sym in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        if not check_signal_cooldown(sym): continue
        
        try:
            # 1. PARAMÈTRES DYNAMIQUES
            asset_params = get_asset_params(sym)
            
            # 2. REGIME
            regime, regime_data = get_market_regime(api, sym)
            
            # LOGIQUE MODE : Veto vs Malus
            if regime == "RANGE":
                continue # Range est toujours toxique, même en mode Normal
            
            regime_malus = 0
            if regime == "UNKNOWN" or regime == "WEAK_TREND":
                if strict_mode: 
                     # En strict, WEAK est accepté avec moins de points, mais UNKNOWN exclu
                     if regime == "UNKNOWN": continue
                else:
                     # En normal, on tolère un peu plus
                     pass
            
            # 3. DATA & INDICATORS
            df = api.get_candles(sym, "M5", 150)
            if df.empty: continue
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            atr = SmartIndicators.calculate_atr(df)
            price = df['close'].iloc[-1]
            atr_pct = (atr / price) * 100
            
            # FILTRE VOLATILITÉ DYNAMIQUE
            if strict_mode:
                if atr_pct < asset_params['min_atr']: continue
            else:
                # Mode Normal : on tolère une volatilité un peu plus basse (80% du seuil)
                if atr_pct < (asset_params['min_atr'] * 0.8): continue
            
            # 4. TRIGGER M5
            rsi_series = SmartIndicators.calculate_rsi_ohlc4(df)
            rsi = rsi_series.iloc[-1]
            trend_hma, _ = SmartIndicators.get_hma_trend(df['close'])
            rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else rsi
            
            signal_type = None
            rsi_cross_up = (rsi_prev <= 50 and rsi >= 50)
            rsi_cross_down = (rsi_prev >= 50 and rsi <= 50)
            
            if trend_hma == 1 and rsi_cross_up: signal_type = "BUY"
            elif trend_hma == -1 and rsi_cross_down: signal_type = "SELL"
            
            if not signal_type: continue
            
            # 5. SCORING ADDITIF (LE MOTEUR)
            current_score = 0.0
            
            # A. Regime Points
            if regime == "TREND": current_score += 2.0
            elif regime == "WEAK_TREND": current_score += 1.0
            
            # B. GPS Points
            mtf = calculate_mtf_gps(api, sym, signal_type)
            if strict_mode:
                if mtf['quality'] == 'C': continue # VETO Strict
            else:
                if mtf['quality'] == 'C': current_score -= 2.0 # MALUS Normal
            
            if mtf['quality'] == 'A': current_score += 3.0
            elif mtf['quality'] == 'B': current_score += 2.0
            
            # C. Currency Strength / Asset Strength
            cs_aligned = False
            base, quote = sym.split('_') if "_" in sym else (None, None)
            
            if asset_params['type'] == 'FOREX' and cs_scores and base and quote:
                gap = cs_scores.get(base, 0) - cs_scores.get(quote, 0)
                if signal_type == "BUY":
                    if gap > 0.5: 
                        current_score += 1.5; cs_aligned = True
                    elif gap < -0.5:
                        if strict_mode: continue # VETO CS
                        else: current_score -= 2.0 # Malus CS
                else: # SELL
                    if gap < -0.5:
                        current_score += 1.5; cs_aligned = True
                    elif gap > 0.5:
                        if strict_mode: continue
                        else: current_score -= 2.0
                        
            else:
                # COMPENSATION POUR GOLD & INDICES
                # Ils n'ont pas de CS, donc on leur donne un "Bonus Structurel" 
                # si le reste est propre pour compenser les 1.5 pts manquants
                if regime == "TREND" and mtf['quality'] in ['A', 'B']:
                    current_score += 1.5 # Bonus compensatoire d'actif unique
            
            # D. Contexte
            if session_active: current_score += 1.0
            elif strict_mode and "JPY" not in sym: current_score -= 1.0
            
            # E. Technique
            fvg, fvg_type = SmartIndicators.detect_fvg(df, atr)
            obv_pump, obv_dump = SmartIndicators.detect_obv_pump(df)
            
            current_score += 1.0 # RSI Freshness bonus de base
            if fvg and ((signal_type == "BUY" and fvg_type == "BULL") or (signal_type == "SELL" and fvg_type == "BEAR")):
                current_score += 0.8
                
            obv_ok = (signal_type == "BUY" and obv_pump) or (signal_type == "SELL" and obv_dump)
            if obv_ok: current_score += 0.7
            
            # F. Volatilité (Points)
            if atr_pct >= asset_params['min_atr']: current_score += 1.0
            
            # 6. FINAL CHECK
            if current_score < min_score: continue
            if detect_correlation_conflict(signals, {'symbol': sym, 'type': signal_type}): continue
            
            # 7. RISK
            sl_mult = 1.5 if mtf['quality'] == 'A' else 2.0
            # Gold/Indices demandent souvent un peu plus d'air
            if asset_params['type'] != 'FOREX': sl_mult += 0.5
                
            sl = price - (atr * sl_mult) if signal_type == "BUY" else price + (atr * sl_mult)
            tp = price + (atr * 3.0) if signal_type == "BUY" else price - (atr * 3.0)
            
            signals.append({
                'symbol': sym, 'type': signal_type, 'price': price,
                'score': current_score, 'regime': regime,
                'mtf': mtf, 'atr_pct': atr_pct, 'time': df['time'].iloc[-1],
                'sl': sl, 'tp': tp, 'rr': 3.0/sl_mult,
                'details': {'cs_aligned': cs_aligned, 'session': session_active, 'fvg': fvg, 'vol': obv_ok}
            })
            st.session_state.signal_history[sym] = datetime.now()
            
        except Exception: continue
            
    bar.empty()
    return sorted(signals, key=lambda x: x['score'], reverse=True)

# ==========================================
# 7. AFFICHAGE
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score']
    label = "⭐ EXCELLENT" if sc >= 8.0 else "✅ BON"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"<div class='timestamp-box'>📅 Signal: {s['time'].strftime('%d/%m %H:%M UTC')}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.3f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        badges = []
        badges.append(f"<span class='badge-regime'>{s['regime']}</span>")
        badges.append(f"<span class='badge-gps'>GPS {s['mtf']['quality']}</span>")
        if s['details']['cs_aligned']: badges.append("<span class='badge-vol' style='background:#3b82f6'>CS ALIGNÉ</span>")
        if s['details']['session']: badges.append("<span class='badge-vol' style='background:#f59e0b'>SESSION</span>")
        if s['details']['fvg']: badges.append("<span class='badge-fvg'>FVG</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Score Additif", f"{sc:.1f}/10")
        c2.metric("GPS Bias", f"D:{s['mtf']['analysis']['D'][0]} W:{s['mtf']['analysis']['W'][0]}")
        c3.metric("Risk:Reward", f"1:{s['rr']:.1f}")

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em;'>STOP LOSS</div><div style='color:#ef4444;font-weight:bold;'>{s['sl']:.5f}</div></div>", unsafe_allow_html=True)
        r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8;font-size:0.8em;'>TAKE PROFIT</div><div style='color:#10b981;font-weight:bold;'>{s['tp']:.5f}</div></div>", unsafe_allow_html=True)

def main():
    st.title("💎 BLUESTAR ULTIMATE")
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.9em;'>Multi-Asset | Dual Mode | Additive Scoring</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Paramètres")
        # LE MODE STRICT EST ACTIVÉ PAR DÉFAUT
        strict_mode = st.checkbox("🔥 Mode Strict (Veto)", value=True)
        
        min_score = st.slider("Score Min", 5.0, 10.0, 7.0, 0.5)
        
        st.markdown("---")
        st.markdown("### 📊 Sensibilité Actifs")
        st.markdown("""
        <div style='font-size:0.8em;color:#94a3b8;'>
        <b>Forex:</b> ATR > 0.045%<br>
        <b>Gold:</b> ATR > 0.08%<br>
        <b>Indices:</b> ATR > 0.12%<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if strict_mode:
            st.warning("🛡️ MODE STRICT: Les signaux imparfaits sont bloqués.")
        else:
            st.info("🌊 MODE NORMAL: Tolérance élargie (Malus au lieu de Veto).")
    
    if st.button("🚀 LANCER LE SCAN", type="primary"):
        st.session_state.cache = {}
        api = OandaClient()
        with st.spinner(f"Scan en cours (Mode {'Strict' if strict_mode else 'Normal'})..."):
            results = run_scan(api, min_score, strict_mode)
        
        if not results:
            st.warning("⚠️ Aucun signal qualifié.")
        else:
            st.success(f"✅ {len(results)} Signaux qualifiés")
            for sig in results:
                display_sig(sig)

if __name__ == "__main__":
    main()
   
