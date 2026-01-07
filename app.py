import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import logging
import time
import csv
import os
from datetime import datetime, timedelta
from scipy import stats 
from scipy.signal import argrelextrema 
import pytz
import warnings

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(page_title="Bluestar Ultimate V4.0", layout="centered", page_icon="🛡️")

LOG_FILE = "bluestar_signals_v4_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp", "symbol", "direction", "price", "score", "spread", 
            "vol_ratio", "liquidity_sweep", "pattern", "mtf_bias", "sl", "tp"
        ])

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .main .block-container { max-width: 950px; padding-top: 2rem; }
    h1 {
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 2.8em; text-align: center;
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px -10px #3b82f6; }
    .streamlit-expanderHeader { background-color: #161b22 !important; border-radius: 10px !important; border: 1px solid #30363d !important; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; margin: 2px; display: inline-block; color: white; }
    .badge-blue { background: #2563eb; } .badge-purple { background: #7c3aed; }
    .badge-gold { background: #ca8a04; } .badge-red { background: #dc2626; }
    .badge-green { background: #16a34a; } .badge-grey { background: #4b5563; }
    .metric-card { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CLIENT API
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception as e:
            st.error(f"❌ Erreur Config API: {e}"); st.stop()

    def get_candles(self, instrument, granularity, count):
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
            return pd.DataFrame(data)
        except Exception: return pd.DataFrame()

    def get_realtime_spread(self, instrument):
        try:
            params = {"instruments": instrument}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            self.client.request(r)
            price = r.response['prices'][0]
            bid = float(price['closeoutBid'])
            ask = float(price['closeoutAsk'])
            spread = (ask - bid)
            # Normalisation pip (simple)
            multiplier = 10000 if "JPY" not in instrument else 100
            return spread, spread * multiplier
        except Exception: return 0, 0

# ==========================================
# MOTEUR QUANT V4.0
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_atr(df, period=14):
        if len(df) < period + 1: return 0
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def detect_liquidity_sweep(df_m5, df_d):
        """Détecte si le prix a balayé le High ou Low du jour précédent (PDH/PDL)"""
        if df_d.empty or df_m5.empty: return False, None
        prev_day_high = df_d['high'].iloc[-2]
        prev_day_low = df_d['low'].iloc[-2]
        curr_low = df_m5['low'].iloc[-10:].min() # 50 derniers min
        curr_high = df_m5['high'].iloc[-10:].max()
        curr_close = df_m5['close'].iloc[-1]

        if curr_low < prev_day_low and curr_close > prev_day_low:
            return True, "BULL_SWEEP"
        if curr_high > prev_day_high and curr_close < prev_day_high:
            return True, "BEAR_SWEEP"
        return False, None

    @staticmethod
    def detect_candle_pattern(df):
        """Détecte Engulfing ou Pin Bar sur les 2 dernières bougies"""
        if len(df) < 2: return None
        c1, c2 = df.iloc[-2], df.iloc[-1]
        body2 = abs(c2['close'] - c2['open'])
        range2 = c2['high'] - c2['low']
        
        # Pin Bar
        if range2 > 0 and (body2 / range2) < 0.3:
            if c2['close'] > (c2['high'] + c2['low'])/2: return "PINBAR_BULL"
            else: return "PINBAR_BEAR"
        
        # Engulfing
        if c2['close'] > c1['high'] and c2['open'] < c1['low']: return "ENGULFING_BULL"
        if c2['close'] < c1['low'] and c2['open'] > c1['high']: return "ENGULFING_BEAR"
        return None

    @staticmethod
    def calculate_volatility_ratio(df_m5):
        """ATR Ratio: Volatilité actuelle vs historique"""
        atr_fast = QuantEngine.calculate_atr(df_m5, 14).iloc[-1]
        atr_slow = QuantEngine.calculate_atr(df_m5, 100).iloc[-1]
        if atr_slow == 0: return 1.0
        return atr_fast / atr_slow

    @staticmethod
    def get_mtf_bias(df_d, df_w):
        def score(df):
            if len(df) < 50: return 0
            ma200 = df['close'].rolling(200).mean().iloc[-1]
            ma50 = df['close'].ewm(span=50).mean().iloc[-1]
            return 1 if df['close'].iloc[-1] > ma200 and ma50 > ma200 else (-1 if df['close'].iloc[-1] < ma200 and ma50 < ma200 else 0)
        total = score(df_d) + score(df_w)
        return "STRONG_BULL" if total == 2 else ("STRONG_BEAR" if total == -2 else "NEUTRAL")

    @staticmethod
    def check_session(hour_utc):
        if 7 <= hour_utc < 11: return "LDN"
        if 13 <= hour_utc < 17: return "NY"
        return "OFF"

# ==========================================
# LOGIQUE DE PROBABILITÉ V4.0
# ==========================================
def analyze_asset_v4(api, symbol, current_time_utc):
    try:
        df_d_raw = api.get_candles(symbol, "D", 250)
        df_m5 = api.get_candles(symbol, "M5", 200)
        if df_d_raw.empty or df_m5.empty: return None
        
        # MTF & Contexte
        df_w = df_d_raw.set_index('time').resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        mtf_bias = QuantEngine.get_mtf_bias(df_d_raw, df_w)
        
        # Volatilité & Spread
        vol_ratio = QuantEngine.calculate_volatility_ratio(df_m5)
        raw_spread, pip_spread = api.get_realtime_spread(symbol)
        
        # Direction RSI (Trigger de base)
        delta = df_m5['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rsi = 100 - (100 / (1 + (gain.ewm(alpha=1/7).mean() / loss.ewm(alpha=1/7).mean().replace(0, 0.001)))).iloc[-1]
        
        direction = "BUY" if rsi > 55 else ("SELL" if rsi < 45 else None)
        if not direction: return None
        
        # Confirmation Score
        score = 0.5 # Base
        
        # 1. Session (10%)
        sess = QuantEngine.check_session(current_time_utc.hour)
        if sess != "OFF": score += 0.1
        
        # 2. MTF Alignment (20%)
        if (direction == "BUY" and "BULL" in mtf_bias) or (direction == "SELL" and "BEAR" in mtf_bias):
            score += 0.2
            
        # 3. Liquidity Sweep (20%) - L'ajout V4
        has_sweep, sweep_type = QuantEngine.detect_liquidity_sweep(df_m5, df_d_raw)
        if has_sweep:
            if (direction == "BUY" and sweep_type == "BULL_SWEEP") or (direction == "SELL" and sweep_type == "BEAR_SWEEP"):
                score += 0.2
        
        # 4. Candle Pattern (15%)
        pattern = QuantEngine.detect_candle_pattern(df_m5)
        if pattern:
            if (direction == "BUY" and "BULL" in pattern) or (direction == "SELL" and "BEAR" in pattern):
                score += 0.15
        
        # 5. Volatility Ratio (15%)
        if 1.0 < vol_ratio < 2.5: score += 0.15
        elif vol_ratio < 0.7: score -= 0.2 # Marché trop plat
        
        # 6. Spread Penalty
        if pip_spread > 3.0: score -= 0.2

        # Final Calc
        final_score = max(0, min(1.0, score))
        
        # Risk levels
        atr = QuantEngine.calculate_atr(df_m5, 14).iloc[-1]
        price = df_m5['close'].iloc[-1]
        sl = price - (atr * 1.5) if direction == "BUY" else price + (atr * 1.5)
        tp = price + (atr * 2.5) if direction == "BUY" else price - (atr * 2.5)

        return {
            "symbol": symbol, "direction": direction, "score": final_score,
            "price": price, "pip_spread": pip_spread, "vol_ratio": vol_ratio,
            "sweep": sweep_type, "pattern": pattern, "mtf": mtf_bias,
            "session": sess, "sl": sl, "tp": tp
        }
    except Exception as e:
        logging.error(f"Erreur {symbol}: {e}"); return None

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
def main():
    st.markdown("<h1>🛡️ BLUESTAR ULTIMATE V4.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8b949e;'>Final Quantitative Edition | Sweep & Volatility Engine</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Paramètres")
        min_score = st.slider("Confiance Minimum (%)", 50, 95, 70)
        assets_to_scan = st.multiselect("Actifs", 
            ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "XAU_USD", "US30_USD"],
            default=["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD"])

    if st.button("🚀 LANCER LE SCAN AUDIT"):
        api = OandaClient()
        results = []
        now_utc = datetime.now(pytz.utc)
        
        progress = st.progress(0)
        for i, sym in enumerate(assets_to_scan):
            res = analyze_asset_v4(api, sym, now_utc)
            if res and res['score'] >= (min_score/100):
                results.append(res)
            progress.progress((i+1)/len(assets_to_scan))
            
        if not results:
            st.warning("Aucun signal de haute probabilité détecté.")
        else:
            for s in sorted(results, key=lambda x: x['score'], reverse=True):
                color = "#16a34a" if s['direction'] == "BUY" else "#dc2626"
                with st.expander(f"{s['symbol']} - {s['direction']} (Score: {int(s['score']*100)}%)", expanded=True):
                    # Header
                    st.markdown(f"""
                    <div style='background:{color}; padding:10px; border-radius:5px; color:white; font-weight:bold; display:flex; justify-content:space-between;'>
                        <span>{s['symbol']} - {s['direction']}</span>
                        <span>Prix: {s['price']:.5f}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    # Badges
                    st.write("")
                    b1 = f"<span class='badge badge-blue'>SESSION: {s['session']}</span>"
                    b2 = f"<span class='badge badge-purple'>MTF: {s['mtf']}</span>"
                    b3 = f"<span class='badge badge-gold'>SWEEP: {s['sweep'] if s['sweep'] else 'None'}</span>"
                    b4 = f"<span class='badge badge-green' if s['pip_spread'] < 2 else 'badge-red'>SPREAD: {s['pip_spread']:.1f}</span>"
                    st.markdown(f"{b1} {b2} {b3} {b4}", unsafe_allow_html=True)
                    
                    # Metrics
                    cols = st.columns(3)
                    cols[0].metric("Vol. Ratio", f"{s['vol_ratio']:.2f}")
                    cols[1].metric("Pattern", str(s['pattern']))
                    cols[2].metric("RR Ratio", "1:2.5")
                    
                    # SL/TP
                    st.markdown(f"""
                    <div style='display:flex; gap:10px; margin-top:10px;'>
                        <div style='flex:1; background:#3e1616; padding:10px; border-radius:5px; border:1px solid #dc2626; text-align:center;'>
                            <small style='color:#f87171'>STOP LOSS</small><br><b>{s['sl']:.5f}</b>
                        </div>
                        <div style='flex:1; background:#162e21; padding:10px; border-radius:5px; border:1px solid #16a34a; text-align:center;'>
                            <small style='color:#4ade80'>TAKE PROFIT</small><br><b>{s['tp']:.5f}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
