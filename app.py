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
    .risk-box { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
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
            st.error(f"❌ Configuration API manquante ou invalide: {e}"); st.stop()

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
            bid = float(price['closeoutBid']); ask = float(price['closeoutAsk'])
            spread = (ask - bid)
            multiplier = 10000 if ("JPY" not in instrument and "X" not in instrument and "_" in instrument) else 100
            if "XAU" in instrument or "XAG" in instrument or "XPT" in instrument: multiplier = 100
            return spread, spread * multiplier
        except Exception: return 0, 0

# ==========================================
# ASSETS & PARAMS
# ==========================================
ASSETS = [
    # 28 Forex Pairs
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    # Commodities
    "XAU_USD", "XPT_USD", "XAG_USD",
    # Indices
    "US30_USD", "NAS100_USD", "SPX500_USD", "DE30_EUR"
]

def get_asset_params(symbol):
    if any(idx in symbol for idx in ["US30", "NAS100", "SPX500", "DE30"]):
        return {'type': 'INDEX', 'atr_threshold': 0.10, 'sl_base': 2.0, 'tp_rr': 3.0}
    if any(met in symbol for met in ["XAU", "XPT", "XAG"]):
        return {'type': 'COMMODITY', 'atr_threshold': 0.06, 'sl_base': 1.8, 'tp_rr': 2.5}
    return {'type': 'FOREX', 'atr_threshold': 0.035, 'sl_base': 1.5, 'tp_rr': 2.0}

# ==========================================
# MOTEUR QUANT V4.0
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_atr(df, period=14):
        if len(df) < period + 1: return pd.Series([0]*len(df))
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def detect_liquidity_sweep(df_m5, df_d):
        if df_d.empty or df_m5.empty: return False, None
        pdh = df_d['high'].iloc[-2]; pdl = df_d['low'].iloc[-2]
        curr_l = df_m5['low'].iloc[-10:].min(); curr_h = df_m5['high'].iloc[-10:].max()
        curr_c = df_m5['close'].iloc[-1]
        if curr_l < pdl and curr_c > pdl: return True, "BULL_SWEEP"
        if curr_h > pdh and curr_c < pdh: return True, "BEAR_SWEEP"
        return False, None

    @staticmethod
    def detect_candle_pattern(df):
        if len(df) < 2: return None
        c1, c2 = df.iloc[-2], df.iloc[-1]
        body2 = abs(c2['close'] - c2['open']); range2 = c2['high'] - c2['low']
        if range2 > 0 and (body2 / range2) < 0.3:
            return "PINBAR_BULL" if c2['close'] > (c2['high']+c2['low'])/2 else "PINBAR_BEAR"
        if c2['close'] > c1['high'] and c2['open'] < c1['low']: return "ENGULFING_BULL"
        if c2['close'] < c1['low'] and c2['open'] > c1['high']: return "ENGULFING_BEAR"
        return None

    @staticmethod
    def calculate_volatility_ratio(df_m5):
        atr_f = QuantEngine.calculate_atr(df_m5, 14).iloc[-1]
        atr_s = QuantEngine.calculate_atr(df_m5, 100).iloc[-1]
        return atr_f / atr_s if atr_s > 0 else 1.0

    @staticmethod
    def get_mtf_bias(df_d, df_w):
        def score(df):
            if len(df) < 50: return 0
            m200 = df['close'].rolling(200).mean().iloc[-1]
            m50 = df['close'].ewm(span=50).mean().iloc[-1]
            c = df['close'].iloc[-1]
            return 1 if c > m200 and m50 > m200 else (-1 if c < m200 and m50 < m200 else 0)
        t = score(df_d) + score(df_w)
        return "STRONG_BULL" if t >= 2 else ("STRONG_BEAR" if t <= -2 else "NEUTRAL")

    @staticmethod
    def check_session(hour_utc):
        if 7 <= hour_utc < 11: return "LDN"
        if 13 <= hour_utc < 17: return "NY"
        return "OFF"

# ==========================================
# ANALYSE & SCAN
# ==========================================
def analyze_asset_v4(api, symbol, current_time_utc):
    try:
        df_d = api.get_candles(symbol, "D", 250); df_m5 = api.get_candles(symbol, "M5", 200)
        if df_d.empty or df_m5.empty: return None
        
        df_w = df_d.set_index('time').resample('W-FRI').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        mtf = QuantEngine.get_mtf_bias(df_d, df_w)
        vol_ratio = QuantEngine.calculate_volatility_ratio(df_m5)
        raw_spread, pip_spread = api.get_realtime_spread(symbol)
        
        # Signal RSI
        d = df_m5['close'].diff()
        rsi = 100 - (100 / (1 + (d.where(d>0,0).ewm(alpha=1/7).mean() / d.where(d<0,0).abs().ewm(alpha=1/7).mean().replace(0,0.001)))).iloc[-1]
        direction = "BUY" if rsi > 55 else ("SELL" if rsi < 45 else None)
        if not direction: return None
        
        score = 0.4
        sess = QuantEngine.check_session(current_time_utc.hour)
        if sess != "OFF": score += 0.15
        if (direction == "BUY" and "BULL" in mtf) or (direction == "SELL" and "BEAR" in mtf): score += 0.20
        has_sweep, sweep_t = QuantEngine.detect_liquidity_sweep(df_m5, df_d)
        if has_sweep and direction[:4] == sweep_t[:4]: score += 0.20
        pattern = QuantEngine.detect_candle_pattern(df_m5)
        if pattern and direction[:4] == pattern.split('_')[1][:4]: score += 0.15
        if 1.0 < vol_ratio < 2.5: score += 0.10
        if pip_spread > 4.0: score -= 0.25

        final_s = max(0, min(1.0, score))
        atr = QuantEngine.calculate_atr(df_m5, 14).iloc[-1]; price = df_m5['close'].iloc[-1]
        p = get_asset_params(symbol)
        sl = price - (atr * p['sl_base']) if direction == "BUY" else price + (atr * p['sl_base'])
        tp = price + (atr * p['tp_rr']) if direction == "BUY" else price - (atr * p['tp_rr'])

        return {
            "symbol": symbol, "direction": direction, "score": final_s, "price": price,
            "pip_spread": pip_spread, "vol_ratio": vol_ratio, "sweep": sweep_t,
            "pattern": pattern, "mtf": mtf, "session": sess, "sl": sl, "tp": tp, "rr": p['tp_rr']
        }
    except Exception: return None

def main():
    st.markdown("<h1>🛡️ BLUESTAR ULTIMATE V4.0</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#8b949e;'>Full Market Scanner | Forex, Metals & Indices</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")
        min_conf = st.slider("Confiance Min (%)", 50, 95, 70)
        st.info(f"Scan de {len(ASSETS)} actifs en cours...")

    if st.button("🔍 LANCER LE SCAN"):
        api = OandaClient(); results = []; now_utc = datetime.now(pytz.utc)
        progress = st.progress(0); status = st.empty()
        
        for i, sym in enumerate(ASSETS):
            status.text(f"Analyse de {sym}...")
            res = analyze_asset_v4(api, sym, now_utc)
            if res and res['score'] >= (min_conf/100): results.append(res)
            progress.progress((i+1)/len(ASSETS))
        
        status.empty(); progress.empty()
        if not results: st.warning("Aucune opportunité détectée.")
        else:
            st.success(f"{len(results)} signaux détectés")
            for s in sorted(results, key=lambda x: x['score'], reverse=True):
                color = "#16a34a" if s['direction'] == "BUY" else "#dc2626"
                with st.expander(f"{s['symbol']} | {s['direction']} | {int(s['score']*100)}%", expanded=True):
                    st.markdown(f"<div style='background:{color};padding:12px;border-radius:8px;color:white;font-weight:900;display:flex;justify-content:space-between;'>"
                                f"<span>{s['symbol']} - {s['direction']}</span><span>{s['price']:.5f}</span></div>", unsafe_allow_html=True)
                    st.write("")
                    cols = st.columns(4)
                    cols[0].markdown(f"<span class='badge badge-blue'>SESS: {s['session']}</span>", unsafe_allow_html=True)
                    cols[1].markdown(f"<span class='badge badge-purple'>{s['mtf']}</span>", unsafe_allow_html=True)
                    cols[2].markdown(f"<span class='badge badge-gold'>SWEEP: {s['sweep']}</span>", unsafe_allow_html=True)
                    cols[3].markdown(f"<span class='badge badge-grey'>SPR: {s['pip_spread']:.1f}</span>", unsafe_allow_html=True)
                    
                    st.write("---")
                    r1, r2 = st.columns(2)
                    r1.markdown(f"<div class='risk-box'><small style='color:#ef4444'>STOP LOSS</small><br><b>{s['sl']:.5f}</b></div>", unsafe_allow_html=True)
                    r2.markdown(f"<div class='risk-box'><small style='color:#10b981'>TAKE PROFIT (1:{s['rr']})</small><br><b>{s['tp']:.5f}</b></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
