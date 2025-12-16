import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import requests
from datetime import datetime, timezone
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Hybrid Pro", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.WARNING)

# ==========================================
# CSS FINTECH
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0f1117; background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%); }
    .main .block-container { max-width: 950px; padding-top: 2rem; }
    h1 { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.8em; text-align: center; }
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); color: white; border: 1px solid rgba(255,255,255,0.1); }
    .streamlit-expanderHeader { background-color: #1e293b !important; border: 1px solid #334155; border-radius: 10px; color: #f8fafc !important; }
    .streamlit-expanderContent { background-color: #161b22; border: 1px solid #334155; border-top: none; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }
    .info-box { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    .risk-box { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-sr { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTES
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "XPT_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

FOREX_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]

if 'cache' not in st.session_state:
    st.session_state.cache = {}
    st.session_state.cache_time = {}
    st.session_state.currency_strength_cache = None
    st.session_state.currency_strength_time = 0

CACHE_DURATION = 30
CURRENCY_STRENGTH_CACHE_DURATION = 300

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
            self.last_request_time = time.time()
        except:
            st.error("⚠️ Erreur Secrets API")
            st.stop()

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < 0.1: time.sleep(0.1 - elapsed)
        self.last_request_time = time.time()

    def get_candles(self, instrument: str, granularity: str, count: int = 150) -> pd.DataFrame:
        cache_key = f"{instrument}_{granularity}"
        if cache_key in st.session_state.cache:
            if time.time() - st.session_state.cache_time.get(cache_key, 0) < CACHE_DURATION:
                return st.session_state.cache[cache_key].copy()
        
        self._rate_limit()
        params = {"count": count, "granularity": granularity, "price": "M"}
        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({
                        'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']),
                        'low': float(c['mid']['l']), 'close': float(c['mid']['c']), 'volume': int(c['volume'])
                    })
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            st.session_state.cache[cache_key] = df.copy()
            st.session_state.cache_time[cache_key] = time.time()
            return df
        except: return pd.DataFrame()

# ==========================================
# INDICATEURS
# ==========================================
def calculate_wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_sma(series, length): return series.rolling(window=length).mean()

def calculate_zlema(series, length):
    lag = int((length - 1) / 2)
    return (series + (series - series.shift(lag))).ewm(span=length, adjust=False).mean()

def calculate_atr(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(df, period=14):
    tr = calculate_atr(df, 1).ewm(span=period, adjust=False).mean()
    up, down = df['high'] - df['high'].shift(1), df['low'].shift(1) - df['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / tr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean()

def get_rsi_ohlc4(df, length=7):
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    delta = ohlc4.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def get_colored_hma(df, length=20):
    src = df['close']
    wma1 = calculate_wma(src, int(length/2))
    wma2 = calculate_wma(src, length)
    hma = calculate_wma(2 * wma1 - wma2, int(np.sqrt(length)))
    trend = pd.Series(np.where(hma > hma.shift(1), 1, -1), index=df.index)
    return hma, trend

def get_nearest_sr(df, price):
    if len(df) < 20: return {'sup': None, 'res': None, 'dist_sup': 999, 'dist_res': 999}
    h, l = df['high'], df['low']
    is_res = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
    is_sup = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))
    res = df[is_res]['high'].values
    sup = df[is_sup]['low'].values
    
    nr = res[res > price].min() if any(res > price) else None
    ns = sup[sup < price].max() if any(sup < price) else None
    dr = ((nr - price)/price*100) if nr else 999
    ds = ((price - ns)/price*100) if ns else 999
    return {'sup': ns, 'res': nr, 'dist_sup': ds, 'dist_res': dr}

def detect_fvg(df):
    if len(df) < 5: return False, False
    return (df['low'] > df['high'].shift(2)).iloc[-5:].any(), (df['high'] < df['low'].shift(2)).iloc[-5:].any()

def get_pips(pair, diff):
    if any(x in pair for x in ["XAU", "US30", "NAS100", "SPX500", "XPT"]): return abs(diff)
    return abs(diff * (100 if "JPY" in pair else 10000))

# ==========================================
# MARKET MAP PRO ENGINE (DIRECT)
# ==========================================
def calculate_currency_strength(api: OandaClient, lookback_days: int = 1) -> Dict[str, float]:
    cache_age = time.time() - st.session_state.currency_strength_time
    if st.session_state.currency_strength_cache and cache_age < CURRENCY_STRENGTH_CACHE_DURATION:
        if sum(abs(x) for x in st.session_state.currency_strength_cache.values()) > 0.001:
            return st.session_state.currency_strength_cache

    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
        env = st.secrets.get("OANDA_ENVIRONMENT", "practice")
        base_url = "https://api-fxtrade.oanda.com" if env == "live" else "https://api-fxpractice.oanda.com"
    except: return {}

    pairs = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD', 'NZD_USD',
        'EUR_GBP', 'EUR_JPY', 'EUR_CHF', 'EUR_AUD', 'EUR_CAD', 'EUR_NZD',
        'GBP_JPY', 'GBP_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',
        'AUD_JPY', 'AUD_CAD', 'AUD_NZD', 'AUD_CHF',
        'CAD_JPY', 'CAD_CHF', 'NZD_JPY', 'NZD_CHF', 'CHF_JPY'
    ]
    
    forex_data = {}
    headers = {"Authorization": f"Bearer {token}"}
    
    for inst in pairs:
        try:
            url = f"{base_url}/v3/instruments/{inst}/candles?count={lookback_days+5}&granularity=D"
            resp = requests.get(url, headers=headers, timeout=2)
            if resp.status_code == 200:
                c = resp.json().get('candles', [])
                if c and len(c) > lookback_days:
                    now = float(c[-1]['mid']['c'])
                    past = float(c[-(lookback_days+1)]['mid']['c'])
                    forex_data[inst] = (now - past) / past * 100
        except: continue

    data = {}
    for s, p in forex_data.items():
        b, q = s.split('_')
        if b not in data: data[b] = []
        if q not in data: data[q] = []
        data[b].append({'pct': p, 'other': q})
        data[q].append({'pct': -p, 'other': b})
    
    scores = {}
    for c, items in data.items():
        sc = 0
        w_sum = 0
        for i in items:
            w = 2.0 if i['other'] in ['USD', 'EUR', 'JPY'] else 1.0
            sc += (i['pct'] * w)
            w_sum += w
        scores[c] = sc / w_sum if w_sum > 0 else 0
        
    if sum(abs(v) for v in scores.values()) > 0.001:
        st.session_state.currency_strength_cache = scores
        st.session_state.currency_strength_time = time.time()
    return scores

def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    """Calcul du score de force (Logique Market Map Pro)"""
    if symbol in FOREX_PAIRS:
        parts = symbol.split('_')
        base, quote = parts[0], parts[1]
        scores = calculate_currency_strength(api)
        
        if base not in scores: 
            return {'score': 0, 'details': 'N/A', 'base_score': 0, 'label': 'Neutre'}
        
        b_score = scores[base]
        q_score = scores[quote]
        
        # Ranking pour le contexte
        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {k: i+1 for i, (k, v) in enumerate(sorted_s)}
        b_rank, q_rank = ranks.get(base, 8), ranks.get(quote, 8)
        
        # Calcul du score (Reste identique pour la sélection)
        score = 0
        label = "Neutre"
        
        if direction == 'BUY':
            if b_rank <= 3 and q_rank >= 6:
                score = 2
                label = "Excellent" # Texte neutre
            elif b_score > q_score:
                score = 1
                label = "Validé"
            else:
                label = "Divergence"
        else: # SELL
            if q_rank <= 3 and b_rank >= 6:
                score = 2
                label = "Excellent"
            elif q_score > b_score:
                score = 1
                label = "Validé"
            else:
                label = "Divergence"
                
        # On renvoie juste les faits, pas d'instruction de trading
        rank_info = f"{base}(#{b_rank}) vs {quote}(#{q_rank})"
        return {'score': score, 'details': label, 'base_score': b_score, 'quote_score': q_score, 'rank_info': rank_info}

    else:
        # Indices/Or
        try:
            df = api.get_candles(symbol, "D", count=2)
            if df.empty: return {'score': 0, 'details': 'N/A', 'base_score': 0, 'label': 'Neutre'}
            
            chg = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100
            score = 0
            label = "Neutre"
            
            if direction == 'BUY':
                if chg > 0.3: score = 2; label = "Impulsion"
                elif chg > 0: score = 1; label = "Positif"
                else: label = "Contre-tendance"
            else:
                if chg < -0.3: score = 2; label = "Chute"
                elif chg < 0: score = 1; label = "Négatif"
                else: label = "Contre-tendance"
                
            return {'score': score, 'details': label, 'base_score': chg, 'rank_info': f"D1: {chg:+.2f}%"}
        except:
            return {'score': 0, 'details': 'Err', 'base_score': 0, 'label': 'Err'}

# ==========================================
# MTF & SCANNER
# ==========================================
def calculate_mtf_score_gps(api, symbol, direction):
    map_tf = {'D1': 'D', 'H4': 'H4', 'H1': 'H1'}
    expected = 'Bullish' if direction == 'BUY' else 'Bearish'
    matches = 0
    w_tot = 0
    weights = {'D1': 2, 'H4': 1, 'H1': 0.5}
    analysis = {}
    
    for name, code in map_tf.items():
        df = api.get_candles(symbol, code, count=250)
        trend = 'Neutral'
        if not df.empty:
            c = df['close']
            p = c.iloc[-1]
            if name in ['D1', 'H4']:
                ma = calculate_sma(c, 200).iloc[-1] if len(c)>200 else calculate_sma(c, 50).iloc[-1]
                trend = 'Bullish' if p > ma else 'Bearish'
            else:
                z = calculate_zlema(c, 50).iloc[-1]
                trend = 'Bullish' if p > z else 'Bearish'
            
            if trend == expected: matches += weights[name]
        w_tot += weights[name]
        analysis[name] = trend
        
    pct = (matches/w_tot)*100
    score = 3 if pct >= 85 else 2 if pct >= 50 else 1 if pct >= 25 else 0
    qual = 'A' if pct >= 85 else 'B' if pct >= 50 else 'C'
    return {'score': score, 'quality': qual, 'alignment': f"{pct:.0f}%", 'analysis': analysis}

def calculate_risk(price, atr, direction, pair, sl_m, tp_m):
    sl_dist = atr * sl_m
    tp_dist = atr * tp_m
    sl = price - sl_dist if direction == 'BUY' else price + sl_dist
    tp = price + tp_dist if direction == 'BUY' else price - tp_dist
    return {'sl': sl, 'tp': tp, 'sl_pips': get_pips(pair, sl_dist), 'tp_pips': get_pips(pair, tp_dist), 'rr': tp_m/sl_m}

def run_scan(api, min_score, risk, sl_m, tp_m):
    sigs = []
    
    # CALCUL SILENCIEUX DE LA FORCE (PAS DE TEXTE)
    calculate_currency_strength(api)
    
    pbar = st.progress(0)
    for i, sym in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        try:
            df = api.get_candles(sym, "M15", count=150)
            if df.empty or len(df) < 50: continue
            
            p = df['close'].iloc[-1]
            atr = calculate_atr(df).iloc[-1]
            rsi = get_rsi_ohlc4(df).iloc[-1]
            hma, trend = get_colored_hma(df)
            hma_val = trend.iloc[-1]
            fvg_b, fvg_s = detect_fvg(df)
            
            # Logic
            typ = None
            if rsi > 50 and hma_val == 1: typ = 'BUY'
            elif rsi < 50 and hma_val == -1: typ = 'SELL'
            
            if typ:
                sc = 0
                sc += 3 # Base
                
                mtf = calculate_mtf_score_gps(api, sym, typ)
                cs = calculate_currency_strength_score(api, sym, typ)
                sc += mtf['score'] + cs['score']
                if (typ == 'BUY' and fvg_b) or (typ == 'SELL' and fvg_s): sc += 1
                
                sr = get_nearest_sr(api.get_candles(sym, "D", 200), p)
                warn = ""
                badge = ""
                
                if typ == 'BUY':
                    if sr['dist_res'] < 0.25: sc -= 2; warn = "Résistance proche"
                    elif sr['dist_sup'] < 0.4: badge = "Rebond Support"
                else:
                    if sr['dist_sup'] < 0.25: sc -= 2; warn = "Support proche"
                    elif sr['dist_res'] < 0.4: badge = "Rejet Résistance"
                
                if sc >= min_score:
                    rm = calculate_risk(p, atr, typ, sym, sl_m, tp_m) if risk else None
                    sigs.append({
                        'symbol': sym, 'type': typ, 'price': p, 'score': sc,
                        'quality': mtf['quality'], 'atr': atr, 'warn': warn, 'badge': badge,
                        'rsi': rsi, 'mtf': mtf, 'cs': cs, 'fvg': (fvg_b if typ=='BUY' else fvg_s),
                        'rm': rm, 'time': df['time'].iloc[-1]
                    })
        except: continue
    pbar.empty()
    return sigs

def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    ago = int((datetime.now(timezone.utc) - s['time'].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()/60)
    
    with st.expander(f"{s['symbol']} | {s['type']} | Score {s['score']}/10 [{s['quality']}]", expanded=True):
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col};display:flex;justify-content:space-between;align-items:center;">
            <div><span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
            <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span></div>
            <div style="text-align:right;"><div style="color:#cbd5e1;">{ago} min ago</div><div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div></div>
        </div>""", unsafe_allow_html=True)
        
        badges = []
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        if s['quality'] == 'A': badges.append("<span class='badge-gps'>🛡️ GPS A+</span>")
        if s['badge']: badges.append(f"<span class='badge-sr'>{s['badge']}</span>")
        if badges: st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        if s['warn']: st.warning(s['warn'])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score Total", s['score'])
        c2.metric("Qualité GPS", s['quality'])
        
        # --- MODIFICATION FORCE/MOMENTUM ---
        # Affichage du Score "2/2" (la Jauge) avec le texte neutre
        score_val = s['cs']['score']
        score_text = f"{score_val}/2"
        delta_label = s['cs']['details'] # Contient "Excellent", "Validé" ou "Divergence" (pas de Buy/Sell)
        
        # Coloration dynamique du delta
        delta_color = "normal" if score_val > 0 else "off"
        
        c3.metric("Momentum (Force)", score_text, delta=delta_label, delta_color=delta_color)
        c4.metric("ATR", f"{s['atr']:.4f}")
        
        if s['rm']:
            st.divider()
            r1, r2, r3 = st.columns(3)
            sl_pip = int(s['rm']['sl_pips']) if "XAU" not in s['symbol'] else f"{s['rm']['sl_pips']:.1f}"
            tp_pip = int(s['rm']['tp_pips']) if "XAU" not in s['symbol'] else f"{s['rm']['tp_pips']:.1f}"
            r1.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>SL</div><div style='color:#ef4444;font-weight:bold'>{s['rm']['sl']:.5f}</div><small>-{sl_pip} pips</small></div>", unsafe_allow_html=True)
            r2.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>TP</div><div style='color:#10b981;font-weight:bold'>{s['rm']['tp']:.5f}</div><small>+{tp_pip} pips</small></div>", unsafe_allow_html=True)
            r3.markdown(f"<div class='risk-box'><div style='color:#94a3b8'>R:R</div><div style='color:white;font-weight:bold'>1:{s['rm']['rr']:.2f}</div></div>", unsafe_allow_html=True)
            
        st.divider()
        k1, k2 = st.columns(2)
        k1.info(f"**Technique**: RSI {s['rsi']:.1f} | Alignement {s['mtf']['alignment']}")
        k2.info(f"**Fondamental**: {s['cs']['rank_info']}")

# ==========================================
# MAIN
# ==========================================
st.title("💎 Bluestar SNP3 Hybrid Pro")
with st.expander("⚙️ Paramètres"):
    c1, c2 = st.columns(2)
    risk = c1.checkbox("Risk Manager", True)
    sl = c2.slider("SL xATR", 1.0, 3.0, 1.5)
    tp = c2.slider("TP xATR", 1.5, 5.0, 2.0)

co1, co2 = st.columns([3,1])
min_sc = co1.slider("Score Min", 4, 10, 6)
if co2.button("🧹 Reset"):
    st.session_state.cache = {}
    st.session_state.currency_strength_cache = None
    st.toast("Cache vidé")

if st.button("🚀 LANCER", type="primary"):
    api = OandaClient()
    res = run_scan(api, min_sc, risk, sl, tp)
    st.success(f"Trouvé: {len(res)}")
    for s in sorted(res, key=lambda x: (x['score'], x['quality']), reverse=True):
        display_sig(s)
      
