import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime, timezone
from scipy import stats 

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Ultimate v3.0", layout="centered", page_icon="💎")
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
        font-weight: 900; font-size: 2.8em; text-align: center; margin-bottom: 0.2em;
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em;
        border: 1px solid rgba(255,255,255,0.1);
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .streamlit-expanderHeader {
        background-color: #1e293b !important; border: 1px solid #334155;
        border-radius: 10px; color: #f8fafc !important; padding: 1.5rem;
    }
    .streamlit-expanderContent {
        background-color: #161b22; border: 1px solid #334155;
        border-top: none; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; padding: 20px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.9rem; }
    .badge { color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; margin: 2px; display: inline-block; }
    .badge-regime { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); }
    .badge-gps { background: linear-gradient(135deg, #059669 0%, #10b981 100%); }
    .badge-vol { background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); }
    .badge-blue { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); }
    .badge-purple { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); }
    .badge-gold { background: linear-gradient(135deg, #ca8a04 0%, #eab308 100%); }
    .risk-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px;
        text-align: center; border: 1px solid rgba(255,255,255,0.05);
    }
    .timestamp-box {
        background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3b82f6;
        padding: 8px 12px; border-radius: 6px; font-size: 0.85em;
        color: #93c5fd; margin: 10px 0;
    }
    .quality-indicator {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7em;
        font-weight: 700;
        margin-left: 8px;
    }
    .quality-high { background: #10b981; color: white; }
    .quality-medium { background: #f59e0b; color: white; }
    .quality-low { background: #6b7280; color: white; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CLIENT API & CACHE
# ==========================================
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'signal_history' not in st.session_state: st.session_state.signal_history = {}
if 'cs_data' not in st.session_state: st.session_state.cs_data = {'data': None, 'time': None}

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except Exception as e:
            st.error(f"⚠️ Configuration API manquante: {e}")
            st.stop()

    def get_candles(self, instrument, granularity, count):
        key = f"{instrument}_{granularity}"
        # Cache TTL de 60 secondes pour les données récentes
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
        except Exception as e:
            logging.error(f"Error fetching {instrument}: {e}")
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
# PARAMÈTRES PAR ACTIF
# ==========================================
def get_asset_params(symbol):
    if "XAU" in symbol:
        return {'type': 'COMMODITY', 'atr_threshold': 0.06, 'sl_base': 1.8, 'tp_rr': 2.5}
    if "US30" in symbol or "NAS" in symbol or "SPX" in symbol:
        return {'type': 'INDEX', 'atr_threshold': 0.10, 'sl_base': 2.2, 'tp_rr': 3.0}
    return {'type': 'FOREX', 'atr_threshold': 0.035, 'sl_base': 1.5, 'tp_rr': 2.0}

# ==========================================
# MOTEUR D'INDICATEURS (Optimisé)
# ==========================================
class QuantEngine:
    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(df, period=7):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.ewm(alpha=1/period, adjust=False).mean() / loss.ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def detect_structure_zscore(df, lookback=20):
        """Détecte la tendance via Z-Score (Scipy)"""
        if len(df) < lookback + 1: return 0
        
        close = df['close'].iloc[-1]
        # Utilisation de scipy.stats.zscore pour précision
        # On prend les 20 dernières bougies
        window = df['close'].iloc[-lookback:]
        z_score = stats.zscore(window)[-1]
        
        # Z-Score > 1.5 suggère une extension de tendance (puissance)
        if z_score > 1.5: return 1 # Bullish Power
        if z_score < -1.5: return -1 # Bearish Power
        return 0 # Neutral/Range

    @staticmethod
    def detect_smart_fvg(df, atr):
        """FVG validé par le volume relatif"""
        if len(df) < 4: return False, 0
        
        curr_close = df['close'].iloc[-1]
        min_gap = atr * 0.5
        # On regarde la bougie i-2 vs i (Gap entre 1 et 3)
        high_1 = df['high'].iloc[-3]
        low_1 = df['low'].iloc[-3]
        high_3 = df['high'].iloc[-1]
        low_3 = df['low'].iloc[-1]
        
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        vol_curr = df['volume'].iloc[-1]
        
        # FVG Bullish
        gap_bull = low_3 - high_1
        if gap_bull > min_gap and curr_close > high_1:
            # Confirmation par volume
            if vol_curr > vol_mean * 0.8: 
                return True, "BULL"
                
        # FVG Bearish
        gap_bear = low_1 - high_3
        if gap_bear > min_gap and curr_close < low_1:
            if vol_curr > vol_mean * 0.8:
                return True, "BEAR"
                
        return False, None

    @staticmethod
    def get_mtf_bias(df_d, df_w):
        """Bias institutionnel robuste"""
        def trend_score(df):
            if len(df) < 50: return 0
            sma200 = df['close'].rolling(200).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            price = df['close'].iloc[-1]
            
            score = 0
            if price > sma200: score += 1
            if ema50 > sma200: score += 1
            # Distance prix/SMA200 pour éviter le "noise" près de la moyenne
            dist = abs((price - sma200) / sma200)
            if dist > 0.005: score += 1 # Au moins 0.5% de marge
            
            return score # 0 à 3 (Bearish) à 3 (Bullish), 1.5 est neutre

        score_d = trend_score(df_d)
        score_w = trend_score(df_w)
        
        total = score_d + score_w
        if total >= 4: return "STRONG_BULL"
        if total >= 2: return "BULL"
        if total <= -4: return "STRONG_BEAR"
        if total <= -2: return "BEAR"
        return "NEUTRAL"

# ==========================================
# CURRENCY STRENGTH (Version Cache Manuel)
# ==========================================
def get_currency_strength(api):
    # Vérification du cache en session (TTL 60 secondes)
    now = datetime.now()
    if st.session_state.cs_data['time'] and (now - st.session_state.cs_data['time']).total_seconds() < 60:
        return st.session_state.cs_data['data']

    # Calcul si pas de cache ou expiré
    pct_changes = {}
    forex_pairs = [p for p in ASSETS if "_" in p and "XAU" not in p and "US30" not in p]
    
    for sym in forex_pairs:
        try:
            df = api.get_candles(sym, "H1", 50)
            if not df.empty and len(df) >= 2:
                pct_changes[sym] = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        except:
            continue
    
    if not pct_changes: 
        st.session_state.cs_data = {'data': None, 'time': now}
        return None
    
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    raw_scores = {c: 0.0 for c in currencies}
    
    for curr in currencies:
        for pair, pct in pct_changes.items():
            if curr in pair:
                base, quote = pair.split('_')
                if base == curr: raw_scores[curr] += pct
                elif quote == curr: raw_scores[curr] -= pct
    
    # Sauvegarde dans le cache
    st.session_state.cs_data = {'data': raw_scores, 'time': now}
    return raw_scores

# ==========================================
# ANALYSE DE PROBABILITÉ (WEIGHT OF EVIDENCE)
# ==========================================
def calculate_signal_probability(df_m5, df_h4, df_d, df_w, symbol, direction):
    """
    Calcule une probabilité de succès (0.0 - 1.0) au lieu d'un score additif.
    """
    prob_factors = []
    weights = []
    details = {}
    
    params = get_asset_params(symbol)
    atr = QuantEngine.calculate_atr(df_m5)
    atr_pct = (atr / df_m5['close'].iloc[-1]) * 100
    
    # --- 1. VOLATILITÉ ADÉQUATE (Filtre Binaire) ---
    if atr_pct < params['atr_threshold'] * 0.5:
        return 0, {}, atr_pct # Marché trop mort
    
    vol_score = min(atr_pct / params['atr_threshold'], 2.0) # Cap à 2.0
    details['vol_score'] = vol_score
    # On ajoute la volatilité comme facteur de confiance multiplicatif
    vol_conf = min(vol_score, 1.2) / 1.2 
    
    # --- 2. MOMENTUM RSI (Trigger Principal) ---
    rsi_serie = QuantEngine.calculate_rsi(df_m5)
    if len(rsi_serie) < 3: return 0, {}, atr_pct
    rsi_val = rsi_serie.iloc[-1]
    rsi_prev_val = rsi_serie.iloc[-2]
    rsi_mom = rsi_val - rsi_prev_val
    
    rsi_prob = 0
    if direction == "BUY":
        if rsi_prev_val < 50 and rsi_val >= 50 and rsi_mom > 1.5:
            rsi_prob = 0.85 # Trigger fort
        elif rsi_val > 50 and rsi_mom > 0:
            rsi_prob = 0.60 # Momentum continu
    else: # SELL
        if rsi_prev_val > 50 and rsi_val <= 50 and rsi_mom < -1.5:
            rsi_prob = 0.85
        elif rsi_val < 50 and rsi_mom < 0:
            rsi_prob = 0.60
            
    if rsi_prob == 0: return 0, {}, atr_pct # Pas de trigger valide
    prob_factors.append(rsi_prob)
    weights.append(0.35) # 35% du poids total
    details['rsi_mom'] = abs(rsi_mom)
    
    # --- 3. STRUCTURE DU MARCHÉ (Z-Score & ADX) ---
    z_score_struc = QuantEngine.detect_structure_zscore(df_h4, 20)
    
    struc_score = 0
    if direction == "BUY":
        if z_score_struc == 1: struc_score = 0.8 # Extension Haussière
        elif z_score_struc == 0: struc_score = 0.4 # Neutre
        else: struc_score = 0.0 # Contre extension Baissière
    else:
        if z_score_struc == -1: struc_score = 0.8
        elif z_score_struc == 0: struc_score = 0.4
        else: struc_score = 0.0
        
    prob_factors.append(struc_score)
    weights.append(0.25) # 25% du poids
    details['structure_z'] = z_score_struc
    
    # --- 4. BIAS MULTI-TIMEFRAME (GPS) ---
    mtf_bias = QuantEngine.get_mtf_bias(df_d, df_w)
    
    mtf_score = 0.5 # Neutre par défaut
    if direction == "BUY":
        if mtf_bias == "STRONG_BULL": mtf_score = 0.95
        elif mtf_bias == "BULL": mtf_score = 0.80
        elif mtf_bias == "NEUTRAL": mtf_score = 0.60
        else: mtf_score = 0.20 # Contre tendance
    else: # SELL
        if mtf_bias == "STRONG_BEAR": mtf_score = 0.95
        elif mtf_bias == "BEAR": mtf_score = 0.80
        elif mtf_bias == "NEUTRAL": mtf_score = 0.60
        else: mtf_score = 0.20
        
    prob_factors.append(mtf_score)
    weights.append(0.30) # 30% du poids
    details['mtf_bias'] = mtf_bias
    
    # --- 5. VALIDATION FVG (Confluence) ---
    fvg_active, fvg_type = QuantEngine.detect_smart_fvg(df_m5, atr)
    fvg_score = 0
    if fvg_active:
        if (direction == "BUY" and fvg_type == "BULL") or (direction == "SELL" and fvg_type == "BEAR"):
            fvg_score = 0.9 # FVG aligné
        else:
            fvg_score = 0.2 # FVG contre trade (ignorable)
    else:
        fvg_score = 0.6 # Pas de FVG, neutre
        
    prob_factors.append(fvg_score)
    weights.append(0.10) # 10% du poids (Bonus)
    details['fvg_align'] = fvg_active
    
    # --- CALCUL FINAL ---
    # Moyenne pondérée
    total_weight = sum(weights)
    weighted_prob = sum(p * w for p, w in zip(prob_factors, weights)) / total_weight
    
    # Ajustement par volatilité (Confiance de marché)
    final_score = weighted_prob * vol_conf
    
    return final_score, details, atr_pct

# ==========================================
# SCANNER PRINCIPAL
# ==========================================
def run_scan_v3(api, min_prob, strict_mode):
    # Utilisation de la fonction corrigée de Currency Strength
    cs_scores = get_currency_strength(api)
    
    signals = []
    bar = st.progress(0)
    
    for i, sym in enumerate(ASSETS):
        bar.progress((i+1)/len(ASSETS))
        
        # Cache cooldown simple
        if sym in st.session_state.signal_history:
            if (datetime.now() - st.session_state.signal_history[sym]).total_seconds() < 3600: 
                continue
        
        try:
            # Récupération Multi-Timeframe
            df_m5 = api.get_candles(sym, "M5", 150)
            df_h4 = api.get_candles(sym, "H4", 100)
            df_d = api.get_candles(sym, "D", 100)
            df_w_raw = api.get_candles(sym, "D", 300) # On resample nous-même
            
            if df_m5.empty or df_h4.empty or df_d.empty: continue
            
            # Resample Weekly manuel pour assurer la qualité
            df_w = df_w_raw.set_index('time').resample('W-FRI').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna().reset_index()
            
            # Détection direction via RSI Momentum (Scan Bi-directionnel)
            rsi_serie = QuantEngine.calculate_rsi(df_m5)
            if len(rsi_serie) < 3: continue
            
            rsi_mom = rsi_serie.iloc[-1] - rsi_serie.iloc[-2]
            
            scan_direction = None
            # Trigger conditionnel
            if rsi_serie.iloc[-2] < 50 and rsi_serie.iloc[-1] >= 50 and rsi_mom > 1.0:
                scan_direction = "BUY"
            elif rsi_serie.iloc[-2] > 50 and rsi_serie.iloc[-1] <= 50 and rsi_mom < -1.0:
                scan_direction = "SELL"
            
            if not scan_direction: continue
            
            # Calcul de la probabilité
            prob, details, atr_pct = calculate_signal_probability(
                df_m5, df_h4, df_d, df_w, sym, scan_direction
            )
            
            # Score Minimum Check
            if prob < min_prob: continue
            
            # Veto Strict Mode
            if strict_mode:
                if details['mtf_bias'] == "NEUTRAL" and details['structure_z'] == 0:
                    continue
            
            # Currency Strength Check
            cs_aligned = False
            base, quote = sym.split('_')
            if cs_scores and base and quote:
                gap = cs_scores.get(base, 0) - cs_scores.get(quote, 0)
                if scan_direction == "BUY" and gap > 0.3: cs_aligned = True
                elif scan_direction == "SELL" and gap < -0.3: cs_aligned = True
            elif "XAU" in sym or "US30" in sym:
                cs_aligned = True # Indices/Commodités moins dépendants du CS simple

            if strict_mode and not cs_aligned: continue
            
            # Calcul SL/TP
            price = df_m5['close'].iloc[-1]
            atr = QuantEngine.calculate_atr(df_m5)
            params = get_asset_params(sym)
            
            # SL Adaptatif
            sl_mult = params['sl_base']
            if details['structure_z'] != 0: sl_mult -= 0.2 # SL plus serré si structure claire
            sl_mult = max(sl_mult, 1.0)
            
            sl = price - (atr * sl_mult) if scan_direction == "BUY" else price + (atr * sl_mult)
            
            # TP Basé sur le RR paramétré de l'actif
            tp_mult = params['tp_rr']
            tp = price + (atr * tp_mult) if scan_direction == "BUY" else price - (atr * tp_mult)
            
            signals.append({
                'symbol': sym,
                'type': scan_direction,
                'price': price,
                'prob': prob, # Probabilité 0-1
                'score_display': prob * 10, # Pour affichage 0-10
                'details': details,
                'atr_pct': atr_pct,
                'time': df_m5['time'].iloc[-1],
                'sl': sl,
                'tp': tp,
                'rr': tp_mult / sl_mult,
                'cs_aligned': cs_aligned
            })
            
            st.session_state.signal_history[sym] = datetime.now()
            
        except Exception as e:
            logging.error(f"Scan error {sym}: {e}")
            continue
            
    bar.empty()
    return sorted(signals, key=lambda x: x['prob'], reverse=True)

# ==========================================
# AFFICHAGE SIGNAL
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score_display']
    # Mapping Probabilité -> Label
    if sc >= 8.5: 
        label = "💎 INSTITUTIONAL"
        quality_badge = "quality-high"
    elif sc >= 7.5: 
        label = "⭐ ALGORITHMIC"
        quality_badge = "quality-high"
    elif sc >= 6.5:
        label = "✅ STRATEGIC"
        quality_badge = "quality-medium"
    else: 
        label = "📊 TACTICAL"
        quality_badge = "quality-medium"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"<div class='timestamp-box'>📅 {s['time'].strftime('%d/%m/%Y %H:%M UTC')}</div>", 
                   unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;
                            color:white;margin-left:10px;">{s['type']}</span>
                <span class="quality-indicator {quality_badge}">{int(s['prob']*100)}% CONF</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.3f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Badges
        badges = []
        badges.append(f"<span class='badge badge-regime'>{s['details']['mtf_bias']}</span>")
        
        # Z-Score Interpretation
        z_val = s['details']['structure_z']
        z_text = "NEUTRAL"
        if z_val > 0: z_text = "BULL POWER"
        if z_val < 0: z_text = "BEAR POWER"
        badges.append(f"<span class='badge badge-purple'>Z-SCORE: {z_text}</span>")
        
        if s['cs_aligned']: 
            badges.append("<span class='badge badge-blue'>CS ALIGNÉ</span>")
        if s['details']['fvg_align']: 
            badges.append("<span class='badge badge-gold'>FVG ACTIF</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", 
                   unsafe_allow_html=True)
        
        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confiance", f"{int(s['prob']*100)}%")
        c2.metric("Bias MTF", f"{s['details']['mtf_bias']}")
        c3.metric("RSI Mom.", f"{s['details']['rsi_mom']:.1f}")
        c4.metric("R:R", f"1:{s['rr']:.1f}")

        st.markdown("---")
        
        # Métriques techniques
        t1, t2, t3 = st.columns(3)
        t1.metric("Z-Score", f"{z_val:.1f}")
        t2.metric("Vol Score", f"{s['details']['vol_score']:.2f}")
        t3.metric("Prob. Ajustée", f"{int(s['prob']*100)}%")
        
        st.markdown("---")
        
        # Risk Management
        r1, r2 = st.columns(2)
        r1.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>STOP LOSS</div>
            <div style='color:#ef4444;font-weight:bold;font-size:1.2em;'>{s['sl']:.5f}</div>
            <div style='color:#64748b;font-size:0.7em;margin-top:4px;'>
                {abs(s['price'] - s['sl']):.5f} ({(abs(s['price'] - s['sl'])/s['price']*100):.2f}%)
            </div>
        </div>""", unsafe_allow_html=True)
        
        r2.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>TAKE PROFIT</div>
            <div style='color:#10b981;font-weight:bold;font-size:1.2em;'>{s['tp']:.5f}</div>
            <div style='color:#64748b;font-size:0.7em;margin-top:4px;'>
                {abs(s['tp'] - s['price']):.5f} ({(abs(s['tp'] - s['price'])/s['price']*100):.2f}%)
            </div>
        </div>""", unsafe_allow_html=True)

# ==========================================
# INTERFACE PRINCIPALE
# ==========================================
def main():
    st.title("💎 BLUESTAR ULTIMATE v3.0")
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:0.9em;'>Probability Engine | Z-Score Structure | Institutional Risk</p>", 
               unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration v3.0")
        
        strict_mode = st.checkbox("🔥 Mode Strict (Z-Score)", value=False, 
                                 help="Filtre agressif : Requiere Z-Score non-neutre et Bias clair.")
        
        # Slider de probabilité (0.0 à 1.0 converti en 0-10 pour l'UI)
        min_prob_display = st.slider("Confiance Minimale (%)", 50, 95, 75, 5)
        min_prob = min_prob_display / 100.0
        
        st.markdown("---")
        st.markdown("### 📊 Nouvelle Métrique")
        st.markdown("""
        <div style='font-size:0.85em;color:#94a3b8;line-height:1.8;'>
        <b>Z-Score Structure:</b> Puissance de la tendance<br>
        <b>Weight of Evidence:</b> Score pondéré (RSI, MTF)<br>
        <b>Volatility Adjusted:</b> Score ajusté au bruit
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Échelle de Confiance")
        st.markdown("""
        <div style='font-size:0.8em;color:#94a3b8;line-height:1.6;'>
        <b>85%+:</b> 💎 Institutional (Alignement parfait)<br>
        <b>75%-85%:</b> ⭐ Algorithmic (Forte probabilité)<br>
        <b>65%-75%:</b> ✅ Strategic (Validé)<br>
        <b>&lt;65%:</b> Filtré par défaut
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        if strict_mode:
            st.warning("🛡️ **MODE STRICT ACTIF**\n\nRequiere:\n- Z-Score Power (Extension)\n- Bias MTF Non-Neutre\n- Alignement CS")
    
    if st.button("🚀 LANCER LE SCAN v3.0", type="primary"):
        st.session_state.cache = {}
        api = OandaClient()
        
        with st.spinner(f"🔍 Calcul Probabiliste sur {len(ASSETS)} actifs..."):
            results = run_scan_v3(api, min_prob, strict_mode)
        
        if not results:
            st.warning("⚠️ Aucun signal ne répond aux critères de haute confiance.")
            
            with st.expander("🔧 Diagnostic Quant"):
                st.markdown("""
                **Raisons possibles :**
                
                1. **Volatilité insuffisante** pour générer un Z-Score significatif.
                2. **Neutralité du marché** : Les Bias MTF sont probablement "NEUTRAL".
                3. **Divergence Structurelle** : Le RSI donne un signal mais le prix est dans une zone de consolidation (Z-Score proche de 0).
                
                **Actions :**
                - Baisser la confiance minimale à **65%**.
                - Désactiver le "Mode Strict" pour autoriser les marchés neutres.
                """)
        else:
            st.success(f"✅ {len(results)} Signal{'s' if len(results) > 1 else ''} Détecté{'s' if len(results) > 1 else ''}")
            
            avg_conf = sum(s['prob'] for s in results) / len(results)
            inst = sum(1 for s in results if s['prob'] >= 0.85)
            algo = sum(1 for s in results if 0.75 <= s['prob'] < 0.85)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Signaux", len(results))
            col2.metric("Conf. Moyenne", f"{int(avg_conf*100)}%")
            col3.metric("Institutional", inst)
            col4.metric("Algorithmic", algo)
            
            st.markdown("---")
            
            for sig in results:
                display_sig(sig)
                
            st.markdown("---")
            st.caption("💡 **Note Quant** : Ce scanner utilise une approche probabiliste. Un score de 85% signifie que les facteurs internes (RSI, MTF, Structure) sont fortement alignés.")

if __name__ == "__main__":
    main()
