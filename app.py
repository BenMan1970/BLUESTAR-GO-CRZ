import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import requests
from datetime import datetime, time as dtime, timezone, timedelta
import time
import logging
from typing import Optional, Dict, List

# ==========================================
# 1. CONFIGURATION & DESIGN "BLUESTAR"
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Final Pro", layout="centered", page_icon="💎")
logging.basicConfig(level=logging.WARNING)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0f1117; background-image: radial-gradient(at 50% 0%, #1f2937 0%, #0f1117 70%); }
    .main .block-container { max-width: 950px; padding-top: 2rem; }
    
    /* TITRE */
    h1 { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.8em; text-align: center; margin-bottom: 0.2em; }
    
    /* BOUTONS */
    .stButton>button { width: 100%; border-radius: 12px; height: 3.5em; font-weight: 700; font-size: 1.1em; background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); color: white; border: 1px solid rgba(255,255,255,0.1); transition: all 0.2s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(37, 99, 235, 0.4); }
    
    /* CARTES */
    .streamlit-expanderHeader { background-color: #1e293b !important; border: 1px solid #334155; border-radius: 10px; color: #f8fafc !important; font-weight: 600; }
    .streamlit-expanderContent { background-color: #161b22; border: 1px solid #334155; border-top: none; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }
    
    /* METRICS */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #f1f5f9; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.85rem; }
    
    /* BADGES */
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: 700; }
    .badge-news { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: 700; }
    
    /* PROGRESS BAR CUSTOM (METER) */
    .meter-container { width: 100%; background-color: #334155; border-radius: 10px; height: 12px; margin-top: 5px; }
    .meter-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DONNÉES & CACHE
# ==========================================
ASSETS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY",
    "XAU_USD", "US30_USD"
]

ALL_CROSSES = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]

if 'cache' not in st.session_state:
    st.session_state.cache = {}
    st.session_state.matrix_cache = None # Pour stocker la force calculée (CurrencyStrengthMeter)

# ==========================================
# 3. CLIENT API & UTILS
# ==========================================
class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except: st.error("⚠️ Config API manquante (secrets.toml)"); st.stop()

    def get_candles(self, instrument: str, granularity: str, count: int) -> pd.DataFrame:
        key = f"{instrument}_{granularity}"
        if key in st.session_state.cache: return st.session_state.cache[key]
        
        try:
            params = {"count": count, "granularity": granularity, "price": "M"}
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            self.client.request(r)
            data = []
            for c in r.response['candles']:
                if c['complete']:
                    data.append({'time': c['time'], 'open': float(c['mid']['o']), 'high': float(c['mid']['h']), 'low': float(c['mid']['l']), 'close': float(c['mid']['c'])})
            df = pd.DataFrame(data)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                st.session_state.cache[key] = df
            return df
        except: return pd.DataFrame()

# ==========================================
# 4. MOTEUR FONDAMENTAL (Le Cerveau)
# ==========================================
class CurrencyStrengthSystem:
    """
    Réplique la logique de calcul de CurrencyStrengthMeter.org
    et prépare les données pour la Market Map de Barchart.
    """
    @staticmethod
    def calculate_matrix(api: OandaClient):
        # Si déjà calculé il y a moins de 5 min, on garde (éviter latence)
        if st.session_state.matrix_cache: return st.session_state.matrix_cache

        with st.spinner("🔄 Calcul de la Matrice de Force (Scan complet du marché)..."):
            scores = {c: 0.0 for c in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']}
            details = {c: [] for c in scores.keys()} # Pour Barchart Map
            
            # On récupère la variation Daily de TOUTES les paires
            for pair in ALL_CROSSES:
                df = api.get_candles(pair, "D", 2)
                if not df.empty and len(df) >= 2:
                    op = df['open'].iloc[-1]
                    cl = df['close'].iloc[-1]
                    pct_change = ((cl - op) / op) * 100
                    
                    base, quote = pair.split('_')
                    
                    # Logique de Panier (Basket)
                    scores[base] += pct_change
                    scores[quote] -= pct_change
                    
                    # Stockage pour le visuel "Barchart"
                    details[base].append({'vs': quote, 'val': pct_change})
                    details[quote].append({'vs': base, 'val': -pct_change})
            
            # Normalisation 0-10 (Comme le site web)
            # On cherche le min et max pour étaler les scores
            vals = list(scores.values())
            if not vals: return None
            min_v, max_v = min(vals), max(vals)
            
            final_scores = {}
            for k, v in scores.items():
                # Formule pour mapper sur 0-10
                if max_v - min_v == 0: norm = 5.0
                else: norm = ((v - min_v) / (max_v - min_v)) * 10.0
                final_scores[k] = norm

            result = {'scores': final_scores, 'details': details}
            st.session_state.matrix_cache = result
            return result

    @staticmethod
    def get_pair_analysis(matrix, base, quote):
        if not matrix: return 5.0, 5.0, 0.0, []
        
        s_base = matrix['scores'].get(base, 5.0)
        s_quote = matrix['scores'].get(quote, 5.0)
        
        # Liste des détails pour le Market Map (Base vs Others)
        # On trie pour voir les meilleures perfs en premier
        map_data = sorted(matrix['details'].get(base, []), key=lambda x: x['val'], reverse=True)
        
        return s_base, s_quote, (s_base - s_quote), map_data

# ==========================================
# 5. INDICATEURS TECHNIQUES & MOTEUR SCAN
# ==========================================
def run_scanner(api, min_tech_score, risk_mode, strict_mode):
    # 1. Calcul FONDAMENTAL (Matrice)
    matrix = CurrencyStrengthSystem.calculate_matrix(api)
    
    signals = []
    pbar = st.progress(0)
    
    for i, symbol in enumerate(ASSETS):
        pbar.progress((i+1)/len(ASSETS))
        
        # Gestion des indices/Or (Pas de calcul de force complet)
        is_forex = symbol in ALL_CROSSES
        
        # Données M5 (Sniper)
        df = api.get_candles(symbol, "M5", 100)
        if df.empty or len(df) < 50: continue
        
        # --- CALCULS TECHNIQUES ---
        close = df['close']
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # HMA (Hull Moving Average) Trend
        def wma(s, l): return s.rolling(l).apply(lambda x: np.dot(x, np.arange(1, l+1))/np.arange(1, l+1).sum(), raw=True)
        wma1 = wma(close, int(20/2)); wma2 = wma(close, 20)
        hma = wma(2*wma1 - wma2, int(np.sqrt(20)))
        trend = 1 if hma.iloc[-1] > hma.iloc[-2] else -1
        
        # Détection du Signal Technique
        signal_type = None
        if current_rsi > 50 and trend == 1: signal_type = "BUY"
        elif current_rsi < 50 and trend == -1: signal_type = "SELL"
        
        if signal_type:
            score = 5 # Base technique
            
            # --- VALIDATION FONDAMENTALE (CSM) ---
            cs_data = {'b_score': 0, 'q_score': 0, 'gap': 0, 'map': []}
            
            if is_forex:
                parts = symbol.split('_')
                base, quote = parts[0], parts[1]
                b_s, q_s, gap, map_d = CurrencyStrengthSystem.get_pair_analysis(matrix, base, quote)
                cs_data = {'b_score': b_s, 'q_score': q_s, 'gap': gap, 'map': map_d}
                
                # LOGIQUE DU SITE "CURRENCYSTRENGTHMETER.ORG"
                # BUY: Base forte (>6), Quote faible (<4)
                # SELL: Base faible (<4), Quote forte (>6)
                
                valid_fund = False
                if signal_type == "BUY":
                    if b_s > q_s: score += 2 # Bon sens
                    if b_s >= 6.0 and q_s <= 4.0: score += 3; valid_fund = True # Configuration Parfaite
                else: # SELL
                    if q_s > b_s: score += 2
                    if q_s >= 6.0 and b_s <= 4.0: score += 3; valid_fund = True
                
                # FILTRE STRICT (MODE SNIPER)
                if strict_mode and not valid_fund:
                    continue # On zappe ce signal
            else:
                # Pour XAU/Indices, on utilise juste le trend D1
                df_d = api.get_candles(symbol, "D", 2)
                chg = (df_d['close'].iloc[-1] - df_d['open'].iloc[-1]) / df_d['open'].iloc[-1]
                if (signal_type == "BUY" and chg > 0) or (signal_type == "SELL" and chg < 0):
                    score += 3
            
            # --- FILTRE FINAL ---
            if score >= min_tech_score:
                signals.append({
                    'symbol': symbol, 'type': signal_type, 'price': close.iloc[-1],
                    'score': score, 'rsi': current_rsi, 'cs': cs_data, 'time': df['time'].iloc[-1]
                })
                
    pbar.empty()
    return signals

# ==========================================
# 6. AFFICHAGE VISUEL (STYLE SITE WEB)
# ==========================================
def draw_meter_bar(label, value, color_hex):
    # Dessine une jauge style "CurrencyStrengthMeter.org"
    # value est entre 0 et 10
    width_pct = min(100, max(0, value * 10))
    st.markdown(f"""
    <div style="margin-bottom:5px;">
        <div style="display:flex;justify-content:space-between;font-size:0.8em;color:#cbd5e1;font-weight:bold;">
            <span>{label}</span>
            <span>{value:.1f}/10</span>
        </div>
        <div class="meter-container">
            <div class="meter-fill" style="width:{width_pct}%; background-color:{color_hex};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_signal_card(s):
    is_buy = s['type'] == 'BUY'
    main_col = "#10b981" if is_buy else "#ef4444" # Vert / Rouge
    bg_grad = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    parts = s['symbol'].split('_')
    base = parts[0]
    quote = parts[1] if len(parts) > 1 else ""

    with st.expander(f"{s['symbol']} | {s['type']} | Score: {s['score']}/10", expanded=True):
        # Header visuel
        st.markdown(f"""
        <div style="background:{bg_grad};padding:15px;border-radius:8px;border-left:5px solid {main_col};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.5em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:3px 8px;border-radius:4px;color:white;margin-left:10px;font-weight:bold;">{s['type']}</span>
            </div>
            <div style="font-size:1.2em;font-weight:bold;color:white;">{s['price']:.5f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # SECTION 1 : VISUEL "CURRENCY STRENGTH METER"
        if quote: # Seulement pour Forex
            st.markdown("##### 📊 Force Relative (0-10)")
            c1, c2 = st.columns(2)
            
            b_score = s['cs']['b_score']
            q_score = s['cs']['q_score']
            
            # Couleurs dynamiques (Vert si fort, Rouge si faible, Gris si moyen)
            def get_col(val): return "#10b981" if val >= 6 else "#ef4444" if val <= 4 else "#94a3b8"
            
            with c1: draw_meter_bar(base, b_score, get_col(b_score))
            with c2: draw_meter_bar(quote, q_score, get_col(q_score))
            
            st.markdown("---")
            
            # SECTION 2 : VISUEL "BARCHART MARKET MAP"
            st.markdown("##### 🗺️ Market Map (Contexte)")
            st.caption(f"Comment {base} performe contre les autres ce jour :")
            
            # On affiche les 6 principales croix pour montrer si c'est "Tout Vert" ou "Tout Rouge"
            cols = st.columns(6)
            map_data = s['cs']['map'][:6] # Top 6 mouvements
            
            for i, item in enumerate(map_data):
                with cols[i]:
                    val = item['val']
                    col = "#10b981" if val > 0 else "#ef4444"
                    arrow = "▲" if val > 0 else "▼"
                    st.markdown(f"""
                    <div style="text-align:center;background:#1e293b;border-radius:5px;padding:5px;">
                        <div style="font-size:0.7em;color:#94a3b8;">vs {item['vs']}</div>
                        <div style="color:{col};font-weight:bold;font-size:0.9em;">{arrow}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# 7. INTERFACE PRINCIPALE
# ==========================================
st.title("💎 Bluestar SNP3 Final Pro")
st.markdown("Scanner Hybrid : Technique M5 + Force Fondamentale D1")

with st.expander("⚙️ Paramètres de Scan", expanded=True):
    c1, c2 = st.columns(2)
    score_min = c1.slider("Score Minimum", 5, 10, 7)
    strict = c2.checkbox("🔥 Mode Sniper (Strict)", value=True, help="Si activé, le scanner rejette le signal si la force des devises n'est pas parfaitement alignée (Ex: Base > 6/10 et Quote < 4/10).")

if st.button("🚀 LANCER LE SCAN DE FORCE", type="primary"):
    # Clear cache manuel si besoin
    # st.session_state.matrix_cache = None 
    
    api = OandaClient()
    results = run_scanner(api, score_min, False, strict)
    
    if not results:
        st.warning("Aucune opportunité trouvée avec ces filtres stricts. Le marché est peut-être en range.")
    else:
        st.success(f"{len(results)} Signaux Détectés !")
        # Tri par score décroissant
        results.sort(key=lambda x: x['score'], reverse=True)
        for sig in results:
            display_signal_card(sig)
