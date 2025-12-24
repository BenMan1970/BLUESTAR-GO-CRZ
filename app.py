import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import time
import logging
from datetime import datetime, timezone, timedelta
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Ultimate", layout="wide", page_icon="💎")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CSS Combiné (App Trading + Dashboard Cards + Map)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    
    .stApp { background-color: #0f1117; }
    
    /* --- HEADERS --- */
    h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900; text-align: center; margin-bottom: 0.2em;
    }
    
    /* --- DASHBOARD CARDS (NOUVEAU) --- */
    .currency-card {
        background-color: #1e293b; border-radius: 10px; padding: 15px;
        margin-bottom: 10px; border: 1px solid #334155; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.2s;
    }
    .currency-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
    
    .card-header { 
        display: flex; justify-content: center; align-items: center; gap: 8px; 
        font-weight: bold; color: #f8fafc; font-size: 1.1rem; margin-bottom: 5px;
    }
    .strength-score { 
        font-size: 2.0rem; font-weight: 800; margin: 5px 0;
        display: flex; justify-content: center; align-items: center; gap: 10px;
    }
    .progress-bg { background-color: #0f172a; height: 6px; border-radius: 3px; width: 100%; margin-top: 10px; }
    .progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
    
    /* COLORS */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }
    
    /* --- SCANNER BADGES --- */
    .badge-gps { padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; color: white; }
    
    /* --- BUTTONS --- */
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em; font-weight: 700;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none;
    }
</style>
""", unsafe_allow_html=True)

# Drapeaux pour l'UI
FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp", "AUD": "au", 
    "CAD": "ca", "NZD": "nz", "CHF": "ch", "XAU": "xk", "US30": "us"
}

# ==========================================
# 2. CLIENT API & DATA
# ==========================================
# Initialisation Session State
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'matrix_data' not in st.session_state: st.session_state.matrix_data = None

ASSETS_FOREX = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]
ASSETS_SPECIAL = {'XAU_USD': 'GOLD', 'US30_USD': 'US30', 'NAS100_USD': 'NAS100'}

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.account_id = st.secrets["OANDA_ACCOUNT_ID"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except:
            st.error("⚠️ Secrets OANDA manquants")
            st.stop()

    def get_candles(self, instrument, granularity, count):
        key = f"{instrument}_{granularity}"
        # Cache simple pour éviter spam API pendant le dev
        if key in st.session_state.cache:
            # Check expiration (1 min)
            timestamp, data = st.session_state.cache[key]
            if (datetime.now() - timestamp).total_seconds() < 60:
                return data

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
                        'low': float(c['mid']['l']), 'close': float(c['mid']['c'])
                    })
            df = pd.DataFrame(data)
            if not df.empty:
                st.session_state.cache[key] = (datetime.now(), df)
            return df
        except:
            return pd.DataFrame()

# ==========================================
# 3. MOTEUR MATHÉMATIQUE (REMPLACE LE SCRAPING)
# ==========================================
class MathCurrencySystem:
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def normalize_score(rsi_value):
        # Transforme RSI 0-100 en Score 0-10
        # RSI 50 = Score 5. RSI 70 = Score 7. RSI 30 = Score 3.
        return ((rsi_value - 50) / 50 + 1) * 5

    @staticmethod
    def process_market_data(api, granularity="H1"):
        """Récupère tout et calcule la matrice mathématiquement"""
        prices = {}
        pct_changes = {}
        
        # 1. Fetch Forex pour Matrice
        progress_text = "📥 Téléchargement des données..."
        my_bar = st.progress(0, text=progress_text)
        
        total_assets = len(ASSETS_FOREX) + len(ASSETS_SPECIAL)
        
        for i, pair in enumerate(ASSETS_FOREX):
            df = api.get_candles(pair, granularity, 100)
            if not df.empty:
                prices[pair] = df['close']
                # % Change pour la Map (sur la dernière bougie close)
                op = df['open'].iloc[-1]
                cl = df['close'].iloc[-1]
                pct_changes[pair] = ((cl - op) / op) * 100
            my_bar.progress((i + 1) / total_assets)

        # 2. Fetch Specials
        scores_special = {}
        pct_special = {}
        
        for i, (sym, name) in enumerate(ASSETS_SPECIAL.items()):
            df = api.get_candles(sym, granularity, 100)
            if not df.empty:
                rsi = MathCurrencySystem.calculate_rsi(df['close'], 14)
                curr_score = MathCurrencySystem.normalize_score(rsi.iloc[-1])
                prev_score = MathCurrencySystem.normalize_score(rsi.iloc[-2])
                scores_special[name] = (curr_score, prev_score)
                
                op = df['open'].iloc[-1]
                cl = df['close'].iloc[-1]
                pct = ((cl - op) / op) * 100
                pct_special[name] = {'pct': pct, 'cat': 'SPECIAL'}
            
            my_bar.progress((len(ASSETS_FOREX) + i + 1) / total_assets)
            
        my_bar.empty()

        if not prices: return None

        # 3. Calcul Matriciel Forex
        df_prices = pd.DataFrame(prices).fillna(method='ffill').fillna(method='bfill')
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
        scores_forex = {}
        
        for curr in currencies:
            total_curr, total_prev, count = 0.0, 0.0, 0
            opponents = [c for c in currencies if c != curr]
            for opp in opponents:
                pair_d = f"{curr}_{opp}"
                pair_i = f"{opp}_{curr}"
                
                rsi_s = None
                if pair_d in df_prices.columns:
                    rsi_s = MathCurrencySystem.calculate_rsi(df_prices[pair_d])
                elif pair_i in df_prices.columns:
                    # RSI de l'inverse (1/prix)
                    rsi_s = MathCurrencySystem.calculate_rsi(1/df_prices[pair_i])
                
                if rsi_s is not None:
                    total_curr += MathCurrencySystem.normalize_score(rsi_s.iloc[-1])
                    total_prev += MathCurrencySystem.normalize_score(rsi_s.iloc[-2])
                    count += 1
            
            if count > 0:
                scores_forex[curr] = (total_curr / count, total_prev / count)
        
        # Structure de retour compatible avec le scanner
        final_scores = {k: v[0] for k, v in scores_forex.items()}
        
        return {
            'scores': final_scores,          # Pour le scanner (valeur actuelle)
            'scores_full': scores_forex,     # Pour l'affichage (actuel, prev)
            'scores_special': scores_special,# Gold/Indices
            'pct_changes': pct_changes,      # Pour la map
            'pct_special': pct_special,      # Pour la map
            'timestamp': datetime.now()
        }

    @staticmethod
    def generate_map_html(matrix_data):
        """Génère le HTML exact de la Market Map"""
        pct_forex = matrix_data['pct_changes']
        pct_special = matrix_data['pct_special']
        
        # Logique couleurs
        def get_col(pct):
            if pct >= 0.15: return "#009900", "white"
            if pct >= 0.01: return "#33cc33", "white"
            if pct <= -0.15: return "#cc0000", "white"
            if pct <= -0.01: return "#ff3300", "white"
            return "#f0f0f0", "#333"

        # Organisation des données Forex par devise de base
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
        forex_map = {c: [] for c in currencies}
        
        for pair, val in pct_forex.items():
            base, quote = pair.split('_')
            forex_map[base].append({'pair': quote, 'pct': val})
            forex_map[quote].append({'pair': base, 'pct': -val})

        # Tri par force globale
        sums = {c: sum(x['pct'] for x in items) for c, items in forex_map.items()}
        sorted_curr = sorted(sums, key=sums.get, reverse=True)

        html = """
        <style>
            body { font-family: 'Roboto', sans-serif; margin: 0; padding: 0; background: transparent; }
            .section-header { color: #94a3b8; font-size: 12px; font-weight: bold; margin: 15px 0 5px 0; border-bottom: 1px solid #334155; }
            .matrix-row { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 5px; }
            .currency-col { display: flex; flex-direction: column; min-width: 85px; gap: 1px; }
            .tile { display: flex; justify-content: space-between; padding: 4px; font-size: 11px; font-weight: bold; }
            .sep { background: #1e293b; color: #f8fafc; padding: 4px; font-size: 12px; font-weight: 900; text-align: center; border-left: 3px solid #3b82f6; }
            .grid-container { display: flex; gap: 10px; }
            .big-box { width: 120px; padding: 10px; display: flex; flex-direction: column; align-items: center; border-radius: 4px; color: white; }
        </style>
        <div class="section-header">FOREX HEATMAP (H1)</div>
        <div class="matrix-row">
        """
        
        for curr in sorted_curr:
            items = forex_map[curr]
            items.sort(key=lambda x: x['pct'], reverse=True)
            html += '<div class="currency-col">'
            
            # Gagnants
            for x in items:
                if x['pct'] >= 0.01:
                    bg, txt = get_col(x['pct'])
                    html += f'<div class="tile" style="background:{bg};color:{txt}"><span>{x["pair"]}</span><span>+{x["pct"]:.2f}%</span></div>'
            
            # Header
            html += f'<div class="sep">{curr}</div>'
            
            # Neutres/Perdants
            for x in items:
                if x['pct'] < 0.01:
                    bg, txt = get_col(x['pct'])
                    sign = "+" if x['pct'] > 0 else ""
                    html += f'<div class="tile" style="background:{bg};color:{txt}"><span>{x["pair"]}</span><span>{sign}{x["pct"]:.2f}%</span></div>'
            html += '</div>'
            
        html += '</div><div class="section-header">ASSETS</div><div class="grid-container">'
        
        for name, data in pct_special.items():
            bg, txt = get_col(data['pct'])
            html += f'<div class="big-box" style="background:{bg};"><span style="font-size:10px">{name}</span><span style="font-size:14px;font-weight:bold">{data["pct"]:+.2f}%</span></div>'
        
        html += '</div>'
        return html

# ==========================================
# 4. SCANNER LOGIC (GPS + MATH MATRIX)
# ==========================================
def ema(series, length): return series.ewm(span=length, adjust=False).mean()

def calculate_mtf_gps(api, symbol, direction):
    # Version simplifiée pour l'exemple (Reprend ta logique GPS complète ici normalement)
    # On simule un score GPS basé sur les données Daily/H4
    try:
        df = api.get_candles(symbol, "H4", 50)
        if df.empty: return 0, 'N/A'
        
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        close = df['close'].iloc[-1]
        
        trend = "BUY" if close > sma50 else "SELL"
        score = 3.0 if trend == direction else 0.0
        return score, 'A' if score == 3 else 'C'
    except:
        return 0, 'N/A'

def run_scanner(api, matrix_data, strict_mode):
    signals = []
    scores = matrix_data['scores'] # Dictionnaire simple {'USD': 7.2, ...}
    
    progress_text = "🕵️ Scan du marché (M5)..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, sym in enumerate(ASSETS_FOREX):
        # 1. Données M5
        df = api.get_candles(sym, "M5", 100)
        if df.empty: continue
        
        # 2. Indicateurs M5
        rsi = MathCurrencySystem.calculate_rsi(df['close']).iloc[-1]
        
        # HMA Trend (Simplifié)
        hma_val = df['close'].rolling(20).mean().iloc[-1] # Placeholder pour HMA
        price = df['close'].iloc[-1]
        trend_m5 = 1 if price > hma_val else -1
        
        signal = None
        if trend_m5 == 1 and 30 < rsi < 60: signal = "BUY"
        elif trend_m5 == -1 and 40 < rsi < 70: signal = "SELL"
        
        if not signal: 
            my_bar.progress((i+1)/len(ASSETS_FOREX))
            continue
            
        # 3. Filtre Fondamental (MATH MATRIX)
        base, quote = sym.split('_')
        s_b = scores.get(base, 5.0)
        s_q = scores.get(quote, 5.0)
        gap = s_b - s_q
        
        fund_valid = False
        if signal == "BUY" and gap > 1.0: fund_valid = True
        if signal == "SELL" and gap < -1.0: fund_valid = True
        
        if strict_mode and not fund_valid: continue
        
        # 4. GPS
        gps_score, gps_quality = calculate_mtf_gps(api, sym, signal)
        
        # Score Final
        total_score = (gps_score * 0.4 + (abs(gap)/2) * 0.4 + 2) # Formule simplifiée
        total_score = min(10, total_score)
        
        if total_score >= 6.0:
            signals.append({
                'symbol': sym, 'type': signal, 'price': price,
                'score': total_score, 'gps': gps_quality,
                's_b': s_b, 's_q': s_q, 'gap': gap, 'rsi': rsi
            })
            
        my_bar.progress((i+1)/len(ASSETS_FOREX))
    
    my_bar.empty()
    return sorted(signals, key=lambda x: x['score'], reverse=True)

# ==========================================
# 5. UI DISPLAY COMPONENTS
# ==========================================
def display_strength_card(name, current, previous):
    delta = current - previous
    if current >= 7: c_txt, c_bg = "text-green", "bg-green"
    elif current >= 5.5: c_txt, c_bg = "text-blue", "bg-blue"
    elif current >= 4: c_txt, c_bg = "text-orange", "bg-orange"
    else: c_txt, c_bg = "text-red", "bg-red"
    
    arrow = "↗" if delta > 0.1 else "↘" if delta < -0.1 else "→"
    flag_code = FLAG_URLS.get(name, "xk")
    
    img_html = "🟡" if name == "GOLD" else "📊" if "30" in name or "100" in name else f'<img src="https://flagcdn.com/48x36/{flag_code}.png" style="width:24px;border-radius:2px;">'
    
    return f"""
    <div class="currency-card">
        <div class="card-header">{img_html} <span>{name}</span></div>
        <div class="strength-score {c_txt}">
            {current:.1f} <span style="font-size:0.6em;color:#64748b">{arrow}</span>
        </div>
        <div class="progress-bg"><div class="progress-fill {c_bg}" style="width:{min(current*10, 100)}%;"></div></div>
    </div>
    """

def display_signal(s):
    col = "#10b981" if s['type'] == 'BUY' else "#ef4444"
    with st.expander(f"{s['symbol']} {s['type']} | Score: {s['score']:.1f}/10", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", s['price'])
        c2.metric("GPS", s['gps'])
        c3.metric("RSI", f"{s['rsi']:.1f}")
        
        base, quote = s['symbol'].split('_')
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);padding:10px;border-radius:5px;margin-top:5px;display:flex;justify-content:space-between;align-items:center;">
            <span><b>{base}</b> ({s['s_b']:.1f})</span>
            <span style="font-weight:bold;color:{col}">GAP: {s['gap']:+.2f}</span>
            <span><b>{quote}</b> ({s['s_q']:.1f})</span>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. MAIN APP
# ==========================================
def main():
    st.title("💎 BLUESTAR SNP3 ULTIMATE")
    
    with st.sidebar:
        st.header("Paramètres")
        scan_tf = st.selectbox("Analyse Timeframe", ["H1", "H4"], index=0, help="TF pour calculer la Force des Devises")
        strict_mode = st.checkbox("Mode Sniper Strict", value=False)
        st.info("Scanner autonome: Utilise l'API OANDA pour calculer la force mathématique (RSI) et générer la Map.")
    
    api = OandaClient()
    
    if st.button("🚀 LANCER L'ANALYSE GLOBALE", type="primary"):
        # 1. CALCUL DU DASHBOARD (FORCE + MAP)
        st.session_state.cache = {} # Reset cache pour fraîcheur
        
        with st.status("🔄 Analyse du marché en cours...", expanded=True) as status:
            st.write("1. Récupération des données H1/H4...")
            matrix = MathCurrencySystem.process_market_data(api, granularity=scan_tf)
            st.session_state.matrix_data = matrix
            
            st.write("2. Recherche des opportunités M5...")
            signals = run_scanner(api, matrix, strict_mode)
            
            status.update(label="✅ Analyse terminée !", state="complete", expanded=False)
        
        # 2. AFFICHAGE DU DASHBOARD (HAUT)
        st.markdown("### 📊 DASHBOARD FONDAMENTAL")
        
        # Cartes Forex
        scores = matrix['scores_full']
        sorted_fx = sorted(scores.keys(), key=lambda x: scores[x][0], reverse=True)
        cols = st.columns(8) # 8 devises
        for i, curr in enumerate(sorted_fx):
            with cols[i]:
                st.markdown(display_strength_card(curr, scores[curr][0], scores[curr][1]), unsafe_allow_html=True)
        
        # Cartes Speciales (Or, Indices)
        if matrix['scores_special']:
            st.write("")
            sp_cols = st.columns(len(matrix['scores_special']))
            for i, (name, val) in enumerate(matrix['scores_special'].items()):
                with sp_cols[i]:
                    st.markdown(display_strength_card(name, val[0], val[1]), unsafe_allow_html=True)
        
        # MARKET MAP
        st.write("")
        with st.expander("🗺️ Market Map & Heatmap", expanded=False):
            html_map = MathCurrencySystem.generate_map_html(matrix)
            components.html(html_map, height=250, scrolling=True)

        # 3. AFFICHAGE DES SIGNAUX (BAS)
        st.markdown("---")
        st.markdown(f"### 🎯 SIGNAUX DE TRADING ({len(signals)})")
        
        if not signals:
            st.warning("Aucun signal ne correspond aux critères stricts.")
        else:
            grid = st.columns(2)
            for i, sig in enumerate(signals):
                with grid[i % 2]:
                    display_signal(sig)

if __name__ == "__main__":
    main()
