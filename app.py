import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURATION & STYLE (VISUEL ULTIMATE)
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Ultimate", layout="wide", page_icon="💎")
logging.basicConfig(level=logging.INFO)

# CSS STRICTEMENT IDENTIQUE À LA VERSION VALIDÉE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0f1117; }
    
    /* DASHBOARD CARDS */
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
    
    /* COULEURS */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }
    
    /* BADGES SCANNER */
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    .badge-gps { background: #334155; color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.75em; font-weight: 700; }
    
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em; font-weight: 700;
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none;
    }
</style>
""", unsafe_allow_html=True)

FLAG_URLS = {
    "USD": "us", "EUR": "eu", "GBP": "gb", "JPY": "jp", "AUD": "au", 
    "CAD": "ca", "NZD": "nz", "CHF": "ch", "XAU": "xk", "US30": "us"
}

ASSETS_FOREX = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_CAD", "EUR_AUD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_CAD", "GBP_AUD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "NZD_JPY", "NZD_CAD", "NZD_CHF", "CHF_JPY"
]
ASSETS_SPECIAL = {'XAU_USD': 'GOLD', 'US30_USD': 'US30', 'NAS100_USD': 'NAS100'}

# ==========================================
# 2. CLIENT API
# ==========================================
if 'cache' not in st.session_state: st.session_state.cache = {}
if 'matrix_data' not in st.session_state: st.session_state.matrix_data = None

class OandaClient:
    def __init__(self):
        try:
            self.access_token = st.secrets["OANDA_ACCESS_TOKEN"]
            self.environment = st.secrets.get("OANDA_ENVIRONMENT", "practice")
            self.client = oandapyV20.API(access_token=self.access_token, environment=self.environment)
        except:
            st.error("⚠️ Config OANDA manquante")
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
                        'low': float(c['mid']['l']), 'close': float(c['mid']['c'])
                    })
            df = pd.DataFrame(data)
            if not df.empty:
                st.session_state.cache[key] = (datetime.now(), df)
            return df
        except:
            return pd.DataFrame()

# ==========================================
# 3. MOTEUR INDICATEURS (AMÉLIORÉ)
# ==========================================
class SmartIndicators:
    @staticmethod
    def calculate_atr(df, period=14):
        h, l, c = df['high'], df['low'], df['close']
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]

    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        rs = gain.ewm(alpha=1/period, adjust=False).mean() / loss.ewm(alpha=1/period, adjust=False).mean()
        return 100 - (100 / (1 + rs))

    @staticmethod
    def normalize_score(rsi_value):
        return ((rsi_value - 50) / 50 + 1) * 5

    @staticmethod
    def get_hma_trend(series, length=20):
        # Hull Moving Average pour direction instantanée
        wma1 = series.rolling(int(length/2)).mean()
        wma2 = series.rolling(length).mean()
        raw_hma = 2 * wma1 - wma2
        hma = raw_hma.rolling(int(np.sqrt(length))).mean()
        
        if len(hma) < 2: return 0, 0
        
        curr = hma.iloc[-1]
        prev = hma.iloc[-2]
        trend = 1 if curr > prev else -1
        return trend, curr

    @staticmethod
    def detect_institutional_fvg(df, atr):
        """Détection FVG 'Smart Money' avec filtre ATR"""
        if len(df) < 5: return False, None
        
        for i in range(1, 4): 
            high_prev = df['high'].iloc[-(i+2)]
            low_curr = df['low'].iloc[-i]
            low_prev = df['low'].iloc[-(i+2)]
            high_curr = df['high'].iloc[-i]
            
            # Gap > 30% ATR pour éviter le bruit
            min_gap = atr * 0.3
            
            if low_curr > high_prev: # Bullish
                if (low_curr - high_prev) > min_gap: return True, "BULL"
            
            if high_curr < low_prev: # Bearish
                if (low_prev - high_curr) > min_gap: return True, "BEAR"
                    
        return False, None

# ==========================================
# 4. TRAITEMENT DONNÉES & MAP HTML
# ==========================================
class MathCurrencySystem:
    @staticmethod
    def process_market_data(api, granularity="H1"):
        prices = {}
        pct_changes = {}
        
        # Barre de progression
        total_assets = len(ASSETS_FOREX) + len(ASSETS_SPECIAL)
        progress_bar = st.progress(0, text="📥 Analyse Matrice...")
        
        # 1. Forex Data
        for i, pair in enumerate(ASSETS_FOREX):
            df = api.get_candles(pair, granularity, 100)
            if not df.empty:
                prices[pair] = df['close']
                op = df['open'].iloc[-1]
                cl = df['close'].iloc[-1]
                pct_changes[pair] = ((cl - op) / op) * 100
            progress_bar.progress((i + 1) / total_assets)

        # 2. Special Assets Data
        scores_special = {}
        pct_special = {}
        for i, (sym, name) in enumerate(ASSETS_SPECIAL.items()):
            df = api.get_candles(sym, granularity, 100)
            if not df.empty:
                rsi = SmartIndicators.calculate_rsi(df['close'], 14)
                scores_special[name] = (SmartIndicators.normalize_score(rsi.iloc[-1]), 
                                      SmartIndicators.normalize_score(rsi.iloc[-2]))
                op = df['open'].iloc[-1]
                cl = df['close'].iloc[-1]
                pct_special[name] = {'pct': ((cl - op) / op) * 100, 'cat': 'SPECIAL'}
            progress_bar.progress((len(ASSETS_FOREX) + i + 1) / total_assets)
            
        progress_bar.empty()
        if not prices: return None

        # 3. Calcul Matrice
        df_prices = pd.DataFrame(prices).ffill().bfill()
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
        scores_forex = {}
        
        for curr in currencies:
            vals_curr, vals_prev = [], []
            for opp in currencies:
                if curr == opp: continue
                pair_d = f"{curr}_{opp}"
                pair_i = f"{opp}_{curr}"
                
                if pair_d in df_prices.columns:
                    rsi = SmartIndicators.calculate_rsi(df_prices[pair_d])
                    vals_curr.append(SmartIndicators.normalize_score(rsi.iloc[-1]))
                    vals_prev.append(SmartIndicators.normalize_score(rsi.iloc[-2]))
                elif pair_i in df_prices.columns:
                    rsi = SmartIndicators.calculate_rsi(1/df_prices[pair_i])
                    vals_curr.append(SmartIndicators.normalize_score(rsi.iloc[-1]))
                    vals_prev.append(SmartIndicators.normalize_score(rsi.iloc[-2]))
            
            if vals_curr:
                scores_forex[curr] = (np.mean(vals_curr), np.mean(vals_prev))
        
        return {
            'scores': {k: v[0] for k, v in scores_forex.items()},
            'scores_full': scores_forex,
            'scores_special': scores_special,
            'pct_changes': pct_changes,
            'pct_special': pct_special
        }

    @staticmethod
    def generate_map_html(matrix_data):
        # Code HTML exact de la version Ultimate
        pct_forex = matrix_data['pct_changes']
        pct_special = matrix_data['pct_special']
        
        def get_col(pct):
            if pct >= 0.15: return "#009900", "white"
            if pct >= 0.01: return "#33cc33", "white"
            if pct <= -0.15: return "#cc0000", "white"
            if pct <= -0.01: return "#ff3300", "white"
            return "#f0f0f0", "#333"

        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
        forex_map = {c: [] for c in currencies}
        
        for pair, val in pct_forex.items():
            base, quote = pair.split('_')
            forex_map[base].append({'pair': quote, 'pct': val})
            forex_map[quote].append({'pair': base, 'pct': -val})

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
            html += f'<div class="currency-col"><div class="sep">{curr}</div>'
            for x in items:
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
# 5. SCANNER (LOGIQUE AMÉLIORÉE, VISUEL ULTIMATE)
# ==========================================
def calculate_mtf_gps(api, symbol, direction):
    # Logique GPS Simplifiée pour l'exemple (placeholder pour ta logique MTF complète)
    try:
        df = api.get_candles(symbol, "H4", 50)
        if df.empty: return 0, "N/A"
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        close = df['close'].iloc[-1]
        trend = "BUY" if close > sma50 else "SELL"
        if trend == direction: return 3.0, "A"
        return 1.5, "B"
    except: return 0, "N/A"

def run_scanner(api, matrix_data, strict_mode):
    signals = []
    scores = matrix_data['scores']
    
    bar = st.progress(0, text="🕵️ Scan M5 (Smart Logic)...")
    
    for i, sym in enumerate(ASSETS_FOREX):
        # 1. Data M5
        df = api.get_candles(sym, "M5", 100)
        if df.empty: continue
        
        # 2. Indicateurs (Nouvelle logique)
        atr = SmartIndicators.calculate_atr(df)
        rsi = SmartIndicators.calculate_rsi(df['close']).iloc[-1]
        trend_hma, _ = SmartIndicators.get_hma_trend(df['close'])
        fvg, fvg_type = SmartIndicators.detect_institutional_fvg(df, atr)
        price = df['close'].iloc[-1]
        
        # 3. Signal Trigger (Contextuel)
        signal = None
        rsi_bonus = 0
        
        if trend_hma == 1: # Tendance HMA Bull
            if 35 <= rsi <= 65: signal = "BUY"
            elif rsi < 35: signal = "BUY" # Rebond
            
        elif trend_hma == -1: # Tendance HMA Bear
            if 35 <= rsi <= 65: signal = "SELL"
            elif rsi > 65: signal = "SELL" # Rebond
            
        if not signal:
            bar.progress((i+1)/len(ASSETS_FOREX))
            continue
            
        # 4. Scoring Soft (Points)
        final_score = 5.0
        
        # Fonda
        base, quote = sym.split('_')
        s_b, s_q = scores.get(base, 5.0), scores.get(quote, 5.0)
        gap = s_b - s_q
        
        if signal == "BUY":
            if gap > 0.5: final_score += 1.5
            elif gap < -0.5: final_score -= 2.5 # Pénalité forte mais pas mortelle
        else: # SELL
            if gap < -0.5: final_score += 1.5
            elif gap > 0.5: final_score -= 2.5
            
        # FVG Bonus (Smart Money)
        if fvg:
            if (signal == "BUY" and fvg_type == "BULL") or (signal == "SELL" and fvg_type == "BEAR"):
                final_score += 1.0
        
        # RSI Quality
        if 45 <= rsi <= 55: final_score += 1.0 # Zone Momentum parfait
        
        # Volatilité check
        atr_pct = (atr / price) * 100
        if atr_pct < 0.04: final_score -= 1.0 # Trop calme
        
        # GPS
        gps_score, gps_qual = calculate_mtf_gps(api, sym, signal)
        final_score += (gps_score * 0.5) # Max +1.5
        
        final_score = min(10.0, final_score)
        
        # Filtrage final
        min_threshold = 7.0 if strict_mode else 5.5
        if final_score < min_threshold:
            bar.progress((i+1)/len(ASSETS_FOREX))
            continue
            
        # Risk Calc
        sl_pips = atr * 1.8
        tp_pips = atr * 3.0
        sl = price - sl_pips if signal == "BUY" else price + sl_pips
        tp = price + tp_pips if signal == "BUY" else price - tp_pips
        
        signals.append({
            'symbol': sym, 'type': signal, 'price': price,
            'score': final_score, 'gps': gps_qual,
            's_b': s_b, 's_q': s_q, 'gap': gap, 'rsi': rsi,
            'fvg': fvg, 'atr_pct': atr_pct, 'sl': sl, 'tp': tp
        })
        
        bar.progress((i+1)/len(ASSETS_FOREX))
    
    bar.empty()
    return sorted(signals, key=lambda x: x['score'], reverse=True)

# ==========================================
# 6. UI DISPLAY COMPONENTS (VISUEL ULTIMATE)
# ==========================================
def display_strength_card(name, current, previous):
    # EXACTEMENT LE VISUEL QUE TU AIMES
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
    # EXACTEMENT LE VISUEL DE TA CAPTURE
    is_buy = s['type'] == 'BUY'
    col = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    # Label score
    if s['score'] >= 8: label = "💎 LEGENDARY"
    elif s['score'] >= 7: label = "✅ BON"
    else: label = "⚠️ MOYEN"
    
    with st.expander(f"{s['symbol']} | {s['type']} | {label} [{s['score']:.1f}/10]", expanded=True):
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;color:white;margin-left:10px;">{s['type']}</span>
                <span style="font-size:0.8em;margin-left:10px;color:#cbd5e1;">M5 Entry</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.2f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # BADGES
        badges = []
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY</span>")
        badges.append(f"<span class='badge-gps'>🛡️ GPS {s['gps']}</span>")
        badges.append(f"<span class='badge-gps' style='background:#64748b'>RSI {s['rsi']:.1f}</span>")
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        # METRICS
        st.markdown("---")
        st.markdown("### 📊 Analyse Fondamentale")
        c1, c2, c3 = st.columns(3)
        c1.metric(s['s_b'] > s['s_q'] and "Base Forte" or "Base Faible", f"{s['s_b']:.1f}/10")
        c2.metric("Écart (Gap)", f"{s['gap']:+.2f}", delta_color="normal")
        c3.metric(s['s_q'] > s['s_b'] and "Quote Forte" or "Quote Faible", f"{s['s_q']:.1f}/10")
        
        st.markdown("### ⚖️ Risk Management")
        r1, r2, r3 = st.columns(3)
        r1.metric("Stop Loss", f"{s['sl']:.5f}", "1.8x ATR")
        r2.metric("Take Profit", f"{s['tp']:.5f}", "3.0x ATR")
        r3.metric("R:R Ratio", "1:1.67")

# ==========================================
# 7. MAIN APP
# ==========================================
def main():
    st.title("💎 BLUESTAR SNP3 ULTIMATE")
    
    with st.sidebar:
        st.header("Paramètres")
        scan_tf = st.selectbox("Analyse Timeframe", ["H1", "H4"], index=0)
        strict_mode = st.checkbox("🔥 Mode Sniper", value=False)
        st.info("Scanner Autonome (Maths + Oanda)")
    
    api = OandaClient()
    
    if st.button("🚀 LANCER L'ANALYSE GLOBALE", type="primary"):
        st.session_state.cache = {}
        
        with st.status("🔄 Analyse du marché en cours...", expanded=True) as status:
            st.write("1. Calcul Matrice Fondamentale...")
            matrix = MathCurrencySystem.process_market_data(api, granularity=scan_tf)
            st.session_state.matrix_data = matrix
            
            st.write("2. Recherche Opportunités M5 (Smart Logic)...")
            signals = run_scanner(api, matrix, strict_mode)
            
            status.update(label="✅ Analyse terminée !", state="complete", expanded=False)
        
        # 1. CARDS DASHBOARD (VISUEL VALIDÉ)
        st.markdown("### 📊 DASHBOARD FONDAMENTAL")
        scores = matrix['scores_full']
        sorted_fx = sorted(scores.keys(), key=lambda x: scores[x][0], reverse=True)
        cols = st.columns(8)
        for i, curr in enumerate(sorted_fx):
            with cols[i]:
                st.markdown(display_strength_card(curr, scores[curr][0], scores[curr][1]), unsafe_allow_html=True)
        
        if matrix['scores_special']:
            st.write("")
            sp_cols = st.columns(len(matrix['scores_special']))
            for i, (name, val) in enumerate(matrix['scores_special'].items()):
                with sp_cols[i]:
                    st.markdown(display_strength_card(name, val[0], val[1]), unsafe_allow_html=True)
        
        # 2. MARKET MAP (VISUEL VALIDÉ)
        st.write("")
        with st.expander("🗺️ Market Map & Heatmap", expanded=False):
            html_map = MathCurrencySystem.generate_map_html(matrix)
            components.html(html_map, height=250, scrolling=True)

        # 3. SIGNALS (VISUEL VALIDÉ)
        st.markdown("---")
        st.markdown(f"### 🎯 SIGNAUX DE TRADING ({len(signals)})")
        
        if not signals:
            st.warning("Aucun signal ne correspond aux critères.")
        else:
            grid = st.columns(2)
            for i, sig in enumerate(signals):
                with grid[i % 2]:
                    display_signal(sig)

if __name__ == "__main__":
    main()
