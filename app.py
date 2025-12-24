import streamlit as st
import pandas as pd
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import logging
from datetime import datetime
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIG & STYLE ( inchangé )
# ==========================================
st.set_page_config(page_title="Bluestar SNP3 Institutional", layout="wide", page_icon="💎")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #0f1117; }
    
    /* CARDS */
    .currency-card {
        background-color: #1e293b; border-radius: 10px; padding: 12px;
        margin-bottom: 8px; border: 1px solid #334155; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .strength-score { font-size: 1.8rem; font-weight: 800; margin: 5px 0; color: #f8fafc; }
    .progress-bg { background-color: #0f172a; height: 6px; border-radius: 3px; width: 100%; margin-top: 8px; }
    .progress-fill { height: 100%; border-radius: 3px; }
    
    /* COLORS */
    .text-green { color: #10B981; } .bg-green { background-color: #10B981; }
    .text-red { color: #EF4444; } .bg-red { background-color: #EF4444; }
    .text-blue { color: #3B82F6; } .bg-blue { background-color: #3B82F6; }
    .text-orange { color: #F59E0B; } .bg-orange { background-color: #F59E0B; }
    
    /* BADGES */
    .badge-fvg { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: bold; }
    .badge-gps { background: #334155; color: #e2e8f0; padding: 3px 8px; border-radius: 4px; font-size: 0.7em; font-weight: bold; }
    
    .stButton>button { background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%); color: white; border: none; font-weight: bold; }
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
ASSETS_SPECIAL = {'XAU_USD': 'GOLD', 'US30_USD': 'US30'}

# ==========================================
# 2. CLIENT API & DATA
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
        # Cache 60s
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
# 3. MOTEUR "TUNED" (INDICATEURS)
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
    def get_hma_trend(series, length=20):
        # Hull Moving Average
        wma1 = series.rolling(int(length/2)).mean() # Simplifié pour la démo, idéalement WMA pondérée
        wma2 = series.rolling(length).mean()
        raw_hma = 2 * wma1 - wma2
        hma = raw_hma.rolling(int(np.sqrt(length))).mean()
        
        curr = hma.iloc[-1]
        prev = hma.iloc[-2]
        trend = 1 if curr > prev else -1
        return trend, curr

    @staticmethod
    def detect_institutional_fvg(df, atr):
        """
        Détection FVG 'Smart Money' :
        1. Cherche un gap (High < Low.shift(2) ou Low > High.shift(2))
        2. Le gap doit être > 30% de l'ATR (filtre le bruit)
        3. Retourne True si un FVG significatif est présent sur les 3 dernières bougies
        """
        if len(df) < 5: return False, None
        
        for i in range(1, 4): # Scan des 3 dernières bougies closes
            high_prev = df['high'].iloc[-(i+2)]
            low_curr = df['low'].iloc[-i]
            
            low_prev = df['low'].iloc[-(i+2)]
            high_curr = df['high'].iloc[-i]
            
            # Bullish FVG
            if low_curr > high_prev:
                gap_size = low_curr - high_prev
                if gap_size > (atr * 0.3): # Filtre ATR
                    return True, "BULL"
            
            # Bearish FVG
            if high_curr < low_prev:
                gap_size = low_prev - high_curr
                if gap_size > (atr * 0.3): # Filtre ATR
                    return True, "BEAR"
                    
        return False, None

# ==========================================
# 4. LOGIQUE SCANNER & SCORING "SOFT"
# ==========================================
def calculate_matrix_scores(api):
    # Calcul simplifié mathématique (identique à v3.0)
    prices = {}
    for sym in ASSETS_FOREX:
        df = api.get_candles(sym, "H1", 50) # H1 pour la tendance fonda
        if not df.empty: prices[sym] = df['close']
    
    if not prices: return {}
    df_p = pd.DataFrame(prices).ffill().bfill()
    
    scores = {}
    currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]
    for curr in currencies:
        vals = []
        for opp in currencies:
            if curr == opp: continue
            pair = f"{curr}_{opp}"
            if pair in df_p.columns:
                rsi = SmartIndicators.calculate_rsi(df_p[pair]).iloc[-1]
                vals.append(((rsi-50)/50+1)*5) # Norm 0-10
            elif f"{opp}_{curr}" in df_p.columns:
                rsi = SmartIndicators.calculate_rsi(1/df_p[f"{opp}_{curr}"]).iloc[-1]
                vals.append(((rsi-50)/50+1)*5)
        scores[curr] = np.mean(vals) if vals else 5.0
    return scores

def run_smart_scanner(api, scores, strict_mode):
    signals = []
    
    progress_text = "🕵️ Scan Institutionnel (M5)..."
    bar = st.progress(0, text=progress_text)
    
    for i, sym in enumerate(ASSETS_FOREX):
        # 1. DATA M5
        df = api.get_candles(sym, "M5", 100)
        if df.empty: continue
        
        # 2. INDICATEURS TECHNIQUES
        atr = SmartIndicators.calculate_atr(df)
        rsi_series = SmartIndicators.calculate_rsi(df['close'])
        rsi = rsi_series.iloc[-1]
        trend_hma, hma_val = SmartIndicators.get_hma_trend(df['close'])
        fvg_detected, fvg_type = SmartIndicators.detect_institutional_fvg(df, atr)
        
        price = df['close'].iloc[-1]
        
        # 3. DÉTECTION SIGNAL (Plus souple)
        # On cherche une concordance HMA + RSI
        signal_type = None
        rsi_quality = 0 # 0-1
        
        # BUY LOGIC
        if trend_hma == 1:
            # RSI "Smart Context": On accepte tout ce qui n'est pas suracheté extrême (>80)
            # Idéalement entre 40 et 65 pour une entrée
            if 35 <= rsi <= 65:
                signal_type = "BUY"
                rsi_quality = 1.0 if 45 <= rsi <= 55 else 0.7
            elif rsi < 35: # Rebond survendu dans tendance haussière
                signal_type = "BUY"
                rsi_quality = 0.8
        
        # SELL LOGIC
        elif trend_hma == -1:
            if 35 <= rsi <= 65:
                signal_type = "SELL"
                rsi_quality = 1.0 if 45 <= rsi <= 55 else 0.7
            elif rsi > 65: # Rebond suracheté dans tendance baissière
                signal_type = "SELL"
                rsi_quality = 0.8
                
        if not signal_type:
            bar.progress((i+1)/len(ASSETS_FOREX))
            continue

        # 4. SCORING SYSTÉMIQUE (Le cœur du tuning)
        final_score = 5.0 # Base neutre
        
        # A. Fondamental (Matrix)
        base, quote = sym.split('_')
        s_b = scores.get(base, 5.0)
        s_q = scores.get(quote, 5.0)
        gap = s_b - s_q
        
        # Scoring dynamique (pas de filtre bloquant, mais grosse pénalité si contre-sens)
        if signal_type == "BUY":
            if gap > 1.5: final_score += 2.5 # Fort support
            elif gap > 0.5: final_score += 1.5 # Support modéré
            elif gap < -0.5: final_score -= 3.0 # Contre tendance fonda majeure !
        else: # SELL
            if gap < -1.5: final_score += 2.5
            elif gap < -0.5: final_score += 1.5
            elif gap > 0.5: final_score -= 3.0

        # B. Technique (Bonus FVG)
        # Le FVG est un BONUS, pas une condition requise
        if fvg_detected:
            # On vérifie que le FVG est dans le sens du trade
            if (signal_type == "BUY" and fvg_type == "BULL") or \
               (signal_type == "SELL" and fvg_type == "BEAR"):
                final_score += 1.0 # Bonus Smart Money
        
        # C. Qualité RSI
        final_score += (rsi_quality * 1.5) # Max +1.5 points
        
        # D. Volatilité (ATR %)
        atr_pct = (atr / price) * 100
        vol_score = 0
        if 0.05 <= atr_pct <= 0.5: vol_score = 1.0 # Volatilité saine M5
        elif atr_pct < 0.05: vol_score = -1.0 # Trop mou (Dead market)
        elif atr_pct > 0.5: vol_score = -0.5 # Trop violent (News?)
        final_score += vol_score

        # E. Ajustement Strict Mode
        # Si activé, on coupe tout ce qui est sous 7.0
        if strict_mode and final_score < 7.0:
            continue
        elif not strict_mode and final_score < 5.0: # Filtre de base
            continue
            
        # Plafond 10
        final_score = min(10.0, max(0.0, final_score))
        
        # Risk Mgmt
        sl_pips = (atr * 1.5)
        tp_pips = (atr * 3.0)
        sl = price - sl_pips if signal_type == "BUY" else price + sl_pips
        tp = price + tp_pips if signal_type == "BUY" else price - tp_pips
        
        signals.append({
            'symbol': sym, 'type': signal_type, 'price': price,
            'score': final_score, 'atr_pct': atr_pct,
            'fvg': fvg_detected, 'rsi': rsi,
            'sl': sl, 'tp': tp,
            'cs': {'base': base, 's_b': s_b, 'quote': quote, 's_q': s_q, 'gap': gap}
        })
        
        bar.progress((i+1)/len(ASSETS_FOREX))
    
    bar.empty()
    return sorted(signals, key=lambda x: x['score'], reverse=True)

# ==========================================
# 5. UI & EXECUTION
# ==========================================
def display_signal_card(s):
    is_buy = s['type'] == "BUY"
    col = "#10b981" if is_buy else "#ef4444"
    bg_grad = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    score_col = "#10b981" if s['score'] >= 8 else "#3b82f6" if s['score'] >= 6 else "#f59e0b"
    
    with st.expander(f"{s['symbol']} | {s['type']} | Score: {s['score']:.1f}/10", expanded=True):
        # Header visuel
        st.markdown(f"""
        <div style="background:{bg_grad};padding:12px;border-radius:8px;border:1px solid {col};display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.4em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 6px;border-radius:4px;color:white;font-size:0.8em;margin-left:8px;">{s['type']}</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.2em;color:white;font-weight:bold;">{s['price']:.5f}</div>
                <div style="font-size:0.7em;color:#cbd5e1;">Vol: {s['atr_pct']:.3f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Badges
        badges = []
        if s['fvg']: badges.append("<span class='badge-fvg'>🦅 SMART MONEY GAP</span>")
        if 40 <= s['rsi'] <= 60: badges.append("<span class='badge-gps'>⚡ MOMENTUM OPTIMAL</span>")
        
        if badges:
            st.markdown(f"<div style='margin-top:8px;text-align:center;'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        # Métriques
        c1, c2, c3 = st.columns(3)
        c1.metric("Score", f"{s['score']:.1f}/10")
        c2.metric("Gap Fonda", f"{s['cs']['gap']:+.2f}")
        c3.metric("RSI M5", f"{s['rsi']:.1f}")
        
        # Détail Matrix
        base, quote = s['cs']['base'], s['cs']['quote']
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);padding:8px;border-radius:5px;margin-top:5px;font-size:0.85em;display:flex;justify-content:space-between;">
            <span>{base}: <b>{s['cs']['s_b']:.1f}</b></span>
            <span>vs</span>
            <span>{quote}: <b>{s['cs']['s_q']:.1f}</b></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk
        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.markdown(f"<div style='color:#ef4444;text-align:center;font-size:0.9em'><b>SL:</b> {s['sl']:.5f}</div>", unsafe_allow_html=True)
        r2.markdown(f"<div style='color:#10b981;text-align:center;font-size:0.9em'><b>TP:</b> {s['tp']:.5f}</div>", unsafe_allow_html=True)

def main():
    st.title("💎 BLUESTAR SNP3 - INSTITUTIONAL TUNED")
    
    with st.sidebar:
        st.header("Paramètres")
        strict_mode = st.checkbox("🔥 Mode Sniper Strict (Score > 7)", value=False)
        st.info("Le mode strict ne montre que les opportunités 'A+'. Désactivez-le pour voir les setups 'B' (Score > 5).")
    
    api = OandaClient()
    
    if st.button("🚀 SCAN INSTITUTIONNEL", type="primary"):
        # 1. Calcul Matrix (Fonda)
        with st.spinner("📊 Calcul de la force des devises (H1)..."):
            scores = calculate_matrix_scores(api)
            
            # Affichage mini dashboard haut
            if scores:
                top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
                flop_3 = sorted(scores.items(), key=lambda x: x[1])[:4]
                
                c1, c2 = st.columns(2)
                with c1: 
                    st.caption("💪 Devises Fortes")
                    st.write(" ".join([f"**{k}** {v:.1f} |" for k,v in top_3]))
                with c2: 
                    st.caption("🩸 Devises Faibles")
                    st.write(" ".join([f"**{k}** {v:.1f} |" for k,v in flop_3]))
        
        # 2. Scan M5
        results = run_smart_scanner(api, scores, strict_mode)
        
        st.markdown(f"### 🎯 Opportunités Détectées ({len(results)})")
        
        if not results:
            st.warning("Aucun setup ne correspond aux critères actuels. Le marché est peut-être en range ou trop calme.")
        else:
            grid = st.columns(2)
            for i, sig in enumerate(results):
                with grid[i % 2]:
                    display_signal_card(sig)

if __name__ == "__main__":
    main()
