import time  # <--- AJOUTER CECI TOUT EN HAUT DU FICHIER

# ... (Le reste du code ne change pas jusqu'à la fonction run_scan) ...

# ==========================================
# SCANNER PRINCIPAL (CORRIGÉ ANTI-FREEZE)
# ==========================================
def run_scan_v32(api, min_prob, strict_mode):
    cs_scores = get_currency_strength_rsi(api)
    signals = []
    
    # Création des éléments d'interface
    progress_bar = st.progress(0)
    status_text = st.empty() # Texte qui va changer dynamiquement
    
    for i, sym in enumerate(ASSETS):
        # Mise à jour de la barre et du texte
        progress = (i + 1) / len(ASSETS)
        progress_bar.progress(progress)
        status_text.markdown(f"⏳ Analyse en cours : **{sym}** ({i+1}/{len(ASSETS)})")
        
        # --- PROTECTION ANTI-FREEZE ---
        # On évite de spammer l'actif s'il a été scanné il y a moins de 5 min
        if sym in st.session_state.signal_history:
            if (datetime.now() - st.session_state.signal_history[sym]).total_seconds() < 300: 
                time.sleep(0.05) # Petite pause quand même
                continue
        
        try:
            # 1. OPTIMISATION : On récupère Daily et Weekly en une seule fois (300 bougies D)
            # Ça économise 30 requêtes API au total
            df_d_raw = api.get_candles(sym, "D", 300)
            
            # Pause de sécurité pour ne pas bloquer l'API Oanda
            time.sleep(0.15) 
            
            df_m5 = api.get_candles(sym, "M5", 288)
            time.sleep(0.15) # Pause entre les requêtes
            
            df_h4 = api.get_candles(sym, "H4", 100)
            
            if df_m5.empty or df_h4.empty or df_d_raw.empty: 
                continue
            
            # Préparation des DataFrames D et W à partir de la même source
            df_d = df_d_raw.iloc[-100:].copy() # Les 100 derniers jours
            
            # Reconstruction Weekly
            df_w = df_d_raw.set_index('time').resample('W-FRI').agg({
                'open':'first', 'high':'max', 'low':'min', 'close':'last'
            }).dropna().reset_index()
            
            # Détection Basique RSI (Trigger)
            rsi_serie = QuantEngine.calculate_rsi(df_m5)
            if len(rsi_serie) < 3: continue
            
            rsi_mom = rsi_serie.iloc[-1] - rsi_serie.iloc[-2]
            scan_direction = None
            
            # Logique Trigger
            if rsi_serie.iloc[-2] < 50 and rsi_serie.iloc[-1] >= 50 and rsi_mom > 0.5:
                scan_direction = "BUY"
            elif rsi_serie.iloc[-2] > 50 and rsi_serie.iloc[-1] <= 50 and rsi_mom < -0.5:
                scan_direction = "SELL"
            
            if not scan_direction: continue
            
            # Calcul Probabilité
            prob, details, atr_pct = calculate_signal_probability(
                df_m5, df_h4, df_d, df_w, sym, scan_direction
            )
            
            if prob < min_prob: continue
            
            # Mode Strict
            if strict_mode:
                if details['mtf_bias'] == "NEUTRAL": continue
            
            # Filtre Corrélation
            temp_signal_obj = {'symbol': sym, 'type': scan_direction}
            if check_dynamic_correlation_conflict(temp_signal_obj, signals, cs_scores):
                continue
            
            # Filtre CS
            cs_aligned = False
            if "_" in sym:
                base, quote = sym.split('_')
                if cs_scores and base in cs_scores and quote in cs_scores:
                    gap = cs_scores.get(base, 0) - cs_scores.get(quote, 0)
                    if scan_direction == "BUY" and gap > 0: cs_aligned = True
                    elif scan_direction == "SELL" and gap < 0: cs_aligned = True
            elif "XAU" in sym or "US30" in sym:
                cs_aligned = True 

            price = df_m5['close'].iloc[-1]
            atr = QuantEngine.calculate_atr(df_m5)
            params = get_asset_params(sym)
            
            sl_mult = params['sl_base']
            if details['structure_z'] != 0: sl_mult -= 0.2
            sl = price - (atr * sl_mult) if scan_direction == "BUY" else price + (atr * sl_mult)
            tp = price + (atr * params['tp_rr']) if scan_direction == "BUY" else price - (atr * params['tp_rr'])
            
            signals.append({
                'symbol': sym,
                'type': scan_direction,
                'price': price,
                'prob': prob,
                'score_display': prob * 10,
                'details': details,
                'atr_pct': atr_pct,
                'exact_time': datetime.now().strftime("%H:%M:%S"),
                'sl': sl,
                'tp': tp,
                'rr': params['tp_rr'],
                'cs_aligned': cs_aligned
            })
            
            st.session_state.signal_history[sym] = datetime.now()
            
        except Exception as e:
            # En cas d'erreur sur un actif, on l'affiche dans la console (pas sur l'app) et on passe au suivant
            print(f"Erreur sur {sym}: {e}")
            time.sleep(1) # Pause de sécurité en cas d'erreur
            continue
            
    progress_bar.empty()
    status_text.empty()
    return sorted(signals, key=lambda x: x['prob'], reverse=True)
     
