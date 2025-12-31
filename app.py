# ==========================================
# AFFICHAGE SIGNAL (V3.4 - UI)
# ==========================================
def display_sig(s):
    is_buy = s['type'] == 'BUY'
    col_type = "#10b981" if is_buy else "#ef4444"
    bg = "linear-gradient(90deg, #064e3b 0%, #065f46 100%)" if is_buy else "linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%)"
    
    sc = s['score_display']
    mid_status = s['details']['midnight_status']
    
    # Ajustement des labels de qualité
    if sc >= 8.0: label, q_badge = "💎 INSTITUTIONAL", "quality-high"
    elif sc >= 7.0: label, q_badge = "⭐ ALGORITHMIC", "quality-high"
    elif sc >= 6.0: label, q_badge = "✅ STRATEGIC", "quality-medium"
    else: label, q_badge = "📊 TACTICAL", "quality-medium"

    with st.expander(f"{s['symbol']}  |  {s['type']}  |  {label}  [{sc:.1f}/10]", expanded=True):
        st.markdown(f"<div class='timestamp-box'>⚡ SIGNAL : {s['exact_time']}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:{bg};padding:15px;border-radius:8px;border:2px solid {col_type};
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8em;font-weight:900;color:white;">{s['symbol']}</span>
                <span style="background:rgba(255,255,255,0.2);padding:2px 8px;border-radius:4px;
                            color:white;margin-left:10px;">{s['type']}</span>
                <span class="quality-indicator {q_badge}">{int(s['prob']*100)}% CONF</span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.4em;font-weight:bold;color:white;">{s['price']:.5f}</div>
                <div style="font-size:0.75em;color:#cbd5e1;">ATR: {s['atr_pct']:.3f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # --- BADGES ---
        badges = []
        
        # Badge Midnight Informatif
        if "OPTIMAL" in mid_status:
            # C'est le top : Bleu/Violet pour signifier "Smart Money"
            badges.append(f"<span class='badge badge-midnight'>🌑 {mid_status}</span>")
        else:
            # C'est standard : Couleur neutre ou attention
            # On informe l'utilisateur qu'il n'est pas en zone optimale mais que le trade reste valide
            badges.append(f"<span class='badge' style='background:#475569;border:1px solid #64748b'>🌗 {mid_status}</span>")
            
        adx = s['details'].get('adx_val', 0)
        if adx >= 25: badges.append(f"<span class='badge badge-trend'>ADX FORT ({int(adx)})</span>")
        
        badges.append(f"<span class='badge badge-regime'>{s['details']['mtf_bias']}</span>")
        
        if s['cs_aligned']: badges.append("<span class='badge badge-blue'>CS ALIGNÉ</span>")
        if s['details']['fvg_align']: badges.append("<span class='badge badge-gold'>FVG ACTIF</span>")
        
        st.markdown(f"<div style='margin-top:10px;text-align:center'>{' '.join(badges)}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        c1, c2, c3 = st.columns(3)
        # Affichage du prix Midnight dans les détails pour référence
        mid_price_disp = s['details']['midnight_val']
        c1.metric("Midnight Open", f"{mid_price_disp:.5f}" if mid_price_disp else "N/A")
        c2.metric("RSI Mom.", f"{s['details']['rsi_mom']:.1f}")
        c3.metric("Z-Score", f"{s['details']['structure_z']:.1f}")

        st.write("")
        r1, r2 = st.columns(2)
        r1.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>STOP LOSS</div>
            <div style='color:#ef4444;font-weight:bold;font-size:1.2em;'>{s['sl']:.5f}</div>
        </div>""", unsafe_allow_html=True)
        r2.markdown(f"""<div class='risk-box'>
            <div style='color:#94a3b8;font-size:0.8em;'>TAKE PROFIT (1:{s['rr']})</div>
            <div style='color:#10b981;font-weight:bold;font-size:1.2em;'>{s['tp']:.5f}</div>
        </div>""", unsafe_allow_html=True)
