def calculate_currency_strength_score(api: OandaClient, symbol: str, direction: str) -> Dict:
    """Score Currency Strength : 0-2 points (CORRIGÉ)"""
    
    if symbol not in FOREX_PAIRS:
        return {
            'score': 0,
            'details': 'Non-Forex',
            'base_score': 0,
            'quote_score': 0,
            'rank_info': 'N/A'
        }
    
    parts = symbol.split('_')
    if len(parts) != 2:
        return {'score': 0, 'details': 'Format invalide', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    base, quote = parts[0], parts[1]
    
    try:
        strength_scores = calculate_currency_strength(api)
    except:
        return {'score': 0, 'details': 'Erreur calcul', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    if base not in strength_scores or quote not in strength_scores:
        return {'score': 0, 'details': 'Données manquantes', 'base_score': 0, 'quote_score': 0, 'rank_info': 'N/A'}
    
    base_score = strength_scores[base]
    quote_score = strength_scores[quote]
    
    # Seuil minimum de différence significative
    MIN_DIFF = 0.15  # 0.15% minimum d'écart pour valider
    
    sorted_currencies = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
    base_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == base)
    quote_rank = next(i for i, (curr, _) in enumerate(sorted_currencies, 1) if curr == quote)
    total_currencies = len(sorted_currencies)
    
    score = 0
    details = []
    
    if direction == 'BUY':
        # Pour BUY EUR_USD : on veut EUR fort (base) et USD faible (quote)
        score_diff = base_score - quote_score
        
        if base_rank <= 3 and quote_rank >= total_currencies - 2 and score_diff > MIN_DIFF:
            score = 2
            details.append(f"✅ {base} TOP3 (#{base_rank}) & {quote} BOTTOM3 (#{quote_rank})")
        elif score_diff > MIN_DIFF:
            score = 1
            details.append(f"📊 {base} > {quote} (Δ: {score_diff:+.2f}%)")
        elif abs(score_diff) <= MIN_DIFF:
            score = 0
            details.append(f"⚠️ Forces équivalentes ({base}: {base_score:.2f}% ≈ {quote}: {quote_score:.2f}%)")
        else:
            score = 0
            details.append(f"⚠️ Divergence : {quote} plus fort que {base} (Δ: {score_diff:+.2f}%)")
    
    else:  # SELL
        # Pour SELL EUR_USD : on veut USD fort (quote) et EUR faible (base)
        score_diff = quote_score - base_score
        
        if quote_rank <= 3 and base_rank >= total_currencies - 2 and score_diff > MIN_DIFF:
            score = 2
            details.append(f"✅ {quote} TOP3 (#{quote_rank}) & {base} BOTTOM3 (#{base_rank})")
        elif score_diff > MIN_DIFF:
            score = 1
            details.append(f"📊 {quote} > {base} (Δ: {score_diff:+.2f}%)")
        elif abs(score_diff) <= MIN_DIFF:
            score = 0
            details.append(f"⚠️ Forces équivalentes ({quote}: {quote_score:.2f}% ≈ {base}: {base_score:.2f}%)")
        else:
            score = 0
            details.append(f"⚠️ Divergence : {base} plus fort que {quote} (Δ: {-score_diff:+.2f}%)")
    
    rank_info = f"{base}:#{base_rank} vs {quote}:#{quote_rank}"
    
    return {
        'score': score,
        'details': ' | '.join(details),
        'base_score': base_score,
        'quote_score': quote_score,
        'rank_info': rank_info
    }
