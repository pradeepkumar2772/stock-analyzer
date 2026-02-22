def strategy_score(win_rate, return_pct, max_dd):

    score = (win_rate * 0.4) + (return_pct * 0.4) - (abs(max_dd) * 0.2)

    if score > 50:
        return "Strong Strategy"
    elif score > 25:
        return "Moderate Strategy"
    else:
        return "Avoid"