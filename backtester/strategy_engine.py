def ema_crossover_strategy(data):

    data['EMA20'] = ema(data, 20)
    data['EMA50'] = ema(data, 50)

    data['Signal'] = 0
    data.loc[data['EMA20'] > data['EMA50'], 'Signal'] = 1

    data['Position'] = data['Signal'].diff()

    return data