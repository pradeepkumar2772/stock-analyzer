import pandas_ta as ta

class StrategyFactory:
    @staticmethod
    def rsi_60_cross(df, config):
        df['rsi'] = ta.rsi(df['close'], length=14)
        long_signal = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        exit_signal = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
        return long_signal, exit_signal

    @staticmethod
    def ema_ribbon(df, config):
        # Using config to allow user-defined EMA lengths
        fast = config.get('ema_fast', 20)
        slow = config.get('ema_slow', 50)
        exit_p = config.get('ema_exit', 30)
        
        df['ema_f'] = ta.ema(df['close'], length=fast)
        df['ema_s'] = ta.ema(df['close'], length=slow)
        df['ema_e'] = ta.ema(df['close'], length=exit_p)
        
        long_signal = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
        exit_signal = (df['ema_f'] < df['ema_e']) & (df['ema_f'].shift(1) >= df['ema_e'].shift(1))
        return long_signal, exit_signal