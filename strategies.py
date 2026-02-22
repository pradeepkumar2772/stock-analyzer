import pandas_ta as ta

class StrategyFactory:
    @staticmethod
    def rsi_60_cross(df, config):
        # We use .get() to ensure we pick the correct column even if yfinance naming varies
        # pandas-ta automatically looks for 'close' (lowercase)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Entry: RSI crosses 60 from below | Exit: RSI crosses 60 from above
        long_signal = (df['rsi'] > 60) & (df['rsi'].shift(1) <= 60)
        exit_signal = (df['rsi'] < 60) & (df['rsi'].shift(1) >= 60)
        return long_signal, exit_signal

    @staticmethod
    def ema_ribbon(df, config):
        fast = config.get('ema_fast', 20)
        slow = config.get('ema_slow', 50)
        exit_p = config.get('ema_exit', 30)
        
        df['ema_f'] = ta.ema(df['close'], length=fast)
        df['ema_s'] = ta.ema(df['close'], length=slow)
        df['ema_e'] = ta.ema(df['close'], length=exit_p)
        
        # Entry: Fast EMA crosses above Slow EMA
        long_signal = (df['ema_f'] > df['ema_s']) & (df['ema_f'].shift(1) <= df['ema_s'].shift(1))
        # Exit: Fast EMA crosses below Exit EMA
        exit_signal = (df['ema_f'] < df['ema_e']) & (df['ema_f'].shift(1) >= df['ema_e'].shift(1))
        return long_signal, exit_signal