import pandas as pd
import numpy as np

class AnalyticsEngine:
    @staticmethod
    def calculate_all_metrics(df_trades, capital, duration):
        wins = df_trades[df_trades['pnl_pct'] > 0]
        losses = df_trades[df_trades['pnl_pct'] <= 0]
        
        # Section 2 & 3: Returns & Drawdown
        total_ret = (df_trades['equity'].iloc[-1] / capital - 1) * 100
        years = max(duration.days / 365.25, 0.1)
        cagr = (((df_trades['equity'].iloc[-1] / capital) ** (1/years)) - 1) * 100
        
        peak = df_trades['equity'].cummax()
        drawdown = (df_trades['equity'] - peak) / peak
        mdd = drawdown.min() * 100
        
        # Section 6: Risk Adjusted
        sharpe = (df_trades['pnl_pct'].mean() / df_trades['pnl_pct'].std() * np.sqrt(252)) if len(df_trades) > 1 else 0
        calmar = abs(cagr / mdd) if mdd != 0 else 0
        
        # Section 8: Streaks
        pnl_bool = (df_trades['pnl_pct'] > 0).astype(int)
        streaks = pnl_bool.groupby((pnl_bool != pnl_bool.shift()).cumsum()).cumcount() + 1
        
        return {
            "total_ret": total_ret, "cagr": cagr, "mdd": mdd,
            "avg_ret": df_trades['pnl_pct'].mean() * 100,
            "win_rate": (len(wins)/len(df_trades)*100),
            "loss_rate": (len(losses)/len(df_trades)*100),
            "sharpe": sharpe, "calmar": calmar,
            "max_w_s": streaks[pnl_bool == 1].max() if 1 in pnl_bool.values else 0,
            "max_l_s": streaks[pnl_bool == 0].max() if 0 in pnl_bool.values else 0,
            "drawdown_series": drawdown,
            "total_trades": len(df_trades),
            "win_trades": len(wins),
            "loss_trades": len(losses)
        }