import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# --- CORE ENGINE ---
def run_backtest(df, config):
    trades = []
    active_trade = None
    
    # Calculate Indicators
    df['ema_fast'] = df['close'].ewm(span=config['ema_f'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_s'], adjust=False).mean()
    
    # ATR for Trailing Stop
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    highest_high = 0

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        if active_trade:
            # Update Trailing Stop based on Highest High since entry
            highest_high = max(highest_high, current['high'])
            trailing_stop = highest_high - (current['atr'] * config['atr_mult'])
            
            # Exit Logic: Signal or Trailing Stop Hit
            if current['low'] <= trailing_stop or prev['ema_fast'] < prev['ema_slow']:
                active_trade['exit_price'] = min(current['open'], trailing_stop)
                active_trade['exit_date'] = current.name
                active_trade['pnl'] = (active_trade['exit_price'] - active_trade['entry_price']) / active_trade['entry_price']
                trades.append(active_trade)
                active_trade = None
                highest_high = 0
        
        # Entry Logic: EMA Cross
        elif prev['ema_fast'] > prev['ema_slow']:
            active_trade = {
                'entry_price': current['open'],
                'entry_date': current.name
            }
            highest_high = current['high']
            
    return pd.DataFrame(trades)

# --- UI ---
st.set_page_config(layout="wide")
st.sidebar.title("🛠️ Pro-Tracer Optimizer")

symbol = st.sidebar.text_input("Stock Symbol", value="BRITANNIA.NS")
mode = st.sidebar.radio("Mode", ["Single Run", "Optimize Parameters"])

if mode == "Optimize Parameters":
    st.header(f"🔍 Optimizing {symbol}")
    
    # Define ranges to test
    ema_ranges = st.sidebar.slider("EMA Fast Range", 5, 50, (10, 30))
    atr_ranges = st.sidebar.slider("ATR Multiplier Range", 1.0, 5.0, (2.0, 4.0))

    if st.button("Start Optimization"):
        data = yf.download(symbol, start="2020-01-01", auto_adjust=True)
        data.columns = [col.lower() for col in data.columns]
        
        results = []
        # Optimization Loop
        for ema_f in range(ema_ranges[0], ema_ranges[1], 5):
            for mult in np.arange(atr_ranges[0], atr_ranges[1], 0.5):
                conf = {'ema_f': ema_f, 'ema_s': 50, 'atr_mult': mult, 'use_sl': False}
                res_df = run_backtest(data.copy(), conf)
                
                if not res_df.empty:
                    total_ret = (res_df['pnl'] + 1).prod() - 1
                    results.append({
                        'EMA Fast': ema_f,
                        'ATR Mult': mult,
                        'Total Return %': total_ret * 100,
                        'Trade Count': len(res_df)
                    })
        
        opt_df = pd.DataFrame(results).sort_values('Total Return %', ascending=False)
        st.write("### Optimization Results")
        st.dataframe(opt_df.style.highlight_max(axis=0, subset=['Total Return %']))
        
        # Visualize
        fig = px.scatter(opt_df, x='EMA Fast', y='ATR Mult', size='Total Return %', color='Total Return %', 
                         title="Heatmap: Best Parameter Combinations")
        st.plotly_chart(fig)

else:
    st.info("Select 'Optimize Parameters' in the sidebar to find the best settings for this stock.")