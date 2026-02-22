import plotly.graph_objects as go
import streamlit as st

class VisualLibrary:
    @staticmethod
    def render_charts(df_trades):
        # Yearly Bar
        yearly = df_trades.groupby(df_trades['exit_date'].dt.year)['pnl_pct'].sum() * 100
        fig1 = go.Figure(data=[go.Bar(x=yearly.index, y=yearly.values, text=yearly.values.round(1), 
                        texttemplate='%{text}%', textposition='outside', marker_color='#3498db')])
        st.plotly_chart(fig1.update_layout(title="1. Return by Period (%)", template="plotly_dark"), use_container_width=True)
        
        # Exits Bar
        exits = df_trades['exit_reason'].value_counts()
        fig2 = go.Figure(data=[go.Bar(x=exits.index, y=exits.values, marker_color='#2ecc71')])
        st.plotly_chart(fig2.update_layout(title="2. Exits Distribution", template="plotly_dark"), use_container_width=True)