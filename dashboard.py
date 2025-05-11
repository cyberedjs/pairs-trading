import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from live_live import zscore_data

st.set_page_config(page_title="Trading Pairs Monitor", layout="wide")

def create_zscore_plot(pair, timestamps, zscores, zscore_st):
    """z-score와 zscore_st를 시각화하는 함수"""
    fig = go.Figure()
    
    # z-score 라인
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=zscores,
        name='z-score',
        line=dict(color='blue')
    ))
    
    # zscore_st 라인
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=zscore_st,
        name='zscore_st',
        line=dict(color='red')
    ))
    
    # 진입/청산 임계값 라인
    fig.add_hline(y=2, line_dash="dash", line_color="green", name="Long Entry")
    fig.add_hline(y=-2, line_dash="dash", line_color="green", name="Short Entry")
    fig.add_hline(y=1, line_dash="dash", line_color="orange", name="Long Exit")
    fig.add_hline(y=-1, line_dash="dash", line_color="orange", name="Short Exit")
    
    fig.update_layout(
        title=f"Z-Score Monitor - {pair}",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )
    
    return fig

def main():
    st.title("Trading Pairs Monitor")
    
    # 사이드바에 현재 시간 표시
    st.sidebar.title("Status")
    st.sidebar.write(f"Last Update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # 활성화된 페어 목록 표시
    if zscore_data['pairs']:
        st.sidebar.subheader("Active Pairs")
        for pair in zscore_data['pairs'].keys():
            st.sidebar.write(f"- {pair}")
    
    # 메인 대시보드
    if zscore_data['timestamps']:
        # 각 페어별 차트 생성
        for pair in zscore_data['pairs'].keys():
            if pair in zscore_data['pairs'] and pair in zscore_data['zscore_st']:
                fig = create_zscore_plot(
                    pair,
                    zscore_data['timestamps'],
                    zscore_data['pairs'][pair],
                    zscore_data['zscore_st'][pair]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 현재 값 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Current Z-Score",
                        f"{zscore_data['pairs'][pair][-1]:.2f}",
                        delta=f"{zscore_data['pairs'][pair][-1] - zscore_data['pairs'][pair][-2]:.2f}"
                    )
                with col2:
                    st.metric(
                        "Current Z-Score ST",
                        f"{zscore_data['zscore_st'][pair][-1]:.2f}",
                        delta=f"{zscore_data['zscore_st'][pair][-1] - zscore_data['zscore_st'][pair][-2]:.2f}"
                    )
    else:
        st.info("Waiting for data...")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(5)  # 5초마다 업데이트 