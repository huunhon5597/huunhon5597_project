import sys
import os
import time
import json
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to the path to allow imports from modules
sys.path.insert(0, os.path.dirname(__file__))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_echarts import st_echarts
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta


# --- Background Job Management ---
if "executor" not in st.session_state:
    st.session_state.executor = ThreadPoolExecutor(max_workers=30)  # Increased for maximum parallel data loading
if "jobs" not in st.session_state:
    st.session_state.jobs = {}  # Stores submitted futures {key: future}

# No need to track shown_success - we just show data without success messages


def get_date_range(period):
    """Get date range for a given period. Uses date only (no time) for consistency."""
    end_date = datetime.now().date()
    days_map = {"1 tháng": 30, "3 tháng": 90, "6 tháng": 180, "1 năm": 365, "2 năm": 730}
    start_date = end_date - timedelta(days=days_map.get(period, 0))
    return start_date, end_date

def navigate_to(menu_name):
    """Navigate to a specific menu using query params."""
    st.query_params._nav_menu = menu_name
    st.rerun()

def render_main_navigation():
    """Render main menu navigation buttons."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏠 Trang chủ", key="nav_home"):
            navigate_to("Trang chủ")
    with col2:
        if st.button("📈 Thị trường", key="nav_market"):
            navigate_to("Thị trường")
    with col3:
        if st.button("💹 Cổ phiếu", key="nav_stock"):
            navigate_to("Cổ phiếu")
    with col4:
        if st.button("🧪 Test", key="nav_test"):
            navigate_to("Test")

def render_submenu_navigation(submenu_key, submenu_options):
    """Render submenu navigation buttons based on current selection."""
    if not submenu_options:
        return
    
    # Create columns for submenu buttons
    cols = st.columns(len(submenu_options))
    for i, (submenu_name, submenu_value) in enumerate(submenu_options.items()):
        with cols[i]:
            # Get icon based on submenu name
            icon = "📊"  # default
            if "Tâm lý" in submenu_name:
                icon = "🧠"
            elif "Phân loại nhà đầu tư" in submenu_name:
                icon = "👥"
            elif "Định giá" in submenu_name:
                icon = "💰"
            elif "Phân loại giao dịch" in submenu_name:
                icon = "🔄"
            
            # Check if current submenu is selected
            is_active = st.session_state.get(submenu_key) == submenu_value
            button_type = "primary" if is_active else "secondary"
            
            if st.button(f"{icon} {submenu_name}", key=f"subnav_{submenu_value}"):
                st.session_state[submenu_key] = submenu_value
                st.rerun()

def get_job_status(key):
    """
    Checks the status of a job.
    Returns: (status, result) where status is one of
             ["not_started", "running", "completed", "error"]
    Note: This function stores job results in session_state when complete.
    """
    # Check if job is running
    if key in st.session_state.jobs:
        future = st.session_state.jobs[key]
        if future.done():
            try:
                result = future.result()
                st.session_state[key] = result
            except Exception as e:
                st.session_state[key] = e
            finally:
                del st.session_state.jobs[key]  # Job is done, remove it
        else:
            return "running", None

    # Check if we have cached result
    if key in st.session_state:
        result = st.session_state[key]
        return ("error", result) if isinstance(result, Exception) else ("completed", result)

    return "not_started", None


# --- UI & App Logic ---
# Force clear session state to ensure fresh rendering
if "annotations_force_refresh" not in st.session_state:
    st.session_state.annotations_force_refresh = True

st.set_page_config(page_title="Market Sentiment", layout="wide")

# Custom CSS for percentage color coding and metric display
st.markdown("""
<style>
/* Fix metric truncation issues */
div[data-testid="metric-container"] {
    min-width: 200px !important;
}

/* Fix value display without truncation */
div[data-testid="metric-container"] > div[data-testid="baseMain"] > div[data-testid="metricValue"] > span {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: none !important;
}

/* Color coding for positive values (green) */
div[data-testid="metric-container"] > div[data-testid="deltaRow"] > div[data-testid="metricDelta"] > span {
    color: #51cf66 !important; /* Green for positive values */
}

/* Color coding for negative values (red) */
div[data-testid="metric-container"] > div[data-testid="deltaRow"] > div[data-testid="metricDelta"] > span.negative {
    color: #ff6b6b !important; /* Red for negative values */
}

/* Remove delta arrow */
div[data-testid="metric-container"] > div[data-testid="deltaRow"] > div[data-testid="metricDelta"] > span:first-child::after {
    content: "";
}
</style>
""", unsafe_allow_html=True)

# --- Cached wrapper functions for market sentiment (defined at top level to avoid redefinition) ---
import concurrent.futures

def _run_with_timeout(func, args=(), kwargs=None, timeout=180):
    """Run a function with a timeout. Returns None on timeout or error."""
    if kwargs is None:
        kwargs = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"Timeout running {func.__name__}")
                return None
            except Exception as e:
                print(f"Error running {func.__name__}: {e}")
                return None
    except Exception as e:
        print(f"Error in timeout wrapper: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def cached_sentiment(start_date, end_date):
    from market_sentiment.sentiment import sentiment
    return sentiment(start_date, end_date)

@st.cache_data(ttl=1800)
def cached_volatility(symbol, end_date, countback, forecast_days=0):
    from market_sentiment.sentiment import volatility
    return volatility(symbol, end_date, countback, forecast_days=forecast_days)

@st.cache_data(ttl=1800)
def cached_high_low_index(start_date, end_date):
    from market_sentiment.sentiment import high_low_index
    try:
        result = high_low_index(start_date, end_date)
        return result if result is not None else pd.DataFrame()
    except Exception as e:
        print(f"Error in cached_high_low_index: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def cached_bpi(start_date, end_date):
    from market_sentiment.sentiment import bpi
    try:
        result = bpi(start_date, end_date)
        return result if result is not None else pd.DataFrame()
    except Exception as e:
        print(f"Error in cached_bpi: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def cached_ma(start_date, end_date):
    from market_sentiment.sentiment import ma
    return ma(start_date, end_date)

@st.cache_data(ttl=1800)
def cached_market_breadth(start_date, end_date):
    from market_sentiment.sentiment import market_breadth
    return market_breadth(start_date, end_date)


# --- Fragment functions for independent chart rendering (reduces flickering) ---
# Each fragment runs independently, allowing charts to load without full page rerun
# Using st.fragment with run_every to auto-refresh when data is ready

@st.fragment
def render_sentiment_fragment(sent_key, start_date_str, end_date_str):
    """Fragment to render Sentiment chart independently."""
    import pandas as pd
    import plotly.graph_objects as go
    
    with st.container():
        st.subheader("🧠 Tâm lý Thị trường & VNINDEX")
        status, data = get_job_status(sent_key)

        if status == "running":
            st.info("Đang tải dữ liệu tâm lý thị trường...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu tâm lý thị trường: {data}")
        elif status == "completed" and data is not None and not data.empty and any(c in data.columns for c in ['short', 'long', 'close']):
            data['time'] = pd.to_datetime(data['time'])
            
            fig = go.Figure()
            
            # Add VNINDEX on right Y-axis
            if 'close' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['time'], y=data['close'], mode='lines',
                    name='VNINDEX', line=dict(color='green', width=2),
                    yaxis='y2'
                ))
            
            # Add Sentiment on left Y-axis
            if 'long' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['time'], y=data['long'], mode='lines',
                    name='Long (trung hạn)', line=dict(color='#1f77b4', width=1.5)
                ))
            if 'short' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['time'], y=data['short'], mode='lines',
                    name='Short (ngắn hạn)', line=dict(color='#ff7f0e', width=1.5)
                ))
            
            # Add background bands for sentiment zones
            bands = [
                (80, 100, '#006400', 0.15), (60, 80, '#2ca02c', 0.12),
                (40, 60, '#ffd700', 0.12), (20, 40, '#ff7f7f', 0.12),
                (0, 20, '#8b0000', 0.15)
            ]
            shapes = []
            for y0, y1, color, opacity in bands:
                shapes.append({
                    'type': 'rect', 'xref': 'x', 'yref': 'y',
                    'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                    'y0': y0, 'y1': y1,
                    'fillcolor': color, 'opacity': opacity, 'layer': 'below', 'line': {'width': 0}
                })
            for thr in [80, 60, 40, 20]:
                shapes.append({
                    'type': 'line', 'xref': 'x', 'yref': 'y',
                    'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                    'y0': thr, 'y1': thr,
                    'line': {'color': 'gray', 'width': 1, 'dash': 'dash'}
                })
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis=dict(
                    title=dict(text='Sentiment', font=dict(color='#1f77b4')),
                    tickfont=dict(color='#1f77b4'),
                    range=[0, 100],
                    side='left'
                ),
                yaxis2=dict(
                    title=dict(text='VNINDEX', font=dict(color='green')),
                    tickfont=dict(color='green'),
                    overlaying='y',
                    side='right'
                ),
                shapes=shapes,
                height=400, hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(l=60, r=60, t=60, b=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            st.subheader("🎯 Ngưỡng Tâm lý Thị trường")
            cols = st.columns(5)
            thresholds = [
                ("Extreme Greed", "80-100", "#006400"),
                ("Greed", "60-80", "#2ca02c"),
                ("Neutral", "40-60", "#ffd700"),
                ("Fear", "20-40", "#8b0000"),
                ("Extreme Fear", "0-20", "#8b0000")
            ]
            for i, (label, range_val, color) in enumerate(thresholds):
                cols[i].markdown(f"""
                <div style="text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 5px; background-color: white;">
                    <div style="font-size: 12px; font-weight: bold; color: {color};">{label}</div>
                    <div style="font-size: 16px; font-weight: bold; color: {color};">{range_val}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if f"{sent_key}_start_time" in st.session_state:
                loading_time_key = f"{sent_key}_loading_time"
                if loading_time_key not in st.session_state:
                    st.session_state[loading_time_key] = time.time() - st.session_state[f"{sent_key}_start_time"]
                st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
            
            if data is not None and not data.empty:
                with st.expander("📊 Xem dữ liệu tâm lý thị trường chi tiết"):
                    st.dataframe(data, width='stretch')
                    st.download_button("Tải xuống dữ liệu CSV", data.to_csv(index=False), f"sentiment_{start_date_str}_{end_date_str}.csv", "text/csv")
        else:
            st.info("Đang tải dữ liệu tâm lý thị trường tự động (mặc định 6 tháng)...")


@st.fragment
def render_volatility_fragment(vol_key, forecast_days, show_forecast, start_date_str, end_date_str):
    """Fragment to render Volatility chart independently."""
    with st.container():
        st.subheader("📈 Biến động Thị trường")
        status, data = get_job_status(vol_key)

        if status == "running":
            st.info("Đang tải dữ liệu biến động...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu biến động: {data}")
        elif status == "completed" and data is not None and not data.empty:
            import pandas as pd
            vol_col = 'volatility' if 'volatility' in data.columns else None
            if vol_col:
                data['time'] = pd.to_datetime(data['time'])
                try:
                    start_date = pd.Timestamp(data['time'].min()).strftime('%Y-%m-%d')
                    end_date = pd.Timestamp(data['time'].max()).strftime('%Y-%m-%d')
                except:
                    start_date, end_date = "N/A", "N/A"
                
                historical_data = data[data['close'].notna()].copy()
                forecast_data = data[data['close'].isna()].copy()
                
                fig_vol = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]], vertical_spacing=0.05)
                
                fig_vol.add_trace(go.Scatter(
                    x=historical_data['time'], y=historical_data['close'], mode='lines',
                    name='VNINDEX Close Price', line=dict(color='blue', width=2)
                ), secondary_y=False)
                
                fig_vol.add_trace(go.Scatter(
                    x=historical_data['time'], y=historical_data[vol_col], mode='lines',
                    name='Historical Volatility', line=dict(color='red', width=2)
                ), secondary_y=True)
                
                # Add dashed line at 0.03 threshold for strong volatility
                fig_vol.add_shape(
                    type="line",
                    x0=0, x1=1, xref="paper",
                    y0=0.03, y1=0.03, yref="y2",
                    line=dict(color="orange", width=2, dash="dash")
                )
                
                # Add annotation for the threshold
                fig_vol.add_annotation(
                    x=1.02, y=0.03, xref="paper", yref="y2",
                    text="Ngưỡng 0.03 (Biến động mạnh)",
                    showarrow=False, xanchor="left", font=dict(color="orange", size=10)
                )
                
                if show_forecast and not forecast_data.empty and vol_col in forecast_data.columns:
                    fig_vol.add_trace(go.Scatter(
                        x=forecast_data['time'], y=forecast_data[vol_col], mode='lines',
                        name='Forecast Volatility', line=dict(color='green', width=2)
                    ), secondary_y=True)
                
                title_suffix = f" (Forecast: {len(forecast_data)} days)" if show_forecast and not forecast_data.empty else ""
                fig_vol.update_layout(
                    title=f'VNINDEX Close Price and Volatility ({start_date} to {end_date}){title_suffix}',
                    xaxis_title='Date', height=400, hovermode='x unified', showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig_vol.update_yaxes(title_text="VNINDEX Price", secondary_y=False, showgrid=False)
                fig_vol.update_yaxes(title_text="Volatility", secondary_y=True, showgrid=False, range=[0, 0.06])
                st.plotly_chart(fig_vol, width='stretch')
                
                if f"{vol_key}_start_time" in st.session_state:
                    loading_time_key = f"{vol_key}_loading_time"
                    if loading_time_key not in st.session_state:
                        st.session_state[loading_time_key] = time.time() - st.session_state[f"{vol_key}_start_time"]
                    st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
                
                with st.expander("📊 Xem dữ liệu biến động chi tiết"):
                    st.dataframe(data, width='stretch')
                    st.download_button("Tải xuống dữ liệu CSV", data.to_csv(index=False), f"volatility_{start_date}_{end_date}.csv", "text/csv")
            else:
                st.error("Không tìm thấy cột dữ liệu biến động.")
        else:
            st.info("Đang tải dữ liệu biến động tự động (mặc định 6 tháng)...")


@st.fragment
def render_highlow_fragment(hl_key, start_date_str, end_date_str):
    """Fragment to render High-Low Index chart independently."""
    with st.container():
        st.subheader("📉 High-Low Index")
        status, data = get_job_status(hl_key)

        if status == "running":
            st.info("Đang tải dữ liệu High-Low Index...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu High-Low Index: {data}")
        elif status == "completed":
            if data is None or (hasattr(data, 'empty') and data.empty):
                st.warning("Không có dữ liệu High-Low Index. Có thể API không khả dụng từ máy chủ deployed.")
            else:
                import pandas as pd
                import plotly.graph_objects as go
                
                data['time'] = pd.to_datetime(data['time'])
                
                fig_hl = go.Figure()
                if 'hl_index' in data.columns:
                    fig_hl.add_trace(go.Scatter(
                        x=data['time'], y=data['hl_index'], mode='lines',
                        name='HL Index', line=dict(color='#1f77b4', width=2),
                        fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
                    ))
                
                # Add colored background zones for overbought/oversold
                shapes = [
                    # Oversold zone (0-30)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 0, 'y1': 30,
                     'fillcolor': 'rgba(46, 204, 113, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                    # Neutral zone (30-70)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 30, 'y1': 70,
                     'fillcolor': 'rgba(241, 196, 15, 0.1)', 'layer': 'below', 'line': {'width': 0}},
                    # Overbought zone (70-100)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 70, 'y1': 100,
                     'fillcolor': 'rgba(231, 76, 60, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                    # Reference lines
                    {'type': 'line', 'xref': 'paper', 'x0': 0, 'x1': 1, 'yref': 'y', 'y0': 30, 'y1': 30, 'line': {'color': 'green', 'width': 1.5, 'dash': 'dash'}},
                    {'type': 'line', 'xref': 'paper', 'x0': 0, 'x1': 1, 'yref': 'y', 'y0': 70, 'y1': 70, 'line': {'color': 'red', 'width': 1.5, 'dash': 'dash'}}
                ]
                fig_hl.update_layout(
                    xaxis_title='Date', 
                    yaxis=dict(title='HL Index', range=[0, 100], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    height=400, hovermode='x unified', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    margin=dict(l=50, r=50, t=60, b=50), 
                    shapes=shapes,
                    annotations=[
                        dict(x=0.02, y=85, xref='paper', yref='y', text='🔴 Overbought', showarrow=False, xanchor='left', font=dict(color='#c0392b', size=11, weight='bold')),
                        dict(x=0.02, y=50, xref='paper', yref='y', text='🟡 Neutral', showarrow=False, xanchor='left', font=dict(color='#f39c12', size=11, weight='bold')),
                        dict(x=0.02, y=15, xref='paper', yref='y', text='🟢 Oversold', showarrow=False, xanchor='left', font=dict(color='#27ae60', size=11, weight='bold'))
                    ]
                )
                fig_hl.update_xaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig_hl, width='stretch')
                
                if f"{hl_key}_start_time" in st.session_state:
                    loading_time_key = f"{hl_key}_loading_time"
                    if loading_time_key not in st.session_state:
                        st.session_state[loading_time_key] = time.time() - st.session_state[f"{hl_key}_start_time"]
                    st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
                
                with st.expander("📊 Xem dữ liệu High-Low Index chi tiết"):
                    st.dataframe(data, width='stretch')
                    st.download_button("Tải xuống dữ liệu CSV", data.to_csv(index=False), f"highlow_{start_date_str}_{end_date_str}.csv", "text/csv")
        else:
            st.info("Đang tải dữ liệu High-Low Index tự động (mặc định 6 tháng)...")


@st.fragment
def render_bpi_fragment(bpi_key, start_date_str, end_date_str):
    """Fragment to render BPI chart independently."""
    with st.container():
        st.subheader("📋 Bullish Percent Index")
        status, data = get_job_status(bpi_key)

        if status == "running":
            st.info("Đang tải dữ liệu BPI...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu BPI: {data}")
        elif status == "completed":
            if data is None or (hasattr(data, 'empty') and data.empty):
                st.warning("Không có dữ liệu BPI. Có thể API không khả dụng từ máy chủ deployed.")
            else:
                import pandas as pd
                import plotly.graph_objects as go
                
                data['time'] = pd.to_datetime(data['time'])
                
                fig_bpi = go.Figure()
                if 'bpi' in data.columns:
                    fig_bpi.add_trace(go.Scatter(
                        x=data['time'], y=data['bpi'], mode='lines',
                        name='BPI', line=dict(color='#9b59b6', width=2),
                        fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.2)'
                    ))
                
                # Add colored background zones
                shapes = [
                    # Oversold zone (0-30)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 0, 'y1': 30,
                     'fillcolor': 'rgba(46, 204, 113, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                    # Neutral zone (30-70)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 30, 'y1': 70,
                     'fillcolor': 'rgba(241, 196, 15, 0.1)', 'layer': 'below', 'line': {'width': 0}},
                    # Overbought zone (70-100)
                    {'type': 'rect', 'xref': 'x', 'yref': 'y', 
                     'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                     'y0': 70, 'y1': 100,
                     'fillcolor': 'rgba(231, 76, 60, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                ]
                
                fig_bpi.update_layout(
                    xaxis_title='Date', yaxis_title='BPI (%)', 
                    yaxis=dict(range=[0, 100], showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                    height=400, hovermode='x unified', showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    margin=dict(l=50, r=50, t=60, b=50),
                    shapes=shapes,
                    annotations=[
                        dict(x=0.98, y=70, xref='paper', yref='y', text='🔴 Overbought (70)', showarrow=False, xanchor='right', font=dict(color='#c0392b', size=11, weight='bold')),
                        dict(x=0.98, y=30, xref='paper', yref='y', text='🟢 Oversold (30)', showarrow=False, xanchor='right', font=dict(color='#27ae60', size=11, weight='bold'))
                    ]
                )
                fig_bpi.update_xaxes(showgrid=False, zeroline=False)
                st.plotly_chart(fig_bpi, width='stretch')
                
                if f"{bpi_key}_start_time" in st.session_state:
                    loading_time_key = f"{bpi_key}_loading_time"
                    if loading_time_key not in st.session_state:
                        st.session_state[loading_time_key] = time.time() - st.session_state[f"{bpi_key}_start_time"]
                    st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
                
                with st.expander("📊 Xem dữ liệu BPI chi tiết"):
                    st.dataframe(data, width='stretch')
                    st.download_button("Tải xuống dữ liệu CSV", data.to_csv(index=False), f"bpi_{start_date_str}_{end_date_str}.csv", "text/csv")
        else:
            st.info("Đang tải dữ liệu BPI tự động (mặc định 6 tháng)...")


@st.fragment
def render_ma_fragment(ma_key, start_date_str, end_date_str):
    """Fragment to render MA chart independently."""
    with st.container():
        st.subheader("➡️ Moving Average")
        status, data = get_job_status(ma_key)

        if status == "running":
            st.info("Đang tải dữ liệu MA...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu MA: {data}")
        elif status == "completed" and data is not None and not data.empty:
            import pandas as pd
            import plotly.graph_objects as go
            
            ma_df = data.copy()
            ma_df['time'] = pd.to_datetime(ma_df['time'])
            
            # Reset index to use integer positions (eliminates date gaps)
            ma_df = ma_df.reset_index(drop=True)
            
            # Get first day of each month for tick positions
            ma_df['year_month'] = ma_df['time'].dt.to_period('M')
            unique_months = ma_df['year_month'].unique()
            
            # Use integer indices for tick positions (no gaps between candles)
            tick_vals = []
            tick_texts = []
            for month in unique_months:
                month_data = ma_df[ma_df['year_month'] == month]
                if not month_data.empty:
                    # Get the first index of each month
                    first_idx = month_data.index[0]
                    tick_vals.append(first_idx)
                    tick_texts.append(month.strftime('%Y-%m'))
            
            # Create x-axis as integer indices (no date gaps)
            x_indices = ma_df.index.tolist()
            
            # Create custom hover text with actual dates
            hover_text = [
                f"Date: {ma_df['time'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                f"Open: {ma_df['open'].iloc[i]:,.2f}<br>" +
                f"High: {ma_df['high'].iloc[i]:,.2f}<br>" +
                f"Low: {ma_df['low'].iloc[i]:,.2f}<br>" +
                f"Close: {ma_df['close'].iloc[i]:,.2f}"
                for i in range(len(ma_df))
            ]
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Candlestick(
                x=x_indices, open=ma_df['open'], high=ma_df['high'],
                low=ma_df['low'], close=ma_df['close'],
                name="VNINDEX", increasing_line_color='green', decreasing_line_color='red',
                increasing_line_width=1, decreasing_line_width=1,
                text=hover_text,  # Custom hover text with dates
                hoverinfo='text'
            ))
            if 'ma50' in ma_df.columns:
                # Custom hover text for MA50
                ma50_hover = [
                    f"Date: {ma_df['time'].iloc[i].strftime('%Y-%m-%d')}<br>MA50: {ma_df['ma50'].iloc[i]:,.2f}" if pd.notna(ma_df['ma50'].iloc[i]) else f"Date: {ma_df['time'].iloc[i].strftime('%Y-%m-%d')}<br>MA50: N/A"
                    for i in range(len(ma_df))
                ]
                fig_ma.add_trace(go.Scatter(
                    x=x_indices, y=ma_df['ma50'], mode='lines',
                    name='MA50', line=dict(color='#1f77b4', width=2),
                    text=ma50_hover, hoverinfo='text'
                ))
            if 'ma200' in ma_df.columns:
                # Custom hover text for MA200
                ma200_hover = [
                    f"Date: {ma_df['time'].iloc[i].strftime('%Y-%m-%d')}<br>MA200: {ma_df['ma200'].iloc[i]:,.2f}" if pd.notna(ma_df['ma200'].iloc[i]) else f"Date: {ma_df['time'].iloc[i].strftime('%Y-%m-%d')}<br>MA200: N/A"
                    for i in range(len(ma_df))
                ]
                fig_ma.add_trace(go.Scatter(
                    x=x_indices, y=ma_df['ma200'], mode='lines',
                    name='MA200', line=dict(color='#2ca02c', width=2),
                    text=ma200_hover, hoverinfo='text'
                ))
            
            fig_ma.update_layout(
                title=f'📊 Moving Average - VNINDEX',
                height=500, 
                template='plotly_white', 
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                xaxis_rangeslider_visible=False, 
                margin=dict(l=50, r=50, t=60, b=50),
                bargap=0, 
                bargroupgap=0
            )
            # Use category-based x-axis (no gaps between candles)
            fig_ma.update_xaxes(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_texts,
                tickangle=45,
                showgrid=False,
                zeroline=False,
                showticklabels=True,
                title_text='Thời gian'
            )
            fig_ma.update_yaxes(title_text="Giá", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            st.plotly_chart(fig_ma, width='stretch')
            
            if f"{ma_key}_start_time" in st.session_state:
                loading_time_key = f"{ma_key}_loading_time"
                if loading_time_key not in st.session_state:
                    st.session_state[loading_time_key] = time.time() - st.session_state[f"{ma_key}_start_time"]
                st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
            
            with st.expander("📊 Xem dữ liệu MA chi tiết"):
                st.dataframe(ma_df, width='stretch')
                st.download_button("Tải xuống dữ liệu CSV", ma_df.to_csv(index=False), f"ma_{start_date_str}_{end_date_str}.csv", "text/csv")
        else:
            st.info("Đang tải dữ liệu MA tự động (mặc định 6 tháng)...")


@st.fragment
def render_breadth_fragment(bread_key, start_date_str, end_date_str):
    """Fragment to render Market Breadth chart independently."""
    with st.container():
        st.subheader("💹 Độ rộng Thị trường")
        status, data = get_job_status(bread_key)

        if status == "running":
            st.info("Đang tải dữ liệu độ rộng thị trường...")
        elif status == "error":
            st.error(f"Lỗi khi tải dữ liệu độ rộng thị trường: {data}")
        elif status == "completed" and data is not None and not data.empty:
            import pandas as pd
            import plotly.graph_objects as go
            
            data['time'] = pd.to_datetime(data['time'])
            
            fig_bread = go.Figure()
            if 'vnindex' in data.columns:
                fig_bread.add_trace(go.Scatter(
                    x=data['time'], y=data['vnindex'], mode='lines',
                    name='VNINDEX', line=dict(color='#51cf66', width=2.5)
                ))
            
            if 'percent' in data.columns:
                fig_bread.add_trace(go.Scatter(
                    x=data['time'], y=data['percent'], mode='lines',
                    name='Tỷ lệ trên EMA50', line=dict(color='#3498db', width=2), yaxis='y2',
                    fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.15)'
                ))
            else:
                st.warning("Dữ liệu Market Breadth không có cột 'percent'. Các cột có sẵn: " + str(data.columns.tolist()))
            
            # Add colored background zones for percentage
            shapes = [
                # Bearish zone (0-0.3)
                {'type': 'rect', 'xref': 'x', 'yref': 'y2', 
                 'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                 'y0': 0, 'y1': 0.3,
                 'fillcolor': 'rgba(231, 76, 60, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                # Neutral zone (0.3-0.7)
                {'type': 'rect', 'xref': 'x', 'yref': 'y2', 
                 'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                 'y0': 0.3, 'y1': 0.7,
                 'fillcolor': 'rgba(241, 196, 15, 0.1)', 'layer': 'below', 'line': {'width': 0}},
                # Bullish zone (0.7-1.0)
                {'type': 'rect', 'xref': 'x', 'yref': 'y2', 
                 'x0': pd.Timestamp(data['time'].min()), 'x1': pd.Timestamp(data['time'].max()),
                 'y0': 0.7, 'y1': 1.0,
                 'fillcolor': 'rgba(46, 204, 113, 0.15)', 'layer': 'below', 'line': {'width': 0}},
                # Reference lines
                {'type': 'line', 'xref': 'paper', 'x0': 0, 'x1': 1, 'yref': 'y2', 'y0': 0.3, 'y1': 0.3, 'line': {'color': '#e74c3c', 'width': 1.5, 'dash': 'dash'}},
                {'type': 'line', 'xref': 'paper', 'x0': 0, 'x1': 1, 'yref': 'y2', 'y0': 0.7, 'y1': 0.7, 'line': {'color': '#27ae60', 'width': 1.5, 'dash': 'dash'}}
            ]
            fig_bread.update_layout(
                xaxis_title='Date', 
                yaxis=dict(title=dict(text='VNINDEX', font=dict(color='#51cf66')), side='left', showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#51cf66')),
                yaxis2=dict(title=dict(text='Tỷ lệ EMA50', font=dict(color='#3498db')), side='right', overlaying='y', range=[0, 1], showgrid=False, tickfont=dict(color='#3498db'), tickformat='.0%'),
                height=450, hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                shapes=shapes,
                margin=dict(l=50, r=50, t=60, b=50),
                annotations=[
                    dict(x=0.02, y=0.85, xref='paper', yref='y2', text='🟢 Bullish (>70%)', showarrow=False, xanchor='left', font=dict(color='#27ae60', size=11, weight='bold')),
                    dict(x=0.02, y=0.5, xref='paper', yref='y2', text='🟡 Neutral', showarrow=False, xanchor='left', font=dict(color='#f39c12', size=11, weight='bold')),
                    dict(x=0.02, y=0.15, xref='paper', yref='y2', text='🔴 Bearish (<30%)', showarrow=False, xanchor='left', font=dict(color='#c0392b', size=11, weight='bold'))
                ]
            )
            fig_bread.update_xaxes(showgrid=False, zeroline=False)
            st.plotly_chart(fig_bread, width='stretch')
            
            if f"{bread_key}_start_time" in st.session_state:
                loading_time_key = f"{bread_key}_loading_time"
                if loading_time_key not in st.session_state:
                    st.session_state[loading_time_key] = time.time() - st.session_state[f"{bread_key}_start_time"]
                st.caption(f"⏱️ Thời gian tải biểu đồ: {st.session_state[loading_time_key]:.2f} giây")
            
            with st.expander("📊 Xem dữ liệu độ rộng thị trường chi tiết"):
                st.dataframe(data, width='stretch')
                st.download_button("Tải xuống dữ liệu CSV", data.to_csv(index=False), f"breadth_{start_date_str}_{end_date_str}.csv", "text/csv")
        else:
            st.info("Đang tải dữ liệu độ rộng thị trường tự động (mặc định 6 tháng)...")


# --- Navigation: default blank page. Main menus: Trang trống, Thị trường, Định giá ---

def clear_content_on_menu_change():
    """Callback to clear old content when menu changes."""
    # Cancel any running jobs
    if "jobs" in st.session_state:
        for key in list(st.session_state.jobs.keys()):
            try:
                future = st.session_state.jobs[key]
                if not future.done():
                    future.cancel()
            except:
                pass
            del st.session_state.jobs[key]
    
    # Clear all cached data keys (but keep executor and infrastructure)
    keys_to_keep = {"executor", "jobs", "annotations_force_refresh", "main_menu"}
    keys_to_remove = [k for k in list(st.session_state.keys()) if k not in keys_to_keep]
    for key in keys_to_remove:
        try:
            del st.session_state[key]
        except:
            pass
    
    # Set flag to clear content on next render
    st.session_state["clear_content"] = True

# Initialize clear content flag
if "clear_content" not in st.session_state:
    st.session_state["clear_content"] = False

# Track previous menu to detect changes
if "prev_main_menu" not in st.session_state:
    st.session_state.prev_main_menu = "Trang chủ"

main_menu_options = ["Trang chủ", "Thị trường", "Cổ phiếu", "Test"]

# Initialize session state for menu if not exists
if "main_menu" not in st.session_state:
    st.session_state.main_menu = "Trang chủ"

# Handle navigation via query params (set menu from quick action buttons)
# Use a separate internal key to avoid conflict with widget's session state
if "_nav_menu" in st.query_params:
    nav_target = st.query_params._nav_menu
    if nav_target in main_menu_options:
        # Set the actual menu directly
        st.session_state.main_menu = nav_target
    # Clear the query param after using it
    st.query_params.clear()

# Get the index from session state - but ensure we use the correct value
current_menu = st.session_state.main_menu
if current_menu not in main_menu_options:
    current_menu = "Trang chủ"

main_menu = st.sidebar.selectbox(
    "Menu chính", 
    main_menu_options, 
    index=main_menu_options.index(current_menu), 
    key="main_menu"
)

# Check if menu changed
menu_changed = main_menu != st.session_state.prev_main_menu
if menu_changed:
    st.session_state.prev_main_menu = main_menu
    st.session_state["clear_content"] = True

# Create main content container - this will hold all menu content
main_container = st.container()

# Check if we should show initial loading state for market sentiment
if main_menu == "Thị trường":
    # Check if any jobs are still running
    # We need to wait for jobs to complete before showing content
    # to avoid flickering
    pass  # Continue to show content normally

# Handle all menu options
if main_menu == "Trang chủ":
    # Import stock data module for Trang chủ
    from stock_data.stock_data import get_stock_history
    
    st.title("🏠 Trang chủ Dashboard")
    st.markdown("---")
    
    # Quick actions - right after title
    st.subheader("⚡ Menu")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("🏠 Trang chủ", key="quick_home"):
            st.query_params._nav_menu = "Trang chủ"
            st.rerun()
    
    with action_col2:
        if st.button("📈 Thị trường", key="quick_market"):
            st.query_params._nav_menu = "Thị trường"
            st.rerun()
    
    with action_col3:
        if st.button("💹 Cổ phiếu", key="quick_valuation"):
            st.query_params._nav_menu = "Cổ phiếu"
            st.rerun()
    
    with action_col4:
        if st.button("🧪 Test", key="quick_test"):
            st.query_params._nav_menu = "Test"
            st.rerun()
    
    st.markdown("---")
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Chào mừng đến với Dashboard Phân Tích Thị Trường")
        st.markdown("""
        Đây là nền tảng phân tích tài chính chuyên nghiệp, cung cấp các công cụ phân tích định giá và theo dõi thị trường.
        
        🎯 **Tính năng chính:**
        - Phân tích P/B, P/E, PEG định giá cổ phiếu
        - Theo dõi tâm lý thị trường và biến động
        - Biểu đồ nến và chỉ số kỹ thuật
        - So sánh với ngành và thị trường
        
        🔧 **Công cụ hỗ trợ:**
        - Dữ liệu thời gian thực
        - Phân tích chi tiết, đa chiều
        - Hỗ trợ ra quyết định đầu tư
        """)
    
    with col2:
        st.subheader("📊 VNINDEX & Thống kê")
        
        # VNINDEX Controls on main dashboard - 2 columns for better balance
        vnindex_col1, vnindex_col2 = st.columns([3, 1])
        
        with vnindex_col1:
            vnindex_period = st.selectbox(
                "Chọn khoảng thời gian",
                ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tùy chỉnh"],
                index=2,
                key="vnindex_period"
            )
            
            if vnindex_period == "Tùy chỉnh":
                vnindex_date_col1, vnindex_date_col2 = st.columns(2)
                with vnindex_date_col1:
                    vnindex_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=90),
                        key="vnindex_start"
                    )
                with vnindex_date_col2:
                    vnindex_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="vnindex_end"
                    )
            else:
                vnindex_start_date, vnindex_end_date = get_date_range(vnindex_period)
        
        with vnindex_col2:
            # Current date display
            st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y')}")
        
        # VNINDEX Chart
        try:
            # Get date range based on selection
            if vnindex_period == "Tùy chỉnh":
                days = (vnindex_end_date - vnindex_start_date).days
                vnindex_data = get_stock_history('VNINDEX', period='day', start_date=vnindex_start_date.strftime('%Y-%m-%d'), end_date=vnindex_end_date.strftime('%Y-%m-%d'))
            else:
                days_map = {"1 tháng": 30, "3 tháng": 90, "6 tháng": 180, "1 năm": 365, "2 năm": 730}
                days = days_map.get(vnindex_period, 180)
                vnindex_data = get_stock_history('VNINDEX', period='day', count_back=days)
            
            with st.spinner(f"Đang tải dữ liệu VNINDEX {vnindex_period}..."):
                # Data loading is already done above, spinner is for visual feedback
                pass
            
            if vnindex_data is not None and not vnindex_data.empty:
                # Create subplots with 2 rows (candlestick + volume)
                fig_vnindex = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3]
                )
                
                # Convert time column to datetime if it's not already
                vnindex_data['time'] = pd.to_datetime(vnindex_data['time'])
                
                # Add candlestick trace with no gaps
                fig_vnindex.add_trace(go.Candlestick(
                    x=vnindex_data['time'],
                    open=vnindex_data['open'],
                    high=vnindex_data['high'],
                    low=vnindex_data['low'],
                    close=vnindex_data['close'],
                    name="VNINDEX",
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    # Remove gaps between candlesticks by setting line width
                    increasing_line_width=1,
                    decreasing_line_width=1
                ), row=1, col=1)
                
                # Calculate volume colors to match candles exactly
                colors = []
                for idx, row in vnindex_data.iterrows():
                    if row['close'] >= row['open']:
                        colors.append('green')  # Green for increasing (same as candlestick)
                    else:
                        colors.append('red')  # Red for decreasing (same as candlestick)
                
                # Add volume bars with direct color assignment and no gaps
                fig_vnindex.add_trace(go.Bar(
                    x=vnindex_data['time'],
                    y=vnindex_data['volume'],
                    name="Volume",
                    marker_color=colors,  # Use direct color list
                    opacity=0.8,
                    width=0.8,  # Slightly less than 1.0 to ensure adjacent bars
                    showlegend=False,
                    # Add hover template to show volume and date information
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>',
                    # Custom text for hover
                    text=vnindex_data['volume'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside',
                    textfont=dict(size=10, color='black')
                ), row=2, col=1)
                
                # Update layout with comprehensive gap removal and proper x-axis configuration
                fig_vnindex.update_layout(
                    title=f'VNINDEX {vnindex_period}',
                    height=400,
                    template='plotly_white',
                    showlegend=False,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    # Remove all bar spacing
                    bargroupgap=0,  # No gap between bar groups
                    bargap=0,       # No gap between individual bars
                    # Configure x-axis (candlestick) - hide tick labels
                    xaxis=dict(
                        type="category",  # Use category type for no gaps
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showticklabels=False,  # Hide tick labels on candlestick chart
                        tickangle=0
                    ),
                    # Configure xaxis2 (volume) - use same category type for no gaps
                    xaxis2=dict(
                        type="category",  # Use category type for no gaps
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        showticklabels=True,   # Show tick labels on volume chart
                        tickangle=0,
                        # Configure tick format to show only year and month
                        tickvals=[],  # We'll set custom tick values below
                        ticktext=[],  # We'll set custom tick text below
                    )
                )
                
                # Update x-axes to ensure no gaps and proper configuration
                fig_vnindex.update_xaxes(
                    type="category",  # Use categorical axis for no gaps between candlesticks
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,  # Hide tick labels on candlestick chart
                    row=1, col=1
                )
                fig_vnindex.update_xaxes(
                    type="category",  # Use category type for no gaps between volume bars
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=True,   # Show tick labels on volume chart
                    tickangle=0,
                    row=2, col=1
                )
                
                fig_vnindex.update_yaxes(title_text="Giá (VNINDEX)", row=1, col=1)
                fig_vnindex.update_yaxes(title_text="Khối lượng", row=2, col=1)
                
                # Create custom tick labels showing only year and month
                # Extract unique months from the data
                vnindex_data['year_month'] = vnindex_data['time'].dt.to_period('M')
                unique_months = vnindex_data['year_month'].unique()
                
                # Create tick values (indices) and tick text (year-month labels)
                tick_vals = []
                tick_texts = []
                
                # Find the first occurrence of each month
                for i, month in enumerate(unique_months):
                    month_data = vnindex_data[vnindex_data['year_month'] == month]
                    first_idx = month_data.index[0]
                    tick_vals.append(first_idx)
                    tick_texts.append(month.strftime('%Y-%m'))
                
                # Update x-axis with custom tick labels
                fig_vnindex.update_xaxes(
                    tickvals=tick_vals,
                    ticktext=tick_texts,
                    tickangle=45,
                    row=2, col=1
                )
                
                st.plotly_chart(fig_vnindex, width='stretch')
                
                # Enhanced Quick Stats with more metrics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                try:
                    current_data = get_stock_history('VNINDEX', period='day', count_back=2)
                    if current_data is not None and not current_data.empty and len(current_data) >= 2:
                        latest = current_data.iloc[-1]
                        previous = current_data.iloc[-2]
                        
                        price_change = latest['close'] - previous['close']
                        price_change_pct = (price_change / previous['close']) * 100
                        volume_change = latest['volume'] - previous['volume']
                        volume_change_pct = (volume_change / previous['volume']) * 100 if previous['volume'] > 0 else 0
                        
                        # Display enhanced metrics with custom HTML for proper color coding
                        price_color = "#51cf66" if price_change >= 0 else "#ff6b6b"
                        volume_color = "#51cf66" if volume_change >= 0 else "#ff6b6b"
                        
                        stats_col1.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">VNINDEX Hiện tại</div>
                            <div style="font-size: 18px; font-weight: bold;">{latest['close']:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col2.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi Giá</div>
                            <div style="font-size: 16px; font-weight: bold; color: {price_color};">{price_change:+,.2f}</div>
                            <div style="font-size: 12px; color: {price_color};">{price_change_pct:+.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col3.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Tổng Khối lượng</div>
                            <div style="font-size: 16px; font-weight: bold;">{latest['volume']:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col4.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi KL</div>
                            <div style="font-size: 16px; font-weight: bold; color: {volume_color};">{volume_change:+,.0f}</div>
                            <div style="font-size: 12px; color: {volume_color};">{volume_change_pct:+.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Display N/A values with custom HTML
                        stats_col1.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">VNINDEX Hiện tại</div>
                            <div style="font-size: 18px; font-weight: bold; color: #999;">N/A</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col2.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi Giá</div>
                            <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                            <div style="font-size: 12px; color: #999;">N/A</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col3.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Tổng Khối lượng</div>
                            <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        stats_col4.markdown(f"""
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi KL</div>
                            <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                            <div style="font-size: 12px; color: #999;">N/A</div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    # Display error values with custom HTML
                    stats_col1.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">VNINDEX Hiện tại</div>
                        <div style="font-size: 18px; font-weight: bold; color: #999;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_col2.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi Giá</div>
                        <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                        <div style="font-size: 12px; color: #999;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_col3.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Tổng Khối lượng</div>
                        <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stats_col4.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">Thay đổi KL</div>
                        <div style="font-size: 16px; font-weight: bold; color: #999;">N/A</div>
                        <div style="font-size: 12px; color: #999;">N/A</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(f"Lỗi khi tải dữ liệu thống kê: {e}")
            else:
                st.warning("Không có dữ liệu VNINDEX để hiển thị hoặc dữ liệu không hợp lệ.")
        except Exception as e:
            st.error(f"Không thể tải dữ liệu VNINDEX: {e}")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Dashboard Phân Tích Thị Trường</strong></p>
        <p>Cung cấp công cụ phân tích tài chính chuyên nghiệp</p>
        <p>📧 Liên hệ: support@dashboard.com | 🌐 Website: www.dashboard.com</p>
    </div>
    """, unsafe_allow_html=True)

elif main_menu == "Thị trường":
    # Import market-related modules only when this menu is selected
    from stock_data.stock_data import get_stock_history
    from market_sentiment.sentiment import sentiment, volatility, high_low_index, market_breadth, bpi, ma
    
    st.header("📈 Thị trường")
    
    # Main navigation buttons
    render_main_navigation()
    st.markdown("---")
    
    # Submenu navigation buttons for Thị trường
    st.subheader("📊 Submenu")
    submenu_cols = st.columns(2)
    with submenu_cols[0]:
        if st.button("🧠 Tâm lý thị trường", key="subnav_thi_truong_tam_ly"):
            st.session_state.thi_truong_submenu = "Tâm lý thị trường"
            st.rerun()
    with submenu_cols[1]:
        if st.button("👥 Phân loại nhà đầu tư", key="subnav_thi_truong_phan_loai"):
            st.session_state.thi_truong_submenu = "Phân loại nhà đầu tư"
            st.rerun()
    
    st.markdown("---")
    
    # Submenu for Thị trường - with placeholder option
    thi_truong_submenu = st.sidebar.selectbox(
        "Chọn submenu", 
        ["-- Chọn --", "Tâm lý thị trường", "Phân loại nhà đầu tư"], 
        key="thi_truong_submenu"
    )
    
    if thi_truong_submenu == "-- Chọn --":
        st.info("👈 Vui lòng chọn submenu từ menu bên trên hoặc thanh bên trái để tiếp tục.")
    
    elif thi_truong_submenu == "Tâm lý thị trường":
        st.subheader("📊 Cài đặt khoảng thời gian")
        
        # Unified period selection in main dashboard area (similar to VNINDEX in Trang chủ)
        period_col1, period_col2 = st.columns([3, 1])
        
        with period_col1:
            unified_period = st.selectbox(
                "Chọn khoảng thời gian",
                ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tùy chỉnh"],
                index=2,
                key="unified_sentiment_period"
            )
            
            if unified_period == "Tùy chỉnh":
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    unified_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=180),
                        key="unified_start_date"
                    )
                with date_col2:
                    unified_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="unified_end_date"
                    )
            else:
                unified_start_date, unified_end_date = get_date_range(unified_period)
        
        with period_col2:
            st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y')}")
        
        # Default forecast settings (used for initial load, actual settings in Volatility tab)
        forecast_days = 10
        show_forecast = True
        
        # Generate keys based on selected period - these change when period changes
        sent_key = f"sent_df_{unified_start_date}_{unified_end_date}"
        vol_key = f"vol_df_{unified_start_date}_{unified_end_date}_{forecast_days}"
        hl_key = f"hl_df_{unified_start_date}_{unified_end_date}"
        bpi_key = f"bpi_df_{unified_start_date}_{unified_end_date}"
        ma_key = f"ma_df_{unified_start_date}_{unified_end_date}"
        bread_key = f"bread_df_{unified_start_date}_{unified_end_date}"
        
        # Submit jobs only if not already in session_state or jobs
        # Note: We check both session_state (completed data) and jobs (running jobs)
        # Use a single start_time for all jobs to track total loading time
        batch_start_time = time.time()
        
        # Submit all jobs simultaneously for parallel execution
        # Each job is submitted directly to executor for immediate parallel execution
        
        if sent_key not in st.session_state and sent_key not in st.session_state.jobs:
            st.session_state.jobs[sent_key] = st.session_state.executor.submit(
                cached_sentiment, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d")
            )
            st.session_state[f"{sent_key}_start_time"] = batch_start_time
        
        # Note: Volatility job is submitted inside the Volatility tab when user changes forecast settings
        # Initial load uses default forecast_days=10
        vol_key_initial = f"vol_df_{unified_start_date}_{unified_end_date}_10"
        if vol_key_initial not in st.session_state and vol_key_initial not in st.session_state.jobs:
            st.session_state.jobs[vol_key_initial] = st.session_state.executor.submit(
                cached_volatility, 'VNINDEX', unified_end_date.strftime("%Y-%m-%d"), 
                (unified_end_date - unified_start_date).days, 10
            )
            st.session_state[f"{vol_key_initial}_start_time"] = batch_start_time
        
        if hl_key not in st.session_state and hl_key not in st.session_state.jobs:
            st.session_state.jobs[hl_key] = st.session_state.executor.submit(
                cached_high_low_index, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d")
            )
            st.session_state[f"{hl_key}_start_time"] = batch_start_time
        
        if bpi_key not in st.session_state and bpi_key not in st.session_state.jobs:
            st.session_state.jobs[bpi_key] = st.session_state.executor.submit(
                cached_bpi, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d")
            )
            st.session_state[f"{bpi_key}_start_time"] = batch_start_time
        
        if ma_key not in st.session_state and ma_key not in st.session_state.jobs:
            st.session_state.jobs[ma_key] = st.session_state.executor.submit(
                cached_ma, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d")
            )
            st.session_state[f"{ma_key}_start_time"] = batch_start_time
        
        if bread_key not in st.session_state and bread_key not in st.session_state.jobs:
            st.session_state.jobs[bread_key] = st.session_state.executor.submit(
                cached_market_breadth, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d")
            )
            st.session_state[f"{bread_key}_start_time"] = batch_start_time

        # Custom CSS for tabs styling
        st.markdown("""
        <style>
        /* Tab styling for Tâm lý thị trường */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            padding: 4px 4px 0px 4px;
            border-bottom: 1px solid #2d2d2d;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            height: 52px;
            font-size: 17px;
            font-weight: 700;
            color: #b0b0b0;
            background-color: #2d2d2d;
            border-radius: 6px 6px 0px 0px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e0e0e;
            color: #51cf66;
            border-color: #51cf66;
            box-shadow: 0px -2px 0px 0px #51cf66 inset;
        }
        .stTabs [data-baseweb="tab-list"] {
            box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        }
        /* Tab content area */
        .stTabs [data-testid="stTabContent"] {
            background-color: #0e0e0e;
            border-radius: 0px 0px 8px 8px;
            border: 1px solid #2d2d2d;
            border-top: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs for market sentiment charts
        tab_sentiment, tab_volatility, tab_highlow, tab_bpi, tab_ma, tab_breadth = st.tabs(["🧠 Sentiment", "📈 Volatility", "📉 High-Low", "📋 BPI", "➡️ MA", "💹 Market Breadth"])

        # --- Sentiment Tab ---
        with tab_sentiment:
            render_sentiment_fragment(sent_key, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

        # --- Volatility Tab ---
        with tab_volatility:
            # Forecast settings inside Volatility tab
            st.subheader("🔮 Dự báo Tương lai (Volatility)")
            forecast_col1, forecast_col2 = st.columns([2, 1])
            with forecast_col1:
                forecast_days_tab = st.slider(
                    "Số ngày dự báo",
                    min_value=5,
                    max_value=30,
                    value=10,
                    key="forecast_days_tab",
                    help="Số ngày dự báo biến động trong tương lai"
                )
            with forecast_col2:
                show_forecast_tab = st.checkbox(
                    "Hiển thị dự báo",
                    value=True,
                    key="show_forecast_tab",
                    help="Bật/tắt hiển thị đường dự báo"
                )
            
            # Update vol_key based on forecast settings from tab
            vol_key = f"vol_df_{unified_start_date}_{unified_end_date}_{forecast_days_tab}"
            
            # Submit volatility job with forecast settings if not already done
            if vol_key not in st.session_state and vol_key not in st.session_state.jobs:
                start_time = time.time()
                st.session_state.jobs[vol_key] = st.session_state.executor.submit(
                    cached_volatility, 'VNINDEX', unified_end_date.strftime("%Y-%m-%d"), 
                    (unified_end_date - unified_start_date).days, forecast_days_tab if show_forecast_tab else 0
                )
                st.session_state[f"{vol_key}_start_time"] = start_time
            
            render_volatility_fragment(vol_key, forecast_days_tab, show_forecast_tab, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

        # --- High-Low Tab ---
        with tab_highlow:
            render_highlow_fragment(hl_key, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

        # --- BPI Tab ---
        with tab_bpi:
            render_bpi_fragment(bpi_key, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

        # --- MA Tab ---
        with tab_ma:
            render_ma_fragment(ma_key, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

        # --- Market Breadth Tab ---
        with tab_breadth:
            render_breadth_fragment(bread_key, unified_start_date.strftime("%Y-%m-%d"), unified_end_date.strftime("%Y-%m-%d"))

    elif thi_truong_submenu == "Phân loại nhà đầu tư":
        st.subheader("👥 Phân loại Nhà Đầu Tư")
        
        # Custom CSS for tabs styling (similar to Tâm lý thị trường)
        st.markdown("""
        <style>
        /* Tab styling for Phân loại nhà đầu tư */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            padding: 4px 4px 0px 4px;
            border-bottom: 1px solid #2d2d2d;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            height: 52px;
            font-size: 17px;
            font-weight: 700;
            color: #b0b0b0;
            background-color: #2d2d2d;
            border-radius: 6px 6px 0px 0px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e0e0e;
            color: #51cf66;
            border-color: #51cf66;
            box-shadow: 0px -2px 0px 0px #51cf66 inset;
        }
        .stTabs [data-baseweb="tab-list"] {
            box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        }
        /* Tab content area */
        .stTabs [data-testid="stTabContent"] {
            background-color: #0e0e0e;
            border-radius: 0px 0px 8px 8px;
            border: 1px solid #2d2d2d;
            border-top: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs for investor classification
        tab_tong_gia_tri, tab_tu_doanh, tab_ca_nhan_trong_nuoc, tab_to_chuc_trong_nuoc, tab_ca_nhan_nuoc_ngoai, tab_to_chuc_nuoc_ngoai = st.tabs([
            "💰 Tổng giá trị", 
            "🏢 Tự doanh", 
            "👤 Cá nhân trong nước", 
            "🏛️ Tổ chức trong nước", 
            "🌍 Cá nhân nước ngoài", 
            "🌐 Tổ chức nước ngoài"
        ])
        
        # --- Tổng giá trị Tab ---
        with tab_tong_gia_tri:
            # Import investor_type function
            from stock_data.stock_data import investor_type, get_stock_history
            
            st.subheader("💰 Tổng giá trị Giao dịch theo Phân loại Nhà đầu tư")
            
            # Period selection (similar to Tâm lý thị trường)
            period_col1, period_col2 = st.columns([3, 1])
            
            with period_col1:
                investor_period = st.selectbox(
                    "Chọn khoảng thời gian",
                    ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tùy chỉnh"],
                    index=2,
                    key="investor_period"
                )
                
                if investor_period == "Tùy chỉnh":
                    date_col1, date_col2 = st.columns(2)
                    with date_col1:
                        investor_start_date = st.date_input(
                            "Ngày bắt đầu",
                            value=datetime.now().date() - timedelta(days=180),
                            key="investor_start_date"
                        )
                    with date_col2:
                        investor_end_date = st.date_input(
                            "Ngày kết thúc",
                            value=datetime.now().date(),
                            key="investor_end_date"
                        )
                else:
                    investor_start_date, investor_end_date = get_date_range(investor_period)
            
            with period_col2:
                st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y')}")
            
            # Fixed symbol as VN-Index
            symbol_investor = "VN-Index"
            
            # Create cache key based on parameters
            investor_key = f"investor_{symbol_investor}_{investor_start_date}_{investor_end_date}"
            stock_key = f"investor_stock_{symbol_investor}_{investor_start_date}_{investor_end_date}"
            
            # Auto-load data (similar to Tâm lý thị trường)
            # Submit jobs if not already in session_state or jobs
            if investor_key not in st.session_state and investor_key not in st.session_state.jobs:
                st.session_state.jobs[investor_key] = st.session_state.executor.submit(
                    investor_type,
                    symbol=symbol_investor,
                    start_date=investor_start_date.strftime('%Y-%m-%d'),
                    end_date=investor_end_date.strftime('%Y-%m-%d')
                )
                st.session_state[f"{investor_key}_start_time"] = time.time()
            
            # Get job status
            inv_status, investor_df = get_job_status(investor_key)
            
            if inv_status == "running":
                st.info(f"Đang tải dữ liệu phân loại nhà đầu tư cho {symbol_investor}...")
            elif inv_status == "error":
                st.error(f"Lỗi khi tải dữ liệu: {investor_df}")
            elif inv_status == "completed" and investor_df is not None and not investor_df.empty:
                try:
                    # Get stock history for close price
                    stock_df = get_stock_history(
                        symbol=symbol_investor.replace('-Index', 'INDEX') if 'Index' in symbol_investor else symbol_investor,
                        period="day",
                        end_date=investor_end_date.strftime('%Y-%m-%d'),
                        count_back=(investor_end_date - investor_start_date).days + 30
                    )
                    
                    # Create the chart
                    fig_investor = make_subplots(
                        rows=1, cols=1,
                        specs=[[{"secondary_y": True}]],
                        vertical_spacing=0.05
                    )
                    
                    # Define professional colors for each investor type
                    colors = {
                        'Tự doanh ròng': '#e74c3c',           # Red - Proprietary trading
                        'Cá nhân trong nước ròng': '#3498db',  # Blue - Domestic individual
                        'Tổ chức trong nước ròng': '#2ecc71',  # Green - Domestic institutional
                        'Cá nhân nước ngoài ròng': '#9b59b6',  # Purple - Foreign individual
                        'Tổ chức nước ngoài ròng': '#f39c12'   # Orange - Foreign institutional
                    }
                    
                    # Add stacked bar traces for investor types
                    investor_columns = [
                        'Tự doanh ròng',
                        'Cá nhân trong nước ròng',
                        'Tổ chức trong nước ròng',
                        'Cá nhân nước ngoài ròng',
                        'Tổ chức nước ngoài ròng'
                    ]
                    
                    for col in investor_columns:
                        if col in investor_df.columns:
                            # Convert to numeric, handling any string values
                            investor_df[col] = pd.to_numeric(investor_df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                            
                            # Create custom hover text with actual dates
                            bar_hover = [
                                f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                                f"{col}: {investor_df[col].iloc[i]:,.0f}"
                                for i in range(len(investor_df))
                            ]
                            
                            fig_investor.add_trace(go.Bar(
                                x=investor_df['Ngày'],
                                y=investor_df[col],
                                name=col,
                                marker_color=colors.get(col, '#888888'),
                                opacity=0.85,
                                text=bar_hover,
                                hoverinfo='text'
                            ), secondary_y=False)
                    
                    # Add close price line if stock data is available
                    if stock_df is not None and not stock_df.empty:
                        stock_df['time'] = pd.to_datetime(stock_df['time'])
                        # Filter stock data to match the selected date range
                        start_datetime = pd.to_datetime(investor_start_date)
                        end_datetime = pd.to_datetime(investor_end_date)
                        stock_df_filtered = stock_df[
                            (stock_df['time'] >= start_datetime) & 
                            (stock_df['time'] <= end_datetime)
                        ]
                        
                        if not stock_df_filtered.empty:
                            # Create hover text for close price
                            close_hover = [
                                f"Ngày: {row['time'].strftime('%Y-%m-%d')}<br>" +
                                f"Giá đóng cửa: {row['close']:,.2f}"
                                for _, row in stock_df_filtered.iterrows()
                            ]
                            
                            fig_investor.add_trace(go.Scatter(
                                x=stock_df_filtered['time'],
                                y=stock_df_filtered['close'],
                                mode='lines',
                                name='Giá đóng cửa',
                                line=dict(color='#1abc9c', width=2.5),
                                text=close_hover,
                                hoverinfo='text'
                            ), secondary_y=True)
                    
                    # Update layout with improved visual hierarchy
                    fig_investor.update_layout(
                        title=dict(
                            text=f'💰 Phân loại Nhà đầu tư - {symbol_investor}',
                            y=0.98,
                            x=0.5,
                            xanchor='center',
                            yanchor='top',
                            font=dict(size=18)
                        ),
                        barmode='relative',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị giao dịch ròng (tỷ đồng)',
                        yaxis2_title='Giá đóng cửa',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.08,
                            xanchor='center',
                            x=0.5
                        ),
                        margin=dict(l=60, r=60, t=100, b=60),
                        bargap=0.2,
                        bargroupgap=0.1
                    )
                    
                    fig_investor.update_xaxes(showgrid=False, zeroline=False)
                    fig_investor.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', secondary_y=False)
                    fig_investor.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_investor, width='stretch')
                    
                    # Show data table
                    with st.expander("📊 Xem dữ liệu chi tiết"):
                        st.dataframe(investor_df, width='stretch')
                        st.download_button(
                            "Tải xuống dữ liệu CSV",
                            investor_df.to_csv(index=False),
                            f"investor_type_{symbol_investor}_{investor_start_date.strftime('%Y%m%d')}_{investor_end_date.strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Lỗi khi tạo biểu đồ: {e}")
            elif inv_status == "completed":
                st.warning(f"Không có dữ liệu phân loại nhà đầu tư cho {symbol_investor}.")
        
        # --- Tự doanh Tab ---
        with tab_tu_doanh:
            st.subheader("🏢 Giao dịch Tự doanh Ròng")
            
            if inv_status == "completed" and investor_df is not None and not investor_df.empty:
                col_name = 'Tự doanh ròng'
                if col_name in investor_df.columns:
                    investor_df[col_name] = pd.to_numeric(investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    # Calculate cumulative value
                    cumulative_values = investor_df[col_name].cumsum()
                    
                    # Create figure with secondary y-axis
                    fig_td = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar chart with conditional colors
                    colors_td = ['#2ecc71' if v >= 0 else '#e74c3c' for v in investor_df[col_name]]
                    
                    bar_hover_td = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị ròng: {investor_df[col_name].iloc[i]:,.0f}<br>" +
                        f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_td.add_trace(go.Bar(
                        x=investor_df['Ngày'],
                        y=investor_df[col_name],
                        name='Giá trị ròng',
                        marker_color=colors_td,
                        opacity=0.85,
                        text=bar_hover_td,
                        hoverinfo='text'
                    ), secondary_y=False)
                    
                    # Add cumulative line
                    line_hover_td = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_td.add_trace(go.Scatter(
                        x=investor_df['Ngày'],
                        y=cumulative_values,
                        mode='lines+markers',
                        name='Giá trị tích lũy',
                        line=dict(color='#3498db', width=2.5),
                        marker=dict(size=4),
                        text=line_hover_td,
                        hoverinfo='text'
                    ), secondary_y=True)
                    
                    fig_td.update_layout(
                        title=f'Tự doanh Ròng - VN-Index ({investor_start_date.strftime("%Y-%m-%d")} đến {investor_end_date.strftime("%Y-%m-%d")})',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị ròng',
                        yaxis2_title='Giá trị tích lũy',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=60, t=80, b=60),
                        bargap=0.15
                    )
                    
                    fig_td.update_xaxes(showgrid=False)
                    fig_td.update_yaxes(showgrid=False, secondary_y=False)
                    fig_td.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_td, width='stretch')
                else:
                    st.warning("Không có dữ liệu Tự doanh ròng.")
            else:
                st.info("Đang tải dữ liệu...")
        
        # --- Cá nhân trong nước Tab ---
        with tab_ca_nhan_trong_nuoc:
            st.subheader("👤 Giao dịch Cá nhân trong nước Ròng")
            
            if inv_status == "completed" and investor_df is not None and not investor_df.empty:
                col_name = 'Cá nhân trong nước ròng'
                if col_name in investor_df.columns:
                    investor_df[col_name] = pd.to_numeric(investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    # Calculate cumulative value
                    cumulative_values = investor_df[col_name].cumsum()
                    
                    # Create figure with secondary y-axis
                    fig_cntn = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar chart with conditional colors
                    colors_cntn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in investor_df[col_name]]
                    
                    bar_hover_cntn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị ròng: {investor_df[col_name].iloc[i]:,.0f}<br>" +
                        f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_cntn.add_trace(go.Bar(
                        x=investor_df['Ngày'],
                        y=investor_df[col_name],
                        name='Giá trị ròng',
                        marker_color=colors_cntn,
                        opacity=0.85,
                        text=bar_hover_cntn,
                        hoverinfo='text'
                    ), secondary_y=False)
                    
                    # Add cumulative line
                    line_hover_cntn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_cntn.add_trace(go.Scatter(
                        x=investor_df['Ngày'],
                        y=cumulative_values,
                        mode='lines+markers',
                        name='Giá trị tích lũy',
                        line=dict(color='#3498db', width=2.5),
                        marker=dict(size=4),
                        text=line_hover_cntn,
                        hoverinfo='text'
                    ), secondary_y=True)
                    
                    fig_cntn.update_layout(
                        title=f'Cá nhân trong nước Ròng - VN-Index ({investor_start_date.strftime("%Y-%m-%d")} đến {investor_end_date.strftime("%Y-%m-%d")})',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị ròng',
                        yaxis2_title='Giá trị tích lũy',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=60, t=80, b=60),
                        bargap=0.15
                    )
                    
                    fig_cntn.update_xaxes(showgrid=False)
                    fig_cntn.update_yaxes(showgrid=False, secondary_y=False)
                    fig_cntn.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_cntn, width='stretch')
                else:
                    st.warning("Không có dữ liệu Cá nhân trong nước ròng.")
            else:
                st.info("Đang tải dữ liệu...")
        
        # --- Tổ chức trong nước Tab ---
        with tab_to_chuc_trong_nuoc:
            st.subheader("🏛️ Giao dịch Tổ chức trong nước Ròng")
            
            if inv_status == "completed" and investor_df is not None and not investor_df.empty:
                col_name = 'Tổ chức trong nước ròng'
                if col_name in investor_df.columns:
                    investor_df[col_name] = pd.to_numeric(investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    # Calculate cumulative value
                    cumulative_values = investor_df[col_name].cumsum()
                    
                    # Create figure with secondary y-axis
                    fig_tctn = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar chart with conditional colors
                    colors_tctn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in investor_df[col_name]]
                    
                    bar_hover_tctn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị ròng: {investor_df[col_name].iloc[i]:,.0f}<br>" +
                        f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_tctn.add_trace(go.Bar(
                        x=investor_df['Ngày'],
                        y=investor_df[col_name],
                        name='Giá trị ròng',
                        marker_color=colors_tctn,
                        opacity=0.85,
                        text=bar_hover_tctn,
                        hoverinfo='text'
                    ), secondary_y=False)
                    
                    # Add cumulative line
                    line_hover_tctn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_tctn.add_trace(go.Scatter(
                        x=investor_df['Ngày'],
                        y=cumulative_values,
                        mode='lines+markers',
                        name='Giá trị tích lũy',
                        line=dict(color='#3498db', width=2.5),
                        marker=dict(size=4),
                        text=line_hover_tctn,
                        hoverinfo='text'
                    ), secondary_y=True)
                    
                    fig_tctn.update_layout(
                        title=f'Tổ chức trong nước Ròng - VN-Index ({investor_start_date.strftime("%Y-%m-%d")} đến {investor_end_date.strftime("%Y-%m-%d")})',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị ròng',
                        yaxis2_title='Giá trị tích lũy',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=60, t=80, b=60),
                        bargap=0.15
                    )
                    
                    fig_tctn.update_xaxes(showgrid=False)
                    fig_tctn.update_yaxes(showgrid=False, secondary_y=False)
                    fig_tctn.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_tctn, width='stretch')
                else:
                    st.warning("Không có dữ liệu Tổ chức trong nước ròng.")
            else:
                st.info("Đang tải dữ liệu...")
        
        # --- Cá nhân nước ngoài Tab ---
        with tab_ca_nhan_nuoc_ngoai:
            st.subheader("🌍 Giao dịch Cá nhân nước ngoài Ròng")
            
            if inv_status == "completed" and investor_df is not None and not investor_df.empty:
                col_name = 'Cá nhân nước ngoài ròng'
                if col_name in investor_df.columns:
                    investor_df[col_name] = pd.to_numeric(investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    # Calculate cumulative value
                    cumulative_values = investor_df[col_name].cumsum()
                    
                    # Create figure with secondary y-axis
                    fig_cnnn = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar chart with conditional colors
                    colors_cnnn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in investor_df[col_name]]
                    
                    bar_hover_cnnn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị ròng: {investor_df[col_name].iloc[i]:,.0f}<br>" +
                        f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_cnnn.add_trace(go.Bar(
                        x=investor_df['Ngày'],
                        y=investor_df[col_name],
                        name='Giá trị ròng',
                        marker_color=colors_cnnn,
                        opacity=0.85,
                        text=bar_hover_cnnn,
                        hoverinfo='text'
                    ), secondary_y=False)
                    
                    # Add cumulative line
                    line_hover_cnnn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_cnnn.add_trace(go.Scatter(
                        x=investor_df['Ngày'],
                        y=cumulative_values,
                        mode='lines+markers',
                        name='Giá trị tích lũy',
                        line=dict(color='#3498db', width=2.5),
                        marker=dict(size=4),
                        text=line_hover_cnnn,
                        hoverinfo='text'
                    ), secondary_y=True)
                    
                    fig_cnnn.update_layout(
                        title=f'Cá nhân nước ngoài Ròng - VN-Index ({investor_start_date.strftime("%Y-%m-%d")} đến {investor_end_date.strftime("%Y-%m-%d")})',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị ròng',
                        yaxis2_title='Giá trị tích lũy',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=60, t=80, b=60),
                        bargap=0.15
                    )
                    
                    fig_cnnn.update_xaxes(showgrid=False)
                    fig_cnnn.update_yaxes(showgrid=False, secondary_y=False)
                    fig_cnnn.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_cnnn, width='stretch')
                else:
                    st.warning("Không có dữ liệu Cá nhân nước ngoài ròng.")
            else:
                st.info("Đang tải dữ liệu...")
        
        # --- Tổ chức nước ngoài Tab ---
        with tab_to_chuc_nuoc_ngoai:
            st.subheader("🌐 Giao dịch Tổ chức nước ngoài Ròng")
            
            if inv_status == "completed" and investor_df is not None and not investor_df.empty:
                col_name = 'Tổ chức nước ngoài ròng'
                if col_name in investor_df.columns:
                    investor_df[col_name] = pd.to_numeric(investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                    
                    # Calculate cumulative value
                    cumulative_values = investor_df[col_name].cumsum()
                    
                    # Create figure with secondary y-axis
                    fig_tcnn = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bar chart with conditional colors
                    colors_tcnn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in investor_df[col_name]]
                    
                    bar_hover_tcnn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị ròng: {investor_df[col_name].iloc[i]:,.0f}<br>" +
                        f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_tcnn.add_trace(go.Bar(
                        x=investor_df['Ngày'],
                        y=investor_df[col_name],
                        name='Giá trị ròng',
                        marker_color=colors_tcnn,
                        opacity=0.85,
                        text=bar_hover_tcnn,
                        hoverinfo='text'
                    ), secondary_y=False)
                    
                    # Add cumulative line
                    line_hover_tcnn = [
                        f"Ngày: {investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                        f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                        for i in range(len(investor_df))
                    ]
                    
                    fig_tcnn.add_trace(go.Scatter(
                        x=investor_df['Ngày'],
                        y=cumulative_values,
                        mode='lines+markers',
                        name='Giá trị tích lũy',
                        line=dict(color='#3498db', width=2.5),
                        marker=dict(size=4),
                        text=line_hover_tcnn,
                        hoverinfo='text'
                    ), secondary_y=True)
                    
                    fig_tcnn.update_layout(
                        title=f'Tổ chức nước ngoài Ròng - VN-Index ({investor_start_date.strftime("%Y-%m-%d")} đến {investor_end_date.strftime("%Y-%m-%d")})',
                        xaxis_title='Thời gian',
                        yaxis_title='Giá trị ròng',
                        yaxis2_title='Giá trị tích lũy',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=60, r=60, t=80, b=60),
                        bargap=0.15
                    )
                    
                    fig_tcnn.update_xaxes(showgrid=False)
                    fig_tcnn.update_yaxes(showgrid=False, secondary_y=False)
                    fig_tcnn.update_yaxes(showgrid=False, secondary_y=True)
                    
                    st.plotly_chart(fig_tcnn, width='stretch')
                else:
                    st.warning("Không có dữ liệu Tổ chức nước ngoài ròng.")
            else:
                st.info("Đang tải dữ liệu...")

elif main_menu == "Cổ phiếu":
    st.header("💹 Cổ phiếu")
    
    # Main navigation buttons
    render_main_navigation()
    st.markdown("---")
    
    # Submenu navigation buttons for Cổ phiếu
    st.subheader("💰 Submenu")
    submenu_cols = st.columns(2)
    with submenu_cols[0]:
        if st.button("💰 Định giá", key="subnav_co_phieu_dinh_gia"):
            st.session_state.co_phieu_submenu = "Định giá"
            st.rerun()
    with submenu_cols[1]:
        if st.button("🔄 Phân loại giao dịch", key="subnav_co_phieu_phan_loai"):
            st.session_state.co_phieu_submenu = "Phân loại giao dịch"
            st.rerun()
    
    st.markdown("---")
    
    # Submenu for Cổ phiếu - with placeholder option
    co_phieu_submenu = st.sidebar.selectbox(
        "Chọn submenu", 
        ["-- Chọn --", "Định giá", "Phân loại giao dịch"], 
        key="co_phieu_submenu"
    )
    
    if co_phieu_submenu == "-- Chọn --":
        st.info("👈 Vui lòng chọn submenu từ menu bên trên hoặc thanh bên trái để tiếp tục.")
    
    elif co_phieu_submenu == "Định giá":
        # Import valuation-related modules only when this menu is selected
        from valuation.valuation import get_pb_pe, ref_pb_pe, get_peg, fireant_valuation, analyst_price_targets
        from stock_data.stock_data import get_stock_history
        
        st.header("📊 Định giá")
        
        # Custom CSS for dark theme compatible tabs spanning full width
        st.markdown("""
        <style>
        /* Dark theme professional tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            padding: 4px 4px 0px 4px;
            border-bottom: 1px solid #2d2d2d;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            height: 52px;
            font-size: 17px;
            font-weight: 700;
            color: #b0b0b0;
            background-color: #2d2d2d;
            border-radius: 6px 6px 0px 0px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e0e0e;
            color: #ff6b6b;
            border-color: #ff6b6b;
            box-shadow: 0px -2px 0px 0px #ff6b6b inset;
        }
        .stTabs [data-baseweb="tab-list"] {
            box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        }
        /* Dark theme content area */
        .stTabs [data-testid="stTabContent"] {
            background-color: #0e0e0e;
            border-radius: 0px 0px 8px 8px;
            border: 1px solid #2d2d2d;
            border-top: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create tabs for P/B, P/E, PEG, Price, Features
        tab_pb, tab_pe, tab_peg, tab_price, tab_features = st.tabs(["📊 P/B", "📈 P/E", "🎯 PEG", "💰 Price", "⚡ Features"])
        
        # Initialize session state for storing charts
        if 'pb_chart_data' not in st.session_state:
            st.session_state.pb_chart_data = None
        if 'pe_chart_data' not in st.session_state:
            st.session_state.pe_chart_data = None
        
        # Cached wrapper for ref_pb_pe (shared between P/B and P/E tabs)
        @st.cache_data(ttl=3600)
        def cached_ref_pb_pe(symbol, start_date=None, end_date=None):
            return ref_pb_pe(symbol, start_date=start_date, end_date=end_date)
        
        # Cached wrapper for get_pb_pe historical data
        @st.cache_data(ttl=3600)
        def cached_get_pb_pe(symbol, start_date=None, end_date=None):
            return get_pb_pe(symbol, start_date=start_date, end_date=end_date)

        # Function to display P/B chart from stored data
        def display_pb_chart(pb_df, pb_ref, symbol):
            # Create chart with dark theme
            fig_pb = go.Figure()
            
            # Add P/B line with area fill
            fig_pb.add_trace(go.Scatter(
                x=pb_df['date'], y=pb_df['pb'], mode='lines', name='P/B',
                line=dict(color='#00BFFF', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0, 191, 255, 0.1)',
            ))

            # Normalize ref to a dict for easy access
            thresholds = {}
            if isinstance(pb_ref, pd.Series):
                try:
                    thresholds = dict(pb_ref.dropna())
                except Exception:
                    thresholds = dict(pb_ref)
            elif isinstance(pb_ref, dict):
                thresholds = {k: v for k, v in pb_ref.items() if v is not None}

            # Display threshold metrics if available
            if thresholds:
                cols = st.columns(4)
                keys_map = [
                    ('pb_ttm_avg', 'PB TTM Avg'),
                    ('pb_ttm_med', 'PB TTM Med'),
                    ('pb_sec_avg', 'PB Sec Avg'),
                    ('pb_sec_med', 'PB Sec Med'),
                ]
                # Get current/last P/B value
                pb_latest = pb_df['pb'].dropna().iloc[-1] if not pb_df['pb'].dropna().empty else None
                
                for i, (k, label) in enumerate(keys_map):
                    val = thresholds.get(k)
                    if val is not None and not pd.isna(val):
                        try:
                            # Add delta for all metrics
                            if pb_latest is not None:
                                pb_diff = ((val - pb_latest) / pb_latest) * 100 if pb_latest != 0 else 0
                                cols[i].metric(label, f"{float(val):.2f}", delta=f"{pb_diff:+.1f}%", delta_color="normal")
                            else:
                                cols[i].metric(label, f"{float(val):.2f}")
                        except Exception:
                            cols[i].metric(label, str(val))

                # Add threshold lines as traces (show in legend)
                colors = ['#FF8C00', '#DC143C', '#32CD32', '#9370DB']
                
                for i, (k, label) in enumerate(keys_map):
                    val = thresholds.get(k)
                    if val is None:
                        continue
                    try:
                        yv = float(val)
                    except Exception:
                        continue
                    
                    # Add horizontal line as trace (appears in legend)
                    fig_pb.add_trace(go.Scatter(
                        x=[pd.Timestamp(pb_df['date'].min()), pd.Timestamp(pb_df['date'].max())],
                        y=[yv, yv],
                        mode='lines',
                        name=f"{label}: {yv:.2f}",
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        showlegend=True,
                        hoverinfo='skip'
                    ))

                # Update layout with dark theme and annotations
                fig_pb.update_layout(
                    title=dict(
                        text=f"<b>📊 P/B Ratio {symbol}</b>",
                        x=0.5, xanchor='center',
                        font=dict(size=18, color='white'),
                        y=0.95
                    ),
                    xaxis_title=dict(text='<b>Date</b>', font=dict(color='white')),
                    yaxis_title=dict(text='<b>P/B</b>', font=dict(color='white')),
                    height=520,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.05,
                        xanchor="center", x=0.5,
                        font=dict(color='white', size=11),
                        bgcolor='rgba(0,0,0,0.3)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,30,30,0.5)',
                    margin=dict(t=120, l=60, r=40, b=60),
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    )
                )
            else:
                # Basic layout without thresholds
                fig_pb.update_layout(
                    title=dict(
                        text=f"<b>📊 P/B Ratio {symbol}</b>",
                        x=0.5, xanchor='center',
                        font=dict(size=18, color='white'),
                        y=0.95
                    ),
                    xaxis_title=dict(text='<b>Date</b>', font=dict(color='white')),
                    yaxis_title=dict(text='<b>P/B</b>', font=dict(color='white')),
                    height=520,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.05,
                        xanchor="center", x=0.5,
                        font=dict(color='white', size=11),
                        bgcolor='rgba(0,0,0,0.3)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,30,30,0.5)',
                    margin=dict(t=80, l=60, r=40, b=60),
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                )
            
            st.plotly_chart(fig_pb, use_container_width=True)

            # Summary section for P/B
            st.markdown("---")
            st.subheader("📋 Tổng hợp P/B")
            
            # Get current values
            pb_latest = pb_df['pb'].dropna().iloc[-1] if not pb_df['pb'].dropna().empty else None
            current_price = pb_df['price'].dropna().iloc[-1] if 'price' in pb_df.columns and not pb_df['price'].dropna().empty else None
            pb_ttm_avg = thresholds.get('pb_ttm_avg') if thresholds else None
            
            if pb_latest is not None and current_price is not None and pb_ttm_avg is not None and pb_ttm_avg != 0:
                # Calculate fair price based on PB TTM Avg
                fair_price = current_price * (pb_ttm_avg / pb_latest)
                price_diff = ((fair_price - current_price) / current_price) * 100
                
                sum_cols = st.columns(4)
                sum_cols[0].metric("Giá hiện tại", f"{current_price:,.0f} VND")
                sum_cols[1].metric("P/B hiện tại", f"{pb_latest:.2f}")
                sum_cols[2].metric("PB TTM Avg", f"{pb_ttm_avg:.2f}")
                sum_cols[3].metric("Giá hợp lý (PB TTM Avg)", f"{fair_price:,.0f} VND", delta=f"{price_diff:+.1f}%", delta_color="normal")
            elif pb_latest is not None and current_price is not None:
                sum_cols = st.columns(3)
                sum_cols[0].metric("Giá hiện tại", f"{current_price:,.0f} VND")
                sum_cols[1].metric("P/B hiện tại", f"{pb_latest:.2f}")
                if pb_ttm_avg is not None:
                    sum_cols[2].metric("PB TTM Avg", f"{pb_ttm_avg:.2f}")
                else:
                    sum_cols[2].metric("PB TTM Avg", "N/A")

            with st.expander("Xem dữ liệu P/B chi tiết"):
                st.dataframe(pb_df.rename(columns={'date': 'time'}), width='stretch')
                st.download_button("Tải xuống P/B CSV", pb_df.to_csv(index=False), f"pb_{symbol}.csv", "text/csv")

        # Function to display P/E chart from stored data
        def display_pe_chart(pe_df, pe_ref, symbol):
            # Create chart with dark theme
            fig_pe = go.Figure()
            
            # Choose pe column candidate
            pe_col = next((c for c in ['pe', 'pe_ttm', 'pe_latest'] if c in pe_df.columns), None)
            if pe_col is None:
                st.error("Không tìm thấy cột P/E trong dữ liệu trả về.")
                return
            
            # Add P/E line with area fill
            fig_pe.add_trace(go.Scatter(
                x=pe_df['date'], y=pe_df[pe_col], mode='lines', name='P/E',
                line=dict(color='#32CD32', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(50, 205, 50, 0.1)',
            ))

            # Normalize ref to a dict for easy access
            thresholds = {}
            if isinstance(pe_ref, pd.Series):
                try:
                    thresholds = dict(pe_ref.dropna())
                except Exception:
                    thresholds = dict(pe_ref)
            elif isinstance(pe_ref, dict):
                thresholds = {k: v for k, v in pe_ref.items() if v is not None}

            # Display threshold metrics if available
            if thresholds:
                cols = st.columns(4)
                keys_map = [
                    ('pe_ttm_avg', 'PE TTM Avg'),
                    ('pe_ttm_med', 'PE TTM Med'),
                    ('pe_sec_avg', 'PE Sec Avg'),
                    ('pe_sec_med', 'PE Sec Med'),
                ]
                # Get current/last P/E value
                pe_latest = pe_df[pe_col].dropna().iloc[-1] if pe_col and not pe_df[pe_col].dropna().empty else None
                
                for i, (k, label) in enumerate(keys_map):
                    val = thresholds.get(k)
                    if val is not None and not pd.isna(val):
                        try:
                            # Add delta for all metrics
                            if pe_latest is not None:
                                pe_diff = ((val - pe_latest) / pe_latest) * 100 if pe_latest != 0 else 0
                                cols[i].metric(label, f"{float(val):.2f}", delta=f"{pe_diff:+.1f}%", delta_color="normal")
                            else:
                                cols[i].metric(label, f"{float(val):.2f}")
                        except Exception:
                            cols[i].metric(label, str(val))

                # Add threshold lines as traces (show in legend)
                colors = ['#FF8C00', '#DC143C', '#1E90FF', '#9370DB']
                
                for i, (k, label) in enumerate(keys_map):
                    val = thresholds.get(k)
                    if val is None:
                        continue
                    try:
                        yv = float(val)
                    except Exception:
                        continue
                    
                    # Add horizontal line as trace (appears in legend)
                    fig_pe.add_trace(go.Scatter(
                        x=[pe_df['date'].min(), pe_df['date'].max()],
                        y=[yv, yv],
                        mode='lines',
                        name=f"{label}: {yv:.2f}",
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                        showlegend=True,
                        hoverinfo='skip'
                    ))

                # Update layout with dark theme
                fig_pe.update_layout(
                    title=dict(
                        text=f"<b>📈 P/E Ratio {symbol}</b>",
                        x=0.5, xanchor='center',
                        font=dict(size=18, color='white'),
                        y=0.95
                    ),
                    xaxis_title=dict(text='<b>Date</b>', font=dict(color='white')),
                    yaxis_title=dict(text='<b>P/E</b>', font=dict(color='white')),
                    height=520,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.05,
                        xanchor="center", x=0.5,
                        font=dict(color='white', size=11),
                        bgcolor='rgba(0,0,0,0.3)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,30,30,0.5)',
                    margin=dict(t=120, l=60, r=40, b=60),
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    )
                )
            else:
                # Basic layout without thresholds
                fig_pe.update_layout(
                    title=dict(
                        text=f"<b>📈 P/E Ratio {symbol}</b>",
                        x=0.5, xanchor='center',
                        font=dict(size=18, color='white'),
                        y=0.95
                    ),
                    xaxis_title=dict(text='<b>Date</b>', font=dict(color='white')),
                    yaxis_title=dict(text='<b>P/E</b>', font=dict(color='white')),
                    height=520,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.05,
                        xanchor="center", x=0.5,
                        font=dict(color='white', size=11),
                        bgcolor='rgba(0,0,0,0.3)'
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30,30,30,0.5)',
                    margin=dict(t=80, l=60, r=40, b=60),
                    xaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                        tickfont=dict(color='white'),
                        title_font=dict(color='white')
                    ),
                )
            
            st.plotly_chart(fig_pe, use_container_width=True)

            # Summary section for P/E
            st.markdown("---")
            st.subheader("📋 Tổng hợp P/E")
            
            # Get current values
            pe_col = 'pe' if 'pe' in pe_df.columns else ('pe_ttm' if 'pe_ttm' in pe_df.columns else None)
            pe_latest = pe_df[pe_col].dropna().iloc[-1] if pe_col and not pe_df[pe_col].dropna().empty else None
            current_price = pe_df['price'].dropna().iloc[-1] if 'price' in pe_df.columns and not pe_df['price'].dropna().empty else None
            pe_ttm_avg = thresholds.get('pe_ttm_avg') if thresholds else None
            
            if pe_latest is not None and current_price is not None and pe_ttm_avg is not None and pe_ttm_avg != 0:
                # Calculate fair price based on PE TTM Avg
                fair_price = current_price * (pe_ttm_avg / pe_latest)
                price_diff = ((fair_price - current_price) / current_price) * 100
                
                sum_cols = st.columns(4)
                sum_cols[0].metric("Giá hiện tại", f"{current_price:,.0f} VND")
                sum_cols[1].metric("P/E hiện tại", f"{pe_latest:.2f}")
                sum_cols[2].metric("PE TTM Avg", f"{pe_ttm_avg:.2f}")
                sum_cols[3].metric("Giá hợp lý (PE TTM Avg)", f"{fair_price:,.0f} VND", delta=f"{price_diff:+.1f}%", delta_color="normal")
            elif pe_latest is not None and current_price is not None:
                sum_cols = st.columns(3)
                sum_cols[0].metric("Giá hiện tại", f"{current_price:,.0f} VND")
                sum_cols[1].metric("P/E hiện tại", f"{pe_latest:.2f}")
                if pe_ttm_avg is not None:
                    sum_cols[2].metric("PE TTM Avg", f"{pe_ttm_avg:.2f}")
                else:
                    sum_cols[2].metric("PE TTM Avg", "N/A")

            with st.expander("Xem dữ liệu P/E chi tiết"):
                st.dataframe(pe_df.rename(columns={'date': 'time'}), use_container_width=True)
                st.download_button("Tải xuống P/E CSV", pe_df.to_csv(index=False), f"pe_{symbol}.csv", "text/csv")

        # Valuation -> P/B implementation
        with tab_pb:
            st.subheader("P/B: Historical series and benchmarks")
            
            # Create columns for symbol input and period selection
            col1, col2 = st.columns([1, 1])
            with col1:
                symbol_pb = st.text_input("Mã cổ phiếu (ví dụ: VCI)", value="", max_chars=10, key="val_symbol_pb")
            with col2:
                pb_period = st.selectbox(
                    "Chọn khoảng thời gian",
                    ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm", "Tùy chỉnh"],
                    index=3,  # Default to "1 năm"
                    key="pb_period"
                )
            
            # If custom period is selected, show date inputs
            if pb_period == "Tùy chỉnh":
                pb_date_col1, pb_date_col2 = st.columns(2)
                with pb_date_col1:
                    pb_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=365),
                        key="pb_start"
                    )
                with pb_date_col2:
                    pb_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="pb_end"
                    )
            
            # Display stored chart if exists
            if st.session_state.pb_chart_data is not None:
                stored_symbol, stored_period, stored_pb_df, stored_pb_ref = st.session_state.pb_chart_data
                if stored_symbol == symbol_pb and stored_period == pb_period:
                    display_pb_chart(stored_pb_df, stored_pb_ref, stored_symbol)
            
            if st.button("Tải P/B", key="load_pb"):
                if not symbol_pb:
                    st.warning("Vui lòng nhập mã cổ phiếu.")
                else:
                    with st.spinner(f"Đang tải dữ liệu P/B cho {symbol_pb}..."):
                        try:
                            # Get date range based on selected period
                            if pb_period == "Tùy chỉnh":
                                start_date = pb_start_date
                                end_date = pb_end_date
                            else:
                                start_date, end_date = get_date_range(pb_period)
                            
                            pb_df = cached_get_pb_pe(symbol_pb, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                            if pb_df is None or pb_df.empty:
                                st.warning("Không có dữ liệu P/B trả về cho cổ phiếu này.")
                            else:
                                # normalize and sort
                                pb_df['date'] = pd.to_datetime(pb_df['date'])
                                pb_df = pb_df.sort_values('date')

                                # get reference thresholds using ref_pb_pe (shared function)
                                pb_ref = None
                                try:
                                    pb_ref, _ = cached_ref_pb_pe(symbol_pb, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                                except Exception as e:
                                    st.warning(f"Không lấy được giá trị tham chiếu từ ref_pb_pe: {e}")

                                # Store in session state (include period)
                                st.session_state.pb_chart_data = (symbol_pb, pb_period, pb_df, pb_ref)
                                
                                # Display the chart
                                display_pb_chart(pb_df, pb_ref, symbol_pb)
                        except Exception as e:
                            st.error(f"Lỗi khi tải dữ liệu P/B: {e}")

        # Valuation -> P/E implementation
        with tab_pe:
            st.subheader("P/E: Historical series and benchmarks")
            
            # Create columns for symbol input and period selection
            col1, col2 = st.columns([1, 1])
            with col1:
                # reuse same symbol input so user doesn't need to retype when switching tabs
                symbol_pe = st.text_input("Mã cổ phiếu (ví dụ: VCI)", value="", max_chars=10, key="val_symbol_pe")
            with col2:
                pe_period = st.selectbox(
                    "Chọn khoảng thời gian",
                    ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm", "Tùy chỉnh"],
                    index=3,  # Default to "1 năm"
                    key="pe_period"
                )
            
            # If custom period is selected, show date inputs
            if pe_period == "Tùy chỉnh":
                pe_date_col1, pe_date_col2 = st.columns(2)
                with pe_date_col1:
                    pe_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=365),
                        key="pe_start"
                    )
                with pe_date_col2:
                    pe_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="pe_end"
                    )
            
            # Display stored chart if exists
            if st.session_state.pe_chart_data is not None:
                stored_symbol, stored_period, stored_pe_df, stored_pe_ref = st.session_state.pe_chart_data
                if stored_symbol == symbol_pe and stored_period == pe_period:
                    display_pe_chart(stored_pe_df, stored_pe_ref, stored_symbol)
            
            if st.button("Tải P/E", key="load_pe"):
                if not symbol_pe:
                    st.warning("Vui lòng nhập mã cổ phiếu.")
                else:
                    with st.spinner(f"Đang tải dữ liệu P/E cho {symbol_pe}..."):
                        try:
                            # Get date range based on selected period
                            if pe_period == "Tùy chỉnh":
                                start_date = pe_start_date
                                end_date = pe_end_date
                            else:
                                start_date, end_date = get_date_range(pe_period)
                            
                            pe_df = cached_get_pb_pe(symbol_pe, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                            if pe_df is None or pe_df.empty:
                                st.warning("Không có dữ liệu P/E trả về cho cổ phiếu này.")
                            else:
                                # normalize and sort
                                pe_df['date'] = pd.to_datetime(pe_df['date'])
                                pe_df = pe_df.sort_values('date')

                                # get reference thresholds using ref_pb_pe (shared function)
                                pe_ref = None
                                try:
                                    _, pe_ref = cached_ref_pb_pe(symbol_pe, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
                                except Exception as e:
                                    st.warning(f"Không lấy được giá trị tham chiếu từ ref_pb_pe: {e}")

                                # Store in session state (include period)
                                st.session_state.pe_chart_data = (symbol_pe, pe_period, pe_df, pe_ref)
                                
                                # Display the chart
                                display_pe_chart(pe_df, pe_ref, symbol_pe)
                        except Exception as e:
                            st.error(f"Lỗi khi tải dữ liệu P/E: {e}")

        # Valuation -> PEG implementation
        with tab_peg:
            # Cached wrapper to avoid repeated API calls
            @st.cache_data(ttl=300, show_spinner=False)  # 5 minutes cache for testing
            def cached_get_peg(symbol):
                return get_peg(symbol)

            st.subheader("PEG: Price/Earnings to Growth")
            symbol = st.text_input("Mã cổ phiếu (ví dụ: VCI)", value="", max_chars=10, key="val_symbol_peg")
            if st.button("Tính PEG", key="load_peg"):
                if not symbol:
                    st.warning("Vui lòng nhập mã cổ phiếu.")
                else:
                    with st.spinner(f"Đang tính PEG cho {symbol}..."):
                        try:
                            peg_data = cached_get_peg(symbol)
                            continue_processing = True  # Default to True, set to False if issues found
                            
                            if peg_data is None:
                                st.warning("Không thể tính PEG cho cổ phiếu này. Có thể do:")
                                st.markdown("""
                                - Mã cổ phiếu không tồn tại hoặc không có dữ liệu
                                - Không có dữ liệu dự báo EPS tăng trưởng
                                - Dữ liệu P/E không hợp lệ
                                - Tốc độ tăng trưởng EPS bằng 0
                                """)
                                continue_processing = False
                            else:
                                # Extract data with validation
                                try:
                                    # Check if required keys exist
                                    if not all(k in peg_data for k in ['peg_ratio', 'pe_ratio', 'eps_growth']):
                                        st.warning("Dữ liệu PEG không đầy đủ.")
                                        st.info(f"Dữ liệu trả về: {peg_data}")
                                        continue_processing = False
                                    else:
                                        # Handle None values - convert to float safely
                                        peg_value = float(peg_data['peg_ratio']) if peg_data['peg_ratio'] is not None else None
                                        pe_ratio = float(peg_data['pe_ratio']) if peg_data['pe_ratio'] is not None else None
                                        eps_growth = float(peg_data['eps_growth']) if peg_data['eps_growth'] is not None else None
                                        
                                        # Get data source if available
                                        data_source = peg_data.get('data_source', None)
                                        if data_source:
                                            st.caption(f"📊 Nguồn dữ liệu EPS: **{data_source}**")

                                        # Validate data values
                                        if peg_value is None:
                                            # PEG is None - check if it's due to negative growth or missing data
                                            note = peg_data.get('note', '')
                                            if 'âm' in note:
                                                # EPS growth is negative - show specific message
                                                st.warning(note)
                                            else:
                                                # Missing or invalid data
                                                st.warning("Dữ liệu PEG không hợp lệ. Có thể API không trả về đủ dữ liệu EPS.")
                                                st.info(f"Chi tiết: {note}")
                                            
                                            # Show debug info
                                            with st.expander("Debug: Xem chi tiết lỗi"):
                                                st.write("PEG Data trả về:", peg_data)
                                            continue_processing = False
                                        else:
                                            # Data is valid, continue with processing
                                            continue_processing = True
                                       
                                except (KeyError, ValueError, TypeError) as e:
                                    st.error(f"Lỗi khi xử lý dữ liệu PEG: {e}")
                                    continue_processing = False
                                
                            # Only proceed with display if data is valid
                            if continue_processing:
                                # Check if EPS growth is positive for PEG interpretation
                                eps_growth_positive = eps_growth > 0
                                
                                # Only show gauge chart if EPS growth is positive
                                if eps_growth_positive:
                                    # Create gauge chart for PEG
                                    fig_gauge = go.Figure()
                                    
                                    # Define ranges for PEG interpretation
                                    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
                                    
                                    # Create the gauge
                                    fig_gauge.add_trace(go.Indicator(
                                        mode="gauge+number+delta",
                                        value=peg_value,
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        title={'text': "PEG Ratio"},
                                        delta={'reference': 1.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                                        gauge={
                                            'axis': {'range': [0, 3], 'dtick': 0.5},
                                            'bar': {'color': "darkblue"},
                                            'steps': [
                                                {'range': [0, 1], 'color': colors[0], 'name': 'Undervalued'},
                                                {'range': [1, 2], 'color': colors[1], 'name': 'Fairly Valued'},
                                                {'range': [2, 3], 'color': colors[2], 'name': 'Overvalued'}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': 2
                                            }
                                        }
                                    ))
                                    
                                    # Add interpretation text
                                    if peg_value < 1:
                                        interpretation = "Cổ phiếu có thể đang bị định giá thấp"
                                        color = "green"
                                    elif peg_value <= 2:
                                        interpretation = "Cổ phiếu được định giá hợp lý"
                                        color = "blue"
                                    else:
                                        interpretation = "Cổ phiếu có thể đang bị định giá cao"
                                        color = "red"
                                    
                                    st.plotly_chart(fig_gauge, width='stretch')
                                    
                                    # Display interpretation
                                    st.markdown(f"<p style='color:{color}; font-size: 18px; font-weight: bold;'>{interpretation}</p>", unsafe_allow_html=True)
                                    
                                    # Display additional information
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("PEG Ratio", f"{peg_value:.2f}")
                                    col2.metric("Ngưỡng hợp lý", "1.5")
                                    col3.metric("Ngưỡng cảnh báo", "2.0")
                                else:
                                    # EPS growth is negative - show warning
                                    st.warning("⚠️ Tăng trưởng EPS âm - Không thể tính PEG có ý nghĩa")
                                    st.info("PEG chỉ có ý nghĩa khi công ty có tốc độ tăng trưởng EPS dương. Với EPS growth âm, PEG sẽ không phản ánh đúng giá trị định giá.")
                                
                                # Add data section below the gauge (show for both cases)
                                st.subheader("📊 Dữ liệu chi tiết")
                                
                                # Create columns for key metrics
                                data_col1, data_col2, data_col3 = st.columns(3)
                                
                                # Display key metrics
                                data_col1.metric("P/E gần nhất", f"{pe_ratio:.2f}")
                                data_col2.metric("Tăng trưởng EPS dự báo", f"{eps_growth:.2f}%")
                                # Get additional data from peg_data
                                eps_current = peg_data.get('eps_current')
                                eps_forward = peg_data.get('eps_forward')
                                if eps_current is not None and eps_forward is not None:
                                    data_col3.metric("EPS hiện tại / dự phóng", f"{float(eps_current):,.0f} / {float(eps_forward):,.0f}")
                                else:
                                    data_col3.metric("Nguồn dữ liệu", "Vietcap")
                                
                                # Add explanation
                                with st.expander("Giải thích PEG"):
                                    st.markdown("""
                                    **PEG (Price/Earnings to Growth)** là chỉ số đánh giá giá trị của cổ phiếu so với tốc độ tăng trưởng lợi nhuận.
                                    
                                    **Công thức:** PEG = P/E / Tốc độ tăng trưởng EPS (%)
                                                                    
                                    **Giải thích:**
                                    - **PEG < 1:** Cổ phiếu có thể bị định giá thấp
                                    - **1 ≤ PEG ≤ 2:** Cổ phiếu được định giá hợp lý
                                    - **PEG > 2:** Cổ phiếu có thể bị định giá cao
                                    
                                    **Lưu ý:** PEG chỉ có ý nghĩa khi công ty có tốc độ tăng trưởng EPS dương.
                                    """)
                                # End of continue_processing if block
                                
                        except Exception as e:
                            st.error(f"Lỗi khi tính PEG: {e}")
                            st.markdown(f"*Chi tiết lỗi: {type(e).__name__}: {str(e)}*")
                            # Try to get more detailed error information
                            try:
                                # Attempt to call get_peg directly to get detailed error info
                                detailed_result = get_peg(symbol)
                                if detailed_result is None:
                                    st.info("Hàm get_peg() đã trả về None - không có dữ liệu hợp lệ")
                                else:
                                    st.info("Hàm get_peg() đã trả về dữ liệu nhưng có lỗi xử lý ở dashboard")
                            except Exception as detailed_e:
                                st.markdown(f"*Lỗi chi tiết từ hàm get_peg: {type(detailed_e).__name__}: {str(detailed_e)}*")

        # Valuation -> Price implementation (show stock price with Fireant valuation)
        with tab_price:
            st.subheader("💰 Giá cổ phiếu và Định giá Fireant")
            
            # Create columns for symbol input and period selection
            price_col1, price_col2 = st.columns([1, 1])
            with price_col1:
                symbol_price = st.text_input("Mã cổ phiếu (ví dụ: SSI, VNM, FPT)", value="", max_chars=10, key="val_symbol_price")
            with price_col2:
                price_period = st.selectbox(
                    "Chọn khoảng thời gian",
                    ["1 tháng", "3 tháng", "6 tháng", "1 năm", "2 năm", "Tùy chỉnh"],
                    index=3,  # Default to "1 năm"
                    key="price_period"
                )
            
            # If custom period is selected, show date inputs
            if price_period == "Tùy chỉnh":
                price_date_col1, price_date_col2 = st.columns(2)
                with price_date_col1:
                    price_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=365),
                        key="price_start"
                    )
                with price_date_col2:
                    price_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="price_end"
                    )
            
            if st.button("Tải dữ liệu Giá", key="load_price"):
                if not symbol_price:
                    st.warning("Vui lòng nhập mã cổ phiếu.")
                else:
                    with st.spinner(f"Đang tải dữ liệu giá cho {symbol_price}..."):
                        try:
                            # Get date range based on selected period
                            if price_period == "Tùy chỉnh":
                                start_date = price_start_date
                                end_date = price_end_date
                            else:
                                start_date, end_date = get_date_range(price_period)
                            
                            # Get stock history data (need at least 250 days for MA200)
                            price_df = get_stock_history(symbol_price, period="day", end_date=end_date.strftime('%Y-%m-%d'), count_back=365)
                            
                            if price_df is None or price_df.empty:
                                st.warning("Không có dữ liệu giá trả về cho cổ phiếu này.")
                            else:
                                # Filter by date range
                                price_df['time'] = pd.to_datetime(price_df['time'])
                                price_df = price_df[(price_df['time'] >= pd.to_datetime(start_date)) & (price_df['time'] <= pd.to_datetime(end_date))]
                                price_df = price_df.sort_values('time')
                                
                                # Calculate MA200 using pandas_ta
                                price_df['ma200'] = ta.sma(price_df['close'], length=200)
                                
                                if price_df.empty:
                                    st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
                                else:
                                    # Get Fireant valuation
                                    fireant_val = fireant_valuation(symbol_price)
                                    
                                    # Get analyst price targets
                                    analyst_targets = analyst_price_targets(symbol_price)
                                    
                                    # Create the chart with dark theme
                                    fig_price = go.Figure()
                                    
                                    # Add shaded background between high and low (analyst targets)
                                    if analyst_targets is not None and analyst_targets.get('high') is not None and analyst_targets.get('low') is not None:
                                        # Add gradient-like effect with multiple rects
                                        fig_price.add_shape(
                                            type="rect",
                                            xref="x", yref="y",
                                            x0=price_df['time'].min(),
                                            x1=price_df['time'].max(),
                                            y0=analyst_targets['low'],
                                            y1=analyst_targets['high'],
                                            fillcolor="rgba(0, 200, 100, 0.15)",
                                            layer="below",
                                            line_width=0,
                                        )
                                        # Add border lines for high and low
                                        fig_price.add_shape(
                                            type="line",
                                            xref="x", yref="y",
                                            x0=price_df['time'].min(),
                                            x1=price_df['time'].max(),
                                            y0=analyst_targets['high'],
                                            y1=analyst_targets['high'],
                                            line=dict(color="rgba(0, 200, 100, 0.6)", width=1, dash="solid"),
                                        )
                                        fig_price.add_shape(
                                            type="line",
                                            xref="x", yref="y",
                                            x0=price_df['time'].min(),
                                            x1=price_df['time'].max(),
                                            y0=analyst_targets['low'],
                                            y1=analyst_targets['low'],
                                            line=dict(color="rgba(0, 200, 100, 0.6)", width=1, dash="solid"),
                                        )
                                    
                                    # Add close price line with area fill
                                    fig_price.add_trace(go.Scatter(
                                        x=price_df['time'], 
                                        y=price_df['close'], 
                                        mode='lines', 
                                        name='Giá đóng cửa',
                                        line=dict(color='#00BFFF', width=2.5),
                                        fill='tozeroy',
                                        fillcolor='rgba(0, 191, 255, 0.1)',
                                    ))
                                    
                                    # Add MA200 line
                                    if price_df['ma200'].notna().any():
                                        fig_price.add_trace(go.Scatter(
                                            x=price_df['time'],
                                            y=price_df['ma200'],
                                            mode='lines',
                                            name='MA200',
                                            line=dict(color='#FFD700', width=2, dash='solid'),
                                        ))
                                    
                                    # Add mean line (analyst price target)
                                    if analyst_targets is not None and analyst_targets.get('mean') is not None:
                                        fig_price.add_trace(go.Scatter(
                                            x=[price_df['time'].min(), price_df['time'].max()],
                                            y=[analyst_targets['mean'], analyst_targets['mean']],
                                            mode='lines',
                                            name='Mean (PT)',
                                            line=dict(color='#FF8C00', width=2, dash='dot'),
                                        ))
                                    
                                    # Add median line (analyst price target)
                                    if analyst_targets is not None and analyst_targets.get('median') is not None:
                                        fig_price.add_trace(go.Scatter(
                                            x=[price_df['time'].min(), price_df['time'].max()],
                                            y=[analyst_targets['median'], analyst_targets['median']],
                                            mode='lines',
                                            name='Median (PT)',
                                            line=dict(color='#9370DB', width=2, dash='dot'),
                                        ))
                                    
                                    # Add Fireant valuation line
                                    if fireant_val is not None:
                                        fig_price.add_trace(go.Scatter(
                                            x=[price_df['time'].min(), price_df['time'].max()],
                                            y=[fireant_val, fireant_val],
                                            mode='lines',
                                            name='Fireant Valuation',
                                            line=dict(color='#FF4444', width=2.5, dash='dash'),
                                        ))
                                    
                                    # Build annotations list
                                    annotations = []
                                    
                                    # Add annotations for analyst targets
                                    if analyst_targets is not None:
                                        if analyst_targets.get('high') is not None:
                                            annotations.append(dict(
                                                x=0.02, y=analyst_targets['high'],
                                                xref='paper', yref='y',
                                                text=f"High: {analyst_targets['high']:,.0f}",
                                                showarrow=False,
                                                xanchor='left',
                                                font=dict(color='rgba(0, 200, 100, 0.8)', size=10),
                                                bgcolor='rgba(0, 0, 0, 0.5)',
                                                borderpad=3
                                            ))
                                        if analyst_targets.get('low') is not None:
                                            annotations.append(dict(
                                                x=0.02, y=analyst_targets['low'],
                                                xref='paper', yref='y',
                                                text=f"Low: {analyst_targets['low']:,.0f}",
                                                showarrow=False,
                                                xanchor='left',
                                                font=dict(color='rgba(0, 200, 100, 0.8)', size=10),
                                                bgcolor='rgba(0, 0, 0, 0.5)',
                                                borderpad=3
                                            ))
                                        if analyst_targets.get('mean') is not None:
                                            annotations.append(dict(
                                                x=0.98, y=analyst_targets['mean'],
                                                xref='paper', yref='y',
                                                text=f"Mean: {analyst_targets['mean']:,.0f}",
                                                showarrow=False,
                                                xanchor='right',
                                                font=dict(color='#FF8C00', size=10, family="Arial Black"),
                                                bgcolor='rgba(0, 0, 0, 0.6)',
                                                borderpad=3
                                            ))
                                        if analyst_targets.get('median') is not None:
                                            annotations.append(dict(
                                                x=0.98, y=analyst_targets['median'],
                                                xref='paper', yref='y',
                                                text=f"Median: {analyst_targets['median']:,.0f}",
                                                showarrow=False,
                                                xanchor='right',
                                                font=dict(color='#9370DB', size=10, family="Arial Black"),
                                                bgcolor='rgba(0, 0, 0, 0.6)',
                                                borderpad=3,
                                                yshift=20
                                            ))
                                    
                                    # Add Fireant annotation
                                    if fireant_val is not None:
                                        annotations.append(dict(
                                            x=0.5, y=fireant_val,
                                            xref='paper', yref='y',
                                            text=f"🔥 Fireant: {fireant_val:,.0f}",
                                            showarrow=True,
                                            arrowhead=2,
                                            arrowcolor='#FF4444',
                                            xanchor='center',
                                            yanchor='bottom',
                                            font=dict(color='#FF4444', size=11, family="Arial Black"),
                                            bgcolor='rgba(0, 0, 0, 0.7)',
                                            borderpad=4
                                        ))
                                    
                                    # Display metrics with better layout
                                    latest_price = price_df['close'].iloc[-1]
                                    
                                    if analyst_targets is not None:
                                        # Calculate upside/downside
                                        mean_pt = analyst_targets.get('mean')
                                        median_pt = analyst_targets.get('median')
                                        
                                        # Main metrics row
                                        st.markdown("**💰 Tổng quan Giá**")
                                        metric_row1 = st.columns([1.8, 1.5, 1.5, 1.8, 1.5])
                                        
                                        # Current Price
                                        metric_row1[0].metric(
                                            "Giá hiện tại", 
                                            f"{latest_price:,.0f}"
                                        )
                                        
                                        # Mean PT
                                        if mean_pt is not None:
                                            mean_diff = ((mean_pt - latest_price) / latest_price) * 100
                                            metric_row1[1].metric(
                                                "Mean (PT)", 
                                                f"{mean_pt:,.0f}",
                                                delta=f"{mean_diff:+.1f}%",
                                                delta_color="normal"
                                            )
                                        else:
                                            metric_row1[1].metric("Mean (PT)", "N/A")
                                        
                                        # Median PT
                                        if median_pt is not None:
                                            median_diff = ((median_pt - latest_price) / latest_price) * 100
                                            metric_row1[2].metric(
                                                "Median (PT)", 
                                                f"{median_pt:,.0f}",
                                                delta=f"{median_diff:+.1f}%",
                                                delta_color="normal"
                                            )
                                        else:
                                            metric_row1[2].metric("Median (PT)", "N/A")
                                        
                                        # PT Range
                                        if analyst_targets.get('high') is not None and analyst_targets.get('low') is not None:
                                            metric_row1[3].metric(
                                                "PT Range", 
                                                f"{analyst_targets['low']:,.0f} - {analyst_targets['high']:,.0f}"
                                            )
                                        else:
                                            metric_row1[3].metric("PT Range", "N/A")
                                        
                                        # Fireant
                                        if fireant_val is not None:
                                            fireant_diff = ((fireant_val - latest_price) / latest_price) * 100
                                            metric_row1[4].metric(
                                                "Fireant", 
                                                f"{fireant_val:,.0f}",
                                                delta=f"{fireant_diff:+.1f}%",
                                                delta_color="normal"
                                            )
                                        else:
                                            metric_row1[4].metric("Fireant", "N/A")
                                    else:
                                        price_col1, price_col2, price_col3 = st.columns(3)
                                        price_col1.metric("Giá đóng cửa mới nhất", f"{latest_price:.2f}")
                                        if fireant_val is not None:
                                            price_col2.metric("Định giá Fireant", f"{fireant_val:.2f}")
                                            diff = latest_price - fireant_val
                                            diff_pct = (diff / fireant_val) * 100 if fireant_val != 0 else 0
                                            price_col3.metric("Chênh lệch", f"{diff:.2f} ({diff_pct:.1f}%)", 
                                                             delta_color="inverse" if diff >= 0 else "normal")
                                        else:
                                            price_col2.metric("Định giá Fireant", "N/A")
                                            price_col3.metric("Chênh lệch", "N/A")
                                    
                                    # Update chart layout with dark theme
                                    fig_price.update_layout(
                                        title=dict(
                                            text=f"<b>📈 Biểu đồ giá cổ phiếu {symbol_price}</b> <span style='font-size:12px'>({start_date} - {end_date})</span>",
                                            x=0.5,
                                            xanchor='center',
                                            font=dict(size=18, color='white')
                                        ),
                                        xaxis_title=dict(text='<b>Ngày</b>', font=dict(color='white')),
                                        yaxis_title=dict(text='<b>Giá (VND)</b>', font=dict(color='white')),
                                        height=550,
                                        hovermode='x unified',
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h", 
                                            yanchor="bottom", 
                                            y=1.08, 
                                            xanchor="center", 
                                            x=0.5,
                                            font=dict(color='white', size=11),
                                            bgcolor='rgba(0,0,0,0.3)'
                                        ),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(30,30,30,0.5)',
                                        margin=dict(t=120, l=60, r=40, b=60),
                                        xaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(255,255,255,0.1)',
                                            tickfont=dict(color='white'),
                                            title_font=dict(color='white')
                                        ),
                                        yaxis=dict(
                                            showgrid=True,
                                            gridcolor='rgba(255,255,255,0.1)',
                                            tickfont=dict(color='white'),
                                            title_font=dict(color='white')
                                        ),
                                        annotations=annotations
                                    )
                                    st.plotly_chart(fig_price, use_container_width=True)
                                    
                                    # Check if analyst targets has valid data
                                    has_analyst_data = analyst_targets is not None and (
                                        analyst_targets.get('high') is not None or 
                                        analyst_targets.get('low') is not None or 
                                        analyst_targets.get('mean') is not None or 
                                        analyst_targets.get('median') is not None
                                    )
                                    
                                    # Show detailed summary
                                    st.markdown("---")
                                    summary_col1, summary_col2 = st.columns(2)
                                    
                                    with summary_col1:
                                        if has_analyst_data:
                                            st.markdown("### 📊 Analyst Price Targets")
                                            pt_data = []
                                            if analyst_targets.get('low'):
                                                pt_data.append(f"**Low:** {analyst_targets['low']:,.0f} VND")
                                            if analyst_targets.get('high'):
                                                pt_data.append(f"**High:** {analyst_targets['high']:,.0f} VND")
                                            if analyst_targets.get('mean'):
                                                pt_data.append(f"**Mean:** {analyst_targets['mean']:,.0f} VND")
                                            if analyst_targets.get('median'):
                                                pt_data.append(f"**Median:** {analyst_targets['median']:,.0f} VND")
                                            for item in pt_data:
                                                st.markdown(f"• {item}")
                                        else:
                                            st.markdown("### 📊 Analyst Price Targets")
                                            st.info("ℹ️ Không có dữ liệu Analyst Price Targets cho mã cổ phiếu này.")
                                            st.caption("Các nhà phân tích chưa đưa ra mục tiêu giá cho cổ phiếu này.")
                                    
                                    with summary_col2:
                                        if fireant_val is not None:
                                            st.markdown("### 🔥 Fireant Valuation")
                                            st.markdown(f"**Giá trị:** {fireant_val:,.0f} VND")
                                            if latest_price and fireant_val:
                                                diff_val = fireant_val - latest_price
                                                diff_pct = (diff_val / latest_price) * 100
                                                st.markdown(f"**Chênh lệch:** {diff_val:+,.0f} VND ({diff_pct:+.1f}%)")
                                                if diff_val > 0:
                                                    st.success("⬆️ Fireant định giá cao hơn giá thị trường")
                                                else:
                                                    st.warning("⬇️ Fireant định giá thấp hơn giá thị trường")
                                        else:
                                            st.markdown("### 🔥 Fireant Valuation")
                                            st.info("ℹ️ Không có thông tin định giá từ Fireant")
                                            st.caption("Hiện tại FireAnt chỉ cung cấp thông tin định giá các cổ phiếu của doanh nghiệp thông thường (không phải ngân hàng, công ty chứng khoán, quỹ).")
                                    
                                    with st.expander("📋 Xem dữ liệu giá chi tiết"):
                                        st.dataframe(price_df, use_container_width=True)
                                        st.download_button("💾 Tải xuống CSV", price_df.to_csv(index=False), f"price_{symbol_price}.csv", "text/csv")
                        except Exception as e:
                            st.error(f"Lỗi khi tải dữ liệu giá: {e}")

        # Features section for Định giá menu
        with tab_features:
            st.subheader("🚀 Tính năng Định giá Nổi bật")
            
            features_col1, features_col2 = st.columns(2)
            
            with features_col1:
                st.markdown("""
                ### 📈 Phân tích Định giá Cổ phiếu
                
                **P/B Ratio (Price-to-Book)**
                - So sánh giá thị trường với giá trị sổ sách kế toán
                - Đánh giá giá trị nội tại của công ty
                - So sánh với mức trung bình ngành (Sector Average)
                - Hiển thị delta (% thay đổi so với ngưỡng)
                - Đường ngưỡng trong legend với giá trị cụ thể
                
                **P/E Ratio (Price-to-Earnings)**
                - Đánh giá giá trị dựa trên khả năng sinh lời
                - So sánh với mức trung bình thị trường và ngành
                - Phân tích xu hướng P/E theo thời gian
                - Hiển thị delta (% thay đổi so với ngưỡng)
                
                **PEG Ratio (Price/Earnings to Growth)**
                - Kết hợp tốc độ tăng trưởng để đánh giá giá trị
                - PEG = P/E / Tốc độ tăng trưởng EPS (%)
                - Giúp phát hiện cổ phiếu bị định giá thấp/hợp lý/cao
                - EPS hiện tại (TTM) và EPS forward dự báo
                """)
            
            with features_col2:
                st.markdown("""
                ### 💰 Định giá Price & Analyst Targets
                
                **Fireant Valuation (Định giá Fireant)**
                - Lấy dữ liệu định giá từ Fireant API
                - Giá trị định giá theo phương pháp tổng hợp (Composed Price)
                - Cập nhật theo thời gian thực
                
                **Analyst Price Targets**
                - Dự báo giá từ các chuyên gia phân tích (valueinvesting.io)
                - High/Low/Mean/Median target prices
                - So sánh với giá hiện tại để tìm cơ hội
                
                ### 🔧 Công cụ Hỗ trợ Phân tích
                
                **Biểu đồ Tương tác**
                - Biểu đồ P/B, P/E, PEG, Price theo thời gian
                - Đường ngưỡng tham chiếu hiển thị trong legend
                - Giao diện dark theme hiện đại
                
                **Phân tích Đa chiều**
                - So sánh với ngành và thị trường
                - Phân tích xu hướng dài hạn
                - Delta metrics hiển thị % chênh lệch
                
                **Dữ liệu Thời gian Thực**
                - Kết nối API Fireant, FiinTrade, valueinvesting.io
                - Cập nhật định kỳ tự động
                """)
            
            st.markdown("---")
    
    elif co_phieu_submenu == "Phân loại giao dịch":
        # Import investor_type function
        from stock_data.stock_data import investor_type, get_stock_history
        
        st.subheader("👥 Phân loại Giao dịch Cổ phiếu")
        
        # Custom CSS for tabs styling
        st.markdown("""
        <style>
        /* Tab styling for Phân loại giao dịch */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            background-color: #1e1e1e;
            border-radius: 8px 8px 0px 0px;
            padding: 4px 4px 0px 4px;
            border-bottom: 1px solid #2d2d2d;
        }
        .stTabs [data-baseweb="tab"] {
            flex: 1;
            height: 52px;
            font-size: 17px;
            font-weight: 700;
            color: #b0b0b0;
            background-color: #2d2d2d;
            border-radius: 6px 6px 0px 0px;
            border: 1px solid #3d3d3d;
            border-bottom: none;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0e0e0e;
            color: #ff6b6b;
            border-color: #ff6b6b;
            box-shadow: 0px -2px 0px 0px #ff6b6b inset;
        }
        .stTabs [data-baseweb="tab-list"] {
            box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        }
        /* Tab content area */
        .stTabs [data-testid="stTabContent"] {
            background-color: #0e0e0e;
            border-radius: 0px 0px 8px 8px;
            border: 1px solid #2d2d2d;
            border-top: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Stock symbol input and period selection
        input_col1, input_col2, input_col3 = st.columns([1, 2, 1])
        
        with input_col1:
            stock_symbol = st.text_input(
                "Mã cổ phiếu",
                value="",
                key="stock_symbol_input",
                help="Nhập mã cổ phiếu (ví dụ: VCB, FPT, VIC...)"
            ).upper()
        
        with input_col2:
            # Period selection
            stock_period = st.selectbox(
                "Chọn khoảng thời gian",
                ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tùy chỉnh"],
                index=2,
                key="stock_period"
            )
            
            if stock_period == "Tùy chỉnh":
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    stock_start_date = st.date_input(
                        "Ngày bắt đầu",
                        value=datetime.now().date() - timedelta(days=180),
                        key="stock_start_date"
                    )
                with date_col2:
                    stock_end_date = st.date_input(
                        "Ngày kết thúc",
                        value=datetime.now().date(),
                        key="stock_end_date"
                    )
            else:
                stock_start_date, stock_end_date = get_date_range(stock_period)
        
        with input_col3:
            st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y')}")
            load_button = st.button("📊 Tải dữ liệu", key="load_stock_investor_data", width='stretch')
        
        # Initialize session state for loaded symbol
        if "loaded_stock_symbol" not in st.session_state:
            st.session_state.loaded_stock_symbol = ""
        if "loaded_stock_start_date" not in st.session_state:
            st.session_state.loaded_stock_start_date = None
        if "loaded_stock_end_date" not in st.session_state:
            st.session_state.loaded_stock_end_date = None
        
        # Handle load button click
        if load_button and stock_symbol:
            st.session_state.loaded_stock_symbol = stock_symbol
            st.session_state.loaded_stock_start_date = stock_start_date
            st.session_state.loaded_stock_end_date = stock_end_date
            
            # Create cache key based on parameters
            stock_investor_key = f"stock_investor_{stock_symbol}_{stock_start_date}_{stock_end_date}"
            
            # Load data
            if stock_investor_key not in st.session_state and stock_investor_key not in st.session_state.jobs:
                st.session_state.jobs[stock_investor_key] = st.session_state.executor.submit(
                    investor_type,
                    symbol=stock_symbol,
                    start_date=stock_start_date.strftime('%Y-%m-%d'),
                    end_date=stock_end_date.strftime('%Y-%m-%d')
                )
                st.session_state[f"{stock_investor_key}_start_time"] = time.time()
        
        # Check if we have a loaded symbol
        if st.session_state.loaded_stock_symbol:
            current_symbol = st.session_state.loaded_stock_symbol
            current_start = st.session_state.loaded_stock_start_date
            current_end = st.session_state.loaded_stock_end_date
            
            # Create cache key
            stock_investor_key = f"stock_investor_{current_symbol}_{current_start}_{current_end}"
            
            # Get job status
            stock_inv_status, stock_investor_df = get_job_status(stock_investor_key)
        else:
            stock_inv_status = "not_started"
            stock_investor_df = None
            current_symbol = ""
            current_start = None
            current_end = None
        
        # Create tabs for investor classification
        tab_tong_gia_tri_cp, tab_tu_doanh_cp, tab_ca_nhan_trong_nuoc_cp, tab_to_chuc_trong_nuoc_cp, tab_ca_nhan_nuoc_ngoai_cp, tab_to_chuc_nuoc_ngoai_cp = st.tabs([
            "💰 Tổng giá trị", 
            "🏢 Tự doanh", 
            "👤 Cá nhân trong nước", 
            "🏛️ Tổ chức trong nước", 
            "🌍 Cá nhân nước ngoài", 
            "🌐 Tổ chức nước ngoài"
        ])
        
        # --- Tổng giá trị Tab ---
        with tab_tong_gia_tri_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"💰 Tổng giá trị Giao dịch - {current_symbol}")
                
                if stock_inv_status == "running":
                    st.info(f"Đang tải dữ liệu phân loại giao dịch cho {current_symbol}...")
                elif stock_inv_status == "error":
                    st.error(f"Lỗi khi tải dữ liệu: {stock_investor_df}")
                elif stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    try:
                        # Get stock history for close price
                        stock_price_df = get_stock_history(
                            symbol=current_symbol,
                            period="day",
                            end_date=current_end.strftime('%Y-%m-%d'),
                            count_back=(current_end - current_start).days + 30
                        )
                        
                        # Create the chart
                        fig_stock_inv = make_subplots(
                            rows=1, cols=1,
                            specs=[[{"secondary_y": True}]],
                            vertical_spacing=0.05
                        )
                        
                        # Define professional colors for each investor type
                        colors = {
                            'Tự doanh ròng': '#e74c3c',
                            'Cá nhân trong nước ròng': '#3498db',
                            'Tổ chức trong nước ròng': '#2ecc71',
                            'Cá nhân nước ngoài ròng': '#9b59b6',
                            'Tổ chức nước ngoài ròng': '#f39c12'
                        }
                        
                        # Add stacked bar traces for investor types
                        investor_columns = [
                            'Tự doanh ròng',
                            'Cá nhân trong nước ròng',
                            'Tổ chức trong nước ròng',
                            'Cá nhân nước ngoài ròng',
                            'Tổ chức nước ngoài ròng'
                        ]
                        
                        for col in investor_columns:
                            if col in stock_investor_df.columns:
                                stock_investor_df[col] = pd.to_numeric(stock_investor_df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                                
                                bar_hover = [
                                    f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                                    f"{col}: {stock_investor_df[col].iloc[i]:,.0f}"
                                    for i in range(len(stock_investor_df))
                                ]
                                
                                fig_stock_inv.add_trace(go.Bar(
                                    x=stock_investor_df['Ngày'],
                                    y=stock_investor_df[col],
                                    name=col,
                                    marker_color=colors.get(col, '#888888'),
                                    opacity=0.85,
                                    text=bar_hover,
                                    hoverinfo='text'
                                ), secondary_y=False)
                        
                        # Add close price line if stock data is available
                        if stock_price_df is not None and not stock_price_df.empty:
                            stock_price_df['time'] = pd.to_datetime(stock_price_df['time'])
                            start_datetime = pd.to_datetime(current_start)
                            end_datetime = pd.to_datetime(current_end)
                            stock_price_df_filtered = stock_price_df[
                                (stock_price_df['time'] >= start_datetime) & 
                                (stock_price_df['time'] <= end_datetime)
                            ]
                            
                            if not stock_price_df_filtered.empty:
                                close_hover = [
                                    f"Ngày: {row['time'].strftime('%Y-%m-%d')}<br>" +
                                    f"Giá đóng cửa: {row['close']:,.2f}"
                                    for _, row in stock_price_df_filtered.iterrows()
                                ]
                                
                                fig_stock_inv.add_trace(go.Scatter(
                                    x=stock_price_df_filtered['time'],
                                    y=stock_price_df_filtered['close'],
                                    mode='lines',
                                    name='Giá đóng cửa',
                                    line=dict(color='#1abc9c', width=2.5),
                                    text=close_hover,
                                    hoverinfo='text'
                                ), secondary_y=True)
                        
                        fig_stock_inv.update_layout(
                            title=dict(
                                text=f'💰 Phân loại Giao dịch - {current_symbol}',
                                y=0.98,
                                x=0.5,
                                xanchor='center',
                                yanchor='top',
                                font=dict(size=18)
                            ),
                            barmode='relative',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị giao dịch ròng (tỷ đồng)',
                            yaxis2_title='Giá đóng cửa',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.08,
                                xanchor='center',
                                x=0.5
                            ),
                            margin=dict(l=60, r=60, t=100, b=60),
                            bargap=0.2,
                            bargroupgap=0.1
                        )
                        
                        fig_stock_inv.update_xaxes(showgrid=False, zeroline=False)
                        fig_stock_inv.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', secondary_y=False)
                        fig_stock_inv.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_stock_inv, width='stretch')
                        
                        with st.expander("📊 Xem dữ liệu chi tiết"):
                            st.dataframe(stock_investor_df, width='stretch')
                            st.download_button(
                                "Tải xuống dữ liệu CSV",
                                stock_investor_df.to_csv(index=False),
                                f"investor_type_{current_symbol}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.csv",
                                "text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Lỗi khi tạo biểu đồ: {e}")
                elif stock_inv_status == "completed":
                    st.warning(f"Không có dữ liệu phân loại giao dịch cho {current_symbol}.")
        
        # --- Tự doanh Tab ---
        with tab_tu_doanh_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"🏢 Giao dịch Tự doanh Ròng - {current_symbol}")
                
                if stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    col_name = 'Tự doanh ròng'
                    if col_name in stock_investor_df.columns:
                        stock_investor_df[col_name] = pd.to_numeric(stock_investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        
                        cumulative_values = stock_investor_df[col_name].cumsum()
                        fig_td_cp = make_subplots(specs=[[{"secondary_y": True}]])
                        colors_td = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stock_investor_df[col_name]]
                        
                        bar_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị ròng: {stock_investor_df[col_name].iloc[i]:,.0f}<br>" +
                            f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_td_cp.add_trace(go.Bar(
                            x=stock_investor_df['Ngày'],
                            y=stock_investor_df[col_name],
                            name='Giá trị ròng',
                            marker_color=colors_td,
                            opacity=0.85,
                            text=bar_hover,
                            hoverinfo='text'
                        ), secondary_y=False)
                        
                        line_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_td_cp.add_trace(go.Scatter(
                            x=stock_investor_df['Ngày'],
                            y=cumulative_values,
                            mode='lines+markers',
                            name='Giá trị tích lũy',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=4),
                            text=line_hover,
                            hoverinfo='text'
                        ), secondary_y=True)
                        
                        fig_td_cp.update_layout(
                            title=f'Tự doanh Ròng - {current_symbol} ({current_start.strftime("%Y-%m-%d")} đến {current_end.strftime("%Y-%m-%d")})',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị ròng',
                            yaxis2_title='Giá trị tích lũy',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=60, t=80, b=60),
                            bargap=0.15
                        )
                        
                        fig_td_cp.update_xaxes(showgrid=False)
                        fig_td_cp.update_yaxes(showgrid=False, secondary_y=False)
                        fig_td_cp.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_td_cp, width='stretch')
                    else:
                        st.warning("Không có dữ liệu Tự doanh ròng.")
                else:
                    st.info("Đang tải dữ liệu...")
        
        # --- Cá nhân trong nước Tab ---
        with tab_ca_nhan_trong_nuoc_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"👤 Giao dịch Cá nhân trong nước Ròng - {current_symbol}")
                
                if stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    col_name = 'Cá nhân trong nước ròng'
                    if col_name in stock_investor_df.columns:
                        stock_investor_df[col_name] = pd.to_numeric(stock_investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        
                        cumulative_values = stock_investor_df[col_name].cumsum()
                        fig_cntn_cp = make_subplots(specs=[[{"secondary_y": True}]])
                        colors_cntn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stock_investor_df[col_name]]
                        
                        bar_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị ròng: {stock_investor_df[col_name].iloc[i]:,.0f}<br>" +
                            f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_cntn_cp.add_trace(go.Bar(
                            x=stock_investor_df['Ngày'],
                            y=stock_investor_df[col_name],
                            name='Giá trị ròng',
                            marker_color=colors_cntn,
                            opacity=0.85,
                            text=bar_hover,
                            hoverinfo='text'
                        ), secondary_y=False)
                        
                        line_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_cntn_cp.add_trace(go.Scatter(
                            x=stock_investor_df['Ngày'],
                            y=cumulative_values,
                            mode='lines+markers',
                            name='Giá trị tích lũy',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=4),
                            text=line_hover,
                            hoverinfo='text'
                        ), secondary_y=True)
                        
                        fig_cntn_cp.update_layout(
                            title=f'Cá nhân trong nước Ròng - {current_symbol} ({current_start.strftime("%Y-%m-%d")} đến {current_end.strftime("%Y-%m-%d")})',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị ròng',
                            yaxis2_title='Giá trị tích lũy',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=60, t=80, b=60),
                            bargap=0.15
                        )
                        
                        fig_cntn_cp.update_xaxes(showgrid=False)
                        fig_cntn_cp.update_yaxes(showgrid=False, secondary_y=False)
                        fig_cntn_cp.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_cntn_cp, width='stretch')
                    else:
                        st.warning("Không có dữ liệu Cá nhân trong nước ròng.")
                else:
                    st.info("Đang tải dữ liệu...")
        
        # --- Tổ chức trong nước Tab ---
        with tab_to_chuc_trong_nuoc_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"🏛️ Giao dịch Tổ chức trong nước Ròng - {current_symbol}")
                
                if stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    col_name = 'Tổ chức trong nước ròng'
                    if col_name in stock_investor_df.columns:
                        stock_investor_df[col_name] = pd.to_numeric(stock_investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        
                        cumulative_values = stock_investor_df[col_name].cumsum()
                        fig_tctn_cp = make_subplots(specs=[[{"secondary_y": True}]])
                        colors_tctn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stock_investor_df[col_name]]
                        
                        bar_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị ròng: {stock_investor_df[col_name].iloc[i]:,.0f}<br>" +
                            f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_tctn_cp.add_trace(go.Bar(
                            x=stock_investor_df['Ngày'],
                            y=stock_investor_df[col_name],
                            name='Giá trị ròng',
                            marker_color=colors_tctn,
                            opacity=0.85,
                            text=bar_hover,
                            hoverinfo='text'
                        ), secondary_y=False)
                        
                        line_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_tctn_cp.add_trace(go.Scatter(
                            x=stock_investor_df['Ngày'],
                            y=cumulative_values,
                            mode='lines+markers',
                            name='Giá trị tích lũy',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=4),
                            text=line_hover,
                            hoverinfo='text'
                        ), secondary_y=True)
                        
                        fig_tctn_cp.update_layout(
                            title=f'Tổ chức trong nước Ròng - {current_symbol} ({current_start.strftime("%Y-%m-%d")} đến {current_end.strftime("%Y-%m-%d")})',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị ròng',
                            yaxis2_title='Giá trị tích lũy',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=60, t=80, b=60),
                            bargap=0.15
                        )
                        
                        fig_tctn_cp.update_xaxes(showgrid=False)
                        fig_tctn_cp.update_yaxes(showgrid=False, secondary_y=False)
                        fig_tctn_cp.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_tctn_cp, width='stretch')
                    else:
                        st.warning("Không có dữ liệu Tổ chức trong nước ròng.")
                else:
                    st.info("Đang tải dữ liệu...")
        
        # --- Cá nhân nước ngoài Tab ---
        with tab_ca_nhan_nuoc_ngoai_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"🌍 Giao dịch Cá nhân nước ngoài Ròng - {current_symbol}")
                
                if stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    col_name = 'Cá nhân nước ngoài ròng'
                    if col_name in stock_investor_df.columns:
                        stock_investor_df[col_name] = pd.to_numeric(stock_investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        
                        cumulative_values = stock_investor_df[col_name].cumsum()
                        fig_cnnn_cp = make_subplots(specs=[[{"secondary_y": True}]])
                        colors_cnnn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stock_investor_df[col_name]]
                        
                        bar_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị ròng: {stock_investor_df[col_name].iloc[i]:,.0f}<br>" +
                            f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_cnnn_cp.add_trace(go.Bar(
                            x=stock_investor_df['Ngày'],
                            y=stock_investor_df[col_name],
                            name='Giá trị ròng',
                            marker_color=colors_cnnn,
                            opacity=0.85,
                            text=bar_hover,
                            hoverinfo='text'
                        ), secondary_y=False)
                        
                        line_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_cnnn_cp.add_trace(go.Scatter(
                            x=stock_investor_df['Ngày'],
                            y=cumulative_values,
                            mode='lines+markers',
                            name='Giá trị tích lũy',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=4),
                            text=line_hover,
                            hoverinfo='text'
                        ), secondary_y=True)
                        
                        fig_cnnn_cp.update_layout(
                            title=f'Cá nhân nước ngoài Ròng - {current_symbol} ({current_start.strftime("%Y-%m-%d")} đến {current_end.strftime("%Y-%m-%d")})',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị ròng',
                            yaxis2_title='Giá trị tích lũy',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=60, t=80, b=60),
                            bargap=0.15
                        )
                        
                        fig_cnnn_cp.update_xaxes(showgrid=False)
                        fig_cnnn_cp.update_yaxes(showgrid=False, secondary_y=False)
                        fig_cnnn_cp.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_cnnn_cp, width='stretch')
                    else:
                        st.warning("Không có dữ liệu Cá nhân nước ngoài ròng.")
                else:
                    st.info("Đang tải dữ liệu...")
        
        # --- Tổ chức nước ngoài Tab ---
        with tab_to_chuc_nuoc_ngoai_cp:
            if not current_symbol:
                st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Tải dữ liệu' để xem biểu đồ.")
            else:
                st.subheader(f"🌐 Giao dịch Tổ chức nước ngoài Ròng - {current_symbol}")
                
                if stock_inv_status == "completed" and stock_investor_df is not None and not stock_investor_df.empty:
                    col_name = 'Tổ chức nước ngoài ròng'
                    if col_name in stock_investor_df.columns:
                        stock_investor_df[col_name] = pd.to_numeric(stock_investor_df[col_name].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        
                        cumulative_values = stock_investor_df[col_name].cumsum()
                        fig_tcnn_cp = make_subplots(specs=[[{"secondary_y": True}]])
                        colors_tcnn = ['#2ecc71' if v >= 0 else '#e74c3c' for v in stock_investor_df[col_name]]
                        
                        bar_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị ròng: {stock_investor_df[col_name].iloc[i]:,.0f}<br>" +
                            f"Tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_tcnn_cp.add_trace(go.Bar(
                            x=stock_investor_df['Ngày'],
                            y=stock_investor_df[col_name],
                            name='Giá trị ròng',
                            marker_color=colors_tcnn,
                            opacity=0.85,
                            text=bar_hover,
                            hoverinfo='text'
                        ), secondary_y=False)
                        
                        line_hover = [
                            f"Ngày: {stock_investor_df['Ngày'].iloc[i].strftime('%Y-%m-%d')}<br>" +
                            f"Giá trị tích lũy: {cumulative_values.iloc[i]:,.0f}"
                            for i in range(len(stock_investor_df))
                        ]
                        
                        fig_tcnn_cp.add_trace(go.Scatter(
                            x=stock_investor_df['Ngày'],
                            y=cumulative_values,
                            mode='lines+markers',
                            name='Giá trị tích lũy',
                            line=dict(color='#3498db', width=2.5),
                            marker=dict(size=4),
                            text=line_hover,
                            hoverinfo='text'
                        ), secondary_y=True)
                        
                        fig_tcnn_cp.update_layout(
                            title=f'Tổ chức nước ngoài Ròng - {current_symbol} ({current_start.strftime("%Y-%m-%d")} đến {current_end.strftime("%Y-%m-%d")})',
                            xaxis_title='Thời gian',
                            yaxis_title='Giá trị ròng',
                            yaxis2_title='Giá trị tích lũy',
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            margin=dict(l=60, r=60, t=80, b=60),
                            bargap=0.15
                        )
                        
                        fig_tcnn_cp.update_xaxes(showgrid=False)
                        fig_tcnn_cp.update_yaxes(showgrid=False, secondary_y=False)
                        fig_tcnn_cp.update_yaxes(showgrid=False, secondary_y=True)
                        
                        st.plotly_chart(fig_tcnn_cp, width='stretch')
                    else:
                        st.warning("Không có dữ liệu Tổ chức nước ngoài ròng.")
                else:
                    st.info("Đang tải dữ liệu...")
    
    # Custom CSS for dark theme compatible tabs spanning full width
    st.markdown("""
    <style>
    /* Dark theme professional tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #1e1e1e;
        border-radius: 8px 8px 0px 0px;
        padding: 4px 4px 0px 4px;
        border-bottom: 1px solid #2d2d2d;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        height: 52px;
        font-size: 17px;
        font-weight: 700;
        color: #b0b0b0;
        background-color: #2d2d2d;
        border-radius: 6px 6px 0px 0px;
        border: 1px solid #3d3d3d;
        border-bottom: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e0e0e;
        color: #ff6b6b;
        border-color: #ff6b6b;
        box-shadow: 0px -2px 0px 0px #ff6b6b inset;
    }
    .stTabs [data-baseweb="tab-list"] {
        box-shadow: 0px 2px 4px rgba(0,0,0,0.3);
    }
    /* Dark theme content area */
    .stTabs [data-testid="stTabContent"] {
        background-color: #0e0e0e;
        border-radius: 0px 0px 8px 8px;
        border: 1px solid #2d2d2d;
        border-top: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
elif main_menu == "Test":
    st.header("🧪 Test API")
    
    # Main navigation buttons
    render_main_navigation()
    st.markdown("---")
    
    st.subheader("Kiểm tra kết nối API")
    
    # Default values
    default_url = "https://valueinvesting.io/company/intrinsic_metric?ticker=SSI.VN"
    default_headers = '''
{
  "accept": "*/*",
  "accept-language": "en-US,en;q=0.9,vi;q=0.8",
  "content-type": "application/x-www-form-urlencoded",
  "priority": "u=1, i",
  "referer": "https://valueinvesting.io/HPG.VN/valuation/intrinsic-value",
  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
'''
    default_payload = "{}"
    
    # Input fields
    api_url = st.text_input("URL", value=default_url, key="api_url")
    
    col_method, col_timeout = st.columns([1, 1])
    with col_method:
        api_method = st.selectbox("Method", ["GET", "POST"], index=0, key="api_method")
    with col_timeout:
        api_timeout = st.number_input("Timeout (giây)", min_value=5, max_value=120, value=30, key="api_timeout")
    
    api_headers = st.text_area("Headers (JSON)", value=default_headers, height=200, key="api_headers")
    api_payload = st.text_area("Payload (JSON)", value=default_payload, height=100, key="api_payload")
    
    # Run button
    if st.button("🚀 Chạy Test API", key="test_api_btn"):
        with st.spinner("Đang kiểm tra API..."):
            try:
                import requests
                import pandas as pd
                
                # Parse headers
                try:
                    try:
                        headers = json.loads(api_headers)
                    except json.JSONDecodeError:
                        headers = eval(api_headers)
                except Exception as e:
                    st.error(f"❌ Headers không phải JSON/Python dict hợp lệ: {e}")
                    st.stop()
                
                # Parse payload
                try:
                    try:
                        payload = json.loads(api_payload)
                    except json.JSONDecodeError:
                        payload = eval(api_payload) if api_payload.strip() else {}
                except Exception as e:
                    st.error(f"❌ Payload không phải JSON hợp lệ: {e}")
                    st.stop()
                
                st.info(f"Đang gọi API: {api_url}")
                st.caption(f"Method: {api_method}, Timeout: {api_timeout}s")
                
                # Make request
                if api_method == "GET":
                    response = requests.get(api_url, headers=headers, data=payload, timeout=api_timeout)
                else:
                    response = requests.post(api_url, headers=headers, data=payload, timeout=api_timeout)
                
                st.divider()
                
                # Show response status
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check if data contains 'data' field (like valueinvesting.io API)
                        if 'data' in data:
                            df = pd.DataFrame(data['data'])
                            st.success(f"✓ API hoạt động - HTTP {response.status_code}")
                            st.write(f"**Số dòng dữ liệu:** {len(df)}")
                            st.dataframe(df)
                            
                            with st.expander("Xem JSON đầy đủ"):
                                st.json(data)
                        else:
                            st.success(f"✓ API hoạt động - HTTP {response.status_code}")
                            st.json(data)
                            
                    except json.JSONDecodeError:
                        st.success(f"✓ API hoạt động - HTTP {response.status_code}")
                        st.warning("⚠ Response không phải JSON hợp lệ")
                        st.text(response.text[:1000])
                else:
                    st.error(f"❌ API lỗi - HTTP {response.status_code}")
                    st.text(response.text[:500])
                    
                # Show response headers
                with st.expander("Xem Response Headers"):
                    st.json(dict(response.headers))
                    
            except requests.exceptions.Timeout:
                st.error(f"❌ Request timeout sau {api_timeout} giây")
            except requests.exceptions.ConnectionError as e:
                st.error(f"❌ Không thể kết nối đến server: {e}")
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                import traceback
                with st.expander("Xem chi tiết lỗi"):
                    st.code(traceback.format_exc())
    
    # Info section
    st.divider()
    with st.expander("📝 Hướng dẫn sử dụng"):
        st.markdown("""
        **Cách sử dụng:**
        1. Nhập URL của API cần test
        2. Chọn method (GET hoặc POST)
        3. Nhập headers dạng JSON **hoặc** Python dict (với dấu ')
        4. Nhập payload dạng JSON hoặc Python dict (cho POST)
        5. Nhấn nút "Chạy Test API"
        
        **Ví dụ headers (Python dict - dùng dấu '):**
        ```python
        {
            'accept': '*/*',
            'Cookie': 'token=abc123',
            'user-agent': 'Mozilla/5.0'
        }
        ```
        
        **Ví dụ headers (JSON - dùng dấu "):**
        ```json
        {
            "accept": "*/*",
            "Cookie": "token=abc123"
        }
        ```
        
        **Lưu ý:** Cookie trong headers có thể hết hạn sau một thời gian.
        """)

# --- Polling for Rerun ---
if "jobs" in st.session_state and st.session_state.jobs:
    time.sleep(1)
    st.rerun()