import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Smart Money Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Custom alert boxes */
    .alert-bullish {
        padding: 15px;
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 5px solid #00c853;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-bearish {
        padding: 15px;
        background-color: rgba(255, 82, 82, 0.1);
        border-left: 5px solid #ff5252;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-neutral {
        padding: 15px;
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1e1e2e;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Ticker header */
    .ticker-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px 25px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED DATA FETCHING FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'SPY')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d')
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Remove any rows with NaN values in critical columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        return data
        
    except Exception as e:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_data(ticker: str, period: str = "5d", interval: str = "30m") -> pd.DataFrame:
    """
    Fetch intraday data for Smart Money Index calculation.
    
    Args:
        ticker: Stock symbol
        period: Time period (max '60d' for intraday data)
        interval: Data interval ('1m', '5m', '15m', '30m', '1h')
    
    Returns:
        DataFrame with intraday OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        return data
        
    except Exception as e:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_options_data(ticker: str) -> tuple:
    """
    Fetch options chain data for the nearest expiry.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Tuple of (calls_df, puts_df, expiry_date) or (None, None, None) on failure
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get available expiration dates
        expirations = stock.options
        
        if not expirations or len(expirations) == 0:
            return None, None, None
        
        # Use nearest expiry
        nearest_expiry = expirations[0]
        
        # Fetch the option chain
        chain = stock.option_chain(nearest_expiry)
        
        return chain.calls, chain.puts, nearest_expiry
        
    except Exception as e:
        return None, None, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock information.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Dictionary with stock info
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception:
        return {}


# =============================================================================
# INDICATOR CALCULATION FUNCTIONS
# =============================================================================

def calculate_rvol(data: pd.DataFrame, lookback: int = 20) -> float:
    """
    Calculate Relative Volume (RVOL).
    
    RVOL = Current Volume / Average Volume (lookback period)
    
    Interpretation:
    - RVOL > 2.0: Unusually high volume, potential institutional activity
    - RVOL > 1.5: Above average interest
    - RVOL < 0.5: Unusually low volume
    
    Args:
        data: DataFrame with 'Volume' column
        lookback: Number of periods for average calculation
    
    Returns:
        RVOL ratio or None if insufficient data
    """
    if data is None or len(data) < lookback:
        return None
    
    # Calculate 20-day average volume (excluding current bar)
    avg_volume = data['Volume'].iloc[-(lookback+1):-1].mean()
    current_volume = data['Volume'].iloc[-1]
    
    if avg_volume == 0 or pd.isna(avg_volume):
        return None
    
    return current_volume / avg_volume


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price (VWAP).
    
    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    where Typical Price = (High + Low + Close) / 3
    
    Used by institutions as a benchmark for trade execution quality.
    Price above VWAP = bullish bias; Price below VWAP = bearish bias.
    
    Args:
        data: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
    
    Returns:
        Series of VWAP values
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    cumulative_tp_vol = (typical_price * data['Volume']).cumsum()
    cumulative_vol = data['Volume'].cumsum()
    
    # Avoid division by zero
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    
    return vwap


def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI).
    
    MFI is a volume-weighted RSI that measures buying and selling pressure.
    
    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price × Volume
    3. Money Flow Ratio = (14-period Positive MF) / (14-period Negative MF)
    4. MFI = 100 - (100 / (1 + Money Flow Ratio))
    
    Interpretation:
    - MFI > 80: Overbought (potential reversal down)
    - MFI < 20: Oversold (potential reversal up)
    - MFI with price divergence: Strong reversal signal
    
    Args:
        data: DataFrame with OHLCV data
        period: Lookback period (default: 14)
    
    Returns:
        Series of MFI values
    """
    if data is None or len(data) < period + 1:
        return pd.Series([np.nan] * len(data) if data is not None else [])
    
    # Calculate Typical Price
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    
    # Calculate Raw Money Flow
    raw_money_flow = typical_price * data['Volume']
    
    # Determine money flow direction
    price_change = typical_price.diff()
    
    # Positive and negative money flow
    positive_flow = raw_money_flow.where(price_change > 0, 0)
    negative_flow = raw_money_flow.where(price_change < 0, 0)
    
    # Rolling sums
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    # Money Flow Ratio (handle division by zero)
    mf_ratio = positive_mf / negative_mf.replace(0, np.nan)
    
    # Money Flow Index
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi


def calculate_smi(intraday_data: pd.DataFrame, ticker: str) -> tuple:
    """
    Calculate the Smart Money Index (SMI).
    
    The SMI is based on the observation that:
    - "Dumb money" (retail) trades at market open based on overnight news/emotion
    - "Smart money" (institutions) trades at market close based on careful analysis
    
    Formula:
    SMI = SMI_yesterday - (First 30 min gain) + (Last hour gain)
    
    Interpretation:
    - Rising SMI: Smart money accumulating
    - Falling SMI: Smart money distributing
    - Divergence from price: Potential reversal signal
    
    Args:
        intraday_data: DataFrame with 30-minute interval data
        ticker: Stock symbol (for reference)
    
    Returns:
        Tuple of (current_smi, smi_dataframe) or (None, None) on failure
    """
    if intraday_data is None or len(intraday_data) < 10:
        return None, None
    
    try:
        # Create a copy to avoid modifying original
        df = intraday_data.copy()
        
        # Extract date from datetime index
        df['Date'] = df.index.date
        df['Time'] = df.index.time
        
        # Get unique trading days
        dates = sorted(df['Date'].unique())
        
        if len(dates) < 2:
            return None, None
        
        smi_values = []
        smi_dates = []
        
        # Initialize SMI at 0
        prev_smi = 0
        
        for date in dates:
            day_data = df[df['Date'] == date].copy()
            
            if len(day_data) < 3:
                continue
            
            # Sort by time to ensure correct order
            day_data = day_data.sort_index()
            
            # First 30 minutes: first candle (if 30m interval)
            first_open = day_data['Open'].iloc[0]
            first_close = day_data['Close'].iloc[0]
            
            if first_open != 0:
                first_30_gain = ((first_close - first_open) / first_open) * 100
            else:
                first_30_gain = 0
            
            # Last hour: last 2 candles (if 30m interval)
            if len(day_data) >= 2:
                last_hour_open = day_data['Open'].iloc[-2]
                last_hour_close = day_data['Close'].iloc[-1]
            else:
                last_hour_open = day_data['Open'].iloc[-1]
                last_hour_close = day_data['Close'].iloc[-1]
            
            if last_hour_open != 0:
                last_hour_gain = ((last_hour_close - last_hour_open) / last_hour_open) * 100
            else:
                last_hour_gain = 0
            
            # SMI Formula: SMI = SMI_yesterday - first_30_gain + last_hour_gain
            smi = prev_smi - first_30_gain + last_hour_gain
            prev_smi = smi
            
            smi_values.append(smi)
            smi_dates.append(date)
        
        if smi_values:
            smi_df = pd.DataFrame({
                'Date': smi_dates,
                'SMI': smi_values
            })
            return smi_values[-1], smi_df
        
        return None, None
        
    except Exception as e:
        return None, None


def get_unusual_options(calls: pd.DataFrame, puts: pd.DataFrame, 
                        current_price: float = None) -> pd.DataFrame:
    """
    Identify unusual options activity (Volume > Open Interest).
    
    When options volume exceeds open interest, it suggests:
    - New positions being opened (not just closing existing positions)
    - Aggressive betting on price movement
    - Potential "informed" trading activity
    
    Args:
        calls: DataFrame of call options
        puts: DataFrame of put options
        current_price: Current stock price for reference
    
    Returns:
        DataFrame of unusual options sorted by Vol/OI ratio
    """
    unusual = []
    
    # Process calls
    if calls is not None and not calls.empty:
        for _, row in calls.iterrows():
            try:
                vol = row.get('volume', 0) or 0
                oi = row.get('openInterest', 0) or 0
                
                if oi > 0 and vol > oi:
                    strike = row.get('strike', 0)
                    
                    # Determine if ITM/OTM/ATM
                    moneyness = "—"
                    if current_price:
                        if strike < current_price * 0.98:
                            moneyness = "ITM"
                        elif strike > current_price * 1.02:
                            moneyness = "OTM"
                        else:
                            moneyness = "ATM"
                    
                    iv = row.get('impliedVolatility', None)
                    iv_display = f"{iv * 100:.1f}%" if iv and not pd.isna(iv) else "N/A"
                    
                    unusual.append({
                        'Strike': strike,
                        'Type': '📈 CALL',
                        'Moneyness': moneyness,
                        'Volume': int(vol),
                        'Open Interest': int(oi),
                        'Vol/OI': round(vol / oi, 2),
                        'Last Price': f"${row.get('lastPrice', 0):.2f}",
                        'IV': iv_display,
                        'Bid': row.get('bid', 0),
                        'Ask': row.get('ask', 0)
                    })
            except Exception:
                continue
    
    # Process puts
    if puts is not None and not puts.empty:
        for _, row in puts.iterrows():
            try:
                vol = row.get('volume', 0) or 0
                oi = row.get('openInterest', 0) or 0
                
                if oi > 0 and vol > oi:
                    strike = row.get('strike', 0)
                    
                    # Determine if ITM/OTM/ATM
                    moneyness = "—"
                    if current_price:
                        if strike > current_price * 1.02:
                            moneyness = "ITM"
                        elif strike < current_price * 0.98:
                            moneyness = "OTM"
                        else:
                            moneyness = "ATM"
                    
                    iv = row.get('impliedVolatility', None)
                    iv_display = f"{iv * 100:.1f}%" if iv and not pd.isna(iv) else "N/A"
                    
                    unusual.append({
                        'Strike': strike,
                        'Type': '📉 PUT',
                        'Moneyness': moneyness,
                        'Volume': int(vol),
                        'Open Interest': int(oi),
                        'Vol/OI': round(vol / oi, 2),
                        'Last Price': f"${row.get('lastPrice', 0):.2f}",
                        'IV': iv_display,
                        'Bid': row.get('bid', 0),
                        'Ask': row.get('ask', 0)
                    })
            except Exception:
                continue
    
    if unusual:
        df = pd.DataFrame(unusual)
        df = df.sort_values('Vol/OI', ascending=False)
        return df
    
    return pd.DataFrame()


def calculate_buy_sell_pressure(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate buy/sell pressure for volume coloring.
    
    Uses the relationship between close and open to determine pressure:
    - Close > Open: Buy pressure (bullish candle)
    - Close < Open: Sell pressure (bearish candle)
    
    Also calculates estimated buy/sell volume based on candle body position.
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        DataFrame with added pressure columns
    """
    df = data.copy()
    
    # Basic pressure based on candle color
    df['Pressure'] = np.where(df['Close'] >= df['Open'], 'Buy', 'Sell')
    
    # Calculate the position of close within the candle range
    candle_range = df['High'] - df['Low']
    close_position = (df['Close'] - df['Low']) / candle_range.replace(0, np.nan)
    
    # Estimated buy volume (volume × close position in range)
    df['Buy_Volume'] = df['Volume'] * close_position.fillna(0.5)
    df['Sell_Volume'] = df['Volume'] * (1 - close_position.fillna(0.5))
    
    return df


# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_candlestick_chart(data: pd.DataFrame, ticker: str, 
                             vwap: pd.Series) -> go.Figure:
    """
    Create an interactive candlestick chart with VWAP overlay and volume.
    
    Features:
    - Candlestick price chart
    - VWAP line overlay
    - Volume bars colored by buy/sell pressure
    - Interactive zoom and pan
    
    Args:
        data: DataFrame with OHLCV data
        ticker: Stock symbol for title
        vwap: Series of VWAP values
    
    Returns:
        Plotly Figure object
    """
    # Calculate buy/sell pressure for volume coloring
    df = calculate_buy_sell_pressure(data)
    
    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('', 'Volume (Smart Money Pressure)')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
            decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350'),
            whiskerwidth=0.5
        ),
        row=1, col=1
    )
    
    # VWAP line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vwap,
            mode='lines',
            name='VWAP',
            line=dict(color='#ffd54f', width=2, dash='dot'),
            hovertemplate='VWAP: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Current price line
    current_price = df['Close'].iloc[-1]
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.5,
        annotation_text=f"${current_price:.2f}",
        annotation_position="right",
        row=1, col=1
    )
    
    # Volume bars colored by pressure
    colors = ['#26a69a' if p == 'Buy' else '#ef5350' for p in df['Pressure']]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(
                color=colors,
                opacity=0.7,
                line=dict(width=0)
            ),
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add 20-period volume SMA
    vol_sma = df['Volume'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vol_sma,
            mode='lines',
            name='Vol SMA(20)',
            line=dict(color='#9e9e9e', width=1),
            hovertemplate='Avg Vol: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> - Price Action & Volume Analysis',
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        showline=True,
        linewidth=1,
        linecolor='gray'
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        row=2, col=1
    )
    
    return fig


def create_smi_chart(smi_df: pd.DataFrame) -> go.Figure:
    """
    Create a Smart Money Index trend chart.
    
    Args:
        smi_df: DataFrame with 'Date' and 'SMI' columns
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # SMI line with gradient fill
    fig.add_trace(
        go.Scatter(
            x=smi_df['Date'],
            y=smi_df['SMI'],
            mode='lines+markers',
            name='SMI',
            line=dict(color='#7c4dff', width=3),
            marker=dict(size=8, color='#7c4dff'),
            fill='tozeroy',
            fillcolor='rgba(124, 77, 255, 0.2)',
            hovertemplate='Date: %{x}<br>SMI: %{y:.2f}<extra></extra>'
        )
    )
    
    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text="Neutral",
        annotation_position="left"
    )
    
    # Add shading for positive/negative regions
    fig.update_layout(
        template='plotly_dark',
        height=280,
        margin=dict(l=20, r=20, t=30, b=30),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            title='SMI Value'
        ),
        hovermode='x'
    )
    
    return fig


def create_mfi_gauge(mfi_value: float) -> go.Figure:
    """
    Create a gauge chart for Money Flow Index.
    
    Args:
        mfi_value: Current MFI value (0-100)
    
    Returns:
        Plotly Figure object
    """
    # Determine color based on value
    if mfi_value > 80:
        bar_color = "#ef5350"  # Red - overbought
    elif mfi_value < 20:
        bar_color = "#26a69a"  # Green - oversold
    else:
        bar_color = "#ffd54f"  # Yellow - neutral
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mfi_value,
        title={'text': "Money Flow Index", 'font': {'size': 16}},
        number={'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(38, 166, 154, 0.3)'},
                {'range': [20, 80], 'color': 'rgba(255, 213, 79, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(239, 83, 80, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': mfi_value
            }
        }
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=200,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='margin-bottom: 0;'>💰 Smart Money Footprint Tracker</h1>
        <p style='color: #888; font-size: 1.1rem;'>
            Track institutional activity using volume analysis, money flow, and options data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    st.sidebar.markdown("## ⚙️ Dashboard Settings")
    
    # Ticker input
    default_tickers = "SPY, AAPL, TSLA, NVDA, QQQ"
    tickers_input = st.sidebar.text_input(
        "📊 Enter Tickers",
        value=default_tickers,
        help="Enter stock or ETF symbols separated by commas (e.g., AAPL, MSFT, SPY)"
    )
    
    # Timeframe selection
    st.sidebar.markdown("### ⏱️ Timeframe")
    timeframe = st.sidebar.selectbox(
        "Select Analysis Period",
        options=["1d", "5d", "1mo", "3mo", "6mo"],
        index=2,
        format_func=lambda x: {
            "1d": "1 Day (Intraday)",
            "5d": "5 Days",
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months"
        }.get(x, x)
    )
    
    # Interval mapping based on timeframe
    interval_map = {
        "1d": "5m",
        "5d": "15m",
        "1mo": "1d",
        "3mo": "1d",
        "6mo": "1d"
    }
    interval = interval_map[timeframe]
    
    st.sidebar.markdown("---")
    
    # Run analysis button
    run_analysis = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True
    )
    
    # ==========================================================================
    # HOW TO READ SECTION (SIDEBAR)
    # ==========================================================================
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📖 How to Read This Dashboard", expanded=False):
        st.markdown("""
        ### 🎯 Smart Money Indicators Explained
        
        ---
        
        **1. Smart Money Index (SMI)**
        
        Tracks the difference between opening (retail/emotional) and closing (institutional/calculated) price movements.
        
        - 📈 **Positive SMI**: Institutions accumulating
        - 📉 **Negative SMI**: Institutions distributing
        - ⚠️ **Divergence from price**: Potential reversal
        
        ---
        
        **2. Relative Volume (RVOL)**
        
        Current volume compared to 20-day average:
        
        - 🔥 **RVOL > 2.0**: Unusual activity (potential institutional trades)
        - 📊 **RVOL 1.0-2.0**: Normal activity
        - 😴 **RVOL < 0.5**: Low interest
        
        *Volume spikes without news often indicate informed trading*
        
        ---
        
        **3. VWAP (Volume-Weighted Avg Price)**
        
        Institutional benchmark price:
        
        - 🟢 **Price > VWAP**: Bullish control
        - 🔴 **Price < VWAP**: Bearish control
        - 💡 Institutions often accumulate below VWAP
        
        ---
        
        **4. Money Flow Index (MFI)**
        
        Volume-weighted RSI (buying/selling pressure):
        
        - 🔴 **MFI > 80**: Overbought (potential top)
        - 🟢 **MFI < 20**: Oversold (potential bottom)
        - ⚠️ **Divergence**: Strong reversal signal
        
        ---
        
        **5. Unusual Options Activity**
        
        Options where Volume > Open Interest suggest:
        
        - 📌 New positions being opened
        - 🎯 Aggressive directional bets
        - 🔍 Potential "informed" trading
        
        *High Vol/OI ratio = higher conviction*
        """)
    
    # Data attribution
    st.sidebar.markdown("---")
    st.sidebar.caption("📊 Data: Yahoo Finance (yfinance)")
    st.sidebar.caption("🛠️ Built with Streamlit & Plotly")
    st.sidebar.caption(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ==========================================================================
    # MAIN CONTENT AREA
    # ==========================================================================
    
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if not tickers:
        st.warning("⚠️ Please enter at least one ticker symbol.")
        return
    
    # Show instructions if analysis hasn't been run
    if not run_analysis:
        st.info("👈 Enter tickers in the sidebar and click **'Run Analysis'** to start tracking Smart Money footprints.")
        
        # Display educational content
        st.markdown("---")
        st.header("📚 Understanding Smart Money")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-box'>
            <h3>🎯 What is "Smart Money"?</h3>
            
            Smart money refers to capital controlled by:
            
            - **Institutional investors** (pension funds, mutual funds)
            - **Hedge funds** and proprietary traders
            - **Market makers** and specialists
            - **Corporate insiders** (within legal limits)
            
            These players typically have:
            - Access to better research and data
            - Larger teams of analysts
            - Faster execution capabilities
            - More capital to move markets
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='info-box'>
            <h3>🔍 Why Track Smart Money?</h3>
            
            **Volume Patterns**
            - Unusual volume without news = potential insider activity
            - Volume precedes price movement
            
            **Timing Patterns**
            - Retail trades emotionally at open
            - Institutions execute strategically at close
            
            **Options Flow**
            - Large options bets signal expectations
            - High volume/OI ratio = aggressive new positions
            
            *"Follow the money" - Large capital moves leave footprints*
            </div>
            """, unsafe_allow_html=True)
        
        # Example interpretation
        st.markdown("---")
        st.subheader("📊 Example Interpretation")
        
        example_cols = st.columns(4)
        
        with example_cols[0]:
            st.markdown("""
            **🔥 High RVOL + Rising Price**
            
            Strong institutional buying. 
            Momentum likely to continue.
            """)
        
        with example_cols[1]:
            st.markdown("""
            **📉 High RVOL + Falling Price**
            
            Institutional distribution.
            Avoid catching falling knife.
            """)
        
        with example_cols[2]:
            st.markdown("""
            **⚠️ MFI Divergence**
            
            Price up but MFI down = 
            Potential reversal ahead.
            """)
        
        with example_cols[3]:
            st.markdown("""
            **🎯 Unusual Call Activity**
            
            High Vol/OI on calls =
            Bullish expectations.
            """)
        
        return
    
    # ==========================================================================
    # ANALYSIS LOOP FOR EACH TICKER
    # ==========================================================================
    
    for ticker in tickers:
        st.markdown("---")
        
        # Ticker header
        st.markdown(f"""
        <div class='ticker-header'>
            <h2 style='margin: 0; color: white;'>📊 {ticker}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner(f"Fetching data for {ticker}..."):
            # Fetch all data
            data = fetch_stock_data(ticker, period=timeframe, interval=interval)
            intraday_data = fetch_intraday_data(ticker, period="5d", interval="30m")
            stock_info = fetch_stock_info(ticker)
            
            # Validate data
            if data is None or data.empty:
                st.error(f"""
                ❌ **Could not fetch data for {ticker}**
                
                Possible reasons:
                - Invalid ticker symbol
                - API rate limit reached
                - Network connectivity issues
                
                Please check the ticker symbol and try again.
                """)
                continue
            
            # ==================================================================
            # CALCULATE ALL INDICATORS
            # ==================================================================
            
            # Basic price info
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Open'].iloc[0]
            pct_change = ((current_price - prev_close) / prev_close) * 100
            
            # RVOL
            rvol = calculate_rvol(data)
            
            # VWAP
            vwap = calculate_vwap(data)
            current_vwap = vwap.iloc[-1]
            
            # MFI
            mfi = calculate_mfi(data)
            current_mfi = mfi.iloc[-1] if not mfi.empty and not pd.isna(mfi.iloc[-1]) else None
            
            # SMI
            smi_current, smi_df = calculate_smi(intraday_data, ticker)
            
            # ==================================================================
            # TOP ROW: METRIC CARDS
            # ==================================================================
            
            st.subheader("📈 Key Metrics")
            
            metric_cols = st.columns(4)
            
            # Price
            with metric_cols[0]:
                delta_color = "normal"
                st.metric(
                    label="💵 Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{pct_change:+.2f}%",
                    delta_color=delta_color
                )
            
            # RVOL
            with metric_cols[1]:
                if rvol is not None:
                    rvol_display = f"{rvol:.2f}x"
                    if rvol > 2.0:
                        rvol_delta = "🔥 HIGH ACTIVITY"
                        st.metric(
                            label="📊 Relative Volume",
                            value=rvol_display,
                            delta=rvol_delta,
                            delta_color="normal"
                        )
                    elif rvol > 1.5:
                        st.metric(
                            label="📊 Relative Volume",
                            value=rvol_display,
                            delta="Above Average",
                            delta_color="normal"
                        )
                    else:
                        st.metric(
                            label="📊 Relative Volume",
                            value=rvol_display,
                            delta="Normal",
                            delta_color="off"
                        )
                else:
                    st.metric(label="📊 Relative Volume", value="N/A")
            
            # MFI
            with metric_cols[2]:
                if current_mfi is not None:
                    mfi_display = f"{current_mfi:.1f}"
                    if current_mfi > 80:
                        mfi_delta = "🔴 Overbought"
                    elif current_mfi < 20:
                        mfi_delta = "🟢 Oversold"
                    else:
                        mfi_delta = "⚪ Neutral"
                    st.metric(
                        label="💰 Money Flow Index",
                        value=mfi_display,
                        delta=mfi_delta,
                        delta_color="off"
                    )
                else:
                    st.metric(label="💰 Money Flow Index", value="N/A")
            
            # VWAP Position
            with metric_cols[3]:
                vwap_pct = ((current_price - current_vwap) / current_vwap) * 100
                if current_price > current_vwap:
                    vwap_delta = f"🟢 +{vwap_pct:.2f}% Above"
                else:
                    vwap_delta = f"🔴 {vwap_pct:.2f}% Below"
                st.metric(
                    label="📍 Price vs VWAP",
                    value=f"${current_vwap:.2f}",
                    delta=vwap_delta,
                    delta_color="off"
                )
            
            # ==================================================================
            # MIDDLE ROW: CANDLESTICK CHART WITH VWAP
            # ==================================================================
            
            st.subheader("📉 Price & Volume Analysis")
            
            fig_candle = create_candlestick_chart(data, ticker, vwap)
            st.plotly_chart(fig_candle, use_container_width=True)
            
            # ==================================================================
            # BOTTOM ROW: OPTIONS AND SMI
            # ==================================================================
            
            bottom_cols = st.columns([1, 1])
            
            # Unusual Options
            with bottom_cols[0]:
                st.subheader("🎯 Unusual Options Activity")
                
                try:
                    options_result = fetch_options_data(ticker)
                    
                    if options_result[0] is not None:
                        calls, puts, expiry = options_result
                        unusual_options = get_unusual_options(calls, puts, current_price)
                        
                        if not unusual_options.empty:
                            st.caption(f"**Expiry: {expiry}** | Showing strikes where Volume > Open Interest")
                            
                            # Display table with formatting
                            display_cols = ['Strike', 'Type', 'Moneyness', 'Volume', 
                                          'Open Interest', 'Vol/OI', 'Last Price', 'IV']
                            
                            st.dataframe(
                                unusual_options[display_cols].head(10),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Summary
                            call_count = len(unusual_options[unusual_options['Type'] == '📈 CALL'])
                            put_count = len(unusual_options[unusual_options['Type'] == '📉 PUT'])
                            
                            if call_count > put_count * 1.5:
                                st.success(f"📈 Bullish bias: {call_count} unusual calls vs {put_count} puts")
                            elif put_count > call_count * 1.5:
                                st.error(f"📉 Bearish bias: {put_count} unusual puts vs {call_count} calls")
                            else:
                                st.info(f"⚖️ Mixed: {call_count} calls, {put_count} puts")
                        else:
                            st.info("No unusual options activity detected for the nearest expiry.")
                    else:
                        st.warning("Options data not available for this ticker.")
                        
                except Exception as e:
                    st.error(f"Error fetching options data: {str(e)}")
            
            # SMI Section
            with bottom_cols[1]:
                st.subheader("🧠 Smart Money Index (SMI)")
                
                if smi_current is not None and smi_df is not None:
                    # SMI value and trend
                    smi_col1, smi_col2 = st.columns([1, 2])
                    
                    with smi_col1:
                        if smi_current > 0:
                            st.metric(
                                label="Current SMI",
                                value=f"{smi_current:.2f}",
                                delta="Accumulation 📈",
                                delta_color="normal"
                            )
                        else:
                            st.metric(
                                label="Current SMI",
                                value=f"{smi_current:.2f}",
                                delta="Distribution 📉",
                                delta_color="inverse"
                            )
                    
                    with smi_col2:
                        # SMI chart
                        fig_smi = create_smi_chart(smi_df)
                        st.plotly_chart(fig_smi, use_container_width=True)
                    
                    # SMI interpretation
                    st.markdown("---")
                    st.markdown("**📊 SMI Interpretation:**")
                    
                    # Calculate SMI trend
                    if len(smi_df) >= 2:
                        smi_trend = smi_df['SMI'].iloc[-1] - smi_df['SMI'].iloc[0]
                        
                        if smi_current > 0 and smi_trend > 0:
                            st.markdown("""
                            <div class='alert-bullish'>
                            <strong>🟢 Bullish Signal</strong><br>
                            Smart money is actively accumulating. The positive and rising SMI suggests 
                            institutions are buying during the last hour of trading while retail sells at open.
                            </div>
                            """, unsafe_allow_html=True)
                        elif smi_current < 0 and smi_trend < 0:
                            st.markdown("""
                            <div class='alert-bearish'>
                            <strong>🔴 Bearish Signal</strong><br>
                            Smart money is distributing. The negative and falling SMI suggests institutions 
                            are selling into strength at close while retail buys emotionally at open.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class='alert-neutral'>
                            <strong>⚪ Mixed Signal</strong><br>
                            SMI shows mixed signals. Monitor for clearer directional bias. 
                            Watch for divergence between SMI trend and price action.
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("""
                    📊 **SMI requires intraday data**
                    
                    The Smart Money Index is calculated using 30-minute candles 
                    to track opening vs. closing price movements. 
                    
                    Limited intraday history may be available on the free tier.
                    """)
            
            # ==================================================================
            # ALERTS SECTION
            # ==================================================================
            
            st.subheader("⚠️ Smart Money Alerts")
            
            alerts = []
            
            # RVOL alerts
            if rvol is not None:
                if rvol > 3.0:
                    alerts.append({
                        'type': 'critical',
                        'icon': '🚨',
                        'title': f'Extreme Volume ({rvol:.2f}x)',
                        'message': 'Volume is 3x+ above average. Major institutional activity likely. Check for news or potential insider trading.'
                    })
                elif rvol > 2.0:
                    alerts.append({
                        'type': 'warning',
                        'icon': '🔥',
                        'title': f'High Volume ({rvol:.2f}x)',
                        'message': 'Unusual volume detected. Possible institutional buying or selling. Monitor price action closely.'
                    })
            
            # MFI alerts
            if current_mfi is not None:
                if current_mfi > 80:
                    alerts.append({
                        'type': 'warning',
                        'icon': '🔴',
                        'title': f'MFI Overbought ({current_mfi:.1f})',
                        'message': 'Large money inflow detected. Stock may be extended. Watch for potential reversal or consolidation.'
                    })
                elif current_mfi < 20:
                    alerts.append({
                        'type': 'opportunity',
                        'icon': '🟢',
                        'title': f'MFI Oversold ({current_mfi:.1f})',
                        'message': 'Large money outflow detected. Stock may be oversold. Watch for potential bounce or accumulation.'
                    })
            
            # VWAP alerts
            if current_price > current_vwap * 1.02:
                alerts.append({
                    'type': 'bullish',
                    'icon': '📈',
                    'title': 'Trading Above VWAP',
                    'message': f'Price is {((current_price/current_vwap)-1)*100:.1f}% above VWAP. Bullish pressure with buyers in control.'
                })
            elif current_price < current_vwap * 0.98:
                alerts.append({
                    'type': 'bearish',
                    'icon': '📉',
                    'title': 'Trading Below VWAP',
                    'message': f'Price is {(1-(current_price/current_vwap))*100:.1f}% below VWAP. Bearish pressure with sellers in control.'
                })
            
            # SMI alerts
            if smi_current is not None:
                if smi_current > 2:
                    alerts.append({
                        'type': 'bullish',
                        'icon': '🧠',
                        'title': 'Strong Smart Money Accumulation',
                        'message': 'SMI significantly positive. Institutions appear to be accumulating shares strategically.'
                    })
                elif smi_current < -2:
                    alerts.append({
                        'type': 'bearish',
                        'icon': '🧠',
                        'title': 'Strong Smart Money Distribution',
                        'message': 'SMI significantly negative. Institutions appear to be distributing shares.'
                    })
            
            # Display alerts
            if alerts:
                alert_cols = st.columns(min(len(alerts), 3))
                for i, alert in enumerate(alerts[:3]):
                    with alert_cols[i % 3]:
                        if alert['type'] in ['bullish', 'opportunity']:
                            st.success(f"**{alert['icon']} {alert['title']}**\n\n{alert['message']}")
                        elif alert['type'] in ['bearish', 'warning']:
                            st.warning(f"**{alert['icon']} {alert['title']}**\n\n{alert['message']}")
                        elif alert['type'] == 'critical':
                            st.error(f"**{alert['icon']} {alert['title']}**\n\n{alert['message']}")
                        else:
                            st.info(f"**{alert['icon']} {alert['title']}**\n\n{alert['message']}")
            else:
                st.info("✅ No significant Smart Money signals detected. Market activity appears normal.")
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p><strong>⚠️ Disclaimer</strong></p>
        <p style='font-size: 0.85rem;'>
        This dashboard is for educational and informational purposes only. 
        It does not constitute financial advice. All indicators are derived from publicly available data 
        and should be used in conjunction with other analysis methods. Past performance does not guarantee future results.
        Always do your own research before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
