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
    Calculate 
