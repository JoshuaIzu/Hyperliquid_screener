import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import requests
import time

# Set page config
st.set_page_config(
    page_title="Hyperliquid Futures Screener",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("📈 Hyperliquid Futures Market Screener")
st.markdown("Track and analyze cryptocurrency futures markets on Hyperliquid")

# Hyperliquid API Endpoints
HYPERLIQUID_INFO_API = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_EXCHANGE_API = "https://api.hyperliquid.xyz/exchange"

# Configuration
BASE_VOL = 0.35
VOL_MULTIPLIER = 1.5
MIN_LIQUIDITY = 50000  # Lower default to 50k
FUNDING_THRESHOLD = 60  # Annualized funding rate threshold (in basis points)

# Initialize session state if not already present
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'scanned_markets' not in st.session_state:
    st.session_state.scanned_markets = pd.DataFrame()
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# Database functions for state management
def load_state_from_db():
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    # Get active trades
    cursor.execute('SELECT value FROM app_state WHERE key = "active_trades"')
    active_trades_row = cursor.fetchone()
    active_trades = json.loads(active_trades_row[0]) if active_trades_row else {}
    
    # Get completed trades
    cursor.execute('SELECT value FROM app_state WHERE key = "completed_trades"')
    completed_trades_row = cursor.fetchone()
    completed_trades = json.loads(completed_trades_row[0]) if completed_trades_row else []
    
    conn.close()
    return active_trades, completed_trades

def save_state_to_db(active_trades, completed_trades):
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    
    # Save active trades
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('active_trades', json.dumps(active_trades)))
    
    # Save completed trades
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('completed_trades', json.dumps(completed_trades)))
    
    conn.commit()
    conn.close()

# Hyperliquid API Functions
@st.cache_data(ttl=30)
def fetch_hyperliquid_meta():
    """Fetch metadata about available markets on Hyperliquid"""
    try:
        response = requests.post(HYPERLIQUID_INFO_API, json={"type": "meta"})
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error fetching metadata: {str(e)}")
        return None

def parse_number_string(value):
    """Parse a number from API that could be a string with commas, scientific notation, or a float"""
    if value is None:
        return 0.0
        
    if isinstance(value, (int, float)):
        return float(value)
        
    if isinstance(value, str):
        # Remove commas
        value = value.replace(',', '')
        
        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            # If it fails (perhaps due to scientific notation or other format)
            try:
                # Try scientific notation
                return float(value)
            except ValueError:
                return 0.0
    
    return 0.0

@st.cache_data(ttl=60)
def fetch_all_markets():
    """Fetch all perpetual contracts from Hyperliquid with comprehensive market data"""
    try:
        # Store debug information
        debug_info = {}
        
        # Start with fetching the metadata to get the list of available coins
        meta_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "meta"})
        
        if meta_response.status_code != 200:
            st.error(f"Meta API request failed with status {meta_response.status_code}")
            return pd.DataFrame()
            
        meta_data = meta_response.json()
        debug_info['meta_data'] = str(type(meta_data))[:100]  # Just store the type to avoid large objects
        
        # Extract coins from universe list
        coins = []
        if 'universe' in meta_data and isinstance(meta_data['universe'], list):
            coins = [asset['name'] for asset in meta_data['universe'] if isinstance(asset, dict) and 'name' in asset]
        
        if not coins:
            st.error("No coins found in meta data")
            return pd.DataFrame()
        
        debug_info['coins'] = coins[:5]  # Store first 5 coins for debugging
        debug_info['total_coins'] = len(coins)
        
        # Get all stats
        stats_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "stats"})
        stats_data = {}
        debug_info['stats_response_code'] = stats_response.status_code
        
        if stats_response.status_code == 200:
            stats_result = stats_response.json()
            debug_info['stats_result_type'] = str(type(stats_result))
            
            # Store the raw JSON for the first few coins to debug parsing issues
            if isinstance(stats_result, list) and len(stats_result) > 0:
                debug_info['stats_sample'] = stats_result[:2]
                
                for stat in stats_result:
                    if isinstance(stat, dict) and 'coin' in stat:
                        stats_data[stat['coin']] = stat
        
        # Get all prices
        prices_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "oracle"})
        prices_data = {}
        debug_info['prices_response_code'] = prices_response.status_code
        
        if prices_response.status_code == 200:
            prices_result = prices_response.json()
            
            if isinstance(prices_result, list):
                for price in prices_result:
                    if isinstance(price, dict) and 'coin' in price and 'price' in price:
                        prices_data[price['coin']] = parse_number_string(price['price'])

        # Get funding rates
        funding_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "funding"})
        funding_data = {}
        
        if funding_response.status_code == 200:
            funding_result = funding_response.json()
            if isinstance(funding_result, list):
                for item in funding_result:
                    if isinstance(item, dict) and 'coin' in item and 'funding' in item:
                        # Convert funding rate to annualized basis points
                        funding_data[item['coin']] = parse_number_string(item['funding']) * 10000 * 365 * 24

        # Process market data for each coin
        markets = []
        skipped_markets = []
        
        for coin in coins:
            try:
                # Get price
                price = prices_data.get(coin, 0)
                
                # Skip coins without price data
                if price <= 0:
                    skipped_markets.append({
                        'symbol': coin,
                        'reason': 'No price data',
                        'min_required': MIN_LIQUIDITY
                    })
                    continue
                
                # Get stats for this coin
                stats = stats_data.get(coin, {})
                
                # Get volume and properly parse it
                volume_24h_raw = stats.get('volume24h', 0)
                volume_24h = parse_number_string(volume_24h_raw)
                
                # Get open interest
                open_interest_raw = stats.get('openInterest', 0)
                open_interest = parse_number_string(open_interest_raw)
                
                # Store info for debugging specific coins
                if coin in ['BTC', 'ETH', 'SOL']:
                    debug_info[f'{coin}_details'] = {
                        'price': price,
                        'volume_raw': volume_24h_raw,
                        'volume_parsed': volume_24h,
                        'oi_raw': open_interest_raw,
                        'oi_parsed': open_interest,
                        'min_required': MIN_LIQUIDITY
                    }
                
                # Skip low liquidity markets
                if volume_24h < MIN_LIQUIDITY:
                    skipped_markets.append({
                        'symbol': coin,
                        'volume24h': volume_24h,
                        'raw_volume': volume_24h_raw,
                        'min_required': MIN_LIQUIDITY
                    })
                    continue
                
                # Get funding rate
                funding_rate = funding_data.get(coin, 0)
                
                markets.append({
                    'symbol': coin,
                    'markPrice': price,
                    'lastPrice': price,
                    'fundingRate': funding_rate,
                    'openInterest': open_interest,
                    'volume24h': volume_24h,
                    'change24h': 0,  # Will be calculated from candles
                    'liquidityScore': volume_24h / open_interest if open_interest > 0 else 0
                })
                
            except Exception as e:
                debug_info[f'error_{coin}'] = str(e)
                continue
        
        # Create DataFrame with all market data
        df = pd.DataFrame(markets)
        
        # Save debug info
        debug_info['markets_count'] = len(markets)
        debug_info['skipped_count'] = len(skipped_markets)
        debug_info['skipped_markets'] = skipped_markets[:5]  # Show first 5 for brevity
        st.session_state.debug_info = debug_info
        
        if df.empty:
            if skipped_markets:
                st.warning(f"All {len(skipped_markets)} markets were skipped due to liquidity threshold. Consider lowering the minimum liquidity.")
            return df
        
        # Only calculate 24h change for top markets to avoid timeout
        for symbol in df['symbol'].head(5):
            try:
                candles = fetch_hyperliquid_candles(symbol, interval='1d', limit=2)
                if candles is not None and len(candles) >= 2:
                    prev_close = candles.iloc[-2]['close']
                    current_close = candles.iloc[-1]['close']
                    df.loc[df['symbol'] == symbol, 'change24h'] = (
                        (current_close - prev_close) / prev_close * 100
                    )
            except Exception as e:
                # Just skip if we can't get change data
                continue
        
        return df.sort_values('volume24h', ascending=False)
        
    except Exception as e:
        st.error(f"Error fetching markets: {str(e)}")
        st.session_state.debug_info['fetch_error'] = str(e)
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_hyperliquid_candles(coin, interval="1h", limit=50):
    """Fetch OHLCV data for a specific coin"""
    try:
        now = int(time.time())
        # Convert limit to appropriate time range based on interval
        if interval == "1h":
            start_time = now - (limit * 3600)
        elif interval == "4h":
            start_time = now - (limit * 3600 * 4)
        elif interval == "1d":
            start_time = now - (limit * 3600 * 24)
        else:
            start_time = now - (limit * 3600)  # Default to hourly
            
        response = requests.post(
            HYPERLIQUID_INFO_API, 
            json={
                "type": "candleSnapshot",
                "coin": coin,
                "interval": interval,
                "startTime": start_time,
                "endTime": now
            }
        )
        data = response.json()
        
        # Convert to DataFrame
        if data and len(data) > 0:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching candles for {coin}: {str(e)}")
        return None

# Fallback orderbook-based volume estimation
def estimate_volume_from_orderbook(coin, price):
    """Estimate 24h volume from orderbook liquidity"""
    try:
        # Get orderbook data
        response = requests.post(HYPERLIQUID_INFO_API, json={"type": "l2Book", "coin": coin})
        if response.status_code != 200:
            return 0
        
        orderbook = response.json()
        if not orderbook:
            return 0

        asks = orderbook.get('asks', [])
        bids = orderbook.get('bids', [])

        # Calculate liquidity within 2% of mid price
        liquid_asks = [
            [float(level[0]), float(level[1])]
            for level in asks 
            if float(level[0]) <= price * 1.02
        ]
        
        liquid_bids = [
            [float(level[0]), float(level[1])]
            for level in bids
            if float(level[0]) >= price * 0.98
        ]
        
        # Sum up the liquidity (size * price) within threshold
        ask_liquidity = sum(level[0] * level[1] for level in liquid_asks)
        bid_liquidity = sum(level[0] * level[1] for level in liquid_bids)
        
        # Estimate volume as total liquidity * typical daily turnover (4x)
        volume_24h = (ask_liquidity + bid_liquidity) * 4
        
        return volume_24h

    except Exception as e:
        return 0

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")
    BASE_VOL = st.slider("Base Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    VOL_MULTIPLIER = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
    
    # Updated MIN_LIQUIDITY with LOWER options
    liquidity_options = {
        "5,000 USD": 5000,
        "10,000 USD": 10000,
        "25,000 USD": 25000,
        "50,000 USD": 50000,
        "100,000 USD": 100000,
        "500,000 USD": 500000,
        "1,000,000 USD": 1000000
    }
    
    selected_liquidity = st.selectbox(
        "Minimum Liquidity",
        options=list(liquidity_options.keys()),
        index=2  # Default to 25,000
    )
    
    MIN_LIQUIDITY = liquidity_options[selected_liquidity]
    
    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        use_orderbook_fallback = st.checkbox(
            "Use orderbook fallback", 
            value=True, 
            help="If enabled, will estimate volume from orderbook when API volume data is missing"
        )
        
        fallback_threshold = st.number_input(
            "Fallback threshold (USD)",
            value=10000.0,
            min_value=1000.0,
            step=1000.0,
            help="Volume threshold below which orderbook fallback will be used"
        )
    
    FUNDING_THRESHOLD = st.slider("Funding Rate Threshold (basis points)", 10, 200, 60, 5)

class ForwardTester:
    def __init__(self):
        self.active_trades = {}
        self.completed_trades = []
        self.load_state()
        
    def load_state(self):
        """Load previous trading state from database"""
        try:
            self.active_trades, self.completed_trades = load_state_from_db()
        except Exception as e:
            st.warning(f"Could not load previous state: {str(e)}. Starting fresh.")
    
    def save_state(self):
        """Save current trading state to database"""
        try:
            save_state_to_db(self.active_trades, self.completed_trades)
        except Exception as e:
            st.error(f"Error saving state to database: {str(e)}")
    
    def execute_trades(self, signals):
        """Execute new trades based on signals"""
        executed = []
        for signal in signals:
            symbol = signal['Symbol']
            
            # Skip if already in active trades
            if symbol in self.active_trades:
                continue
                
            if signal['Signal'] != "HOLD":
                self.active_trades[symbol] = {
                    'Symbol': symbol,
                    'entry_price': signal['Price'],
                    'entry_time': datetime.now().isoformat(),
                    'direction': signal['Signal'],
                    'tp_price': float(signal['TP']) if signal['TP'] != "-" else None,
                    'sl_price': float(signal['SL']) if signal['SL'] != "-" else None,
                    'reason': signal['Reason'],
                    'funding_rate': signal.get('Funding Rate', 0),
                    'status': 'OPEN'
                }
                executed.append(f"📝 New {signal['Signal']} trade for {symbol} at {signal['Price']}")
        
        self.save_state()
        return executed
    
    def update_trades(self):
        """Check open trades for TP/SL hits"""
        to_remove = []
        updates = []
        
        # Get current prices for all coins
        markets_df = fetch_all_markets()
        prices_dict = {row['symbol']: row['markPrice'] for _, row in markets_df.iterrows()} if not markets_df.empty else {}
        
        for symbol, trade in self.active_trades.items():
            try:
                current_price = prices_dict.get(symbol)
                if not current_price:
                    continue
                    
                entry_price = trade['entry_price']
                
                # Check for TP/SL
                if trade['direction'] == "LONG":
                    if trade['tp_price'] and current_price >= trade['tp_price']:
                        trade['exit_reason'] = "TP Hit"
                    elif trade['sl_price'] and current_price <= trade['sl_price']:
                        trade['exit_reason'] = "SL Hit"
                elif trade['direction'] == "SHORT":
                    if trade['tp_price'] and current_price <= trade['tp_price']:
                        trade['exit_reason'] = "TP Hit"
                    elif trade['sl_price'] and current_price >= trade['sl_price']:
                        trade['exit_reason'] = "SL Hit"
                
                # Mark for removal if closed
                if 'exit_reason' in trade:
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['status'] = 'CLOSED'
                    trade['pct_change'] = ((current_price - entry_price)/entry_price)*100 if trade['direction'] == "LONG" else ((entry_price - current_price)/entry_price)*100
                    self.completed_trades.append(trade)
                    to_remove.append(symbol)
                    updates.append(f"✅ Trade closed: {symbol} | Reason: {trade['exit_reason']} | PnL: {trade['pct_change']:.2f}%")
            
            except Exception as e:
                updates.append(f"Error updating {symbol}: {str(e)}")
        
        # Remove closed trades
        for symbol in to_remove:
            self.active_trades.pop(symbol)
        
        self.save_state()
        return updates
    
    def get_performance_report(self):
        """Generate performance report"""
        if not self.completed_trades:
            return "No completed trades yet", pd.DataFrame()
            
        df = pd.DataFrame(self.completed_trades)
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds()/3600
        else:
            df['duration'] = 0
        
        stats = {
            'total_trades': len(df),
            'win_rate': len(df[df['pct_change'] > 0])/len(df) if len(df) > 0 else 0,
            'avg_pnl': df['pct_change'].mean() if 'pct_change' in df.columns else 0,
            'avg_duration_hours': df['duration'].mean() if 'duration' in df.columns else 0
        }
        
        return stats, df

    def reset_all_trades(self):
        """Reset all trades in the database"""
        self.active_trades = {}
        self.completed_trades = []
        self.save_state()
        return "All trades have been reset"

# Create a separate database connection for caching market data
def init_market_cache():
    conn = sqlite3.connect('market_cache.db')
    cursor = conn.cursor()
    
    # Create markets table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_cache (
        symbol TEXT PRIMARY KEY,
        data TEXT,
        timestamp INTEGER
    )
    ''')
    
    # Create OHLCV table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ohlcv_cache (
        symbol TEXT,
        timeframe TEXT,
        data TEXT,
        timestamp INTEGER,
        PRIMARY KEY (symbol, timeframe)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize cache tables
init_market_cache()

# Function to analyze and generate signals in manageable chunks
def generate_signals(markets_df):
    """Generate trading signals based on volume analysis and funding rates"""
    if markets_df is None or markets_df.empty:
        return []
    
    signals = []
    
    # Limit analysis to top markets by volume to avoid timeout
    # Use a smaller number if you're experiencing timeouts
    max_markets = 15
    top_markets = markets_df.head(max_markets)
    total_markets = len(top_markets)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (index, market) in enumerate(top_markets.iterrows()):
        symbol = market['symbol']
        status_text.text(f"Analyzing {symbol}... ({i+1}/{total_markets})")
        
        # Update progress bar at reasonable intervals
        if i % 2 == 0 or i == total_markets - 1:
            progress_bar.progress((i + 1) / total_markets)
        
        # Fetch hourly data - limited to 48h to reduce processing
        df = fetch_hyperliquid_candles(symbol, interval='1h', limit=48) 
        if df is None or len(df) < 6:  # Need at least a few hours of data
            continue
            
        # Calculate volume metrics
        avg_vol = df['volume'].mean()
        recent_vol = df['volume'].iloc[-1]
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
        
        signal = "HOLD"
        reason = ""
        tp = "-"
        sl = "-"
        
        # Get funding rate
        funding_rate = market['fundingRate']
        
        # Check volume surge and funding rate for signal
        if vol_surge >= VOL_MULTIPLIER and recent_vol > BASE_VOL * avg_vol:
            # Check price action for direction
            recent_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            
            # Consider funding rate in signal generation
            if recent_change > 0.01 and funding_rate < -FUNDING_THRESHOLD:
                # Long signal with negative funding (we get paid to hold long)
                signal = "LONG"
                reason = f"Vol surge {vol_surge:.2f}x with bullish price action and favorable funding rate ({funding_rate:.2f} bps)"
                tp = str(round(df['close'].iloc[-1] * 1.05, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 0.97, 4))  # 3% SL
            elif recent_change < -0.01 and funding_rate > FUNDING_THRESHOLD:
                # Short signal with positive funding (we get paid to hold short)
                signal = "SHORT"
                reason = f"Vol surge {vol_surge:.2f}x with bearish price action and favorable funding rate ({funding_rate:.2f} bps)"
                tp = str(round(df['close'].iloc[-1] * 0.95, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 1.03, 4))  # 3% SL
            elif recent_change > 0.01:
                # Regular long signal (without favorable funding)
                signal = "LONG"
                reason = f"Vol surge {vol_surge:.2f}x with bullish price action"
                tp = str(round(df['close'].iloc[-1] * 1.05, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 0.97, 4))  # 3% SL
            elif recent_change < -0.01:
                # Regular short signal (without favorable funding)
                signal = "SHORT"
                reason = f"Vol surge {vol_surge:.2f}x with bearish price action"
                tp = str(round(df['close'].iloc[-1] * 0.95, 4))  # 5% TP
                sl = str(round(df['close'].iloc[-1] * 1.03, 4))  # 3% SL
        
        signals.append({
            'Symbol': symbol,
            'Price': df['close'].iloc[-1],
            'Mark Price': market['markPrice'],
            'Signal': signal,
            'Volume 24h': market['volume24h'],
            'Open Interest': market['openInterest'],
            'Funding Rate': funding_rate,
            'Vol Surge': vol_surge,
            'Change 24h': market['change24h'],
            'Reason': reason,
            'TP': tp,
            'SL': sl
        })
    
    progress_bar.empty()
    status_text.empty()
    
    return signals

# Function to scan markets with better error handling 
def scan_markets():
    """Scan markets and store them in session state with comprehensive error handling"""
    # Clear previous markets and signals
    st.session_state.scanned_markets = pd.DataFrame()
    st.session_state.signals = []
    
    # First, fetch the raw market data
    try:
        with st.spinner("Fetching market data..."):
            markets_df = fetch_all_markets()
            
            if markets_df.empty:
                # Check debug info for clues
                if 'debug_info' in st.session_state:
                    debug = st.session_state.debug_info
                    if 'skipped_markets' in debug and debug['skipped_markets']:
                        skipped = debug['skipped_markets']
                        sample_market = skipped[0] if skipped else {}
                        st.warning(f"Found data but all markets were below the liquidity threshold. Sample: {sample_market}")
                        
                        # Show debugging info
                        if 'BTC_details' in debug:
                            st.info("Debug info for BTC:")
                            st.json(debug['BTC_details'])
                        
                        # Suggest lowering the threshold
                        current_threshold = MIN_LIQUIDITY
                        if current_threshold > 5000:  # If not already at lowest
                            lower_threshold = current_threshold / 2
                            st.info(f"Try lowering the minimum liquidity threshold to {lower_threshold:,.0f} USD")
                    else:
                        st.error("No market data found. Check API connection.")
                return
                
            # Store markets in session state
            st.session_state.scanned_markets = markets_df
            st.success(f"Found {len(markets_df)} markets meeting the minimum liquidity threshold of ${MIN_LIQUIDITY:,} USD.")
    
    except Exception as e:
        st.error(f"Error during market scan: {str(e)}")
        return
    
    # Then, generate signals in a separate step
    try:
        if not markets_df.empty:
            with st.spinner("Analyzing markets for trading signals..."):
                signals = generate_signals(markets_df)
                st.session_state.signals = signals
                
                actionable_count = len([s for s in signals if s['Signal'] != 'HOLD'])
                if actionable_count > 0:
                    st.success(f"Analysis complete. Found {actionable_count} actionable signals.")
                else:
                    st.info("Analysis complete. No actionable signals found with current parameters.")
    
    except Exception as e:
        st.error(f"Error during signal generation: {str(e)}")
        # At least we have the market data, even if signal generation failed

# Initialize the forward tester
tester = ForwardTester()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database", "Debug"])

# Tab 1: Market Scanner
with tab1:
    # Display current liquidity setting
    st.info(f"Current Minimum Liquidity: ${MIN_LIQUIDITY:,} USD")
    
    if st.button("Scan Markets", use_container_width=True):
        scan_markets()
    
    # Show raw markets data regardless of signal generation
    if not st.session_state.scanned_markets.empty:
        st.subheader("All Markets")
        
        # Format the data for display
        display_df = st.session_state.scanned_markets.copy()
        
        # Show the table with nice formatting
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                'symbol': st.column_config.TextColumn("Symbol"),
                'markPrice': st.column_config.NumberColumn("Mark Price", format="%.4f"),
                'volume24h': st.column_config.NumberColumn("24h Volume", format="$%.2f"),
                'openInterest': st.column_config.NumberColumn("Open Interest", format="$%.2f"),
                'fundingRate': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                'change24h': st.column_config.NumberColumn("24h Change", format="%.2f%%")
            }
        )
    
    # Show signals if available
    if st.session_state.signals:
        st.subheader("Trading Signals")
        signals_df = pd.DataFrame(st.session_state.signals)
        
        # Filter options
        signal_filter = st.multiselect("Filter by Signal", 
                                      options=['LONG', 'SHORT', 'HOLD'], 
                                      default=['LONG', 'SHORT'])
        
        if signal_filter:
            filtered_df = signals_df[signals_df['Signal'].isin(signal_filter)]
            
            # Display signals table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    'Price': st.column_config.NumberColumn("Price", format="%.4f"),
                    'Volume 24h': st.column_config.NumberColumn("Volume 24h", format="$%.2f"),
                    'Funding Rate': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                    'Vol Surge': st.column_config.NumberColumn("Volume Surge", format="%.2fx")
                }
            )
            
            # Execute trades button
            actionable_signals = [s for s in st.session_state.signals if s['Signal'] != "HOLD" and s['Signal'] in signal_filter]
            if actionable_signals and st.button("Execute Selected Signals"):
                results = tester.execute_trades(actionable_signals)
                for result in results:
                    st.success(result)

# Tab 2: Active Trades
with tab2:
    st.header("Active Trades")
    
    if st.button("Update Trades"):
        updates = tester.update_trades()
        for update in updates:
            st.info(update)
    
    if tester.active_trades:
        # Convert active trades to DataFrame for display
        active_df = pd.DataFrame.from_dict(tester.active_trades, orient='index')
        
        # Get current prices
        markets_df = fetch_all_markets()
        prices_dict = {row['symbol']: row['markPrice'] for _, row in markets_df.iterrows()} if not markets_df.empty else {}
        
        # Add current price and P&L columns
        for idx, row in active_df.iterrows():
            try:
                current_price = prices_dict.get(idx, 0)
                entry_price = row['entry_price']
                
                active_df.at[idx, 'current_price'] = current_price
                
                # Calculate unrealized P&L
                if row['direction'] == "LONG":
                    pnl = ((current_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    pnl = ((entry_price - current_price) / entry_price) * 100
                    
                active_df.at[idx, 'unrealized_pnl'] = pnl
            except Exception as e:
                active_df.at[idx, 'current_price'] = "Error"
                active_df.at[idx, 'unrealized_pnl'] = 0
        
        # Display active trades
        st.dataframe(active_df, use_container_width=True)
        
        # Individual trade charts
        st.subheader("Individual Trade Charts")
        selected_trade = st.selectbox("Select a trade to view", list(tester.active_trades.keys()))
        
        if selected_trade:
            timeframe = st.selectbox("Timeframe", ['1h', '4h', '1d'], index=0)
            ohlcv_df = fetch_hyperliquid_candles(selected_trade, interval=timeframe, limit=50)  # Reduced from 100
            
            if ohlcv_df is not None:
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=ohlcv_df['timestamp'],
                    open=ohlcv_df['open'],
                    high=ohlcv_df['high'],
                    low=ohlcv_df['low'],
                    close=ohlcv_df['close'],
                    name="OHLC"
                )])
                
                # Add entry, TP and SL lines
                trade_data = tester.active_trades[selected_trade]
                
                fig.add_hline(y=trade_data['entry_price'], line_width=1, line_dash="dash", 
                              line_color="yellow", annotation_text="Entry")
                
                if trade_data['tp_price']:
                    fig.add_hline(y=trade_data['tp_price'], line_width=1, line_dash="dash", 
                                 line_color="green", annotation_text="TP")
                    
                if trade_data['sl_price']:
                    fig.add_hline(y=trade_data['sl_price'], line_width=1, line_dash="dash", 
                                 line_color="red", annotation_text="SL")
                
                # Layout
                fig.update_layout(
                    title=f"{selected_trade} - {timeframe} Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active trades at the moment.")

# Tab 3: Completed Trades
with tab3:
    st.header("Completed Trades")
    
    # Reset button with confirmation
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Reset All Trades"):
            with col2:
                if st.checkbox("I confirm I want to reset all trades"):
                    result = tester.reset_all_trades()
                    st.success(result)
    
    # Get completed trades
    stats, completed_df = tester.get_performance_report()
    
    if isinstance(completed_df, pd.DataFrame) and len(completed_df) > 0:
        # Display completed trades
        st.dataframe(completed_df, use_container_width=True)
        
        # Trade outcomes pie chart
        if 'pct_change' in completed_df.columns:
            fig = px.pie(
                names=['Winning Trades', 'Losing Trades'],
                values=[
                    len(completed_df[completed_df['pct_change'] > 0]),
                    len(completed_df[completed_df['pct_change'] <= 0])
                ],
                color=['green', 'red'],
                title="Trade Outcomes"
            )
            st.plotly_chart(fig)
    else:
        st.info("No completed trades yet.")

# Tab 4: Performance
with tab4:
    st.header("Trading Performance")
    
    stats, completed_df = tester.get_performance_report()
    
    if isinstance(stats, dict):
        stats_df = pd.DataFrame([stats])
        st.dataframe(stats_df, use_container_width=True)
        
        # Show performance charts if we have data
        if isinstance(completed_df, pd.DataFrame) and len(completed_df) > 0 and 'pct_change' in completed_df.columns:
            # Cumulative performance chart
            if 'exit_time' in completed_df.columns:
                completed_df_sorted = completed_df.sort_values('exit_time')
                completed_df_sorted['cumulative_pnl'] = completed_df_sorted['pct_change'].cumsum()
                
                fig = px.line(
                    completed_df_sorted,
                    x='exit_time', 
                    y='cumulative_pnl',
                    title="Cumulative P&L (%)",
                    labels={'exit_time': 'Date', 'cumulative_pnl': 'Cumulative P&L (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # P&L distribution
            fig = px.histogram(
                completed_df,
                x='pct_change',
                nbins=20,
                title="P&L Distribution",
                labels={'pct_change': 'P&L (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(stats)  # This will display "No completed trades yet"

# Tab 5: Database Management
with tab5:
    st.header("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Information")
        if os.path.exists('trading_state.db'):
            conn = sqlite3.connect('trading_state.db')
            cursor = conn.cursor()
            
            # Get database size
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            conn.close()
            
            st.write(f"Database Size: {db_size/1024:.2f} KB")
            st.write("Tables:")
            for table in tables:
                st.write(f"- {table[0]}")
        else:
            st.write("Database file not found.")
    
    with col2:
        st.subheader("Database Actions")
        if st.button("Backup Database"):
            try:
                import shutil
                backup_file = f"trading_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2('trading_state.db', backup_file)
                st.success(f"Database backed up to {backup_file}")
            except Exception as e:
                st.error(f"Backup failed: {str(e)}")
        
        if st.button("Clear Cache"):
            try:
                if os.path.exists('market_cache.db'):
                    conn = sqlite3.connect('market_cache.db')
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM market_cache")
                    cursor.execute("DELETE FROM ohlcv_cache")
                    conn.commit()
                    conn.close()
                    st.cache_data.clear()
                    st.success("Cache cleared successfully")
            except Exception as e:
                st.error(f"Failed to clear cache: {str(e)}")

# Tab 6: Debug Information
with tab6:
    st.header("Debug Information")
    
    st.subheader("API Connection Status")
    
    # Test API connection button
    if st.button("Test API Connection"):
        try:
            meta_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "meta"})
            stats_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "stats"})
            oracle_response = requests.post(HYPERLIQUID_INFO_API, json={"type": "oracle"})
            
            st.success(f"Meta API: {meta_response.status_code}")
            st.success(f"Stats API: {stats_response.status_code}")
            st.success(f"Oracle API: {oracle_response.status_code}")
            
            if meta_response.status_code == 200:
                st.write("Meta Data Sample:")
                meta_json = meta_response.json()
                if 'universe' in meta_json and len(meta_json['universe']) > 0:
                    st.json(meta_json['universe'][0])
            
            if stats_response.status_code == 200:
                st.write("Stats Data Sample:")
                stats_json = stats_response.json()
                if isinstance(stats_json, list) and len(stats_json) > 0:
                    st.json(stats_json[0])
            
            if oracle_response.status_code == 200:
                st.write("Oracle Data Sample:")
                oracle_json = oracle_response.json()
                if isinstance(oracle_json, list) and len(oracle_json) > 0:
                    st.json(oracle_json[0])
        except Exception as e:
            st.error(f"API connection test failed: {str(e)}")
    
    # Show debug info if available
    if st.session_state.debug_info:
        st.subheader("Last Scan Debug Info")
        
        # Show important debug information first
        if 'BTC_details' in st.session_state.debug_info:
            st.write("BTC Market Details:")
            st.json(st.session_state.debug_info['BTC_details'])
        
        # Markets summary
        if 'markets_count' in st.session_state.debug_info:
            st.write(f"Markets found: {st.session_state.debug_info['markets_count']}")
            st.write(f"Markets skipped: {st.session_state.debug_info['skipped_count']}")
        
        # Show sample skipped markets
        if 'skipped_markets' in st.session_state.debug_info and st.session_state.debug_info['skipped_markets']:
            st.subheader("Sample Skipped Markets")
            skipped_df = pd.DataFrame(st.session_state.debug_info['skipped_markets'])
            st.dataframe(skipped_df)
        
        # Test parse_number_string function
        st.subheader("Test Number Parsing")
        test_values = ["1,234,567", "1.23e6", "123456", 123456, None]
        
        for val in test_values:
            try:
                result = parse_number_string(val)
                st.write(f"Input: {type(val)} {val} → Output: {result}")
            except Exception as e:
                st.write(f"Input: {type(val)} {val} → Error: {str(e)}")
        
        # Show any errors
        errors = {k: v for k, v in st.session_state.debug_info.items() if k.startswith('error')}
        if errors:
            st.subheader("Errors")
            for k, v in errors.items():
                st.error(f"{k}: {v}")
    
    # Cache status
    st.subheader("Cache Status")
    if os.path.exists('market_cache.db'):
        conn = sqlite3.connect('market_cache.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM market_cache")
        market_cache_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ohlcv_cache")
        ohlcv_cache_count = cursor.fetchone()[0]
        
        conn.close()
        
        st.write(f"Markets in cache: {market_cache_count}")
        st.write(f"OHLCV data sets in cache: {ohlcv_cache_count}")
    else:
        st.write("Cache database not found.")

# Update timestamp in the footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
