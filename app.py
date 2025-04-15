import streamlit as st
import numpy as np
import pandas as pd
import json
import os
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
MIN_LIQUIDITY = 5000000
FUNDING_THRESHOLD = 60  # Annualized funding rate threshold (in basis points)

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

@st.cache_data(ttl=30)
def fetch_hyperliquid_all_mids():
    """Fetch all mid prices for markets"""
    try:
        response = requests.post(HYPERLIQUID_INFO_API, json={"type": "allMids"})
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error fetching mid prices: {str(e)}")
        return None

@st.cache_data(ttl=30)
def fetch_hyperliquid_funding_rates():
    """Fetch funding rates for all markets"""
    try:
        response = requests.post(HYPERLIQUID_INFO_API, json={"type": "fundingHistory"})
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error fetching funding rates: {str(e)}")
        return None

@st.cache_data(ttl=30)
def fetch_hyperliquid_market_info():
    """Fetch information about markets including open interest"""
    try:
        response = requests.post(HYPERLIQUID_INFO_API, json={"type": "stats"})
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error fetching market info: {str(e)}")
        return None

@st.cache_data(ttl=300)
def fetch_hyperliquid_candles(coin, interval="1h", limit=500):
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

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")
    BASE_VOL = st.slider("Base Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    VOL_MULTIPLIER = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
    MIN_LIQUIDITY = st.number_input("Minimum Liquidity (USD)", 1000000, 20000000, 5000000, 1000000)
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
        mids = fetch_hyperliquid_all_mids()
        prices_dict = {item["coin"]: float(item["mid"]) for item in mids} if mids else {}
        
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

# Function to fetch all markets and filter
@st.cache_data(ttl=300)
def fetch_all_markets():
    """Fetch all markets from Hyperliquid"""
    try:
        meta = fetch_hyperliquid_meta()
        if not meta or not meta.get('universe'):
            return pd.DataFrame()
            
        # Get market names
        markets = [asset['name'] for asset in meta['universe']]
        
        # Get mid prices
        mids = fetch_hyperliquid_all_mids()
        prices_dict = {item["coin"]: float(item["mid"]) for item in mids} if mids else {}
        
        # Get funding rates
        funding_data = fetch_hyperliquid_funding_rates()
        funding_dict = {}
        if funding_data:
            for item in funding_data:
                if item and 'coin' in item and 'fundingRate' in item:
                    # Convert funding rate from decimal to basis points and annualize (x365x24)
                    funding_dict[item['coin']] = float(item['fundingRate']) * 10000 * 365 * 24
        
        # Get market stats (includes open interest)
        stats_data = fetch_hyperliquid_market_info()
        stats_dict = {}
        if stats_data:
            for item in stats_data:
                if 'coin' in item:
                    stats_dict[item['coin']] = item
        
        results = []
        for market in markets:
            try:
                price = prices_dict.get(market, 0)
                funding_rate = funding_dict.get(market, 0)
                
                # Get volume and open interest from stats
                market_stats = stats_dict.get(market, {})
                volume_24h = float(market_stats.get('volume24h', 0)) if market_stats else 0
                open_interest = float(market_stats.get('openInterest', 0)) if market_stats else 0
                
                # Skip low liquidity markets
                if volume_24h < MIN_LIQUIDITY:
                    continue
                
                results.append({
                    'symbol': market,
                    'last_price': price,
                    'mark_price': price,  # Same as last price in this case
                    'volume_24h': volume_24h,
                    'funding_rate': funding_rate,
                    'open_interest': open_interest,
                    'change_24h': 0  # Not available directly, would need to calculate from candles
                })
            except Exception as e:
                st.warning(f"Error processing market {market}: {str(e)}")
                
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error fetching markets: {str(e)}")
        return pd.DataFrame()

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

# Function to analyze and generate signals
def generate_signals(markets_df=None):
    """Generate trading signals based on volume analysis and funding rates"""
    if markets_df is None or markets_df.empty:
        markets_df = fetch_all_markets()
    
    signals = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (index, market) in enumerate(markets_df.iterrows()):
        symbol = market['symbol']
        status_text.text(f"Analyzing {symbol}...")
        
        # Fetch hourly data
        df = fetch_hyperliquid_candles(symbol, interval='1h', limit=168)  # 7 days
        if df is None or len(df) < 24:
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
        funding_rate = market['funding_rate']
        
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
            'Mark Price': market['mark_price'],
            'Signal': signal,
            'Volume 24h': market['volume_24h'],
            'Open Interest': market['open_interest'],
            'Funding Rate': funding_rate,
            'Vol Surge': vol_surge,
            'Change 24h': market['change_24h'] if 'change_24h' in market else 0,
            'Reason': reason,
            'TP': tp,
            'SL': sl
        })
        
        # Update progress
        progress_bar.progress((i + 1) / len(markets_df))
    
    progress_bar.empty()
    status_text.empty()
    
    return signals

# Initialize the forward tester
tester = ForwardTester()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database"])

# Tab 1: Market Scanner
with tab1:
    if st.button("Scan Markets"):
        with st.spinner("Scanning all Hyperliquid perpetual markets..."):
            markets_df = fetch_all_markets()
            st.session_state.signals = generate_signals(markets_df)
    
    if 'signals' in st.session_state and st.session_state.signals:
        signals_df = pd.DataFrame(st.session_state.signals)
        
        # Filter options
        signal_filter = st.multiselect("Filter by Signal", 
                                       options=['LONG', 'SHORT', 'HOLD'], 
                                       default=['LONG', 'SHORT'])
        
        filtered_df = signals_df[signals_df['Signal'].isin(signal_filter)]
        
        # Display signals table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Execute trades button
        if st.button("Execute Selected Signals"):
            actionable_signals = [s for s in st.session_state.signals if s['Signal'] != "HOLD" and s['Signal'] in signal_filter]
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
        mids = fetch_hyperliquid_all_mids()
        prices_dict = {item["coin"]: float(item["mid"]) for item in mids} if mids else {}
        
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
            ohlcv_df = fetch_hyperliquid_candles(selected_trade, interval=timeframe, limit=100)
            
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
    if st.button("Reset All Trades"):
        if st.checkbox("I confirm I want to reset all trades"):
            result = tester.reset_all_trades()
            st.success(result)
    
    # Get completed trades
    stats, completed_df = tester.get_performance_report()
    
    if len(completed_df) > 0:
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
        if len(completed_df) > 0 and 'pct_change' in completed_df.columns:
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

# Update timestamp in the footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
