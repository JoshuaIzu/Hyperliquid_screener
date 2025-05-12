import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import time
from hyperliquid.info import Info
from hyperliquid.utils import constants
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

# Set page config
st.set_page_config(
    page_title="Hyperliquid Futures Screener",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("📈 Hyperliquid Futures Market Screener")
st.markdown("Track and analyze cryptocurrency futures markets on Hyperliquid")

# ======================== PYDANTIC MODELS ========================
class MarketLevel(BaseModel):
    side: str  # 'b' for bid, 'a' for ask
    price: float
    size: float

class OrderBook(BaseModel):
    levels: List[MarketLevel]
    
    @validator('levels')
    def validate_levels(cls, v):
        if not v:
            raise ValueError("Order book levels cannot be empty")
        return v

class MarketMeta(BaseModel):
    name: str
    szDecimals: int
    pxDecimals: int
    minSize: float

class MarketUniverse(BaseModel):
    universe: List[MarketMeta]

class CandleData(BaseModel):
    timestamp: int
    time_close: int
    symbol: str
    interval: str
    open: float
    close: float
    high: float
    low: float
    volume: float
    num: int

class FundingRate(BaseModel):
    fundingRate: float
    time: int

class MarketInfo(BaseModel):
    symbol: str = Field(..., alias="name")
    markPrice: float
    lastPrice: float
    fundingRate: float
    volume24h: float
    change24h: float

class Signal(BaseModel):
    Symbol: str
    Price: float
    MarkPrice: float
    Signal: str
    Volume24h: float
    FundingRate: float
    VolSurge: float
    Change24h: float
    Reason: str
    TP: str
    SL: str

class Trade(BaseModel):
    Symbol: str
    entry_price: float
    entry_time: str
    direction: str
    tp_price: Optional[float]
    sl_price: Optional[float]
    reason: str
    funding_rate: float
    status: str
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pct_change: Optional[float] = None

# ======================== CONFIGURATION ========================
BASE_VOL = 0.35
VOL_MULTIPLIER = 1.5
MIN_LIQUIDITY = 50000  # Fallback value
if 'MIN_LIQUIDITY' not in st.session_state:
    st.session_state.MIN_LIQUIDITY = MIN_LIQUIDITY
FUNDING_THRESHOLD = 60  # Annualized funding rate threshold (in basis points)

# Initialize Hyperliquid Info client
@st.cache_resource
def init_hyperliquid():
    return Info(skip_ws=True)

info = init_hyperliquid()

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'scanned_markets' not in st.session_state:
    st.session_state.scanned_markets = pd.DataFrame()
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# ======================== UTILITY CLASSES ========================
class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
    
    def wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

rate_limiter = RateLimiter(calls_per_second=3)

# ======================== HELPER FUNCTIONS ========================
def api_request_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1 * (2 ** attempt))
    return None

def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def estimate_volume_from_orderbook(symbol):
    try:
        book_response = api_request_with_retry(info.l2_snapshot, symbol)
        book = OrderBook(**book_response) if book_response else None
        
        if not book or len(book.levels) < 2:
            return 0
            
        bids = [level for level in book.levels if level.side == 'b']
        asks = [level for level in book.levels if level.side == 'a']
        
        if not bids or not asks:
            return 0
            
        top_bids = sorted(bids, key=lambda x: -x.price)[:5]
        top_asks = sorted(asks, key=lambda x: x.price)[:5]
        
        bid_liquidity = sum(bid.size for bid in top_bids)
        ask_liquidity = sum(ask.size for ask in top_asks)
        
        bid_price = top_bids[0].price if top_bids else 0
        ask_price = top_asks[0].price if top_asks else 0
        
        if bid_price <= 0 or ask_price <= 0 or bid_liquidity <= 0 or ask_liquidity <= 0:
            return 0
            
        if bid_price > ask_price:
            bid_price, ask_price = ask_price, bid_price
        
        mid_price = (bid_price + ask_price) / 2
        bid_value = bid_liquidity * mid_price
        ask_value = ask_liquidity * mid_price
        total_liquidity_value = bid_value + ask_value
        
        volume_multiplier = 20 if symbol in ["BTC", "ETH"] else 10
        volume = total_liquidity_value * volume_multiplier
        
        return max(volume, 1000)
    except Exception as e:
        return 1000 if symbol in ["BTC", "ETH"] else 0

# ======================== DATA FETCHING ========================
@st.cache_data(ttl=60)
def fetch_all_markets():
    try:
        debug_info = {}
        meta_response = api_request_with_retry(info.meta)
        meta = MarketUniverse(**meta_response)
        debug_info['total_markets'] = len(meta.universe)
        
        all_prices = api_request_with_retry(info.all_mids)
        
        market_data = []
        skipped_markets = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, coin in enumerate(meta.universe):
            symbol = coin.name
            
            try:
                progress.progress((i + 1) / len(meta.universe))
                status.text(f"Processing {i+1}/{len(meta.universe)}: {symbol}")
                
                price = safe_float(all_prices.get(symbol))
                if not price:
                    skipped_markets.append({
                        'symbol': symbol,
                        'reason': 'Could not get price data',
                        'min_required': MIN_LIQUIDITY
                    })
                    continue
                
                volume_24h = estimate_volume_from_orderbook(symbol)
                
                funding_rate = 0
                try:
                    funding_response = api_request_with_retry(info.funding_history, symbol, 1)
                    if funding_response and len(funding_response) > 0:
                        funding_data = FundingRate(**funding_response[0])
                        funding_rate = safe_float(funding_data.fundingRate) * 10000 * 3 * 365
                except Exception as e:
                    debug_info[f'funding_error_{symbol}'] = str(e)
                
                if volume_24h < MIN_LIQUIDITY:
                    skipped_markets.append({
                        'symbol': symbol,
                        'volume24h': volume_24h,
                        'min_required': MIN_LIQUIDITY
                    })
                    continue
                
                change_24h = 0
                try:
                    end_time = int(datetime.now().timestamp() * 1000)
                    start_time = end_time - (24 * 60 * 60 * 1000)
                    
                    candles_response = api_request_with_retry(info.candles_snapshot, symbol, '1h', start_time, end_time)
                    if candles_response and len(candles_response) > 0:
                        candles = [CandleData(**candle) for candle in candles_response]
                        if len(candles) > 1:
                            oldest_price = safe_float(candles[0].open)
                            latest_price = safe_float(candles[-1].close)
                            if oldest_price > 0:
                                change_24h = ((latest_price - oldest_price) / oldest_price) * 100
                except Exception as e:
                    debug_info[f'change_error_{symbol}'] = str(e)
                
                market_data.append(MarketInfo(
                    name=symbol,
                    markPrice=price,
                    lastPrice=price,
                    fundingRate=funding_rate,
                    volume24h=volume_24h,
                    change24h=change_24h,
                ))
                
            except Exception as e:
                debug_info[f'error_{symbol}'] = str(e)
                continue
        
        progress.empty()
        status.empty()
        
        df = pd.DataFrame([m.dict() for m in market_data])
        
        if df.empty:
            debug_info['empty_result'] = True
            debug_info['min_liquidity'] = MIN_LIQUIDITY
            st.warning(f"No markets met the minimum liquidity threshold of ${MIN_LIQUIDITY:,}.")
            
            if st.session_state.get('force_include_major', False):
                st.info("Using fallback data for major coins.")
                fallback_data = [
                    MarketInfo(name='BTC', markPrice=83000, lastPrice=83000, fundingRate=5, volume24h=500000, change24h=1.2),
                    MarketInfo(name='ETH', markPrice=3000, lastPrice=3000, fundingRate=10, volume24h=300000, change24h=0.8),
                    MarketInfo(name='SOL', markPrice=150, lastPrice=150, fundingRate=15, volume24h=100000, change24h=2.5)
                ]
                return pd.DataFrame([m.dict() for m in fallback_data])
        
        debug_info['markets_processed'] = len(market_data)
        debug_info['markets_skipped'] = len(skipped_markets)
        st.session_state.debug_info = debug_info
        
        return df.sort_values('volume24h', ascending=False) if not df.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching markets: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_hyperliquid_candles(symbol, interval="1h", limit=50):
    try:
        timeframe_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
        timeframe = timeframe_map.get(interval, '1h')
        
        end_time = int(datetime.now().timestamp() * 1000)
        
        if interval == '1h':
            start_time = end_time - (limit * 60 * 60 * 1000)
        elif interval == '4h':
            start_time = end_time - (limit * 4 * 60 * 60 * 1000)
        else:
            start_time = end_time - (limit * 24 * 60 * 60 * 1000)
        
        ohlcv = api_request_with_retry(info.candles_snapshot, symbol, timeframe, start_time, end_time)
        
        if not ohlcv or len(ohlcv) == 0:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'time_close', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'num'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        st.error(f"Error fetching candles for {symbol}: {str(e)}")
        return None

# ======================== DATABASE FUNCTIONS ========================
def init_market_cache():
    conn = sqlite3.connect('market_cache.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS market_cache (
        symbol TEXT PRIMARY KEY,
        data TEXT,
        timestamp INTEGER
    )
    ''')
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

init_market_cache()

def load_state_from_db():
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    cursor.execute('SELECT value FROM app_state WHERE key = "active_trades"')
    active_trades_row = cursor.fetchone()
    active_trades = json.loads(active_trades_row[0]) if active_trades_row else {}
    cursor.execute('SELECT value FROM app_state WHERE key = "completed_trades"')
    completed_trades_row = cursor.fetchone()
    completed_trades = json.loads(completed_trades_row[0]) if completed_trades_row else []
    conn.close()
    return active_trades, completed_trades

def save_state_to_db(active_trades, completed_trades):
    conn = sqlite3.connect('trading_state.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('active_trades', json.dumps(active_trades)))
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('completed_trades', json.dumps(completed_trades)))
    conn.commit()
    conn.close()

# ======================== TRADING LOGIC ========================
class ForwardTester:
    def __init__(self):
        self.active_trades: Dict[str, Trade] = {}
        self.completed_trades: List[Trade] = []
        self.load_state()
        
    def load_state(self):
        try:
            active_trades, completed_trades = load_state_from_db()
            self.active_trades = {k: Trade(**v) for k, v in active_trades.items()}
            self.completed_trades = [Trade(**t) for t in completed_trades]
        except Exception as e:
            st.warning(f"Could not load previous state: {str(e)}. Starting fresh.")
    
    def save_state(self):
        try:
            save_state_to_db(
                {k: v.dict() for k, v in self.active_trades.items()},
                [t.dict() for t in self.completed_trades]
            )
        except Exception as e:
            st.error(f"Error saving state to database: {str(e)}")
    
    def execute_trades(self, signals: List[Signal]):
        executed = []
        for signal in signals:
            symbol = signal.Symbol
            
            if symbol in self.active_trades:
                continue
                
            if signal.Signal != "HOLD":
                self.active_trades[symbol] = Trade(
                    Symbol=symbol,
                    entry_price=signal.Price,
                    entry_time=datetime.now().isoformat(),
                    direction=signal.Signal,
                    tp_price=float(signal.TP) if signal.TP != "-" else None,
                    sl_price=float(signal.SL) if signal.SL != "-" else None,
                    reason=signal.Reason,
                    funding_rate=signal.FundingRate,
                    status='OPEN'
                )
                executed.append(f"📝 New {signal.Signal} trade for {symbol} at {signal.Price}")
        
        self.save_state()
        return executed
    
    def update_trades(self):
        to_remove = []
        updates = []
        
        markets_df = fetch_all_markets()
        if not markets_df.empty:
            prices_dict = {row['symbol']: row['markPrice'] for _, row in markets_df.iterrows()}
            
            for symbol, trade in self.active_trades.items():
                try:
                    current_price = prices_dict.get(symbol)
                    if not current_price:
                        continue
                        
                    entry_price = trade.entry_price
                    
                    if trade.direction == "LONG":
                        if trade.tp_price and current_price >= trade.tp_price:
                            trade.exit_reason = "TP Hit"
                        elif trade.sl_price and current_price <= trade.sl_price:
                            trade.exit_reason = "SL Hit"
                    elif trade.direction == "SHORT":
                        if trade.tp_price and current_price <= trade.tp_price:
                            trade.exit_reason = "TP Hit"
                        elif trade.sl_price and current_price >= trade.sl_price:
                            trade.exit_reason = "SL Hit"
                    
                    if hasattr(trade, 'exit_reason'):
                        trade.exit_price = current_price
                        trade.exit_time = datetime.now().isoformat()
                        trade.status = 'CLOSED'
                        trade.pct_change = ((current_price - entry_price)/entry_price)*100 if trade.direction == "LONG" else ((entry_price - current_price)/entry_price)*100
                        self.completed_trades.append(trade)
                        to_remove.append(symbol)
                        updates.append(f"✅ Trade closed: {symbol} | Reason: {trade.exit_reason} | PnL: {trade.pct_change:.2f}%")
                
                except Exception as e:
                    updates.append(f"Error updating {symbol}: {str(e)}")
            
            for symbol in to_remove:
                self.active_trades.pop(symbol)
            
            self.save_state()
        
        return updates
    
    def get_performance_report(self):
        if not self.completed_trades:
            return "No completed trades yet", pd.DataFrame()
            
        df = pd.DataFrame([t.dict() for t in self.completed_trades])
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
        self.active_trades = {}
        self.completed_trades = []
        self.save_state()
        return "All trades have been reset"

# ======================== SIGNAL GENERATION ========================
def generate_signals(markets_df):
    if markets_df is None or markets_df.empty:
        return []
    
    signals = []
    top_markets = markets_df.head(15)
    total_markets = len(top_markets)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (index, market) in enumerate(top_markets.iterrows()):
        symbol = market['symbol']
        status_text.text(f"Analyzing {symbol}... ({i+1}/{total_markets})")
        
        df = fetch_hyperliquid_candles(symbol, interval='1h', limit=48)
        if df is None or len(df) < 6:
            continue
            
        avg_vol = df['volume'].mean()
        recent_vol = df['volume'].iloc[-1]
        vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
        
        signal = "HOLD"
        reason = ""
        tp = "-"
        sl = "-"
        
        funding_rate = market['fundingRate']
        
        if vol_surge >= VOL_MULTIPLIER and recent_vol > BASE_VOL * avg_vol:
            recent_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            
            if recent_change > 0.01 and funding_rate < -FUNDING_THRESHOLD:
                signal = "LONG"
                reason = f"Vol surge {vol_surge:.2f}x with bullish price action and favorable funding rate ({funding_rate:.2f} bps)"
                tp = str(round(df['close'].iloc[-1] * 1.05, 4))
                sl = str(round(df['close'].iloc[-1] * 0.97, 4))
            elif recent_change < -0.01 and funding_rate > FUNDING_THRESHOLD:
                signal = "SHORT"
                reason = f"Vol surge {vol_surge:.2f}x with bearish price action and favorable funding rate ({funding_rate:.2f} bps)"
                tp = str(round(df['close'].iloc[-1] * 0.95, 4))
                sl = str(round(df['close'].iloc[-1] * 1.03, 4))
        
        signals.append(Signal(
            Symbol=symbol,
            Price=df['close'].iloc[-1],
            MarkPrice=market['markPrice'],
            Signal=signal,
            Volume24h=market['volume24h'],
            FundingRate=funding_rate,
            VolSurge=vol_surge,
            Change24h=market['change24h'],
            Reason=reason,
            TP=tp,
            SL=sl
        ))
    
    progress_bar.empty()
    status_text.empty()
    return signals

def scan_markets():
    st.session_state.scanned_markets = pd.DataFrame()
    st.session_state.signals = []
    
    try:
        with st.spinner("Fetching market data from Hyperliquid..."):
            markets_df = fetch_all_markets()
            
            if markets_df.empty:
                if 'debug_info' in st.session_state:
                    debug = st.session_state.debug_info
                    if 'skipped_samples' in debug and debug['skipped_samples']:
                        st.warning(f"Found data but all markets were below the liquidity threshold.")
                return
                
            st.session_state.scanned_markets = markets_df
            st.success(f"Found {len(markets_df)} markets meeting the minimum liquidity threshold of ${MIN_LIQUIDITY:,} USD.")
    
    except Exception as e:
        st.error(f"Error during market scan: {str(e)}")
        return
    
    try:
        if not markets_df.empty:
            with st.spinner("Analyzing markets for trading signals..."):
                signals = generate_signals(markets_df)
                st.session_state.signals = signals
                
                actionable_count = len([s for s in signals if s.Signal != 'HOLD'])
                if actionable_count > 0:
                    st.success(f"Analysis complete. Found {actionable_count} actionable signals.")
                else:
                    st.info("Analysis complete. No actionable signals found with current parameters.")
    except Exception as e:
        st.error(f"Error during signal generation: {str(e)}")

# ======================== STREAMLIT UI ========================
# Initialize the forward tester
tester = ForwardTester()

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")
    BASE_VOL = st.slider("Base Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    VOL_MULTIPLIER = st.slider("Volume Multiplier", 1.0, 3.0, 1.5, 0.1)
    
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
        index=1
    )
    
    MIN_LIQUIDITY = liquidity_options[selected_liquidity]
    
    with st.expander("Advanced Settings"):
        st.session_state.use_orderbook_fallback = st.checkbox(
            "Use orderbook fallback", 
            value=True, 
            help="Estimate volume from orderbook when API volume data is missing"
        )
        
        fallback_threshold = st.number_input(
            "Fallback threshold (USD)",
            value=5000.0,
            min_value=1000.0,
            step=1000.0
        )
        
        api_calls_per_second = st.slider(
            "API calls per second",
            min_value=1,
            max_value=10,
            value=3
        )
        rate_limiter.calls_per_second = api_calls_per_second
    
    FUNDING_THRESHOLD = st.slider("Funding Rate Threshold (basis points)", 10, 200, 60, 5)
    
    with st.expander("Debugging Tools"):
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        
        debug_enabled = st.checkbox(
            "Enable Debug Mode",
            value=st.session_state.debug_mode,
            key="debug_mode_toggle",
            on_change=lambda: st.session_state.update(debug_mode=not st.session_state.debug_mode)
        )
        
        if st.session_state.debug_mode:
            st.warning("Debug Mode is ON - This may slow down the application")
            st.session_state.log_api_responses = st.checkbox("Log API Responses", value=True)
            st.session_state.force_include_major = st.checkbox("Force Include Major Coins", value=True)
            
            debug_min_liquidity = st.number_input(
                "Debug Min Liquidity",
                min_value=1000,
                max_value=100000,
                value=5000,
                step=1000
            )
            
            if st.button("Apply Debug Settings"):
                st.session_state.MIN_LIQUIDITY = debug_min_liquidity
                st.success(f"Applied debug settings. Min liquidity now: ${st.session_state.MIN_LIQUIDITY:,}")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database", "Debug"])

# Tab 1: Market Scanner
with tab1:
    st.info(f"Current Minimum Liquidity: ${MIN_LIQUIDITY:,} USD")
    
    if st.button("Scan Markets", use_container_width=True):
        scan_markets()
    
    if not st.session_state.scanned_markets.empty:
        st.subheader("All Markets")
        st.dataframe(
            st.session_state.scanned_markets,
            use_container_width=True,
            column_config={
                'symbol': st.column_config.TextColumn("Symbol"),
                'markPrice': st.column_config.NumberColumn("Mark Price", format="%.4f"),
                'volume24h': st.column_config.NumberColumn("24h Volume", format="$%.2f"),
                'fundingRate': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                'change24h': st.column_config.NumberColumn("24h Change", format="%.2f%%")
            }
        )
    
    if st.session_state.signals:
        st.subheader("Trading Signals")
        signals_df = pd.DataFrame([s.dict() for s in st.session_state.signals])
        
        signal_filter = st.multiselect("Filter by Signal", 
                                     options=['LONG', 'SHORT', 'HOLD'], 
                                     default=['LONG', 'SHORT'])
        
        if signal_filter:
            filtered_df = signals_df[signals_df['Signal'].isin(signal_filter)]
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    'Price': st.column_config.NumberColumn("Price", format="%.4f"),
                    'Volume24h': st.column_config.NumberColumn("Volume 24h", format="$%.2f"),
                    'FundingRate': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                    'VolSurge': st.column_config.NumberColumn("Volume Surge", format="%.2fx")
                }
            )
            
            actionable_signals = [s for s in st.session_state.signals if s.Signal != "HOLD" and s.Signal in signal_filter]
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
        active_df = pd.DataFrame([t.dict() for t in tester.active_trades.values()])
        
        markets_df = fetch_all_markets()
        prices_dict = {row['symbol']: row['markPrice'] for _, row in markets_df.iterrows()} if not markets_df.empty else {}
        
        for idx, row in active_df.iterrows():
            try:
                current_price = prices_dict.get(row['Symbol'], 0)
                entry_price = row['entry_price']
                
                active_df.at[idx, 'current_price'] = current_price
                
                if row['direction'] == "LONG":
                    pnl = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl = ((entry_price - current_price) / entry_price) * 100
                    
                active_df.at[idx, 'unrealized_pnl'] = pnl
            except Exception:
                active_df.at[idx, 'current_price'] = "Error"
                active_df.at[idx, 'unrealized_pnl'] = 0
        
        st.dataframe(active_df, use_container_width=True)
        
        st.subheader("Individual Trade Charts")
        selected_trade = st.selectbox("Select a trade to view", list(tester.active_trades.keys()))
        
        if selected_trade:
            timeframe = st.selectbox("Timeframe", ['1h', '4h', '1d'], index=0)
            ohlcv_df = fetch_hyperliquid_candles(selected_trade, interval=timeframe, limit=50)
            
            if ohlcv_df is not None:
                fig = go.Figure(data=[go.Candlestick(
                    x=ohlcv_df['timestamp'],
                    open=ohlcv_df['open'],
                    high=ohlcv_df['high'],
                    low=ohlcv_df['low'],
                    close=ohlcv_df['close'],
                    name="OHLC"
                )])
                
                trade_data = tester.active_trades[selected_trade]
                
                fig.add_hline(y=trade_data.entry_price, line_width=1, line_dash="dash", 
                              line_color="yellow", annotation_text="Entry")
                
                if trade_data.tp_price:
                    fig.add_hline(y=trade_data.tp_price, line_width=1, line_dash="dash", 
                                 line_color="green", annotation_text="TP")
                    
                if trade_data.sl_price:
                    fig.add_hline(y=trade_data.sl_price, line_width=1, line_dash="dash", 
                                 line_color="red", annotation_text="SL")
                
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
    
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Reset All Trades"):
            with col2:
                if st.checkbox("I confirm I want to reset all trades"):
                    result = tester.reset_all_trades()
                    st.success(result)
    
    stats, completed_df = tester.get_performance_report()
    
    if isinstance(completed_df, pd.DataFrame) and len(completed_df) > 0:
        st.dataframe(completed_df, use_container_width=True)
        
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
        
        if isinstance(completed_df, pd.DataFrame) and len(completed_df) > 0 and 'pct_change' in completed_df.columns:
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
            
            fig = px.histogram(
                completed_df,
                x='pct_change',
                nbins=20,
                title="P&L Distribution",
                labels={'pct_change': 'P&L (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(stats)

# Tab 5: Database Management
with tab5:
    st.header("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Information")
        if os.path.exists('trading_state.db'):
            conn = sqlite3.connect('trading_state.db')
            cursor = conn.cursor()
            cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
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
    
    debug_enabled = st.checkbox(
        "Enable Debug Mode",
        value=st.session_state.get('debug_mode', False),
        key="debug_tab_toggle",
        on_change=lambda: st.session_state.update(debug_mode=not st.session_state.debug_mode)
    )
    
    if debug_enabled != st.session_state.get('debug_mode_prev', None):
        st.session_state.debug_mode_prev = debug_enabled
        st.success(f"Debug mode {'enabled' if debug_enabled else 'disabled'}")
    
    if st.button("Test Hyperliquid Connection"):
        try:
            meta = info.meta()
            st.success("✅ Hyperliquid connection established")
            st.write(f"Total markets: {len(meta['universe'])}")
            
            if len(meta['universe']) > 0:
                sample_symbol = meta['universe'][0]['name']
                st.write(f"Sample market: {sample_symbol}")
                
                ticker = info.all_mids()
                st.write(f"Current mid price: {ticker.get(sample_symbol, 'N/A')}")
                
                try:
                    book = info.l2_snapshot(sample_symbol)
                    if book and 'levels' in book and book['levels']:
                        st.success(f"✅ Successfully fetched orderbook for {sample_symbol}")
                        bids = [level for level in book['levels'] if level[0] == 'b']
                        asks = [level for level in book['levels'] if level[0] == 'a']
                        st.write(f"Found {len(bids)} bids and {len(asks)} asks")
                        if bids:
                            st.write(f"Top bid: {bids[0]}")
                        if asks:
                            st.write(f"Top ask: {asks[0]}")
                    else:
                        st.error(f"❌ Failed to fetch valid orderbook for {sample_symbol}")
                except Exception as e:
                    st.error(f"❌ Orderbook test failed: {str(e)}")
        except Exception as e:
            st.error(f"❌ Hyperliquid connection test failed: {str(e)}")
    
    if st.session_state.debug_info:
        st.subheader("Last Scan Debug Info")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Markets", 
                      st.session_state.debug_info.get('total_markets', 'N/A'))
        with cols[1]:
            st.metric("Markets Processed", 
                      st.session_state.debug_info.get('markets_processed', 'N/A'))
        with cols[2]:
            st.metric("Markets Skipped", 
                      st.session_state.debug_info.get('markets_skipped', 'N/A'))
        
        if 'BTC_details' in st.session_state.debug_info:
            st.subheader("BTC Market Details")
            st.json(st.session_state.debug_info['BTC_details'])
            
            if 'BTC_raw_book' in st.session_state.debug_info:
                with st.expander("Raw BTC Orderbook Data"):
                    st.code(st.session_state.debug_info['BTC_raw_book'])
        
        if 'skipped_samples' in st.session_state.debug_info and st.session_state.debug_info['skipped_samples']:
            st.subheader("Sample Skipped Markets")
            st.json(st.session_state.debug_info['skipped_samples'])
            
        errors = {k: v for k, v in st.session_state.debug_info.items() if 'error' in k.lower()}
        if errors:
            with st.expander("Error Details"):
                for k, v in errors.items():
                    st.error(f"{k}: {v}")
        
        if 'sample_price_response' in st.session_state.debug_info:
            with st.expander("Sample API Response"):
                st.code(st.session_state.debug_info['sample_price_response'])
    
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
        
        if st.button("Clear All Cache"):
            try:
                conn = sqlite3.connect('market_cache.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM market_cache")
                cursor.execute("DELETE FROM ohlcv_cache")
                conn.commit()
                conn.close()
                st.cache_data.clear()
                st.success("✅ Cache cleared successfully")
                st.info("Please scan markets again to refresh data")
            except Exception as e:
                st.error(f"❌ Failed to clear cache: {str(e)}")
    else:
        st.write("Cache database not found.")
        
    st.subheader("Rate Limiter Settings")
    current_rate = st.session_state.get('api_rate', rate_limiter.calls_per_second)
    new_rate = st.slider("API calls per second", 1, 10, int(current_rate), 1)
    
    if new_rate != current_rate:
        rate_limiter.calls_per_second = new_rate
        st.session_state.api_rate = new_rate
        st.success(f"Rate limiter updated to {new_rate} calls per second")
        
# Update timestamp in the footer
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
