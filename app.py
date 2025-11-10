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
import math
import random
import asyncio
import nest_asyncio
import logging
from hyperliquid.info import Info
from hyperliquid.utils import constants
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logfire

# Import WebSocket functionality
from hyperliquid_websocket import (
    HyperliquidWebSocketManager, 
    initialize_websocket_manager, 
    get_websocket_manager,
    get_websocket_orderbook,
    get_websocket_candles
)
# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Initialize logging
#logfire.configure(send_to_logfire="if-token-present")

logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler('hyperliquid_screener.log'),
                       #logfire.LogfireLoggingHandler(),
                       logging.StreamHandler() # Also log to console
                   ]
)

# Initialize the Hyperliquid Info client
info = Info(constants.MAINNET_API_URL, skip_ws=False)

# ======================== UTILITY FUNCTIONS ========================
def parse_orderbook_levels(raw_levels: Any) -> list:
    """Helper to parse raw orderbook level data."""
    parsed = []
    for level_group in raw_levels:
        for level in level_group:
            if isinstance(level, dict) and "px" in level and "sz" in level:
                parsed.append({
                    "side": "b" if level.get("n", 0) > 0 else "a",
                    "price": float(level["px"]),
                    "size": float(level["sz"])
                })
    return parsed

def parse_candles(raw_candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parses raw candle data from the API into a more usable format."""
    parsed = []
    if not raw_candles:
        return parsed
    for candle in raw_candles:
        if isinstance(candle, dict):
            parsed.append({
                'timestamp': candle.get('t', 0),
                'time_close': candle.get('t', 0) + 3600000,  # Assuming 1h interval for now
                'symbol': candle.get('s', ''),
                'interval': candle.get('i', ''),
                'open': float(candle.get('o', 0.0)),
                'close': float(candle.get('c', 0.0)),
                'high': float(candle.get('h', 0.0)),
                'low': float(candle.get('l', 0.0)),
                'volume': float(candle.get('v', 0.0)),
                'num': candle.get('n', 0)
            })
    return parsed

async def async_api_request(api_call, *args, **kwargs):
    """A wrapper for making asynchronous API requests."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        try:
            result = await loop.run_in_executor(executor, lambda: api_call(*args, **kwargs))
            return result
        except Exception as e:
            logging.error(f"API request failed: {e}")
            return None

def validate_api_responses():
    """Validates that the API is responsive."""
    try:
        # A simple check to see if we can get metadata for a common coin
        response = info.meta()
        if response and 'universe' in response:
            return True
    except Exception as e:
        logging.error(f"API validation failed: {e}")
    return False

def apply_fallback_for_major_coins():
    """Returns a default list of major coins when the main fetch fails."""
    major_coins = ["BTC", "ETH"]
    st.warning(f"Falling back to major coins: {', '.join(major_coins)}")
    
    # Create a minimal DataFrame with default values for major coins
    fallback_data = []
    for coin in major_coins:
        fallback_data.append({
            'symbol': coin,
            'markPrice': 0.0,  # Will be updated in real-time
            'lastPrice': 0.0,
            'fundingRate': 0.0,
            'volume24h': 100000.0,  # Default high volume
            'change24h': 0.0,
            'spreadBps': 10.0,
            'liquidityScore': 1000.0
        })
    return pd.DataFrame(fallback_data)


# Set page config
st.set_page_config(
    page_title="Hyperliquid Futures Screener",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("📈 Hyperliquid Futures Market Screener")
st.markdown("Track and analyze cryptocurrency futures markets on Hyperliquid")

# Rate limiting notice
if st.button("🔄 Refresh API Status"):
    st.rerun()

with st.expander("ℹ️ API Rate Limiting Info", expanded=False):
    st.info("""
    **Enhanced Rate Limiting Active**
    
    This version includes improved rate limiting to prevent 429 (Too Many Requests) errors:
    - Reduced concurrent requests (3 max instead of 8)
    - Conservative API call frequency (1-2 calls/second)
    - Exponential backoff on 429 errors
    - Automatic retry logic with circuit breaker
    - Smaller batch sizes (5-10 markets per batch)
    
    **What this means:**
    - More reliable data fetching
    - Longer processing times but fewer failures
    - Better respect for API limits
    - Automatic recovery from rate limit errors
    """)
    
    if 'last_429_count' in st.session_state:
        st.metric("Recent 429 Errors", st.session_state.get('last_429_count', 0))
    
    current_time = datetime.now().strftime('%H:%M:%S')
    st.caption(f"Status checked at: {current_time}")

st.divider()

# Initialize session state
def init_session_state():
    if 'signals' not in st.session_state:
        st.session_state.signals = []
    if 'scanned_markets' not in st.session_state:
        st.session_state.scanned_markets = pd.DataFrame()
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = {}
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []
    if 'MIN_LIQUIDITY' not in st.session_state:
        st.session_state.MIN_LIQUIDITY = 1000
    if 'ws_enabled' not in st.session_state:
        st.session_state.ws_enabled = False
    if 'ws_manager' not in st.session_state:
        st.session_state.ws_manager = None
    if 'ws_orderbooks' not in st.session_state:
        st.session_state.ws_orderbooks = {}
    if 'ws_candles' not in st.session_state:
        st.session_state.ws_candles = {}

init_session_state()
# Auto initialize WebSocket if enabled
if st.session_state.get('ws_enabled', False) and not st.session_state.get('ws_manager'):
    try:
        asyncio.run(initialize_websocket_manager())
        st.session_state.ws_connected = True
        logging.info("Auto WebSocket connection established.")
    except Exception as e:
        logging.error(f"Auto WebSocket initialization failed: {e}")
        st.session_state.ws_connected = False

# ======================== PYDANTIC MODELS ========================
class MarketLevel(BaseModel):
    side: str
    price: float
    size: float

class OrderBook(BaseModel):
    levels: List[MarketLevel]
    
    @field_validator('levels')
    @classmethod
    def validate_levels(cls, v):
        if not v:
            raise ValueError("Order book levels cannot be empty")
        return v
    
    def get_best_bid_ask(self) -> tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices from the order book"""
        bids = [level.price for level in self.levels if level.side == 'b']
        asks = [level.price for level in self.levels if level.side == 'a']
        
        best_bid = max(bids) if bids else None
        best_ask = min(asks) if asks else None
        
        return best_bid, best_ask

class MarketMeta(BaseModel):
    name: str
    szDecimals: Optional[int] = None
    pxDecimals: Optional[int] = None
    minSize: Optional[float] = None

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
    symbol: str
    markPrice: float
    lastPrice: float
    fundingRate: float
    volume24h: float
    change24h: float
    spreadBps: Optional[float] = 999.0
    liquidityScore: Optional[float] = 0.0
    spreadPct: Optional[float] = 0.0
    spreadRatio: Optional[float] = 0.0
    bidLiquidity: Optional[float] = 0.0
    askLiquidity: Optional[float] = 0.0
    bidDepth1pct: Optional[float] = 0.0
    askDepth1pct: Optional[float] = 0.0
    imbalance: Optional[float] = 0.0
    bestBid: Optional[float] = 0.0
    bestAsk: Optional[float] = 0.0

class Signal(BaseModel):
    Symbol: str
    Price: float
    MarkPrice: float
    Signal: str
    Volume24h: float
    FundingRate: float
    VolSurge: float
    Change24h: float
    spreadBps: Optional[float] = None
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

# ======================== UTILITY CLASSES ========================
class MicroPrice:
    def __init__(self, n_imb: int = 10, n_spread: int = 5, alpha: float = 2.0):
        """
        Improved MicroPrice calculator based on Sasha Stoikov's approach
        
        Args:
            n_imb: Number of imbalance buckets (discretization)
            n_spread: Maximum spread size in ticks to consider
            alpha: Sensitivity parameter for price adjustment
        """
        self.n_imb = n_imb
        self.n_spread = n_spread
        self.alpha = alpha
        self.transition_matrices = None
        self.ticksize = None
        self.price_adjustments = None
        
    def calculate(self, best_bid: float, best_ask: float, bid_size: float, ask_size: float, n_steps: int = 6) -> float:
        """
        Calculate micro-price given current market state
        
        Args:
            best_bid: Current best bid price
            best_ask: Current best ask price
            bid_size: Current best bid size
            ask_size: Current best ask size
            n_steps: Number of price moves to look ahead (default 6 as per theory)
            
        Returns:
            Micro-price estimate
        """
        # Input validation
        if not all(isinstance(x, (int, float)) for x in [best_bid, best_ask, bid_size, ask_size]):
            raise ValueError("All inputs must be numeric")
        if best_bid < 0 or best_ask < 0 or bid_size < 0 or ask_size < 0:
            raise ValueError("Inputs must be non-negative")
        if best_ask < best_bid:
            best_bid, best_ask = best_ask, best_bid
        if bid_size == 0 and ask_size == 0:
            raise ValueError("At least one of bid_size or ask_size must be positive")
            
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        
        # Handle edge cases
        if bid_size == 0 or ask_size == 0 or self.price_adjustments is None:
            return mid_price
            
        # Calculate current state
        imbalance = bid_size / (bid_size + ask_size)
        
        # If we have transition matrices (from historical data), use n-step ahead prediction
        if self.transition_matrices is not None and n_steps > 1:
            spread_bucket = min(int(np.round(spread / self.ticksize)), self.n_spread - 1)
            imb_bucket = min(int(imbalance * self.n_imb), self.n_imb - 1)
            state_idx = spread_bucket * self.n_imb + imb_bucket
            
            # Compute n-step ahead state distribution
            state_probs = np.zeros(self.n_imb * self.n_spread)
            state_probs[state_idx] = 1.0
            
            for _ in range(n_steps):
                state_probs = np.dot(state_probs, self.transition_matrices)
                
            # Compute expected price adjustment
            adjustment = np.dot(state_probs, self.price_adjustments)
        else:
            # Fall back to basic micro-price calculation if no transition matrices
            delta = imbalance - 0.5
            adjustment = spread * math.tanh(self.alpha * delta)
        
        micro_price = mid_price + adjustment
        
        # Ensure micro-price stays within bid-ask spread
        return max(best_bid, min(best_ask, micro_price))
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the micro-price model to historical data (optional)
        
        Args:
            df: DataFrame with historical market data containing:
                - bid: bid prices
                - ask: ask prices
                - bs: bid sizes
                - as: ask sizes
                - time: timestamp
        """
        if df.empty:
            return
            
        # Prepare data similar to prep_data_sym
        spread = df['ask'] - df['bid']
        self.ticksize = np.round(min(spread.loc[spread > 0]) * 100) / 100
        
        df['spread'] = np.round((df['ask'] - df['bid']) / self.ticksize) * self.ticksize
        df['mid'] = (df['bid'] + df['ask']) / 2
        df = df.loc[(df.spread <= self.n_spread * self.ticksize) & (df.spread > 0)]
        
        df['imb'] = df['bs'] / (df['bs'] + df['as'])
        df['imb_bucket'] = pd.qcut(df['imb'], self.n_imb, labels=False)
        
        # Future states
        df['next_mid'] = df['mid'].shift(-1)
        df['next_spread'] = df['spread'].shift(-1)
        df['next_imb_bucket'] = df['imb_bucket'].shift(-1)
        
        # Price changes
        df['dM'] = np.round((df['next_mid'] - df['mid']) / self.ticksize * 2) * self.ticksize / 2
        df = df.loc[(df.dM <= self.ticksize * 1.1) & (df.dM >= -self.ticksize * 1.1)]
        
        # Symmetrize data
        df2 = df.copy(deep=True)
        df2['imb_bucket'] = self.n_imb - 1 - df2['imb_bucket']
        df2['next_imb_bucket'] = self.n_imb - 1 - df2['next_imb_bucket']
        df2['dM'] = -df2['dM']
        df2['mid'] = -df2['mid']
        
        df_combined = pd.concat([df, df2])
        
        # Estimate transition matrices (simplified version)
        no_move = df_combined[df_combined['dM'] == 0]
        move = df_combined[df_combined['dM'] != 0]
        
        # Count state transitions
        transition_counts = no_move.groupby(['spread', 'imb_bucket', 'next_spread', 'next_imb_bucket']).size().unstack(fill_value=0)
        
        # Normalize to probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=1)
        
        # Estimate price adjustments
        price_changes = move.groupby(['spread', 'imb_bucket'])['dM'].mean().unstack(fill_value=0)
        
        # Store the learned parameters
        self.transition_matrices = transition_probs.values
        self.price_adjustments = price_changes.values.flatten()

# ======================== ASYNC RATE LIMITER ========================

class AsyncRateLimiter:
    """Enhanced async rate limiter with exponential backoff for 429 errors"""
    
    def __init__(self, calls_per_second: int = 2, burst_size: int = 4, max_retries: int = 5):
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size
        self.max_retries = max_retries
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        self.consecutive_429s = 0
        self.backoff_until = 0
    
    async def acquire(self):
        async with self.lock:
            # Check if we're in backoff period
            now = time.time()
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                logging.warning(f"Rate limiter in backoff mode, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            
            # Update tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.calls_per_second)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Wait for next token
            wait_time = (1 - self.tokens) / self.calls_per_second
            await asyncio.sleep(wait_time)
            self.tokens = 0
    
    async def handle_429_error(self):
        """Handle 429 errors with exponential backoff"""
        async with self.lock:
            self.consecutive_429s += 1
            # Track 429 errors in session state
            if hasattr(st, 'session_state'):
                st.session_state.last_429_count = getattr(st.session_state, 'last_429_count', 0) + 1
            
            # Exponential backoff: 2^attempts seconds, max 60 seconds
            backoff_time = min(60, 2 ** self.consecutive_429s)
            self.backoff_until = time.time() + backoff_time
            logging.warning(f"429 error #{self.consecutive_429s}, backing off for {backoff_time}s")
    
    def reset_429_counter(self):
        """Reset 429 counter on successful request"""
        self.consecutive_429s = 0
        self.backoff_until = 0

class RetryableAPICall:
    """Wrapper for API calls with retry logic and 429 handling"""
    
    def __init__(self, rate_limiter: AsyncRateLimiter, max_retries: int = 3):
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
    
    async def execute(self, api_func, *args, **kwargs):
        """Execute API call with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()
                
                # Execute API call
                result = await self._execute_api_call(api_func, *args, **kwargs)
                
                # Reset 429 counter on success
                self.rate_limiter.reset_429_counter()
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Check for 429 error
                if "429" in error_str or "Too Many Requests" in error_str:
                    await self.rate_limiter.handle_429_error()
                    
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff per attempt
                        logging.warning(f"429 error, attempt {attempt + 1}/{self.max_retries}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                
                # For non-429 errors or final attempt
                if attempt == self.max_retries - 1:
                    logging.error(f"API call failed after {self.max_retries} attempts: {error_str}")
                    raise
                else:
                    # Brief wait for other errors
                    await asyncio.sleep(0.5)
        
        raise Exception(f"API call failed after {self.max_retries} attempts")
    
    async def _execute_api_call(self, api_func, *args, **kwargs):
        """Execute the actual API call in thread executor"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, lambda: api_func(*args, **kwargs))

# ======================== ENHANCED ASYNC FETCHING WITH GATHER ========================

class ValidatedHyperliquidFetcher:
    """Enhanced fetcher with Pydantic validation and asyncio.gather"""
    
    def __init__(self, info_client, max_concurrent: int = 3, calls_per_second: int = 2):
        self.info = info_client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.rate_limiter = AsyncRateLimiter(calls_per_second)
        self.api_caller = RetryableAPICall(self.rate_limiter)
        self.validation_stats = {
            'total_processed': 0,
            'validation_errors': 0,
            'warnings': 0,
            'gather_errors': 0
        }
        
    async def fetch_validated_markets_with_gather(self, symbols: List[str], 
                                                min_liquidity: float = 10000,
                                                batch_size: int = 10) -> List[MarketInfo]:
        """
        Fetch markets using asyncio.gather with Pydantic validation
        Processes in batches to manage memory and API limits
        """
        st.info("🚀 Fetching market data with asyncio.gather + Pydantic validation...")
        
        all_validated_markets = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        # Process symbols in batches
        for batch_idx in range(0, len(symbols), batch_size):
            batch_symbols = symbols[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            st.subheader(f"📦 Processing Batch {batch_num}/{total_batches}")
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            try:
                # Create tasks for concurrent execution
                status_text.text(f"🔄 Creating {len(batch_symbols)} concurrent tasks...")
                
                tasks = [
                    self._fetch_single_market_validated(symbol, batch_idx * batch_size + i, len(batch_symbols))
                    for i, symbol in enumerate(batch_symbols)
                ]
                
                status_text.text(f"⚡ Executing {len(tasks)} tasks with asyncio.gather...")
                
                # Execute all tasks concurrently with gather
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results from gather
                validated_batch = self._process_gather_results(
                    batch_symbols, batch_results, min_liquidity
                )
                
                # Update progress
                progress_bar.progress(1.0)
                processing_time = time.time() - start_time
                
                # Display batch results
                self._display_batch_results(
                    batch_num, batch_symbols, validated_batch, processing_time
                )
                
                all_validated_markets.extend(validated_batch)
                
            except Exception as e:
                st.error(f"❌ Batch {batch_num} failed: {str(e)}")
                logging.error(f"Batch {batch_num} failed: {str(e)}")
            
            finally:
                progress_bar.empty()
                status_text.empty()
                
                # Longer delay between batches to avoid rate limits
                if batch_num < total_batches:
                    delay_time = min(5.0, 2.0 + (self.rate_limiter.consecutive_429s * 0.5))
                    st.info(f"⏳ Waiting {delay_time}s before next batch to respect rate limits...")
                    await asyncio.sleep(delay_time)
        
        # Final validation and filtering
        final_markets = self._final_validation_and_filtering(all_validated_markets, min_liquidity)
        
        # Display final statistics
        self._display_final_stats(symbols, final_markets)
        
        return final_markets
    
    async def _fetch_single_market_validated(self, symbol: str, index: int, total: int) -> Optional[MarketInfo]:
        """
        Fetch and validate a single market's data
        This is the core function that gets called by gather()
        """
        async with self.semaphore:  # Rate limiting
            try:
                # Create subtasks for different data types
                price_data = await self._fetch_price_data(symbol)
                
                # Price is needed for volume, so fetch it first.
                if isinstance(price_data, Exception) or not price_data.get('mark_price', 0):
                    logging.warning(f"No valid price for {symbol}, skipping.")
                    return None

                mark_price = price_data.get('mark_price', 0.0)

                # Now fetch all other data in parallel
                volume_data, funding_data, change_data, orderbook_data = await asyncio.gather(
                    self._fetch_volume_data(symbol, mark_price),
                    self._fetch_funding_data(symbol),
                    self._fetch_price_change_data(symbol),
                    self._fetch_orderbook_metrics(symbol),
                    return_exceptions=True
                )
                
                # Construct market data dictionary
                raw_market_data = {
                    "symbol": symbol,
                    "markPrice": mark_price,
                    "lastPrice": self._safe_extract(price_data, 'last_price', 0.0),
                    "fundingRate": self._safe_extract(funding_data, 'funding_rate', 0.0),
                    "volume24h": self._safe_extract(volume_data, 'volume_24h', 10000.0),
                    "change24h": self._safe_extract(change_data, 'change_24h', 0.0),
                    "spreadBps": self._safe_extract(orderbook_data, 'spread_bps', 999.0),
                    "liquidityScore": self._safe_extract(orderbook_data, 'liquidity_score', 0.0),
                    "spreadPct": self._safe_extract(orderbook_data, 'spread_pct', 0.0),
                    "spreadRatio": self._safe_extract(orderbook_data, 'spread_ratio', 0.0),
                    "bidLiquidity": self._safe_extract(orderbook_data, 'bid_liquidity', 0.0),
                    "askLiquidity": self._safe_extract(orderbook_data, 'ask_liquidity', 0.0),
                    "bidDepth1pct": self._safe_extract(orderbook_data, 'bid_depth_1pct', 0.0),
                    "askDepth1pct": self._safe_extract(orderbook_data, 'ask_depth_1pct', 0.0),
                    "imbalance": self._safe_extract(orderbook_data, 'imbalance', 0.0),
                    "bestBid": self._safe_extract(orderbook_data, 'best_bid', 0.0),
                    "bestAsk": self._safe_extract(orderbook_data, 'best_ask', 0.0)
                }
                
                # Validate with Pydantic
                validated_market = MarketInfo(**raw_market_data)
                
                if validated_market:
                    self.validation_stats['total_processed'] += 1
                    return validated_market
                else:
                    self.validation_stats['validation_errors'] += 1
                    return None
                    
            except Exception as e:
                logging.error(f"Error fetching {symbol}: {str(e)}")
                self.validation_stats['gather_errors'] += 1
                return None
    
    def _process_gather_results(self, symbols: List[str], results: List[Any], 
                              min_liquidity: float) -> List[MarketInfo]:
        """Process results from asyncio.gather()"""
        validated_markets = []
        exceptions_count = 0
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                exceptions_count += 1
                logging.error(f"Gather exception for {symbol}: {str(result)}")
            elif result is not None and isinstance(result, MarketInfo):
                # Apply liquidity filter
                if result.volume24h >= min_liquidity:
                    validated_markets.append(result)
                else:
                    logging.debug(
                        f"Filtered out {symbol}: volume {result.volume24h} < {min_liquidity}"
                    )
        
        self.validation_stats['gather_errors'] += exceptions_count
        return validated_markets
    
    async def _fetch_price_data(self, symbol: str) -> Dict[str, float]:
        """Fetch price data with enhanced retry logic"""
        try:
            # Use the new retryable API caller
            all_mids = await self.api_caller.execute(
                lambda: self.info.all_mids()
            )
            
            price = float(all_mids.get(symbol, 0))
            return {"mark_price": price, "last_price": price}
            
        except Exception as e:
            logging.error(f"Price fetch failed for {symbol}: {str(e)}")
            # Return a safe fallback instead of raising
            return {"mark_price": 0.0, "last_price": 0.0}
    
    async def _fetch_volume_data(self, symbol: str, price: float) -> Dict[str, float]:
        """Fetch 24h volume data and convert to USD with retry logic"""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (24 * 60 * 60 * 1000)
            
            candles = await self.api_caller.execute(
                lambda: self.info.candles_snapshot(symbol, "1h", start_time, end_time)
            )
            
            if candles and isinstance(candles, list):
                coin_volume = sum(float(candle.get("v", 0)) for candle in candles)
                usd_volume = coin_volume * price
                return {"volume_24h": max(usd_volume, 10000)}
            
            # Fallback to orderbook estimation
            return await self._estimate_volume_from_orderbook(symbol, price)
            
        except Exception as e:
            logging.warning(f"Volume fetch failed for {symbol}: {str(e)}")
            return {"volume_24h": 10000.0}  # Safe fallback
    
    async def _fetch_funding_data(self, symbol: str) -> Dict[str, float]:
        """Fetch funding rate data with retry logic"""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (24 * 60 * 60 * 1000)
            
            funding_history = await self.api_caller.execute(
                lambda: self.info.funding_history(symbol, start_time)
            )
            
            if funding_history and len(funding_history) > 0:
                latest_funding = funding_history[0]
                raw_rate = float(latest_funding.get("fundingRate", 0))
                funding_bps = raw_rate * 10000 * 3  # Convert to annualized bps
                return {"funding_rate": funding_bps}
            
            return {"funding_rate": 0.0}
            
        except Exception as e:
            logging.warning(f"Funding fetch failed for {symbol}: {str(e)}")
            return {"funding_rate": 0.0}
    
    async def _fetch_price_change_data(self, symbol: str) -> Dict[str, float]:
        """Fetch 24h price change with retry logic"""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (48 * 60 * 60 * 1000)
            
            candles = await self.api_caller.execute(
                lambda: self.info.candles_snapshot(symbol, "1h", start_time, end_time)
            )
            
            if candles and len(candles) >= 24:
                old_price = float(candles[0].get("o", 0))
                current_price = float(candles[-1].get("c", 0))
                
                if old_price > 0:
                    change_pct = ((current_price - old_price) / old_price) * 100
                    return {"change_24h": change_pct}
            
            return {"change_24h": 0.0}
            
        except Exception as e:
            logging.warning(f"Price change fetch failed for {symbol}: {str(e)}")
            return {"change_24h": 0.0}
    
    async def _fetch_orderbook_metrics(self, symbol: str) -> Dict[str, float]:
        """Fetch orderbook metrics (spread, liquidity) with retry logic"""
        try: 
            # Determine symbol formats: if already a pair use directly, else try fallbacks
            if "/" in symbol:
                symbol_formats = [symbol]
            else:
                symbol_formats = [f"{symbol}/USDC:USDC", f"{symbol}:USDC", symbol]
            orderbook = None
            for perp_symbol in symbol_formats:
                try:
                    orderbook = await self.api_caller.execute(
                        lambda sym=perp_symbol: self.info.l2_snapshot(sym)
                    )
                    if orderbook and "levels" in orderbook:
                        logging.debug(f"Orderbook fetched with format: {perp_symbol}")
                        break
                except Exception as e:
                    logging.debug(f"Orderbook format {perp_symbol} failed: {e}")
                    continue
            
            if not orderbook or "levels" not in orderbook:
                return {
                    "spread_bps": 999.0,
                    "spread_pct": 0,
                    "spread_ratio": 0,
                    "liquidity_score": 0.0,
                    "bid_liquidity": 0,
                    "ask_liquidity": 0,
                    "bid_depth_1pct": 0,
                    "ask_depth_1pct": 0,
                    "imbalance": 0,
                    "best_bid": 0,
                    "best_ask": 0
                }
            
            # Process orderbook
            parsed_levels = ValidatedHyperliquidFetcher.parse_orderbook_levels(orderbook["levels"])
            
            if parsed_levels:
                bids = sorted(
                    [level for level in parsed_levels if level["side"] == 'b'],
                    key=lambda x: x["price"],
                    reverse=True  # Sort bids descending (best bid first)
                )
                asks = sorted(
                    [level for level in parsed_levels if level["side"] == 'a'],
                    key=lambda x: x["price"]  # Sort asks ascending (best ask first)
                )
                
                if bids and asks:
                    best_bid = bids[0]["price"]  # Already the max due to sorting
                    best_ask = asks[0]["price"]  # Already the min due to sorting
                    
                    # Improved spread calculation with safety checks
                    try:
                        mid_price = (best_bid + best_ask) / 2
                        spread_abs = best_ask - best_bid
                        spread_bps = (spread_abs / mid_price) * 10000 if mid_price > 0 else 999.0
                        
                        # Additional spread metrics
                        spread_pct = spread_abs / mid_price * 100 if mid_price > 0 else 0
                        spread_ratio = spread_abs / mid_price if mid_price > 0 else 0
                    except ZeroDivisionError:
                        spread_bps = 999.0
                        spread_pct = 0
                        spread_ratio = 0
                    
                    # Enhanced liquidity calculation
                    def calculate_depth_levels(levels, depth_pct=0.5):
                        """Calculate liquidity within price depth percentage"""
                        if not levels:
                            return 0
                            
                        top_price = levels[0]["price"]
                        depth_price = top_price * (1 - depth_pct/100) if levels[0]["side"] == 'b' else top_price * (1 + depth_pct/100)
                        
                        total_size = 0
                        for level in levels:
                            if ((level["side"] == 'b' and level["price"] >= depth_price) or
                                (level["side"] == 'a' and level["price"] <= depth_price)):
                                total_size += level["size"]
                            else:
                                break
                        return total_size
                    
                    # Calculate liquidity using multiple metrics
                    bid_liquidity_10_levels = sum(level["size"] for level in bids[:10])
                    ask_liquidity_10_levels = sum(level["size"] for level in asks[:10])
                    
                    bid_liquidity_1pct = calculate_depth_levels(bids, 1.0)
                    ask_liquidity_1pct = calculate_depth_levels(asks, 1.0)
                    
                    # Weighted liquidity score
                    liquidity_score = (
                        0.4 * (bid_liquidity_10_levels + ask_liquidity_10_levels) +
                        0.6 * (bid_liquidity_1pct + ask_liquidity_1pct)
                    )
                    
                    # Additional market depth metrics
                    bid_ask_imbalance = (bid_liquidity_10_levels - ask_liquidity_10_levels) / \
                                       (bid_liquidity_10_levels + ask_liquidity_10_levels) if (bid_liquidity_10_levels + ask_liquidity_10_levels) > 0 else 0
                    
                    return {
                        "spread_bps": spread_bps,
                        "spread_pct": spread_pct,
                        "spread_ratio": spread_ratio,
                        "liquidity_score": liquidity_score,
                        "bid_liquidity": bid_liquidity_10_levels,
                        "ask_liquidity": ask_liquidity_10_levels,
                        "bid_depth_1pct": bid_liquidity_1pct,
                        "ask_depth_1pct": ask_liquidity_1pct,
                        "imbalance": bid_ask_imbalance,
                        "best_bid": best_bid,
                        "best_ask": best_ask
                    }
            
            return {
                "spread_bps": 999.0,
                "spread_pct": 0,
                "spread_ratio": 0,
                "liquidity_score": 0.0,
                "bid_liquidity": 0,
                "ask_liquidity": 0,
                "bid_depth_1pct": 0,
                "ask_depth_1pct": 0,
                "imbalance": 0,
                "best_bid": 0,
                "best_ask": 0
            }
            
        except Exception as e:
            logging.warning(f"Orderbook fetch failed for {symbol}: {str(e)}")
            return {
                "spread_bps": 999.0,
                "spread_pct": 0,
                "spread_ratio": 0,
                "liquidity_score": 0.0,
                "bid_liquidity": 0,
                "ask_liquidity": 0,
                "bid_depth_1pct": 0,
                "ask_depth_1pct": 0,
                "imbalance": 0,
                "best_bid": 0,
                "best_ask": 0
            }
    
    async def _estimate_volume_from_orderbook(self, symbol: str, price: float) -> Dict[str, float]:
        """Estimate volume from orderbook when candles fail and convert to USD"""
        try:
            # Determine symbol formats for estimation
            if "/" in symbol:
                symbol_formats = [symbol]
            else:
                symbol_formats = [f"{symbol}/USDC:USDC", f"{symbol}:USDC", symbol]
            loop = asyncio.get_event_loop()
            orderbook = None
            for perp_symbol in symbol_formats:
                try:
                    orderbook = await loop.run_in_executor(
                        self.executor,
                        lambda sym=perp_symbol: self.info.l2_snapshot(sym)
                    )
                    if orderbook and "levels" in orderbook:
                        logging.debug(f"Estimation orderbook fetched with format: {perp_symbol}")
                        break
                except Exception as e:
                    logging.debug(f"Estimation format {perp_symbol} failed: {e}")
                    continue
            
            if orderbook and "levels" in orderbook:
                parsed_levels = ValidatedHyperliquidFetcher.parse_orderbook_levels(orderbook["levels"])
                total_liquidity_coin = sum(level["size"] for level in parsed_levels)
                
                if total_liquidity_coin > 0:
                    estimated_volume = total_liquidity_coin * price * 0.1  # Conservative multiplier
                    return {"volume_24h": max(estimated_volume, 10000)}
            
            return {"volume_24h": 10000.0}
            
        except Exception as e:
            logging.warning(f"Orderbook volume estimation failed for {symbol}: {str(e)}")
            return {"volume_24h": 10000.0}
    
    def _safe_extract(self, data: Any, key: str, default: Any) -> Any:
        """Safely extract data, handling exceptions"""
        if isinstance(data, Exception):
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        return default

    @staticmethod
    def parse_orderbook_levels(raw_levels: Any) -> list:
        """Helper to parse raw orderbook level data."""
        parsed = []
        for level_group in raw_levels:
            for level in level_group:
                if isinstance(level, dict) and "px" in level and "sz" in level:
                    parsed.append({
                        "side": "b" if level.get("n", 0) > 0 else "a",
                        "price": float(level["px"]),
                        "size": float(level["sz"])
                    })
        return parsed
    
    def _display_batch_results(self, batch_num: int, symbols: List[str], 
                             validated_markets: List[MarketInfo], processing_time: float):
        """Display batch processing results"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Symbols", len(symbols))
        with col2:
            st.metric("✅ Validated", len(validated_markets))
        with col3:
            st.metric("⚡ Speed", f"{processing_time:.1f}s")
        with col4:
            st.metric("🎯 Success Rate", f"{len(validated_markets)/len(symbols)*100:.1f}%")
        
        if validated_markets:
            # Show sample of validated markets
            sample_df = pd.DataFrame([
                {
                    "Symbol": m.symbol,
                    "Price": f"${m.markPrice:.4f}",
                    "Volume": f"${m.volume24h:,.0f}",
                    "Funding": f"{m.fundingRate:.1f} bps"
                }
                for m in validated_markets[:5]
            ])
            
            st.dataframe(sample_df, use_container_width=True)
    
    def _final_validation_and_filtering(self, markets: List[MarketInfo], 
                                      min_liquidity: float) -> List[MarketInfo]:
        """Final validation and filtering step"""
        st.info("🔬 Final validation and filtering...")
        
        # Additional business logic validation
        validated_markets = []
        
        for market in markets:
            try:
                # Re-validate critical fields
                if market.markPrice <= 0:
                    continue
                    
                if market.volume24h < min_liquidity:
                    continue
                
                # Check for suspicious data
                if market.spreadBps and market.spreadBps > 500:  # > 5% spread
                    logging.warning(
                        f"Wide spread for {market.symbol}: {market.spreadBps:.1f} bps"
                    )
                
                validated_markets.append(market)
                
            except Exception as e:
                logging.error(f"Final validation failed for {market.symbol}: {str(e)}")
        
        # Sort by volume descending
        validated_markets.sort(key=lambda x: x.volume24h, reverse=True)
        
        return validated_markets
    
    def _display_final_stats(self, original_symbols: List[str], final_markets: List[MarketInfo]):
        """Display final processing statistics"""
        st.success("🎉 Market fetching completed!")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📋 Total Symbols", len(original_symbols))
        with col2:
            st.metric("✅ Successfully Processed", self.validation_stats['total_processed'])
        with col3:
            st.metric("🎯 Final Markets", len(final_markets))
        with col4:
            st.metric("❌ Validation Errors", self.validation_stats['validation_errors'])
        with col5:
            st.metric("⚠️ Gather Errors", self.validation_stats['gather_errors'])
        
        if final_markets:
            overall_success_rate = len(final_markets) / len(original_symbols) * 100
            st.metric("📊 Overall Success Rate", f"{overall_success_rate:.1f}%")

# ======================== FETCH ALL MARKETS ========================
async def fetch_all_markets():
    """
    Updated fetch_all_markets function with validated concurrency
    """
    try:
        init_session_state()
        logging.info("Starting validated concurrent market fetch")

        if not validate_api_responses():
            st.error("API validation failed. Check debug info.")
            return pd.DataFrame()

        # Get metadata
        api_caller = RetryableAPICall(AsyncRateLimiter())
        meta_response = await api_caller.execute(info.meta)
        if not meta_response or 'universe' not in meta_response:
            st.error("Failed to fetch market metadata")
            return pd.DataFrame()
        
        symbols = [coin['name'] for coin in meta_response['universe']]
        st.info(f"📋 Found {len(symbols)} symbols to process")
        
        if not symbols:
            st.warning("No symbols found in metadata")
            return pd.DataFrame()

        # Initialize the validated fetcher with conservative rate limits
        fetcher = ValidatedHyperliquidFetcher(
            info,
            max_concurrent=2,  # Reduced from default
            calls_per_second=1  # Much more conservative
        )
    
        # Fetch markets with the new gather-based method (smaller batches)
        validated_markets = await fetcher.fetch_validated_markets_with_gather(
            symbols, min_liquidity=st.session_state.MIN_LIQUIDITY, batch_size=5
        )
        
        markets_df = pd.DataFrame()
        if validated_markets:
            # Convert to DataFrame
            market_dicts = [market.model_dump() for market in validated_markets]
            markets_df = pd.DataFrame(market_dicts)

        if markets_df.empty:
            st.warning(f"No markets met the minimum liquidity threshold of ${st.session_state.MIN_LIQUIDITY:,}.")
            if st.session_state.get('force_include_major', False):
                return apply_fallback_for_major_coins()
            return pd.DataFrame()

        # Sort by volume descending
        sorted_df = markets_df.sort_values('volume24h', ascending=False)
        st.success(f"🎉 Successfully fetched {len(sorted_df)} markets!")
        return sorted_df

    except Exception as e:
        st.error(f"Critical error fetching markets: {str(e)}")
        logging.error(f"Critical error fetching markets: {str(e)}")
        return pd.DataFrame()

# ======================== DATA FETCHING ========================
async def fetch_hyperliquid_candles(symbol, interval="1h", limit=50, use_websocket=True):
    debug_info = getattr(st.session_state, 'debug_info', {})
    
    # Try WebSocket first if enabled
    if use_websocket and st.session_state.get('ws_enabled', False):
        try:
            ws_data = await get_websocket_candles(symbol, interval)
            if ws_data:
                # Process WebSocket candle data
                parsed_candles = [ws_data]  # WebSocket gives us latest candle
                df = pd.DataFrame(parsed_candles, columns=['timestamp', 'time_close', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'num', 'coin'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                debug_info[f'candles_source_{symbol}'] = 'websocket'
                st.session_state.debug_info = debug_info
                
                # If we only have one candle from WebSocket, fetch more from REST for historical context
                if len(df) < limit:
                    logging.info(f"WebSocket provided 1 candle for {symbol}, fetching more from REST API for historical context")
                    rest_df = await fetch_hyperliquid_candles_rest(symbol, interval, limit-1)
                    if rest_df is not None and not rest_df.empty:
                        # Combine WebSocket (latest) with REST (historical)
                        combined_df = pd.concat([rest_df, df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        debug_info[f'candles_combined_{symbol}'] = f'WebSocket: 1, REST: {len(rest_df)}, Combined: {len(combined_df)}'
                        st.session_state.debug_info = debug_info
                        return combined_df
                
                return df
        except Exception as e:
            logging.warning(f"WebSocket candles fetch failed for {symbol}, falling back to REST: {e}")
            debug_info[f'candles_ws_error_{symbol}'] = str(e)
    
    # Fallback to REST API
    return await fetch_hyperliquid_candles_rest(symbol, interval, limit)

async def fetch_hyperliquid_candles_rest(symbol, interval="1h", limit=50):
    """REST API version of candle fetching"""
    debug_info = getattr(st.session_state, 'debug_info', {})
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
        
        # Always try base symbol first for candles
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        symbol_formats = [base_symbol, f"{base_symbol}/USDC:USDC"]
        ohlcv = None
        errors = []
        
        for format_idx, sym_format in enumerate(symbol_formats):
            try:
                debug_info[f'trying_format_{symbol}_{format_idx}'] = sym_format
                ohlcv = await async_api_request(info.candles_snapshot, sym_format, timeframe, start_time, end_time)
                if ohlcv and isinstance(ohlcv, list) and len(ohlcv) > 0:
                    debug_info[f'candles_success_{symbol}'] = f'Found data using format: {sym_format}'
                    break
                else:
                    errors.append(f"Empty response for format {sym_format}")
            except Exception as e:
                error_msg = f"Error with format {sym_format}: {str(e)}"
                errors.append(error_msg)
                debug_info[f'candles_format_error_{symbol}_{format_idx}'] = error_msg
        
        if not ohlcv or len(ohlcv) == 0:
            debug_info[f'candles_empty_{symbol}'] = f'No candle data returned for any format. Errors: {errors}'
            logging.warning(f"No candle data for {symbol}. Tried formats: {symbol_formats}")
            st.session_state.debug_info = debug_info
            return None
            
        parsed_ohlcv = parse_candles(ohlcv)
        df = pd.DataFrame(parsed_ohlcv, columns=['timestamp', 'time_close', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'num'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        debug_info[f'candles_source_{symbol}'] = 'rest_api'
        st.session_state.debug_info = debug_info
        return df
    except Exception as e:
        debug_info[f'candles_error_{symbol}'] = str(e)
        st.session_state.debug_info = debug_info
        st.error(f"Error fetching candles for {symbol}: {str(e)}")
        logging.error(f"Error fetching candles for {symbol}: {str(e)}")
        return None

# The rest of the code (Database Functions, Trading Logic, Signal Generation, Streamlit UI) remains unchanged unless further issues are identified.
# Below are placeholders for the remaining sections to ensure completeness.

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

async def fetch_realtime_orderbook(symbol: str, use_websocket: bool = True) -> Dict[str, Any]:
    """
    Fetch real-time orderbook data for a specific market with WebSocket fallback to REST API
    
    Args:
        symbol: Market symbol (e.g. 'BTC')
        use_websocket: Whether to try WebSocket first
        
    Returns:
        Dictionary with parsed orderbook data including:
        - bids: List of bid levels sorted from highest to lowest
        - asks: List of ask levels sorted from lowest to highest
        - metrics: Various orderbook metrics
    """
    # Try WebSocket first if enabled
    if use_websocket and st.session_state.get('ws_enabled', False):
        try:
            ws_data = await get_websocket_orderbook(symbol)
            if ws_data and ws_data.get('levels'):
                # Process WebSocket data 
                parsed_levels = ws_data['levels']
                
                if parsed_levels:
                    bids = sorted(
                        [level for level in parsed_levels if level["side"] == 'b'],
                        key=lambda x: x["price"],
                        reverse=True
                    )
                    asks = sorted(
                        [level for level in parsed_levels if level["side"] == 'a'],
                        key=lambda x: x["price"]
                    )
                    
                    if bids and asks:
                        best_bid = bids[0]["price"]
                        best_ask = asks[0]["price"]
                        
                        # Calculate metrics
                        mid_price = (best_bid + best_ask) / 2
                        spread_abs = best_ask - best_bid
                        spread_bps = (spread_abs / mid_price) * 10000 if mid_price > 0 else 999.0
                        spread_pct = spread_abs / mid_price * 100 if mid_price > 0 else 0
                        
                        # Calculate liquidity
                        bid_liquidity = sum(level["size"] for level in bids[:10])
                        ask_liquidity = sum(level["size"] for level in asks[:10])
                        
                        # Depth calculation 
                        def calculate_depth_levels(levels, depth_pct=1.0):
                            if not levels:
                                return 0
                            top_price = levels[0]["price"]
                            depth_price = top_price * (1 - depth_pct/100) if levels[0]["side"] == 'b' else top_price * (1 + depth_pct/100)
                            
                            total_size = 0
                            for level in levels:
                                if ((level["side"] == 'b' and level["price"] >= depth_price) or
                                    (level["side"] == 'a' and level["price"] <= depth_price)):
                                    total_size += level["size"]
                                else:
                                    break
                            return total_size
                        
                        bid_depth_1pct = calculate_depth_levels(bids, 1.0)
                        ask_depth_1pct = calculate_depth_levels(asks, 1.0)
                        
                        # Liquidity score
                        liquidity_score = (
                            0.4 * (bid_liquidity + ask_liquidity) +
                            0.6 * (bid_depth_1pct + ask_depth_1pct)
                        )
                        
                        # Imbalance
                        imbalance = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity) if (bid_liquidity + ask_liquidity) > 0 else 0
                        
                        return {
                            "bids": bids[:20],  # Return top 20 levels
                            "asks": asks[:20],
                            "metrics": {
                                "spread_bps": spread_bps,
                                "spread_pct": spread_pct,
                                "liquidity_score": liquidity_score,
                                "bid_liquidity": bid_liquidity,
                                "ask_liquidity": ask_liquidity,
                                "bid_depth_1pct": bid_depth_1pct,
                                "ask_depth_1pct": ask_depth_1pct,
                                "imbalance": imbalance,
                                "best_bid": best_bid,
                                "best_ask": best_ask,
                                "mid_price": mid_price,
                                "source": "websocket"
                            }
                        }
        except Exception as e:
            logging.warning(f"WebSocket orderbook fetch failed for {symbol}, falling back to REST: {e}")
    
    # Fallback to REST API
    try:
        perp_symbol = f"{symbol}/USDC:USDC"
        
        # Create a rate limiter for this function
        if not hasattr(fetch_realtime_orderbook, 'rate_limiter'):
            fetch_realtime_orderbook.rate_limiter = AsyncRateLimiter(calls_per_second=1)
        
        await fetch_realtime_orderbook.rate_limiter.acquire()
        
        # Create a new Info client for this request to avoid threading issues
        local_info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
        # Use thread executor to make the sync call async
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            orderbook = await loop.run_in_executor(
                executor, 
                lambda: local_info.l2_snapshot(perp_symbol)
            )
        
        if not orderbook or "levels" not in orderbook:
            return {
                "bids": [],
                "asks": [],
                "metrics": {
                    "spread_bps": 999.0,
                    "spread_pct": 0,
                    "liquidity_score": 0.0,
                    "mid_price": 0.0,
                    "source": "rest_api_failed"
                }
            }
        
        # Process orderbook
        parsed_levels = parse_orderbook_levels(orderbook["levels"])
        
        if parsed_levels:
            bids = sorted(
                [level for level in parsed_levels if level["side"] == 'b'],
                key=lambda x: x["price"],
                reverse=True  # Sort bids descending (best bid first)
            )
            asks = sorted(
                [level for level in parsed_levels if level["side"] == 'a'],
                key=lambda x: x["price"]  # Sort asks ascending (best ask first)
            )
            
            if bids and asks:
                best_bid = bids[0]["price"]
                best_ask = asks[0]["price"]
                
                # Calculate metrics
                mid_price = (best_bid + best_ask) / 2
                spread_abs = best_ask - best_bid
                spread_bps = (spread_abs / mid_price) * 10000 if mid_price > 0 else 999.0
                spread_pct = spread_abs / mid_price * 100 if mid_price > 0 else 0
                
                # Calculate liquidity
                bid_liquidity = sum(level["size"] for level in bids[:10])
                ask_liquidity = sum(level["size"] for level in asks[:10])
                
                # Depth calculation 
                def calculate_depth_levels(levels, depth_pct=1.0):
                    if not levels:
                        return 0
                    top_price = levels[0]["price"]
                    depth_price = top_price * (1 - depth_pct/100) if levels[0]["side"] == 'b' else top_price * (1 + depth_pct/100)
                    
                    total_size = 0
                    for level in levels:
                        if ((level["side"] == 'b' and level["price"] >= depth_price) or
                            (level["side"] == 'a' and level["price"] <= depth_price)):
                            total_size += level["size"]
                        else:
                            break
                    return total_size
                
                bid_depth_1pct = calculate_depth_levels(bids, 1.0)
                ask_depth_1pct = calculate_depth_levels(asks, 1.0)
                
                # Liquidity score
                liquidity_score = (
                    0.4 * (bid_liquidity + ask_liquidity) +
                    0.6 * (bid_depth_1pct + ask_depth_1pct)
                )
                
                # Imbalance
                imbalance = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity) if (bid_liquidity + ask_liquidity) > 0 else 0
                
                return {
                    "bids": bids[:20],  # Return top 20 levels
                    "asks": asks[:20],
                    "metrics": {
                        "spread_bps": spread_bps,
                        "spread_pct": spread_pct,
                        "liquidity_score": liquidity_score,
                        "bid_liquidity": bid_liquidity,
                        "ask_liquidity": ask_liquidity,
                        "bid_depth_1pct": bid_depth_1pct,
                        "ask_depth_1pct": ask_depth_1pct,
                        "imbalance": imbalance,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": mid_price,
                        "source": "rest_api"
                    }
                }
        
        return {
            "bids": [],
            "asks": [],
            "metrics": {
                "spread_bps": 999.0,
                "spread_pct": 0,
                "liquidity_score": 0.0,
                "mid_price": 0.0,
                "source": "rest_api_empty"
            }
        }
        
    except Exception as e:
        logging.error(f"Error fetching real-time orderbook for {symbol}: {str(e)}")
        return {
            "bids": [],
            "asks": [],
            "metrics": {
                "spread_bps": 999.0,
                "spread_pct": 0,
                "liquidity_score": 0.0,
                "mid_price": 0.0,
                "error": str(e),
                "source": "error"
            }
        }

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
            logging.error(f"Could not load state: {str(e)}")
    
    def save_state(self):
        try:
            save_state_to_db(
                {k: v.model_dump() for k, v in self.active_trades.items()},
                [t.model_dump() for t in self.completed_trades]
            )
        except Exception as e:
            st.error(f"Error saving state to database: {str(e)}")
            logging.error(f"Error saving state: {str(e)}")
    
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
        markets_df = asyncio.run(fetch_all_markets())
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
                    logging.error(f"Error updating trade {symbol}: {str(e)}")
            for symbol in to_remove:
                self.active_trades.pop(symbol)
        self.save_state()
        return updates
    
    def get_performance_report(self):
        if not self.completed_trades:
            return "No completed trades yet", pd.DataFrame()
        df = pd.DataFrame([t.model_dump() for t in self.completed_trades])
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
async def generate_signals(markets_df):
    debug_info = getattr(st.session_state, 'debug_info', {})
    if markets_df is None or markets_df.empty:
        st.warning("No market data available for signal generation")
        debug_info['signals_error'] = 'Empty markets DataFrame'
        st.session_state.debug_info = debug_info
        logging.warning("No market data for signal generation")
        return []
    
    signals = []
    top_markets = markets_df.head(15)
    total_markets = len(top_markets)
    micro_price_calc = MicroPrice(n_imb=10, n_spread=5, alpha=2.0)
    progress_bar = st.progress(0)
    status_text = st.empty()
    MICRO_PRICE_THRESHOLD = 0.0005  # Reduced from 0.001 to 0.0005 (0.05%)
    FUNDING_THRESHOLD = 60  # Default value if not available from sidebar

    # Create rate limiter for signal generation
    signal_rate_limiter = AsyncRateLimiter(calls_per_second=1)
    signal_api_caller = RetryableAPICall(signal_rate_limiter)

    # Enhanced signal generation with gather
    st.info("🔬 Generating signals with asyncio.gather...")
    
    async def process_market_signal(index, market):
        symbol = market['symbol']
        perp_symbol = f"{symbol}/USDC:USDC"
        status_text.text(f"Analyzing {symbol}... ({index + 1}/{total_markets})")
        debug_info = getattr(st.session_state, 'debug_info', {})
        
        try:
            # Create concurrent tasks for signal generation with retry logic
            orderbook_task = signal_api_caller.execute(lambda: info.l2_snapshot(perp_symbol))
            candles_task = fetch_hyperliquid_candles(symbol, interval='1h', limit=48)
            
            # Execute tasks concurrently with gather
            orderbook_response, df = await asyncio.gather(
                orderbook_task, candles_task, return_exceptions=True
            )
            
            # Handle orderbook response
            if isinstance(orderbook_response, Exception) or not orderbook_response or 'levels' not in orderbook_response:
                debug_info[f'no_orderbook_{symbol}'] = 'Empty or invalid order book'
                st.session_state.debug_info = debug_info
                return None
                
            parsed_levels = ValidatedHyperliquidFetcher.parse_orderbook_levels(orderbook_response['levels'])
            
            # Convert parsed levels to MarketLevel objects
            market_levels = []
            for level in parsed_levels:
                try:
                    market_levels.append(MarketLevel(
                        side=level['side'],
                        price=level['price'], 
                        size=level['size']
                    ))
                except Exception as e:
                    continue  # Skip invalid levels
            
            if not market_levels:
                debug_info[f'no_valid_levels_{symbol}'] = 'No valid market levels'
                st.session_state.debug_info = debug_info
                return None
                
            book = OrderBook(levels=market_levels)
            bids = [level for level in book.levels if level.side == 'b']
            asks = [level for level in book.levels if level.side == 'a']
            
            if not bids or not asks:
                debug_info[f'no_bids_asks_{symbol}'] = 'No bids or asks in order book'
                st.session_state.debug_info = debug_info
                return None
            
            # Use multiple levels for more stable calculation
            best_bid = max(bid.price for bid in bids)
            best_ask = min(ask.price for ask in asks)
            
            # Consider top 3 price levels instead of just best bid/ask
            top_n = 3
            top_bids = sorted([(bid.price, bid.size) for bid in bids], key=lambda x: x[0], reverse=True)[:top_n]
            top_asks = sorted([(ask.price, ask.size) for ask in asks], key=lambda x: x[0])[:top_n]
            
            # Calculate aggregated size across top levels
            bid_size = sum(size for _, size in top_bids)
            ask_size = sum(size for _, size in top_asks)
            
            # Calculate micro-price with improved inputs
            micro_price = micro_price_calc.calculate(best_bid, best_ask, bid_size, ask_size)
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            
            # Handle candles response
            if isinstance(df, Exception) or df is None or len(df) < 6:
                debug_info[f'no_candles_{symbol}'] = 'Insufficient candle data'
                st.session_state.debug_info = debug_info
                logging.warning(f"Insufficient candle data for {symbol}")
                return None
                
            recent_vols = df['volume'].tail(3)
            avg_vol = df['volume'].mean()
            recent_vol = recent_vols.mean()
            vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
            vol_consistent = all(v > avg_vol * 0.7 for v in recent_vols)
            
            df['hlrange'] = df['high'] - df['low']
            avg_range = df['hlrange'].mean()
            volatility_factor = min(max(avg_range / df['close'].iloc[-1], 0.01), 0.05)
            
            signal = "HOLD"
            reason = ""
            tp = "-"
            sl = "-"
            funding_rate = market['fundingRate']
            current_price = df['close'].iloc[-1]
            micro_price_deviation = (micro_price - mid_price) / mid_price
            
            # Lower threshold for more signals
            price_threshold = max(MICRO_PRICE_THRESHOLD, volatility_factor * 0.25)
            max_spread_bps = 75  # Increased from 50 to allow more assets
            tp_distance = max(volatility_factor * 5, 0.02)
            sl_distance = max(volatility_factor * 3, 0.015)
            
            # Add debug output
            debug_info[f'micro_price_{symbol}'] = micro_price
            debug_info[f'mid_price_{symbol}'] = mid_price
            debug_info[f'deviation_{symbol}'] = micro_price_deviation
            debug_info[f'threshold_{symbol}'] = price_threshold
            debug_info[f'best_bid_{symbol}'] = best_bid
            debug_info[f'best_ask_{symbol}'] = best_ask
            debug_info[f'bid_size_{symbol}'] = bid_size
            debug_info[f'ask_size_{symbol}'] = ask_size
            
            # Correct signal logic interpretation
            if spread_bps <= max_spread_bps and vol_surge >= 1.3 and vol_consistent:
                local_funding_threshold = FUNDING_THRESHOLD / 2
                
                # Consider book imbalance correctly - REVERSED from previous logic
                # micro_price < mid_price indicates selling pressure (LONG signal)
                # micro_price > mid_price indicates buying pressure (SHORT signal)
                if micro_price_deviation < -price_threshold and funding_rate < -local_funding_threshold:
                    signal = "LONG"  # Buy when selling pressure (micro_price < mid) and negative funding
                    reason = f"Selling pressure detected: μP {micro_price_deviation:.4f}, Vol surge {vol_surge:.2f}x, Funding {funding_rate:.2f} bps"
                    tp = str(round(current_price * (1 + tp_distance), 4))
                    sl = str(round(current_price * (1 - sl_distance), 4))
                elif micro_price_deviation > price_threshold and funding_rate > local_funding_threshold:
                    signal = "SHORT"  # Sell when buying pressure (micro_price > mid) and positive funding
                    reason = f"Buying pressure detected: μP {micro_price_deviation:.4f}, Vol surge {vol_surge:.2f}x, Funding {funding_rate:.2f} bps"
                    tp = str(round(current_price * (1 - tp_distance), 4))
                    sl = str(round(current_price * (1 + sl_distance), 4))
            
            st.session_state.debug_info = debug_info
            return Signal(
                Symbol=symbol,
                Price=current_price,
                MarkPrice=market['markPrice'],
                Signal=signal,
                Volume24h=market['volume24h'],
                FundingRate=funding_rate,
                VolSurge=vol_surge,
                Change24h=market['change24h'],
                spreadBps=spread_bps,
                Reason=reason,
                TP=tp,
                SL=sl
            )
            
        except Exception as e:
            debug_info[f'error_{symbol}'] = str(e)
            st.session_state.debug_info = debug_info
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    # Create tasks for all markets first
    tasks = [process_market_signal(i, market) for i, (_, market) in enumerate(top_markets.iterrows())]
    
    # Execute signal generation tasks in smaller batches to reduce API pressure
    batch_size = 3  # Process 3 signals at a time
    all_signals = []
    
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        
        st.info(f"⚡ Processing signal batch {batch_num}/{total_batches} ({len(batch_tasks)} signals)...")
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process batch results
        for result in batch_results:
            if isinstance(result, Exception):
                logging.error(f"Signal generation exception: {str(result)}")
            elif result is not None:
                all_signals.append(result)
        
        # Small delay between signal batches
        if i + batch_size < len(tasks):
            await asyncio.sleep(1.0)
    
    signals = all_signals

    progress_bar.progress(1.0)
    status_text.empty()
    debug_info['signals_generated'] = len(signals)
    st.session_state.debug_info = debug_info
    logging.info(f"Generated {len(signals)} signals using asyncio.gather")
    return signals

# ======================== SCAN MARKETS ========================
def scan_markets():
    init_session_state()
    st.session_state.scanned_markets = pd.DataFrame()
    st.session_state.signals = []
    
    try:
        with st.spinner("Fetching market data from Hyperliquid..."):
            markets_df = asyncio.run(fetch_all_markets())
            if markets_df.empty:
                debug = st.session_state.debug_info
                if 'skipped_samples' in debug and debug['skipped_samples']:
                    st.warning(f"All markets were below the liquidity threshold of ${MIN_LIQUIDITY:,}.")
                else:
                    st.warning("No market data fetched. Check debug info.")
                logging.warning("No markets fetched")
                return
            st.session_state.scanned_markets = markets_df
            st.success(f"Found {len(markets_df)} markets meeting the minimum liquidity threshold of ${MIN_LIQUIDITY:,} USD.")
    except Exception as e:
        debug_info = getattr(st.session_state, 'debug_info', {})
        debug_info['scan_error'] = str(e)
        st.session_state.debug_info = debug_info
        st.error(f"Error during market scan: {str(e)}")
        logging.error(f"Market scan error: {str(e)}")
        return
    
    try:
        if not markets_df.empty:
            with st.spinner("Analyzing markets for trading signals..."):
                signals = asyncio.run(generate_signals(markets_df))
                st.session_state.signals = signals
                actionable_count = len([s for s in signals if s.Signal != 'HOLD'])
                if actionable_count > 0:
                    st.success(f"Analysis complete. Found {actionable_count} actionable signals.")
                else:
                    st.info("Analysis complete. No actionable signals found with current parameters.")
    except Exception as e:
        debug_info = getattr(st.session_state, 'debug_info', {})
        debug_info['signal_gen_error'] = str(e)
        st.session_state.debug_info = debug_info
        st.error(f"Error during signal generation: {str(e)}")
        logging.error(f"Signal generation error: {str(e)}")
        return

# ======================== STREAMLIT UI ========================
tester = ForwardTester()

with st.sidebar:
    st.header("Parameters")
    BASE_VOL = st.slider("Base Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    
    liquidity_options = {
        "1,000 USD": 1000,
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
        index=0
    )
    
    MIN_LIQUIDITY = liquidity_options[selected_liquidity]
    st.session_state.MIN_LIQUIDITY = MIN_LIQUIDITY
    
    # WebSocket Controls
    with st.expander("🔗 WebSocket Settings", expanded=True):
        st.markdown("**Real-time Data Connection**")
        
        ws_enabled = st.checkbox(
            "Enable WebSocket Connection",
            value=st.session_state.get('ws_enabled', False),
            help="Use WebSocket for real-time order book and candle data to reduce API rate limiting"
        )
        
        if ws_enabled != st.session_state.get('ws_enabled', False):
            st.session_state.ws_enabled = ws_enabled
            if ws_enabled:
                st.info("🚀 WebSocket enabled! This will reduce REST API calls and provide real-time data.")
            else:
                st.info("🔄 WebSocket disabled. Using REST API only.")
        
        if ws_enabled:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect WebSocket", use_container_width=True):
                    with st.spinner("Initializing WebSocket connection..."):
                        try:
                            async def init_ws():
                                ws_manager = await initialize_websocket_manager()
                                if ws_manager:
                                    st.session_state.ws_manager = ws_manager
                                    return True
                                return False
                            
                            success = asyncio.run(init_ws())
                            
                            if success:
                                st.success("✅ WebSocket connected successfully!")
                                st.session_state.ws_connected = True
                            else:
                                st.error("❌ Failed to connect WebSocket")
                                st.session_state.ws_connected = False
                        except Exception as e:
                            st.error(f"❌ WebSocket connection error: {str(e)}")
                            st.session_state.ws_connected = False
            
            with col2:
                if st.button("🔌 Disconnect WebSocket", use_container_width=True):
                    if st.session_state.get('ws_manager'):
                        try:
                            asyncio.run(st.session_state.ws_manager.stop())
                            st.session_state.ws_manager = None
                            st.session_state.ws_connected = False
                            st.success("🔌 WebSocket disconnected.")
                        except Exception as e:
                            st.error(f"❌ WebSocket disconnection error: {str(e)}")
                            st.success("🔄 WebSocket disconnected")
                        except Exception as e:
                            st.error(f"Error disconnecting: {str(e)}")
            
            # WebSocket Status
            if st.session_state.get('ws_connected', False):
                st.success("🟢 WebSocket Status: Connected")
                
                # Show real-time data stats
                if st.session_state.get('ws_orderbooks'):
                    st.metric("📊 Real-time Order Books", len(st.session_state.ws_orderbooks))
                if st.session_state.get('ws_candles'):
                    st.metric("📈 Real-time Candles", len(st.session_state.ws_candles))
            else:
                st.warning("🔴 WebSocket Status: Disconnected")
    
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
        
        st.subheader("Rate Limiting Settings")
        
        api_calls_per_second = st.slider(
            "API calls per second",
            min_value=1.0,
            max_value=10.0,
            value=st.session_state.get('api_calls_per_second', 4.0),
            step=0.5,
            help="Maximum API calls per second for the fetcher. Lower values help prevent 429 rate limit errors."
        )
        st.session_state.api_calls_per_second = api_calls_per_second
        
        max_retries = st.slider(
            "Max retries",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of retry attempts when standalone API requests fail"
        )
        st.session_state.api_max_retries = max_retries
        
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database", "Debug"])

with tab1:
    st.info(f"Current Minimum Liquidity: ${MIN_LIQUIDITY:,} USD")
    if st.button("Scan Markets", use_container_width=True):
        scan_markets()
    if not st.session_state.scanned_markets.empty:
        st.subheader("All Markets")
        display_df = st.session_state.scanned_markets.rename(columns={
            'symbol': 'Symbol',
            'markPrice': 'Mark Price',
            'volume24h': 'Volume (24h)',
            'fundingRate': 'Funding Rate (bps)',
            'change24h': 'Change (24h)',
            'spreadBps': 'Spread (bps)',
            'liquidityScore': 'Liquidity Score',
            'bidLiquidity': 'Bid Liquidity',
            'askLiquidity': 'Ask Liquidity',
            'bidDepth1pct': 'Bid Depth (1%)',
            'askDepth1pct': 'Ask Depth (1%)',
            'imbalance': 'Order Imbalance',
            'bestBid': 'Best Bid',
            'bestAsk': 'Best Ask'
        })
        
        # Create tabs for basic and advanced metrics
        basic_tab, advanced_tab = st.tabs(["Basic Metrics", "Advanced Metrics"])
        
        with basic_tab:
            st.dataframe(
                display_df[['Symbol', 'Mark Price', 'Volume (24h)', 'Funding Rate (bps)', 'Change (24h)', 'Spread (bps)', 'Liquidity Score']],
                use_container_width=True,
                column_config={
                    'Symbol': st.column_config.TextColumn("Symbol"),
                    'Mark Price': st.column_config.NumberColumn("Mark Price", format="%.4f"),
                    'Volume (24h)': st.column_config.NumberColumn("Volume (24h)", format="$%.2f"),
                    'Funding Rate (bps)': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                    'Change (24h)': st.column_config.NumberColumn("Change (24h)", format="%.2f%%"),
                    'Spread (bps)': st.column_config.NumberColumn("Spread (bps)", format="%.2f"),
                    'Liquidity Score': st.column_config.NumberColumn("Liquidity Score", format="%.2f")
                }
            )
            
        with advanced_tab:
            st.dataframe(
                display_df[['Symbol', 'Bid Liquidity', 'Ask Liquidity', 'Bid Depth (1%)', 'Ask Depth (1%)', 'Order Imbalance', 'Best Bid', 'Best Ask']],
                use_container_width=True,
                column_config={
                    'Symbol': st.column_config.TextColumn("Symbol"),
                    'Bid Liquidity': st.column_config.NumberColumn("Bid Liquidity", format="%.2f"),
                    'Ask Liquidity': st.column_config.NumberColumn("Ask Liquidity", format="%.2f"),
                    'Bid Depth (1%)': st.column_config.NumberColumn("Bid Depth (1%)", format="%.2f"),
                    'Ask Depth (1%)': st.column_config.NumberColumn("Ask Depth (1%)", format="%.2f"),
                    'Order Imbalance': st.column_config.NumberColumn("Order Imbalance", format="%.2f"),
                    'Best Bid': st.column_config.NumberColumn("Best Bid", format="%.4f"),
                    'Best Ask': st.column_config.NumberColumn("Best Ask", format="%.4f")
                }
        )
    # Add orderbook visualization for a selected market
    if not st.session_state.scanned_markets.empty:
        st.subheader("Orderbook Visualization")
        symbols = st.session_state.scanned_markets['symbol'].tolist()
        selected_symbol = st.selectbox("Select Market for Orderbook Analysis", options=symbols)
        
        col1, col2 = st.columns(2)
        show_cached = col1.button("Show Cached Orderbook")
        show_live = col2.button("Fetch Live Orderbook")
        
        if selected_symbol and (show_cached or show_live):
            with st.spinner(f"Fetching {'live' if show_live else 'cached'} orderbook data for {selected_symbol}..."):
                try:
                    # Get data based on whether we're showing cached or live data
                    if show_live:
                        # Fetch real-time orderbook
                        orderbook_data = asyncio.run(fetch_realtime_orderbook(selected_symbol))
                        
                        # Extract metrics
                        metrics = orderbook_data['metrics']
                        market_data = {
                            'bestBid': metrics.get('best_bid', 0),
                            'bestAsk': metrics.get('best_ask', 0),
                            'spreadBps': metrics.get('spread_bps', 0),
                            'imbalance': metrics.get('imbalance', 0),
                            'bidLiquidity': metrics.get('bid_liquidity', 0),
                            'askLiquidity': metrics.get('ask_liquidity', 0),
                            'bidDepth1pct': metrics.get('bid_depth_1pct', 0),
                            'askDepth1pct': metrics.get('ask_depth_1pct', 0),
                            'liquidityScore': metrics.get('liquidity_score', 0)
                        }
                        
                        # Create a DataFrame of the orderbook levels for visualization
                        bid_df = pd.DataFrame(orderbook_data['bids'])
                        ask_df = pd.DataFrame(orderbook_data['asks'])
                        
                        # Show the raw orderbook data
                        st.subheader("Live Orderbook Data")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Bids (Buy Orders)")
                            if not bid_df.empty:
                                st.dataframe(
                                    bid_df,
                                    use_container_width=True,
                                    column_config={
                                        'price': st.column_config.NumberColumn("Price", format="%.4f"),
                                        'size': st.column_config.NumberColumn("Size", format="%.2f")
                                    }
                                )
                            else:
                                st.write("No bid data available")
                                
                        with col2:
                            st.write("Asks (Sell Orders)")
                            if not ask_df.empty:
                                st.dataframe(
                                    ask_df,
                                    use_container_width=True,
                                    column_config={
                                        'price': st.column_config.NumberColumn("Price", format="%.4f"),
                                        'size': st.column_config.NumberColumn("Size", format="%.2f")
                                    }
                                )
                            else:
                                st.write("No ask data available")
                                
                        st.success("✅ Live orderbook data fetched successfully")
                    else:
                        # Get the cached market data for this symbol
                        market_data = st.session_state.scanned_markets[st.session_state.scanned_markets['symbol'] == selected_symbol].iloc[0]
                    
                    # Create metrics display
                    st.subheader(f"Orderbook Metrics for {selected_symbol}")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Best Bid", f"{market_data.get('bestBid', 0):.4f}")
                    with col2:
                        st.metric("Best Ask", f"{market_data.get('bestAsk', 0):.4f}")
                    with col3:
                        st.metric("Spread (bps)", f"{market_data.get('spreadBps', 0):.2f}")
                    with col4:
                        st.metric("Imbalance", f"{market_data.get('imbalance', 0):.2f}")
                    
                    # Create liquidity metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Bid Liquidity", f"{market_data.get('bidLiquidity', 0):.2f}")
                    with col2:
                        st.metric("Ask Liquidity", f"{market_data.get('askLiquidity', 0):.2f}")
                    with col3:
                        st.metric("Bid Depth (1%)", f"{market_data.get('bidDepth1pct', 0):.2f}")
                    with col4:
                        st.metric("Ask Depth (1%)", f"{market_data.get('askDepth1pct', 0):.2f}")
                    
                    # Create a visual representation of the orderbook imbalance
                    st.subheader("Order Book Imbalance")
                    imbalance = market_data.get('imbalance', 0)
                    bid_ratio = (1 + imbalance) / 2  # Convert from [-1,1] to [0,1] for bid side
                    ask_ratio = (1 - imbalance) / 2  # Convert from [-1,1] to [0,1] for ask side
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Bid Side"],
                        y=[bid_ratio * 100],
                        name="Bid Liquidity",
                        marker_color="green"
                    ))
                    fig.add_trace(go.Bar(
                        x=["Ask Side"],
                        y=[ask_ratio * 100],
                        name="Ask Liquidity",
                        marker_color="red"
                    ))
                    
                    fig.update_layout(
                        title=f"Order Book Imbalance for {selected_symbol}",
                        yaxis=dict(title="Percentage of Total Liquidity"),
                        barmode="group"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Liquidity Score
                    st.subheader("Liquidity Analysis")
                    liquidity_score = market_data.get('liquidityScore', 0)
                    
                    # Create a gauge chart for liquidity score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=liquidity_score,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Liquidity Score"},
                        gauge={
                            "axis": {"range": [0, max(10000, liquidity_score * 1.2)]},
                            "steps": [
                                {"range": [0, 1000], "color": "red"},
                                {"range": [1000, 5000], "color": "yellow"},
                                {"range": [5000, max(10000, liquidity_score * 1.2)], "color": "green"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": st.session_state.MIN_LIQUIDITY
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add depth chart if we have live orderbook data
                    if show_live and 'bid_df' in locals() and 'ask_df' in locals() and not bid_df.empty and not ask_df.empty:
                        st.subheader("Orderbook Depth Chart")
                        
                        # Calculate cumulative sizes
                        bid_df = bid_df.sort_values('price', ascending=False)
                        ask_df = ask_df.sort_values('price', ascending=True)
                        
                        bid_df['cumulative_size'] = bid_df['size'].cumsum()
                        ask_df['cumulative_size'] = ask_df['size'].cumsum()
                        
                        # Create depth chart
                        fig = go.Figure()
                        
                        # Add bid trace
                        fig.add_trace(go.Scatter(
                            x=bid_df['price'],
                            y=bid_df['cumulative_size'],
                            name="Bids",
                            line=dict(color='green', width=2),
                            fill='tozeroy'
                        ))
                        
                        # Add ask trace
                        fig.add_trace(go.Scatter(
                            x=ask_df['price'],
                            y=ask_df['cumulative_size'],
                            name="Asks",
                            line=dict(color='red', width=2),
                            fill='tozeroy'
                        ))
                        
                        # Update layout
                        mid_price = (market_data.get('bestBid', 0) + market_data.get('bestAsk', 0)) / 2
                        spread_percent = market_data.get('spreadBps', 0) / 10000  # Convert bps to percentage
                        
                        # Calculate a reasonable x-axis range (±5% from mid price)
                        price_range = mid_price * 0.05
                        x_min = max(0, mid_price - price_range)
                        x_max = mid_price + price_range
                        
                        fig.update_layout(
                            title=f"Orderbook Depth Chart for {selected_symbol}",
                            xaxis_title="Price",
                            yaxis_title="Cumulative Size",
                            hovermode="x unified",
                            xaxis=dict(range=[x_min, x_max])
                        )
                        
                        # Add a vertical line at the mid price
                        fig.add_shape(
                            type="line",
                            x0=mid_price, y0=0,
                            x1=mid_price, y1=max(bid_df['cumulative_size'].max(), ask_df['cumulative_size'].max()) if not bid_df.empty and not ask_df.empty else 0,
                            line=dict(color="blue", width=1, dash="dash")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error visualizing orderbook: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
        
    if st.session_state.signals:
        st.subheader("Trading Signals")
        signals_df = pd.DataFrame([s.model_dump() for s in st.session_state.signals])
        signal_filter = st.multiselect("Filter by Signal", options=['LONG', 'SHORT', 'HOLD'], default=['LONG', 'SHORT'])
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

with tab2:
    st.header("Active Trades")
    if st.button("Update Trades"):
        updates = tester.update_trades()
        for update in updates:
            st.info(update)
    if tester.active_trades:
        active_df = pd.DataFrame([t.model_dump() for t in tester.active_trades.values()])
        markets_df = asyncio.run(fetch_all_markets())
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
            ohlcv_df = asyncio.run(fetch_hyperliquid_candles(selected_trade, interval=timeframe, limit=50))
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
                fig.add_hline(y=trade_data.entry_price, line_width=1, line_dash="dash", line_color="yellow", annotation_text="Entry")
                if trade_data.tp_price:
                    fig.add_hline(y=trade_data.tp_price, line_width=1, line_dash="dash", line_color="green", annotation_text="TP")
                if trade_data.sl_price:
                    fig.add_hline(y=trade_data.sl_price, line_width=1, line_dash="dash", line_color="red", annotation_text="SL")
                    fig.add_hline(y=trade_data.sl_price, line_width=1, line_dash="dash", line_color="red", annotation_text="SL")
                fig.update_layout(
                    title=f"{selected_trade} - {timeframe} Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active trades at the moment.")

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
                    title="P&L Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(stats)

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
                shutil.copy('trading_state.db', backup_file)
                st.success(f"Database backed up as {backup_file}")
            except Exception as e:
                st.error(f"Failed to backup database: {str(e)}")
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
                logging.error(f"Cache clear failed: {str(e)}")

with tab6:
    st.header("Debug Information")
    debug_enabled = st.checkbox(
        "Enable Debug Mode",
        value=st.session_state.get('debug_mode', False),
        key="debug_tab_toggle",
        on_change=lambda: st.session_state.update(debug_mode=not st.session_state.get('debug_mode', False))
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
                    book = info.l2_snapshot(f"{sample_symbol}/USDC:USDC")
                    if book and 'levels' in book and book['levels']:
                        st.success(f"✅ Successfully fetched orderbook for {sample_symbol}")
                        parsed_levels = parse_orderbook_levels(book['levels'])
                        bids = [level for level in parsed_levels if level['side'] == 'b']
                        asks = [level for level in parsed_levels if level['side'] == 'a']
                        st.write(f"Found {len(bids)} bids and {len(asks)} asks")
                        if bids:
                            st.write(f"Top bid: Price={bids[0]['price']}, Size={bids[0]['size']}")
                        if asks:
                            st.write(f"Top ask: Price={asks[0]['price']}, Size={asks[0]['size']}")
                    else:
                        st.error(f"❌ Failed to fetch valid orderbook for {sample_symbol}")
                except Exception as e:
                    st.error(f"❌ Orderbook test failed: {str(e)}")
                    logging.error(f"Orderbook test failed for {sample_symbol}: {str(e)}")
                try:
                    end_time = int(datetime.now().timestamp() * 1000)
                    start_time = end_time - (24 * 60 * 60 * 1000)
                    candles = info.candles_snapshot(sample_symbol, '1h', start_time, end_time)  # Use base symbol
                    if candles and isinstance(candles, list):
                        st.success(f"✅ Successfully fetched candles for {sample_symbol}")
                        st.write(f"Found {len(candles)} candles")
                    else:
                        st.error(f"❌ Failed to fetch valid candles for {sample_symbol}")
                except Exception as e:
                    st.error(f"❌ Candles test failed: {str(e)}")
                    logging.error(f"Candles test failed for {sample_symbol}: {str(e)}")
                try:
                    end_time = int(datetime.now().timestamp() * 1000)
                    start_time = end_time - (24 * 60 * 60 * 1000)
                    funding = info.funding_history(sample_symbol, start_time)
                    if funding and isinstance(funding, list):
                        st.success(f"✅ Successfully fetched funding history for {sample_symbol}")
                        st.write(f"Found {len(funding)} funding records")
                    else:
                        st.error(f"❌ Failed to fetch valid funding history for {sample_symbol}")
                except Exception as e:
                    st.error(f"❌ Funding history test failed: {str(e)}")
                    logging.error(f"Funding history test failed for {sample_symbol}: {str(e)}")
        except Exception as e:
            st.error(f"❌ Hyperliquid connection test failed: {str(e)}")
            logging.error(f"Hyperliquid connection test failed: {str(e)}")
    
    if st.session_state.debug_info:
        st.subheader("Last Scan Debug Info")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total Markets", st.session_state.debug_info.get('total_markets', 'N/A'))
        with cols[1]:
            st.metric("Markets Processed", st.session_state.debug_info.get('markets_processed', 'N/A'))
        with cols[2]:
            st.metric("Markets Skipped", st.session_state.debug_info.get('markets_skipped', 'N/A'))
        
        if 'skipped_samples' in st.session_state.debug_info and st.session_state.debug_info['skipped_samples']:
            st.subheader("Sample Skipped Markets")
            st.json(st.session_state.debug_info['skipped_samples'])
        
        errors = {k: v for k, v in st.session_state.debug_info.items() if 'error' in k.lower()}
        if errors:
            with st.expander("Error Details"):
                for k, v in errors.items():
                    st.error(f"{k}: {v}")
    
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
                logging.error(f"Clear cache failed: {str(e)}")
    else:
        st.write("Cache database not found.")
    
    st.subheader("Rate Limiter Settings")
    current_rate = st.session_state.get('api_rate', 4.0)
    rate = st.slider("API calls per second", 1.0, 10.0, current_rate, 0.5)
    if rate != current_rate:
        st.session_state.api_rate = rate
        st.success(f"Rate limiter updated to {rate} calls per second")
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (WAT)")
