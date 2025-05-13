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
import logging
from hyperliquid.info import Info
from hyperliquid.utils import constants
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import math
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Set up logging
logging.basicConfig(filename='hyperliquid_screener.log', level=logging.DEBUG)

# Set page config
st.set_page_config(
    page_title="Hyperliquid Futures Screener",
    page_icon="📊",
    layout="wide",
)

# Title and description
st.title("📈 Hyperliquid Futures Market Screener")
st.markdown("Track and analyze cryptocurrency futures markets on Hyperliquid")

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

init_session_state()

# ======================== PYDANTIC MODELS ========================
class MarketLevel(BaseModel):
    side: str  # 'b' for bid, 'a' for ask
    price: float
    size: float

class OrderBook(BaseModel):
    levels: List[MarketLevel]
    
    @validator('levels', allow_reuse=True)
    def validate_levels(cls, v):
        if not v:
            raise ValueError("Order book levels cannot be empty")
        return v

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
MIN_LIQUIDITY = st.session_state.MIN_LIQUIDITY
FUNDING_THRESHOLD = 60
BATCH_SIZE = 50

# Initialize Hyperliquid Info client
@st.cache_resource
def init_hyperliquid():
    return Info(skip_ws=True)

info = init_hyperliquid()

# ======================== UTILITY CLASSES ========================
class RateLimiter:
    def __init__(self, calls_per_second=4):
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

rate_limiter = RateLimiter(calls_per_second=4)
volume_fetcher = VolumeFetcher(info, rate_limit_calls_per_second=4)

class VolumeFetcher:
    def __init__(self, info_client: Info, rate_limit_calls_per_second: int = 4):
        self.info = info_client
        self.rate_limiter = RateLimiter(calls_per_second=rate_limit_calls_per_second)

    def get_volume_24h(self, symbol: str) -> Optional[float]:
        """Fetch the 24-hour trading volume for a given symbol using candlestick data, with order book fallback.

        Args:
            symbol (str): The trading pair (e.g., "BTC" or "BTC/USDC:USDC").

        Returns:
            Optional[float]: The estimated 24-hour volume in USD, or None if fetching fails.
        """
        try:
            # Prepare timestamps for the last 24 hours
            end_time = int(datetime.utcnow().timestamp() * 1000)  # Current time in UTC milliseconds
            start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago in UTC milliseconds

            # Step 1: Try fetching volume from candlestick data
            candles = self._fetch_candles_snapshot(symbol, "1h", start_time, end_time)
            if candles and isinstance(candles, list) and len(candles) > 0:
                total_volume = sum(safe_float(candle.get("v", 0)) for candle in candles)
                logging.info(f"Volume from candles for {symbol}: ${total_volume:.2f}")
                return max(total_volume, 10000)  # Minimum fallback volume

            # Step 2: Fallback to order book estimation if candles fail
            logging.warning(f"Candlestick data unavailable for {symbol}, falling back to order book")
            return self._estimate_volume_from_orderbook(symbol)

        except Exception as e:
            logging.error(f"Failed to get volume for {symbol}: {str(e)}")
            return 10000  # Default fallback volume for major coins or errors

    def _fetch_candles_snapshot(self, name: str, interval: str, start_time: int, end_time: int) -> Any:
        """Helper method to fetch candlestick data with rate limiting and error handling."""
        self.rate_limiter.wait()
        try:
            req = {
                "coin": self.info.name_to_coin.get(name, name),  # Handle symbol conversion
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time
            }
            response = self.info.post("/info", {"type": "candleSnapshot", "req": req})
            if response and isinstance(response, list):
                return response
            logging.warning(f"Invalid candles response for {name}: {str(response)[:200]}")
            return None
        except Exception as e:
            logging.error(f"Candles snapshot error for {name}: {str(e)}")
            return None

    def _estimate_volume_from_orderbook(self, symbol: str) -> float:
        """Estimate volume based on order book liquidity."""
        self.rate_limiter.wait()
        try:
            book_response = self.info.l2_snapshot(symbol)
            if not book_response or "levels" not in book_response:
                logging.warning(f"Invalid order book for {symbol}")
                return 10000

            parsed_levels = self._parse_orderbook_levels(book_response["levels"])
            bids = [level for level in parsed_levels if level["side"] == "b"]
            asks = [level for level in parsed_levels if level["side"] == "a"]

            if not bids or not asks:
                logging.warning(f"No bids or asks in order book for {symbol}")
                return 10000

            top_bids = sorted(bids, key=lambda x: -x["price"])[:20]
            top_asks = sorted(asks, key=lambda x: x["price"])[:20]
            bid_liquidity = sum(level["size"] for level in top_bids)
            ask_liquidity = sum(level["size"] for level in top_asks)
            mid_price = (top_bids[0]["price"] + top_asks[0]["price"]) / 2

            estimated_volume = (bid_liquidity + ask_liquidity) * mid_price
            logging.info(f"Estimated volume from order book for {symbol}: ${estimated_volume:.2f}")
            return max(estimated_volume, 10000)
        except Exception as e:
            logging.error(f"Order book estimation failed for {symbol}: {str(e)}")
            return 10000

    def _parse_orderbook_levels(self, raw_levels: Any) -> list:
        """Parse raw order book levels into a standardized format."""
        parsed = []
        for level_group in raw_levels:
            for level in level_group:
                if isinstance(level, dict) and "px" in level and "sz" in level:
                    parsed.append({
                        "side": "b" if level.get("n", 0) > 0 else "a",
                        "price": safe_float(level["px"]),
                        "size": safe_float(level["sz"])
                    })
        return parsed

class MicroPrice:
    def __init__(self, alpha: float = 2.0, volatility_factor: float = 1.0):
        self.alpha = alpha
        self.volatility_factor = volatility_factor
    
    def calculate(self, best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> float:
        if not all(isinstance(x, (int, float)) for x in [best_bid, best_ask, bid_size, ask_size]):
            raise ValueError("All inputs must be numeric")
        if best_bid < 0 or best_ask < 0 or bid_size < 0 or ask_size < 0:
            raise ValueError("Inputs must be non-negative")
        if best_ask < best_bid:
            best_bid, best_ask = best_ask, best_bid
        if bid_size == 0 and ask_size == 0:
            raise ValueError("At least one of bid_size or ask_size must be positive")
        mid_price = (best_bid + best_ask) / 2.0
        if bid_size == 0 or ask_size == 0:
            return mid_price
        spread = best_ask - best_bid
        imbalance = bid_size / (bid_size + ask_size)
        effective_alpha = self.alpha * self.volatility_factor
        delta = imbalance - 0.5
        adjustment = spread * math.tanh(effective_alpha * delta)
        micro_price = mid_price + adjustment
        return max(best_bid, min(best_ask, micro_price))

# ======================== PARSERS FOR RAW API DATA ========================
def parse_orderbook_levels(raw_levels):
    parsed = []
    for level_group in raw_levels:
        for level in level_group:
            if isinstance(level, dict) and 'px' in level and 'sz' in level:
                parsed.append({
                    'side': 'b' if level.get('n', 0) > 0 else 'a',
                    'price': safe_float(level['px']),
                    'size': safe_float(level['sz'])
                })
        else:
            debug_info = getattr(st.session_state, 'debug_info', {})
            debug_info[f'parse_error_orderbook'] = f"Unexpected level format: {level_group}"
            st.session_state.debug_info = debug_info
    return parsed

def parse_candles(raw_candles):
    parsed = []
    for candle in raw_candles:
        if isinstance(candle, dict):
            parsed.append({
                'timestamp': candle.get('t', 0),
                'time_close': candle.get('T', 0),
                'symbol': candle.get('s', ''),
                'interval': candle.get('i', ''),
                'open': safe_float(candle.get('o'), 0.0),
                'close': safe_float(candle.get('c'), 0.0),
                'high': safe_float(candle.get('h'), 0.0),
                'low': safe_float(candle.get('l'), 0.0),
                'volume': safe_float(candle.get('v'), 0.0),
                'num': candle.get('n', 0)
            })
        else:
            debug_info = getattr(st.session_state, 'debug_info', {})
            debug_info[f'parse_error_candle'] = f"Unexpected candle format: {candle}"
            st.session_state.debug_info = debug_info
    return parsed

# ======================== HELPER FUNCTIONS ========================
def api_request_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            response = func(*args, **kwargs)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"API request failed after {max_retries} retries: {str(e)}")
                raise
            time.sleep(1 * (2 ** attempt))
    return None

async def async_api_request(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: api_request_with_retry(func, *args, **kwargs))
    return result

def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def validate_symbol(symbol, meta_universe):
    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
    return any(coin.name == base_symbol for coin in meta_universe)

# ======================== API CONNECTION VALIDATION ========================
def validate_api_responses():
    try:
        meta_response = api_request_with_retry(info.meta)
        if not meta_response or 'universe' not in meta_response:
            debug_info = getattr(st.session_state, 'debug_info', {})
            debug_info['meta_error'] = 'Invalid or empty meta response'
            st.session_state.debug_info = debug_info
            logging.error("Failed to fetch market metadata")
            return False
        all_prices = api_request_with_retry(info.all_mids)
        if not all_prices or not isinstance(all_prices, dict):
            debug_info = getattr(st.session_state, 'debug_info', {})
            debug_info['mids_error'] = 'Invalid or empty mids response'
            st.session_state.debug_info = debug_info
            logging.error("Failed to fetch mid prices")
            return False
        return True
    except Exception as e:
        debug_info = getattr(st.session_state, 'debug_info', {})
        debug_info['api_validation_error'] = str(e)
        st.session_state.debug_info = debug_info
        logging.error(f"API validation failed: {str(e)}")
        return False

# ======================== FALLBACK LOGIC FOR MAJOR COINS ========================
def apply_fallback_for_major_coins():
    st.info("Using fallback data for major coins.")
    fallback_data = [
        MarketInfo(symbol='BTC', markPrice=83000, lastPrice=83000, fundingRate=5, volume24h=500000, change24h=1.2),
        MarketInfo(symbol='ETH', markPrice=3000, lastPrice=3000, fundingRate=10, volume24h=300000, change24h=0.8),
        MarketInfo(symbol='SOL', markPrice=150, lastPrice=150, fundingRate=15, volume24h=100000, change24h=2.5)
    ]
    return pd.DataFrame([m.dict() for m in fallback_data])

# ======================== FETCH ALL MARKETS ========================
async def fetch_all_markets():
    try:
        init_session_state()
        logging.info("Starting async market fetch")
        debug_info = st.session_state.debug_info

        if not validate_api_responses():
            st.error("API validation failed. Check debug info.")
            return pd.DataFrame()

        meta_response = await async_api_request(info.meta)
        meta = MarketUniverse(**meta_response)
        debug_info['total_markets'] = len(meta.universe)
        if not meta.universe:
            debug_info['meta_empty'] = 'No markets in universe'
            st.warning("No markets found in metadata")
            return pd.DataFrame()

        all_prices = await async_api_request(info.all_mids)
        meta_lookup = {coin.name: coin.dict() for coin in meta.universe}

        coins = meta.universe
        total_coins = len(coins)
        batches = [coins[i:i + BATCH_SIZE] for i in range(0, total_coins, BATCH_SIZE)]
        debug_info['total_batches'] = len(batches)
        st.session_state.batch_results = []

        market_data = []
        skipped_markets = []
        batch_results = []

        for batch_idx, batch in enumerate(batches):
            batch_data = []
            batch_skipped = []
            batch_start = batch_idx * BATCH_SIZE + 1
            batch_end = min((batch_idx + 1) * BATCH_SIZE, total_coins)

            progress = st.progress(0)
            status = st.empty()
            status.text(f"Processing batch {batch_idx + 1}/{len(batches)} ({batch_start}-{batch_end}/{total_coins})")

            async def process_coin(coin, idx_in_batch):
                symbol = coin.name
                perp_symbol = f"{symbol}/USDC:USDC"
                debug_info = getattr(st.session_state, 'debug_info', {})
                try:
                    if not validate_symbol(perp_symbol, meta.universe):
                        batch_skipped.append({
                            'symbol': symbol,
                            'reason': 'Invalid or unsupported symbol',
                            'min_required': MIN_LIQUIDITY
                        })
                        debug_info[f'invalid_symbol_{symbol}'] = f'Unsupported symbol: {perp_symbol}'
                        st.session_state.debug_info = debug_info
                        return None

                    price = safe_float(all_prices.get(symbol))
                    if not price:
                        batch_skipped.append({
                            'symbol': symbol,
                            'reason': 'No valid price data',
                            'min_required': MIN_LIQUIDITY
                        })
                        debug_info[f'skip_no_price_{symbol}'] = 'Missing price'
                        st.session_state.debug_info = debug_info
                        return None

                    # Fetch volume using VolumeFetcher
                    volume_24h = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: volume_fetcher.get_volume_24h(perp_symbol)
                    )
                    debug_info[f'volume_fetch_{symbol}'] = f'Volume: ${volume_24h:.2f}'

                    if volume_24h < MIN_LIQUIDITY:
                        if symbol in ["BTC", "ETH", "SOL"]:
                            volume_24h = max(volume_24h, MIN_LIQUIDITY * 1.5)
                            debug_info[f'volume_fallback_{symbol}'] = f'Used fallback volume: {volume_24h}'
                        else:
                            batch_skipped.append({
                                'symbol': symbol,
                                'volume24h': volume_24h,
                                'min_required': MIN_LIQUIDITY
                            })
                            debug_info[f'skip_low_volume_{symbol}'] = f'Volume {volume_24h} < {MIN_LIQUIDITY}'
                            st.session_state.debug_info = debug_info
                            return None

                    # Fetch funding rate
                    funding_rate = 10.0 if symbol in ["BTC", "ETH", "SOL"] else 5.0
                    try:
                        end_time = int(datetime.now().timestamp() * 1000)
                        start_time = end_time - (24 * 60 * 60 * 1000)
                        funding_response = await async_api_request(info.funding_history, symbol, start_time)
                        debug_info[f'funding_response_{symbol}'] = str(funding_response)
                        if funding_response and isinstance(funding_response, list) and len(funding_response) > 0:
                            for entry in funding_response[:5]:
                                try:
                                    funding_data = FundingRate(**entry)
                                    raw_rate = safe_float(funding_data.fundingRate)
                                    debug_info[f'funding_raw_{symbol}'] = f'Raw rate: {raw_rate}, Time: {funding_data.time}'
                                    funding_rate = raw_rate * 10000 * 3
                                    debug_info[f'funding_success_{symbol}'] = f'Funding rate: {funding_rate:.2f} bps'
                                    break
                                except Exception as e:
                                    debug_info[f'funding_entry_error_{symbol}'] = f'Entry error: {str(e)}'
                            else:
                                debug_info[f'funding_empty_{symbol}'] = 'No valid funding entries'
                        else:
                            debug_info[f'funding_empty_{symbol}'] = 'Empty or invalid funding response'
                    except Exception as e:
                        debug_info[f'funding_error_{symbol}'] = str(e)
                        logging.error(f"Funding rate fetch failed for {symbol}: {str(e)}")

                    # Calculate 24-hour price change
                    change_24h = 1.0 if symbol in ["BTC", "ETH", "SOL"] else 0.5
                    try:
                        end_time = int(datetime.now().timestamp() * 1000)
                        start_time = end_time - (48 * 60 * 60 * 1000)
                        for timeframe in ['1h', '4h', '1d']:
                            candles_response = await async_api_request(info.candles_snapshot, perp_symbol, timeframe, start_time, end_time)
                            debug_info[f'candles_response_{symbol}_{timeframe}'] = str(candles_response)
                            if candles_response and isinstance(candles_response, list) and len(candles_response) >= 2:
                                parsed_candles = parse_candles(candles_response)
                                candles = [CandleData(**candle) for candle in parsed_candles]
                                valid_candles = [c for c in candles if end_time - 24*60*60*1000 <= c.timestamp <= end_time]
                                if len(valid_candles) >= 2:
                                    oldest_price = safe_float(valid_candles[0].open)
                                    latest_price = safe_float(valid_candles[-1].close)
                                    debug_info[f'candles_raw_{symbol}'] = f'Oldest: {oldest_price}, Latest: {latest_price}, Count: {len(valid_candles)}'
                                    if oldest_price > 0:
                                        change_24h = ((latest_price - oldest_price) / oldest_price) * 100
                                        debug_info[f'change_success_{symbol}'] = f'Change: {change_24h:.2f}%, Timeframe: {timeframe}'
                                        break
                                else:
                                    debug_info[f'candles_insufficient_{symbol}'] = f'Only {len(valid_candles)} valid candles in {timeframe}'
                            else:
                                debug_info[f'candles_empty_{symbol}'] = f'Empty or invalid candles response in {timeframe}'
                    except Exception as e:
                        debug_info[f'candles_error_{symbol}'] = str(e)
                        logging.error(f"Candles fetch failed for {symbol}: {str(e)}")

                    st.session_state.debug_info = debug_info
                    return MarketInfo(
                        symbol=symbol,
                        markPrice=price,
                        lastPrice=price,
                        fundingRate=funding_rate,
                        volume24h=volume_24h,
                        change24h=change_24h,
                    )
                except Exception as e:
                    debug_info[f'error_{symbol}'] = str(e)
                    st.session_state.debug_info = debug_info
                    logging.error(f"Processing failed for {symbol}: {str(e)}")
                    return None

            tasks = [process_coin(coin, i) for i, coin in enumerate(batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if result is not None:
                    batch_data.append(result)

            market_data.extend(batch_data)
            skipped_markets.extend(batch_skipped)

            progress.progress(1.0)
            status.empty()

            batch_df = pd.DataFrame([m.dict() for m in batch_data]) if batch_data else pd.DataFrame()
            batch_results.append({
                'batch_number': batch_idx + 1,
                'markets_processed': len(batch_data),
                'markets_skipped': len(batch_skipped),
                'data': batch_df
            })
            st.session_state.batch_results = batch_results

            st.subheader(f"Batch {batch_idx + 1} Results")
            if not batch_df.empty:
                st.write(f"Processed {len(batch_df)} markets in batch {batch_idx + 1}")
                st.dataframe(
                    batch_df,
                    use_container_width=True,
                    column_config={
                        'symbol': st.column_config.TextColumn("Symbol"),
                        'markPrice': st.column_config.NumberColumn("Mark Price", format="%.4f"),
                        'volume24h': st.column_config.NumberColumn("24h Volume", format="$%.2f"),
                        'fundingRate': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                        'change24h': st.column_config.NumberColumn("24h Change", format="%.2f%%")
                    }
                )
            else:
                st.warning(f"No markets met liquidity threshold in batch {batch_idx + 1}")

        df = pd.DataFrame([m.dict() for m in market_data])
        if df.empty:
            debug_info['empty_result'] = True
            debug_info['min_liquidity'] = MIN_LIQUIDITY
            st.warning(f"No markets met the minimum liquidity threshold of ${MIN_LIQUIDITY:,}.")
            logging.warning(f"No markets met liquidity threshold of ${MIN_LIQUIDITY:,}")
            if st.session_state.get('force_include_major', False):
                return apply_fallback_for_major_coins()

        debug_info['markets_processed'] = len(market_data)
        debug_info['markets_skipped'] = len(skipped_markets)
        debug_info['skipped_samples'] = skipped_markets[:5]
        st.session_state.debug_info = debug_info

        # Sort by volume24h descending as the default
        sorted_df = df.sort_values('volume24h', ascending=False)
        debug_info['sorted_columns'] = f"Sorted by volume24h: {sorted_df[['symbol', 'volume24h']].head().to_dict()}"
        return sorted_df

    except Exception as e:
        debug_info = getattr(st.session_state, 'debug_info', {})
        debug_info['critical_error'] = str(e)
        st.session_state.debug_info = debug_info
        st.error(f"Critical error fetching markets: {str(e)}")
        logging.error(f"Critical error fetching markets: {str(e)}")
        return pd.DataFrame()

# ======================== DATA FETCHING ========================
async def fetch_hyperliquid_candles(symbol, interval="1h", limit=50):
    debug_info = getattr(st.session_state, 'debug_info', {})
    try:
        timeframe_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
        timeframe = timeframe_map.get(interval, '1h')
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (48 * 60 * 60 * 1000)
        ohlcv = await async_api_request(info.candles_snapshot, symbol, timeframe, start_time, end_time)
        debug_info[f'candles_response_{symbol}_{timeframe}'] = str(ohlcv)
        if not ohlcv or len(ohlcv) == 0:
            debug_info[f'candles_empty_{symbol}'] = 'No candle data returned'
            logging.warning(f"No candle data for {symbol}")
            st.session_state.debug_info = debug_info
            return None
        parsed_ohlcv = parse_candles(ohlcv)
        df = pd.DataFrame(parsed_ohlcv, columns=['timestamp', 'time_close', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'num'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        st.session_state.debug_info = debug_info
        return df
    except Exception as e:
        debug_info[f'candles_error_{symbol}'] = str(e)
        st.session_state.debug_info = debug_info
        st.error(f"Error fetching candles for {symbol}: {str(e)}")
        logging.error(f"Error fetching candles for {symbol}: {str(e)}")
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
            logging.error(f"Could not load state: {str(e)}")
    
    def save_state(self):
        try:
            save_state_to_db(
                {k: v.dict() for k, v in self.active_trades.items()},
                [t.dict() for t in self.completed_trades]
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
        loop = asyncio.get_event_loop()
        markets_df = loop.run_until_complete(fetch_all_markets())
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
    micro_price_calc = MicroPrice(alpha=2.0, volatility_factor=1.0)
    progress_bar = st.progress(0)
    status_text = st.empty()
    MICRO_PRICE_THRESHOLD = 0.001

    async def process_market(index, market):
        symbol = market['symbol']
        perp_symbol = f"{symbol}/USDC:USDC"
        status_text.text(f"Analyzing {symbol}... ({index + 1}/{total_markets})")
        debug_info = getattr(st.session_state, 'debug_info', {})
        try:
            book_response = await async_api_request(info.l2_snapshot, perp_symbol)
            if not book_response or 'levels' not in book_response:
                debug_info[f'no_orderbook_{symbol}'] = 'Empty or invalid order book'
                st.session_state.debug_info = debug_info
                return None
            parsed_levels = parse_orderbook_levels(book_response['levels'])
            book = OrderBook(levels=parsed_levels)
            bids = [level for level in book.levels if level.side == 'b']
            asks = [level for level in book.levels if level.side == 'a']
            if not bids or not asks:
                debug_info[f'no_bids_asks_{symbol}'] = 'No bids or asks in order book'
                st.session_state.debug_info = debug_info
                return None
            best_bid = max(bid.price for bid in bids)
            best_ask = min(ask.price for ask in asks)
            bid_size = sum(bid.size for bid in bids if bid.price == best_bid)
            ask_size = sum(ask.size for ask in asks if ask.price == best_ask)
            micro_price = micro_price_calc.calculate(best_bid, best_ask, bid_size, ask_size)
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            df = await fetch_hyperliquid_candles(perp_symbol, interval='1h', limit=48)
            if df is None or len(df) < 6:
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
            price_threshold = max(MICRO_PRICE_THRESHOLD, volatility_factor * 0.5)
            max_spread_bps = 50
            tp_distance = max(volatility_factor * 5, 0.02)
            sl_distance = max(volatility_factor * 3, 0.015)
            if spread_bps <= max_spread_bps and vol_surge >= 1.5 and vol_consistent:
                if micro_price_deviation > price_threshold and funding_rate < -FUNDING_THRESHOLD:
                    signal = "LONG"
                    reason = f"Micro-price above mid: {micro_price_deviation:.4f}, Vol surge {vol_surge:.2f}x, Funding {funding_rate:.2f} bps"
                    tp = str(round(current_price * (1 + tp_distance), 4))
                    sl = str(round(current_price * (1 - sl_distance), 4))
                elif micro_price_deviation < -price_threshold and funding_rate > FUNDING_THRESHOLD:
                    signal = "SHORT"
                    reason = f"Micro-price below mid: {micro_price_deviation:.4f}, Vol surge {vol_surge:.2f}x, Funding {funding_rate:.2f} bps"
                    tp = str(round(current_price * (1 - tp_distance), 4))
                    sl = str(round(current_price * (1 + sl_distance), 4))
            return Signal(
                Symbol=symbol,
                Price=current_price,
                MarkPrice=market['markPrice'],
                Signal=signal,
                Volume24h=market['volume24h'],
                FundingRate=funding_rate,
                VolSurge=vol_surge,
                Change24h=market['change24h'],
                Reason=reason,
                TP=tp,
                SL=sl
            )
        except Exception as e:
            debug_info[f'error_{symbol}'] = str(e)
            st.session_state.debug_info = debug_info
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    tasks = [process_market(i, market) for i, (_, market) in enumerate(top_markets.iterrows())]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if result is not None:
            signals.append(result)

    progress_bar.progress(1.0)
    status_text.empty()
    debug_info['signals_generated'] = len(signals)
    st.session_state.debug_info = debug_info
    logging.info(f"Generated {len(signals)} signals")
    return signals

# ======================== SCAN MARKETS ========================
def scan_markets():
    init_session_state()
    st.session_state.scanned_markets = pd.DataFrame()
    st.session_state.signals = []
    
    try:
        with st.spinner("Fetching market data from Hyperliquid..."):
            loop = asyncio.get_event_loop()
            markets_df = loop.run_until_complete(fetch_all_markets())
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
                loop = asyncio.get_event_loop()
                signals = loop.run_until_complete(generate_signals(markets_df))
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
            value=4
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Market Scanner", "Active Trades", "Completed Trades", "Performance", "Database", "Debug"])

with tab1:
    st.info(f"Current Minimum Liquidity: ${MIN_LIQUIDITY:,} USD")
    if st.button("Scan Markets", use_container_width=True):
        scan_markets()
    if not st.session_state.scanned_markets.empty:
        st.subheader("All Markets")
        # Rename columns for better display and ensure consistency
        display_df = st.session_state.scanned_markets.rename(columns={
            'symbol': 'Symbol',
            'markPrice': 'Mark Price',
            'volume24h': 'Volume (24h)',
            'fundingRate': 'Funding Rate (bps)',
            'change24h': 'Change (24h)'
        })
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                'Symbol': st.column_config.TextColumn("Symbol"),
                'Mark Price': st.column_config.NumberColumn("Mark Price", format="%.4f"),
                'Volume (24h)': st.column_config.NumberColumn("Volume (24h)", format="$%.2f"),
                'Funding Rate (bps)': st.column_config.NumberColumn("Funding Rate (bps)", format="%.2f"),
                'Change (24h)': st.column_config.NumberColumn("Change (24h)", format="%.2f%%")
            }
        )
    if st.session_state.signals:
        st.subheader("Trading Signals")
        signals_df = pd.DataFrame([s.dict() for s in st.session_state.signals])
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
        active_df = pd.DataFrame([t.dict() for t in tester.active_trades.values()])
        loop = asyncio.get_event_loop()
        markets_df = loop.run_until_complete(fetch_all_markets())
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
            loop = asyncio.get_event_loop()
            ohlcv_df = loop.run_until_complete(fetch_hyperliquid_candles(selected_trade, interval=timeframe, limit=50))
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
                    candles = info.candles_snapshot(f"{sample_symbol}/USDC:USDC", '1h', start_time, end_time)
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
    current_rate = st.session_state.get('api_rate', rate_limiter.calls_per_second)
    rate = st.slider("API calls per second", 1, 10, int(current_rate), 1)
    if rate != current_rate:
        rate_limiter.calls_per_second = rate
        st.session_state.api_rate = rate
        st.success(f"Rate limiter updated to {rate} calls per second")
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (WAT)")
