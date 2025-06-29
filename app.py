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
import random

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

# Set up logging
logging.basicConfig(filename='hyperliquid_screener.log', level=logging.INFO)

# Set page config for mobile and desktop compatibility
st.set_page_config(
    page_title="Hyperliquid Futures Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and description (use st.write instead of st.markdown to avoid regex issues)
st.title("Hyperliquid Futures Screener")
st.write("Track and analyze cryptocurrency futures markets on Hyperliquid")

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

# Pydantic Models
class MarketLevel(BaseModel):
    side: str
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

# Rate Limiter
class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.backoff_factor = 2.0
    
    def wait(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
    
    def backoff(self, attempt: int) -> float:
        return (self.backoff_factor ** attempt) + random.uniform(0.1, 0.5)

# Volume Fetcher
class VolumeFetcher:
    def __init__(self, info_client: Info, rate_limit_calls_per_second: int = 2):
        self.info = info_client
        self.rate_limiter = RateLimiter(calls_per_second=rate_limit_calls_per_second)

    def get_volume_24h(self, symbol: str) -> Optional[float]:
        try:
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = end_time - (24 * 60 * 60 * 1000)
            candles = self._fetch_candles_snapshot(symbol, "1h", start_time, end_time)
            if candles and isinstance(candles, list) and len(candles) > 0:
                total_volume = sum(safe_float(candle.get("v", 0)) for candle in candles)
                logging.info(f"Volume for {symbol}: ${total_volume:.2f}")
                return max(total_volume, 10000)
            return self._estimate_volume_from_orderbook(symbol)
        except Exception as e:
            logging.error(f"Failed to get volume for {symbol}: {str(e)}")
            return 10000

    def _fetch_candles_snapshot(self, name: str, interval: str, start_time: int, end_time: int) -> Any:
        self.rate_limiter.wait()
        base_symbol = name.split('/')[0] if '/' in name else name
        symbol_formats = [base_symbol, f"{base_symbol}/USDC:USDC"]
        for sym_format in symbol_formats:
            try:
                req = {
                    "coin": sym_format,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
                response = self.info.post("/info", {"type": "candleSnapshot", "req": req})
                if response and isinstance(response, list) and len(response) > 0:
                    return response
            except Exception as e:
                logging.warning(f"Candles fetch failed for {sym_format}: {str(e)}")
        return None

    def _estimate_volume_from_orderbook(self, symbol: str) -> float:
        self.rate_limiter.wait()
        try:
            book_response = self.info.l2_snapshot(symbol)
            if not book_response or "levels" not in book_response:
                return 10000
            parsed_levels = self._parse_orderbook_levels(book_response["levels"])
            bids = [level for level in parsed_levels if level["side"] == "b"]
            asks = [level for level in parsed_levels if level["side"] == "a"]
            if not bids or not asks:
                return 10000
            top_bids = sorted(bids, key=lambda x: -x["price"])[:5]
            top_asks = sorted(asks, key=lambda x: x["price"])[:5]
            bid_liquidity = sum(level["size"] for level in top_bids)
            ask_liquidity = sum(level["size"] for level in top_asks)
            mid_price = (top_bids[0]["price"] + top_asks[0]["price"]) / 2
            estimated_volume = (bid_liquidity + ask_liquidity) * mid_price
            return max(estimated_volume, 10000)
        except Exception as e:
            logging.error(f"Order book estimation failed for {symbol}: {str(e)}")
            return 10000

    def _parse_orderbook_levels(self, raw_levels: Any) -> list:
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

# MicroPrice Calculator
class MicroPrice:
    def __init__(self, alpha: float = 2.0, volatility_factor: float = 1.0):
        self.alpha = alpha
        self.volatility_factor = volatility_factor
    
    def calculate(self, best_bid: float, best_ask: float, bid_size: float, ask_size: float) -> float:
        try:
            if not all(isinstance(x, (int, float)) for x in [best_bid, best_ask, bid_size, ask_size]):
                raise ValueError("All inputs must be numeric")
            if best_bid < 0 or best_ask < 0 or bid_size < 0 or ask_size < 0:
                raise ValueError("Inputs must be non-negative")
            if best_ask < best_bid:
                best_bid, best_ask = best_ask, best_bid
            if bid_size == 0 and ask_size == 0:
                return (best_bid + best_ask) / 2.0
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
        except Exception as e:
            logging.error(f"MicroPrice error: {str(e)}")
            return (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0

# Configuration
BASE_VOL = 0.35
MIN_LIQUIDITY = st.session_state.MIN_LIQUIDITY
FUNDING_THRESHOLD = 60
BATCH_SIZE = 10

# Initialize Hyperliquid Client
@st.cache_resource
def init_hyperliquid():
    return Info(skip_ws=True)

info = init_hyperliquid()
rate_limiter = RateLimiter(calls_per_second=2)
volume_fetcher = VolumeFetcher(info, rate_limit_calls_per_second=2)

# Helper Functions
def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

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
    return parsed

@st.cache_data(ttl=300)
def api_request_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            response = func(*args, **kwargs)
            return response
        except Exception as e:
            is_rate_limit_error = str(e).find('429') >= 0 or str(e).lower().find('rate limit') >= 0
            if attempt == max_retries - 1:
                logging.error(f"API request failed after {max_retries} retries: {str(e)}")
                raise
            backoff = rate_limiter.backoff(attempt)
            if is_rate_limit_error:
                backoff *= 2
            time.sleep(backoff)
    return None

async def async_api_request(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: api_request_with_retry(func, *args, **kwargs))
    return result

def validate_symbol(symbol, meta_universe):
    base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
    return any(coin.name == base_symbol for coin in meta_universe)

def validate_api_responses():
    try:
        meta_response = api_request_with_retry(info.meta)
        if not meta_response or 'universe' not in meta_response:
            return False
        all_prices = api_request_with_retry(info.all_mids)
        if not all_prices or not isinstance(all_prices, dict):
            return False
        return True
    except Exception as e:
        logging.error(f"API validation failed: {str(e)}")
        return False

def apply_fallback_for_major_coins():
    fallback_data = [
        MarketInfo(symbol='BTC', markPrice=83000, lastPrice=83000, fundingRate=5, volume24h=500000, change24h=1.2),
        MarketInfo(symbol='ETH', markPrice=3000, lastPrice=3000, fundingRate=10, volume24h=300000, change24h=0.8),
        MarketInfo(symbol='SOL', markPrice=150, lastPrice=150, fundingRate=15, volume24h=100000, change24h=2.5)
    ]
    return pd.DataFrame([m.dict() for m in fallback_data])

# Fetch All Markets
async def fetch_all_markets():
    try:
        init_session_state()
        if not validate_api_responses():
            st.write("API validation failed. Using fallback data.")
            return apply_fallback_for_major_coins()

        meta_response = await async_api_request(info.meta)
        meta = MarketUniverse(**meta_response)
        if not meta.universe:
            return apply_fallback_for_major_coins()

        all_prices = await async_api_request(info.all_mids)
        coins = meta.universe
        total_coins = len(coins)
        batches = [coins[i:i + BATCH_SIZE] for i in range(0, total_coins, BATCH_SIZE)]
        market_data = []
        skipped_markets = []

        for batch_idx, batch in enumerate(batches):
            batch_data = []
            batch_skipped = []
            progress = st.progress(0)
            status = st.empty()
            status.text(f"Processing batch {batch_idx + 1}/{len(batches)}")

            async def process_coin(coin):
                symbol = coin.name
                perp_symbol = f"{symbol}/USDC:USDC"
                try:
                    if not validate_symbol(perp_symbol, meta.universe):
                        batch_skipped.append({'symbol': symbol, 'reason': 'Invalid symbol'})
                        return None
                    price = safe_float(all_prices.get(symbol))
                    if not price:
                        batch_skipped.append({'symbol': symbol, 'reason': 'No price data'})
                        return None
                    volume_24h = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: volume_fetcher.get_volume_24h(symbol)
                    )
                    if volume_24h < MIN_LIQUIDITY and symbol not in ["BTC", "ETH", "SOL"]:
                        batch_skipped.append({'symbol': symbol, 'volume24h': volume_24h})
                        return None
                    elif symbol in ["BTC", "ETH", "SOL"]:
                        volume_24h = max(volume_24h, MIN_LIQUIDITY * 1.5)

                    funding_rate = 10.0 if symbol in ["BTC", "ETH", "SOL"] else 5.0
                    try:
                        end_time = int(datetime.now().timestamp() * 1000)
                        start_time = end_time - (24 * 60 * 60 * 1000)
                        funding_response = await async_api_request(info.funding_history, symbol, start_time)
                        if funding_response and isinstance(funding_response, list) and len(funding_response) > 0:
                            funding_rate = safe_float(funding_response[0].get('fundingRate', 0)) * 10000 * 3
                    except Exception as e:
                        logging.error(f"Funding rate fetch failed for {symbol}: {str(e)}")

                    change_24h = 1.0 if symbol in ["BTC", "ETH", "SOL"] else 0.5
                    try:
                        end_time = int(datetime.now().timestamp() * 1000)
                        start_time = end_time - (48 * 60 * 60 * 1000)
                        candles_response = await async_api_request(info.candles_snapshot, symbol, '1h', start_time, end_time)
                        if candles_response and isinstance(candles_response, list) and len(candles_response) >= 2:
                            parsed_candles = parse_candles(candles_response)
                            candles = [CandleData(**candle) for candle in parsed_candles]
                            valid_candles = [c for c in candles if end_time - 24*60*60*1000 <= c.timestamp <= end_time]
                            if len(valid_candles) >= 2:
                                oldest_price = safe_float(valid_candles[0].open)
                                latest_price = safe_float(valid_candles[-1].close)
                                if oldest_price > 0:
                                    change_24h = ((latest_price - oldest_price) / oldest_price) * 100
                    except Exception as e:
                        logging.error(f"Candles fetch failed for {symbol}: {str(e)}")

                    return MarketInfo(
                        symbol=symbol,
                        markPrice=price,
                        lastPrice=price,
                        fundingRate=funding_rate,
                        volume24h=volume_24h,
                        change24h=change_24h,
                    )
                except Exception as e:
                    logging.error(f"Processing failed for {symbol}: {str(e)}")
                    return None

            tasks = [process_coin(coin) for coin in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if result is not None:
                    batch_data.append(result)

            market_data.extend(batch_data)
            skipped_markets.extend(batch_skipped)
            progress.progress(1.0)
            status.empty()

        df = pd.DataFrame([m.dict() for m in market_data])
        if df.empty:
            return apply_fallback_for_major_coins()
        return df.sort_values('volume24h', ascending=False)

    except Exception as e:
        st.write(f"Error fetching markets: {str(e)}")
        return apply_fallback_for_major_coins()

# Fetch Candles
async def fetch_hyperliquid_candles(symbol, interval="1h", limit=24):
    try:
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (limit * 60 * 60 * 1000)
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
        ohlcv = await async_api_request(info.candles_snapshot, base_symbol, interval, start_time, end_time)
        if not ohlcv or not isinstance(ohlcv, list):
            return None
        parsed_ohlcv = parse_candles(ohlcv)
        df = pd.DataFrame(parsed_ohlcv, columns=['timestamp', 'time_close', 'symbol', 'interval', 'open', 'close', 'high', 'low', 'volume', 'num'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.write(f"Error fetching candles for {symbol}: {str(e)}")
        return None

# Database Functions
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
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('active_trades', json.dumps(active_trades)))
    cursor.execute('''
    INSERT OR REPLACE INTO app_state (key, value)
    VALUES (?, ?)
    ''', ('completed_trades', json.dumps(completed_trades)))
    conn.commit()
    conn.close()

# Trading Logic
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
            logging.error(f"Could not load state: {str(e)}")
    
    def save_state(self):
        try:
            save_state_to_db(
                {k: v.dict() for k, v in self.active_trades.items()},
                [t.dict() for t in self.completed_trades]
            )
        except Exception as e:
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
                executed.append(f"New {signal.Signal} trade for {symbol} at {signal.Price}")
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
                        updates.append(f"Trade closed: {symbol} | Reason: {trade.exit_reason} | PnL: {trade.pct_change:.2f}%")
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

# Signal Generation
async def generate_signals(markets_df):
    if markets_df is None or markets_df.empty:
        st.write("No market data for signal generation")
        return []
    
    signals = []
    top_markets = markets_df.head(5)  # Reduced for iPhone 7
    total_markets = len(top_markets)
    micro_price_calc = MicroPrice(alpha=2.0, volatility_factor=1.0)
    progress_bar = st.progress(0)
    status_text = st.empty()
    MICRO_PRICE_THRESHOLD = 0.001

    async def process_market(index, market):
        symbol = market['symbol']
        perp_symbol = f"{symbol}/USDC:USDC"
        status_text.text(f"Analyzing {symbol}... ({index + 1}/{total_markets})")
        try:
            book_response = await async_api_request(info.l2_snapshot, perp_symbol)
            if not book_response or 'levels' not in book_response:
                return None
            parsed_levels = parse_orderbook_levels(book_response['levels'])
            book = OrderBook(levels=[MarketLevel(**level) for level in parsed_levels])
            bids = [level for level in book.levels if level.side == 'b']
            asks = [level for level in book.levels if level.side == 'a']
            if not bids or not asks:
                return None
            best_bid = max(bid.price for bid in bids)
            best_ask = min(ask.price for ask in asks)
            bid_size = sum(bid.size for bid in bids if bid.price == best_bid)
            ask_size = sum(ask.size for ask in asks if ask.price == best_ask)
            micro_price = micro_price_calc.calculate(best_bid, best_ask, bid_size, ask_size)
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000
            df = await fetch_hyperliquid_candles(symbol, interval='1h', limit=12)  # Reduced limit
            if df is None or len(df) < 6:
                return None
            current_price = df['close'].iloc[-1]
            recent_vols = df['volume'].tail(3)
            avg_vol = df['volume'].mean()
            recent_vol = recent_vols.mean()
            vol_surge = recent_vol / avg_vol if avg_vol > 0 else 0
            vol_consistent = all(v > avg_vol * 0.7 for v in recent_vols)
            df['hlrange'] = df['high'] - df['low']
            avg_range = df['hlrange'].mean()
            volatility_factor = min(max(avg_range / current_price, 0.01), 0.05)
            signal = "HOLD"
            reason = ""
            tp = "-"
            sl = "-"
            funding_rate = market['fundingRate']
            micro_price_deviation = (micro_price - mid_price) / mid_price
            price_threshold = max(MICRO_PRICE_THRESHOLD, volatility_factor * 0.5)
            max_spread_bps = 50
            tp_distance = max(volatility_factor * 5, 0.02)
            sl_distance = max(volatility_factor * 3, 0.015)
            if spread_bps <= max_spread_bps and vol_surge >= 1.5 and vol_consistent:
                if micro_price_deviation > price_threshold and funding_rate < -FUNDING_THRESHOLD:
                    signal = "LONG"
                    reason = f"Micro-price deviation: {micro_price_deviation:.4f}, Volume surge: {vol_surge:.2f}x"
                    tp = str(round(current_price * (1 + tp_distance), 4))
                    sl = str(round(current_price * (1 - sl_distance), 4))
                elif micro_price_deviation < -price_threshold and funding_rate > FUNDING_THRESHOLD:
                    signal = "SHORT"
                    reason = f"Micro-price deviation: {micro_price_deviation:.4f}, Volume surge: {vol_surge:.2f}x"
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
            logging.error(f"Error analyzing {symbol}: {str(e)}")
            return None

    tasks = [process_market(i, market) for i, (_, market) in enumerate(top_markets.iterrows())]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if result is not None:
            signals.append(result)

    progress_bar.progress(1.0)
    status_text.empty()
    return signals

# Scan Markets
def scan_markets():
    init_session_state()
    st.session_state.scanned_markets = pd.DataFrame()
    st.session_state.signals = []
    
    try:
        with st.spinner("Fetching market data..."):
            loop = asyncio.get_event_loop()
            markets_df = loop.run_until_complete(fetch_all_markets())
            if markets_df.empty:
                st.write("No market data fetched.")
                return
            st.session_state.scanned_markets = markets_df
            st.write(f"Found {len(markets_df)} markets.")
    except Exception as e:
        st.write(f"Error during market scan: {str(e)}")
        return
    
    try:
        if not markets_df.empty:
            with st.spinner("Analyzing markets..."):
                loop = asyncio.get_event_loop()
                signals = loop.run_until_complete(generate_signals(markets_df))
                st.session_state.signals = signals
                actionable_count = len([s for s in signals if s.Signal != 'HOLD'])
                if actionable_count > 0:
                    st.write(f"Found {actionable_count} actionable signals.")
                else:
                    st.write("No actionable signals found.")
    except Exception as e:
        st.write(f"Error during signal generation: {str(e)}")
        return

# Streamlit UI
tester = ForwardTester()

with st.sidebar:
    st.header("Settings")
    BASE_VOL = st.slider("Volume Threshold", 0.1, 2.0, 0.35, 0.05)
    liquidity_options = {
        "1,000 USD": 1000,
        "5,000 USD": 5000
    }
    selected_liquidity = st.selectbox("Minimum Liquidity", options=list(liquidity_options.keys()))
    MIN_LIQUIDITY = liquidity_options[selected_liquidity]
    st.session_state.MIN_LIQUIDITY = MIN_LIQUIDITY
    api_calls_per_second = st.slider("API calls per second", 1.0, 3.0, 2.0, 0.5)
    rate_limiter.calls_per_second = api_calls_per_second

tab1, tab2, tab3 = st.tabs(["Markets", "Active Trades", "Completed Trades"])

with tab1:
    st.write(f"Min Liquidity: ${MIN_LIQUIDITY:,} USD")
    if st.button("Scan Markets"):
        scan_markets()
    if not st.session_state.scanned_markets.empty:
        st.subheader("Markets")
        display_df = st.session_state.scanned_markets.rename(columns={
            'symbol': 'Symbol',
            'markPrice': 'Price',
            'volume24h': 'Volume (24h)',
            'fundingRate': 'Funding Rate',
            'change24h': 'Change (24h)'
        })
        st.dataframe(display_df, use_container_width=True)
    if st.session_state.signals:
        st.subheader("Signals")
        signals_df = pd.DataFrame([s.dict() for s in st.session_state.signals])
        signals_df['Reason'] = signals_df['Reason'].str.replace(r'[<>]', '', regex=True)  # Sanitize output
        signal_filter = st.multiselect("Filter Signals", ['LONG', 'SHORT', 'HOLD'], ['LONG', 'SHORT'])
        if signal_filter:
            filtered_df = signals_df[signals_df['Signal'].isin(signal_filter)]
            st.dataframe(filtered_df, use_container_width=True)
            actionable_signals = [s for s in st.session_state.signals if s.Signal != "HOLD" and s.Signal in signal_filter]
            if actionable_signals and st.button("Execute Signals"):
                results = tester.execute_trades(actionable_signals)
                for result in results:
                    st.write(result)

with tab2:
    st.header("Active Trades")
    if st.button("Update Trades"):
        updates = tester.update_trades()
        for update in updates:
            st.write(update)
    if tester.active_trades:
        active_df = pd.DataFrame([t.dict() for t in tester.active_trades.values()])
        st.dataframe(active_df, use_container_width=True)
    else:
        st.write("No active trades.")

with tab3:
    st.header("Completed Trades")
    stats, completed_df = tester.get_performance_report()
    if isinstance(completed_df, pd.DataFrame) and len(completed_df) > 0:
        st.dataframe(completed_df, use_container_width=True)
    else:
        st.write("No completed trades.")
