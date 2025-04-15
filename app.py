import streamlit as st
import pandas as pd
import ccxt
import time
from datetime import datetime

# Initialize Exchange with proper config
@st.cache_resource
def init_exchange():
    return ccxt.hyperliquid({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True
        }
    })

exchange = init_exchange()

# Improved market data fetcher
def get_market_data(symbol):
    try:
        # Get order book for price data
        orderbook = exchange.fetch_order_book(symbol)
        if not orderbook or not orderbook['bids'] or not orderbook['asks']:
            return None
            
        bid = orderbook['bids'][0][0]
        ask = orderbook['asks'][0][0]
        price = (bid + ask) / 2
        
        # Estimate volume from order book depth
        depth_price_range = price * 0.02  # ±2% depth
        bid_volume = sum(amount for (bid_price, amount) in orderbook['bids'] if bid_price >= price - depth_price_range)
        ask_volume = sum(amount for (ask_price, amount) in orderbook['asks'] if ask_price <= price + depth_price_range)
        volume = (bid_volume + ask_volume) * price * 4  # Turnover factor
        
        return {
            'symbol': symbol.split(':')[0],  # Extract base symbol
            'price': price,
            'volume': volume,
            'bid': bid,
            'ask': ask
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Main market scanner function
def scan_markets():
    try:
        markets = exchange.load_markets()
        perp_markets = [
            symbol for symbol, market in markets.items() 
            if market['type'] == 'swap' 
            and not market['spot']
            and not symbol.endswith('USDC:USDC')
        ]
        
        market_data = []
        for symbol in perp_markets[:20]:  # Limit to top 20 for demo
            data = get_market_data(symbol)
            if data:
                market_data.append(data)
        
        return pd.DataFrame(market_data)
    except Exception as e:
        st.error(f"Market scan failed: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.set_page_config(page_title="Hyperliquid Scanner", layout="wide")
st.title("Hyperliquid Market Scanner")

if st.button("Scan Markets"):
    with st.spinner("Loading market data..."):
        df = scan_markets()
        
        if not df.empty:
            st.success(f"Found {len(df)} markets")
            st.dataframe(df)
            
            # Basic price chart for first market
            if len(df) > 0:
                try:
                    trades = exchange.fetch_trades(df.iloc[0]['symbol'] + ':USDC', limit=100)
                    if trades:
                        trades_df = pd.DataFrame(trades)
                        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
                        st.line_chart(trades_df.set_index('datetime')['price'])
                except:
                    st.warning("Could not load trade history")
        else:
            st.warning("No market data found")

# Debug info
with st.expander("Debug Info"):
    st.write("Exchange API status:", exchange.has)
    st.write("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
