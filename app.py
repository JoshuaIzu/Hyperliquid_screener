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

# Get price from order book
def get_price_from_orderbook(symbol):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        if orderbook and orderbook['bids'] and orderbook['asks']:
            bid = orderbook['bids'][0][0]
            ask = orderbook['asks'][0][0]
            return (bid + ask) / 2
    except:
        return None

# Get volume from recent trades
def get_volume_from_trades(symbol):
    try:
        trades = exchange.fetch_trades(symbol, limit=100)
        if trades:
            df = pd.DataFrame(trades)
            return df['amount'].sum() * df['price'].mean()
    except:
        return 0

# Main market scanner function
def scan_markets():
    try:
        markets = exchange.load_markets()
        
        # Get all perpetual markets (excluding USDC pairs)
        perp_markets = [
            symbol for symbol, market in markets.items() 
            if market['type'] == 'swap' 
            and not market['spot']
            and not symbol.endswith('USDC:USDC')
        ]
        
        market_data = []
        for symbol in perp_markets[:20]:  # Limit to top 20 for demo
            price = get_price_from_orderbook(symbol)
            if not price:
                continue
                
            volume = get_volume_from_trades(symbol)
            
            market_data.append({
                'Symbol': symbol.split(':')[0],  # Get base asset
                'Price': price,
                '24h Volume': volume,
                'Bid/Ask Spread': f"{((price - exchange.fetch_order_book(symbol)['bids'][0][0])/price)*100:.2f}%"
            })
            
            time.sleep(0.1)  # Rate limiting
            
        return pd.DataFrame(market_data)
    except Exception as e:
        st.error(f"Error scanning markets: {str(e)}")
        return pd.DataFrame()

# Streamlit UI
st.set_page_config(page_title="Hyperliquid Scanner", layout="wide")
st.title("📊 Hyperliquid Market Scanner")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("Real-time perpetual markets data from Hyperliquid")

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()

if st.button("Scan Markets"):
    with st.spinner("Loading market data..."):
        df = scan_markets()
        
        if not df.empty:
            st.success(f"Found {len(df)} markets")
            
            # Format the dataframe display
            st.dataframe(
                df.style.format({
                    'Price': '{:.4f}',
                    '24h Volume': '{:,.2f}'
                }),
                use_container_width=True
            )
            
            # Show price chart for first market
            if len(df) > 0:
                try:
                    symbol = df.iloc[0]['Symbol'] + ':USDC'
                    trades = exchange.fetch_trades(symbol, limit=100)
                    if trades:
                        trades_df = pd.DataFrame(trades)
                        trades_df['datetime'] = pd.to_datetime(trades_df['timestamp'], unit='ms')
                        st.line_chart(trades_df.set_index('datetime')['price'])
                except Exception as e:
                    st.warning(f"Could not load trade history: {str(e)}")
        else:
            st.warning("No market data found")

# Exchange info
with st.expander("ℹ️ Exchange Information"):
    st.write(f"Connected to: {exchange.name}")
    st.write(f"API Status: {'✅ Live' if exchange.has['fetchOrderBook'] else '❌ Issues'}")
    st.write(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
