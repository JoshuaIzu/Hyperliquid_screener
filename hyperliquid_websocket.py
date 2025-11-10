import asyncio
import json
import logging
import websockets
import streamlit as st
from typing import Dict, Any, Optional, Callable, List

# WebSocket URL for Hyperliquid
HYPERLIQUID_WSS = "wss://api.hyperliquid.xyz/ws"

class HyperliquidWebSocket:
    """Manages a single WebSocket connection and its subscriptions."""
    def __init__(self, url: str):
        self.url = url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.connection_lock = asyncio.Lock()
        self.data_cache = {
            "orderbook": {},
            "candles": {}
        }

    async def connect(self):
        """Establishes the WebSocket connection."""
        async with self.connection_lock:
            if self.ws and self.ws.open:
                logging.info("WebSocket already connected.")
                return
            try:
                self.ws = await websockets.connect(self.url)
                self.is_running = True
                logging.info("WebSocket connection established.")
            except Exception as e:
                logging.error(f"Failed to connect to WebSocket: {e}")
                self.ws = None
                self.is_running = False

    async def _subscribe(self, subscription_msg: Dict[str, Any]):
        """Sends a subscription message to the WebSocket."""
        if self.ws and self.ws.open:
            try:
                await self.ws.send(json.dumps(subscription_msg))
                sub_id = self._get_subscription_id(subscription_msg)
                self.subscriptions[sub_id] = subscription_msg
                logging.info(f"Sent subscription: {subscription_msg}")
            except Exception as e:
                logging.error(f"Failed to send subscription {subscription_msg}: {e}")
        else:
            logging.warning("WebSocket not connected. Cannot subscribe.")

    def _get_subscription_id(self, msg: Dict[str, Any]) -> str:
        """Generates a unique ID for a subscription message."""
        if msg["method"] == "subscribe":
            sub_type = msg["subscription"]["type"]
            coin = msg["subscription"].get("coin", "")
            interval = msg["subscription"].get("interval", "")
            return f"{sub_type}_{coin}_{interval}".lower()
        return ""

    async def listen(self):
        """Listens for incoming messages and processes them."""
        if not self.ws:
            logging.warning("Cannot listen, WebSocket is not connected.")
            return
        
        while self.is_running:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                self._process_message(data)
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed. Attempting to reconnect...")
                await self.reconnect()
            except Exception as e:
                logging.error(f"Error during WebSocket listen: {e}")
                await asyncio.sleep(5) # Avoid rapid-fire errors

    def _process_message(self, data: Dict[str, Any]):
        """Processes a single message from the WebSocket."""
        channel = data.get("channel")
        if channel == "l2Book":
            coin = data.get("data", {}).get("coin", "").upper()
            if coin:
                self.data_cache["orderbook"][coin] = data["data"]
                logging.debug(f"Updated orderbook for {coin}")
        elif channel == "candle":
            candle_data = data.get("data", {})
            coin = candle_data.get("c", "").upper()
            interval = candle_data.get("i", "")
            if coin and interval:
                key = f"{coin}_{interval}"
                self.data_cache["candles"][key] = candle_data
                logging.debug(f"Updated {interval} candle for {coin}")

    async def subscribe_to_orderbook(self, symbol: str):
        """Subscribes to the L2 order book for a given symbol."""
        subscription_msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": symbol.upper()}
        }
        await self._subscribe(subscription_msg)

    async def subscribe_to_candles(self, symbol: str, interval: str):
        """Subscribes to candle data for a given symbol and interval."""
        subscription_msg = {
            "method": "subscribe",
            "subscription": {"type": "candle", "coin": symbol.upper(), "interval": interval}
        }
        await self._subscribe(subscription_msg)

    async def close(self):
        """Closes the WebSocket connection."""
        self.is_running = False
        if self.ws and self.ws.open:
            try:
                await self.ws.close()
                logging.info("WebSocket connection closed.")
            except Exception as e:
                logging.error(f"Error closing WebSocket: {e}")
        self.ws = None

    async def reconnect(self):
        """Handles reconnection logic."""
        await self.close()
        logging.info("Reconnecting to WebSocket...")
        await asyncio.sleep(5)  # Wait before reconnecting
        await self.connect()
        if self.ws:
            # Re-subscribe to all previous subscriptions
            for sub_msg in self.subscriptions.values():
                await self._subscribe(sub_msg)
            # Restart the listener task
            asyncio.create_task(self.listen())

class HyperliquidWebSocketManager:
    """Manages the lifecycle of the HyperliquidWebSocket and related tasks."""
    def __init__(self, url: str = HYPERLIQUID_WSS):
        self.ws_client = HyperliquidWebSocket(url)
        self.listen_task: Optional[asyncio.Task] = None

    async def start(self):
        """Connects the client and starts the listener task."""
        await self.ws_client.connect()
        if self.ws_client.is_running:
            self.listen_task = asyncio.create_task(self.ws_client.listen())
            logging.info("WebSocket manager started.")
        else:
            logging.error("WebSocket manager failed to start.")
            raise ConnectionError("WebSocket manager failed to start.")

    async def stop(self):
        """Stops the listener task and closes the connection."""
        if self.listen_task and not self.listen_task.done():
            self.listen_task.cancel()
        await self.ws_client.close()
        logging.info("WebSocket manager stopped.")

    async def subscribe(self, subscriptions: List[Dict[str, Any]]):
        """
        Subscribes to a list of channels.
        Example: [{"type": "l2Book", "coin": "BTC"}, {"type": "candle", "coin": "ETH", "interval": "1h"}]
        """
        for sub in subscriptions:
            if sub["type"] == "l2Book":
                await self.ws_client.subscribe_to_orderbook(sub["coin"])
            elif sub["type"] == "candle":
                await self.ws_client.subscribe_to_candles(sub["coin"], sub["interval"])

    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves the latest orderbook data for a symbol from the cache."""
        return self.ws_client.data_cache["orderbook"].get(symbol.upper())

    def get_candle(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Retrieves the latest candle data for a symbol and interval from the cache."""
        key = f"{symbol.upper()}_{interval}"
        return self.ws_client.data_cache["candles"].get(key)

# --- Streamlit Session State Integration ---

async def initialize_websocket_manager() -> Optional[HyperliquidWebSocketManager]:
    """Initializes and starts the WebSocket manager, storing it in session state."""
    if 'ws_manager' not in st.session_state or st.session_state.ws_manager is None:
        try:
            manager = HyperliquidWebSocketManager()
            await manager.start()
            st.session_state.ws_manager = manager
            logging.info("WebSocketManager initialized and stored in session state.")
            return manager
        except Exception as e:
            logging.error(f"Failed to initialize WebSocketManager: {e}")
            st.session_state.ws_manager = None
            return None
    return st.session_state.ws_manager

def get_websocket_manager() -> Optional[HyperliquidWebSocketManager]:
    """Retrieves the WebSocket manager from session state."""
    return st.session_state.get('ws_manager')

async def get_websocket_orderbook(symbol: str) -> Optional[Dict[str, Any]]:
    """Utility function to get orderbook data using the session's manager."""
    manager = get_websocket_manager()
    if manager:
        # First, ensure subscription exists
        await manager.subscribe([{"type": "l2Book", "coin": symbol}])
        # Give it a moment to receive data
        await asyncio.sleep(0.5)
        return manager.get_orderbook(symbol)
    return None

async def get_websocket_candles(symbol: str, interval: str) -> Optional[Dict[str, Any]]:
    """Utility function to get candle data using the session's manager."""
    manager = get_websocket_manager()
    if manager:
        # First, ensure subscription exists
        await manager.subscribe([{"type": "candle", "coin": symbol, "interval": interval}])
        # Give it a moment to receive data
        await asyncio.sleep(0.5)
        return manager.get_candle(symbol, interval)
    return None
