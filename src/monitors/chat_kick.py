"""
Kick chat monitor using WebSocket (Pusher)

Kick uses Pusher for real-time chat. This module connects to their
WebSocket endpoint to receive chat messages.
"""
import asyncio
import json
import logging
import time
from typing import Optional

import pysher
from pysher.connection import Connection
import websocket
import websockets

from .chat_base import BaseChatMonitor, ChatMessage
from ..utils.kick_api import KickAPI

logger = logging.getLogger(__name__)


# --- THE FIX: Force URL to US2 Cluster ---
class CustomPusherConnection(Connection):
    def _connect(self):
        # 1. FORCE the URL to use the 'us2' cluster.
        # This overrides whatever wrong URL pysher was trying to use.
        self.url = f"wss://ws-us2.pusher.com/app/{self.key}?protocol=7&client=python&version=1.0&flash=false" # type: ignore
        
        # 2. Inject the Headers as before
        self.socket = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            header={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Origin": "https://kick.com",
                "Referer": "https://kick.com/"
            }
        )
        self.socket.run_forever()

pysher.Connection = CustomPusherConnection


class KickChatMonitor(BaseChatMonitor):
    """
    Monitors Kick chat via Pusher WebSocket.
    
    Kick uses Pusher for their chat system. We connect directly to the
    WebSocket endpoint and subscribe to the channel's chatroom.
    
    Uses authenticated API calls to get chatroom ID (bypasses Cloudflare).
    """
    
    # Kick's Pusher configuration
    PUSHER_KEY = "32cbd69e4b950bf97679"
    PUSHER_CLUSTER = "us2"
    
    def __init__(self, channel: str):
        """
        Initialize Kick chat monitor.
        
        Args:
            channel: Kick channel name (username)
        """
        super().__init__(channel.lower())
        self._websocket = None
        self._chatroom_id: Optional[int] = None
        self._api = KickAPI()  # Authenticated API client
        
        # Stats tracking for scoring
        self._msg_count_window = 0
        self._last_tick = time.time()
    
    @property
    def platform_name(self) -> str:
        return "Kick"
    
    def _get_websocket_url(self) -> str:
        """Get the Pusher WebSocket URL"""
        return (
            f"wss://ws-{self.PUSHER_CLUSTER}.pusher.com/app/{self.PUSHER_KEY}"
            f"?protocol=7&client=js&version=7.0.0&flash=false"
        )
    
    async def _get_chatroom_id(self) -> Optional[int]:
        """
        Fetch the chatroom ID for the channel using authenticated API.
        This bypasses Cloudflare blocking.
        """
        logger.info(f"Fetching chatroom ID for '{self.channel}' via authenticated API...")
        
        chatroom_id = await self._api.get_chatroom_id(self.channel)
        
        if chatroom_id:
            logger.info(f"Got chatroom ID: {chatroom_id}")
        else:
            logger.warning("Authenticated API failed, trying fallback...")
            chatroom_id = await self._get_chatroom_id_fallback()
        
        return chatroom_id
    
    async def _get_chatroom_id_fallback(self) -> Optional[int]:
        """
        Fallback method using multiple approaches to get chatroom ID.
        Tries different endpoints and methods.
        """
        import aiohttp
        
        # Method 1: Try kick.com internal API v2 (different endpoint)
        endpoints = [
            f"https://kick.com/api/v2/channels/{self.channel}",
            f"https://kick.com/api/v1/channels/{self.channel}",
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Origin": "https://kick.com",
            "Referer": "https://kick.com/",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        
        for url in endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        logger.debug(f"Trying {url}: status={response.status}")
                        if response.status == 200:
                            data = await response.json()
                            
                            # Try to extract chatroom ID from various locations
                            chatroom_id = None
                            if isinstance(data, dict):
                                if "chatroom" in data:
                                    chatroom_id = data["chatroom"].get("id")
                                elif "chatroom_id" in data:
                                    chatroom_id = data["chatroom_id"]
                                elif "id" in data:
                                    # Some responses have id at root level
                                    chatroom_id = data["id"]
                            
                            if chatroom_id:
                                logger.info(f"Fallback got chatroom ID from {url}: {chatroom_id}")
                                return chatroom_id
                        elif response.status == 403:
                            logger.debug(f"403 from {url}, trying next...")
                            continue
            except Exception as e:
                logger.debug(f"Error with {url}: {e}")
                continue
        
        logger.error("All fallback methods failed to get chatroom ID")
        return None
    
    async def _connect(self):
        """Connect to Kick chat via Pusher WebSocket"""
        # First, get the chatroom ID
        self._chatroom_id = await self._get_chatroom_id()
        
        if not self._chatroom_id:
            logger.error("Could not get chatroom ID - chat monitoring disabled")
            return
        
        # Connect to Pusher with proper headers
        # IMPORTANT: Must use us2 cluster and include Origin/Referer headers
        url = self._get_websocket_url()
        logger.info(f"Connecting to Pusher: {url}")
        
        extra_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://kick.com",
            "Referer": "https://kick.com/"
        }
        
        self._websocket = await websockets.connect(url, additional_headers=extra_headers)
        
        # Wait for connection established message
        response = await self._websocket.recv()
        data = json.loads(response)
        logger.info(f"Pusher response: {data}")
        
        if data.get("event") == "pusher:connection_established":
            logger.info("Connected to Kick Pusher")
            
            # Subscribe to the chatroom channel
            channel_name = f"chatrooms.{self._chatroom_id}.v2"
            subscribe_msg = {
                "event": "pusher:subscribe",
                "data": {
                    "auth": "",
                    "channel": channel_name
                }
            }
            logger.info(f"Subscribing to channel: {channel_name}")
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to chatroom {self._chatroom_id}")
        else:
            logger.warning(f"Unexpected connection response: {data}")
    
    async def _listen(self):
        """Listen for chat messages"""
        if not self._websocket:
            logger.error("No websocket connection for chat listening")
            return
        
        logger.info("Starting chat message listener...")
        
        while self._is_running:
            try:
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=30.0
                )
                
                data = json.loads(message)
                event = data.get("event", "")
                
                # DEBUG: Log all events
                logger.debug(f"Pusher event: {event}")
                
                # Handle chat messages
                if event == "App\\Events\\ChatMessageEvent":
                    #logger.info(f"Chat message received!")
                    self._handle_chat_message(data)
                
                # Handle subscription success
                elif event == "pusher_internal:subscription_succeeded":
                    logger.info("Successfully subscribed to chat - ready to receive messages!")
                
                # Handle subscription error
                elif event == "pusher_internal:subscription_error":
                    logger.error(f"Subscription error: {data}")
                
                # Respond to pings
                elif event == "pusher:ping":
                    await self._websocket.send(json.dumps({"event": "pusher:pong"}))
                
                # Log unknown events for debugging
                elif event and not event.startswith("pusher:"):
                    logger.info(f"Unknown event: {event} - {data}")
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                logger.debug("Sending keepalive ping...")
                try:
                    await self._websocket.send(json.dumps({"event": "pusher:ping"}))
                except:
                    break
            except websockets.ConnectionClosed:
                logger.warning("Kick WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Kick chat error: {e}")
                await asyncio.sleep(1.0)
    
    def _handle_chat_message(self, data: dict):
        """Parse and emit a chat message"""
        try:
            # Parse the nested JSON data
            message_data = json.loads(data.get("data", "{}"))
            
            sender = message_data.get("sender", {})
            
            chat_msg = ChatMessage(
                timestamp=time.time(),
                username=sender.get("username", "unknown"),
                message=message_data.get("content", ""),
                platform="kick",
                is_subscriber=sender.get("is_subscriber", False),
                is_moderator=sender.get("is_moderator", False),
                badges=[b.get("type", "") for b in sender.get("badges", [])],
            )
            
            # Track message count for scoring
            self._msg_count_window += 1
            
            self._emit_message(chat_msg)
            
        except Exception as e:
            logger.debug(f"Failed to parse Kick message: {e}")
    
    def get_messages_per_second(self) -> float:
        """
        Calculate messages per second since last call.
        Used by ScoringEngine for chat velocity scoring.
        
        Returns:
            Messages per second rate
        """
        current_time = time.time()
        elapsed = current_time - self._last_tick
        
        if elapsed >= 1.0:
            mps = self._msg_count_window / elapsed
            self._msg_count_window = 0
            self._last_tick = current_time
            return mps
        
        return 0.0
    
    async def _disconnect(self):
        """Disconnect from Kick chat"""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    channel = sys.argv[1] if len(sys.argv) > 1 else "xqc"
    
    with KickChatMonitor(channel) as chat:
        print(f"Monitoring Kick chat: {channel}")
        
        while True:
            msg = chat.get_message(timeout=1.0)
            if msg:
                print(f"[{msg.username}]: {msg.message}")
