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

import websockets

from .chat_base import BaseChatMonitor, ChatMessage

logger = logging.getLogger(__name__)


class KickChatMonitor(BaseChatMonitor):
    """
    Monitors Kick chat via Pusher WebSocket.
    
    Kick uses Pusher for their chat system. We connect directly to the
    WebSocket endpoint and subscribe to the channel's chatroom.
    """
    
    # Kick's Pusher configuration
    PUSHER_KEY = "eb1d5f283081a78b932c"
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
        Fetch the chatroom ID for the channel.
        This requires making an HTTP request to Kick's API.
        """
        import aiohttp
        
        url = f"https://kick.com/api/v2/channels/{self.channel}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        chatroom_id = data.get("chatroom", {}).get("id")
                        logger.info(f"Got chatroom ID: {chatroom_id}")
                        return chatroom_id
                    else:
                        logger.error(f"Failed to get channel info: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching chatroom ID: {e}")
            return None
    
    async def _connect(self):
        """Connect to Kick chat via Pusher WebSocket"""
        # First, get the chatroom ID
        self._chatroom_id = await self._get_chatroom_id()
        
        if not self._chatroom_id:
            logger.error("Could not get chatroom ID - chat monitoring disabled")
            return
        
        # Connect to Pusher
        url = self._get_websocket_url()
        self._websocket = await websockets.connect(url)
        
        # Wait for connection established message
        response = await self._websocket.recv()
        data = json.loads(response)
        
        if data.get("event") == "pusher:connection_established":
            logger.info("Connected to Kick Pusher")
            
            # Subscribe to the chatroom channel
            subscribe_msg = {
                "event": "pusher:subscribe",
                "data": {
                    "auth": "",
                    "channel": f"chatrooms.{self._chatroom_id}.v2"
                }
            }
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to chatroom {self._chatroom_id}")
        else:
            logger.warning(f"Unexpected connection response: {data}")
    
    async def _listen(self):
        """Listen for chat messages"""
        if not self._websocket:
            return
        
        while self._is_running:
            try:
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=30.0
                )
                
                data = json.loads(message)
                event = data.get("event", "")
                
                # Handle chat messages
                if event == "App\\Events\\ChatMessageEvent":
                    self._handle_chat_message(data)
                
                # Handle subscription success
                elif event == "pusher_internal:subscription_succeeded":
                    logger.info("Successfully subscribed to chat")
                
                # Respond to pings
                elif event == "pusher:ping":
                    await self._websocket.send(json.dumps({"event": "pusher:pong"}))
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
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
            
            self._emit_message(chat_msg)
            
        except Exception as e:
            logger.debug(f"Failed to parse Kick message: {e}")
    
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
