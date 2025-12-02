"""
Twitch chat monitor using twitchio
"""
import asyncio
import logging
import time
from typing import Optional

from .chat_base import BaseChatMonitor, ChatMessage

logger = logging.getLogger(__name__)

# Try to import twitchio
try:
    from twitchio.ext import commands
    from twitchio import Message
    TWITCHIO_AVAILABLE = True
except ImportError:
    TWITCHIO_AVAILABLE = False
    logger.warning("twitchio not installed - Twitch chat monitoring disabled")


class TwitchChatMonitor(BaseChatMonitor):
    """
    Monitors Twitch chat using twitchio IRC client.
    
    Note: For read-only access, you can use an anonymous connection.
    For full features, provide OAuth token.
    """
    
    def __init__(self, channel: str, oauth_token: Optional[str] = None):
        """
        Initialize Twitch chat monitor.
        
        Args:
            channel: Twitch channel name (without #)
            oauth_token: Optional OAuth token for authenticated access
        """
        super().__init__(channel.lower().lstrip('#'))
        self.oauth_token = oauth_token or "oauth:anonymous"  # Anonymous works for reading
        self._bot = None
        self._connected_event = asyncio.Event()
    
    @property
    def platform_name(self) -> str:
        return "Twitch"
    
    async def _connect(self):
        """Connect to Twitch IRC"""
        if not TWITCHIO_AVAILABLE:
            raise ImportError("twitchio is required for Twitch chat monitoring")
        
        # Create a minimal bot class
        monitor = self
        
        class ChatBot(commands.Bot):
            def __init__(bot_self):
                # Use anonymous login for read-only
                super().__init__(
                    token=monitor.oauth_token,
                    prefix='!',
                    initial_channels=[monitor.channel]
                )
            
            async def event_ready(bot_self):
                logger.info(f"Connected to Twitch as {bot_self.nick}")
                monitor._connected_event.set()
            
            async def event_message(bot_self, message: Message):
                if message.echo:
                    return
                
                # Extract message data
                chat_msg = ChatMessage(
                    timestamp=time.time(),
                    username=message.author.name if message.author else "unknown",
                    message=message.content,
                    platform="twitch",
                    is_subscriber=message.author.is_subscriber if message.author else False,
                    is_moderator=message.author.is_mod if message.author else False,
                    badges=[],
                )
                
                monitor._emit_message(chat_msg)
        
        self._bot = ChatBot()
        logger.info(f"Connecting to Twitch channel: {self.channel}")
    
    async def _listen(self):
        """Run the bot"""
        if self._bot:
            await self._bot.start()
    
    async def _disconnect(self):
        """Disconnect from Twitch"""
        if self._bot:
            await self._bot.close()
            self._bot = None


# Fallback implementation using raw IRC if twitchio is not available
class TwitchChatMonitorRaw(BaseChatMonitor):
    """
    Fallback Twitch chat monitor using raw IRC.
    Works without twitchio dependency.
    """
    
    def __init__(self, channel: str, oauth_token: Optional[str] = None):
        super().__init__(channel.lower().lstrip('#'))
        self.oauth_token = oauth_token
        self._reader = None
        self._writer = None
    
    @property
    def platform_name(self) -> str:
        return "Twitch"
    
    async def _connect(self):
        """Connect to Twitch IRC directly"""
        self._reader, self._writer = await asyncio.open_connection(
            'irc.chat.twitch.tv', 6667
        )
        
        # Anonymous login
        nick = "justinfan" + str(int(time.time()) % 100000)
        
        self._writer.write(f"NICK {nick}\r\n".encode())
        self._writer.write(f"JOIN #{self.channel}\r\n".encode())
        await self._writer.drain()
        
        logger.info(f"Connected to Twitch IRC as {nick}")
    
    async def _listen(self):
        """Listen for IRC messages"""
        while self._is_running:
            try:
                line = await asyncio.wait_for(
                    self._reader.readline(), 
                    timeout=30.0
                )
                
                if not line:
                    break
                
                line = line.decode('utf-8', errors='ignore').strip()
                
                # Respond to PING
                if line.startswith('PING'):
                    self._writer.write(f"PONG {line[5:]}\r\n".encode())
                    await self._writer.drain()
                    continue
                
                # Parse PRIVMSG
                if 'PRIVMSG' in line:
                    self._parse_privmsg(line)
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                self._writer.write(b"PING :keepalive\r\n")
                await self._writer.drain()
            except Exception as e:
                logger.error(f"IRC error: {e}")
                break
    
    def _parse_privmsg(self, line: str):
        """Parse IRC PRIVMSG line"""
        try:
            # Format: :username!user@user.tmi.twitch.tv PRIVMSG #channel :message
            parts = line.split(' ', 3)
            if len(parts) < 4:
                return
            
            username = parts[0].split('!')[0].lstrip(':')
            message_text = parts[3].lstrip(':')
            
            chat_msg = ChatMessage(
                timestamp=time.time(),
                username=username,
                message=message_text,
                platform="twitch",
            )
            
            self._emit_message(chat_msg)
            
        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")
    
    async def _disconnect(self):
        """Disconnect from IRC"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()


# Export the best available implementation
if TWITCHIO_AVAILABLE:
    TwitchChat = TwitchChatMonitor
else:
    TwitchChat = TwitchChatMonitorRaw


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    channel = sys.argv[1] if len(sys.argv) > 1 else "shroud"
    
    with TwitchChat(channel) as chat:
        print(f"Monitoring {channel}...")
        
        while True:
            msg = chat.get_message(timeout=1.0)
            if msg:
                print(f"[{msg.username}]: {msg.message}")
