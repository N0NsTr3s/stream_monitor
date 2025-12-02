"""
Base class for chat monitors
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    timestamp: float  # Unix timestamp
    username: str
    message: str
    platform: str
    is_subscriber: bool = False
    is_moderator: bool = False
    emote_count: int = 0
    badges: List[str] = field(default_factory=list)
    
    @property
    def length(self) -> int:
        return len(self.message)
    
    @property
    def has_caps(self) -> bool:
        """Check if message is mostly caps (excitement indicator)"""
        alpha_chars = [c for c in self.message if c.isalpha()]
        if len(alpha_chars) < 3:
            return False
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        return caps_ratio > 0.7


class BaseChatMonitor(ABC):
    """Abstract base class for platform-specific chat monitors"""
    
    def __init__(self, channel: str):
        """
        Initialize chat monitor.
        
        Args:
            channel: Channel name or ID to monitor
        """
        self.channel = channel
        self._message_queue: Queue[ChatMessage] = Queue(maxsize=1000)
        self._is_running = False
        self._thread: Optional[Thread] = None
        self._callbacks: List[Callable[[ChatMessage], None]] = []
        
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name"""
        pass
    
    @abstractmethod
    async def _connect(self):
        """Connect to the chat service"""
        pass
    
    @abstractmethod
    async def _disconnect(self):
        """Disconnect from the chat service"""
        pass
    
    @abstractmethod
    async def _listen(self):
        """Main loop to listen for messages"""
        pass
    
    def on_message(self, callback: Callable[[ChatMessage], None]):
        """Register a callback for new messages"""
        self._callbacks.append(callback)
    
    def _emit_message(self, message: ChatMessage):
        """Emit a message to queue and callbacks"""
        try:
            self._message_queue.put_nowait(message)
        except:
            # Queue full, drop oldest
            try:
                self._message_queue.get_nowait()
                self._message_queue.put_nowait(message)
            except:
                pass
        
        for callback in self._callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_message(self, timeout: float = 0.1) -> Optional[ChatMessage]:
        """Get next message from queue"""
        try:
            return self._message_queue.get(timeout=timeout)
        except:
            return None
    
    def get_all_messages(self) -> List[ChatMessage]:
        """Get all pending messages"""
        messages = []
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except:
                break
        return messages
    
    def _run_async(self):
        """Run the async event loop in a thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"Chat monitor error: {e}")
        finally:
            loop.close()
    
    async def _async_main(self):
        """Main async entry point"""
        await self._connect()
        try:
            await self._listen()
        finally:
            await self._disconnect()
    
    def start(self):
        """Start the chat monitor in a background thread"""
        if self._is_running:
            return
        
        self._is_running = True
        self._thread = Thread(target=self._run_async, daemon=True)
        self._thread.start()
        logger.info(f"{self.platform_name} chat monitor started for {self.channel}")
    
    def stop(self):
        """Stop the chat monitor"""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"{self.platform_name} chat monitor stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
