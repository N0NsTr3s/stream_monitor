"""
YouTube Live Chat monitor using pytchat
"""
import asyncio
import logging
import time
import re
from typing import Optional

from .chat_base import BaseChatMonitor, ChatMessage

logger = logging.getLogger(__name__)

# Try to import pytchat
try:
    import pytchat
    PYTCHAT_AVAILABLE = True
except ImportError:
    PYTCHAT_AVAILABLE = False
    logger.warning("pytchat not installed - YouTube chat monitoring disabled")


class YouTubeChatMonitor(BaseChatMonitor):
    """
    Monitors YouTube Live Chat using pytchat.
    
    pytchat scrapes the live chat without requiring API keys.
    """
    
    def __init__(self, video_id: str):
        """
        Initialize YouTube chat monitor.
        
        Args:
            video_id: YouTube video/stream ID (the v= parameter or full URL)
        """
        # Extract video ID if full URL provided
        video_id = self._extract_video_id(video_id)
        super().__init__(video_id)
        self._chat = None
    
    @staticmethod
    def _extract_video_id(url_or_id: str) -> str:
        """Extract video ID from URL or return as-is if already an ID"""
        # Already an ID
        if len(url_or_id) == 11 and '/' not in url_or_id:
            return url_or_id
        
        # Various YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=)([\w-]{11})',
            r'(?:youtube\.com/live/)([\w-]{11})',
            r'(?:youtu\.be/)([\w-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        # Return as-is, let pytchat handle it
        return url_or_id
    
    @property
    def platform_name(self) -> str:
        return "YouTube"
    
    async def _connect(self):
        """Initialize pytchat connection"""
        if not PYTCHAT_AVAILABLE:
            raise ImportError("pytchat is required for YouTube chat monitoring")
        
        try:
            self._chat = pytchat.create(video_id=self.channel)
            logger.info(f"Connected to YouTube Live Chat: {self.channel}")
        except Exception as e:
            logger.error(f"Failed to connect to YouTube chat: {e}")
            raise
    
    async def _listen(self):
        """Listen for chat messages"""
        if not self._chat:
            return
        
        while self._is_running and self._chat.is_alive():
            try:
                # pytchat is sync, so we run in executor
                messages = await asyncio.get_event_loop().run_in_executor(
                    None, self._get_messages
                )
                
                for msg in messages:
                    self._emit_message(msg)
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"YouTube chat error: {e}")
                await asyncio.sleep(1.0)
    
    def _get_messages(self) -> list:
        """Sync method to get messages from pytchat"""
        messages = []
        
        if not self._chat:
            return messages
        
        for c in self._chat.get().sync_items():
            chat_msg = ChatMessage(
                timestamp=time.time(),
                username=c.author.name,
                message=c.message,
                platform="youtube",
                is_subscriber=c.author.isChatModerator or c.author.isChatOwner,
                is_moderator=c.author.isChatModerator,
                badges=[],
            )
            messages.append(chat_msg)
        
        return messages
    
    async def _disconnect(self):
        """Disconnect from YouTube chat"""
        if self._chat:
            self._chat.terminate()
            self._chat = None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if not PYTCHAT_AVAILABLE:
        print("pytchat not installed. Run: pip install pytchat")
        sys.exit(1)
    
    video_id = sys.argv[1] if len(sys.argv) > 1 else "jfKfPfyJRdk"  # lofi girl
    
    with YouTubeChatMonitor(video_id) as chat:
        print(f"Monitoring YouTube chat: {video_id}")
        
        while True:
            msg = chat.get_message(timeout=1.0)
            if msg:
                print(f"[{msg.username}]: {msg.message}")
