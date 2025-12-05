"""
YouTube Live Chat monitor using Official YouTube Data API v3
"""
import asyncio
import logging
import re
from typing import Optional

from .chat_base import BaseChatMonitor, ChatMessage
from ..utils.youtube_chat_api import YouTubeOfficialChat

logger = logging.getLogger(__name__)


class YouTubeChatMonitor(BaseChatMonitor):
    """
    Monitors YouTube Live Chat using the official YouTube Data API v3.
    
    Provides reliable, official access to live chat without scraping.
    Requires YOUTUBE_API_KEY environment variable.
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
        self._api_chat: Optional[YouTubeOfficialChat] = None
    
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
        """Initialize official YouTube API connection"""
        try:
            self._api_chat = YouTubeOfficialChat(self.channel)
            if self._api_chat.start():
                logger.info(f"Connected to YouTube Live Chat: {self.channel}")
            else:
                logger.error("Failed to connect to YouTube chat")
                raise ConnectionError("YouTube chat connection failed")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube chat: {e}")
            raise
    
    async def _listen(self):
        """Listen for chat messages using official API"""
        if not self._api_chat:
            return
        
        while self._is_running:
            try:
                # Update and get new messages
                messages = self._api_chat.update()
                
                # Convert API messages to ChatMessage format
                for api_msg in messages:
                    chat_msg = ChatMessage(
                        timestamp=api_msg.timestamp,
                        username=api_msg.username,
                        message=api_msg.message,
                        platform="youtube",
                        is_subscriber=api_msg.is_moderator or api_msg.is_owner,
                        is_moderator=api_msg.is_moderator,
                        badges=["owner"] if api_msg.is_owner else (["moderator"] if api_msg.is_moderator else []),
                    )
                    # Emit message to base class (updates all listeners)
                    self._emit_message(chat_msg)
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"YouTube chat error: {e}")
                await asyncio.sleep(1.0)
    
    def get_messages_per_second(self) -> float:
        """
        Calculate messages per second since last call.
        Used by ScoringEngine for chat velocity scoring.
        
        Returns:
            Messages per second rate
        """
        if self._api_chat:
            return self._api_chat.get_messages_per_second()
        return 0.0
    
    async def _disconnect(self):
        """Disconnect from YouTube chat"""
        if self._api_chat:
            self._api_chat.stop()
            self._api_chat = None


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("YOUTUBE_API_KEY"):
        print("YOUTUBE_API_KEY not found. Set it in your .env file")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    video_id = sys.argv[1] if len(sys.argv) > 1 else "jfKfPfyJRdk"  # lofi girl
    
    async def main():
        with YouTubeChatMonitor(video_id) as chat:
            print(f"Monitoring YouTube chat: {video_id}")
            
            while True:
                msg = chat.get_message(timeout=2.0)
                if msg:
                    print(f"[{msg.username}]: {msg.message}")
    
    asyncio.run(main())
