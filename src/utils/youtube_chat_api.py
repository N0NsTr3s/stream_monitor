"""
Official YouTube API Chat Monitor
Uses the YouTube Data API v3 for reliable live chat access.
"""
import time
import requests
import logging
import os
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class YouTubeChatMessage:
    """Represents a YouTube chat message"""
    username: str
    message: str
    timestamp: float
    is_moderator: bool = False
    is_owner: bool = False


class YouTubeOfficialChat:
    """
    Monitor YouTube Live Chat using the official Google YouTube Data API v3.
    
    Provides reliable access to live chat messages without scraping.
    Respects API rate limits and polling intervals sent by YouTube.
    """
    
    def __init__(self, video_id: str):
        """
        Initialize YouTube official chat monitor.
        
        Args:
            video_id: YouTube video/stream ID
        """
        self.video_id = video_id
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        
        if not self.api_key:
            logger.error("YOUTUBE_API_KEY not found in environment variables")
            raise ValueError("YOUTUBE_API_KEY required for official YouTube chat API")
        
        self.live_chat_id: Optional[str] = None
        self.next_page_token: Optional[str] = None
        self.next_poll_time: float = 0.0
        self.is_ready: bool = False
        
        # Message tracking
        self.msg_count_window: int = 0
        self.last_tick: float = time.time()
        self.messages_buffer: List[YouTubeChatMessage] = []
        # Sample & hold: current messages-per-second computed from last batch
        self.current_mps: float = 0.0
        # Last time we pushed/observed the held value (for external samplers)
        self.last_update_push: float = time.time()
    
    def start(self) -> bool:
        """
        Step 1: Get the Chat ID from the Video ID.
        
        Returns:
            True if connected successfully, False otherwise
        """
        logger.info(f"Fetching Live Chat ID for: {self.video_id}")
        
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "liveStreamingDetails",
            "id": self.video_id,
            "key": self.api_key
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if "items" not in data or len(data["items"]) == 0:
                logger.error("Video not found in YouTube API")
                return False

            details = data["items"][0].get("liveStreamingDetails", {})
            self.live_chat_id = details.get("activeLiveChatId")
            
            if self.live_chat_id:
                logger.info(f"Connected to YouTube chat. Chat ID: {self.live_chat_id}")
                self.is_ready = True
                # Initial fetch to get the first page token
                self._fetch_messages(initial=True)
                return True
            else:
                logger.warning("Video is not live or chat is disabled")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"YouTube API connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to YouTube chat: {e}")
            return False
    
    def update(self) -> List[YouTubeChatMessage]:
        """
        Call this regularly in your main loop.
        Respects YouTube's pollingIntervalMillis to avoid rate limiting.
        
        Returns:
            List of new messages since last update
        """
        if not self.is_ready:
            return []

        current_time = time.time()
        
        # Only fetch if the wait time has passed
        if current_time >= self.next_poll_time:
            self._fetch_messages()
        
        # Return buffered messages and clear
        messages = self.messages_buffer[:]
        self.messages_buffer.clear()
        
        return messages
    
    def _fetch_messages(self, initial: bool = False):
        """
        Fetch new messages from the live chat.
        
        Args:
            initial: If True, don't process messages (just get page token)
        """
        url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
        params = {
            "liveChatId": self.live_chat_id,
            "part": "snippet,authorDetails",
            "key": self.api_key
        }
        
        # Use page token to get only new messages
        if self.next_page_token:
            params["pageToken"] = self.next_page_token

        try:
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                # Update token for next fetch
                self.next_page_token = data.get("nextPageToken")
                
                # Update wait time (CRITICAL - respects YouTube's rate limits)
                # API sends milliseconds, convert to seconds + buffer
                wait_ms = data.get("pollingIntervalMillis", 5000)
                self.next_poll_time = time.time() + (wait_ms / 1000.0) + 0.5
                
                # Process messages (skip on initial fetch)
                items = data.get("items", [])
                # Calculate polling interval from API (milliseconds -> seconds)
                wait_ms = data.get("pollingIntervalMillis", 5000)
                polling_interval_sec = wait_ms / 1000.0

                # Update token and next poll time
                self.next_page_token = data.get("nextPageToken")
                self.next_poll_time = time.time() + polling_interval_sec + 0.5

                if not initial:
                    # Buffer individual messages for emission
                    for item in items:
                        msg = self._parse_message(item)
                        if msg:
                            self.messages_buffer.append(msg)
                            self.msg_count_window += 1

                    # SAMPLE & HOLD: compute messages-per-second based on the batch
                    count = len(items)
                    if polling_interval_sec > 0:
                        self.current_mps = count / polling_interval_sec
                    else:
                        self.current_mps = 0.0

                    logger.debug(f"Fetched {len(items)} chat messages; batch MPS={self.current_mps:.2f}")
            
            elif resp.status_code == 403:
                logger.error("403 Forbidden - YouTube API quota exceeded or chat closed")
                self.is_ready = False
            else:
                logger.warning(f"YouTube API returned status {resp.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning("YouTube API request timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"YouTube API fetch error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching messages: {e}")
    
    def _parse_message(self, item: dict) -> Optional[YouTubeChatMessage]:
        """
        Parse a YouTube API message item into our ChatMessage format.
        
        Args:
            item: Message item from YouTube API response
            
        Returns:
            YouTubeChatMessage or None if parsing fails
        """
        try:
            snippet = item.get("snippet", {})
            author_details = item.get("authorDetails", {})
            
            return YouTubeChatMessage(
                username=author_details.get("displayName", "Unknown"),
                message=snippet.get("displayMessage", ""),
                timestamp=time.time(),
                is_moderator=author_details.get("isChatModerator", False),
                is_owner=author_details.get("isChatOwner", False)
            )
        except Exception as e:
            logger.debug(f"Failed to parse message: {e}")
            return None
    
    def get_messages_per_second(self) -> float:
        """
        Calculate messages per second since last call.
        Used for chat scoring.
        
        Returns:
            Messages per second rate
        """
        # Return the last computed batch MPS (sample-and-hold).
        # This lets external code poll more frequently and receive a
        # stable MPS value until the next YouTube batch updates it.
        return float(self.current_mps)
    
    def stop(self):
        """Stop monitoring"""
        self.is_ready = False
        logger.info("YouTube chat monitoring stopped")
