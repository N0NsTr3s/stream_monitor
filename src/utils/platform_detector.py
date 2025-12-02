"""
Platform detector - Identifies streaming platform from URL
"""
import re
from enum import Enum
from typing import Optional, Tuple
from urllib.parse import urlparse


class Platform(Enum):
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    KICK = "kick"
    UNKNOWN = "unknown"


class PlatformDetector:
    """Detects streaming platform and extracts channel/video info from URLs"""
    
    # URL patterns for each platform
    PATTERNS = {
        Platform.TWITCH: [
            r"(?:https?://)?(?:www\.)?twitch\.tv/(\w+)",
            r"(?:https?://)?(?:www\.)?twitch\.tv/videos/(\d+)",
        ],
        Platform.YOUTUBE: [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/live/([\w-]+)",
            r"(?:https?://)?youtu\.be/([\w-]+)",
            r"(?:https?://)?(?:www\.)?youtube\.com/channel/([\w-]+)/live",
            r"(?:https?://)?(?:www\.)?youtube\.com/@([\w-]+)/live",
        ],
        Platform.KICK: [
            r"(?:https?://)?(?:www\.)?kick\.com/(\w+)",
        ],
    }
    
    @classmethod
    def detect(cls, url: str) -> Tuple[Platform, Optional[str]]:
        """
        Detect the platform and extract the channel/video identifier.
        
        Args:
            url: The stream URL to analyze
            
        Returns:
            Tuple of (Platform, identifier) where identifier is channel name or video ID
        """
        url = url.strip()
        
        for platform, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, url, re.IGNORECASE)
                if match:
                    return platform, match.group(1)
        
        return Platform.UNKNOWN, None
    
    @classmethod
    def get_platform_name(cls, url: str) -> str:
        """Get human-readable platform name from URL"""
        platform, _ = cls.detect(url)
        return platform.value.title()
    
    @classmethod
    def is_supported(cls, url: str) -> bool:
        """Check if the URL is from a supported platform"""
        platform, _ = cls.detect(url)
        return platform != Platform.UNKNOWN
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """Normalize URL to standard format"""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url


if __name__ == "__main__":
    # Test URLs
    test_urls = [
        "https://www.twitch.tv/shroud",
        "twitch.tv/ninja",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/live/abc123",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://kick.com/xqc",
        "https://unknown.com/stream",
    ]
    
    for url in test_urls:
        platform, identifier = PlatformDetector.detect(url)
        print(f"{url}")
        print(f"  -> Platform: {platform.value}, ID: {identifier}")
        print()
