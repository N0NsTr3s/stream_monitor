"""
Configuration settings for Stream Monitor
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class StreamConfig:
    """Configuration for stream capture"""
    quality: str = "best"  # streamlink quality option
    buffer_seconds: int = 30  # seconds to keep in circular buffer
    

@dataclass
class ClipConfig:
    """Configuration for clip generation"""
    pre_roll_seconds: float = 3.0  # seconds before trigger to include
    post_roll_seconds: float = 5.0  # seconds after score drops to include
    output_dir: Path = field(default_factory=lambda: Path("clips"))
    format: str = "mp4"
    

@dataclass
class ScoringConfig:
    """Configuration for the scoring engine"""
    # Weights for different signals (should sum to 1.0)
    audio_weight: float = 0.4
    chat_weight: float = 0.6
    
    # Thresholds
    trigger_threshold: float = 0.7  # Score to start recording
    release_threshold: float = 0.3  # Score to stop recording
    
    # Audio settings
    audio_rms_baseline: float = 0.1  # Baseline RMS for normalization
    audio_spike_multiplier: float = 3.0  # How much above baseline = max score
    
    # Chat settings
    chat_window_seconds: float = 5.0  # Window for measuring chat velocity
    chat_baseline_mps: float = 1.0  # Baseline messages per second
    chat_spike_multiplier: float = 5.0  # How much above baseline = max score


@dataclass
class TwitchConfig:
    """Twitch API configuration"""
    client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("TWITCH_CLIENT_ID")
    )
    client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("TWITCH_CLIENT_SECRET")
    )
    oauth_token: Optional[str] = field(
        default_factory=lambda: os.getenv("TWITCH_OAUTH_TOKEN")
    )


@dataclass
class YouTubeConfig:
    """YouTube API configuration"""
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("YOUTUBE_API_KEY")
    )


@dataclass
class KickConfig:
    """Kick configuration"""
    # Kick uses Pusher WebSocket, usually no auth needed for public chat
    pusher_key: str = "eb1d5f283081a78b932c"  # Kick's public Pusher key
    pusher_cluster: str = "us2"


@dataclass
class Config:
    """Main configuration container"""
    stream: StreamConfig = field(default_factory=StreamConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    kick: KickConfig = field(default_factory=KickConfig)
    
    def __post_init__(self):
        # Ensure output directory exists
        self.clip.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
