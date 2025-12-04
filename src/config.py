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
    buffer_seconds: int = 30  # Restored to 30s since we use disk buffer now
    

@dataclass
class ClipConfig:
    """Configuration for clip generation"""
    pre_roll_seconds: float = 5.0  # seconds before trigger to include
    post_roll_seconds: float = 5.0  # seconds after score drops to include (Increased from 3.0)
    min_duration: float = 10.0  # Minimum clip duration in seconds (Increased from 8.0)
    output_dir: Path = field(default_factory=lambda: Path("clips"))
    format: str = "mp4"
    fps: float = 60.0  # Target FPS for clips
    

@dataclass
class ScoringConfig:
    """Configuration for the scoring engine (Z-Score based)"""
    # Weights for different signals (should sum to ~1.0)
    # Audio and Chat valued more than Video (webcams are noisy)
    audio_weight: float = 0.5
    chat_weight: float = 0.4
    video_weight: float = 0.1
    
    # Z-Score Thresholds (adaptive to stream's baseline)
    # 2.0 = Sensitive (Clips often)
    # 3.0 = Standard (Good moments)
    # 4.0 = Strict (Only massive spikes)
    trigger_threshold: float = 3.0  # Combined Z-Score to start recording
    release_threshold: float = 1.0  # Combined Z-Score to stop recording
    
    # Calibration - time to gather baseline data for Z-scores
    calibration_seconds: float = 60.0  # 60 seconds to build rolling average
    
    # Audio settings
    audio_rms_baseline: float = 0.12  # Higher baseline = needs louder audio to trigger
    audio_spike_multiplier: float = 3.5  # Higher multiplier = needs bigger spikes
    
    # Chat settings
    chat_window_seconds: float = 5.0  # Window for measuring chat velocity
    chat_baseline_mps: float = 2.0  # Baseline messages per second (Increased from 1.0)
    chat_spike_multiplier: float = 5.0  # How much above baseline = max score
    chat_latency_seconds: float = 5.0  # Delay chat score to match video latency

    # Video settings

    # Video settings
    video_motion_baseline: float = 17.0  # Higher baseline to ignore camera pans (was 12.0)
    video_spike_multiplier: float = 5.0  # Higher multiplier to require more chaos (was 4.0)


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
