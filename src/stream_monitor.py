"""
Stream Monitor - Main controller that orchestrates all components
"""
import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .config import Config, config
from .monitors.stream_capture import get_stream_capture, CV2_AVAILABLE
from .monitors.chat_twitch import TwitchChatMonitor
from .monitors.chat_youtube import YouTubeChatMonitor
from .monitors.chat_kick import KickChatMonitor
from .analysis.audio_analyzer import AudioAnalyzer
from .analysis.chat_analyzer import ChatAnalyzer
from .analysis.scoring_engine import ScoringEngine, ClipTrigger
from .utils.platform_detector import PlatformDetector, Platform
from .utils.circular_buffer import CircularBuffer

# Use OpenCV clipper if available, otherwise FFmpeg clipper
if CV2_AVAILABLE:
    from .utils.clipper import Clipper
else:
    from .utils.ffmpeg_clipper import FFmpegClipper as Clipper

logger = logging.getLogger(__name__)


class StreamMonitor:
    """
    Main controller for stream monitoring and clip creation.
    
    Orchestrates:
    - Stream capture (video/audio via streamlink)
    - Chat monitoring (platform-specific)
    - Signal analysis (audio RMS, chat velocity)
    - Scoring engine (combined importance score)
    - Clip creation (buffer + save)
    """
    
    def __init__(self, url: str, cfg: Optional[Config] = None):
        """
        Initialize stream monitor.
        
        Args:
            url: Stream URL (Twitch, YouTube, or Kick)
            cfg: Optional configuration, uses global config if not provided
        """
        self.url = url
        self.config = cfg or config
        
        # Detect platform
        self.platform, self.channel_id = PlatformDetector.detect(url)
        
        if self.platform == Platform.UNKNOWN:
            raise ValueError(f"Unsupported platform for URL: {url}")
        
        logger.info(f"Detected platform: {self.platform.value}, channel: {self.channel_id}")
        
        # Components (initialized in start())
        self._stream: Optional[StreamCapture] = None
        self._chat_monitor = None
        self._audio_analyzer: Optional[AudioAnalyzer] = None
        self._chat_analyzer: Optional[ChatAnalyzer] = None
        self._scoring_engine: Optional[ScoringEngine] = None
        self._buffer: Optional[CircularBuffer] = None
        self._clipper: Optional[Clipper] = None
        
        # State
        self._is_running = False
        self._threads = []
        self._shutdown_event = threading.Event()
        
        # Stats
        self._frames_processed = 0
        self._clips_created = 0
        self._start_time: Optional[float] = None
    
    def _create_chat_monitor(self):
        """Create platform-specific chat monitor"""
        if self.platform == Platform.TWITCH:
            return TwitchChatMonitor(
                self.channel_id,
                oauth_token=self.config.twitch.oauth_token
            )
        elif self.platform == Platform.YOUTUBE:
            return YouTubeChatMonitor(self.channel_id)
        elif self.platform == Platform.KICK:
            return KickChatMonitor(self.channel_id)
        else:
            return None
    
    def _initialize_components(self):
        """Initialize all components"""
        # Stream capture - auto-selects OpenCV or FFmpeg based on availability
        self._stream = get_stream_capture(
            self.url,
            quality=self.config.stream.quality
        )
        
        # Chat monitor
        self._chat_monitor = self._create_chat_monitor()
        
        # Analyzers
        self._audio_analyzer = AudioAnalyzer(
            baseline_rms=self.config.scoring.audio_rms_baseline,
            spike_multiplier=self.config.scoring.audio_spike_multiplier
        )
        
        self._chat_analyzer = ChatAnalyzer(
            window_seconds=self.config.scoring.chat_window_seconds,
            baseline_mps=self.config.scoring.chat_baseline_mps,
            spike_multiplier=self.config.scoring.chat_spike_multiplier
        )
        
        # Scoring engine
        self._scoring_engine = ScoringEngine(
            audio_weight=self.config.scoring.audio_weight,
            chat_weight=self.config.scoring.chat_weight,
            trigger_threshold=self.config.scoring.trigger_threshold,
            release_threshold=self.config.scoring.release_threshold,
            pre_roll_seconds=self.config.clip.pre_roll_seconds,
            post_roll_seconds=self.config.clip.post_roll_seconds
        )
        
        # Register clip callbacks
        self._scoring_engine.on_clip_start(self._on_clip_start)
        self._scoring_engine.on_clip_end(self._on_clip_end)
        
        # Buffer
        fps = 30.0  # Will be updated from stream
        self._buffer = CircularBuffer(
            max_seconds=self.config.stream.buffer_seconds,
            fps=fps
        )
        
        # Clipper
        self._clipper = Clipper(
            output_dir=self.config.clip.output_dir,
            fps=fps
        )
    
    def _on_clip_start(self, trigger: ClipTrigger):
        """Called when a clip recording should start"""
        logger.info(f"🔴 Clip triggered! Reason: {trigger.reason}, Score: {trigger.peak_score:.2f}")
    
    def _on_clip_end(self, trigger: ClipTrigger):
        """Called when a clip recording should end"""
        if not self._buffer or not self._clipper:
            return
        
        # Save clip asynchronously
        self._clipper.save_clip_async(
            self._buffer,
            trigger.start_time,
            trigger.end_time,
            prefix=f"{self.platform.value}_{self.channel_id}",
            reason=trigger.reason,
            callback=self._on_clip_saved
        )
    
    def _on_clip_saved(self, path: Optional[Path]):
        """Called when a clip has been saved"""
        if path:
            self._clips_created += 1
            logger.info(f"✅ Clip saved: {path}")
        else:
            logger.warning("❌ Failed to save clip")
    
    def _video_loop(self):
        """Main video processing loop"""
        logger.info("Video processing loop started")
        
        frame_interval = 1.0 / 30.0  # Target 30fps processing
        last_frame_time = time.time()
        
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Read frame
                ret, frame = self._stream.read_frame()
                
                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                
                # Add to buffer
                self._buffer.add_video_frame(frame, current_time)
                
                self._frames_processed += 1
                
                # Rate limit processing
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
                last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Video loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Video processing loop stopped")
    
    def _audio_loop(self):
        """Audio processing loop"""
        logger.info("Audio processing loop started")
        
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Read audio chunk
                audio_data = self._stream.read_audio(timeout=0.1)
                
                if audio_data is not None:
                    current_time = time.time()
                    
                    # Add to buffer
                    self._buffer.add_audio_chunk(audio_data, current_time)
                    
                    # Analyze audio
                    metrics = self._audio_analyzer.analyze(audio_data)
                    
                    # Update scoring engine
                    self._scoring_engine.update(audio_score=metrics.normalized_score)
                
            except Exception as e:
                logger.error(f"Audio loop error: {e}")
                time.sleep(0.1)
        
        logger.info("Audio processing loop stopped")
    
    def _chat_loop(self):
        """Chat processing loop"""
        logger.info("Chat processing loop started")
        
        while self._is_running and not self._shutdown_event.is_set():
            try:
                # Get chat messages
                messages = self._chat_monitor.get_all_messages()
                
                for msg in messages:
                    # Add to analyzer
                    self._chat_analyzer.add_message(
                        msg.username,
                        msg.message,
                        msg.emote_count
                    )
                
                # Analyze chat
                metrics = self._chat_analyzer.analyze()
                
                # Update scoring engine
                self._scoring_engine.update(chat_score=metrics.normalized_score)
                
                time.sleep(0.5)  # Chat analysis rate
                
            except Exception as e:
                logger.error(f"Chat loop error: {e}")
                time.sleep(1.0)
        
        logger.info("Chat processing loop stopped")
    
    def _status_loop(self):
        """Status display loop"""
        while self._is_running and not self._shutdown_event.is_set():
            try:
                elapsed = time.time() - self._start_time
                fps = self._frames_processed / max(elapsed, 1)
                
                status = (
                    f"\r[{self.platform.value.upper()}] "
                    f"Score: {self._scoring_engine.current_score:.2f} | "
                    f"State: {self._scoring_engine.state.value:10} | "
                    f"Buffer: {self._buffer.get_duration():.1f}s | "
                    f"Clips: {self._clips_created} | "
                    f"FPS: {fps:.1f} | "
                    f"Runtime: {elapsed:.0f}s"
                )
                
                print(status, end='', flush=True)
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Status loop error: {e}")
                time.sleep(1.0)
    
    def start(self) -> bool:
        """
        Start monitoring the stream.
        
        Returns:
            True if started successfully
        """
        if self._is_running:
            logger.warning("Monitor already running")
            return True
        
        logger.info(f"Starting stream monitor for {self.url}")
        
        # Initialize components
        self._initialize_components()
        
        # Start stream capture
        if not self._stream.start():
            logger.error("Failed to start stream capture")
            return False
        
        # Update FPS from stream
        actual_fps = self._stream.get_frame_rate()
        self._buffer.fps = actual_fps
        self._clipper.fps = actual_fps
        self._clipper.resolution = self._stream.get_resolution()
        
        logger.info(f"Stream: {self._stream.get_resolution()}, {actual_fps} FPS")
        
        # Start chat monitor
        if self._chat_monitor:
            self._chat_monitor.start()
        
        self._is_running = True
        self._start_time = time.time()
        self._shutdown_event.clear()
        
        # Start processing threads
        threads = [
            ("video", self._video_loop),
            ("audio", self._audio_loop),
            ("chat", self._chat_loop),
            ("status", self._status_loop),
        ]
        
        for name, target in threads:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        
        logger.info("Stream monitor started successfully")
        return True
    
    def stop(self):
        """Stop monitoring"""
        if not self._is_running:
            return
        
        logger.info("Stopping stream monitor...")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Stop components
        if self._stream:
            self._stream.stop()
        
        if self._chat_monitor:
            self._chat_monitor.stop()
        
        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=5.0)
        
        self._threads.clear()
        
        print()  # New line after status
        logger.info(f"Stream monitor stopped. Clips created: {self._clips_created}")
    
    def run(self):
        """Run the monitor until interrupted"""
        if not self.start():
            return
        
        try:
            while self._is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n")
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Stream Monitor - Automatically clip exciting moments from live streams"
    )
    
    parser.add_argument(
        "url",
        help="Stream URL (Twitch, YouTube, or Kick)"
    )
    
    parser.add_argument(
        "-q", "--quality",
        default="best",
        help="Stream quality (best, worst, 720p, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="clips",
        help="Output directory for clips"
    )
    
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.7,
        help="Score threshold to trigger clip (0.0-1.0)"
    )
    
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=0.3,
        help="Score threshold to end clip (0.0-1.0)"
    )
    
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=3.0,
        help="Seconds before trigger to include"
    )
    
    parser.add_argument(
        "--post-roll",
        type=float,
        default=5.0,
        help="Seconds after release to include"
    )
    
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=0.4,
        help="Weight for audio signal (0.0-1.0)"
    )
    
    parser.add_argument(
        "--chat-weight",
        type=float,
        default=0.6,
        help="Weight for chat signal (0.0-1.0)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Update config from args
    config.stream.quality = args.quality
    config.clip.output_dir = Path(args.output)
    config.scoring.trigger_threshold = args.trigger_threshold
    config.scoring.release_threshold = args.release_threshold
    config.clip.pre_roll_seconds = args.pre_roll
    config.clip.post_roll_seconds = args.post_roll
    config.scoring.audio_weight = args.audio_weight
    config.scoring.chat_weight = args.chat_weight
    
    # Validate URL
    if not PlatformDetector.is_supported(args.url):
        print(f"Error: Unsupported platform for URL: {args.url}")
        print("Supported platforms: Twitch, YouTube, Kick")
        sys.exit(1)
    
    # Run monitor
    print(f"Starting Stream Monitor for: {args.url}")
    print(f"Output directory: {config.clip.output_dir}")
    print("Press Ctrl+C to stop\n")
    
    monitor = StreamMonitor(args.url)
    monitor.run()


if __name__ == "__main__":
    main()
