"""
Stream capture module using streamlink
Resolves stream URLs and provides access to video/audio data
"""
import logging
import subprocess
import threading
from typing import Optional, Generator, Tuple
from queue import Queue, Empty

import streamlink
import numpy as np

# Try to import cv2, provide fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from ..utils.platform_detector import PlatformDetector, Platform

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Captures video/audio stream using streamlink.
    Provides frames via OpenCV and raw audio via FFmpeg.
    """
    
    def __init__(self, url: str, quality: str = "best"):
        """
        Initialize stream capture.
        
        Args:
            url: Stream URL (Twitch, YouTube, or Kick)
            quality: Stream quality (best, worst, 720p, etc.)
        """
        self.url = PlatformDetector.normalize_url(url)
        self.quality = quality
        self.platform, self.channel_id = PlatformDetector.detect(self.url)
        
        self._stream_url: Optional[str] = None
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._audio_process: Optional[subprocess.Popen] = None
        self._is_running = False
        
        # Audio buffer
        self._audio_queue: Queue = Queue(maxsize=100)
        self._audio_thread: Optional[threading.Thread] = None
        
        logger.info(f"StreamCapture initialized for {self.platform.value}: {self.channel_id}")
    
    def _resolve_stream_url(self) -> Optional[str]:
        """Resolve the stream URL using streamlink"""
        try:
            streams = streamlink.streams(self.url)
            
            if not streams:
                logger.error(f"No streams found for {self.url}")
                return None
            
            available = list(streams.keys())
            logger.info(f"Available qualities: {available}")
            
            # Try requested quality, fall back to best
            if self.quality in streams:
                stream = streams[self.quality]
            elif "best" in streams:
                stream = streams["best"]
            else:
                stream = streams[available[0]]
            
            stream_url = stream.url
            logger.info(f"Resolved stream URL: {stream_url[:100]}...")
            return stream_url
            
        except streamlink.StreamlinkError as e:
            logger.error(f"Streamlink error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error resolving stream: {e}")
            return None
    
    def start(self) -> bool:
        """
        Start capturing the stream.
        
        Returns:
            True if successfully started, False otherwise
        """
        if self._is_running:
            logger.warning("Stream capture already running")
            return True
        
        if not CV2_AVAILABLE:
            logger.error("OpenCV (cv2) not available - install with: pip install opencv-python")
            return False
        
        # Resolve stream URL
        self._stream_url = self._resolve_stream_url()
        if not self._stream_url:
            return False
        
        # Initialize video capture
        self._video_capture = cv2.VideoCapture(self._stream_url)
        if not self._video_capture.isOpened():
            logger.error("Failed to open video capture")
            return False
        
        # Start audio capture in background thread
        self._start_audio_capture()
        
        self._is_running = True
        logger.info("Stream capture started successfully")
        return True
    
    def _start_audio_capture(self):
        """Start FFmpeg process for audio capture"""
        # FFmpeg command to extract audio as raw PCM
        cmd = [
            "ffmpeg",
            "-i", self._stream_url,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "44100",  # 44.1kHz sample rate
            "-ac", "1",  # Mono
            "-f", "s16le",  # Raw PCM format
            "-loglevel", "quiet",
            "pipe:1"  # Output to stdout
        ]
        
        try:
            self._audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=4096
            )
            
            # Start thread to read audio data
            self._audio_thread = threading.Thread(target=self._audio_reader, daemon=True)
            self._audio_thread.start()
            logger.info("Audio capture started")
            
        except FileNotFoundError:
            logger.warning("FFmpeg not found - audio capture disabled")
            self._audio_process = None
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            self._audio_process = None
    
    def _audio_reader(self):
        """Background thread to read audio data from FFmpeg"""
        chunk_size = 4410  # ~100ms of audio at 44.1kHz mono
        
        while self._is_running and self._audio_process:
            try:
                data = self._audio_process.stdout.read(chunk_size * 2)  # 2 bytes per sample
                if data:
                    # Convert to numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    try:
                        self._audio_queue.put_nowait(audio_data)
                    except:
                        # Queue full, drop oldest
                        try:
                            self._audio_queue.get_nowait()
                            self._audio_queue.put_nowait(audio_data)
                        except:
                            pass
                else:
                    break
            except Exception as e:
                logger.error(f"Audio reader error: {e}")
                break
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single video frame.
        
        Returns:
            Tuple of (success, frame) where frame is BGR numpy array
        """
        if not self._video_capture or not self._is_running:
            return False, None
        
        ret, frame = self._video_capture.read()
        return ret, frame
    
    def read_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Read audio chunk from buffer.
        
        Args:
            timeout: How long to wait for audio data
            
        Returns:
            Audio samples as float32 numpy array, or None if no data
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_rate(self) -> float:
        """Get the stream's frame rate"""
        if self._video_capture:
            return self._video_capture.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get stream resolution as (width, height)"""
        if self._video_capture:
            w = int(self._video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 1920, 1080
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        while self._is_running:
            ret, frame = self.read_frame()
            if ret and frame is not None:
                yield frame
            else:
                break
    
    def stop(self):
        """Stop capturing"""
        self._is_running = False
        
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        
        if self._audio_process:
            self._audio_process.terminate()
            self._audio_process = None
        
        logger.info("Stream capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def get_stream_capture(url: str, quality: str = "best"):
    """
    Factory function to get the best available stream capture implementation.
    
    Returns OpenCV-based capture if available, otherwise FFmpeg-based capture.
    
    Args:
        url: Stream URL
        quality: Stream quality
        
    Returns:
        StreamCapture or FFmpegStreamCapture instance
    """
    if CV2_AVAILABLE:
        logger.info("Using OpenCV-based stream capture")
        return StreamCapture(url, quality)
    else:
        logger.info("OpenCV not available, using FFmpeg-based stream capture")
        from .ffmpeg_capture import FFmpegStreamCapture
        return FFmpegStreamCapture(url, quality)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python stream_capture.py <stream_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    # Use factory function to get best available implementation
    with get_stream_capture(url) as stream:
        print(f"Resolution: {stream.get_resolution()}")
        print(f"FPS: {stream.get_frame_rate()}")
        
        for i, frame in enumerate(stream.frames()):
            print(f"Frame {i}: {frame.shape}")
            if i >= 10:
                break
