"""
Stream capture module using streamlink
Resolves stream URLs and provides access to video/audio data
"""
import logging
import subprocess
import threading
import time
from typing import Optional, Generator, Tuple, Deque
from queue import Queue, Empty

import streamlink
import numpy as np
import shutil

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
    Provides raw stream chunks for buffering and decoded frames for analysis.
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
        self._stream_fd = None # Streamlink file descriptor
        
        # Processes for decoding
        self._video_process: Optional[subprocess.Popen] = None
        self._audio_process: Optional[subprocess.Popen] = None
        
        self._is_running = False
        
        # Check for ffmpeg
        self._ffmpeg_path = shutil.which("ffmpeg")
        if not self._ffmpeg_path:
            logger.warning("FFmpeg not found in PATH. Capture will be limited.")
        
        # Buffers
        self._raw_queue: Queue = Queue(maxsize=100) # For raw chunks
        self._audio_queue: Queue = Queue(maxsize=100) # For decoded audio
        self._video_queue: Queue = Queue(maxsize=10) # For decoded video frames
        
        # Threads
        self._stream_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._video_thread: Optional[threading.Thread] = None
        
        # Stats
        self._fps = 30.0
        self._width = 1920
        self._height = 1080
        
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
        
        if not self._ffmpeg_path:
            logger.error("FFmpeg is required for this capture mode")
            return False
        
        # Resolve stream URL
        self._stream_url = self._resolve_stream_url()
        if not self._stream_url:
            return False
        
        # Open stream with streamlink
        try:
            streams = streamlink.streams(self.url)
            if self.quality in streams:
                self._stream_fd = streams[self.quality].open()
            elif "best" in streams:
                self._stream_fd = streams["best"].open()
            else:
                self._stream_fd = streams[list(streams.keys())[0]].open()
        except Exception as e:
            logger.error(f"Failed to open stream: {e}")
            return False

        self._is_running = True
        
        # Start decoding processes
        self._start_decoders()
        
        # Start stream reader thread
        self._stream_thread = threading.Thread(target=self._stream_reader, daemon=True)
        self._stream_thread.start()
        
        logger.info("Stream capture started successfully")
        return True
    
    def _start_decoders(self):
        """Start FFmpeg processes for video and audio decoding"""
        
        # Video Decoder: Reads from pipe, outputs raw video frames
        # We scale to 640x360 for analysis to save CPU
        video_cmd = [
            self._ffmpeg_path,
            "-i", "pipe:0",
            "-vf", "fps=15,scale=640:360", # Low FPS and resolution for analysis
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "pipe:1"
        ]
        
        self._video_process = subprocess.Popen(
            video_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7 # Large buffer
        )
        
        # Audio Decoder: Reads from pipe, outputs raw audio samples
        audio_cmd = [
            self._ffmpeg_path,
            "-i", "pipe:0",
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            "-loglevel", "error",
            "pipe:1"
        ]
        
        self._audio_process = subprocess.Popen(
            audio_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**6
        )
        
        # Start reader threads
        self._video_thread = threading.Thread(target=self._video_reader, daemon=True)
        self._video_thread.start()
        
        self._audio_thread = threading.Thread(target=self._audio_reader, daemon=True)
        self._audio_thread.start()

    def _stream_reader(self):
        """Reads raw chunks from stream and distributes them"""
        chunk_size = 32 * 1024 # 32KB chunks
        
        while self._is_running and self._stream_fd:
            try:
                data = self._stream_fd.read(chunk_size)
                if not data:
                    logger.warning("Stream ended")
                    break
                
                # 1. Send to raw buffer (for clipping)
                try:
                    self._raw_queue.put_nowait(data)
                except:
                    pass # Drop if full (shouldn't happen if consumer is fast)
                
                # 2. Send to Video Decoder
                if self._video_process and self._video_process.stdin:
                    try:
                        self._video_process.stdin.write(data)
                        self._video_process.stdin.flush()
                    except Exception:
                        pass
                
                # 3. Send to Audio Decoder
                if self._audio_process and self._audio_process.stdin:
                    try:
                        self._audio_process.stdin.write(data)
                        self._audio_process.stdin.flush()
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Stream reader error: {e}")
                break
        
        self._is_running = False

    def _video_reader(self):
        """Reads decoded video frames for analysis"""
        # 640x360 * 3 bytes (BGR)
        frame_size = 640 * 360 * 3
        
        while self._is_running and self._video_process:
            try:
                data = self._video_process.stdout.read(frame_size)
                if len(data) == frame_size:
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((360, 640, 3))
                    
                    # Clear queue if full to always have latest frame
                    if self._video_queue.full():
                        try:
                            self._video_queue.get_nowait()
                        except:
                            pass
                    
                    self._video_queue.put(frame)
                else:
                    if not data:
                        break
            except Exception as e:
                logger.error(f"Video reader error: {e}")
                break

    def _audio_reader(self):
        """Reads decoded audio samples for analysis"""
        chunk_size = 4410 * 2 # ~100ms
        
        while self._is_running and self._audio_process:
            try:
                data = self._audio_process.stdout.read(chunk_size)
                if data:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if self._audio_queue.full():
                        try:
                            self._audio_queue.get_nowait()
                        except:
                            pass
                            
                    self._audio_queue.put(audio_data)
                else:
                    break
            except Exception as e:
                logger.error(f"Audio reader error: {e}")
                break

    def read_chunk(self) -> Optional[bytes]:
        """Read a raw stream chunk"""
        try:
            return self._raw_queue.get_nowait()
        except Empty:
            return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a decoded video frame (low res for analysis)"""
        try:
            frame = self._video_queue.get_nowait()
            return True, frame
        except Empty:
            return True, None # Return True to keep loop alive, but None frame

    def read_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read decoded audio chunk"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_rate(self) -> float:
        return 30.0 # Approximate
    
    def get_resolution(self) -> Tuple[int, int]:
        return 1920, 1080 # Approximate
    
    def stop(self):
        """Stop capturing"""
        self._is_running = False
        
        if self._stream_fd:
            try:
                self._stream_fd.close()
            except:
                pass
        
        if self._video_process:
            self._video_process.terminate()
        
        if self._audio_process:
            self._audio_process.terminate()
        
        logger.info("Stream capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

def get_stream_capture(url: str, quality: str = "best"):
    return StreamCapture(url, quality)
