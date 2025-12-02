"""
FFmpeg-based stream capture module (OpenCV-free alternative)
Uses FFmpeg subprocess for video/audio capture when OpenCV is unavailable.
"""
import logging
import subprocess
import threading
import time
from typing import Optional, Generator, Tuple
from queue import Queue, Empty
from pathlib import Path

import numpy as np
import streamlink

from ..utils.platform_detector import PlatformDetector, Platform

logger = logging.getLogger(__name__)


class FFmpegStreamCapture:
    """
    Captures video/audio stream using FFmpeg directly.
    Works without OpenCV - pure FFmpeg subprocess approach.
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
        self._video_process: Optional[subprocess.Popen] = None
        self._audio_process: Optional[subprocess.Popen] = None
        self._is_running = False
        
        # Frame settings
        self._width = 1920
        self._height = 1080
        self._fps = 30.0
        
        # Buffers
        self._frame_queue: Queue = Queue(maxsize=30)
        self._audio_queue: Queue = Queue(maxsize=100)
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        
        logger.info(f"FFmpegStreamCapture initialized for {self.platform.value}: {self.channel_id}")
    
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
    
    def _probe_stream(self) -> bool:
        """Probe stream to get actual resolution and fps"""
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
                self._stream_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    self._width = int(parts[0])
                    self._height = int(parts[1])
                    # Parse frame rate (e.g., "30/1" or "30000/1001")
                    fps_parts = parts[2].split('/')
                    if len(fps_parts) == 2:
                        self._fps = float(fps_parts[0]) / float(fps_parts[1])
                    else:
                        self._fps = float(fps_parts[0])
                    
                    logger.info(f"Stream info: {self._width}x{self._height} @ {self._fps:.2f} fps")
                    return True
            
            logger.warning("Could not probe stream, using defaults")
            return True  # Continue anyway with defaults
            
        except FileNotFoundError:
            logger.warning("ffprobe not found, using default resolution")
            return True
        except Exception as e:
            logger.warning(f"Probe error: {e}, using defaults")
            return True
    
    def start(self) -> bool:
        """Start capturing the stream."""
        if self._is_running:
            logger.warning("Stream capture already running")
            return True
        
        # Check for ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            logger.error("FFmpeg not found! Install from https://ffmpeg.org/")
            return False
        
        # Resolve stream URL
        self._stream_url = self._resolve_stream_url()
        if not self._stream_url:
            return False
        
        # Probe stream for resolution
        self._probe_stream()
        
        self._is_running = True
        
        # Start video capture thread
        self._video_thread = threading.Thread(target=self._video_reader, daemon=True)
        self._video_thread.start()
        
        # Start audio capture thread
        self._audio_thread = threading.Thread(target=self._audio_reader, daemon=True)
        self._audio_thread.start()
        
        logger.info("FFmpeg stream capture started successfully")
        return True
    
    def _video_reader(self):
        """Background thread to read video frames from FFmpeg"""
        # FFmpeg command to output raw video frames
        cmd = [
            "ffmpeg",
            "-i", self._stream_url,
            "-an",  # No audio
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-loglevel", "error",
            "pipe:1"
        ]
        
        try:
            self._video_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self._width * self._height * 3 * 2
            )
            
            frame_size = self._width * self._height * 3
            
            while self._is_running and self._video_process.poll() is None:
                raw_frame = self._video_process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    continue
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3))
                
                try:
                    self._frame_queue.put_nowait((time.time(), frame))
                except:
                    # Queue full, drop oldest
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait((time.time(), frame))
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Video reader error: {e}")
        finally:
            if self._video_process:
                self._video_process.terminate()
    
    def _audio_reader(self):
        """Background thread to read audio from FFmpeg"""
        # FFmpeg command to output raw audio
        cmd = [
            "ffmpeg",
            "-i", self._stream_url,
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            "-f", "s16le",
            "-loglevel", "error",
            "pipe:1"
        ]
        
        try:
            self._audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=4096
            )
            
            chunk_size = 4410  # ~100ms of audio at 44.1kHz mono
            
            while self._is_running and self._audio_process.poll() is None:
                data = self._audio_process.stdout.read(chunk_size * 2)  # 2 bytes per sample
                
                if data:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    try:
                        self._audio_queue.put_nowait((time.time(), audio_data))
                    except:
                        try:
                            self._audio_queue.get_nowait()
                            self._audio_queue.put_nowait((time.time(), audio_data))
                        except:
                            pass
            
        except Exception as e:
            logger.error(f"Audio reader error: {e}")
        finally:
            if self._audio_process:
                self._audio_process.terminate()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single video frame."""
        try:
            timestamp, frame = self._frame_queue.get(timeout=1.0)
            return True, frame
        except Empty:
            return False, None
    
    def read_frame_with_timestamp(self) -> Tuple[bool, Optional[float], Optional[np.ndarray]]:
        """Read a frame with its timestamp."""
        try:
            timestamp, frame = self._frame_queue.get(timeout=1.0)
            return True, timestamp, frame
        except Empty:
            return False, None, None
    
    def read_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read audio chunk from buffer."""
        try:
            timestamp, audio = self._audio_queue.get(timeout=timeout)
            return audio
        except Empty:
            return None
    
    def read_audio_with_timestamp(self, timeout: float = 0.1) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Read audio chunk with timestamp."""
        try:
            timestamp, audio = self._audio_queue.get(timeout=timeout)
            return timestamp, audio
        except Empty:
            return None, None
    
    def get_frame_rate(self) -> float:
        """Get the stream's frame rate"""
        return self._fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get stream resolution as (width, height)"""
        return self._width, self._height
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        while self._is_running:
            ret, frame = self.read_frame()
            if ret and frame is not None:
                yield frame
    
    def stop(self):
        """Stop capturing"""
        self._is_running = False
        
        if self._video_process:
            self._video_process.terminate()
            self._video_process = None
        
        if self._audio_process:
            self._audio_process.terminate()
            self._audio_process = None
        
        logger.info("FFmpeg stream capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python ffmpeg_capture.py <stream_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    
    with FFmpegStreamCapture(url) as stream:
        print(f"Resolution: {stream.get_resolution()}")
        print(f"FPS: {stream.get_frame_rate()}")
        
        for i, frame in enumerate(stream.frames()):
            print(f"Frame {i}: {frame.shape}")
            if i >= 10:
                break
