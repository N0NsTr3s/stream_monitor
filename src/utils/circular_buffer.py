"""
Circular Buffer - Maintains a rolling buffer of stream data
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Deque, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferFrame:
    """A single frame in the buffer"""
    timestamp: float
    video_frame: Optional[np.ndarray] = None
    audio_chunk: Optional[np.ndarray] = None


class CircularBuffer:
    """
    Circular buffer for storing recent stream data.
    
    Maintains a rolling buffer of video frames and audio chunks
    that can be dumped to create clips with pre-roll.
    """
    
    def __init__(
        self,
        max_seconds: float = 30.0,
        fps: float = 30.0,
        audio_chunks_per_second: float = 10.0
    ):
        """
        Initialize circular buffer.
        
        Args:
            max_seconds: Maximum duration to keep in buffer
            fps: Expected video frame rate
            audio_chunks_per_second: Expected audio chunks per second
        """
        self.max_seconds = max_seconds
        self.fps = fps
        self.audio_chunks_per_second = audio_chunks_per_second
        
        # Calculate buffer sizes
        max_video_frames = int(max_seconds * fps * 1.5)  # 1.5x for safety
        max_audio_chunks = int(max_seconds * audio_chunks_per_second * 1.5)
        
        # Separate buffers for video and audio (different rates)
        self._video_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_video_frames)
        self._audio_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_audio_chunks)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Stats
        self._video_count = 0
        self._audio_count = 0
        self._start_time: Optional[float] = None
    
    def add_video_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """
        Add a video frame to the buffer.
        
        Args:
            frame: Video frame as numpy array (BGR)
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            
            self._video_buffer.append((timestamp, frame.copy()))
            self._video_count += 1
    
    def add_audio_chunk(self, chunk: np.ndarray, timestamp: Optional[float] = None):
        """
        Add an audio chunk to the buffer.
        
        Args:
            chunk: Audio samples as numpy array
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            
            self._audio_buffer.append((timestamp, chunk.copy()))
            self._audio_count += 1
    
    def get_video_frames(
        self,
        start_time: float,
        end_time: float
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Get video frames in a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of (timestamp, frame) tuples
        """
        with self._lock:
            return [
                (ts, frame) for ts, frame in self._video_buffer
                if start_time <= ts <= end_time
            ]
    
    def get_audio_chunks(
        self,
        start_time: float,
        end_time: float
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Get audio chunks in a time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of (timestamp, chunk) tuples
        """
        with self._lock:
            return [
                (ts, chunk) for ts, chunk in self._audio_buffer
                if start_time <= ts <= end_time
            ]
    
    def get_buffer_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the time range currently in buffer"""
        with self._lock:
            if not self._video_buffer and not self._audio_buffer:
                return None, None
            
            # Find earliest and latest timestamps
            timestamps = []
            if self._video_buffer:
                timestamps.extend([ts for ts, _ in self._video_buffer])
            if self._audio_buffer:
                timestamps.extend([ts for ts, _ in self._audio_buffer])
            
            if timestamps:
                return min(timestamps), max(timestamps)
            return None, None
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        start, end = self.get_buffer_range()
        if start and end:
            return end - start
        return 0.0
    
    @property
    def video_frame_count(self) -> int:
        """Get number of video frames in buffer"""
        with self._lock:
            return len(self._video_buffer)
    
    @property
    def audio_chunk_count(self) -> int:
        """Get number of audio chunks in buffer"""
        with self._lock:
            return len(self._audio_buffer)
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._video_buffer.clear()
            self._audio_buffer.clear()
            self._start_time = None
    
    def __len__(self):
        """Get total items in buffer"""
        return self.video_frame_count + self.audio_chunk_count


if __name__ == "__main__":
    # Test the buffer
    buffer = CircularBuffer(max_seconds=5.0)
    
    # Simulate adding frames
    for i in range(150):  # 5 seconds at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buffer.add_video_frame(frame)
        time.sleep(0.033)  # ~30fps
    
    print(f"Video frames: {buffer.video_frame_count}")
    print(f"Duration: {buffer.get_duration():.2f}s")
    print(f"Range: {buffer.get_buffer_range()}")
