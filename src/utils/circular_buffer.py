"""
Circular Buffer - Maintains a rolling buffer of stream data
"""
import logging
import threading
import time
import tempfile
import os
import atexit
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Deque, Tuple, Generator

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
    
    Uses a memory-mapped file for video storage to reduce RAM usage.
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
        self._max_video_frames = int(max_seconds * fps * 1.5)  # 1.5x for safety
        max_audio_chunks = int(max_seconds * audio_chunks_per_second * 1.5)
        
        # Audio buffer (stays in memory as it's small)
        self._audio_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=max_audio_chunks)
        
        # Video buffer (disk-backed)
        self._video_mmap: Optional[np.memmap] = None
        self._video_file_path: Optional[str] = None
        self._video_timestamps = [0.0] * self._max_video_frames
        self._head = 0
        self._is_full = False
        self._video_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Stats
        self._audio_count = 0
        self._start_time: Optional[float] = None
        
        # Register cleanup
        atexit.register(self.close)
    
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
            
            # Initialize mmap on first frame
            if self._video_mmap is None:
                try:
                    h, w, c = frame.shape
                    shape = (self._max_video_frames, h, w, c)
                    dtype = frame.dtype
                    
                    # Create temp file
                    fd, self._video_file_path = tempfile.mkstemp(suffix='.raw_video')
                    os.close(fd)
                    
                    logger.info(f"Creating disk buffer at {self._video_file_path} ({self._max_video_frames} frames)")
                    
                    # Create mmap
                    self._video_mmap = np.memmap(
                        self._video_file_path, 
                        dtype=dtype, 
                        mode='w+', 
                        shape=shape
                    )
                except Exception as e:
                    logger.error(f"Failed to create disk buffer: {e}")
                    return

            # Write frame to mmap
            try:
                self._video_mmap[self._head] = frame
                self._video_timestamps[self._head] = timestamp
                
                self._head = (self._head + 1) % self._max_video_frames
                if self._head == 0:
                    self._is_full = True
                
                self._video_count = self._max_video_frames if self._is_full else self._head
            except Exception as e:
                logger.error(f"Error writing to disk buffer: {e}")
    
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
            if self._video_mmap is None:
                return []
                
            frames = []
            
            # Determine range of valid frames
            count = self._max_video_frames if self._is_full else self._head
            start_idx = self._head if self._is_full else 0
            
            # Iterate through valid frames
            # Note: This iterates all frames to find matches. 
            # Could be optimized with binary search if timestamps are strictly monotonic.
            for i in range(count):
                idx = (start_idx + i) % self._max_video_frames
                ts = self._video_timestamps[idx]
                
                if start_time <= ts <= end_time:
                    # Copy frame to detach from mmap and allow safe usage
                    frames.append((ts, self._video_mmap[idx].copy()))
            
            return frames
    
    def yield_video_frames(
        self,
        start_time: float,
        end_time: float
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Yield video frames in a time range one by one.
        This is memory efficient as it doesn't load all frames at once.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Yields:
            Tuple of (timestamp, frame)
        """
        # 1. Identify the range of indices we want
        indices_to_read = []
        
        with self._lock:
            if self._video_mmap is None:
                return
            
            count = self._max_video_frames if self._is_full else self._head
            start_idx = self._head if self._is_full else 0
            
            # Scan for matching timestamps
            for i in range(count):
                idx = (start_idx + i) % self._max_video_frames
                ts = self._video_timestamps[idx]
                if start_time <= ts <= end_time:
                    indices_to_read.append((idx, ts))
        
        # 2. Yield frames
        for idx, expected_ts in indices_to_read:
            frame_copy = None
            with self._lock:
                # Verify it hasn't been overwritten
                if self._video_timestamps[idx] == expected_ts:
                    # Copy is essential here to release the mmap dependency 
                    frame_copy = self._video_mmap[idx].copy()
            
            if frame_copy is not None:
                yield expected_ts, frame_copy
            else:
                # Frame was overwritten, stop or skip
                logger.warning(f"Frame at {expected_ts} was overwritten during save")
                break
    
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
            video_ts = []
            if self._video_mmap is not None:
                count = self._max_video_frames if self._is_full else self._head
                if count > 0:
                    # Oldest is at head if full, else 0
                    oldest_idx = self._head if self._is_full else 0
                    # Newest is at head-1
                    newest_idx = (self._head - 1) % self._max_video_frames
                    
                    video_ts = [self._video_timestamps[oldest_idx], self._video_timestamps[newest_idx]]

            audio_ts = []
            if self._audio_buffer:
                audio_ts = [self._audio_buffer[0][0], self._audio_buffer[-1][0]]
            
            all_ts = video_ts + audio_ts
            if all_ts:
                return min(all_ts), max(all_ts)
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
            return self._max_video_frames if self._is_full else self._head
    
    @property
    def audio_chunk_count(self) -> int:
        """Get number of audio chunks in buffer"""
        with self._lock:
            return len(self._audio_buffer)
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._head = 0
            self._is_full = False
            self._video_count = 0
            self._audio_buffer.clear()
            self._start_time = None
            # Note: We don't delete the mmap file here, just reset indices
            # It will be overwritten
    
    def close(self):
        """Close and cleanup resources"""
        with self._lock:
            if self._video_mmap is not None:
                try:
                    # Flush changes to disk
                    self._video_mmap.flush()
                    # Close mmap
                    self._video_mmap._mmap.close()
                    del self._video_mmap
                    self._video_mmap = None
                except Exception as e:
                    logger.error(f"Error closing mmap: {e}")
            
            if self._video_file_path and os.path.exists(self._video_file_path):
                try:
                    os.unlink(self._video_file_path)
                    logger.info(f"Deleted disk buffer: {self._video_file_path}")
                except Exception as e:
                    logger.error(f"Error deleting disk buffer file: {e}")
                finally:
                    self._video_file_path = None

    def __len__(self):
        """Get total items in buffer"""
        return self.video_frame_count + self.audio_chunk_count
    
    def __del__(self):
        self.close()


if __name__ == "__main__":
    # Test the buffer
    logging.basicConfig(level=logging.INFO)
    buffer = CircularBuffer(max_seconds=5.0)
    
    # Simulate adding frames
    print("Adding frames...")
    for i in range(150):  # 5 seconds at 30fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some data to verify it's working
        frame[0, 0] = [i % 255, 0, 0] 
        buffer.add_video_frame(frame)
        time.sleep(0.01)
    
    print(f"Video frames: {buffer.video_frame_count}")
    print(f"Duration: {buffer.get_duration():.2f}s")
    print(f"Range: {buffer.get_buffer_range()}")
    
    # Test retrieval
    start, end = buffer.get_buffer_range()
    if start and end:
        frames = buffer.get_video_frames(start, end)
        print(f"Retrieved {len(frames)} frames")
        if frames:
            print(f"First frame pixel: {frames[0][1][0,0]}")
            print(f"Last frame pixel: {frames[-1][1][0,0]}")
            
    buffer.close()
