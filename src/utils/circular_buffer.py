"""
Circular Buffer - Maintains a rolling buffer of raw stream packets
"""
import logging
import threading
import time
from collections import deque
from typing import Optional, Deque, Tuple, List

logger = logging.getLogger(__name__)

class CircularBuffer:
    """
    Circular buffer for storing raw stream packets (bytes).
    
    Maintains a rolling buffer of (timestamp, bytes) tuples.
    This is much more memory efficient than storing decoded frames.
    """
    
    def __init__(
        self,
        max_seconds: float = 30.0,
        fps: float = 30.0, # Unused, kept for compatibility
        audio_chunks_per_second: float = 10.0 # Unused
    ):
        self.max_seconds = max_seconds
        self.fps = fps
        
        # Buffer stores (timestamp, data_bytes)
        self._buffer: Deque[Tuple[float, bytes]] = deque()
        self._total_bytes = 0
        
        self._lock = threading.RLock()
        self._start_time: Optional[float] = None
    
    def add_stream_chunk(self, chunk: bytes, timestamp: Optional[float] = None):
        """
        Add a raw stream chunk to the buffer.
        """
        if not chunk:
            return
            
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            
            self._buffer.append((timestamp, chunk))
            self._total_bytes += len(chunk)
            
            # Prune old data
            self._prune(timestamp)
            
    def _prune(self, current_time: float):
        """Remove data older than max_seconds"""
        cutoff = current_time - self.max_seconds
        
        while self._buffer and self._buffer[0][0] < cutoff:
            _, chunk = self._buffer.popleft()
            self._total_bytes -= len(chunk)
            
    def get_stream_data(self, start_time: float, end_time: float) -> bytes:
        """
        Get concatenated raw stream bytes for a time range.
        """
        with self._lock:
            chunks = [
                chunk for ts, chunk in self._buffer
                if start_time <= ts <= end_time
            ]
            return b"".join(chunks)
            
    def get_buffer_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the time range currently in buffer"""
        with self._lock:
            if not self._buffer:
                return None, None
            return self._buffer[0][0], self._buffer[-1][0]
            
    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        start, end = self.get_buffer_range()
        if start and end:
            return end - start
        return 0.0
        
    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._total_bytes = 0
            self._start_time = None
            
    def close(self):
        self.clear()
