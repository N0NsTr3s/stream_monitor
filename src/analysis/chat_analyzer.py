"""
Chat analyzer - Analyzes chat stream for activity/excitement levels
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChatMetrics:
    """Chat analysis results"""
    timestamp: float
    messages_per_second: float  # Message velocity
    unique_chatters: int  # Unique users in window
    avg_message_length: float
    caps_ratio: float  # Ratio of caps messages (excitement)
    emote_density: float  # Emotes per message
    normalized_score: float  # 0.0 - 1.0 score


@dataclass
class ChatEvent:
    """Represents a chat message for analysis"""
    timestamp: float
    username: str
    message: str
    has_caps: bool = False
    emote_count: int = 0


class ChatAnalyzer:
    """
    Analyzes chat stream for activity patterns and excitement indicators.
    
    Tracks message velocity, unique chatters, and excitement indicators
    like caps lock usage and emote spam.
    """
    
    # Common excitement indicators
    HYPE_KEYWORDS = {
        'pog', 'poggers', 'pogchamp', 'lets go', 'letsgoooo',
        'omg', 'holy', 'wow', 'insane', 'crazy', 'clutch',
        'gg', 'ez', 'lol', 'lmao', 'hype', 'w', 'dub',
        'noway', 'no way', 'what', 'howww'
    }
    
    def __init__(
        self,
        window_seconds: float = 5.0,
        baseline_mps: float = 1.0,
        spike_multiplier: float = 5.0,
        history_seconds: float = 30.0
    ):
        """
        Initialize chat analyzer.
        
        Args:
            window_seconds: Window for calculating metrics
            baseline_mps: Expected baseline messages per second
            spike_multiplier: How many times above baseline = max score
            history_seconds: Seconds of history to keep
        """
        self.window_seconds = window_seconds
        self.baseline_mps = baseline_mps
        self.spike_multiplier = spike_multiplier
        
        # Message buffer
        self._messages: Deque[ChatEvent] = deque(maxlen=1000)
        
        # Metrics history
        max_metrics = int(history_seconds * 2)  # 2 metrics per second
        self._history: Deque[ChatMetrics] = deque(maxlen=max_metrics)
        
        # Adaptive baseline
        # Store ~60 seconds of history (assuming ~2 updates/sec)
        self._mps_history: Deque[float] = deque(maxlen=120)
        self._adaptive_baseline: Optional[float] = None
        
        # State
        self._last_score = 0.0
        self._last_analysis_time = 0.0
        
        # Latency buffer for score delay
        self._score_buffer: Deque[Tuple[float, float]] = deque()  # (timestamp, score)
    
    def add_message(
        self,
        username: str,
        message: str,
        emote_count: int = 0
    ):
        """
        Add a chat message for analysis.
        
        Args:
            username: Chat username
            message: Message text
            emote_count: Number of emotes in message
        """
        # Check for caps excitement
        has_caps = self._is_caps_message(message)
        
        event = ChatEvent(
            timestamp=time.time(),
            username=username,
            message=message,
            has_caps=has_caps,
            emote_count=emote_count
        )
        
        self._messages.append(event)
    
    def _is_caps_message(self, message: str) -> bool:
        """Check if message is mostly caps (excitement indicator)"""
        alpha_chars = [c for c in message if c.isalpha()]
        if len(alpha_chars) < 3:
            return False
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        return caps_ratio > 0.7
    
    def _has_hype_keywords(self, message: str) -> bool:
        """Check if message contains hype keywords"""
        message_lower = message.lower()
        return any(kw in message_lower for kw in self.HYPE_KEYWORDS)
    
    def analyze(self) -> ChatMetrics:
        """
        Analyze recent chat activity.
        
        Returns:
            ChatMetrics with analysis results
        """
        timestamp = time.time()
        cutoff = timestamp - self.window_seconds
        
        # Get messages in window
        recent_messages = [m for m in self._messages if m.timestamp > cutoff]
        
        # Calculate metrics
        message_count = len(recent_messages)
        mps = message_count / self.window_seconds
        
        # Unique chatters
        unique_chatters = len(set(m.username for m in recent_messages))
        
        # Average message length
        if recent_messages:
            avg_length = sum(len(m.message) for m in recent_messages) / message_count
        else:
            avg_length = 0.0
        
        # Caps ratio (excitement)
        if recent_messages:
            caps_count = sum(1 for m in recent_messages if m.has_caps)
            caps_ratio = caps_count / message_count
        else:
            caps_ratio = 0.0
        
        # Emote density
        if recent_messages:
            total_emotes = sum(m.emote_count for m in recent_messages)
            emote_density = total_emotes / message_count
        else:
            emote_density = 0.0
        
        # Update MPS history for adaptive baseline
        self._mps_history.append(mps)
        
        # Calculate adaptive baseline
        if len(self._mps_history) >= 10:
            self._adaptive_baseline = float(np.median(list(self._mps_history)))
        
        # Use adaptive or configured baseline
        baseline = self._adaptive_baseline or self.baseline_mps
        
        # Calculate normalized score
        # Higher score for more messages, caps, and hype words
        if mps > baseline:
            velocity_score = min(1.0, (mps - baseline) / (baseline * (self.spike_multiplier - 1)))
        else:
            velocity_score = 0.0
        
        # Boost score based on excitement indicators
        excitement_boost = (caps_ratio * 0.2) + (min(emote_density, 2.0) * 0.1)
        
        # Combined score
        raw_score = velocity_score + excitement_boost
        normalized_score = min(1.0, raw_score)
        
        # Smooth the score
        smoothed_score = 0.6 * normalized_score + 0.4 * self._last_score
        self._last_score = smoothed_score
        
        metrics = ChatMetrics(
            timestamp=timestamp,
            messages_per_second=mps,
            unique_chatters=unique_chatters,
            avg_message_length=avg_length,
            caps_ratio=caps_ratio,
            emote_density=emote_density,
            normalized_score=smoothed_score
        )
        
        self._history.append(metrics)
        self._last_analysis_time = timestamp
        
        return metrics
    
    def get_delayed_score(self, latency_seconds: float = 0.0) -> float:
        """
        Get the score from 'latency_seconds' ago.
        Useful for aligning chat (fast) with video (slow).
        """
        if latency_seconds <= 0:
            return self._last_score
            
        current_time = time.time()
        target_time = current_time - latency_seconds
        
        # Add current score to buffer
        self._score_buffer.append((current_time, self._last_score))
        
        # Remove old scores (keep 2x latency just in case)
        while self._score_buffer and self._score_buffer[0][0] < current_time - (latency_seconds * 2):
            self._score_buffer.popleft()
            
        # Find score closest to target time
        # Since buffer is sorted by time, we can iterate or bisect
        # For small buffers, iteration is fine
        closest_score = 0.0
        min_diff = float('inf')
        
        for ts, score in self._score_buffer:
            diff = abs(ts - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_score = score
            else:
                # If diff starts increasing, we passed the target
                break
                
        return closest_score

    def get_current_score(self) -> float:
        """Get the current normalized chat score (0.0 - 1.0)"""
        # Re-analyze if stale
        if time.time() - self._last_analysis_time > 0.5:
            self.analyze()
        
        if self._history:
            return self._history[-1].normalized_score
        return 0.0
    
    def get_average_score(self, seconds: float = 3.0) -> float:
        """Get average score over recent seconds"""
        if not self._history:
            return 0.0
        
        cutoff = time.time() - seconds
        recent = [m.normalized_score for m in self._history if m.timestamp > cutoff]
        
        if recent:
            return sum(recent) / len(recent)
        return 0.0
    
    def get_peak_score(self, seconds: float = 3.0) -> float:
        """Get peak score over recent seconds"""
        if not self._history:
            return 0.0
        
        cutoff = time.time() - seconds
        recent = [m.normalized_score for m in self._history if m.timestamp > cutoff]
        
        if recent:
            return max(recent)
        return 0.0
    
    def get_messages_per_second(self) -> float:
        """Get current messages per second"""
        if self._history:
            return self._history[-1].messages_per_second
        return 0.0
    
    def get_history(self, seconds: float = 10.0) -> list:
        """Get recent metrics history"""
        cutoff = time.time() - seconds
        return [m for m in self._history if m.timestamp > cutoff]
    
    def reset(self):
        """Reset analyzer state"""
        self._messages.clear()
        self._history.clear()
        self._mps_history.clear()
        self._adaptive_baseline = None
        self._last_score = 0.0


# Import numpy for median calculation
try:
    import numpy as np # type: ignore
except ImportError:
    # Fallback without numpy
    class np:
        @staticmethod
        def median(values):
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
            return sorted_values[n//2]


if __name__ == "__main__":
    # Test with synthetic chat
    analyzer = ChatAnalyzer(baseline_mps=2.0)
    
    # Simulate normal chat
    for i in range(5):
        analyzer.add_message(f"user{i}", "normal message here")
    
    metrics = analyzer.analyze()
    print(f"Normal: MPS={metrics.messages_per_second:.2f}, Score={metrics.normalized_score:.4f}")
    
    # Simulate hype chat
    for i in range(20):
        analyzer.add_message(f"user{i}", "POGGERS LETS GOOOOO")
    
    metrics = analyzer.analyze()
    print(f"Hype: MPS={metrics.messages_per_second:.2f}, Score={metrics.normalized_score:.4f}")
