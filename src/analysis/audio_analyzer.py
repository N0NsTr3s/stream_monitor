"""
Audio analyzer - Analyzes audio stream for volume/excitement levels
"""
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    """Audio analysis results"""
    timestamp: float
    rms: float  # Root mean square (volume level)
    peak: float  # Peak amplitude
    is_loud: bool  # Above threshold
    normalized_score: float  # 0.0 - 1.0 score


class AudioAnalyzer:
    """
    Analyzes audio stream for volume levels and excitement indicators.
    
    Uses RMS (Root Mean Square) to measure volume and detects
    spikes that may indicate exciting moments (screaming, cheering, etc.)
    """
    
    def __init__(
        self,
        baseline_rms: float = 0.1,
        spike_multiplier: float = 3.0,
        history_seconds: float = 5.0,
        sample_rate: int = 44100
    ):
        """
        Initialize audio analyzer.
        
        Args:
            baseline_rms: Expected baseline RMS level (0.0-1.0)
            spike_multiplier: How many times above baseline = max score
            history_seconds: Seconds of history to keep
            sample_rate: Audio sample rate
        """
        self.baseline_rms = baseline_rms
        self.spike_multiplier = spike_multiplier
        self.sample_rate = sample_rate
        
        # History buffer
        max_samples = int(history_seconds * 10)  # Store 10 metrics per second
        self._history: Deque[AudioMetrics] = deque(maxlen=max_samples)
        
        # Adaptive baseline
        self._rms_history: Deque[float] = deque(maxlen=100)
        self._adaptive_baseline: Optional[float] = None
        
        # State
        self._last_rms = 0.0
        self._last_score = 0.0
    
    def analyze(self, audio_data: np.ndarray) -> AudioMetrics:
        """
        Analyze an audio chunk.
        
        Args:
            audio_data: Audio samples as float32 numpy array (-1.0 to 1.0)
            
        Returns:
            AudioMetrics with analysis results
        """
        timestamp = time.time()
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # Calculate peak
        peak = np.max(np.abs(audio_data))
        
        # Update RMS history for adaptive baseline
        self._rms_history.append(rms)
        
        # Calculate adaptive baseline (median of recent RMS values)
        if len(self._rms_history) >= 10:
            self._adaptive_baseline = np.median(list(self._rms_history))
        
        # Use adaptive baseline if available, otherwise use configured
        baseline = self._adaptive_baseline or self.baseline_rms
        
        # Normalize score based on how much above baseline
        # Score of 0.0 = at/below baseline
        # Score of 1.0 = spike_multiplier times above baseline
        if rms > baseline:
            ratio = (rms - baseline) / (baseline * (self.spike_multiplier - 1))
            normalized_score = min(1.0, ratio)
        else:
            normalized_score = 0.0
        
        # Smooth the score a bit to avoid jitter
        smoothed_score = 0.7 * normalized_score + 0.3 * self._last_score
        self._last_score = smoothed_score
        
        # Determine if "loud"
        is_loud = rms > baseline * 1.5
        
        metrics = AudioMetrics(
            timestamp=timestamp,
            rms=rms,
            peak=peak,
            is_loud=is_loud,
            normalized_score=smoothed_score
        )
        
        self._history.append(metrics)
        self._last_rms = rms
        
        return metrics
    
    def get_current_score(self) -> float:
        """Get the current normalized audio score (0.0 - 1.0)"""
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
    
    def is_currently_loud(self) -> bool:
        """Check if audio is currently loud"""
        if self._history:
            return self._history[-1].is_loud
        return False
    
    def get_history(self, seconds: float = 5.0) -> list:
        """Get recent history"""
        cutoff = time.time() - seconds
        return [m for m in self._history if m.timestamp > cutoff]
    
    def reset(self):
        """Reset analyzer state"""
        self._history.clear()
        self._rms_history.clear()
        self._adaptive_baseline = None
        self._last_rms = 0.0
        self._last_score = 0.0


if __name__ == "__main__":
    # Test with synthetic audio
    analyzer = AudioAnalyzer()
    
    # Simulate quiet audio
    quiet_audio = np.random.randn(4410).astype(np.float32) * 0.05
    metrics = analyzer.analyze(quiet_audio)
    print(f"Quiet: RMS={metrics.rms:.4f}, Score={metrics.normalized_score:.4f}")
    
    # Simulate loud audio
    loud_audio = np.random.randn(4410).astype(np.float32) * 0.5
    metrics = analyzer.analyze(loud_audio)
    print(f"Loud: RMS={metrics.rms:.4f}, Score={metrics.normalized_score:.4f}")
    
    # Simulate very loud audio (spike)
    spike_audio = np.random.randn(4410).astype(np.float32) * 0.9
    metrics = analyzer.analyze(spike_audio)
    print(f"Spike: RMS={metrics.rms:.4f}, Score={metrics.normalized_score:.4f}")
