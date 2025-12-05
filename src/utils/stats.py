"""
Rolling Statistics - Tracks history and detects anomalies/spikes using Z-Scores
"""
import numpy as np
from collections import deque
from typing import Tuple


class RollingStats:
    """
    Tracks a rolling window of values and provides statistical analysis.
    Uses Z-Scores to detect how 'abnormal' a value is relative to recent history.
    
    Z-Score interpretation:
        0.0 = Exactly average
        1.0 = Slightly above average
        2.0 = Moderately high
        3.0 = SPIKE (Clip worthy!)
        4.0+ = Extremely rare spike
    """
    
    def __init__(self, window_size: int = 300, min_samples: int = 30):
        """
        Initialize rolling statistics tracker.
        
        Args:
            window_size: Number of samples to keep (e.g., 300 @ 1/sec = 5 mins)
            min_samples: Minimum samples needed for Z-score calculation
        """
        self.window: deque = deque(maxlen=window_size)
        self.min_samples = min_samples

    def update(self, value: float):
        """Add a new value to the history."""
        self.window.append(value)

    def get_stats(self) -> Tuple[float, float]:
        """
        Get current statistics.
        
        Returns:
            Tuple of (mean, std_dev)
        """
        if not self.window:
            return 0.0, 0.0
        
        # Convert to numpy array for fast math
        data = np.array(self.window)
        mean = float(np.mean(data))
        std = float(np.std(data))
        
        return mean, std

    def get_z_score(self, value: float) -> float:
        """
        Returns how 'abnormal' the value is compared to rolling average.
        
        This is the core metric for adaptive clipping:
            0.0 = Exactly average
            1.0 = Slightly high
            2.0 = Notable spike
            3.0 = SPIKE (Clip worthy!)
            
        Args:
            value: The current value to evaluate
            
        Returns:
            Z-score (standard deviations above mean). Returns 0 if not enough data.
        """
        # Need minimum samples for reliable statistics
        if len(self.window) < self.min_samples:
            return 0.0  # Not enough data yet (Calibrating)
        
        data = np.array(self.window)
        mean = float(np.mean(data))
        std = float(np.std(data))
        
        # Prevent division by zero if stream is silent/static
        # If std is very small, the signal is flat - no spikes possible
        if std < 0.0001:
            # Only consider it a spike if value is significantly above mean
            if value > mean * 2.0:
                return 3.0  # Moderate spike for flat signal
            return 0.0
        
        # Calculate Z-Score: (Current - Average) / Volatility
        z_score = (value - mean) / std
        return z_score

    def is_anomaly(self, value: float, threshold_sigma: float = 3.0) -> bool:
        """
        Check if a value is significantly higher than the rolling average.
        
        Args:
            value: The value to check
            threshold_sigma: How many standard deviations above average to trigger.
                           2.0 = Moderate, 3.0 = Hard, 4.0 = Very Hard.
        
        Returns:
            True if value is an anomaly (spike)
        """
        z_score = self.get_z_score(value)
        return z_score >= threshold_sigma
    
    def get_normalized_score(self, value: float) -> float:
        """
        Get a normalized score (0-1) based on how many std devs above mean.
        
        Args:
            value: The value to normalize
            
        Returns:
            Normalized score (0.0 to 1.0, clamped)
        """
        z_score = self.get_z_score(value)
        
        # Map z-score to 0-1 range (0 std = 0.0, 3 std = 1.0)
        normalized = max(0.0, z_score / 3.0)
        return min(1.0, normalized)
    
    @property
    def count(self) -> int:
        """Get number of samples in history"""
        return len(self.window)
    
    @property
    def is_warmed_up(self) -> bool:
        """Check if we have enough samples for reliable stats (at least 30)"""
        return len(self.window) >= 30
