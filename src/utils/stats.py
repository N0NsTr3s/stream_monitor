"""
Rolling Statistics - Tracks history and detects anomalies/spikes
"""
import numpy as np
from collections import deque
from typing import Tuple


class RollingStats:
    """
    Tracks a rolling window of values and provides statistical analysis.
    Used to detect spikes/anomalies relative to recent history.
    """
    
    def __init__(self, window_size: int = 300):
        """
        Initialize rolling statistics tracker.
        
        Args:
            window_size: Number of samples to keep (e.g., 300 @ 1/sec = 5 mins)
        """
        self.window: deque = deque(maxlen=window_size)

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
        mean, std = self.get_stats()
        
        # If the signal is flat (std=0), use a fallback check
        if std == 0:
            return value > mean * 2  # Fallback: 2x the flatline
        
        # The Formula: Is Value > Average + (Sensitivity * Variance)
        limit = mean + (threshold_sigma * std)
        return value > limit
    
    def get_normalized_score(self, value: float) -> float:
        """
        Get a normalized score (0-1) based on how many std devs above mean.
        
        Args:
            value: The value to normalize
            
        Returns:
            Normalized score (0.0 to 1.0, clamped)
        """
        mean, std = self.get_stats()
        
        if std == 0:
            if mean == 0:
                return min(1.0, value)
            return min(1.0, value / mean) if mean > 0 else 0.0
        
        # How many std devs above mean
        z_score = (value - mean) / std
        
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
