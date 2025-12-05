"""
Video analyzer - Analyzes video frames for motion and visual excitement
"""
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, Any

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VideoMetrics:
    """Video analysis results"""
    timestamp: float
    motion_score: float  # Amount of motion (0.0 - 1.0+)
    brightness: float    # Average brightness
    is_high_motion: bool # Above threshold
    normalized_score: float  # 0.0 - 1.0 score


class VideoAnalyzer:
    """
    Analyzes video frames for motion and visual excitement.
    
    Uses frame differencing to detect motion intensity.
    High motion often correlates with exciting moments in games/streams.
    """
    
    def __init__(
        self,
        baseline_motion: float = 5.0,  # Pixel difference threshold
        spike_multiplier: float = 4.0,
        history_seconds: float = 5.0,
        center_crop_ratio: float = 0.7  # Focus on center 70% of frame
    ):
        """
        Initialize video analyzer.
        
        Args:
            baseline_motion: Expected baseline motion level
            spike_multiplier: How many times above baseline = max score
            history_seconds: Seconds of history to keep
            center_crop_ratio: Ratio of frame to keep (0.7 = center 70%, crops 15% from each edge)
        """
        self.baseline_motion = baseline_motion
        self.spike_multiplier = spike_multiplier
        self.center_crop_ratio = center_crop_ratio
        
        # History buffer
        max_samples = int(history_seconds * 30)  # Store 30 metrics per second
        self._history: Deque[VideoMetrics] = deque(maxlen=max_samples)
        
        # Adaptive baseline
        # Store ~60 seconds of history (30fps * 60 = 1800 frames)
        self._motion_history: Deque[float] = deque(maxlen=1800)
        self._adaptive_baseline: Optional[float] = None
        
        # State
        self._prev_frame = None
        self._prev_brightness: Optional[float] = None
        self._last_score = 0.0
        
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, video analysis will be disabled")
    
    def analyze(self, frame: Any) -> VideoMetrics:
        """
        Analyze a video frame.
        
        Args:
            frame: Video frame (numpy array from cv2)
            
        Returns:
            VideoMetrics with analysis results
        """
        timestamp = time.time()
        
        if not CV2_AVAILABLE or frame is None:
            return VideoMetrics(timestamp, 0.0, 0.0, False, 0.0)
        
        # Convert to grayscale for motion detection
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Crop to center region to focus on main content
            # This ignores margins (chat overlays, webcam corners, etc.)
            if self.center_crop_ratio < 1.0:
                h, w = gray.shape
                margin_x = int(w * (1 - self.center_crop_ratio) / 2)
                margin_y = int(h * (1 - self.center_crop_ratio) / 2)
                gray = gray[margin_y:h-margin_y, margin_x:w-margin_x]
            
            # Calculate brightness
            brightness = float(np.mean(gray))
            
            # Calculate motion
            motion_score = 0.0
            brightness_diff = 0.0
            
            if self._prev_frame is not None:
                # ROBUST MOTION DETECTION:
                # 1. Apply Gaussian blur to reduce noise/grain
                blurred_prev = cv2.GaussianBlur(self._prev_frame, (21, 21), 0)
                blurred_curr = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # 2. Calculate absolute difference
                frame_delta = cv2.absdiff(blurred_prev, blurred_curr)
                
                # 3. Threshold: Only count pixels that changed significantly (> 25 intensity)
                _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
                
                # 4. Dilate to fill in gaps
                thresh = cv2.dilate(thresh, None, iterations=2) # type: ignore
                
                # 5. Count moving pixels and normalize by total pixels
                motion_pixels = cv2.countNonZero(thresh)
                h, w = gray.shape
                motion_score = (motion_pixels / (h * w)) * 100  # Scale to 0-100 range
                
                # Calculate brightness change (sudden dark/light)
                if self._prev_brightness is not None:
                    brightness_diff = abs(brightness - self._prev_brightness)
            
            self._prev_frame = gray
            self._prev_brightness = brightness
            
            # Combine motion and brightness change
            # Brightness change is often more significant for events like "lights out" or flashes
            # Reduced brightness weight to avoid false positives
            combined_activity = motion_score + (brightness_diff * 0.5)
            
            # Update motion history for adaptive baseline
            self._motion_history.append(combined_activity)
            
            # Calculate adaptive baseline (median of recent motion)
            # Require at least 30 samples (1 sec) to start using adaptive
            if len(self._motion_history) >= 30:
                self._adaptive_baseline = float(np.median(list(self._motion_history)))
            
            # Use adaptive baseline if available, otherwise use configured
            # Increase minimum baseline to avoid noise triggering high scores
            baseline = max(self._adaptive_baseline or self.baseline_motion, 5.0)
            
            # Normalize score
            # Avoid division by zero
            baseline = max(baseline, 1.0)
            
            if combined_activity > baseline:
                ratio = (combined_activity - baseline) / (baseline * (self.spike_multiplier - 1))
                normalized_score = min(1.0, ratio)
            else:
                normalized_score = 0.0
            
            # Smooth the score
            smoothed_score = 0.7 * normalized_score + 0.3 * self._last_score
            self._last_score = smoothed_score
            
            # Determine if "high motion"
            is_high_motion = combined_activity > baseline * 2.0
            
            metrics = VideoMetrics(
                timestamp=timestamp,
                motion_score=combined_activity,
                brightness=brightness,
                is_high_motion=is_high_motion,
                normalized_score=smoothed_score
            )
            
            self._history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return VideoMetrics(timestamp, 0.0, 0.0, False, 0.0)
    
    def get_current_score(self) -> float:
        """Get the current normalized video score (0.0 - 1.0)"""
        if self._history:
            return self._history[-1].normalized_score
        return 0.0
    
    def reset(self):
        """Reset analyzer state"""
        self._history.clear()
        self._motion_history.clear()
        self._adaptive_baseline = None
        self._prev_frame = None
        self._prev_brightness = None
        self._last_score = 0.0
