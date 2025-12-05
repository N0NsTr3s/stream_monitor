
"""Scoring Engine - Combines signals from audio, video, and chat analyzers"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Deque, Callable, List

from ..utils.stats import RollingStats

logger = logging.getLogger(__name__)


class RecordingState(Enum):
    """State machine for recording"""
    IDLE = "idle"
    TRIGGERED = "triggered"  # Score exceeded threshold
    RECORDING = "recording"  # Currently recording
    COOLDOWN = "cooldown"  # Post-roll period
    FINALIZING = "finalizing"  # Waiting to see if we should merge with next clip


@dataclass
class ScoreEvent:
    """A scored moment in time"""
    timestamp: float
    audio_score: float
    chat_score: float
    video_score: float
    combined_score: float
    state: RecordingState


@dataclass
class ClipTrigger:
    """Information about a triggered clip"""
    start_time: float
    trigger_time: float
    end_time: Optional[float] = None
    peak_score: float = 0.0
    reason: str = ""


class ScoringEngine:
    """
    Combines signals from multiple analyzers using Z-Score based adaptive scoring.
    
    Instead of fixed thresholds, we use Z-Scores to detect moments that are
    significantly more intense than the stream's rolling average.
    
    Z-Score Thresholds:
        2.0 = Sensitive (Clips often)
        3.0 = Standard (Good moments)
        4.0 = Strict (Only massive spikes)
    
    Manages a state machine for clip recording:
    - IDLE: Monitoring, waiting for score to exceed threshold
    - TRIGGERED: Score exceeded threshold, recording started
    - RECORDING: Actively recording while score is high
    - COOLDOWN: Post-roll period after score dropped
    """
    
    def __init__(
        self,
        audio_weight: float = 0.5,
        chat_weight: float = 0.4,
        video_weight: float = 0.1,
        trigger_threshold: float = 2.5,  # Z-Score threshold (2.5 = 2.5 sigma)
        release_threshold: float = 1.0,  # Z-Score to stop recording
        pre_roll_seconds: float = 5.0,
        post_roll_seconds: float = 5.0,
        min_clip_duration: float = 5.0,
        cooldown_seconds: float = 10.0,
        calibration_seconds: float = 20.0
    ):

        self.audio_weight = audio_weight
        self.chat_weight = chat_weight
        self.video_weight = video_weight
        self.trigger_threshold = trigger_threshold
        self.release_threshold = release_threshold
        self.pre_roll_seconds = pre_roll_seconds
        self.post_roll_seconds = post_roll_seconds
        self.min_clip_duration = min_clip_duration
        self.cooldown_seconds = cooldown_seconds
        self.calibration_seconds = calibration_seconds
        
        # State
        self._state = RecordingState.IDLE
        self._current_trigger: Optional[ClipTrigger] = None
        self._last_clip_end: float = 0.0
        self._finalizing_start_time: float = 0.0
        self._release_time: float = 0.0  # When score first dropped below release threshold
        self._start_time = time.time()
        
        # Score history
        self._history: Deque[ScoreEvent] = deque(maxlen=600)  # ~60 seconds at 10Hz
        
        # Rolling stats for anomaly detection (5 minutes of history at ~1 update/sec)
        self._audio_stats = RollingStats(window_size=300, min_samples=30)
        self._chat_stats = RollingStats(window_size=300, min_samples=10)  # Chat updates slower
        self._video_stats = RollingStats(window_size=300, min_samples=30)
        
        # Current scores
        self._audio_score = 0.0
        self._chat_score = 0.0
        self._video_score = 0.0
        self._combined_score = 0.0
        
        # Callbacks
        self._on_clip_start: List[Callable[[ClipTrigger], None]] = []
        self._on_clip_end: List[Callable[[ClipTrigger], None]] = []
    
    @property
    def is_calibrating(self) -> bool:
        """Check if currently in calibration phase (need enough samples for Z-scores)"""
        time_ok = (time.time() - self._start_time) >= self.calibration_seconds
        # Also require minimum samples in rolling stats (matching each stat's min_samples)
        samples_ok = (
            self._audio_stats.count >= self._audio_stats.min_samples and
            self._chat_stats.count >= self._chat_stats.min_samples and
            self._video_stats.count >= self._video_stats.min_samples
        )
        return not (time_ok and samples_ok)

    def on_clip_start(self, callback: Callable[[ClipTrigger], None]):
        """Register callback for when a clip recording should start"""
        self._on_clip_start.append(callback)
    
    def on_clip_end(self, callback: Callable[[ClipTrigger], None]):
        """Register callback for when a clip recording should end"""
        self._on_clip_end.append(callback)
    
    def update(
        self,
        audio_score: Optional[float] = None,
        chat_score: Optional[float] = None,
        video_score: Optional[float] = None
    ) -> float:
        """
        Update scores and process state machine using Z-Score adaptive scoring.
        
        Args:
            audio_score: Current audio score (0.0-1.0), None to keep previous
            chat_score: Current chat score (0.0-1.0), None to keep previous
            video_score: Current video score (0.0-1.0), None to keep previous
            
        Returns:
            Combined Z-Score (higher = more intense relative to average)
        """
        timestamp = time.time()
        
        # Update individual scores and rolling stats
        if audio_score is not None:
            self._audio_score = audio_score
            self._audio_stats.update(audio_score)
        if chat_score is not None:
            self._chat_score = chat_score
            self._chat_stats.update(chat_score)
        if video_score is not None:
            self._video_score = video_score
            self._video_stats.update(video_score)
        
        # Get Z-Scores (How weird/abnormal is this value compared to recent history?)
        # A score of 3.0 means "Very High Spike" (3 standard deviations above average)
        audio_z = self._audio_stats.get_z_score(self._audio_score)
        chat_z = self._chat_stats.get_z_score(self._chat_score)
        video_z = self._video_stats.get_z_score(self._video_score)
        
        # Weighted combination of Z-Scores
        # We value Audio and Chat more than Video (webcams are noisy)
        # We ignore negative Z-Scores (quiet/still moments) using max(0, ...)
        self._combined_score = (
            (max(0.0, audio_z) * self.audio_weight) + 
            (max(0.0, chat_z) * self.chat_weight) + 
            (max(0.0, video_z) * self.video_weight)
        )
        
        # Record to history
        event = ScoreEvent(
            timestamp=timestamp,
            audio_score=self._audio_score,
            chat_score=self._chat_score,
            video_score=self._video_score,
            combined_score=self._combined_score,
            state=self._state
        )
        self._history.append(event)
        
        # Process state machine
        self._process_state(timestamp)
        
        return self._combined_score
    
    def _process_state(self, timestamp: float):
        """Process state machine transitions"""
        
        if self.is_calibrating:
            # Do not trigger during calibration
            return

        if self._state == RecordingState.IDLE:
            # Check if we should start recording
            if self._combined_score >= self.trigger_threshold:
                # Check cooldown
                if timestamp - self._last_clip_end >= self.cooldown_seconds:
                    self._start_recording(timestamp)
        
        elif self._state == RecordingState.TRIGGERED:
            # Just triggered, transition to recording
            self._state = RecordingState.RECORDING
            logger.debug("State: TRIGGERED -> RECORDING")
        
        elif self._state == RecordingState.RECORDING:
            # Update peak score
            if self._current_trigger:
                self._current_trigger.peak_score = max(
                    self._current_trigger.peak_score,
                    self._combined_score
                )
            
            # Check if score dropped below release threshold
            if self._combined_score < self.release_threshold:
                # Only release if we've met the minimum duration requirement
                # Projected duration = current duration + post_roll
                current_duration = timestamp - self._current_trigger.start_time # type: ignore
                projected_duration = current_duration + self.post_roll_seconds
                
                if projected_duration >= self.min_clip_duration:
                    self._state = RecordingState.COOLDOWN
                    self._release_time = timestamp  # Track when score dropped
                    logger.debug("State: RECORDING -> COOLDOWN")
                # Else: continue recording until min duration is met
        
        elif self._state == RecordingState.COOLDOWN:
            # Check if post-roll period is complete
            if self._current_trigger:
                cooldown_start = self._current_trigger.trigger_time + (
                    timestamp - self._current_trigger.trigger_time
                )
                
                # Check if enough time has passed since score dropped
                time_below_threshold = self._get_time_below_threshold()
                
                if time_below_threshold >= self.post_roll_seconds:
                    # Instead of ending immediately, enter FINALIZING state
                    # This allows us to merge with a subsequent clip if it starts soon
                    self._state = RecordingState.FINALIZING
                    self._finalizing_start_time = timestamp
                    logger.debug("State: COOLDOWN -> FINALIZING")
                
                # Re-trigger if score goes back up
                elif self._combined_score >= self.trigger_threshold:
                    self._state = RecordingState.RECORDING
                    logger.debug("State: COOLDOWN -> RECORDING (re-triggered)")
                    
        elif self._state == RecordingState.FINALIZING:
            # If score spikes again during the merge window (pre-roll duration),
            # go back to recording and extend the current clip
            if self._combined_score >= self.trigger_threshold:
                self._state = RecordingState.RECORDING
                logger.info("Merging with subsequent clip event!")
                logger.debug("State: FINALIZING -> RECORDING (merged)")
            
            # If merge window passed, finalize the clip
            elif timestamp - self._finalizing_start_time >= self.pre_roll_seconds:
                self._end_recording(timestamp)
    
    def _start_recording(self, timestamp: float):
        """Start a new clip recording"""
        self._state = RecordingState.TRIGGERED
        
        self._current_trigger = ClipTrigger(
            start_time=timestamp - self.pre_roll_seconds,
            trigger_time=timestamp,
            peak_score=self._combined_score,
            reason=self._determine_reason()
        )
        
        logger.info(f"Clip triggered! Score: {self._combined_score:.2f}, Reason: {self._current_trigger.reason}")
        
        # Notify callbacks
        for callback in self._on_clip_start:
            try:
                callback(self._current_trigger)
            except Exception as e:
                logger.error(f"Clip start callback error: {e}")
    
    def _end_recording(self, timestamp: float):
        """End the current clip recording"""
        if not self._current_trigger:
            return  # Already ended, prevent duplicate emissions
        
        # Calculate correct end time: when score dropped + post-roll period
        # This ensures we capture 5 seconds after the score falls below threshold
        end_time = self._release_time + self.post_roll_seconds
        self._current_trigger.end_time = end_time
        
        duration = end_time - self._current_trigger.start_time
        
        # Enforce minimum clip duration - extend end time if needed
        if duration < self.min_clip_duration:
            # Extend clip to meet minimum duration
            self._current_trigger.end_time = self._current_trigger.start_time + self.min_clip_duration
            duration = self.min_clip_duration
            logger.info(f"Extended clip to minimum duration: {duration:.1f}s")
        
        logger.info(f"Clip ended. Duration: {duration:.1f}s, Peak: {self._current_trigger.peak_score:.2f}")
        
        # Save current trigger and clear it BEFORE emitting callbacks
        # This prevents re-entry if callback takes time
        trigger_to_emit = self._current_trigger
        self._current_trigger = None
        
        # Emit clip
        for callback in self._on_clip_end:
            try:
                callback(trigger_to_emit)
            except Exception as e:
                logger.error(f"Clip end callback error: {e}")
        
        self._last_clip_end = timestamp
        self._state = RecordingState.IDLE
        logger.debug("State: COOLDOWN -> IDLE")
    
    def _get_time_below_threshold(self) -> float:
        """Get how long the score has been below release threshold"""
        if not self._history:
            return 0.0
        
        time_below = 0.0
        for event in reversed(self._history):
            if event.combined_score >= self.release_threshold:
                break
            time_below = time.time() - event.timestamp
        
        return time_below
    
    def _determine_reason(self) -> str:
        """Determine what triggered the clip"""
        reasons = []
        
        if self._audio_score > 0.5:
            reasons.append("loud_audio")
        if self._chat_score > 0.5:
            reasons.append("chat_hype")
        if self._video_score > 0.5:
            reasons.append("high_motion")
        
        return "+".join(reasons) if reasons else "unknown"
    
    @property
    def state(self) -> RecordingState:
        """Get current state"""
        return self._state
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._state in (
            RecordingState.TRIGGERED,
            RecordingState.RECORDING,
            RecordingState.COOLDOWN,
            RecordingState.FINALIZING
        )
    
    @property
    def current_score(self) -> float:
        """Get current combined score"""
        return self._combined_score
    
    @property
    def audio_score(self) -> float:
        """Get current audio score (raw value)"""
        return self._audio_score
    
    @property
    def chat_score(self) -> float:
        """Get current chat score (raw value)"""
        return self._chat_score
    
    @property
    def video_score(self) -> float:
        """Get current video/frame score (raw value)"""
        return self._video_score
    
    @property
    def audio_z_score(self) -> float:
        """Get current audio Z-Score (how abnormal compared to average)"""
        return self._audio_stats.get_z_score(self._audio_score)
    
    @property
    def chat_z_score(self) -> float:
        """Get current chat Z-Score (how abnormal compared to average)"""
        return self._chat_stats.get_z_score(self._chat_score)
    
    @property
    def video_z_score(self) -> float:
        """Get current video Z-Score (how abnormal compared to average)"""
        return self._video_stats.get_z_score(self._video_score)
    
    @property
    def current_trigger(self) -> Optional[ClipTrigger]:
        """Get current clip trigger info"""
        return self._current_trigger
    
    def get_history(self, seconds: float = 10.0) -> list:
        """Get recent score history"""
        cutoff = time.time() - seconds
        return [e for e in self._history if e.timestamp > cutoff]
    
    def reset(self):
        """Reset engine state"""
        self._state = RecordingState.IDLE
        self._current_trigger = None
        self._history.clear()
        self._audio_score = 0.0
        self._chat_score = 0.0
        self._video_score = 0.0
        self._combined_score = 0.0


if __name__ == "__main__":
    import random
    
    logging.basicConfig(level=logging.INFO)
    
    engine = ScoringEngine(
        trigger_threshold=0.6,
        release_threshold=0.3,
        post_roll_seconds=2.0
    )
    
    def on_start(trigger):
        print(f"🔴 CLIP START: {trigger.reason}")
    
    def on_end(trigger):
        duration = trigger.end_time - trigger.start_time
        print(f"⏹️ CLIP END: {duration:.1f}s, peak={trigger.peak_score:.2f}")
    
    engine.on_clip_start(on_start)
    engine.on_clip_end(on_end)
    
    # Simulate scores over time
    for i in range(100):
        # Simulate a spike at i=30
        if 30 <= i <= 50:
            audio = 0.7 + random.random() * 0.2
            chat = 0.8 + random.random() * 0.2
            video = 0.6 + random.random() * 0.2
        else:
            audio = random.random() * 0.3
            chat = random.random() * 0.2
            video = random.random() * 0.2
        
        score = engine.update(audio, chat, video)
        print(f"t={i}: audio={audio:.2f}, chat={chat:.2f}, video={video:.2f}, combined={score:.2f}, state={engine.state.value}")
        
        time.sleep(0.1)
