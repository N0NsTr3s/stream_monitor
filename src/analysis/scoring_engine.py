"""Scoring Engine - Combines signals from audio, video, and chat analyzers"""

import logging
import time
from typing import Optional

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
        
    def _end_recording(self, timestamp: float):
        """Finalize a recording window using configured pre/post roll and min length.
        Stores a simple trigger dict in self._current_trigger for downstream clipper.
        Adds debug logging so you can see exactly which times were chosen.
        """
        if getattr(self, "_current_trigger", None) is None:
            logger.debug("End recording called but no current trigger present")
            return

        # Resolve configured pre/post/min (fall back to reasonable defaults)
        pre_roll = getattr(self, "pre_roll", None) or getattr(self, "_pre_roll", None) or 3.0
        post_roll = getattr(self, "post_roll", None) or getattr(self, "_post_roll", None) or 5.0
        min_length = getattr(self, "min_clip_length", None) or getattr(self, "_min_clip_length", None) or 1.0

        trigger_ts = getattr(self._current_trigger, "timestamp", None)
        # support dict-like or object-like trigger
        if trigger_ts is None:
            trigger_ts = self._current_trigger.get("timestamp") if isinstance(self._current_trigger, dict) else None

        if trigger_ts is None:
            # fallback: use last history entry timestamp
            trigger_ts = self._history[-1].timestamp if self._history else time.time()

        # Compute clip range
        start = max(0.0, trigger_ts - float(pre_roll))
        end = float(timestamp) + float(post_roll)
        duration = end - start

        if duration < float(min_length):
            end = start + float(min_length)
            duration = end - start

        reason = self._determine_reason() if hasattr(self, "_determine_reason") else "unknown"

        # Save trigger info (keep simple, downstream code can adapt)
        clip_info = {
            "start": start,
            "end": end,
            "duration": duration,
            "reason": reason,
            "timestamp": time.time(),
            "trigger_timestamp": trigger_ts,
            "pre_roll": pre_roll,
            "post_roll": post_roll,
            "min_length": min_length,
        }
        self._current_trigger = clip_info

        logger.debug(
            "Finalized clip: start=%.3f end=%.3f duration=%.3f reason=%s (pre=%s post=%s min=%s)",
            start, end, duration, reason, pre_roll, post_roll, min_length,
        )

    
    def _get_time_below_threshold(self) -> float:
        """Return how long (seconds) the combined score has been continuously below threshold."""
        if not self._history:
            return 0.0
        # Walk history from newest back until score >= threshold
        threshold = getattr(self, "trigger_threshold", None) or getattr(self, "_trigger_threshold", None) or 3.0
        t = time.time()
        elapsed = 0.0
        # assume history entries have .timestamp and .combined attributes
        for entry in reversed(self._history):
            if getattr(entry, "combined", None) is None:
                # try alternative name
                score = getattr(entry, "score", getattr(entry, "value", None))
            else:
                score = entry.combined
            if score is None:
                # cannot determine, stop counting
                break
            if score >= threshold:
                break
            elapsed = t - entry.timestamp
            t = entry.timestamp
        return elapsed
    
    
    def _determine_reason(self) -> str:
        reasons = []
        
        if getattr(self, "_audio_score", 0.0) > 0.5:
            reasons.append("audio")
        if getattr(self, "_chat_score", 0.0) > 0.5:
            reasons.append("chat")
        if getattr(self, "_video_score", 0.0) > 0.5:
            reasons.append("video")
        
        return "+".join(reasons) if reasons else "unknown"
    
    @property
    def state(self) -> RecordingState:
        """Get current state"""
        return self._state
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._state == RecordingState.RECORDING
    
    @property
    def current_score(self) -> float:
        """Get current combined score"""
        return getattr(self, "_combined_score", 0.0)
    
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
    def current_trigger(self) -> Optional[dict]:
        """Get current clip trigger info (dict for introspection)."""
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
    )
    
    def on_start(trigger):
        print(f"🔴 CLIP START: {trigger.reason}")
    
    def on_end(trigger):
        duration = trigger.end_time - trigger.start_time
        print(f"⏹️ CLIP END: {duration:.1f}s, peak={trigger.peak_score:.2f}")
    
    engine.on_clip_start(on_start)
    engine.on_clip_end(on_end)
    """
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
    """
    time.sleep(0.1)
