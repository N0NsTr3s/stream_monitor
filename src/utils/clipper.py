"""
Clipper - Saves buffered stream data as video clips
"""
import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# Try to import cv2, provide fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from .circular_buffer import CircularBuffer

logger = logging.getLogger(__name__)


class Clipper:
    """
    Saves stream buffer contents as video clips.
    
    Uses OpenCV to write video frames and FFmpeg to mux with audio.
    """
    
    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1920, 1080),
        audio_sample_rate: int = 44100,
        video_codec: str = "mp4v",
        output_format: str = "mp4"
    ):
        """
        Initialize clipper.
        
        Args:
            output_dir: Directory to save clips
            fps: Video frame rate
            resolution: Video resolution (width, height)
            audio_sample_rate: Audio sample rate
            video_codec: OpenCV video codec
            output_format: Output file format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.resolution = resolution
        self.audio_sample_rate = audio_sample_rate
        self.video_codec = video_codec
        self.output_format = output_format
        
        # Track saved clips
        self._clip_count = 0
        self._save_lock = threading.Lock()
    
    def _generate_filename(self, prefix: str = "clip") -> Path:
        """Generate a unique filename for a clip"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._clip_count += 1
        filename = f"{prefix}_{timestamp}_{self._clip_count:04d}.{self.output_format}"
        return self.output_dir / filename
    
    def save_clip(
        self,
        buffer: CircularBuffer,
        start_time: float,
        end_time: float,
        prefix: str = "clip",
        reason: str = ""
    ) -> Optional[Path]:
        """
        Save a clip from the buffer.
        
        Args:
            buffer: CircularBuffer containing the stream data
            start_time: Start timestamp
            end_time: End timestamp
            prefix: Filename prefix
            reason: Reason/tag for the clip
            
        Returns:
            Path to saved clip, or None if failed
        """
        with self._save_lock:
            try:
                return self._save_clip_internal(
                    buffer, start_time, end_time, prefix, reason
                )
            except Exception as e:
                logger.error(f"Failed to save clip: {e}")
                return None
    
    def _save_clip_internal(
        self,
        buffer: CircularBuffer,
        start_time: float,
        end_time: float,
        prefix: str,
        reason: str
    ) -> Optional[Path]:
        """Internal method to save a clip"""
        
        if not CV2_AVAILABLE:
            logger.error("OpenCV (cv2) not available - cannot save clips")
            return None
        
        # Get frames from buffer
        video_frames = buffer.get_video_frames(start_time, end_time)
        audio_chunks = buffer.get_audio_chunks(start_time, end_time)
        
        if not video_frames:
            logger.warning("No video frames in buffer for clip")
            return None
        
        logger.info(f"Saving clip: {len(video_frames)} frames, {len(audio_chunks)} audio chunks")
        
        # Generate output path
        if reason:
            prefix = f"{prefix}_{reason.replace('+', '_')}"
        output_path = self._generate_filename(prefix)
        
        # Save video frames
        video_path = self._save_video_frames(video_frames, output_path)
        
        if not video_path:
            return None
        
        # If we have audio, mux it with the video
        if audio_chunks:
            final_path = self._mux_audio(video_path, audio_chunks, output_path)
            if final_path:
                return final_path
        
        return video_path
    
    def _save_video_frames(
        self,
        frames: List[Tuple[float, np.ndarray]],
        output_path: Path
    ) -> Optional[Path]:
        """Save video frames using OpenCV"""
        
        if not frames:
            return None
        
        # Get resolution from first frame
        _, first_frame = frames[0]
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        temp_path = output_path.with_suffix('.temp.avi')
        
        writer = cv2.VideoWriter(
            str(temp_path),
            fourcc,
            self.fps,
            (width, height)
        )
        
        if not writer.isOpened():
            logger.error("Failed to create video writer")
            return None
        
        try:
            # Sort frames by timestamp
            sorted_frames = sorted(frames, key=lambda x: x[0])
            
            # Write frames
            for timestamp, frame in sorted_frames:
                # Ensure frame is correct format
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Resize if needed
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                
                writer.write(frame)
            
            writer.release()
            logger.info(f"Saved video to {temp_path}")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error writing video: {e}")
            writer.release()
            return None
    
    def _mux_audio(
        self,
        video_path: Path,
        audio_chunks: List[Tuple[float, np.ndarray]],
        output_path: Path
    ) -> Optional[Path]:
        """Mux audio with video using FFmpeg"""
        
        # Sort and concatenate audio chunks
        sorted_audio = sorted(audio_chunks, key=lambda x: x[0])
        audio_data = np.concatenate([chunk for _, chunk in sorted_audio])
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            audio_temp = f.name
            audio_data.astype(np.float32).tofile(f)
        
        try:
            # Use FFmpeg to mux audio and video
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-f", "f32le",
                "-ar", str(self.audio_sample_rate),
                "-ac", "1",
                "-i", audio_temp,
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                "-shortest",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Clean up temp files
                os.unlink(audio_temp)
                os.unlink(video_path)
                logger.info(f"Saved clip with audio to {output_path}")
                return output_path
            else:
                logger.warning(f"FFmpeg muxing failed: {result.stderr}")
                # Fall back to video-only
                os.rename(video_path, output_path.with_suffix('.avi'))
                return output_path.with_suffix('.avi')
                
        except FileNotFoundError:
            logger.warning("FFmpeg not found - saving video without audio")
            os.rename(video_path, output_path.with_suffix('.avi'))
            return output_path.with_suffix('.avi')
        except Exception as e:
            logger.error(f"Error muxing audio: {e}")
            return video_path
        finally:
            # Clean up temp audio file
            try:
                os.unlink(audio_temp)
            except:
                pass
    
    def save_clip_async(
        self,
        buffer: CircularBuffer,
        start_time: float,
        end_time: float,
        prefix: str = "clip",
        reason: str = "",
        callback: Optional[callable] = None
    ):
        """
        Save a clip asynchronously in a background thread.
        
        Args:
            buffer: CircularBuffer containing the stream data
            start_time: Start timestamp
            end_time: End timestamp
            prefix: Filename prefix
            reason: Reason/tag for the clip
            callback: Optional callback(path) when complete
        """
        def save_task():
            # Create a snapshot of the buffer data
            video_frames = buffer.get_video_frames(start_time, end_time)
            audio_chunks = buffer.get_audio_chunks(start_time, end_time)
            
            # Save directly without going through buffer again
            path = self._save_direct(video_frames, audio_chunks, prefix, reason)
            
            if callback:
                callback(path)
        
        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()
        return thread
    
    def _save_direct(
        self,
        video_frames: List[Tuple[float, np.ndarray]],
        audio_chunks: List[Tuple[float, np.ndarray]],
        prefix: str,
        reason: str
    ) -> Optional[Path]:
        """Save frames directly without buffer lookup"""
        
        if not video_frames:
            return None
        
        with self._save_lock:
            if reason:
                prefix = f"{prefix}_{reason.replace('+', '_')}"
            output_path = self._generate_filename(prefix)
            
            video_path = self._save_video_frames(video_frames, output_path)
            
            if video_path and audio_chunks:
                return self._mux_audio(video_path, audio_chunks, output_path)
            
            return video_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test clipper
    buffer = CircularBuffer(max_seconds=10.0)
    clipper = Clipper(output_dir=Path("./test_clips"))
    
    # Add some test frames
    for i in range(150):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.add_video_frame(frame)
        time.sleep(0.033)
    
    start, end = buffer.get_buffer_range()
    if start and end:
        path = clipper.save_clip(buffer, start, end, prefix="test", reason="demo")
        if path:
            print(f"Saved test clip to: {path}")
