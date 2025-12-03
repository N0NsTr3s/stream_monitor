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
from typing import Optional, List, Tuple, Iterable

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
        
        # Get frames from buffer (generator)
        video_frames = buffer.yield_video_frames(start_time, end_time)
        audio_chunks = buffer.get_audio_chunks(start_time, end_time)
        
        logger.info(f"Saving clip: {end_time - start_time:.1f}s duration")
        
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
        
        # Fallback if no audio or muxing failed
        if video_path.exists():
            final_path = output_path.with_suffix('.avi')
            if final_path.exists():
                os.unlink(final_path)
            os.rename(video_path, final_path)
            return final_path
        
        return None
    
    def _save_video_frames(
        self,
        frames: Iterable[Tuple[float, np.ndarray]],
        output_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Optional[Path]:
        """Save video frames using OpenCV"""
        
        writer = None
        temp_path = output_path.with_suffix('.temp.avi')
        frames_written = 0
        
        try:
            # If start/end times are provided, we can enforce constant frame rate
            # by duplicating frames if needed.
            if start_time is not None and end_time is not None:
                # Use iterator to pull frames as needed
                frame_iter = iter(frames)
                
                try:
                    next_ts, next_frame = next(frame_iter)
                    current_frame = next_frame
                except StopIteration:
                    logger.warning("No frames in generator")
                    return None
                
                # Initialize writer
                height, width = current_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                writer = cv2.VideoWriter(
                    str(temp_path),
                    fourcc,
                    self.fps,
                    (width, height)
                )
                
                if not writer.isOpened():
                    logger.error("Failed to create video writer")
                    return None
                
                # Iterate through time steps
                frame_duration = 1.0 / self.fps
                current_time = start_time
                
                while current_time < end_time:
                    # Advance to the correct frame for this timestamp
                    while next_ts is not None and next_ts <= current_time:
                        current_frame = next_frame
                        try:
                            next_ts, next_frame = next(frame_iter)
                        except StopIteration:
                            next_ts = None # No more frames
                            break
                    
                    # Write current frame
                    # Ensure frame is correct format
                    if current_frame.dtype != np.uint8:
                        current_frame = current_frame.astype(np.uint8)
                    
                    # Resize if needed
                    if current_frame.shape[:2] != (height, width):
                        current_frame = cv2.resize(current_frame, (width, height))
                    
                    writer.write(current_frame)
                    frames_written += 1
                    current_time += frame_duration
                    
            else:
                # Legacy mode: just write frames as they come
                for timestamp, frame in frames:
                    # Initialize writer on first frame
                    if writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                        writer = cv2.VideoWriter(
                            str(temp_path),
                            fourcc,
                            self.fps,
                            (width, height)
                        )
                        if not writer.isOpened():
                            logger.error("Failed to create video writer")
                            return None

                    # Ensure frame is correct format
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Resize if needed
                    if frame.shape[:2] != (writer.get(cv2.CAP_PROP_FRAME_HEIGHT), writer.get(cv2.CAP_PROP_FRAME_WIDTH)):
                         # Get current writer dimensions
                        h = int(writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w = int(writer.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame = cv2.resize(frame, (w, h))
                    
                    writer.write(frame)
                    frames_written += 1
            
            if writer:
                writer.release()
                
            if frames_written == 0:
                logger.warning("No frames written to video")
                return None
                
            logger.info(f"Saved video to {temp_path} ({frames_written} frames)")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error writing video: {e}")
            if writer:
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
        
        # Convert to int16 for better compatibility
        # Clip values to -1.0 to 1.0 before scaling to avoid overflow
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
            audio_temp = f.name
            audio_int16.tofile(f)
        
        try:
            # Use FFmpeg to mux audio and video
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-f", "s16le",
                "-ar", str(self.audio_sample_rate),
                "-ac", "1",
                "-i", audio_temp,
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "128k",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                # Clean up temp video file
                try:
                    if video_path.exists():
                        os.unlink(video_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp video: {e}")
                
                logger.info(f"Saved clip with audio to {output_path}")
                return output_path
            else:
                logger.warning(f"FFmpeg muxing failed: {result.stderr}")
                # Fall back to video-only (rename temp file)
                fallback_path = output_path.with_suffix('.avi')
                
                # Delete the failed mp4 if it exists
                if output_path.exists():
                    try:
                        os.unlink(output_path)
                    except:
                        pass
                        
                if video_path.exists():
                    if fallback_path.exists():
                        try:
                            os.unlink(fallback_path)
                        except:
                            pass
                    os.rename(video_path, fallback_path)
                return fallback_path
                
        except Exception as e:
            logger.error(f"Error muxing audio: {e}")
            # Fall back to video-only
            fallback_path = output_path.with_suffix('.avi')
            
            # Delete the failed mp4 if it exists
            if output_path.exists():
                try:
                    os.unlink(output_path)
                except:
                    pass
            
            if video_path.exists():
                if fallback_path.exists():
                    try:
                        os.unlink(fallback_path)
                    except:
                        pass
                os.rename(video_path, fallback_path)
            return fallback_path
            
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
            # Use generator to stream frames from disk buffer
            video_frames = buffer.yield_video_frames(start_time, end_time)
            audio_chunks = buffer.get_audio_chunks(start_time, end_time)
            
            # Save directly without going through buffer again
            path = self._save_direct(video_frames, audio_chunks, prefix, reason, start_time, end_time)
            
            if callback:
                callback(path)
        
        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()
        return thread
    
    def _save_direct(
        self,
        video_frames: Iterable[Tuple[float, np.ndarray]],
        audio_chunks: List[Tuple[float, np.ndarray]],
        prefix: str,
        reason: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Optional[Path]:
        """Save frames directly without buffer lookup"""
        
        # Note: We can't check if video_frames is empty here easily
        
        with self._save_lock:
            if reason:
                prefix = f"{prefix}_{reason.replace('+', '_')}"
            output_path = self._generate_filename(prefix)
            
            video_path = self._save_video_frames(video_frames, output_path, start_time, end_time)
            
            if video_path and audio_chunks:
                final_path = self._mux_audio(video_path, audio_chunks, output_path)
                if final_path:
                    return final_path
            
            # Fallback if no audio or muxing failed
            if video_path and video_path.exists():
                final_path = output_path.with_suffix('.avi')
                if final_path.exists():
                    os.unlink(final_path)
                os.rename(video_path, final_path)
                return final_path
            
            return None


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
