"""
FFmpeg-based clipper (OpenCV-free alternative)
Saves clips using FFmpeg subprocess when OpenCV is unavailable.
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

from .circular_buffer import CircularBuffer

logger = logging.getLogger(__name__)


class FFmpegClipper:
    """
    Saves stream buffer contents as video clips using FFmpeg.
    Works without OpenCV - pure FFmpeg approach.
    """
    
    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1920, 1080),
        audio_sample_rate: int = 44100,
        output_format: str = "mp4"
    ):
        """
        Initialize clipper.
        
        Args:
            output_dir: Directory to save clips
            fps: Video frame rate
            resolution: Video resolution (width, height)
            audio_sample_rate: Audio sample rate
            output_format: Output file format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.resolution = resolution
        self.audio_sample_rate = audio_sample_rate
        self.output_format = output_format
        
        # Track saved clips
        self._clip_count = 0
        self._save_lock = threading.Lock()
        
        # Check for ffmpeg
        self._ffmpeg_available = self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except FileNotFoundError:
            logger.error("FFmpeg not found! Install from https://ffmpeg.org/")
            return False
    
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
        if not self._ffmpeg_available:
            logger.error("FFmpeg not available - cannot save clips")
            return None
        
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
        
        # Get resolution from first frame
        _, first_frame = video_frames[0]
        height, width = first_frame.shape[:2]
        
        # Sort frames by timestamp
        sorted_frames = sorted(video_frames, key=lambda x: x[0])
        
        # Create temp file for raw video
        video_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        video_temp_path = video_temp.name
        
        try:
            # Write raw frames to temp file
            for _, frame in sorted_frames:
                # Ensure BGR format and uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                video_temp.write(frame.tobytes())
            
            video_temp.close()
            
            # Prepare FFmpeg command
            if audio_chunks:
                # Sort and concatenate audio
                sorted_audio = sorted(audio_chunks, key=lambda x: x[0])
                audio_data = np.concatenate([chunk for _, chunk in sorted_audio])
                
                # Save audio to temp file
                audio_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
                audio_temp_path = audio_temp.name
                
                # Convert float32 to int16 for raw audio
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_temp.write(audio_int16.tobytes())
                audio_temp.close()
                
                # FFmpeg with video and audio
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}",
                    "-r", str(self.fps),
                    "-i", video_temp_path,
                    "-f", "s16le",
                    "-ar", str(self.audio_sample_rate),
                    "-ac", "1",
                    "-i", audio_temp_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
            else:
                # FFmpeg video only
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}",
                    "-r", str(self.fps),
                    "-i", video_temp_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                audio_temp_path = None
            
            # Run FFmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"Saved clip to {output_path}")
                return output_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
        finally:
            # Clean up temp files
            try:
                os.unlink(video_temp_path)
            except:
                pass
            
            if audio_chunks:
                try:
                    os.unlink(audio_temp_path)
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
        """
        def save_task():
            # Snapshot the buffer data
            video_frames = buffer.get_video_frames(start_time, end_time)
            audio_chunks = buffer.get_audio_chunks(start_time, end_time)
            
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
        
        if not video_frames or not self._ffmpeg_available:
            return None
        
        with self._save_lock:
            if reason:
                prefix = f"{prefix}_{reason.replace('+', '_')}"
            output_path = self._generate_filename(prefix)
            
            # Get resolution from first frame
            _, first_frame = video_frames[0]
            height, width = first_frame.shape[:2]
            
            # Sort frames
            sorted_frames = sorted(video_frames, key=lambda x: x[0])
            
            # Create temp file for raw video
            video_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
            video_temp_path = video_temp.name
            
            try:
                for _, frame in sorted_frames:
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    video_temp.write(frame.tobytes())
                video_temp.close()
                
                if audio_chunks:
                    sorted_audio = sorted(audio_chunks, key=lambda x: x[0])
                    audio_data = np.concatenate([chunk for _, chunk in sorted_audio])
                    
                    audio_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
                    audio_temp_path = audio_temp.name
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    audio_temp.write(audio_int16.tobytes())
                    audio_temp.close()
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{width}x{height}", "-r", str(self.fps),
                        "-i", video_temp_path,
                        "-f", "s16le", "-ar", str(self.audio_sample_rate), "-ac", "1",
                        "-i", audio_temp_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k",
                        "-shortest", "-movflags", "+faststart",
                        str(output_path)
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{width}x{height}", "-r", str(self.fps),
                        "-i", video_temp_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-movflags", "+faststart",
                        str(output_path)
                    ]
                    audio_temp_path = None
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    logger.info(f"Saved clip to {output_path}")
                    return output_path
                else:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    return None
                
            finally:
                try:
                    os.unlink(video_temp_path)
                except:
                    pass
                if audio_chunks:
                    try:
                        os.unlink(audio_temp_path)
                    except:
                        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .circular_buffer import CircularBuffer
    
    # Test clipper
    buffer = CircularBuffer(max_seconds=10.0)
    clipper = FFmpegClipper(output_dir=Path("./test_clips"))
    
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
