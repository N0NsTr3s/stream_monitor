"""
Clipper - Saves buffered stream data as video clips
"""
import logging
import os
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from .circular_buffer import CircularBuffer

logger = logging.getLogger(__name__)


class Clipper:
    """
    Saves stream buffer contents as video clips.
    Writes raw stream bytes to file.
    """
    
    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0, # Unused for raw clips
        resolution: Tuple[int, int] = (1920, 1080), # Unused
        audio_sample_rate: int = 44100, # Unused
        video_codec: str = "mp4v", # Unused
        output_format: str = "ts" # Default to TS for raw stream
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.fps = fps # Kept for compatibility
        self.resolution = resolution # Kept for compatibility
        
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
        
        # Get raw bytes from buffer
        data = buffer.get_stream_data(start_time, end_time)
        
        if not data:
            logger.warning("No data in buffer for clip")
            return None
        
        logger.info(f"Saving clip: {len(data) / 1024 / 1024:.2f} MB")
        
        # Generate output path
        if reason:
            prefix = f"{prefix}_{reason.replace('+', '_')}"
        output_path = self._generate_filename(prefix)
        
        # Write raw bytes
        try:
            with open(output_path, "wb") as f:
                f.write(data)
            
            logger.info(f"Saved raw clip to {output_path}")
            
            # Optional: Remux to MP4 if requested and not TS
            if self.output_format == "mp4":
                return self._remux_to_mp4(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error writing clip: {e}")
            return None

    def _remux_to_mp4(self, input_path: Path) -> Optional[Path]:
        """Remux TS to MP4 using FFmpeg"""
        output_path = input_path.with_suffix(".mp4")
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Delete original
                try:
                    os.unlink(input_path)
                except:
                    pass
                logger.info(f"Remuxed to {output_path}")
                return output_path
            else:
                logger.warning(f"Remux failed: {result.stderr}")
                return input_path # Return original
                
        except Exception as e:
            logger.error(f"Error remuxing: {e}")
            return input_path

    def save_clip_async(
        self,
        buffer: CircularBuffer,
        start_time: float,
        end_time: float,
        prefix: str = "clip",
        reason: str = "",
        callback: Optional[callable] = None
    ):
        """Save a clip asynchronously"""
        def save_task():
            path = self.save_clip(buffer, start_time, end_time, prefix, reason)
            if callback:
                callback(path)
        
        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()
        return thread
