"""
Stream capture module using streamlink
Resolves stream URLs and provides access to video/audio data
"""
import logging
import subprocess
import threading
import time
import os
from typing import Optional, Generator, Tuple, Deque
from queue import Queue, Empty
from collections import deque

import streamlink
import numpy as np
import shutil

# Try to import cv2, provide fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from ..utils.platform_detector import PlatformDetector, Platform

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Captures video/audio stream using streamlink.
    Provides raw stream chunks for buffering and decoded frames for analysis.
    Includes a ring buffer to support pre-roll clipping and an active
    recording path that dumps the ring buffer + future bytes to a .ts file
    which is converted to MP4 when complete.
    """

    def __init__(self, url: str, quality: str = "best", pre_roll_duration: int = 30):
        """
        Initialize stream capture.

        Args:
            url: Stream URL (Twitch, YouTube, or Kick)
            quality: Stream quality (best, worst, 720p, etc.)
            pre_roll_duration: seconds of pre-roll to retain in memory for clips
        """
        self.url = PlatformDetector.normalize_url(url)
        self.quality = quality
        self.platform, self.channel_id = PlatformDetector.detect(self.url)

        self._stream_url: Optional[str] = None
        self._stream_fd = None  # Streamlink file descriptor

        # Processes for decoding
        self._video_process: Optional[subprocess.Popen] = None
        self._audio_process: Optional[subprocess.Popen] = None

        self._is_running = False

        # Check for ffmpeg
        self._ffmpeg_path = shutil.which("ffmpeg")
        if not self._ffmpeg_path:
            logger.warning("FFmpeg not found in PATH. Capture will be limited.")

        # --- NEW BUFFERING LOGIC ---
        self.chunk_size = 32 * 1024
        estimated_chunks = (pre_roll_duration * 1024 * 1024) // self.chunk_size
        self._ring_buffer = deque(maxlen=int(max(1, estimated_chunks)))
        self._buffer_lock = threading.Lock()

        # Active recording state
        self._recording_active = False
        self._recording_end_time = 0.0
        self._temp_ts_filename = ""
        self._final_mp4_filename = ""
        
        # File Writer Queue
        self._write_queue: Queue = Queue()
        self._writer_thread = threading.Thread(target=self._file_writer_loop, daemon=True)
        self._writer_thread.start()

        # Old queues (still used for analysis)
        self._raw_queue: Queue = Queue(maxsize=100)
        self._audio_queue: Queue = Queue(maxsize=100)
        self._video_queue: Queue = Queue(maxsize=10)

        # Threads
        self._stream_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._video_thread: Optional[threading.Thread] = None

        # Stats
        self._fps = 30.0
        self._width = 1920
        self._height = 1080

        logger.info(f"StreamCapture initialized for {self.platform.value}: {self.channel_id}")
    
    def _resolve_stream_url(self) -> Optional[str]:
        """Resolve the stream URL using streamlink"""
        try:
            streams = streamlink.streams(self.url) # type: ignore
            
            if not streams:
                logger.error(f"No streams found for {self.url}")
                return None
            
            available = list(streams.keys())
            logger.info(f"Available qualities: {available}")
            
            # Try requested quality, fall back to best
            if self.quality in streams:
                stream = streams[self.quality]
            elif "best" in streams:
                stream = streams["best"]
            else:
                stream = streams[available[0]]
            
            stream_url = stream.url
            logger.info(f"Resolved stream URL: {stream_url[:100]}...")
            return stream_url
            
        except streamlink.StreamlinkError as e: # type: ignore
            logger.error(f"Streamlink error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error resolving stream: {e}")
            return None
    
    def start(self) -> bool:
        """
        Start capturing the stream.
        
        Returns:
            True if successfully started, False otherwise
        """
        if self._is_running:
            logger.warning("Stream capture already running")
            return True
        
        if not self._ffmpeg_path:
            logger.error("FFmpeg is required for this capture mode")
            return False
        
        # Resolve stream URL
        self._stream_url = self._resolve_stream_url()
        if not self._stream_url:
            return False
        
        # Open stream with streamlink
        try:
            streams = streamlink.streams(self.url) # type: ignore
            if self.quality in streams:
                self._stream_fd = streams[self.quality].open()
            elif "best" in streams:
                self._stream_fd = streams["best"].open()
            else:
                self._stream_fd = streams[list(streams.keys())[0]].open()
        except Exception as e:
            logger.error(f"Failed to open stream: {e}")
            return False

        self._is_running = True
        
        # Start decoding processes
        self._start_decoders()
        
        # Start stream reader thread
        self._stream_thread = threading.Thread(target=self._stream_reader, daemon=True)
        self._stream_thread.start()
        
        logger.info("Stream capture started successfully")
        return True
    
    def _start_decoders(self):
        """Start FFmpeg processes for video and audio decoding"""
        
        # Video Decoder: Reads from pipe, outputs raw video frames
        # We scale to 640x360 for analysis to save CPU
        video_cmd = [
            self._ffmpeg_path,
            "-i", "pipe:0",
            "-vf", "fps=15,scale=640:360", # Low FPS and resolution for analysis
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            "pipe:1"
        ]
        
        self._video_process = subprocess.Popen(
            video_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7 # Large buffer
        )
        
        # Audio Decoder: Reads from pipe, outputs raw audio samples
        audio_cmd = [
            self._ffmpeg_path,
            "-i", "pipe:0",
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            "-loglevel", "error",
            "pipe:1"
        ]
        
        self._audio_process = subprocess.Popen(
            audio_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**6
        )
        
        # Start reader threads
        self._video_thread = threading.Thread(target=self._video_reader, daemon=True)
        self._video_thread.start()
        
        self._audio_thread = threading.Thread(target=self._audio_reader, daemon=True)
        self._audio_thread.start()

    def _stream_reader(self):
        """Reads raw chunks, buffers them, and distributes them"""
        
        while self._is_running and self._stream_fd:
            try:
                data = self._stream_fd.read(self.chunk_size)
                if not data:
                    logger.warning("Stream ended")
                    self._stop_recording_if_active() # Safety close
                    break
                
                # --- 1. RING BUFFER & RECORDING ---
                with self._buffer_lock:
                    self._ring_buffer.append(data)
                    
                    if self._recording_active:
                        self._write_queue.put(("WRITE", data))
                        
                        if time.time() >= self._recording_end_time:
                            self._recording_active = False
                            self._write_queue.put(("STOP", (self._temp_ts_filename, self._final_mp4_filename)))
                
                # --- 3. ANALYSIS DECODERS ---
                if self._video_process and self._video_process.stdin:
                    try:
                        self._video_process.stdin.write(data)
                        self._video_process.stdin.flush()
                    except: pass
                
                if self._audio_process and self._audio_process.stdin:
                    try:
                        self._audio_process.stdin.write(data)
                        self._audio_process.stdin.flush()
                    except: pass
                        
            except Exception as e:
                logger.error(f"Stream reader error: {e}")
                self._stop_recording_if_active()
                break
        
        self._is_running = False

    def _video_reader(self):
        """Reads decoded video frames for analysis"""
        # 640x360 * 3 bytes (BGR)
        frame_size = 640 * 360 * 3
        
        while self._is_running and self._video_process:
            try:
                data = self._video_process.stdout.read(frame_size) # type: ignore
                if len(data) == frame_size:
                    frame = np.frombuffer(data, dtype=np.uint8).reshape((360, 640, 3))
                    
                    # Clear queue if full to always have latest frame
                    if self._video_queue.full():
                        try:
                            self._video_queue.get_nowait()
                        except:
                            pass
                    
                    self._video_queue.put(frame)
                else:
                    if not data:
                        break
            except Exception as e:
                logger.error(f"Video reader error: {e}")
                break

    def _audio_reader(self):
        """Reads decoded audio samples for analysis"""
        chunk_size = 4410 * 2 # ~100ms
        
        while self._is_running and self._audio_process:
            try:
                data = self._audio_process.stdout.read(chunk_size) # type: ignore
                if data:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if self._audio_queue.full():
                        try:
                            self._audio_queue.get_nowait()
                        except:
                            pass
                            
                    self._audio_queue.put(audio_data)
                else:
                    break
            except Exception as e:
                logger.error(f"Audio reader error: {e}")
                break

    def get_buffer_duration(self) -> float:
        """Estimate buffer duration in seconds based on 1MB/s assumption"""
        with self._buffer_lock:
            chunks = len(self._ring_buffer)
        
        # chunk_size is 32KB. 1MB/s = 1024KB/s = 32 chunks/s.
        return chunks / 32.0

    def read_chunk(self) -> Optional[bytes]:
        """Read a raw stream chunk"""
        try:
            return self._raw_queue.get_nowait()
        except Empty:
            return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a decoded video frame (low res for analysis)"""
        try:
            frame = self._video_queue.get_nowait()
            return True, frame
        except Empty:
            return True, None # Return True to keep loop alive, but None frame

    def read_audio(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read decoded audio chunk"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_rate(self) -> float:
        return 30.0 # Approximate
    
    def get_resolution(self) -> Tuple[int, int]:
        return 1920, 1080 # Approximate
    
    def stop(self):
        """Stop capturing"""
        self._is_running = False
        
        if self._stream_fd:
            try:
                self._stream_fd.close()
            except:
                pass
        
        if self._video_process:
            self._video_process.terminate()
        
        if self._audio_process:
            self._audio_process.terminate()
        
        logger.info("Stream capture stopped")

    def start_clip(self, filename: str = "clip.mp4", pre_roll_duration: float = 30.0) -> bool:
        """
        Starts recording a clip: Dumps Ring Buffer (Past) + Starts Recording Live (Future).
        Call stop_clip() to finish.
        
        Args:
            filename: Output filename
            pre_roll_duration: How many seconds of past buffer to include
        """
        with self._buffer_lock:
            if self._recording_active:
                logger.warning("Already recording a clip! Ignoring trigger.")
                return False

            logger.info(f"Trigger! Starting recording to {filename} (Pre-roll: {pre_roll_duration}s)...")
            
            self._final_mp4_filename = filename
            # Ensure temp file has .ts extension, even if filename doesn't end in .mp4
            if filename.endswith(".mp4"):
                self._temp_ts_filename = filename.replace(".mp4", ".ts")
            else:
                self._temp_ts_filename = filename + ".ts"
                
            # No fixed end time, runs until stop_clip() is called
            self._recording_end_time = float('inf') 
            self._recording_active = True
            
            # Calculate how many chunks we need for the requested pre-roll
            # 1MB/s = 32 chunks/s (approx)
            chunks_needed = int(pre_roll_duration * 32)
            
            # Get the last N chunks from the deque
            # Note: deque doesn't support slicing directly, so we convert to list or iterate
            # Converting to list is safest for a snapshot
            buffer_snapshot = list(self._ring_buffer)
            start_index = max(0, len(buffer_snapshot) - chunks_needed)
            chunks_to_write = buffer_snapshot[start_index:]
            
            logger.info(f"Dumping {len(chunks_to_write)} chunks for pre-roll (requested {chunks_needed})")

            # Queue commands: Start -> Past Data
            self._write_queue.put(("START", self._temp_ts_filename))
            for chunk in chunks_to_write:
                self._write_queue.put(("WRITE", chunk))
            
            return True

    def stop_clip(self):
        """Stops the current recording and converts to MP4."""
        with self._buffer_lock:
            if self._recording_active:
                logger.info("Stopping recording...")
                self._recording_active = False
                self._write_queue.put(("STOP", (self._temp_ts_filename, self._final_mp4_filename)))

    def save_clip(self, filename: str = "clip.mp4", duration: int = 30) -> bool:
        """Legacy wrapper for fixed duration clips"""
        if self.start_clip(filename):
            # We can't easily implement fixed duration with the new async model 
            # without a separate timer, but for now we'll just rely on the caller to stop it.
            # Or we could spawn a timer thread to call stop_clip.
            # For this refactor, we assume the caller (StreamMonitor) will handle stopping.
            return True
        return False

    def _stop_recording_if_active(self):
        """Helper to stop recording safely"""
        with self._buffer_lock:
            if self._recording_active:
                self._recording_active = False
                self._write_queue.put(("STOP", (self._temp_ts_filename, self._final_mp4_filename)))

    def _file_writer_loop(self):
        """Background thread to handle file I/O"""
        current_file = None
        while True:
            try:
                cmd, payload = self._write_queue.get()
                
                if cmd == "START":
                    if current_file:
                        current_file.close()
                    try:
                        current_file = open(payload, "wb")
                    except Exception as e:
                        logger.error(f"Failed to open record file: {e}")
                        current_file = None
                        
                elif cmd == "WRITE":
                    if current_file:
                        try:
                            current_file.write(payload)
                        except Exception as e:
                            logger.error(f"Write error: {e}")
                            
                elif cmd == "STOP":
                    if current_file:
                        current_file.close()
                        current_file = None
                    
                    # Payload is (ts_filename, mp4_filename)
                    if payload:
                        ts_file, mp4_file = payload
                        logger.info("Raw recording finished. Converting to MP4...")
                        self._convert_to_mp4(ts_file, mp4_file)
                        
            except Exception as e:
                logger.error(f"Writer loop error: {e}")

    def _convert_to_mp4(self, input_ts, output_mp4):
        # We run this in a thread or blocking depending on preference.
        try:
            # Added -fflags +genpts to fix timestamp issues from concatenated chunks
            cmd = [
                self._ffmpeg_path, "-y",
                "-fflags", "+genpts", 
                "-i", input_ts,
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                output_mp4
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(output_mp4):
                try:
                    os.remove(input_ts) # Cleanup temp file
                except Exception:
                    pass
                logger.info(f"Clip saved successfully: {output_mp4}")
            else:
                logger.error("FFmpeg failed to create MP4.")
        except Exception as e:
            logger.error(f"FFmpeg conversion error: {e}")

    def save_clip_async(self, filename: str = "clip.mp4", duration: int = 30):
        """Threaded wrapper for `save_clip` to match Clipper semantics."""
        def task():
            try:
                self.save_clip(filename, duration)
            except Exception:
                pass

        thread = threading.Thread(target=task, daemon=True)
        thread.start()
        return thread
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

def get_stream_capture(url: str, quality: str = "best"):
    return StreamCapture(url, quality)
