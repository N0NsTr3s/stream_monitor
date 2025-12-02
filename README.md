# Stream Monitor

Automatically create clips from exciting moments in live streams.

## Features

- **Multi-Platform Support**: Twitch, YouTube Live, and Kick
- **Real-time Analysis**: Audio volume detection and chat activity monitoring  
- **Smart Clipping**: Scoring algorithm detects exciting moments
- **Configurable**: Adjust thresholds, pre/post-roll, signal weights
- **Automatic Buffer**: Never miss the start of a moment with circular buffer

## How It Works

1. **Stream Capture**: Uses `streamlink` to capture video/audio from the stream
2. **Chat Monitoring**: Platform-specific chat clients (twitchio, pytchat, WebSocket)
3. **Signal Analysis**: 
   - Audio RMS (volume) - detects shouting, excitement
   - Chat velocity (messages/second) - detects hype moments
4. **Scoring Engine**: Combines signals with configurable weights
5. **Clip Creation**: When score exceeds threshold, saves clip with pre-roll buffer

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Requirements

- **FFmpeg**: Required for audio processing and video muxing
  - Windows: `choco install ffmpeg` or download from https://ffmpeg.org/
  - Linux: `sudo apt install ffmpeg`
  - Mac: `brew install ffmpeg`

## Usage

```bash
# Basic usage - just provide a stream URL
python run.py https://www.twitch.tv/shroud

# YouTube Live
python run.py https://www.youtube.com/watch?v=VIDEO_ID

# Kick
python run.py https://kick.com/xqc

# With options
python run.py https://www.twitch.tv/shroud \
    --quality 720p \
    --output ./my_clips \
    --trigger-threshold 0.6 \
    --pre-roll 5 \
    --post-roll 10 \
    --verbose
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-q, --quality` | `best` | Stream quality (best, worst, 720p, etc.) |
| `-o, --output` | `clips` | Output directory for saved clips |
| `--trigger-threshold` | `0.7` | Score to start recording (0.0-1.0) |
| `--release-threshold` | `0.3` | Score to stop recording (0.0-1.0) |
| `--pre-roll` | `3.0` | Seconds before trigger to include |
| `--post-roll` | `5.0` | Seconds after release to include |
| `--audio-weight` | `0.4` | Weight for audio signal |
| `--chat-weight` | `0.6` | Weight for chat signal |
| `-v, --verbose` | `false` | Enable debug logging |

## Configuration

Copy `.env.example` to `.env` and fill in optional credentials:

```bash
# Twitch OAuth (optional - for authenticated chat features)
TWITCH_OAUTH_TOKEN=oauth:your_token_here

# YouTube API Key (optional - pytchat works without it)
YOUTUBE_API_KEY=your_key_here
```

## Project Structure

```
stream_monitor/
├── src/
│   ├── monitors/
│   │   ├── stream_capture.py   # Video/audio capture via streamlink
│   │   ├── chat_twitch.py      # Twitch IRC client
│   │   ├── chat_youtube.py     # YouTube Live Chat via pytchat
│   │   └── chat_kick.py        # Kick chat via WebSocket
│   ├── analysis/
│   │   ├── audio_analyzer.py   # RMS volume analysis
│   │   ├── chat_analyzer.py    # Chat velocity analysis
│   │   └── scoring_engine.py   # Combined scoring + state machine
│   ├── utils/
│   │   ├── platform_detector.py # URL parsing + platform detection
│   │   ├── circular_buffer.py   # Ring buffer for pre-roll
│   │   └── clipper.py           # Video/audio saving
│   ├── config.py               # Configuration management
│   └── stream_monitor.py       # Main controller
├── clips/                      # Output directory
├── requirements.txt
├── run.py                      # CLI entry point
└── README.md
```

## Scoring Algorithm

The scoring engine combines two signals:

```
Combined Score = (Audio Score × Audio Weight) + (Chat Score × Chat Weight)
```

**Audio Score** is based on RMS volume:
- Baseline is adaptively calculated from recent audio
- Score increases as volume exceeds baseline
- Max score when volume is `spike_multiplier` times baseline

**Chat Score** is based on message velocity:
- Messages per second in a sliding window
- Bonus for caps lock messages (excitement indicator)
- Bonus for emote density

**State Machine**:
1. `IDLE`: Monitoring, waiting for score > trigger threshold
2. `TRIGGERED`: Score exceeded threshold, recording starts
3. `RECORDING`: Continues while score is high
4. `COOLDOWN`: Post-roll period after score drops below release threshold

## Known Limitations

- **Kick.com**: May require browser cookies due to Cloudflare protection
- **YouTube**: Private/unlisted streams may not work with pytchat
- **Performance**: High-resolution streams may require significant CPU

## License

MIT
