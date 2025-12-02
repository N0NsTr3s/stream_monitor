#!/usr/bin/env python
"""
Twitch OAuth Token Manager - CLI Entry Point

Gets and manages Twitch OAuth tokens with automatic refresh.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.auth.twitch_auth import main

if __name__ == "__main__":
    asyncio.run(main())
