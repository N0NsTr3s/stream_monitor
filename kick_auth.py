"""
Kick Authentication CLI Wrapper
"""
import sys
import os
from pathlib import Path

# Add root to path to allow imports from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.auth.kick_auth import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
