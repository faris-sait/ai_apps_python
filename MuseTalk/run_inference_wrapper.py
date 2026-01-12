#!/usr/bin/env python3
"""Wrapper script to run MuseTalk inference with proper Python path"""
import sys
from pathlib import Path

# Add MuseTalk directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Now run the actual inference script
if __name__ == "__main__":
    from scripts import inference
    inference.main()
