"""
Workaround for mmpose dependency issues
This module creates dummy mmpose imports to allow the app to run
while using mediapipe or other alternatives for pose detection
"""

import sys
from unittest.mock import MagicMock

# Create mock mmpose module
mmpose = MagicMock()
mmpose.apis = MagicMock()
mmpose.structures = MagicMock()

# Add to sys.modules so imports don't fail
sys.modules['mmpose'] = mmpose
sys.modules['mmpose.apis'] = mmpose.apis
sys.modules['mmpose.structures'] = mmpose.structures

print("✓ mmpose workaround loaded")
