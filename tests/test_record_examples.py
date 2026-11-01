"""Video capture rejects resizing instead of silently corrupting rawvideo."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("record_examples", TOOLS / "record_examples.py")
RECORDER = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(RECORDER)
finally:
    sys.path.pop(0)


class TestVideoDimensions(unittest.TestCase):
    def test_resize_is_rejected_before_writing_wrong_frame_bytes(self):
        process = Mock()
        process.wait.return_value = 0
        with patch.object(RECORDER.subprocess, "Popen", return_value=process):
            clip = RECORDER.Clip(Path("unused.mp4"), 60)
            clip.write(np.zeros((12, 16, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "Capture dimensions changed"):
                clip.write(np.zeros((14, 16, 3), dtype=np.uint8))
            clip.close()
        self.assertEqual(clip.count, 1)
        self.assertEqual(process.stdin.write.call_count, 1)
        self.assertEqual(len(process.stdin.write.call_args.args[0]), 12 * 16 * 3)


if __name__ == "__main__":
    unittest.main()
