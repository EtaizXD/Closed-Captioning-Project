import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_recognition import (
    _minimum_available_memory_gb,
    _build_transcribe_kwargs,
    _normalise_sensitivity
)

class TestSentenceRecognition(unittest.TestCase):
    def test_minimum_available_memory_gb(self):
        """Test RAM requirement safety guard for different model sizes and devices"""
        self.assertEqual(_minimum_available_memory_gb("large-v3", "cuda"), 0.0)
        self.assertEqual(_minimum_available_memory_gb("large-v3", "cpu"), 8.0)
        self.assertEqual(_minimum_available_memory_gb("medium", "cpu"), 4.0)
        self.assertEqual(_minimum_available_memory_gb("small", "cpu"), 2.0)
        self.assertEqual(_minimum_available_memory_gb("tiny", "cpu"), 0.0)

    def test_normalise_sensitivity(self):
        """Test sensitivity argument coercion to standard levels"""
        self.assertEqual(_normalise_sensitivity(True), "sensitive")
        self.assertEqual(_normalise_sensitivity(False), "off")
        self.assertEqual(_normalise_sensitivity("ultra"), "ultra")
        self.assertEqual(_normalise_sensitivity("SENSITIVE"), "sensitive")
        self.assertEqual(_normalise_sensitivity("invalid_value"), "off")

    def test_build_transcribe_kwargs(self):
        """Test build transcribe kwargs based on sensitivity tier"""
        off_kwargs = _build_transcribe_kwargs("off")
        self.assertTrue(off_kwargs["word_timestamps"])
        self.assertNotIn("no_speech_threshold", off_kwargs)

        sensitive_kwargs = _build_transcribe_kwargs("sensitive")
        self.assertEqual(sensitive_kwargs["no_speech_threshold"], 0.2)

        ultra_kwargs = _build_transcribe_kwargs("ultra")
        self.assertEqual(ultra_kwargs["no_speech_threshold"], 0.05)
        self.assertEqual(ultra_kwargs["beam_size"], 10)


if __name__ == '__main__':
    unittest.main()
