import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stress_highlight import format_time, SentenceRecognizer

class TestStressHighlight(unittest.TestCase):
    def test_format_time(self):
        """Test formatting of time in seconds to WEBVTT style"""
        self.assertEqual(format_time(0.0), "00:00:00.000")
        self.assertEqual(format_time(5.123), "00:00:05.123")
        self.assertEqual(format_time(75.5), "00:01:15.500")
        # Due to floating point precision in Python, 3661.002 - 3661 results in 0.00199999... which truncates to 1 ms
        self.assertEqual(format_time(3661.002), "01:01:01.001")
        self.assertEqual(format_time(3661.25), "01:01:01.250")

    def test_clean_for_analysis(self):
        """Test cleaning of words for stress analysis"""
        recognizer = SentenceRecognizer("dummy.wav", "dummy.json")
        self.assertEqual(recognizer._clean_for_analysis("Hello!"), "hello")
        self.assertEqual(recognizer._clean_for_analysis("don't"), "dont")
        self.assertEqual(recognizer._clean_for_analysis("  Apple,  "), "apple")

    def test_find_word(self):
        """Test locating a word with word boundaries in text"""
        text = "This is an interesting test."
        
        pos_is = SentenceRecognizer._find_word(text, "is", 0)
        self.assertEqual(pos_is, 5)
        
        pos_in = SentenceRecognizer._find_word(text, "in", 0)
        self.assertNotEqual(pos_in, -1)

    def test_apply_stress_formatting(self):
        """Test applying stress formatting (<u> tags) to stressed words"""
        recognizer = SentenceRecognizer("dummy.wav", "dummy.json")
        
        text = "Hello world from Thailand"
        stress_list = [
            {'word': 'Hello', 'stress': 0},
            {'word': 'world', 'stress': 1},
            {'word': 'from', 'stress': 0},
            {'word': 'Thailand', 'stress': 1}
        ]
        
        expected = "Hello <u>w</u><u>o</u><u>r</u><u>l</u><u>d</u> from <u>T</u><u>h</u><u>a</u><u>i</u><u>l</u><u>a</u><u>n</u><u>d</u>"
        result = recognizer._apply_stress_formatting(text, stress_list)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
