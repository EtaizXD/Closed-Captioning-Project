import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Ensure the parent directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_analyzer import AudioAnalyzer

class TestAudioAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create an instance with a dummy file path
        self.analyzer = AudioAnalyzer("dummy_audio.wav")

    def test_safe_n_fft(self):
        """Test calculation of safe n_fft for FFT operations"""
        self.assertEqual(self.analyzer._safe_n_fft(15), 0)
        self.assertEqual(self.analyzer._safe_n_fft(30), 0)
        self.assertEqual(self.analyzer._safe_n_fft(35), 32)
        self.assertEqual(self.analyzer._safe_n_fft(100), 64)
        self.assertEqual(self.analyzer._safe_n_fft(512), 512)

    def test_peak_pitches(self):
        """Test finding peak pitches"""
        self.assertEqual(self.analyzer._peak_pitches(np.array([]), np.array([])), [])

        pitches = np.array([
            [100.0, 150.0, 0.0],
            [200.0, 250.0, 300.0]
        ])
        magnitudes = np.array([
            [0.1, 0.9, 0.1],
            [0.8, 0.2, 0.7]
        ])
        expected = [200.0, 150.0, 300.0]
        result = self.analyzer._peak_pitches(pitches, magnitudes)
        self.assertEqual(result, expected)

    @patch('soundfile.read')
    def test_calculate_average_amplitude(self, mock_sf_read):
        """Test calculating average amplitude in a given time range"""
        dummy_audio = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        mock_sf_read.return_value = (dummy_audio, 1000)

        avg_amp = self.analyzer._calculate_average_amplitude(0.0, 0.003)
        self.assertAlmostEqual(avg_amp, 0.2)
        self.assertEqual(self.analyzer._calculate_average_amplitude(0.005, 0.001), 0.0)

    @patch('soundfile.read')
    def test_get_min_max_amplitudes(self, mock_sf_read):
        """Test getting minimum and maximum amplitudes in a time range"""
        dummy_audio = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        mock_sf_read.return_value = (dummy_audio, 1000)

        min_amp, max_amp = self.analyzer._get_min_max_amplitudes("dummy_audio.wav", 0.0, 0.004)
        self.assertAlmostEqual(min_amp, -0.4)
        self.assertAlmostEqual(max_amp, 0.3)


if __name__ == '__main__':
    unittest.main()
