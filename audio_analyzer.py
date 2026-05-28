import numpy as np
import librosa
import soundfile as sf


class AudioAnalyzer:
    """Cached audio analyzer.

    The audio file is read at most once per backend (librosa, soundfile) per
    instance. Reusing a single :class:`AudioAnalyzer` across the whole stress
    pipeline avoids reloading the entire file for every word/segment.
    """

    # Minimum samples needed before piptrack can run with a sane n_fft.
    _MIN_PITCH_SAMPLES = 32

    def __init__(self, audio_file):
        self.audio_file = audio_file
        self._librosa_audio = None  # tuple (y, sr) loaded lazily
        self._sf_audio = None       # tuple (data, sample_rate) loaded lazily

    # ------------------------------------------------------------------ helpers
    def _librosa_data(self):
        if self._librosa_audio is None:
            y, sr = librosa.load(self.audio_file, sr=None)
            self._librosa_audio = (y, sr)
        return self._librosa_audio

    def _soundfile_data(self):
        if self._sf_audio is None:
            data, sample_rate = sf.read(self.audio_file)
            self._sf_audio = (data, sample_rate)
        return self._sf_audio

    def _segment_samples(self, start_time, end_time):
        """Return the librosa segment for ``[start_time, end_time]``.

        Returns ``(y_segment, sr)``. ``y_segment`` may be shorter than
        requested if the times exceed the file length; it may also be empty.
        """
        y, sr = self._librosa_data()
        start_sample = max(0, int(start_time * sr))
        end_sample = min(len(y), int(end_time * sr))
        if end_sample <= start_sample:
            return np.empty(0, dtype=y.dtype), sr
        return y[start_sample:end_sample], sr

    @staticmethod
    def _peak_pitches(pitches, magnitudes):
        """Return the per-frame peak pitch (Hz) where pitch > 0."""
        if pitches.size == 0 or magnitudes.size == 0:
            return []
        peak_bins = magnitudes.argmax(axis=0)
        frames = np.arange(pitches.shape[1])
        peak_pitches = pitches[peak_bins, frames]
        return peak_pitches[peak_pitches > 0].tolist()

    @staticmethod
    def _safe_n_fft(num_samples):
        """Pick the largest power-of-two n_fft that fits in ``num_samples``.

        Returns 0 when the segment is too short to run piptrack reliably.
        """
        if num_samples < AudioAnalyzer._MIN_PITCH_SAMPLES:
            return 0
        return 2 ** int(np.floor(np.log2(num_samples)))

    # --------------------------------------------------------------- pitch APIs
    def _get_pitch_avg(self, start_time, end_time, n_fft=512):
        y_segment, sr = self._segment_samples(start_time, end_time)
        n_fft = self._safe_n_fft(len(y_segment))
        if n_fft == 0:
            return 0.0

        pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr, n_fft=n_fft)
        pitch_values = self._peak_pitches(pitches, magnitudes)
        return float(np.mean(pitch_values)) if pitch_values else 0.0

    def _get_pitch_max(self, start_time, end_time):
        y_segment, sr = self._segment_samples(start_time, end_time)
        n_fft = self._safe_n_fft(len(y_segment))
        if n_fft == 0:
            return 0.0

        pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr, n_fft=n_fft)
        pitch_values = self._peak_pitches(pitches, magnitudes)
        return float(np.max(pitch_values)) if pitch_values else 0.0

    def _get_pitch_top10_avg(self, start_time, end_time, chunk_size=0.1):
        y_segment, sr = self._segment_samples(start_time, end_time)
        if y_segment.size == 0:
            return 0.0

        chunk_length = int(chunk_size * sr)
        if chunk_length <= 0:
            return 0.0
        num_chunks = len(y_segment) // chunk_length
        if num_chunks == 0:
            return 0.0

        avg_pitches = []
        for i in range(num_chunks):
            chunk_start = i * chunk_length
            chunk_end = chunk_start + chunk_length
            y_chunk = y_segment[chunk_start:chunk_end]

            n_fft = self._safe_n_fft(len(y_chunk))
            if n_fft == 0:
                avg_pitches.append(0.0)
                continue

            pitches, magnitudes = librosa.core.piptrack(y=y_chunk, sr=sr, n_fft=n_fft)
            pitch_values = self._peak_pitches(pitches, magnitudes)
            avg_pitches.append(float(np.mean(pitch_values)) if pitch_values else 0.0)

        top_10 = sorted(avg_pitches, reverse=True)[:10]
        return float(np.mean(top_10)) if top_10 else 0.0

    # ----------------------------------------------------------- amplitude APIs
    def _calculate_average_amplitude(self, start_time, end_time):
        try:
            audio_data, sample_rate = self._soundfile_data()
            start_sample = max(0, int(start_time * sample_rate))
            end_sample = min(len(audio_data), int(end_time * sample_rate))
            if end_sample <= start_sample:
                return 0.0
            audio_segment = audio_data[start_sample:end_sample]
            amplitude = np.abs(audio_segment)
            return float(np.mean(amplitude))
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def _get_min_max_amplitudes(self, audio_file_path, start_time, end_time):
        try:
            # ``audio_file_path`` is kept for backward compatibility, but we use
            # the cached audio data when it points to the same file we already
            # loaded. Otherwise fall back to a fresh read.
            if audio_file_path == self.audio_file:
                audio_data, sample_rate = self._soundfile_data()
            else:
                audio_data, sample_rate = sf.read(audio_file_path)

            start_sample = max(0, int(start_time * sample_rate))
            end_sample = min(len(audio_data), int(end_time * sample_rate))
            if end_sample <= start_sample:
                return 0.0, 0.0

            audio_segment = audio_data[start_sample:end_sample]
            min_amplitude = float(np.min(audio_segment))
            max_amplitude = float(np.max(audio_segment))
            return min_amplitude, max_amplitude

        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
