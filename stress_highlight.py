import os
import re
import json
import numpy as np
from audio_analyzer import AudioAnalyzer


def format_time(seconds):
    """Helper function to format time for VTT files"""
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class SentenceRecognizer:
    def __init__(self, audio_file, json_file):
        self.audio_file = audio_file
        self.json_file = json_file
        # ``text_sentences`` holds the raw segment text. ``sentences`` holds the
        # per-word data for segments that produced at least one usable word.
        # ``sentence_indices`` records, for each entry in ``sentences``, the
        # index of the matching entry in ``text_sentences`` so we can pair the
        # two lists back together even when some segments are skipped.
        self.text_sentences = []
        self.sentences = []
        self.sentence_indices = []
        # Share a single AudioAnalyzer across the whole pipeline so that the
        # cached librosa/soundfile data is reused for every word/segment.
        self._analyzer = AudioAnalyzer(audio_file)

    def collect_data(self):
        """Collect data from JSON file while preserving original text"""
        with open(self.json_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        for segment in data["segments"]:
            text_index = len(self.text_sentences)
            self.text_sentences.append(segment["text"])

            words_data = self._extract_words(segment)
            if words_data:
                self.sentences.extend(words_data)
                # Each call to ``_extract_words`` returns at most one list.
                self.sentence_indices.extend([text_index] * len(words_data))

    def _extract_words(self, segment):
        """Extract words while preserving original text and timing"""
        try:
            words_in_sentence = []
            all_words = []

            for word_info in segment.get("words", []):
                original_word = word_info["word"]
                analysis_word = self._clean_for_analysis(original_word)

                start = word_info["start"]
                end = word_info["end"]

                # Ensure valid time range so amplitude/pitch helpers have a
                # non-empty window to inspect.
                if start == end:
                    end += 0.01

                amp = self._analyzer._calculate_average_amplitude(start, end)
                if amp is not None:
                    words_in_sentence.append({
                        'original': original_word,
                        'analysis': analysis_word,
                        'start': start,
                        'end': end,
                        'amplitude': amp,
                    })

            if words_in_sentence:
                all_words.append(words_in_sentence)

            return all_words

        except Exception as e:
            print(f"Error in _extract_words: {str(e)}")
            return []

    def _clean_for_analysis(self, word):
        """Clean word for stress analysis only"""
        analysis_word = re.sub(r'[^\w\s]', '', word)
        return analysis_word.strip().lower()

    def _calculate_stress(self):
        """Calculate stress values based purely on audio analysis"""
        try:
            stress_lists = []

            for sentence in self.sentences:
                if not sentence:
                    stress_lists.append([])
                    continue

                sent_start = sentence[0]['start']
                sent_end = sentence[-1]['end']
                sent_amp = self._analyzer._calculate_average_amplitude(sent_start, sent_end)
                sent_pitch = self._analyzer._get_pitch_avg(sent_start, sent_end)

                all_amps = [word_data['amplitude'] for word_data in sentence]
                max_amp = max(all_amps) if all_amps else 0

                word_stress_list = []
                for word_data in sentence:
                    original_word = word_data['original']

                    word_amp = word_data['amplitude']
                    word_pitch = self._analyzer._get_pitch_avg(
                        word_data['start'],
                        word_data['end'],
                    )

                    stress = 0
                    if word_amp > (sent_amp * 1.5) and word_amp >= (max_amp * 0.8):
                        stress = 1
                    elif word_pitch and sent_pitch and word_pitch > (sent_pitch * 1.4):
                        stress = 1

                    word_stress_list.append({
                        'word': original_word.strip(),
                        'stress': stress,
                    })

                stress_lists.append(word_stress_list)

            return stress_lists

        except Exception as e:
            print(f"Error in _calculate_stress: {str(e)}")
            return []

    @staticmethod
    def _find_word(text, word, current_pos):
        """Locate ``word`` in ``text`` at or after ``current_pos`` honouring
        word boundaries.

        ``str.find`` would happily match short words like ``"in"`` inside
        ``"interesting"`` which then mis-aligns the underline output. We use
        ``\\b`` for tokens that start/end with word characters; for tokens
        whose edges are already punctuation we fall back to a literal match
        because ``\\b`` does not anchor between two non-word characters.
        """
        if not word:
            return -1

        pattern_parts = []
        if word[:1].isalnum() or word[:1] == "_":
            pattern_parts.append(r"\b")
        pattern_parts.append(re.escape(word))
        if word[-1:].isalnum() or word[-1:] == "_":
            pattern_parts.append(r"\b")
        pattern = "".join(pattern_parts)

        try:
            match = re.compile(pattern).search(text, current_pos)
        except re.error:
            return text.find(word, current_pos)
        if match:
            return match.start()
        # Fall back to a substring search so words containing characters that
        # were not in the original transcription (rare) still get processed.
        return text.find(word, current_pos)

    def _apply_stress_formatting(self, text, stress_list):
        """Apply stress formatting while preserving whitespace and punctuation"""
        try:
            if not text or not stress_list:
                return text

            result = []
            current_pos = 0

            def is_letter(char):
                return char.isalpha()

            for word_data in stress_list:
                word = word_data['word']
                if not word.strip():
                    continue

                word_pos = self._find_word(text, word, current_pos)
                if word_pos == -1:
                    continue

                result.append(text[current_pos:word_pos])

                if word_data['stress'] == 1:
                    parts = word.split()
                    formatted_parts = []
                    for part in parts:
                        formatted_chars = []
                        for char in part:
                            if is_letter(char):
                                formatted_chars.append(f"<u>{char}</u>")
                            else:
                                formatted_chars.append(char)
                        formatted_parts.append(''.join(formatted_chars))
                    result.append(" ".join(formatted_parts))
                else:
                    result.append(word)

                current_pos = word_pos + len(word)

            result.append(text[current_pos:])

            return "".join(result)

        except Exception as e:
            print(f"Error in _apply_stress_formatting: {str(e)}")
            return text

    def generate_vtt(self, vtt_filename):
        """Generate VTT file with stress formatting."""
        self.collect_data()
        stress_lists = self._calculate_stress()

        formatted_segments = []
        for i, sentence in enumerate(self.sentences):
            if not sentence:
                continue
            if i >= len(stress_lists):
                break
            text_index = self.sentence_indices[i] if i < len(self.sentence_indices) else i
            if text_index >= len(self.text_sentences):
                continue
            segment_text = self.text_sentences[text_index]

            formatted_text = self._apply_stress_formatting(segment_text, stress_lists[i])
            if not formatted_text:
                continue

            start_time = sentence[0]['start']
            end_time = sentence[-1]['end']
            formatted_segments.append((start_time, end_time, formatted_text))

        with open(vtt_filename, "w", encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, (start, end, text) in enumerate(formatted_segments, 1):
                f.write(f"{i}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text}\n\n")


if __name__ == "__main__":
    audio_file = "input_audio.wav"
    json_file = "whisper_output.json"
    vtt_file = "output.vtt"

    recognizer = SentenceRecognizer(audio_file, json_file)
    recognizer.generate_vtt(vtt_file)