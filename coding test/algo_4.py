import os
import re
import json
import numpy as np
import soundfile as sf
import nltk
from nltk.corpus import cmudict

# Ensure nltk resources are downloaded
nltk.download("cmudict")

# Load CMU Pronouncing Dictionary
pronouncing_dict = cmudict.dict()

# Dictionary of word-to-number mappings
word_to_number = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


# Helper function to format time for VTT files
def format_time(seconds):
    milliseconds = int(seconds * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(
        hours, minutes, seconds, milliseconds % 1000
    )


# Main class for sentence recognition and VTT generation
class SentenceRecognizer:
    def __init__(self, audio_file, json_file):
        self.audio_file = audio_file
        self.json_file = json_file
        self.text_sentences = []
        self.sentences = []

    def sentence_recognition(self):
        command = f'whisper "{self.audio_file}" --model medium.en --word_timestamps True --highlight_words True'
        os.popen(command).read()

    def collect_data(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)
        for segment in data["segments"]:
            words_data = self._extract_words(segment)
            self.sentences.extend(words_data)
            self.text_sentences.append(segment["text"])

    def _extract_words(self, segment):
        words_in_sentence = []
        all_words = []
        for idx, word_info in enumerate(segment["words"]):
            word = self._format_word(word_info["word"].strip())
            start = word_info["start"]
            end = word_info["end"]

            if start == end:
                end += 0.01  # Ensure a valid time range

            amp = self._calculate_average_amplitude(start, end)
            syllable_count = self._get_syllable_count(word)

            if self._check_space(word):
                words_list = word.split()
                time_avg = (end - start) / len(words_list)

                for sub_word in words_list:
                    sub_word = self._replace_words_with_numbers(sub_word)
                    sub_start = start
                    sub_end = start + time_avg
                    sub_amp = self._calculate_average_amplitude(sub_start, sub_end)
                    word_tuple = (sub_word, sub_start, sub_end, sub_amp)
                    words_in_sentence.append(word_tuple)
                    start += time_avg
            else:
                word_tuple = (word, start, end, amp)
                words_in_sentence.append(word_tuple)

            if (
                self._contains_punctuation(word_info["word"])
                or idx == len(segment["words"]) - 1
            ):
                all_words.append(words_in_sentence)
                words_in_sentence = []

        if words_in_sentence:
            all_words.append(words_in_sentence)

        return all_words

    def _calculate_average_amplitude(self, start_time, end_time):
        try:
            audio_data, sample_rate = sf.read(self.audio_file)
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_segment = audio_data[start_sample:end_sample]
            amplitude = np.abs(audio_segment)
            return np.mean(amplitude)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def _format_word(self, word):
        replacements = {
            "!": "",
            ".": "",
            ",": "",
            "?": "",
            "-": "",
            "0": "zero ",
            "1": "one ",
            "2": "two ",
            "3": "three ",
            "4": "four ",
            "5": "five ",
            "6": "six ",
            "7": "seven ",
            "8": "eight ",
            "9": "nine ",
        }
        for char, replacement in replacements.items():
            word = word.replace(char, replacement)
        return word.strip()

    def _check_space(self, word):
        return " " in word

    def _contains_punctuation(self, word):
        punctuation_marks = [".", "?", "!"]
        return any(char in punctuation_marks for char in word)

    def _replace_words_with_numbers(self, word):
        for char, replacement in word_to_number.items():
            word = word.replace(char, replacement)
        return word.strip()

    def _get_syllable_count(self, word):
        if word.lower() in pronouncing_dict:
            pronunciation = pronouncing_dict[word.lower()][0]
            syllable_count = sum(
                1 for phoneme in pronunciation if phoneme[-1].isdigit()
            )
            return syllable_count
        return None

    def split_text(self):
        result = []
        pattern = r"([.?!])"
        for text in self.text_sentences:
            split_texts = re.split(pattern, text)
            sentences = [
                "".join(pair)
                for pair in zip(split_texts[::2], split_texts[1::2])
                if pair[0].strip()
            ]
            result.extend(sentences)
        return result

    def _get_stress_value(self, sentence_amplitude, word_amplitude):
        if word_amplitude > (sentence_amplitude * 2):
            return 2  # High pitch
        elif word_amplitude >= sentence_amplitude:
            return 1  # Primary stress
        else:
            return 0  # No stress

    def format_texts(self, texts, stress_lists):
        formatted_texts = []
        for text, stress_list in zip(texts, stress_lists):
            formatted_text = self._apply_stress_formatting(text, stress_list)
            formatted_texts.append(formatted_text)
        return formatted_texts

    def _apply_stress_formatting(self, texts, stress_list):
        # print(stress_list)
        formatted_result = []

        # Index for keeping track of position in the text
        text_index = 0

        # Process each (word, stress) tuple in the stress list
        for word, stress in stress_list:
            # Find the start index of this word in the text
            start = texts.find(word, text_index)
            end = start + len(word)

            # Add the text leading up to the word (if there are punctuation/symbols, preserve them)
            formatted_result.append(texts[text_index:start])

            # Format the word based on the stress value
            if stress == 2:
                formatted_word = f"<b>{word.upper()}</b>"
            elif stress == 1:
                formatted_word = f"<u>{word}</u>"
            else:
                formatted_word = word

            # Add the formatted word to the result
            formatted_result.append(formatted_word)

            # Update text_index to the end of the current word
            text_index = end

        # Add any remaining text after the last word
        formatted_result.append(texts[text_index:])
        formatted_html = "".join(formatted_result)
        # Join all parts to create the final formatted string
        return formatted_html.strip()

    def combine_texts_with_timing(self, formatted_texts, sentence_data):
        formatted_sentences = []
        for formatted_text, sentence in zip(formatted_texts, sentence_data):
            start_time = sentence[0][1]
            end_time = sentence[-1][2]
            formatted_sentences.append((start_time, end_time, formatted_text))
        return formatted_sentences

    def write_vtt(self, vtt_filename, formatted_sentences):
        with open(vtt_filename, "w") as f:
            f.write("WEBVTT\n\n")
            for index, (start, end, text) in enumerate(formatted_sentences, start=1):
                f.write(f"{index}\n")
                f.write(f"{format_time(start)} --> {format_time(end)}\n")
                f.write(f"{text}\n\n")

    def generate_vtt(self, vtt_filename):
        self.collect_data()
        texts = self.split_text()
        stress_lists = self._calculate_stress()
        formatted_texts = self.format_texts(texts, stress_lists)
        formatted_sentences = self.combine_texts_with_timing(
            formatted_texts, self.sentences
        )
        self.write_vtt(vtt_filename, formatted_sentences)

    def _calculate_stress(self):
        stress_lists = []
        for sentence in self.sentences:
            sentence_amplitude = self._calculate_average_amplitude(
                sentence[0][1], sentence[-1][2]
            )
            
            total_amplitude = 0
            for word_info in sentence:
                word, start_time, end_time, amp = word_info
                total_amplitude += amp
                # print(f"{word}, {start_time}, {end_time}, {amp}")
            total_amplitude = total_amplitude / len(sentence)
            # print(total_amplitude)
            
            word_stress_list = []
            for word_info in sentence:
                word, start_time, end_time, amp = word_info
                syllable_count = self._get_syllable_count(word)

                if syllable_count and syllable_count > 1:
                    syllables = self._split_word_into_syllables(word, syllable_count)
                    # Calculate the average duration per syllable
                    time_avg = (end_time - start_time) / syllable_count

                    sub_start = start_time
                    for i, sub_word in enumerate(syllables):
                        # If it's the last subword, use the original end time
                        if i == len(syllables) - 1:
                            sub_end = end_time
                        else:
                            # Calculate sub_end based on the average duration per syllable
                            sub_end = sub_start + time_avg
                        sub_amp = self._calculate_average_amplitude(sub_start, sub_end)
                        stress_value = self._get_stress_value(
                            total_amplitude, sub_amp
                        )
                        # print(f"{sub_word} {stress_value}")
                        print(f"{sub_word}, {sub_start}, {sub_end}, {sub_amp}, {total_amplitude}, {stress_value}")
                        word_stress_list.append((sub_word, stress_value))

                        # Update sub_start to the next time interval
                        sub_start = sub_end

                else:
                    stress_value = self._get_stress_value(total_amplitude, amp)
                    word_stress_list.append((word, stress_value))
                    # print(f"{word} {stress_value}")
                    print(f"{word}, {start_time}, {end_time}, {amp}, {total_amplitude}, {stress_value}")
            print("")
            stress_lists.append(word_stress_list)
        return stress_lists

    def _split_word_into_syllables(self, word, num_syllables):
        vowels = "aeiouy"
        syllables = []
        current_syllable = ""
        for letter in word:
            if letter.lower() in vowels:
                if current_syllable:
                    syllables.append(current_syllable)
                    current_syllable = ""
            current_syllable += letter
        if current_syllable:
            syllables.append(current_syllable)

        if len(syllables) != num_syllables:
            syllables = []
            syllable_length = len(word) // num_syllables
            remaining_chars = len(word) % num_syllables
            start_index = 0
            for i in range(num_syllables):
                end_index = start_index + syllable_length
                if i < remaining_chars:
                    end_index += 1
                syllables.append(word[start_index:end_index])
                start_index = end_index
        return syllables


vtt_file_name = "stress_closed_caption.vtt"
audio_file = "vdo23.wav"
json_file = "vdo23.json"

recognizer = SentenceRecognizer(audio_file, json_file)
recognizer.generate_vtt(vtt_file_name)
