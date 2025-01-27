import os
import re
import json
import numpy as np
import nltk
from nltk.corpus import cmudict
from audio_analyzer import AudioAnalyzer

nltk.download("cmudict")
pronouncing_dict = cmudict.dict()

def format_time(seconds):
    """Helper function to format time for VTT files"""
    milliseconds = int(seconds * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

class SentenceRecognizer:
    def __init__(self, audio_file, json_file):
        self.audio_file = audio_file
        self.json_file = json_file
        self.text_sentences = []
        self.sentences = []

    def collect_data(self):
        """Collect data from JSON file while preserving original text"""
        with open(self.json_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        for segment in data["segments"]:
            # Store original text
            self.text_sentences.append(segment["text"])
            
            # Process words for stress detection only
            words_data = self._extract_words(segment)
            if words_data:
                self.sentences.extend(words_data)

    def _extract_words(self, segment):
        """Extract words while preserving original text and timing"""
        try:
            audio_analyzer = AudioAnalyzer(self.audio_file)
            words_in_sentence = []
            all_words = []

            for word_info in segment["words"]:
                # Keep original word text but clean for stress analysis
                original_word = word_info["word"]
                analysis_word = self._clean_for_analysis(original_word)
                
                start = word_info["start"]
                end = word_info["end"]
                
                # Ensure valid time range
                if start == end:
                    end += 0.01

                # Calculate amplitude for stress detection
                amp = audio_analyzer._calculate_average_amplitude(start, end)
                if amp is not None:
                    words_in_sentence.append({
                        'original': original_word,
                        'analysis': analysis_word,
                        'start': start,
                        'end': end,
                        'amplitude': amp
                    })

            if words_in_sentence:
                all_words.append(words_in_sentence)

            return all_words

        except Exception as e:
            print(f"Error in _extract_words: {str(e)}")
            return []

    def _clean_for_analysis(self, word):
        """Clean word for stress analysis only"""
        # Remove punctuation for analysis but keep word intact
        analysis_word = re.sub(r'[^\w\s]', '', word)
        return analysis_word.strip().lower()

    def _calculate_stress(self):
        """Calculate stress values based purely on audio analysis"""
        try:
            audio_analyzer = AudioAnalyzer(self.audio_file)
            stress_lists = []

            for sentence in self.sentences:
                if not sentence:
                    continue

                # คำนวณค่าเฉลี่ยของประโยค
                sent_start = sentence[0]['start']
                sent_end = sentence[-1]['end']
                sent_amp = audio_analyzer._calculate_average_amplitude(sent_start, sent_end)
                sent_pitch = audio_analyzer._get_pitch_avg(sent_start, sent_end)

                # เก็บค่า amplitude ทั้งหมดเพื่อหาค่าสูงสุด
                all_amps = [word_data['amplitude'] for word_data in sentence]
                max_amp = max(all_amps) if all_amps else 0

                word_stress_list = []
                for word_data in sentence:
                    # ใช้ข้อความดั้งเดิมที่มีช่องว่าง
                    original_word = word_data['original']
                    
                    # คำนวณค่าต่างๆ สำหรับการวิเคราะห์
                    word_amp = word_data['amplitude']
                    word_pitch = audio_analyzer._get_pitch_avg(
                        word_data['start'], 
                        word_data['end']
                    )

                    # ตรวจสอบความดังและระดับเสียง
                    stress = 0
                    if word_amp > (sent_amp * 1.5) and word_amp >= (max_amp * 0.8):
                        stress = 1
                    elif word_pitch and sent_pitch and word_pitch > (sent_pitch * 1.4):
                        stress = 1

                    word_stress_list.append({
                        'word': original_word.strip(),  # เก็บคำที่ไม่มีช่องว่างหน้า-หลัง
                        'stress': stress
                    })

                if word_stress_list:
                    stress_lists.append(word_stress_list)

            return stress_lists

        except Exception as e:
            print(f"Error in _calculate_stress: {str(e)}")
            return []

    def _apply_stress_formatting(self, text, stress_list):
        """Apply stress formatting while preserving whitespace"""
        try:
            if not text or not stress_list:
                return text

            result = []
            current_pos = 0
            
            for word_data in stress_list:
                word = word_data['word']
                
                if not word.strip():
                    continue
                    
                # หาตำแหน่งของคำในข้อความ
                word_pos = text.find(word, current_pos)
                if word_pos == -1:
                    continue
                
                # เพิ่มข้อความก่อนคำปัจจุบัน (รวมช่องว่าง)
                result.append(text[current_pos:word_pos])
                
                # จัดการการขีดเส้นใต้โดยแยกพยางค์
                if word_data['stress'] == 1:
                    # แยกคำตามช่องว่าง
                    parts = word.split()
                    formatted_parts = []
                    for part in parts:
                        # ตรวจสอบว่าควรขีดเส้นใต้หรือไม่
                        if len(part) > 1:  # ขีดเส้นใต้เฉพาะคำที่มีความยาวมากกว่า 1 ตัวอักษร
                            formatted_parts.append(f"<u>{part}</u>")
                        else:
                            formatted_parts.append(part)
                    # รวมคำกลับด้วยช่องว่าง
                    result.append(" ".join(formatted_parts))
                else:
                    result.append(word)
                
                current_pos = word_pos + len(word)
            
            # เพิ่มข้อความที่เหลือ
            result.append(text[current_pos:])
            
            return "".join(result)

        except Exception as e:
            print(f"Error in _apply_stress_formatting: {str(e)}")
            return text

    def generate_vtt(self, vtt_filename):
        """Generate VTT file with stress formatting"""
        try:
            self.collect_data()
            stress_lists = self._calculate_stress()

            # Combine original segments with stress formatting
            formatted_segments = []
            for segment, stress_list in zip(self.text_sentences, stress_lists):
                formatted_text = self._apply_stress_formatting(
                    segment, 
                    stress_list
                )
                if formatted_text:
                    start_time = self.sentences[len(formatted_segments)][0]['start']
                    end_time = self.sentences[len(formatted_segments)][-1]['end']
                    formatted_segments.append((start_time, end_time, formatted_text))

            # Write to VTT file
            with open(vtt_filename, "w", encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for i, (start, end, text) in enumerate(formatted_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{format_time(start)} --> {format_time(end)}\n")
                    f.write(f"{text}\n\n")

        except Exception as e:
            print(f"Error in generate_vtt: {str(e)}")

if __name__ == "__main__":
    audio_file = "input_audio.wav"
    json_file = "whisper_output.json"
    vtt_file = "output.vtt"
    
    recognizer = SentenceRecognizer(audio_file, json_file)
    recognizer.generate_vtt(vtt_file)