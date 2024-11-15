import librosa
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from pydub import AudioSegment


# Function to extract pitch using librosa
def extract_pitches(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    
    high_pitches = []
    times = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            high_pitches.append(pitch)
            times.append(t * (1.0 / sr))

    return high_pitches, times


# Function to recognize speech using SpeechRecognition
def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    
    with audio_file as source:
        audio_data = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"


# Function to split audio into words
def split_audio_to_words(audio_path):
    recognizer = sr.Recognizer()
    words = []

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            words = recognizer.recognize_google(audio_data, show_all=True)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

    word_timings = []
    if words and 'alternative' in words:
        for alternative in words['alternative']:
            if 'timestamps' in alternative:
                word_timings.extend(alternative['timestamps'])

    return word_timings


# Load audio file and extract pitches
audio_path = 'vdo23.wav'
high_pitches, times = extract_pitches(audio_path)


# Recognize speech and get word timings
word_timings = split_audio_to_words(audio_path)


# Identify words with high pitch
high_pitch_words = []
all_words_with_pitch = []

for word in word_timings:
    word_start = word[1]
    word_end = word[2]
    word_text = word[0]

    # Check if the pitch during the word is high
    word_pitch = []
    for i, time in enumerate(times):
        if word_start <= time <= word_end:
            word_pitch.append(high_pitches[i])
            if high_pitches[i] > 200:  # Threshold for high pitch, e.g., 200 Hz
                high_pitch_words.append(word_text)
                break
    all_words_with_pitch.append((word_text, word_pitch))


# Print all words and their pitches
for word, pitch in all_words_with_pitch:
    print(f"Word: {word}, Pitch: {pitch}")

# Print high pitch words
print("Words with high pitch:", high_pitch_words)


# Plot pitch over time
plt.figure(figsize=(14, 5))
plt.plot(times, high_pitches, label='Pitch')
plt.axhline(y=200, color='r', linestyle='--', label='High pitch threshold')  # Example threshold
plt.title('Pitch over Time')
plt.xlabel('Time (s)')
plt.ylabel('Pitch (Hz)')
plt.legend()
plt.show()
