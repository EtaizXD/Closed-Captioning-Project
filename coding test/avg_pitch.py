from pydub import AudioSegment
import parselmouth
import numpy as np
import os

def get_pitch_segment(audio_file, start_time, end_time):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Convert audio to mono (if stereo)
    audio = audio.set_channels(1)

    # Extract the segment from start_time to end_time
    start_ms = int(start_time * 1000)  # Convert to milliseconds
    end_ms = int(end_time * 1000)

    audio_segment = audio[start_ms:end_ms]

    # Export audio segment to WAV (Praat requires WAV format)
    wav_file = "temp_segment.wav"
    audio_segment.export(wav_file, format="wav")

    # Load the WAV file with Parselmouth
    snd = parselmouth.Sound(wav_file)

    # Extract pitch
    pitch = snd.to_pitch()

    # Get pitch values (excluding NaN values)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[~np.isnan(pitch_values)]  # Remove NaN values

    # Optionally remove the temporary WAV file
    os.remove(wav_file)

    return pitch_values



# Audio file path
audio_file_path = "vdo23.wav"
start_time = 32.04
end_time = 32.28 

pitch_values = get_pitch_segment(audio_file_path, start_time, end_time)
positive_pitch_values = pitch_values[pitch_values > 0]

average_positive_pitch = np.mean(positive_pitch_values) if len(positive_pitch_values) > 0 else np.nan
average_pitch = np.mean(pitch_values)
min_pitch = np.min(pitch_values)
max_pitch = np.max(pitch_values)
top_10_percent = np.percentile(pitch_values, 90)
top_10_values = pitch_values[pitch_values >= top_10_percent]
avg_top_10_pitch = np.mean(top_10_values) if len(top_10_values) > 0 else np.nan

if average_positive_pitch is None or average_positive_pitch == 0 or np.isnan(average_positive_pitch):
    average_positive_pitch = 300
if avg_top_10_pitch is None or avg_top_10_pitch == 0 or np.isnan(avg_top_10_pitch):
    avg_top_10_pitch = 300

# Print all the data
print(f"Average pitch for values > 0: {average_positive_pitch:.2f} Hz")
print(f"Average pitch: {average_pitch:.2f} Hz")
print(f"Minimum pitch: {min_pitch:.2f} Hz")
print(f"Maximum pitch: {max_pitch:.2f} Hz")
print(f"Average of top 10% pitch values: {avg_top_10_pitch:.2f} Hz")
