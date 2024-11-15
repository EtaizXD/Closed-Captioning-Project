import librosa
import numpy as np

def _get_pitch_segment(audio_path, start_time, end_time):

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate the sample indices for the start and end times
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract the segment within the specified time range
    y_segment = y[start_sample:end_sample]
    
    # Compute the pitch (fundamental frequency)
    pitches, magnitudes = librosa.core.piptrack(y=y_segment, sr=sr)
    
    # Select the pitches for each frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Only consider positive pitch values
            pitch_values.append(pitch)
    
    # Calculate the average pitch
    avg_pitch = np.mean(pitch_values) if pitch_values else 0
    return avg_pitch

# Example usage
audio_path = 'vdo23.wav'  # Replace with the path to your audio file
start_time = 8.0  # seconds
end_time = 8.46  # seconds
average_pitch = _get_pitch_segment(audio_path, start_time, end_time)
print(f'The average pitch between {start_time} and {end_time} seconds is {average_pitch} Hz')
