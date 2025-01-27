import os

"""
Using command to terminal for sentence recognition
Input : Audio file name (string)
Output : transcription files (json, srt, tsv, txt, vtt)
"""

class SentenceRecognition:
    def recognize(self, audio_file):
        # Construct the full file path
        class_directory = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(class_directory, audio_file)
        # file_path = os.path.join("static", "files", audio_file)

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"Error: File '{audio_file}' not found.")
            return

        # Run the recognition command
        command = f'whisper "{file_path}" --model large-v2 --word_timestamps True --highlight_words True'
        output = os.popen(command).read()
        print(output)

"""
recognize = SentenceRecognition()    
recognize.recognize("Unit 6 I need to buy new clothes.mp3")

"""