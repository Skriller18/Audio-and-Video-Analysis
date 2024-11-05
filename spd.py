import argparse
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import numpy as np
import gc
import re

# Set up argument parser
parser = argparse.ArgumentParser(description="Process an audio file for speaker diarization and transcription.")
parser.add_argument("input_audio", type=str, help="Path to the input audio file")
args = parser.parse_args()

# Load the pyannote pipeline for speaker diarization
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization", use_auth_token="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")

def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768

def millisec(timeStr):
    # Handle different time formats
    if '.' in timeStr:
        # Format: seconds.milliseconds
        return int(float(timeStr) * 1000)
    elif ':' in timeStr:
        # Format: HH:MM:SS.milliseconds
        hours, minutes, seconds = timeStr.split(':')
        return int(hours) * 3600000 + int(minutes) * 60000 + int(float(seconds) * 1000)
    else:
        # Assume it's already in milliseconds
        return int(timeStr)

# Process the input audio file
audio = AudioSegment.from_file(args.input_audio)
audio.export("processed_audio.wav", format="wav")

# Perform speaker diarization
diarization = pipeline("processed_audio.wav")

del pipeline
gc.collect()

audio = AudioSegment.from_wav("processed_audio.wav")
audio = audio.set_frame_rate(16000)

model = whisper.load_model("small.en")

# Open a text file to write the transcription results
with open("transcription_results.txt", "w") as file:
    # Process each segment
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * 1000)  # Convert to milliseconds
        end = int(turn.end * 1000)  # Convert to milliseconds

        # Extract the audio segment and transcribe it
        segment = audio[start:end]
        segment.export("segment.wav", format="wav")
        result = model.transcribe("segment.wav")
        
        # Write the result to the file
        file.write(f"Speaker {speaker}: {result['text']}\n")
        print(f"Speaker {speaker}: {result['text']}")

# Clean up
del model
gc.collect()