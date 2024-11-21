import argparse
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

# Set up argument parser
parser = argparse.ArgumentParser(description="Process an audio file for speaker diarization and transcription.")
parser.add_argument("input_audio", type=str, help="Path to the input audio file")
args = parser.parse_args()

# Load the pyannote pipeline for speaker diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Load and process the audio file
audio = AudioSegment.from_file(args.input_audio)
audio.export("processed_audio.wav", format="wav")

# Perform speaker diarization
diarization = pipeline("processed_audio.wav")

# Load the Whisper model for transcription
whisper_model = whisper.load_model("base")

# Transcribe the audio
transcription = whisper_model.transcribe("processed_audio.wav")

# Save the combined diarization and transcript results
with open("combined_results.txt", "w") as result_file:
    current_speaker = None
    transcript_index = 0
    transcript_text = transcription["text"]
    
    # Iterate over diarization results and map them to the transcript
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker != current_speaker:
            result_file.write(f"\n\nspeaker_{speaker}:\n")
            current_speaker = speaker
        
        # Extract the transcript segment corresponding to the current turn
        start_time = turn.start
        end_time = turn.end