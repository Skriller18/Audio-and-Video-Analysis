import librosa
import soundfile as sf
from pydub import AudioSegment

# Load an audio file
audio_file_path = 'G:\\Projects\\ANN-Project\\original.mp3'
audio_data, rate = librosa.load(audio_file_path, sr=None)

# Reduce noise
import noisereduce as nr
reduced_noise = nr.reduce_noise(y=audio_data, sr=rate)

# Save the processed audio to a new WAV file
output_file_path_wav = 'G:\\Projects\\ANN-Project\\processed.wav'
sf.write(output_file_path_wav, reduced_noise, rate)

# Convert WAV to MP3
output_file_path_mp3 = 'G:\\Projects\\ANN-Project\\processed.mp3'
sound = AudioSegment.from_wav(output_file_path_wav)
sound.export(output_file_path_mp3, format="mp3")

'''import pyaudio
import numpy as np
import noisereduce as nr

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio stream parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Function to process the audio stream
def process_stream(data, frame_count, time_info, status):
    audio_data = np.frombuffer(data, dtype=np.float32)
    # Reduce noise
    reduced_noise = nr.reduce_noise(audio_data, RATE)
    return (reduced_noise.tobytes(), pyaudio.paContinue)

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=process_stream)

# Start the stream
stream.start_stream()

# Keep the stream active
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate PyAudio
    p.terminate()'''