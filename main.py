import streamlit as st
import cv2
import numpy as np
import tempfile
import face_recognition
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
from PIL import Image
import io
import torch
import sounddevice as sd
import wave
import gc
import os

class VideoProcessor:
    def __init__(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def process_frame(self, frame):
        if self.process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

class AudioProcessor:
    def __init__(self):
        self.pipeline = None
        self.model = None

    def initialize_models(self, device):
        if self.pipeline is None:
            # Monkey patch numpy.NAN before importing Pipeline
            import numpy as np
            if not hasattr(np, 'NAN'):
                np.NAN = float('nan')
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization", 
                use_auth_token="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"
            )
            self.pipeline.to(device)
        
        if self.model is None:
            self.model = whisper.load_model("small", device=device)

    def cleanup(self):
        try:
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            if self.model is not None:
                del self.model
                self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def process_audio(self, audio_path, language):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.initialize_models(device)

            # Convert input audio to wav format
            audio = AudioSegment.from_file(audio_path)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(temp_wav.name, format="wav")

            # Perform diarization
            try:
                diarization = self.pipeline(temp_wav.name)
            except Exception as e:
                st.error(f"Diarization failed: {str(e)}")
                return "Error: Could not perform speaker diarization"

            # Process audio for transcription
            audio = AudioSegment.from_wav(temp_wav.name)
            audio = audio.set_frame_rate(16000)

            transcript = []
            try:
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start = int(turn.start * 1000)
                    end = int(turn.end * 1000)

                    # Extract segment
                    segment = audio[start:end]
                    temp_segment = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    segment.export(temp_segment.name, format="wav")
                    
                    # Transcribe segment
                    try:
                        result = self.model.transcribe(temp_segment.name, language=language)
                        if result and 'text' in result:
                            transcript.append(f"Speaker {speaker}: {result['text'].strip()}")
                    except Exception as e:
                        st.warning(f"Failed to transcribe segment: {str(e)}")
                        continue
                    finally:
                        # Clean up segment file
                        temp_segment.close()
                        if os.path.exists(temp_segment.name):
                            os.unlink(temp_segment.name)

            except Exception as e:
                st.error(f"Error processing audio segments: {str(e)}")
                return "Error: Failed to process audio segments"

            if not transcript:
                return "No speech detected or transcription failed"

            return "\n".join(transcript)
        
        except Exception as e:
            st.error(f"General transcription error: {str(e)}")
            return f"Error: {str(e)}"
        
        finally:
            # Clean up temporary wav file
            if 'temp_wav' in locals():
                temp_wav.close()
                if os.path.exists(temp_wav.name):
                    os.unlink(temp_wav.name)
            
            # Clean up models and GPU memory
            self.cleanup()

def record_audio(duration=5, sample_rate=16000):
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    st.write("Recording finished.")
    
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_audio_file.name, 'wb') as wavefile:
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(sample_rate)
        wavefile.writeframes(recording.tobytes())
    return temp_audio_file.name

def process_video(video_source):
    video_capture = None
    try:
        if video_source == "Upload Video":
            video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
            if video_file:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_file.write(video_file.read())
                temp_video_file.close()
                video_capture = cv2.VideoCapture(temp_video_file.name)
                os.unlink(temp_video_file.name)
        else:  # Use Webcam
            video_capture = cv2.VideoCapture(0)

        if video_capture is not None and video_capture.isOpened():
            try:
                known_face_encodings = np.load('encodings.npy')
                with open('string_array.txt', 'r') as file:
                    known_face_names = [line.strip() for line in file]

                video_processor = VideoProcessor(known_face_encodings, known_face_names)
                stframe = st.empty()

                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    processed_frame = video_processor.process_frame(frame)
                    img_array = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_array)
                    stframe.image(pil_image, caption="Processed Frame", use_container_width=True)

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

    finally:
        if video_capture is not None:
            video_capture.release()

def main():
    st.title("Video and Audio Processor")

    # Initialize session states
    if "audio_processor" not in st.session_state:
        st.session_state.audio_processor = None
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcription_requested" not in st.session_state:
        st.session_state.transcription_requested = False

    # Video Section
    st.header("Video Processing")
    video_source = st.radio("Select Video Source", ("Upload Video", "Use Webcam"))
    process_video(video_source)

    # Audio Section
    st.header("Audio Processing")
    audio_source = st.radio("Select Audio Source", ("Upload Audio", "Record Audio"))
    
    if audio_source == "Upload Audio":
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
        if audio_file:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(audio_file.read())
            st.session_state.audio_path = temp_audio_file.name
            temp_audio_file.close()

    elif audio_source == "Record Audio":
        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        if st.button("Start Recording"):
            st.session_state.audio_path = record_audio(duration)

    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path, format='audio/wav')

        language_option = st.selectbox("Select Language", ["English", "Hindi"])
        language_code = "hi" if language_option == "Hindi" else "en"

        if st.button("Start Transcription"):
            st.session_state.transcription_requested = True

        if st.session_state.transcription_requested:
            try:
                with st.spinner("Transcribing audio..."):
                    if st.session_state.audio_processor is None:
                        st.session_state.audio_processor = AudioProcessor()
                    
                    transcript = st.session_state.audio_processor.process_audio(
                        st.session_state.audio_path, 
                        language_code
                    )
                    st.text_area("Transcript", transcript, height=300)
                    
                    # Clean up the temporary audio file
                    if os.path.exists(st.session_state.audio_path):
                        os.unlink(st.session_state.audio_path)
                        st.session_state.audio_path = None
                    
                    st.session_state.transcription_requested = False
                    
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")
                st.session_state.transcription_requested = False

if __name__ == "__main__":
    main()
