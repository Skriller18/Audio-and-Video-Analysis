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

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

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
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token="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")
        self.pipeline.to(device)
        self.model = whisper.load_model("small", device=device)

    def process_audio(self, audio_path, language):
        audio = AudioSegment.from_file(audio_path)
        audio.export("processed_audio.wav", format="wav")

        diarization = self.pipeline("processed_audio.wav")

        audio = AudioSegment.from_wav("processed_audio.wav")
        audio = audio.set_frame_rate(16000)

        transcript = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = int(turn.start * 1000)
            end = int(turn.end * 1000)

            segment = audio[start:end]
            segment.export("segment.wav", format="wav")
            result = self.model.transcribe("segment.wav", language=language)
            
            transcript.append(f"Speaker {speaker}: {result['text']}")

        return "\n".join(transcript)

def main():
    st.title("Video and Audio Processor")

    # Upload video file
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if video_file:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_path = temp_video_file.name

        # Use OpenCV to process the video
        video_capture = cv2.VideoCapture(temp_video_path)
        known_face_encodings = np.load('encodings.npy')
        with open('string_array.txt', 'r') as file:
            known_face_names = [line.strip() for line in file]

        video_processor = VideoProcessor(known_face_encodings, known_face_names)

        # Video processing
        stframe = st.empty()
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            processed_frame = video_processor.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            image = Image.open(io.BytesIO(buffer))
            stframe.image(image, caption="Processed Frame", use_column_width=True)

        video_capture.release()

    # Upload audio file
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    if audio_file:
        st.audio(audio_file, format='audio/wav')

        language_option = st.selectbox("Select Language", ["English", "Hindi"])
        language_code = "hi" if language_option == "Hindi" else "en"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(audio_file.read())
            audio_path = temp_audio_file.name
        
        audio_processor = AudioProcessor()
        transcript = audio_processor.process_audio(audio_path, language_code)
        st.text_area("Transcript", transcript, height=300)

if __name__ == "__main__":
    main()
