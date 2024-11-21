import sys
import os
import cv2
import numpy as np
import face_recognition
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
from queue import Queue

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

class VideoProcessor(QThread):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, known_face_encodings, known_face_names):
        super().__init__()
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.frame_queue = Queue(maxsize=5)
        self.running = True

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame = self.process_frame(frame)
                self.frame_processed.emit(processed_frame)

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

    def stop(self):
        self.running = False
        self.wait()

class AudioProcessor:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token="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")
        self.pipeline.to(device)
        self.model = whisper.load_model("small.en", device=device)

    def process_audio(self, audio_path):
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
            result = self.model.transcribe("segment.wav")
            
            transcript.append(f"Speaker {speaker}: {result['text']}")

        return "\n".join(transcript)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video and Audio Processor")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.video_label = QLabel()
        self.layout.addWidget(self.video_label)

        self.right_layout = QVBoxLayout()
        self.layout.addLayout(self.right_layout)

        self.upload_video_button = QPushButton("Upload Video")
        self.upload_video_button.clicked.connect(self.upload_video)
        self.right_layout.addWidget(self.upload_video_button)

        self.upload_audio_button = QPushButton("Upload Audio")
        self.upload_audio_button.clicked.connect(self.upload_audio)
        self.right_layout.addWidget(self.upload_audio_button)

        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.right_layout.addWidget(self.transcript_text)

        known_face_encodings = np.load('encodings.npy')
        with open('string_array.txt', 'r') as file:
            known_face_names = [line.strip() for line in file]

        self.video_processor = VideoProcessor(known_face_encodings, known_face_names)
        self.video_processor.frame_processed.connect(self.update_frame)
        self.video_processor.start()

        self.audio_processor = AudioProcessor()

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)

        self.video_capture = None
        self.video_path = None
        self.audio_path = None

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_path = file_name
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.timer.start(30)

    def upload_audio(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if file_name:
            self.audio_path = file_name
            self.process_audio()

    def process_audio(self):
        if self.audio_path:
            transcript = self.audio_processor.process_audio(self.audio_path)
            self.transcript_text.setText(transcript)

    def read_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                if not self.video_processor.frame_queue.full():
                    self.video_processor.frame_queue.put(frame)
            else:
                self.timer.stop()
                self.video_capture.release()
                self.video_capture = None

    def update_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.video_processor.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())