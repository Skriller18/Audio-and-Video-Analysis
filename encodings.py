import face_recognition
import os
import numpy as np

# Directory containing the photos
photos_directory = "cast"  # Update this path as necessary

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

photo_filenames = [f for f in os.listdir(photos_directory) if f.endswith(('.JPG', '.jpeg', '.png', '.jpg'))]

# Loop to load each photo, get its encoding, and append to the lists
for filename in photo_filenames:
    photo_path = os.path.join(photos_directory, filename)
    if os.path.exists(photo_path):
        image = face_recognition.load_image_file(photo_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            name, _ = os.path.splitext(filename)
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)  # Assign a unique name or modify as needed

# Save encodings and names
np.save('encodings.npy', known_face_encodings)

with open('string_array.txt', 'w') as file:
    for item in known_face_names:
        file.write(f"{item}\n")

print("Encoding generation complete.")
