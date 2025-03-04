import tensorflow as tf
import data_loading_functions
import Design_deep_neural_network
import cv2
import numpy as np
from typing import List
import os

# Verify if the file exists
mouth_cascade_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\haarcascade_mcs_mouth.xml'
frontal_face_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\haarcascade_frontalface_default.xml'

def process_video(video_frames, chunk_size=75):
    chunks = []
    for i in range(0, len(video_frames), chunk_size):
        chunk = video_frames[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunk = np.expand_dims(chunk, axis=0)  # Add batch dimension
            chunks.append(chunk)
    return chunks

def detect_mouth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(frontal_face_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            return frame[y + my:y + my + mh, x + mx:x + mx + mw]
    return frame  # fallback to the original frame if no mouth detected

def resize_frame(frame, target_size=(46, 140)):
    return cv2.resize(frame, target_size)


def load_video(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames
        frame = detect_mouth(frame)  # Detect mouth in the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame = resize_frame(frame, target_size=(46, 140))  # Resize to desired size
        frames.append(frame)

    cap.release()

    # Convert list of frames to a numpy array
    frames = np.array(frames)

    print(f"Total frames loaded: {len(frames)}")  # Debugging: Check total frame count

    # Normalize the frames
    mean = np.mean(frames)
    std = np.std(frames)
    frames = (frames - mean) / std

    # Add channel dimension (1 channel for grayscale)
    frames = frames.reshape(frames.shape[0], 46, 140, 1)

    return tf.cast(frames, tf.float32)

# Load the trained model
Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')

# Path to your video file
#video_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\m1m1.mp4'

video_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\data\\s1\\sriuzp.mpg'

# Load the video (getting all frames)
sample1 = load_video(video_path)
#sample1 = data_loading_functions.load_video(video_path)
print("sample1 shape:", sample1.shape)

# Process the video into chunks
chunks = process_video(sample1, chunk_size=75)

# Collect all predictions
predictions = []

for chunk in chunks:
    # Ensure the chunk is in the correct batch format and make prediction
    pred = Design_deep_neural_network.model.predict(chunk)
    predictions.append(pred)

# Concatenate all predictions
predictions = np.concatenate(predictions, axis=0)
print("predictions:", predictions)

# Decode predictions (pass correct input lengths)
input_lengths = [predictions.shape[1]] * predictions.shape[0]  # For each batch
decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_lengths, greedy=True)[0][0].numpy()

# Print the predicted text
print('~' * 100, 'PREDICTIONS')
print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])
