import tensorflow as tf
import data_loading_functions
import Design_deep_neural_network
import cv2
import numpy as np
from typing import List
import os
import dlib  # For mouth detection using dlib

# Path to the dlib shape predictor
predictor_path = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def process_video(video_frames, chunk_size=75):
    chunks = []
    for i in range(0, len(video_frames), chunk_size):
        chunk = video_frames[i:i + chunk_size]
        if len(chunk) == chunk_size:
            chunk = np.expand_dims(chunk, axis=0)  # Add batch dimension
            chunks.append(chunk)
    return chunks


def detect_mouth(frame):
    # Convert frame to grayscale for dlib detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    for face in faces:
        # Get the landmarks for the detected face
        landmarks = predictor(gray, face)

        # Get the mouth coordinates (mouth is located from 48 to 67 in 68-landmark model)
        mouth_points = []
        for i in range(48, 68):
            point = landmarks.part(i)
            mouth_points.append((point.x, point.y))

        # Create a mask for the mouth region using the landmarks
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [np.array(mouth_points)], 255)

        # Extract the mouth region from the image
        mouth_region = cv2.bitwise_and(frame, frame, mask=mask)

        # Get the bounding box for the mouth region
        x, y, w, h = cv2.boundingRect(np.array(mouth_points))
        cropped_mouth = mouth_region[y:y + h, x:x + w]

        return cropped_mouth
    return frame  # fallback to the original frame if no mouth detected


def resize_frame(frame, target_size=(46, 140)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return cv2.resize(gray_frame, target_size)


def load_video_new(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use mouth detection to get the cropped mouth region
        cropped_mouth = detect_mouth(frame)

        if cropped_mouth is not None:
            # Resize for consistency
            resized_mouth = cv2.resize(cropped_mouth, (140, 46))
            resized_mouth = cv2.cvtColor(resized_mouth, cv2.COLOR_BGR2GRAY)
            resized_mouth = np.expand_dims(resized_mouth, axis=1)
            # Normalize the frame
            mean = np.mean(resized_mouth)
            std = np.std(resized_mouth) + 1e-6  # Avoid division by zero
            normalized_frame = (resized_mouth - mean) / std

            frames.append(normalized_frame)

            # Show the cropped and processed frame
            #cv2.imshow("Mouth Detection", resized_mouth)

        else:
            # If no mouth is detected, skip this frame or handle as needed
            print("Mouth not detected in the current frame.")
            continue

    cap.release()
    cv2.destroyAllWindows()

    # Convert list of frames to a NumPy array
    return np.array(frames, dtype=np.float32)


# Load the trained model
Design_deep_neural_network.model.load_weights(
    'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')

# Path to your video file
video_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\data\\s1\\sriuzp.mpg'
video_path1 = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\test_eng_vid.mp4"

# Load the video (getting all frames) using the new function
sample1 = load_video_new(video_path1)
print("original sample1:", sample1.shape)

sample1 = np.expand_dims(sample1, axis=-1)
print("sample1 shape after expand dims:", sample1.shape)

sample1 = np.transpose(sample1, (2, 0, 1, 3, 4))  # Swap axes to match (frames, height, width, channels)

print("sample1 shape transposed:", sample1.shape)




# Process the video into chunks
chunks = process_video(sample1, chunk_size=75)

# Collect all predictions
predictions = []

for chunk in chunks:
    print(f"Processing chunk with shape: {chunk.shape}")
    pred = Design_deep_neural_network.model.predict(chunk)
    predictions.append(pred)

print("predictions:", predictions)

# Concatenate all predictions
# predictions = np.concatenate(predictions, axis=0)
# print("predictions:", predictions)

# Decode predictions (pass correct input lengths)
#input_lengths = [predictions.shape[1]] * predictions.shape[0]  # For each batch
# decoded = tf.keras.backend.ctc_decode(predictions, input_length=input_lengths, greedy=True)[0][0].numpy()
#
# # Print the predicted text
# print('~' * 100, 'PREDICTIONS')
# print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])
