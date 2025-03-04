import cv2
import gdown
from typing import List
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import imageio
import os
import dlib


# for downloading data

# url = "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL"
# output = 'data.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('data.zip')

def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    # capturing frames from location: 190:236 ...
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# vocabulary for text generation
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
vocab_georgian = [x for x in "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ'?!123456789 "]
#function for converting numbers into chars and chars to num
# oov_token = if no char then returns empty string

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

#print(f"the vocabulary is: {char_to_num.get_vocabulary()}"
#      f"(size ={char_to_num.vocabulary_size()}")

def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, '', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data', 's1', f"{file_name}.mpg")
    alignment_path = os.path.join('data', 'alignments', 's1', f"{file_name}.align")
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

test_path = ".\\data\\s1\\bbal6n.mpg"

# getting file name from data
#print(tf.convert_to_tensor(test_path).numpy().decode('utf-8').split("\\")[-1].split('.')[0])

frames, alignments = load_data(tf.convert_to_tensor(test_path))
#print([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])


a = tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])
def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

# CREATE DATA PIPELINE

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

#splitting data for training
train = data.take(450)
test = data.skip(450)


frames, alignments = data.as_numpy_iterator().next()

test1 = data.as_numpy_iterator()
val = test1.next()


#imageio.mimsave('./animation-00.gif', val[0][1], fps=10)
# print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]))
# print(tf.strings.reduce_join([num_to_char(word) for word in val[1][1]]))

# Paths to models
face_model_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\haarcascade_frontalface_default.xml'
shape_predictor_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\shape_predictor_68_face_landmarks.dat'

# Load face detector and shape predictor from dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)


def detect_mouth(frame):
    """Detect and return the mouth region using dlib landmarks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Dlib
    faces = face_detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = shape_predictor(gray, face)

        # Points for the mouth (usually points 48 to 67 in the 68 landmarks model)
        mouth_points = []
        for i in range(48, 68):
            mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))

        # Create a convex hull around the mouth region
        mouth_points = np.array(mouth_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Crop the mouth region from the image
        cropped_mouth = frame[y:y + h, x:x + w]

        return cropped_mouth  # Return the cropped mouth region

    return None  # Return None if no mouth is detected


def load_video_new(path: str, save_gif=False) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    gif_frames = []  # For saving processed frames as a GIF

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use mouth detection to get the cropped mouth region
        cropped_mouth = detect_mouth(frame)

        if cropped_mouth is not None:
            # Resize for consistency
            resized_mouth = cv2.resize(cropped_mouth, (140, 46))

            # Normalize the frame
            mean = np.mean(resized_mouth)
            std = np.std(resized_mouth) + 1e-6  # Avoid division by zero
            normalized_frame = (resized_mouth - mean) / std

            frames.append(normalized_frame)

            # Show the cropped and processed frame
            cv2.imshow("Mouth Detection", resized_mouth)

            # Store frame for GIF creation
            gif_frames.append(resized_mouth)

        else:
            # If no mouth is detected, skip this frame or handle as needed
            print("Mouth not detected in the current frame.")
            continue

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save GIF if requested
    if save_gif:
        imageio.mimsave("processed_mouth_new_111.gif", gif_frames, fps=10)

    # Convert list of frames to a NumPy array
    return np.array(frames, dtype=np.float32)


# Test with a sample video
video_path = ".\\data\\s1\\bbal6n.mpg"
video_path1 = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\test_eng_vid.mp4"
processed_frames = load_video_new(video_path1, save_gif=True)













