import cv2
import gdown
from typing import List
import numpy as np
import keras.src.layers
import tensorflow as tf
from matplotlib import pyplot as plt
import imageio
import os


def preprocess_frame(frame):
    """ Preprocess the frame to remove dense regions, normalize shades, and enhance visibility. """

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Crop frame (focus on region of interest)
    cropped = gray[190:236, 80:220]

    # Normalize pixel values to range [0,1] for ML models
    normalized = cropped.astype(np.float32) / 255.0

    return normalized


def load_video_mpg(path: str) -> np.ndarray:
    """ Load MPG video, process each frame, and return as a NumPy array. """

    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        frames.append(processed_frame)

    cap.release()

    # Convert to NumPy array with shape (num_frames, height, width, 1) for ML models
    frames_array = np.expand_dims(np.array(frames, dtype=np.float32), axis=-1)

    return frames_array

# vocabulary for text generation
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
vocab_georgian = [x for x in "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ'?!123456789 "]


char_to_num = keras.src.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = keras.src.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


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
    frames = load_video_mpg(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments


test_path = ".\\data\\s1\\bbal6n.mpg"

frames_array = load_video_mpg(test_path)
frames_uint8 = (frames_array.squeeze() * 255).astype(np.uint8)
#imageio.mimsave('./cv2_animation_new.gif', frames_uint8, fps=10)

# getting file name from data
tf.convert_to_tensor(test_path).numpy().decode('utf-8').split("\\")[-1].split('.')[0]

frames, alignments = load_data(tf.convert_to_tensor(test_path))

print([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])
a = tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])

def mappable_function(path: str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# CREATE DATA PIPELINE

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

test = data.as_numpy_iterator()
val = test.next()

frames_to_save = np.squeeze(val[0][0])
frames_to_save = (frames_to_save * 255).astype(np.uint8)

imageio.mimsave('./animation_0_new.gif', frames_to_save, fps=10)










