import cv2
import gdown
from typing import List
import numpy as np
from keras import layers
import tensorflow as tf
from matplotlib import pyplot as plt
import imageio
import os

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
data = data.shuffle(500)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

frames, alignments = data.as_numpy_iterator().next()

test = data.as_numpy_iterator()
val = test.next()


imageio.mimsave('./animation-00.gif', val[0][1], fps=10)
print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]))
print(tf.strings.reduce_join([num_to_char(word) for word in val[1][1]]))













