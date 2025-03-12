import cv2
import gdown
from typing import List
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import imageio
import os
import dlib
from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

my_video_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\data\\s1\\sriuzp.mpg'
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


def load_video_test(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    # capturing frames from location: 190:236 ...
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        #frame = tf.image.rgb_to_grayscale(frame)
        #frame = frame.numpy() # my added
        # cv2.imshow("frames window:", frame)
        # cv2.waitKey(1)
        cropped_frame = frame[500:600, 250:450, :]
        resized_frame = cv2.resize(cropped_frame, (140, 46))
        resized_frame = tf.image.rgb_to_grayscale(resized_frame)
        frames.append(resized_frame)
        # cv2.imshow("Resized Frame", resized_frame)
        # cv2.waitKey(1)
        #frames.append(frame[500:600, 250:450, :])
        # cv2.imshow("frames window:", frames[_])
        # cv2.waitKey(2)

    cap.release()
    cv2.destroyAllWindows()

    #frames_np = np.array(frames, dtype=np.float32)

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


predictor_path = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\shape_predictor_68_face_landmarks.dat"

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# vocabulary for text generation
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
vocab_georgian = [x for x in "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ'?!123456789 "]
#function for converting numbers into chars and chars to num
# oov_token = if no char then returns empty string

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


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

def load_data_new(path: str):
    path = bytes.decode(path.numpy())
    #frames = load_video_new(path)
    #frames = load_video(path)
    frames = load_video_test(path)
    #frames = load_video_test_1(path)
    return frames


model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))


#load_data_new_Result = load_data_new(tf.convert_to_tensor(my_video_path))
# frames_75 = load_data_new_Result[:75]
# frames_75 = tf.expand_dims(frames_75, axis=0)
# yhat1 = model.predict(frames_75)
#yhat1 = model.predict(tf.expand_dims(load_data_new_Result, axis=0))
#yhat = model.predict(tf.expand_dims(load_data_new_Result, axis=0))

#decoded = tf.keras.backend.ctc_decode(yhat1, input_length=[75], greedy=True)[0][0].numpy()

# print('~'*100)
# print("our text")
# print([tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded])


#test_path = ".\\data\\s1\\bbal6n.mpg"

# getting file name from data
#print(tf.convert_to_tensor(test_path).numpy().decode('utf-8').split("\\")[-1].split('.')[0])

#frames, alignments = load_data(tf.convert_to_tensor(test_path))
#print([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])


#a = tf.strings.reduce_join([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])
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


imageio.mimsave('./animation-00.gif', val[0][1], fps=10)
print(tf.strings.reduce_join([num_to_char(word) for word in val[1][0]]))
print(tf.strings.reduce_join([num_to_char(word) for word in val[1][1]]))














