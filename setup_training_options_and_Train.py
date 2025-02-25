import tensorflow as tf
import numpy as np
import data_loading_functions
from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import Design_deep_neural_network
import os


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    # input length should be 75, label_length = 40 in our case
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75, 75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            # print('Original:', tf.strings.reduce_join([data_loading_functions.vocab[word] + '' for word in data[1][x]]).numpy().decode('utf-8'))
            # print('Prediction:', tf.strings.reduce_join([data_loading_functions.vocab[word] + '' for word in decoded[x]]).numpy().decode('utf-8'))
            # print('-' * 100)
            print('Original:', tf.strings.reduce_join(data_loading_functions.num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(data_loading_functions.num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~' * 100)


Design_deep_neural_network.model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint.weights.h5'), monitor='loss', save_weights_only=True)
#checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint'), monitor='loss', save_weights_only=True)
schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(data_loading_functions.test)

Design_deep_neural_network.model.fit(data_loading_functions.train, validation_data=data_loading_functions.test, epochs=50, callbacks=[checkpoint_callback, schedule_callback, example_callback])

