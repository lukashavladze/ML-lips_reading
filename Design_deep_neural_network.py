from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import data_loading_functions
import tensorflow as tf

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

model.add(Dense(data_loading_functions.char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))


yhat = model.predict(data_loading_functions.val[0])

prediction_test = tf.strings.reduce_join([data_loading_functions.num_to_char(tf.argmax(x)) for x in yhat[1]])

# print(model.input_shape)
# print(model.output_shape)
# returning what our model predicted
#print(tf.argmax(yhat[0], axis=1))

#print(tf.strings.reduce_join([data_loading_functions.num_to_char(tf.argmax(x)) for x in yhat[0]]))