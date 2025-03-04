import gdown
import data_loading_functions
import Design_deep_neural_network
import setup_training_options_and_Train
import tensorflow as tf

# url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
# output = 'checkpoints.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('checkpoints.zip', 'model')

#Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\model\\checkpoint')

#Design_deep_neural_network.model.save('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')

Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')

test_data = data_loading_functions.test.as_numpy_iterator()
sample = test_data.next()
yhat = Design_deep_neural_network.model.predict(sample[0])



print('~'*100, 'REAL TEXT')
print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in sample[1]])

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75,75], greedy=True)[0][0].numpy()

print('~'*100, 'PREDICTIONS')
print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])

# test on video

sample = data_loading_functions.load_data(tf.convert_to_tensor('.\\data\\s1\\bras9a.mpg'))
print('~'*100, 'REAL TEXT')

print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in [sample[1]]])
yhat = Design_deep_neural_network.model.predict(tf.expand_dims(sample[0], axis=0))

decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print('~'*100, 'PREDICTIONS')
print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])



