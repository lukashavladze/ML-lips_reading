import gdown
import data_loading_functions
import Design_deep_neural_network
import setup_training_options_and_Train
import tensorflow as tf
from new_test_runs import load_video_new

# url = 'https://drive.google.com/uc?id=1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y'
# output = 'checkpoints.zip'
# gdown.download(url, output, quiet=False)
# gdown.extractall('checkpoints.zip', 'model')

#Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\model\\checkpoint')

#Design_deep_neural_network.model.save('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')

Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modelh5\\model.h5')
#Design_deep_neural_network.model.load_weights('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\modellipnet\\overlapped-weights368.h5')

# test_data = data_loading_functions.test.as_numpy_iterator()
# sample = test_data.next()
# yhat = Design_deep_neural_network.model.predict(sample[0])



# print('~'*100, 'REAL TEXT')
# print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in sample[1]])
#
# decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75,75], greedy=True)[0][0].numpy()

# print('~'*100, 'PREDICTIONS')
# print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])

# test on video
#my_video_path = "C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\test_eng_vid.mp4"
#my_video_path = 'C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\data\\s1\\sriuzp.mpg'

#load_data_new_Result = data_loading_functions.load_data_new(my_video_path)
#load_data_new_Result = data_loading_functions.load_data(my_video_path)

#sample_Result = data_loading_functions.load_data_new(tf.convert_to_tensor(my_video_path))

#sample = data_loading_functions.load_data_new(tf.convert_to_tensor("C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\m1m1m1.mp4"))
#sample = data_loading_functions.load_data_new(tf.convert_to_tensor('C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\data\\s1\\sriuzp.mpg'))
sample = data_loading_functions.load_data_new(tf.convert_to_tensor("C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\trimmed_video.mp4"))

# sample_test = load_video_new("C:\\Users\\LSHAVLADZE\\PycharmProjects\\ML-lips_reading\\test_eng_vid.mp4")
# print("-"*100)
# print("sample", sample_test)
# print("-"*100)
# print("sample[0]", sample_test[0])
# print("-"*100)
# print("sample[1]", sample_test[1])
# print('~'*100, 'REAL TEXT')

#print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in [sample[1]]])
#yhat = Design_deep_neural_network.model.predict(tf.expand_dims(sample[0], axis=0))
yhat = Design_deep_neural_network.model.predict(tf.expand_dims(sample, axis=0))
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()

print('~'*100, 'PREDICTIONS of my video')
print([tf.strings.reduce_join([data_loading_functions.num_to_char(word) for word in sentence]) for sentence in decoded])



