import keras
import numpy as np

from keras import backend as K

from keras.models import Sequential, model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Reshape, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import SeparableConv2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger

import os, random
import h5py

np.set_printoptions(threshold=np.nan)

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)))

def binary_accuracy_np(y_true, y_pred):
	compare = np.equal(y_true.astype(int), np.round(y_pred).astype(int)).flatten().astype(int)
	correct = np.mean(np.min(compare))
	simular = np.mean(compare)

	print("correct {}".format(correct))
	print("simular {}".format(simular))

	return np.mean(np.stack([simular, simular]))

def binary_accuracy_validate(y_true, y_pred):
	correct = K.mean(K.min(K.cast(K.equal(y_true, K.round(y_pred)), 'float32')))
	simular = K.mean(K.equal(y_true, K.round(y_pred)))

	return K.mean(K.stack([correct, simular, simular]))

def binary_accuracy_perfect(y_true, y_pred):
	return K.mean(K.min(K.cast(K.equal(y_true, K.round(y_pred)), 'float32')))

class Tak_Train(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		self.tak_size = 5
		self.tak_height = 64
		self.hidden_units = 100
		self.train_batch_size = 100
		self.validate_batch_size = 30
		self.epochs = 90
		self.dropout_rate = 0.1

		self.opt = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

	def define_LSTM_model(self, validate=False):
		print("Setup Model")
		self.model = Sequential()
		#self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		#self.model.add(BatchNormalization())
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, activation='relu', return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='softmax'))#activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

		self.weights_save = "7-LSTM"
		#self.model.load_weights(os.path.join(os.getcwd(), self.weights_save, "White-weights-improvement-0.961.hdf5"))
		self.load_weights()

		if not validate:
			self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=[binary_accuracy_validate, binary_accuracy_perfect])
		else:
			self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=[correct])
			

		self.model.summary()

	def load_weights(self):
		if not os.path.exists(self.weights_save):
			os.makedirs(self.weights_save)

		#Save Config
		with open(os.path.join(os.getcwd(), self.weights_save, "model.json"), "w") as f:
			f.write(self.model.to_json())

		#Setup Model
		training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), self.weights_save)) if filename.endswith(".hdf5")]
		if len(training_files) != 0:
			training_files = sorted(training_files)
			print("Loading previous weights file " + training_files[-1])
			self.model.load_weights(os.path.join(os.getcwd(), self.weights_save, training_files[-1]))

	def define_Comb_model(self):
		print("Setup Model")
		self.model = Sequential()
		self.model.add(SeparableConv2D(20, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())
		
		self.model.add(Dense(self.tak_height, activation='relu'))
		#self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

		self.weights_save = "7-COMB"
		self.load_weights()
		self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=[binary_accuracy_validate, binary_accuracy_perfect])
		self.model.summary()

	def training_files_generator(self, file_names):
		print("Start Training Generator")

		all_x_train = None
		all_y_train = None

		left_overs = False
		start_index = 0
		end_index = 0
		left_over_size = 0

		for file_name in file_names:
			#print("Getting Training data from {}".format(file_name))

			with h5py.File(os.path.join(os.getcwd(), "ptn", file_name), 'r') as hf:
				x_train = hf["x_train"][:]
				y_train = hf["y_train"][:]

				#Shuffle data randomly but equally
				seed = np.random.randint(2**31)

				x_random = np.random.RandomState(seed)
				y_random = np.random.RandomState(seed)

				x_random.shuffle(x_train)
				y_random.shuffle(y_train)

				#With New file Reset sizes
				array_size = x_train.shape[0]
				start_index = 0
				end_index = 0

				while (end_index + self.train_batch_size) < array_size:
					#Update indexes
					start_index = end_index
					end_index = start_index + self.train_batch_size - left_over_size
					left_over_size = 0

					#print("Start_index: {}, End_index: {}, Array_size: {}".format(start_index, start_index + self.train_batch_size, array_size))

					#Set Return Values
					if left_overs == True:
						all_x_train = np.concatenate((all_x_train, x_train[start_index:end_index]), axis=0)
						all_y_train = np.concatenate((all_y_train, y_train[start_index:end_index]), axis=0)
						left_overs = False

					else:
						all_x_train = x_train[start_index:end_index]
						all_y_train = y_train[start_index:end_index]

					#print("Returning (x_shape: {}, y_shape:{})".format(all_x_train.shape, all_y_train.shape))

					yield (all_x_train, all_y_train)

				#Check for leftover data and add to all_trains
				if array_size > end_index:
					all_x_train = x_train[end_index:array_size]
					all_y_train = y_train[end_index:array_size]
					left_over_size = array_size - end_index
					left_overs = True

	def predict(self, x):
		x_pred = np.array([x])
		#print(x_pred.shape)
		ret = self.model.predict(x_pred)
		#print(ret.shape)
		return np.around(ret[0])

	def train_generator(self, training_generator, validation_generator):
		#Make generator to return data from training file
		callback1 = ModelCheckpoint(os.path.join(os.getcwd(), self.weights_save, "White-weights-improvement-{binary_accuracy_validate:.3f}.hdf5"), monitor='binary_accuracy_validate', verbose=2, save_best_only=True, mode='max')
		#callback2 = keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), self.weights_save), histogram_freq=0, write_graph=True, write_images=True)

		history = self.model.fit_generator(training_generator, self.train_batch_size, epochs=self.epochs, callbacks=[callback1], validation_data=validation_generator, validation_steps=self.validate_batch_size, verbose=1)
		#pyplot.plot(history.history['loss'])
		#pyplot.plot(history.history['acc'])
		#pyplot.plot(history.history['val_acc'])
		#pyplot.plot(history.history['val_loss'])
		#pyplot.title('model train vs validation loss')
		#pyplot.ylabel('loss')
		#pyplot.xlabel('epoch')
		#pyplot.legend(['loss', 'val_loss'], loc='upper right')
		#pyplot.savefig(os.path.join(os.getcwd(), self.weights_save, "loss.png"), bbox_inches='tight')


	def count_inputs(self, hd5f_files):
		count = 0
		for h5 in hd5f_files:
			with h5py.File(os.path.join(os.getcwd(), "ptn", h5), 'r') as hf:
				x_train = hf["x_train"][:]
				count += x_train.shape[0]

		return count

	def validate_all(self, validate_generator, count):
		steps = self.train_batch_size // count
		score = self.model.evaluate_generator(validate_generator, steps)
		print("Validation (Accuracy) = {}".format(score))
		return score


def main():
	test = Tak_Train()

	#test.define_LSTM_model(False)
	test.define_Comb_model()

	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	random.shuffle(white_train_files)
	
	#count = test.count_inputs(white_train_files[:-5])
	#print("Training on {} inputs".format(count))
	training_generator = test.training_files_generator(white_train_files[:-5])

	#count = test.count_inputs(white_train_files[-5:])
	#print("Validation on {} inputs".format(count))
	validation_generator = test.training_files_generator(white_train_files[-5:])

	test.train_generator(training_generator, validation_generator)


def validate():
	fill_correct = 0
	count = 0

	test = Tak_Train()

	test.define_LSTM_model(False)
	print("Loaded model")

	# load weights into new model
	#test.model.load_weights(os.path.join("7-LSTM","White-weights-improvement-64-0.99.hdf5"))
	#print("Loaded model from disk")

	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	random.shuffle(white_train_files)

	validation_all = test.training_files_generator(white_train_files)

	for x_data_array, y_data_array in validation_all:
		for index in range(x_data_array.shape[0]):
			#print("Array shape {}".format(x_data_array.shape))
			valid = binary_accuracy_np(test.predict(x_data_array[index]), y_data_array[index])
			print(valid)
			if valid == 1:
				fill_correct += 1
			count += 1
			print("{} correct out of {}".format(fill_correct, count))


	


if __name__ == '__main__':
	#validate()
	for _ in range(5):
		main()
