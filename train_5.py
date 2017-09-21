import keras
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Reshape, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger

from matplotlib import pyplot
import os, random
import h5py

class Tak_Train(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		self.tak_size = 5
		self.tak_height = 64
		self.hidden_units = 625
		self.train_batch_size = 100
		self.validate_batch_size = 30
		self.epochs = 1000
		self.dropout_rate = 0.3

		self.opt = keras.optimizers.Nadam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

	def define_LSTM_model(self):
		print("Setup Model")
		self.model = Sequential()
		#self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		#self.model.add(BatchNormalization())
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, return_sequences=True, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True, dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate))
		self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

		self.weights_save = "7-LSTM"
		self.load_weights()

		self.model.compile(loss='mean_squared_logarithmic_error', optimizer=self.opt, metrics=['accuracy'])

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
			sorted(training_files)
			print("Loading previous weights file " + training_files[-1])
			self.model.load_weights(os.path.join(os.getcwd(), self.weights_save, training_files[-1]))

	def define_Conv_model(self):
		print("Setup Model")
		self.model = Sequential()
		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())
		
		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.weights_save = "5-CONV"
		self.load_weights()

		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		self.model.summary()

		self.weights_save = "5-CONV"

	def define_Comb_model(self):
		print("Setup Model")
		self.model = Sequential()
		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())
		
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, return_sequences=True, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))


		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.weights_save = "7-COMB"
		self.load_weights()
		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

		self.model.summary()

	def define_Comb2_model(self):
		print("Setup Model")
		self.model = Sequential()
		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())
		
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, return_sequences=True, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())
		
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, return_sequences=True, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

		self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu'))
		self.model.add(BatchNormalization())

		self.weights_save = "10-COMB"
		self.load_weights()
		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

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
						all_y_train = np.concatenate((all_y_train, x_train[start_index:end_index]), axis=0)
						left_overs = False

					else:
						all_x_train = x_train[start_index:end_index]
						all_y_train = y_train[start_index:end_index]

					#print("Returning (x_shape: {}, y_shape:{})".format(all_x_train.shape, all_y_train.shape))

					yield (all_x_train, all_y_train)

				#Check for leftover data and add to all_trains
				if array_size > end_index:
					all_x_train = x_train[end_index:array_size]
					all_y_train = x_train[end_index:array_size]
					left_over_size = array_size - end_index
					left_overs = True

	def train_generator(self, training_generator, validation_generator):
		#Make generator to return data from training file
		callback1 = ModelCheckpoint(os.path.join(os.getcwd(), self.weights_save, "White-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"), monitor='val_acc', verbose=2, save_best_only=True, mode='max')
		#callback2 = keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), self.weights_save), histogram_freq=0, write_graph=True, write_images=True)

		history = self.model.fit_generator(training_generator, self.train_batch_size, epochs=self.epochs, callbacks=[callback1], validation_data=validation_generator, validation_steps=self.validate_batch_size, verbose=1)
		pyplot.plot(history.history['loss'])
		#pyplot.plot(history.history['acc'])
		#pyplot.plot(history.history['val_acc'])
		pyplot.plot(history.history['val_loss'])
		pyplot.title('model train vs validation loss')
		pyplot.ylabel('loss')
		pyplot.xlabel('epoch')
		pyplot.legend(['loss', 'val_loss'], loc='upper right')
		pyplot.savefig(os.path.join(os.getcwd(), self.weights_save, "loss.png"), bbox_inches='tight')


	def count_inputs(self, hd5f_files):
		count = 0
		for h5 in hd5f_files:
			with h5py.File(os.path.join(os.getcwd(), "ptn", h5), 'r') as hf:
				x_train = hf["x_train"][:]
				count += x_train.shape[0]

		return count

	def validate_all(self, validate_generator, files):
		self.count_inputs()
		training_files_generator
		score = self.model.evaluate_generator(validate_generator, steps)
		print("Validation (Accuracy) = {}".format(score[1]))


if __name__ == '__main__':
	test = Tak_Train()

	test.define_LSTM_model()

	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	random.shuffle(white_train_files)
	
	count = test.count_inputs(white_train_files[:-5])
	print("Training on {} inputs".format(count))
	training_generator = test.training_files_generator(white_train_files[:-5])

	count = test.count_inputs(white_train_files[-5:])
	print("Validation on {} inputs".format(count))
	validation_generator = test.training_files_generator(white_train_files[-5:])

	test.train_generator(training_generator, validation_generator)

	validation_all = test.training_files_generator(white_train_files)

	count = test.count_inputs(white_train_files)

	test.validate_all(validate_all, count)