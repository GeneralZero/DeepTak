import keras
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Reshape, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint

import pickle, os, zipfile, random, math
import h5py
import pandas as pd

class Tak_Train(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		self.tak_size = 5
		self.tak_height = 64
		self.hidden_units = 1024
		self.train_batch_size = 1000

		self.iterations = 3
		self.epochs = 100

		self.define_model()

	def define_model(self):
		print("Setup Model")
		self.model = Sequential()
		#self.model.add(Conv2D(self.tak_height, (1, 1), activation='relu', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		#self.model.add(BatchNormalization())
		self.model.add(Reshape((self.tak_size * self.tak_size, self.tak_height), input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(LSTM(self.hidden_units, return_sequences=True, input_shape=(self.tak_size * self.tak_size, self.tak_height)))
		self.model.add(BatchNormalization())

		#self.model.add(LSTM(self.hidden_units, return_sequences=True))
		#self.model.add(BatchNormalization())

		#self.model.add(LSTM(self.hidden_units, return_sequences=True))
		#self.model.add(BatchNormalization())

		self.model.add(Dense(self.tak_height, activation='relu'))
		self.model.add(Reshape((self.tak_size, self.tak_size, self.tak_height), input_shape=(self.tak_size * self.tak_size, self.tak_height)))

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
			print("Getting Training data from {}".format(file_name))

			with h5py.File(os.path.join(os.getcwd(), "ptn", file_name), 'r') as hf:
				x_train = hf["x_train"][:]
				y_train = hf["y_train"][:]

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

					print("Returning (x_shape: {}, y_shape:{})".format(all_x_train.shape, all_y_train.shape))

					yield (all_x_train, all_y_train)

				#Check for leftover data and add to all_trains
				if array_size > end_index:
					all_x_train = x_train[end_index:array_size]
					all_y_train = x_train[end_index:array_size]
					left_over_size = array_size - end_index
					left_overs = True

	def train(self, file_names):
		print("Startring Training")
		callback = ModelCheckpoint("weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

		for file_name in file_names:
			with h5py.File(os.path.join(os.getcwd(), "ptn", file_name), 'r') as hf:
				x_train = hf["x_train"][:]
				y_train = hf["y_train"][:]

				print("Getting Training data from {}".format(file_name))

				self.model.fit(x_train, y_train, shuffle=True, callbacks=[callback], batch_size=x_train.shape[0], validation_split=0.95, epochs=self.epochs, verbose=2)

	def train_generator(self, train_generator):
		#Make generator to return data from training file

		callback = ModelCheckpoint("weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

		self.model.fit_generator(train_generator, self.train_batch_size, epochs=self.epochs, callbacks=[callback], verbose=2)


if __name__ == '__main__':
	test = Tak_Train()


	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	print(white_train_files)
	generator = test.training_files_generator(white_train_files)

	test.train_generator(generator)