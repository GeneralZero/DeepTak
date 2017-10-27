import keras
import numpy as np

from keras import backend as K

from keras.models import Sequential, model_from_json
from keras.metrics import binary_accuracy

from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Reshape, Dense, Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv2D, Conv1D, UpSampling2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger

import os, random
import h5py

np.set_printoptions(threshold=np.nan)

def move_accuracy_validate(y_true, y_pred):
	types = K.cast(K.equal(y_true[:,0], K.round(y_pred[:,0])), 'float32')

	sum0 = K.mean(types)

	place1 = K.cast(K.equal(y_true[:,1], K.round(y_pred[:,1])), 'float32')
	place2 = K.cast(K.equal(y_true[:,2], K.round(y_pred[:,2])), 'float32')
	place3 = K.cast(K.equal(y_true[:,3], K.round(y_pred[:,3])), 'float32')

	sum1 = K.mean(K.stack([place1, place2, place3]))

	move1 = K.cast(K.equal(y_true[:,4], K.round(y_pred[:,4])), 'float32')
	move2 = K.cast(K.equal(y_true[:,5], K.round(y_pred[:,5])), 'float32')
	move3 = K.cast(K.equal(y_true[:,6], K.round(y_pred[:,6])), 'float32')
	move4 = K.cast(K.equal(y_true[:,7], K.round(y_pred[:,7])), 'float32')
	move5 = K.cast(K.equal(y_true[:,8], K.round(y_pred[:,8])), 'float32')
	move6 = K.cast(K.equal(y_true[:,9], K.round(y_pred[:,9])), 'float32')
	move7 = K.cast(K.equal(y_true[:,10], K.round(y_pred[:,10])), 'float32')
	move8 = K.cast(K.equal(y_true[:,11], K.round(y_pred[:,11])), 'float32')

	sum2 = K.mean(K.stack([move1, move2, move3, move4, move5, move6, move7, move8]))

	return K.mean(K.stack([sum0, sum2, sum0, sum1]))

class Tak_Train(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		self.tak_size = 5
		self.tak_height = 64
		self.hidden_units = 1500
		self.number_of_samples = 100
		self.train_batch_size = 100
		self.validate_batch_size = 30
		self.epochs = 100
		self.dropout_rate = 0.1

		self.opt = keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

	def load_weights(self):
		if not os.path.exists(self.weights_save):
			os.makedirs(self.weights_save)

		#Save Config
		with open(os.path.join(os.getcwd(), self.weights_save, "model.json"), "w") as f:
			f.write(self.model.to_json())

		#Setup Model
		if os.path.exists(os.path.join(os.getcwd(), self.weights_save, "best.hdf5")):
			 self.model.load_weights(os.path.join(os.getcwd(), self.weights_save, "best.hdf5"))
		else:
			training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), self.weights_save)) if filename.endswith(".hdf5")]
			if len(training_files) != 0:
				training_files = sorted(training_files)
				print("Loading previous weights file " + training_files[-1])
				self.model.load_weights(os.path.join(os.getcwd(), self.weights_save, training_files[-1]))

	def define_Conv2_model(self):
		print("Setup Model")
		self.model = Sequential()

		#self.model.add(UpSampling2D(5, data_format='channels_last', input_shape=(self.tak_size, self.tak_size, self.tak_height)))
		self.model.add(Dense(20, input_shape=(self.tak_size, self.tak_size, 64)))
		self.model.add(Conv2D(2000, 4, data_format='channels_last'))
		self.model.add(Activation('relu'))

		self.model.add(Flatten())
		self.model.add(Dense(1000))
		self.model.add(Activation('relu'))

		self.model.add(Dense(250))
		self.model.add(Activation('relu'))

		self.model.add(Dense(75))
		self.model.add(Activation('relu'))

		self.model.add(Dense(24))
		self.model.add(Activation('relu'))

		self.model.add(Dense(12))
		#
		self.weights_save = "3-CONV"
		self.load_weights()
		self.model.compile(loss='mean_squared_error', optimizer=self.opt, metrics=[move_accuracy_validate])
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

				while (end_index + self.number_of_samples) < array_size:
					#Update indexes
					start_index = end_index
					end_index = start_index + self.number_of_samples - left_over_size
					left_over_size = 0

					#print("Start_index: {}, End_index: {}, Array_size: {}".format(start_index, start_index + self.number_of_samples, array_size))

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
		callback1 = ModelCheckpoint(os.path.join(os.getcwd(), self.weights_save, "best.hdf5"), monitor='val_move_accuracy_validate', verbose=2, save_best_only=True, mode='max')
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
	test.define_Conv2_model()
	pass
	#test.define_Conv_model()

	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	random.shuffle(white_train_files)
	
	#count1 = test.count_inputs(white_train_files[:-3])
	#print("Training on {} inputs".format(count))
	training_generator = test.training_files_generator(white_train_files[:-5])

	#count2 = test.count_inputs(white_train_files[-3:])
	#print("Validation on {} inputs".format(count))
	validation_generator = test.training_files_generator(white_train_files[-5:])

	test.train_generator(training_generator, validation_generator)


def validate():
	fill_correct = 0
	count = 0

	test = Tak_Train()

	test.define_Conv2_model()
	print("Loaded model")

	# load weights into new model
	test.model.load_weights(os.path.join(os.getcwd(),"3-CONV","White-weights-improvement-0.834.hdf5"))
	#print("Loaded model from disk")

	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_train")]
	random.shuffle(white_train_files)

	validation_all = test.training_files_generator(white_train_files)

	for x_data_array, y_data_array in validation_all:
		for index in range(x_data_array.shape[0]):
			#print("Array shape {}".format(x_data_array.shape))
			valid = np.equal(test.predict(x_data_array[index]), y_data_array[index])
			#print(valid)
			if valid.all():
				fill_correct += 1
			count += 1
		print("{} correct out of {}".format(fill_correct, count))

if __name__ == '__main__':
	validate()
	#for _ in range(500):
	#	try:
	#		main()
	#	except:
	#		pass

