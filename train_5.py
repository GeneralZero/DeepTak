import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM

from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D

import pickle, os, zipfile, random

class Tak_Train(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		self.tak_size = 5
		self.hidden_units = 100
		self.batch_size = 32

		self.iterations = 3
		self.epochs = 3

		#self.define_model()

	def load_data_from_zip(self, white=True):
		zip_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".zip")]

		if white:
			white_zip_files = [filename for filename in zip_files if filename.startswith("White")]
			#White zip files
			for white_zip_file in white_zip_files:
				#print(white_zip_file)
				game_zip_file = zipfile.ZipFile(os.path.join(os.getcwd(), "ptn", white_zip_file))

				yield {name: game_zip_file.read(name) for name in game_zip_file.namelist()}

		else:
			black_zip_files = [filename for filename in zip_files if filename.startswith("Black")]
			#Black zip files
			for black_zip_file in black_zip_files:
				game_zip_file = zipfile.ZipFile(os.path.join(os.getcwd(), "ptn", black_zip_file))

				yield {name: game_zip_file.read(name) for name in game_zip_file.namelist()}

	def load_data_from_pickle(self, zip_files_translations):
		tak_games = []

		for translation_zip in zip_files_translations:
			for file_name in translation_zip:
				#print(file_name)

				game_array = pickle.loads(translation_zip[file_name])
				#print(np.array(game_array).shape)
				tak_games.append(game_array)
				
		return tak_games

	def load_data(self, split=0.9):
		print("Getting Load Data")

		#Load White Data
		white_tak_games = self.load_data_from_pickle(self.load_data_from_zip(True))

		random.shuffle(white_tak_games)

		#Get White Input and Output data
		white_x_train, white_y_train = self.board_to_training_data(white_tak_games, is_white=True)

		print(white_x_train.shape)
		print(white_y_train.shape)

		#Split White to train and test data
		white_x_test = white_x_train[int(math.ceil(split*len(white_x_train))):]
		white_x_train = white_x_train[:int(math.ceil(split*len(white_x_train)))]
		
		white_y_test = white_y_train[int(math.ceil(split*len(white_y_train))):]
		white_y_train = white_y_train[:int(math.ceil(split*len(white_y_train)))]

		print("Training White Win data")


		self.train(white_x_train, white_y_train)

		#Load Black Data
		black_tak_games = self.load_data_from_pickle(self.load_data_from_zip(False))

		#Get black Input and Output data
		black_x_train, black_y_train = self.board_to_training_data(np.random.shuffle(black_tak_games), is_white=False)

		#Split black to train and test data
		black_x_test = black_x_train[int(math.ceil(split*len(black_x_train))):]
		black_x_train = black_x_train[:int(math.ceil(split*len(black_x_train)))]
		
		black_y_test = black_y_train[int(math.ceil(split*len(black_y_train))):]
		black_y_train = black_y_train[:int(math.ceil(split*len(black_y_train)))]

		print("Training Black Win Data")

		self.train(black_x_train, black_y_train)

		print("Finish Training Data")
		print("Saving Weights")

		self.model.save_weights("Tak_wights.h5")

	def board_to_training_data(self, all_tak_game_states, is_white=True):
		x_data = []
		y_data = []

		for game_index, tak_game_states in enumerate(all_tak_game_states):
			print("Generating Traing Data for Game Index {}".format(game_index))

			#Start Game
			is_white_move = False

			pre_moves = []
			post_moves = []

			print(tak_game_states)
			
			for move_index, game_state in enumerate(tak_game_states):
				if move_index == 2:
					is_white_move = False

				if is_white == is_white_move:
					#Is Black and is black move or is white and is white move
					#Pre_move
					pre_moves.append(np.array(game_state))
				else:
					#Is black and is white move or is white and is black move
					#Post_move
					post_moves.append(np.array(game_state))


				#Update
				is_white_move = not is_white_move

			x_data.append(np.array(pre_moves))
			y_data.append(np.array(post_moves))

		return np.array(x_data), np.array(y_data)


	def define_model(self):
		print("Setup Model")
		self.model = Sequential()

		self.model.add(LSTM(self.hidden_units, input_shape=(None, self.tak_size, self.tak_size), return_sequences=True))
		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(LSTM(self.hidden_units, return_sequences=True))
		self.model.add(LSTM(self.hidden_units, return_sequences=True))

		self.model.compile(loss='mean_squared_error', metrics=['accuracy'])

		self.model.summary()

	def train(self, x_train, y_train):
		for iters in range(1, self.iterations):
			print("Itteration #{}".format(iters))

			self.model.fit(x_train, y_train,
				batch_size=self.batch_size,
				epochs=self.epochs)


if __name__ == '__main__':
	test = Tak_Train()

	test.load_data()