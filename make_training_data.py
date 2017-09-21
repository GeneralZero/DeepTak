import pickle, os, zipfile, random, math
import h5py
import numpy as np

np.set_printoptions(threshold=np.nan)

class gen_Tak(object):
	"""docstring for Tak_Train"""
	def __init__(self):
		pass

	def load_data_from_h5(self, file_name=None, white=True):
		print(file_name)
		if file_name != None:
			if white:
				print("Reading file {}".format(file_name))
				with h5py.File(os.path.join(os.getcwd(), "ptn", file_name), 'r') as hf:
					names = [name for name in hf]
					random.shuffle(names)
					for name in names:
						yield hf[name][:]
		else:
			zip_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]

			if white:
				white_zip_files = [filename for filename in zip_files if filename.startswith("White_Win")]
				#White zip files
				for white_zip_file in white_zip_files:
					print("Reading file {}".format(white_zip_file))
					with h5py.File(os.path.join(os.getcwd(), "ptn", white_zip_file), 'r') as hf:
						names = [name for name in hf]
						random.shuffle(names)
						for name in names:
							yield hf[name][:]

			else:
				black_zip_files = [filename for filename in zip_files if filename.startswith("Black_Win")]
				#Black zip files
				for black_zip_file in black_zip_files:
					with h5py.File(os.path.join(os.getcwd(), "ptn", black_zip_file)) as hf:
						yield [hf[name][:] for name in hf]

	def generate_training_data(self, file_name, part):
		print("Generating Training Data")

		all_x_train = None 
		all_y_train = None

		first = True

		i=0
		#Load White Data
		for index, white_tak_games in enumerate(self.load_data_from_h5(file_name, True)):
			#Get White Input and Output data
			#random.shuffle(white_tak_games)
			#print(white_tak_games.shape)
			#print("Finish Reading File")

			(white_x_train, white_y_train) = self.game_to_training_data(white_tak_games, index, is_white=True)
			#print(white_x_train.shape)
			#print(white_y_train.shape)

			if first:
				all_x_train = white_x_train
				all_y_train = white_y_train
				first = False

			elif len(white_x_train) != 0 or len(white_y_train) != 0:
				all_x_train = np.concatenate((all_x_train, white_x_train), axis=0)
				all_y_train = np.concatenate((all_y_train, white_y_train), axis=0)

			if index % 5000 == 4999:
				print("Finished index: {}".format(index))
				print(all_x_train.shape)
				print(all_y_train.shape)

				print("Saving data to White_train_rot{}_part{}.h5".format(part,i))

				with h5py.File(os.path.join(os.getcwd(), "ptn", "White_train_rot{}_part{}.h5".format(part,i)), "w") as hf:
					hf.create_dataset("x_train", data=all_x_train, compression="gzip", compression_opts=9)
					hf.create_dataset("y_train", data=all_y_train, compression="gzip", compression_opts=9)

				i+=1
				first = True

		print("Saving data to White_train_rot{}_part{}.h5".format(part,i))

		with h5py.File(os.path.join(os.getcwd(), "ptn", "White_train_rot{}_part{}.h5".format(part,i)), "w") as hf:
			hf.create_dataset("x_train", data=all_x_train, compression="gzip", compression_opts=9)
			hf.create_dataset("y_train", data=all_y_train, compression="gzip", compression_opts=9)

		print("Finished")
		print(all_x_train.shape)
		print(all_y_train.shape)



	def game_to_training_data(self, tak_game_states, game_index, is_white=True):
		x_data = []
		y_data = []

		#Start Game
		is_white_move = True

		total_moves = len(tak_game_states)
		#print(total_moves)

		if is_white and total_moves % 2 == 1:
			#Win by bad play on Black
			return ([], [])

		if not is_white and total_moves % 2 == 0:
			#Win by bad play on Black
			return ([], [])
		
		for move_index, game_state in enumerate(tak_game_states):

			if is_white == is_white_move:
				#Is Black and is black move or is white and is white move
				#Pre_move
				x_data.append(np.array(game_state, dtype=int))
			else:
				#Is black and is white move or is white and is black move
				#Post_move
				if move_index == 0:
					##Skip empty board for black
					pass
				#print("Diff{}  ".format(move_index), len(game_state))
				y_data.append(np.array(game_state, dtype=int))


			#Update
			is_white_move = not is_white_move

		return (np.array(x_data), np.array(y_data))

if __name__ == '__main__':
	test = gen_Tak()
	
	training_files = [filename for filename in os.listdir(os.path.join(os.getcwd(), "ptn")) if filename.endswith(".h5")]
	white_train_files = [filename for filename in training_files if filename.startswith("White_Win_size_5_rot")]
	white_train_files = sorted(white_train_files)
	test.generate_training_data(white_train_files[2], 2)