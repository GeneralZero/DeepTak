def count_inputs(self, hd5f_files)
	count = 0
	with h5py.File(os.path.join(os.getcwd(), "ptn", file_name), 'r') as hf:
		x_train = hf["x_train"][:]
		count += x_train.shape[0]

	return count

def validate_all(self, validate_generator):
	score = self.model.evaluate_generator(validate_generator, steps)
	print("Validation (Accuracy) = {}".format(score[1]))

#https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications


#https://keras.io/regularizers/
#https://github.com/fchollet/keras/issues/1498
#https://github.com/fchollet/keras/issues/2745

#Dropout
#https://keras.io/layers/core/

#Keras
#keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

###
###
###
def get_internal_cell(self)
	out_list = []
	for element in cell:
		for key, value in self.encode.itteritems():
			if value = element:
				out_list.append(key)

	return out_list

def set_np_game_board(self, move_board):
	#Get Rows
	for x, row in enumerate(self.board):
		for y, cell in enumerate(row):
			move_cell = self.get_internal_cell(move_board[x][y])
			self.board[x][y] = move_cell



def get_move_from_new_board(self, move_board):
	changes = []
	#Get Rows
	for x, row in enumerate(self.board):
		for y, cell in enumerate(row):
			#Convert cell to be compared
			move_cell = self.get_internal_cell(move_board[x][y])


			if len(cell) == len(move_cell):
				if cell != move_cell:
					print("Change in the elements at the index x:{}, y:{}".format())
					changes.append((x,y))
			else:
				print("Change in number of elements at index x:{}, y:{}".format())
				changes.append((x,y))