from board import TakBoard
import numpy as np
import h5py, os
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
	test = TakBoard(5)
	test2 = TakBoard(5)
	before = None
	after = None

	with h5py.File(os.path.join(os.getcwd(), "ptn", "White_train_rot0_part0.h5"), 'r') as hf:
		#37 move
		#54
		before = hf["x_train"][:][37]
		after = hf["y_train"][:][37]

	white_move=True


	#print(before)
	#print(after)

	test.set_np_game_board(before, white_move)

	#for x in test.get_current_string_board():
	#	print(x)

	test2.set_np_game_board(after, not white_move)
	changes = test.get_move_from_new_board(after)

	#for x in test2.get_current_string_board():
	#	print(x)

	print(changes)

	if len(changes) >= 2:
		print()


	#Place 
	if len(changes) == 1:
		change = changes[0]
		out = test.get_index_from_int(change['x'],change['y'])
		print(out)

		if len(change["move_cell"]) == 1:
			test.place("", out)
		else:
			test.place(change["move_cell"][0], out)
	else:
		#Move
		index_list = []

		for change in changes:
			out = test.get_index_from_int(change['x'],change['y'])
			index_list.append(out)

		print(index_list)