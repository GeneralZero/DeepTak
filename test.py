from board import TakBoard
import numpy as np
import h5py, os
np.set_printoptions(threshold=np.nan)

if __name__ == '__main__':
	test = TakBoard(5)
	before = None
	after = None

	with h5py.File(os.path.join(os.getcwd(), "ptn", "White_train_rot0_part0.h5"), 'r') as hf:
		# complex move 15,34,65,67,90,171,173,181
		# move
		# place
		before = hf["x_train"][:]
		after = hf["y_train"][:]

		for x, _ in enumerate(before):

			white_move=True

			test.set_np_game_board(before[x], white_move)

			changes = test.get_move_from_new_board(after[x])

			#Place 
			if len(changes) == 1:
				change = changes[0]

				if len(change["move_cell"]) == 1:
					test.place("", change["index"])
				else:
					test.place(change["move_cell"][0], change["index"])

				print("[Place] {} {}".format(change["move_cell"][0], change["index"]))
			else:
				#Move
				movement_array = [row for row in changes]

				start = ""
				end = ""

				reverse = False

				for index, change in enumerate(changes):
					if change["diff"] > 0:
						#print("Start is " + change["index"])

						start = change["index"]
						movement_array.pop(index)

						if index == 0:
							movement_array = movement_array[::-1] 
							end = changes[-1]["index"]
						else:
							end = changes[0]["index"]
							#print(changes[0])
						break

				count_array = []
				for elem in movement_array:
					count_array.append(elem["diff"] * -1)

				print("[Move]  Start: {}, End: {}, Array: {}".format(start, end, count_array))
