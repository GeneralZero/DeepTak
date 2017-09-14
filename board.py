import numpy as np
#Flat White = 0 110
#Flat Black = 1 111
#Wall White = 2 010
#Wall Black = 3 011
#Caps White = 4 100
#Caps Black = 5 101

#Top is higher
#ex. (50101)8

# White (piece % 2) == 0
# Black (piece % 2) == 1

# Road worthy (piece & 0x02) == 0

# Top of stack (piece & 0x06) != 1

"""
|6|
|0|1|2|3|4|5|
________
"""

class TakBoard():
	"""docstring for TakBoard"""
	def __init__(self, size):
		self.capstone_player1 = False
		self.capstone_player2 = False

		self.player1_turn = True
		
		self.board_size = size
		self.max_height = 64
		self.board = [[[] for x in range(self.board_size)] for x in range(self.board_size)]


	def get_current_string_board(self):
		return self.board

	def get_numpy_board(self):
		encode = {
			"w": 1,
			"b": 2,
			"sw": 3,
			"sb": 4,
			"ww": 3,
			"wb": 4,
			"cw": 5,
			"cb": 6,
		}

		board_array=[]
		
		for row_index, rows in enumerate(self.board):
			row_array = []
			for col_index, cols in enumerate(rows):
				cell = []
				for height in cols:
					cell.append(encode[height.lower()])
				
				cell = np.pad(np.array(cell), (0, self.max_height - len(cell)), 'constant')
				row_array.append(cell)
			board_array.append(row_array)

		return np.array(board_array)

	def get_square(self, grid_location):
		x = (ord(grid_location[0].upper()) - ord("A"))
		y =  self.board_size - int(grid_location[1:])
		return self.board[y][x]

	def set_square(self, grid_location, peices):
		x = (ord(grid_location[0].upper()) - ord("A"))
		y =  self.board_size - int(grid_location[1:])
		self.board[y][x] = peices

	def append_square(self, grid_location, peice):
		x = (ord(grid_location[0].upper()) - ord("A"))
		y =  self.board_size - int(grid_location[1:])
		self.board[y][x].append(peice)

	def place(self, piece, grid_location, is_white):
		#print("Place: {}, gridloc:{} square:{}".format(piece, grid_location, self.get_square(grid_location)))

		#self.pretty_print_board()

		if self.get_square(grid_location) != []:
			raise Exception("Invalid Placement Location: gridlocation={}, currentsquare={}".format(grid_location, self.get_square(grid_location)))

		if is_white:
			color = "w"
		else:
			color = "b"

		if piece == None or piece == "":
			place_peice = color

		elif piece.lower() == "w" or piece.lower() == "s":
			place_peice = "S"+ color

		elif piece.lower() == "c":
			place_peice = "C"+ color
		
		else:
			raise ValueError("Invalid piece: {}".format(piece))

		#Place on board
		x = (ord(grid_location[0].upper()) - ord("A"))
		y =  self.board_size - int(grid_location[1:])
		self.board[y][x].append(place_peice)

		#Change turn
		self.player1_turn = not self.player1_turn

	def move(self, start, end, move_array):
		#Valid Size
		if np.sum(move_array) > self.board_size:
			raise Exception("Moving more tiles than board size")

		#print("Move: s:{}, e:{} square:{}".format(start, end, self.get_square(start)))

		count = np.sum(move_array)
		current_square = start
		#self.pretty_print_board()

		##TODO: Add wall smash to move

		#Valid Move
		if start[0] == end[0]:
			#Up and Down
			if int(start[1:]) > int(end[1:]):
				#Down

				#Set Start
				pop_array = self.get_square(start)[-count:]
				self.set_square(start, self.get_square(start)[:-count])

				for pops in move_array:
					current_square = current_square[0] + str(int(current_square[1:]) -1)
					for x in range(pops):
						self.append_square(current_square, pop_array.pop(0))

			else:
				#Up

				#Set Start
				pop_array = self.get_square(start)[-count:]
				self.set_square(start, self.get_square(start)[:-count])

				for pops in move_array:
					current_square = current_square[0] + str(int(current_square[1:]) +1)
					for x in range(pops):
						self.append_square(current_square, pop_array.pop(0))

		elif start[1:] == end[1:]:
			#left and right
			if start[0] > end[0]:
				#Left
				
				#Set Start
				pop_array = self.get_square(start)[-count:]
				self.set_square(start, self.get_square(start)[:-count])

				for pops in move_array:
					current_square = chr(ord(current_square[0]) - 1) + current_square[1:]
					for x in range(pops):
						self.append_square(current_square, pop_array.pop(0))

			else:
				#Right
				
				#Set Start
				pop_array = self.get_square(start)[-count:]
				self.set_square(start, self.get_square(start)[:-count])

				for pops in move_array:
					current_square = chr(ord(current_square[0]) + 1) + current_square[1:]
					for x in range(pops):
						self.append_square(current_square, pop_array.pop(0))
		else:
			raise Exception("Move is not up, down, left, or right")

		#Change turn
		self.player1_turn = not self.player1_turn

if __name__ == '__main__':
	p= TakBoard(5)

	p.place("", "E1", False)
	p.place("", "D1", True)
	p.place("", "D2", True)
	p.place("", "D3", False)
	p.place("", "C2", True)

	p.place("", "E2", False)
	p.place("", "E3", True)
	p.place("", "D4", False)
	p.place("", "B2", True)
	p.move("D3", "E3", [1])
	p.place("C", "D3", True)
	p.place("", "E4", False)
	p.move("D3", "E3", [1])
	p.place("", "A2", False)
	p.move("E3", "E1", [1, 2])
	test = p.get_numpy_board()
	print(test.shape)
	
	p.place("", "A3", False)
	p.place("", "A1", True)
	p.move("A2", "B2", [1])
	p.place("", "A2", True)
	p.move("A3", "A2", [1])
	p.place("", "C3", True)
	p.place("", "B3", False)
	p.place("", "B4", True)
	p.place("", "C4", False)
	p.place("", "B1", True)
	p.place("W", "C1", False)
	p.move("E1", "C1", [2, 1])