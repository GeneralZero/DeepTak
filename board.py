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
		self.board = np.zeros(self.board_size * self.board_size + 3, dtype=np.uint64)
		self.board.fill(0)
		#16 levels

	def possoble_placements(self):
		return [x for x in self.board if x == 0]

	def get_current_board(self):
		#Set Player Turn
		self.board[self.board_size * self.board_size] = self.player1_turn

		#Set Player 1 capstone
		self.board[self.board_size * self.board_size + 1] = self.capstone_player1

		#Set Player 2 capstone
		self.board[self.board_size * self.board_size + 2] = self.capstone_player2

		return self.board

	def get_current_string_board(self):
		board = []
		for x in range(self.board_size -1, -1, -1):
			row = []
			for i in range(self.board_size):
				test = self.pretty_print_cell(self.board[x*self.board_size + i])
				row.append(test)
			board.append(row)
		
		return board


	def winner(self):
		top_array_road_white = [self.peak_from_index(x) % 2 == 0 and self.peak_from_index(x) & 0x02 == 0 for x in xrange(self.board_size * self.board_size)]
		top_array_road_black = [self.peak_from_index(x) % 2 == 1 and self.peak_from_index(x) & 0x02 == 0 for x in xrange(self.board_size * self.board_size)]

		black_win = find_road(top_array_road_black, 1)
		white_win = find_road(top_array_road_white, 0)

		if black_win != None:
			return ["Black", black_win]

		if white_win != None:
			return ["White", white_win]
				
		#look at edges and try to connect

	"""
	def find_roads(self, top_array, color):
		edges = [range(0,self.board_size) + range(self.board_size,self.board_size*self.board_size+1,self.board_size)]

		for x in edges:
			if top_array[x]:
				#Edge is color start 

				#Connect all roads to list
				#See if a right set in list see if a left set in list
				#See if a top set in list see if a bottom set in list"""

	def index_size(self, index):
		stack = self.board[index]
		size = 0

		while stack > 0:
			size +=1
			stack >> 3

		return min(self.board_size, size)

	def possoble_moves(self, color):
		#Color white=0, black=1
		controlled_stacks = [x for x in self.board if self.peak_from_index(x) % 2 == color]

		top_stacks = [x for x in self.board if self.peak_from_index(x) & 0x06 != 0]


		#start,end,placementarray
		moves = []

		for index in controlled_stacks:
			max_stack_move = self.index_size(index)

			length_ends = [1,1,1,1]#Top,Right,Bottom,Left
			ends = []
			#Get Left right ends
			for x in range(max_stack_move+1):
				#Right
				if x+1 != length_ends[1] and (index +length_ends[1])%self.board_size !=0 and (index + length_ends[1] not in top_stacks):
					ends.append(index +length_ends[1])

				#Left
				if x+1 != length_ends[3] and (index +length_ends[3])%self.board_size !=0 and (index + length_ends[3] not in top_stacks):
					ends.append(index +length_ends[3])

				#Top
				if x+1 != length_ends[0] and (index +length_ends[0])%self.board_size !=0 and (index + length_ends[0] not in top_stacks):
					ends.append(index +length_ends[0])
				
				#Bottom
				if x+1 != length_ends[2] and (index +length_ends[2])%self.board_size !=0 and (index + length_ends[2] not in top_stacks):
					ends.append(index +length_ends[2])


		#Add capstone break walls if capstone
			
			#check all directons for walls/capstones and end of line
			pass

	def index_to_grid(self, index):
		return chr(ord('A')+ (index%self.board_size)) + str((index//self.board_size)+1)

	def grid_to_index(self, grid):
		return self.board_size * (ord(grid[0]) - ord('A')) + int(grid[1:]) - 1

	def get_square(self, grid_location):
		return self.board[self.grid_to_index(grid_location)]

	def is_square_empty(self, index):
		return (0x07 & self.board[index]) == 0x07

	def place(self, piece, grid_location, is_white):
		#print("Place: {}, gridloc:{} square:{}".format(piece, grid_location, self.get_square(grid_location)))

		#self.pretty_print_board()

		if self.get_square(grid_location) != 0:
			raise Exception("Invalid Placement Location: gridlocation={}, currentsquare={}".format(grid_location, self.get_square(grid_location)))

		place_peice = int(6 + is_white)

		if piece == None or piece == "":
			place_peice = int(6 + is_white)

		elif piece.lower() == "w":
			place_peice = int(2 + is_white)

		elif piece.lower() == "c":
			place_peice = int(4 + is_white)
		
		else:
			raise ValueError("Invalid piece: {}".format(piece))

		#Place on board
		self.board[self.grid_to_index(grid_location)] = place_peice

		#Change turn
		self.player1_turn = not self.player1_turn

	def pop_from_index(self, index):
		if index > -1 and index < self.board_size * self.board_size:
			top = 0x07 & int(self.board[index])
			self.board[index] = int(self.board[index]) >> 3
			return top

	def peak_from_index(self, index):
		if index > -1 and index < self.board_size * self.board_size:
			top = 0x07 & int(self.board[index])
			return top

	def push_to_index(self, index, piece):
		if index > -1 and index < self.board_size * self.board_size:
			self.board[index] = int(self.board[index]) << 3
			self.board[index] += piece

	def pretty_print_board(self):
		for x in range(4, -1, -1):
			print("{},{},{},{},{}".format("(" + "".join(self.pretty_print_cell(self.board[x*5])) + ")",
										  "(" + "".join(self.pretty_print_cell(self.board[x*5+1])) + ")",
										  "(" + "".join(self.pretty_print_cell(self.board[x*5+2])) + ")",
										  "(" + "".join(self.pretty_print_cell(self.board[x*5+3])) + ")",
										  "(" + "".join(self.pretty_print_cell(self.board[x*5+4])) + ")"))

	def pretty_print_cell(self, index):
		ret = []
		while index > 0:
			#Get Top
			top = 0x07 & int(index)
			index = int(index) >> 3

			#Flats
			if (0x06 & top == 6):
				if (top % 2 == 1):
					ret.append("b")
				else:
					ret.append("w")

			#Caps
			elif (0x04 & top == 4):
				if (top % 2 == 1):
					ret.append("Cb")
				else:
					ret.append("Cw")

			#Standing
			elif (0x02 & top == 2):
				if (top % 2 == 1):
					ret.append("Sb")
				else:
					ret.append("Sw")

		return ret

	def move(self, start, end, move_array):
		#Valid Size
		if np.sum(move_array) > self.board_size:
			raise Exception("Moving more tiles than board size")

		int_start = self.grid_to_index(start)
		int_end = self.grid_to_index(end)

		#print("Move: s:{}, e:{} square:{}".format(start, end, self.get_square(start)))

		#self.pretty_print_board()

		#Valid Move
		if start[0] == end[0]:
			#Up and Down
			if int_start > int_end:
				#Down
				current_index = int_end
				pop_array = []
				for count in reversed(move_array):
					for x in range(count):
						pop_array.append(self.pop_from_index(int_start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index - self.board_size
			else:
				#Up
				current_index = int_end
				pop_array = []
				for count in reversed(move_array):
					for x in range(count):
						pop_array.append(self.pop_from_index(int_start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index + self.board_size

		elif start[1] == end[1]:
			#left and right
			if int_start > int_end:
				#Left
				current_index = int_end
				pop_array = []
				for count in reversed(move_array):
					for x in range(count):
						pop_array.append(self.pop_from_index(int_start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index - 1
			else:
				#Right
				current_index = int_end
				pop_array = []
				for count in reversed(move_array):
					for x in range(count):
						pop_array.append(self.pop_from_index(int_start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index + 1
		else:
			raise Exception("Move is not up, down, left, or right")

		#Change turn
		self.player1_turn = not self.player1_turn

if __name__ == '__main__':
	p= TakBoard(5)

	print(p.grid_to_index("E5"))
	print(p.index_to_grid(24))
	#grid = "E5"

	#print (5 * (ord('E') - ord('A')) + int(grid[1:]))