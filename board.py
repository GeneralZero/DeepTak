import numpy as np
#Flat White = 0 000
#Flat Black = 1 001
#Wall White = 2 010
#Wall Black = 3 011
#Caps White = 4 101
#Caps Black = 5 100

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
		self.board = np.zero(self.board_size * self.board_size, dtype=uint128)
		self.white_ai_data = np.zero(self.board_size * self.board_size, dtype=uint32)
		self.black_ai_flags = np.zero(self.board_size * self.board_size, dtype=uint32)
		self.white_ai_data = np.zero(self.board_size * self.board_size, dtype=uint32)
		self.black_ai_flags = np.zero(self.board_size * self.board_size, dtype=uint32)
		#16 levels

	def possoble_placements(self):
		return [x for x in self.board if x == 0]

	def possoble_moves(self, color):
		#Color white=0, black=1
		controlled_stacks = [x for x in self.board if self.peak_from_index(x) % 2 == color]

		for stack in controlled_stacks:
			#check all directons
			pass

	def index_to_grid(self, index):
		return chr(ord('A')+ (index%self.board_size)) + str(index/self.board_size)

	def grid_to_index(self, grid):
		return self.board_size * (ord('A') - ord(grid[0])) + int(grid[1:])

	def get_square(self, grid_location):
		return self.board[self.grid_to_index(grid_location)]

	def place(self, piece, grid_location):
		index = self.get_square(grid_location)
		if self.get_square(grid_location) != 0:
			raise Exception("Invalid Placement Location")
		self.board[self.grid_to_index(grid_location)] = num_type

	def pop_from_index(self, index):
		if index > -1 and index < self.board_size * self.board_size:
			top = 0x07 & self.board[index]
			self.board[index] = self.board[index] >> 3
			return top

	def peak_from_index(self, index):
		if index > -1 and index < self.board_size * self.board_size:
			top = 0x07 & self.board[index]
			return top

	def push_to_index(self, index, piece):
		if index > -1 and index < self.board_size * self.board_size:
			self.board[index] = self.board[index] << 3
			self.board[index] += piece

	def move(self, start, end, move_array):
		#Valid Size
		if np.sum(move_array) > self.board_size:
			raise Exception("Moving more tiles than board size")

		start = self.grid_to_index(start)
		end = self.grid_to_index(end)

		#Valid Move
		if start[0] == end[0]:
			#Up and Down
			if start > end:
				#Down
				current_index = end
				pop_array = []
				for count in reversed(move_array):
					for x in xrange(count):
						pop_array.append(self.pop_from_index(start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index - self.board_size
			else:
				#Up
				current_index = end
				pop_array = []
				for count in reversed(move_array):
					for x in xrange(count):
						pop_array.append(self.pop_from_index(start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index + self.board_size

		elif start[1] == end[1]:
			#left and right
			if self.grid_to_index(start) > self.grid_to_index(end):
				#Left
				current_index = end
				pop_array = []
				for count in reversed(move_array):
					for x in xrange(count):
						pop_array.append(self.pop_from_index(start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index - 1
			else:
				#Right
				current_index = end
				pop_array = []
				for count in reversed(move_array):
					for x in xrange(count):
						pop_array.append(self.pop_from_index(start))
					
					for top in reversed(pop_array):
						self.push_to_index(current_index, top)
					current_index = current_index + 1
		else:
			raise Exception("Move is not up, down, left, or right")