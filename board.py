import numpy as np
#Flat White = 0
#Flat Black = 1
#Wall White = 2
#Wall Black = 3
#Caps White = 4
#Caps Black = 5
#Top is lower on 

class TakBoard(object):
	"""docstring for TakBoard"""
	def __init__(self, size):
		self.capstone_player1 = False
		self.capstone_player2 = False
		
		self.board_size = size
		self.board = np.zero(self.board_size * self.board_size)

	def index_to_grid(self, index):
		return chr(ord('A')+ (index%self.board_size)) + str(index/self.board_size)

	def grid_to_index(self, grid):
		return self.board_size * (ord('A') - ord(grid[0])) + int(grid[1:])

	def get_square(self, grid_location):
		return self.board[self.grid_to_index(grid_location)]

	def place(self, num_type, grid_location):
		index = self.get_square(grid_location)
		if self.get_square(grid_location) != 0:
			raise Exception("Invalid Placement Location")
		self.board[self.grid_to_index(grid_location)] = num_type

	def move(self, start, end, move_array):
		#Valid Size
		if np.sum(move_array) > self.board_size:
			raise Exception("Moving more tiles than board size")

		#Valid Move
		if start[0] == end[0]:
			#Up and Down
			if self.grid_to_index(start) > self.grid_to_index(end):
				#Down
			else:
				#Up

		elif start[1] == end[1]:
			#left and right
			if self.grid_to_index(start) > self.grid_to_index(end):
				#Left
			else:
				#Right

		else:
			raise Exception("Move is not up, down, left, or right")

		
			if
		#Figure out direction of move

		#Count peices
