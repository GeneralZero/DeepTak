import numpy as np


game_test1 = []

test = np.sum([
				np.array([[0,3], [1,2]]), 
				np.array([[0], [5,6,1]])
			  ], axis=2)

print(test)