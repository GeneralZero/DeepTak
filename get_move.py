import numpy as np

import keras, os, random, h5py, datetime
import numpy as np

from train_5 import Tak_Train

from board import TakBoard
np.set_printoptions(threshold=np.nan)


def precision(y_true, y_pred):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	equal_matrix
	true_positives = K.sum(K.round(y_true * y_pred))
	predicted_positives = K.sum(K.round(y_pred))
	precision_ret = true_positives / (predicted_positives + K.epsilon())
	return precision_ret


game = TakBoard(5)

#Make Moves
game.place("", "B5")
game.place("", "B1")
game.place("", "C1")
game.place("", "B4")

begin = datetime.datetime.now()

ai = Tak_Train()
ai.define_LSTM_model()

end = datetime.datetime.now()

print("Startup time: {}s".format(end - begin))

before = game.get_numpy_board()
after = ai.predict(before)

print(before)

print()

print(after)

final = datetime.datetime.now()

print("Pick move time: {}s".format(final - end))

game.get_move_from_new_board(after)



