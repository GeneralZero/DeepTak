import numpy as np

import keras, os, random, h5py
import numpy as np

from train_5 import Tak_Train

from board import TakBoard


game = TakBoard(5)

#Make Moves
game.place("", "A1")
game.place("", "3E")
game.place("", "4E")

ai = Tak_Train()
ai.define_LSTM_model()

before = game.get_numpy_board()
after = ai.predict(before)

game.get_move_from_new_board(after)