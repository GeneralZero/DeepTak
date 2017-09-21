import numpy as np

import keras
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Reshape, Dense
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger


if __name__ == '__main__':
	#keras.load_model(os.path.join(os.getcwd(), "7-LSTM", "model.json"))
	