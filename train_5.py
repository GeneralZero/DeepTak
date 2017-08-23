import tensorflow as tf
import numpy as np

import pickle, os

###Start

def load_from_pickle():

	file_list = [filename for filename in os.listdir('.\\ptn') if filename.startswith("gamedata_")]
	for files in file_list:
		#print(files)
		yield pickle.load(open(".\\ptn\\" + files, "rb"))

def load_data(split=0.9):
	dataset = []

	train_dataset = dataset[:int(math.ceil(split*len(dataset)))]
	test_dataset = dataset[int(math.ceil(split*len(dataset))):]

	return train_dataset, test_dataset

def train():
	test = load_from_pickle()
	for x in test:
		print(len(x))
		print(x)
		return


if __name__ == '__main__':
	train()