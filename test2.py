import numpy as np
np.set_printoptions(threshold=np.nan)

def binary_accuracy(y_true, y_pred):
	return K.mean(K.equal(y_true, K.round(y_pred)))

def precision(y_true, y_pred):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(y_true * y_pred))
	predicted_positives = K.sum(K.round(y_pred))
	precision_ret = true_positives / (predicted_positives + K.epsilon())
	return precision_ret