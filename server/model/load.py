import numpy as np
import keras.models
from keras.models import load_model
from scipy.misc import imread, imresize,imshow
import tensorflow as tf


def init(): 
	loaded_model = load_model(r'./model/model4b-Copy1.05-1.27.hdf5')
	graph = tf.get_default_graph()
	class_dictionary = np.load(r'./model/class_indices.npy').item()
	print('initialization successfull')
	return loaded_model,graph,class_dictionary