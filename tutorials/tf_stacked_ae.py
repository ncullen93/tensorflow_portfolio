"""
Stacked Autoencoder
"""
from collections import OrderedDict
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers, framework


def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
	with tf.Session() as sess:
		if model_path:
			saver.restore(sess, model_path)
		X_test = mnist.test.images[:n_test_digits]
		outputs_val = outputs.eval(feed_dict={X: X_test})

	fig = plt.figure(figsize=(8, 3 * n_test_digits))
	for digit_index in range(n_test_digits):
		plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
		plot_image(X_test[digit_index])
		plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
		plot_image(outputs_val[digit_index])



class tfStackedAutoEncoder(object):
	"""
	Train a Stacked AE in a single graph, using multiple "phases"
	"""


	def __init__(self, h_layers, lr=1e-2, l2_pen=1e-4, nb_epoch=100, 
			batch_size=None, act_fn='elu'):
		self.h_layers = h_layers
		self.lr = lr
		self.nb_epoch = nb_epoch
		self.batch_size = batch_size
		self.l2_pen = l2_pen
		
		if act_fn == 'elu':
			self.act_fn=tf.nn.elu
		elif act_fn == 'relu':
			self.act_fn = tf.nn.relu
		else:
			self.act_fn = act_fn

		self.reg_fn = layers.l2_regularizer(l2_pen)
		self.init_fn = layers.variance_scaling_initializer()

	def fit(self, X):
		X_data = X

		X = tf.placeholder(tf.float32, shape=(None,X_data.shape[-1]))

		# create weights/bias dictionary
		layer_shapes = [X_data.shape[1]] + self.h_layers + [X_data.shape[1]]
		w_dict = OrderedDict()
		b_dict = OrderedDict()
		with tf.name_scope('vars'):
			for i in range(len(layer_shapes)-1):
				w_name = 'w_%i'%i
				b_name = 'b_%i'%i
				w_shape = (layer_shapes[i],layer_shapes[i+1])
				w_dict[w_name] = tf.Variable(self.init_fn(w_shape), dtype=tf.float32, name=w_name)
				b_shape = layer_shapes[i+1]
				b_dict[b_name] = tf.Variable(tf.zeros(b_shape), dtype=tf.float32, name=b_name)

		# create layers
		layers = {}
		layers[0] = X
		with tf.name_scope('layers'):
			for i, (w,b) in enumerate(zip(w_dict.values(),b_dict.values())):
				if i < len(w_dict)-1:
					layers[i+1] = self.act_fn(tf.matmul(layers[i],w)+b) 
				else:
					layers[i+1] = tf.matmul(layers[i],w)+b
		
		# create phases
		mid_idx = int((len(w_dict)-1) / 2)
		n_phases = mid_idx
		phase_training_ops = []

		for phase_idx in range(n_phases):
			pass

if __name__=='__main__':
	tf.reset_default_graph()
	from keras.datasets.mnist import load_data
	(xtrain,xtest),(ytrain,ytest) = load_data()
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	xtrain = MinMaxScaler((0,1)).fit_transform(xtrain.astype('float64').reshape(xtrain.shape[0],-1))


	h_layers = [300, 150, 300]
	act_fn = 'elu'
	lr = 1e-2
	l2_pen = 1e-4
	sae = tfStackedAutoEncoder(h_layers=h_layers)
	layers = sae.fit(xtrain)














