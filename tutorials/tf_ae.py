"""
Autoencoder in TF

Using tf.contrib.layers.stack:
# FULLY CONNECTED:
layers.stack(x, layers.fully_connected, [32, 64, 128], scope='fc')
# CONV:
layers.stack(x, layers.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')

"""

from __future__ import division

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from collections import OrderedDict

class tfAutoEncoder(object):

	def __init__(self, h_layers, lr=1e-2, l2_pen=1e-4,
			nb_epoch=100, batch_size=None, activation_fn='elu',
			tied_weights=False, denoise=False):
		self.lr = lr
		self.nb_epoch = nb_epoch
		self.batch_size = batch_size
		self.l2_pen = l2_pen
		self.h_layers = h_layers
		self.tied_weights = tied_weights
		self.denoise = denoise
		if activation_fn == 'elu':
			self.activation_fn=tf.nn.elu
		elif activation_fn == 'relu':
			self.activation_fn = tf.nn.relu
		else:
			self.activation_fn = None

	def fit(self, X):
		h_layers 	= self.h_layers
		assert len(h_layers) >= 1, 'Must give at least one layer.'
		l2_pen 		= self.l2_pen
		nb_epoch 	= self.nb_epoch
		if X.ndim > 2:
			X_data	 	= X.reshape(X.shape[0],-1)
		else:
			X_data = X
		n_inputs 	= X_data.shape[1]
		n_outputs 	= n_inputs
		if self.batch_size is None:
			self.batch_size = n_inputs
		batch_size 	= self.batch_size
		lr 			= self.lr
		tied_weights= self.tied_weights

		activation_fn 	= self.activation_fn
		regularizer_fn 	= layers.l2_regularizer(self.l2_pen)
		initializer_fn 	= layers.variance_scaling_initializer()

		X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
		is_training = tf.placeholder(tf.bool,shape=(),name='is_training')
		if self.denoise == True:
			keep_prob = 0.5
			print 'Adding Denoising Dropout w/ prob: ' , keep_prob
			X = layers.dropout(X, keep_prob=keep_prob, is_training=is_training)

		weights_dict = OrderedDict()
		bias_dict = OrderedDict()
		layer_shapes = [n_inputs] + h_layers + [n_outputs]
		# create weights and bias variables
		with tf.name_scope('weights'):
			for i in range(len(layer_shapes)-1):
				w_key = 'w_%i'%i
				w_shape = (layer_shapes[i],layer_shapes[i+1])
				if tied_weights == False:
					weights_dict[w_key] = tf.Variable(initializer_fn(w_shape), dtype=tf.float32,
						name=w_key)
				elif tied_weights == True:
					mid_idx = (len(layer_shapes)-1)/ 2
					if i >=  mid_idx:
						tied_i = mid_idx - ((i - mid_idx)+1)
						w_key = 'w_%i'%i
						tied_w_key = 'w_%i'%tied_i
						weights_dict[w_key] = tf.transpose(weights_dict[tied_w_key], name=w_key)
					else:
						weights_dict[w_key] = tf.Variable(initializer_fn(w_shape), 
							dtype=tf.float32, name=w_key)
				b_key = 'b_%i'%i
				b_shape = layer_shapes[i+1]
				bias_dict[b_key] = tf.Variable(tf.zeros(b_shape), name=b_key)
		# create the layers
		with tf.name_scope('layers'):
			for i in range(len(weights_dict)):
				if i == 0:
					# first layer : input = X
					hidden = activation_fn(tf.matmul(X,weights_dict['w_%i'%i])+bias_dict['b_%i'%i])
				elif i > 0 and i < len(weights_dict)-1:
					# middle layers
					hidden = activation_fn(tf.matmul(hidden,weights_dict['w_%i'%i])+bias_dict['b_%i'%i])
				else:
					# final layer : no activation fn
					out = tf.matmul(hidden,weights_dict['w_%i'%i])+bias_dict['b_%i'%i]
		
		with tf.name_scope('loss'):
			model_loss 	= tf.reduce_mean(tf.square(out-X), name='loss')
			if tied_weights == False:
				reg_loss 	= [regularizer_fn(w) for _,w in weights_dict.items()]
			else:
				reg_loss 	= [regularizer_fn(w) for _,w in weights_dict.items()[:mid_idx]]
			total_loss 	= model_loss + reg_loss
			optimizer 	= tf.train.AdamOptimizer(learning_rate=lr)
			training_op = optimizer.minimize(total_loss)

		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)

			for epoch in range(nb_epoch):
				for b_idx in range(int(X_data.shape[0]/batch_size)):
					x_batch = X_data[b_idx*batch_size:(b_idx+1)*batch_size]
					# run training op
					sess.run(training_op, feed_dict={X:x_batch, is_training:True})
				# check loss after each epoch
				epoch_loss = sess.run(model_loss, feed_dict={X:X_data,
					is_training:False})
				print 'Epoch: ' , epoch, ' Loss: ', epoch_loss

			# get weights dict actual values
			for k in weights_dict:
				weights_dict[k] = sess.run(weights_dict[k])
			for k in bias_dict:
				bias_dict[k] = sess.run(bias_dict[k])

		self.weights_ 	= weights_dict
		self.biases_ 	= bias_dict



class slimAutoEncoder(object):

	def __init__(self, h_layers, lr=1e-2, l2_pen=1e-4,
			nb_epoch=100, batch_size=None, activation_fn='elu'):
		self.lr = lr
		self.nb_epoch = nb_epoch
		self.batch_size = batch_size
		self.l2_pen = l2_pen
		self.h_layers = h_layers
		if activation_fn == 'elu':
			self.activation_fn=tf.nn.elu
		elif activation_fn == 'relu':
			self.activation_fn = tf.nn.relu
		else:
			self.activation_fn = None

	def fit(self, X):
		h_layers = self.h_layers
		assert len(h_layers) >= 1, 'Must give at least one layer.'
		l2_pen = self.l2_pen
		nb_epoch  =self.nb_epoch
		X_data = X.reshape(X.shape[0],-1)
		n_inputs = X_data.shape[1]
		n_outputs = n_inputs
		if self.batch_size is None:
			self.batch_size = n_inputs
		batch_size = self.batch_size
		lr = self.lr

		# CONTSTRUCTION PHASE
		X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
		with arg_scope(
				[layers.fully_connected],
				activation_fn=self.activation_fn,
				weights_regularizer=layers.l2_regularizer(l2_pen),
				weights_initializer=layers.variance_scaling_initializer()):
			hidden = layers.stack(X, layers.fully_connected, h_layers, scope='hidden')
			out = layers.fully_connected(hidden, n_outputs, activation_fn=None, scope='out')

		model_loss 	= tf.reduce_mean(tf.square(out-X))
		reg_loss 	= tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		total_loss 	= model_loss + reg_loss

		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		training_op = optimizer.minimize(total_loss)

		init = tf.global_variables_initializer()

		#from tensorflow.examples.tutorials.mnist import input_data
		#mnist = input_data.read_data_sets("/tmp/data/")
		with tf.Session() as sess:
			sess.run(init)

			for epoch in range(nb_epoch):
				for b_idx in range(int(X_data.shape[0]/batch_size)):
					x_batch = X_data[b_idx*batch_size:(b_idx+1)*batch_size]
					# run training op
					sess.run(training_op, feed_dict={X:x_batch})
				# check loss after each epoch
				epoch_loss = sess.run(model_loss, feed_dict={X:X_data})
				print 'Epoch: ' , epoch, ' Loss: ', epoch_loss

			# get weights from fully_connected layer(s)
			weights = []
			for i in range(1,len(h_layers)+1):
				with tf.variable_scope('hidden/hidden_%i'%i, reuse=True):
					h_w = tf.get_variable('weights')
					h_w = sess.run(h_w)
					weights.append(h_w)
			with tf.variable_scope('out', reuse=True):
				o_w = tf.get_variable('weights')
				weights.append(sess.run(o_w))
		
		self.weights_ = weights


if __name__=='__main__':
	tf.reset_default_graph()
	from keras.datasets.mnist import load_data
	(xtrain,xtest),(ytrain,ytest) = load_data()
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	xtrain = MinMaxScaler((0,1)).fit_transform(xtrain.astype('float32').reshape(xtrain.shape[0],-1))
	#tfae = slimAutoEncoder(h_layers=[300,150,300], 
	#	batch_size=150 ,nb_epoch=5, lr=1e-2, l2_pen=1e-4, activation_fn='elu')
	#tfae.fit(xtrain)
	tfae = tfAutoEncoder(h_layers=[300,150,300], 
		batch_size=150 ,nb_epoch=5, lr=1e-2, l2_pen=1e-4, activation_fn='elu',
		tied_weights=False, denoise=False)
	tfae.fit(xtrain)
	#w,b = tfae.weights_, tfae.biases_






