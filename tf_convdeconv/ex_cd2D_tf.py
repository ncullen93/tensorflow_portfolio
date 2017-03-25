"""
Conv-Deconv-2D network in tensorflow

Considerations:
	- input shape
	- number of layers
	- number of kernels per layer
	- kernel size per layer
	- convolutional stride
		- note: stride=2 cuts image size in half in all dimensions
	- batch norm & dropout
	- softmax output

tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
	-> input shape 	= [batch, height, width, in_channels]
	-> kernel shape = [width, weight, in_channels, out_channels]
	-> strides = [1, s, s, 1]
tf.nn.conv2d(input, filter, strides, padding, name=None)
	-> input shape 	= [batch, in_height, in_width, in_channels]
	-> kernel shape = [width, weight, out_channels, in_channels]
	-> strides = [1, s, s, 1]
python /Users/nick/.local/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=test_logs

"""
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers, framework
tf.reset_default_graph()

def to_categorical(y, nb_classes):
	y = np.array(y, dtype='int').ravel()
	n = y.shape[0]
	categorical = np.zeros((n, nb_classes))
	categorical[np.arange(n), y] = 1
	return categorical

### DATA LOADING PHASE ###
x_orig = np.random.randn(50,20,20,1)
y_orig = x_orig.copy()
y_orig[y_orig<0] = 0
y_orig[y_orig>0] = 1
nb_classes = 2
# convert y to one-hot representation for softmax classification
y_onehot 	= np.empty((y_orig.shape[0],np.prod(y_orig.shape[1:]),nb_classes))
y_flat 		= y_orig.reshape(y_orig.shape[0],-1)
for i in range(y_orig.shape[0]):
	y_onehot[i] = to_categorical(y_flat[i],nb_classes)


### CONSTRUCTION PHASE ###
# input/output shapes
in_shape 		= [None]+list(x_orig.shape[1:])
orig_out_shape 	= list(y_orig.shape[1:])
soft_out_shape 	= [None]+list(y_onehot.shape[1:])

# hyper-parameters
INIT_FN 	= layers.xavier_initializer()
STRIDE 		= 1
LEARN_RATE 	= 1e-3
NB_EPOCH 	= 100
BATCH_SIZE	= 10

# input/target placeholders
X = tf.placeholder(tf.float32,shape=in_shape,name='X')
y = tf.placeholder(tf.int32,shape=soft_out_shape,name='Y')

## MODEL ##
# CONV LAYERS #
with tf.variable_scope('conv_layers'):
	with framework.arg_scope([layers.conv2d],
							weights_initializer=INIT_FN,
							padding='SAME',
							stride=STRIDE):
		conv1 	= layers.conv2d(X, 5, (3,3), scope='conv1')
		conv2 	= layers.conv2d(conv1, 3, (3,3), scope='conv2')

# DECONV LAYERS #
with tf.variable_scope('deconv_layers'):
	with framework.arg_scope([layers.conv2d_transpose],
							weights_initializer=INIT_FN,
							padding='SAME',
							stride=STRIDE):
		deconv1 = layers.conv2d_transpose(conv2, 3, (3,3), scope='deconv1')
		deconv2 = layers.conv2d_transpose(deconv1, 2, (3,3), scope='deconv2')

with tf.variable_scope('softmax_layer'):
	soft_shape = [tf.shape(deconv2)[0], np.prod(orig_out_shape), nb_classes]
	softmax_reshape = tf.reshape(deconv2,soft_shape)

## LOSS ##
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(softmax_reshape, y))

## OPTIMIZER ##
with tf.variable_scope('train'):
	optimizer 	= tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
	train_op 	= optimizer.minimize(loss)

## EVALUATORS ##
with tf.name_scope('eval'):
	prob_map 	= tf.nn.softmax(softmax_reshape)
	soft_flat 	= tf.reshape(softmax_reshape,[-1,nb_classes]) # (logits, classes)
	y_flat 		= tf.reshape(tf.argmax(y,2),[-1]) # (classes,)
	correct   	= tf.nn.in_top_k(soft_flat,y_flat, 1)
	accuracy 	= tf.reduce_mean(tf.cast(correct, tf.float32))


### EXECUTION PHASE ###
init = tf.global_variables_initializer()
print('Running Graph')
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(NB_EPOCH):
		for b_idx in range(int(x_orig.shape[0]/BATCH_SIZE)):
			xbatch = x_orig[b_idx*BATCH_SIZE:(b_idx+1)*BATCH_SIZE]
			ybatch = y_onehot[b_idx*BATCH_SIZE:(b_idx+1)*BATCH_SIZE]
			# run train op
			sess.run(train_op, feed_dict={X:xbatch,y:ybatch})
		# get test statistics
		test_acc 	= sess.run(accuracy, feed_dict={X:xbatch,y:ybatch})
		test_loss 	= sess.run(loss, feed_dict={X:xbatch,y:ybatch})
		print('Epoch : %i , Test Loss : %.04f, Test Acc: %.04f' % (epoch, test_loss, test_acc))


