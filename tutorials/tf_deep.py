
import numpy as np

import tensorflow as tf
tf.reset_default_graph()


def xavier_variable(n_in, n_out):
	lim = np.sqrt(2)*np.sqrt(6/float(n_in+n_out))
	return tf.Variable(tf.random_uniform(shape=(n_in,n_out), 
		minval=-lim, maxval=lim))

def leaky_relu(x, name=None):
	return tf.maximum(0.01*x, x, name=name)

from keras.datasets.mnist import load_data
(xtrain,ytrain),(xtest,ytest) = load_data()

from sklearn.preprocessing import MinMaxScaler
xtrain = MinMaxScaler((0,1)).fit_transform(xtrain.reshape(-1,28*28).astype('float32'))
xtest = MinMaxScaler((0,1)).fit_transform(xtest.reshape(-1,28*28).astype('float32'))

from keras.utils.np_utils import to_categorical
ytrain 	= to_categorical(ytrain,10)
ytest 	= to_categorical(ytest, 10)


from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.framework import arg_scope

# CONSTRUCTION PHASE #
X = tf.placeholder(tf.float32,shape=(None,784),name='X')
y = tf.placeholder(tf.float32,shape=(None,10),name='Y')
is_training = tf.placeholder(tf.bool, shape=(),name='is_training')

bn_params = {
    'is_training': is_training,
    'decay': 0.99,
    'updates_collections': None
}

with tf.name_scope('model'):
	with arg_scope(
			[fully_connected],
			normalizer_fn=batch_norm,
			normalizer_params=bn_params):
		hidden1 = fully_connected(X, 300, scope='hidden1')
		hidden2 = fully_connected(hidden1, 100, scope='hidden2')
		logits 	= fully_connected(hidden2, 10, scope='outputs',
			activation_fn=None)

with tf.name_scope('loss'):
	xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
	loss = tf.reduce_mean(xentropy, name='loss')

lr=1e-3
with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, tf.argmax(y,axis=1), 1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))


# EVALUATION PHASE #
batch_size = 32
nb_epoch = 20
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	print 'Begin Training:'
	for epoch in range(nb_epoch):
		for batch_idx in range(int(xtrain.shape[0]/batch_size)):
			# get batch
			xbatch = xtrain[batch_idx*batch_size:(batch_idx+1)*batch_size]
			ybatch = ytrain[batch_idx*batch_size:(batch_idx+1)*batch_size]

			# run training ops
			sess.run(training_op,
				feed_dict={is_training:True, X:xbatch, y:ybatch})

		# evaluate accuracy on test set
		test_acc = sess.run(accuracy,
			feed_dict={is_training:False, X:xtest,y:ytest})
		print 'Epoch: ', epoch, ' Test Acc: ' , test_acc

#########################
### GRADIENT CLIPPING ###
#########################
## basically, the optimizer.minimize fn is performing 'compute_gradients', then 'apply_gradients'
# create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
# compute gradients on loss -> returns list of (grad,variable) pairs
grads_and_vars = optimizer.compute_gradients(loss)
# now clip the gradients
thresh=1.0
clipped_grads_and_vars = [(tf.clip_by_value(g,-thresh,thresh),v) for g,v in grads_and_vars]
# make the training op ('apply_gradients' here instead of 'minimize')
training_op = optimizer.apply_gradients(capped_gvs)









