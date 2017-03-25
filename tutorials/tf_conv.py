"""
Convolutional network in tensorflow, trained on MNIST
"""
import tensorflow as tf
from tensorflow.contrib import layers, framework

import numpy as np
import os

from keras.datasets.mnist import load_data
from keras.utils import np_utils

tf.reset_default_graph()

(xtrain,_ytrain),(xtest,_ytest) = load_data()
xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1)
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255

ytrain = np_utils.to_categorical(_ytrain, 10).astype('int32')
ytest = np_utils.to_categorical(_ytest, 10).astype('int32')

## CONTSTRUCTION PHASE

batch_size=32

# create placeholders for data
X = tf.placeholder(tf.float32,shape=(None,28,28,1),name='X')
y = tf.placeholder(tf.int32,shape=(None,10),name='y')

# create weight initializations
init_fn = layers.xavier_initializer()

# create model
with tf.variable_scope('layers'):
    # CONV 1
    with framework.arg_scope([layers.conv2d,layers.fully_connected],
        weights_initializer=init_fn):
        conv1   = layers.conv2d(X, 32, (3,3), 2, padding='VALID', scope='conv1')
        pool1   = layers.max_pool2d(conv1, (2,2), scope='pool1')
        # CONV 2
        conv2   = layers.conv2d(pool1, 32, (3,3), 2,padding='SAME', scope='conv2')
        pool2   = layers.max_pool2d(conv2, (2,2), scope='pool2')
        # FLATTEN
        flat_dim = np.prod(pool2.get_shape().as_list()[1:])
        flatten = tf.reshape(pool2, (-1,flat_dim))
        # FC 3
        fc3     = layers.fully_connected(flatten, 128, scope='fc3')
        # FC 4
        fc4     = layers.fully_connected(fc3, 10, scope='fc4')

# create loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc4, y))

# create evaluators
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(fc4, tf.argmax(y,1), 1) # requires a 1-D target
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# create training op
lr = 1e-3
with tf.variable_scope('train'):
    optimizer   = tf.train.AdamOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss)

nb_epoch    = 10
batch_size  = 32
init = tf.global_variables_initializer()

print 'Running Graph'
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(nb_epoch):
        for b_idx in range(int(xtrain.shape[0]/batch_size)):
            xbatch = xtrain[b_idx*batch_size:(b_idx+1)*batch_size]
            ybatch = ytrain[b_idx*batch_size:(b_idx+1)*batch_size]
            # run training op
            sess.run(training_op, feed_dict={X:xbatch,y:ybatch})
        # get test accuracy 
        acc = sess.run(accuracy, feed_dict={X:xtest,y:ytest})
        print 'Epoch : %i , Test Accuracy : %02f' % (epoch, acc)





