"""
ARCHITECTURE: 64 -> 25 -> Sigmoid -> 64
INIT: Xavier Uniform
Sparsity: KL Divergence w/ rho = 0.01 , beta = 3
LOSS: SSE
"""
from __future__ import division
import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, framework, opt

tf.reset_default_graph()

# create patches
x_orig = scipy.io.loadmat('IMAGES.mat')['IMAGES']
xtrain = np.zeros((10000,64))
for i in range(10000):
    i_idx = np.random.choice(10)
    x_idx = np.random.choice(500)
    y_idx = np.random.choice(500)
    patch = x_orig[x_idx:x_idx+8,y_idx:y_idx+8,i_idx]
    patch = patch.flatten()
    xtrain[i,:] = patch

# normalize data
xtrain = xtrain - np.mean(xtrain)
std_dev = 3 * np.std(xtrain)
xtrain = np.maximum(np.minimum(xtrain, std_dev), -std_dev) / std_dev
xtrain = (xtrain + 1) * 0.4 + 0.1
xtrain = xtrain.astype('float32')

# create variables
X = tf.Variable(xtrain, 'xtrain')

init_fn = layers.xavier_initializer()
weights = {
    'w0' : tf.Variable(init_fn((64,25)),name='w0'),
    'w1' : tf.Variable(init_fn((25,64)),name='w1')
}
biases = {
    'b0' : tf.Variable(tf.zeros(25),name='b0'),
    'b1' : tf.Variable(tf.zeros(64),name='b1')
}

# create hidden layers
with tf.name_scope('layers'):
    hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(X,weights['w0']),biases['b0']))
    output  = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, weights['w1']),biases['b1']))

def kl_divergence(p, q):
    return p * tf.log((p / q)+1e-5) + (1 - p) * tf.log(((1 - p) / (1 - q))+1e-5)

# create sparsity regularizers
with tf.name_scope('regs'):
    mean_act1   = tf.reduce_mean(hidden1, 0)
    sparsity1   = 3.0 * tf.reduce_sum(kl_divergence(mean_act1,0.01))
    decay1      = 1e-3 * tf.reduce_sum(tf.square(weights['w0']))
    decay2      = 1e-3 * tf.reduce_sum(tf.square(weights['w1']))

# create loss
with tf.name_scope('loss'):
    sse = tf.reduce_sum(tf.square(output-X))
    total_loss = sse + sparsity1 + decay1 + decay2

# create train ops
with tf.name_scope('train'):
    optimizer = opt.ScipyOptimizerInterface(total_loss, 
        method='L-BFGS-B', options={'maxiter': 10000})

# create initializer
init = tf.global_variables_initializer()

print 'Running Optimization..'
with tf.Session() as sess:
    sess.run(init)
    optimizer.minimize(sess)
    # turn tensors into numpy arrays
    for k in weights:
        weights[k] = sess.run(weights[k])
    for k in biases:
        biases[k] = sess.run(biases[k])
"""
w = weights['w0']
import matplotlib.pyplot as plt
# run: %matplotlib inline from jupyter console
for i in range(w.shape[1]):
    plt.imshow(w[:,i].reshape(8,8),cmap='gray')
    plt.show()
"""



