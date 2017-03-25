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
y_onehot    = np.empty((y_orig.shape[0],np.prod(y_orig.shape[1:]),nb_classes))
y_flat      = y_orig.reshape(y_orig.shape[0],-1)
for i in range(y_orig.shape[0]):
    y_onehot[i] = to_categorical(y_flat[i],nb_classes)


### CONSTRUCTION PHASE ###
# input/output shapes
input_shape     = [None]+list(x_orig.shape[1:])
target_shape    = list(y_orig.shape[1:])
output_shape    = [None]+list(y_onehot.shape[1:])

# hyper-parameters
INIT_FN     = layers.xavier_initializer()
STRIDE      = 1
LEARN_RATE  = 1e-3
NB_EPOCH    = 100
BATCH_SIZE  = 10


# input/target placeholders
X = tf.placeholder(tf.float32,shape=input_shape,name='X')
y = tf.placeholder(tf.uint8,shape=output_shape,name='Y')

"""

tf.nn.conv2d(input, filter, strides, padding, name=None)
    -> input shape  = [batch, in_height, in_width, in_channels]
    -> kernel shape = [width, weight, out_channels, in_channels]
    -> strides = [1, s, s, 1]
tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
    -> input shape  = [batch, height, width, in_channels]
    -> kernel shape = [width, weight, in_channels, out_channels]
    -> strides = [1, s, s, 1]
"""
#tf.reset_default_graph()

def resize_image(x, image_shape):
    up_x = tf.image.resize_images(x, [image_shape[0], image_shape[1]], 
        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(up_x,tf.float32)

def conv2d_layer(x, n_kernels, kernel_size, stride=1, scope=None):
    # create variables
    if scope is None:
        scope = 'conv%i'%(1+(len([v for v in tf.global_variables() if 'conv' in v.name])/2))
    with tf.variable_scope(scope):
        W = tf.get_variable('weights', shape=(kernel_size,kernel_size,x.get_shape().as_list()[-1],n_kernels),
             initializer=layers.xavier_initializer())
        b = tf.get_variable('bias', shape=(n_kernels,),initializer=tf.zeros_initializer)
    # make operation
    out = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return tf.nn.relu(out)


def deconv2d_layer(x, n_kernels, kernel_size, stride=1, scope=None):
    # create variables
    if scope is None:
        scope = 'conv%i'%(1+(len([v for v in tf.global_variables() if 'conv' in v.name])/2))
    with tf.variable_scope(scope):
        W = tf.get_variable('weights', shape=(kernel_size,kernel_size,n_kernels,x.get_shape().as_list()[-1]),
            initializer=layers.xavier_initializer())
        b = tf.get_variable('bias', shape=(n_kernels,), initializer=tf.zeros_initializer)
    # calculate output shape
    output_shape = [tf.shape(x)[0], tf.shape(x)[1]*stride, tf.shape(x)[2]*stride, tf.shape(W)[-2]]
    output_shape = tf.stack(list(output_shape))
    # make operation
    out = tf.nn.conv2d_transpose(x, W, strides=[1,stride,stride,1], padding='SAME',
        output_shape=output_shape)
    out = tf.nn.bias_add(out,b)
    return tf.nn.relu(out)

def lrelu(x, leak=0.2, scope=None):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    out = f1 * x + f2 * tf.abs(x)
    return out

### CREATION PHASE ###
## MODEL ##

## CONV LAYERS ##
with tf.variable_scope('conv_layers'):
    conv1 = conv2d_layer(X, n_kernels=3, kernel_size=5, stride=1, scope='conv1')
    conv2 = conv2d_layer(conv1, n_kernels=5, kernel_size=5, stride=1, scope='conv2')
    conv3 = conv2d_layer(conv2, n_kernels=5, kernel_size=5, stride=1, scope='conv3')

## DECONV LAYERS ##
with tf.variable_scope('deconv_layers'):
    deconv1 = deconv2d_layer(conv3, n_kernels=3, kernel_size=5, stride=1,scope='deconv0')
    deconv2 = deconv2d_layer(deconv1, n_kernels=1, kernel_size=5, stride=1,scope='deconv1')
    deconv3 = deconv2d_layer(deconv2, n_kernels=1, kernel_size=5, stride=1,scope='deconv2')

### EXECUTION PHASE ###
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    c = sess.run(deconv3,feed_dict={X:x_orig})
    print(c.shape)










