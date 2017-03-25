

import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers, framework

def to_categorical(y, nb_classes):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def process_inputs_3D(X,y, nb_classes):
    # expand y dims if necessary
    
    if X.ndim == 4:
        X = np.expand_dims(X,axis=-1)

    if np.argmin(X.shape[1:])==0:
        X = X.transpose(0,2,3,4,1)

    if y is not None:
        if y.ndim == 4:
            y = np.expand_dims(y,axis=-1)

        if np.argmin(y.shape[1:])==0:
            y = y.transpose(0,2,3,4,1)

        #nb_classes = len(np.unique(y))
        # convert y to one-hot representation for softmax classification
        y_onehot    = np.empty((y.shape[0],np.prod(y.shape[1:]),nb_classes))
        y_flat      = y.reshape(y.shape[0],-1)
        for i in range(y.shape[0]):
            y_onehot[i] = to_categorical(y_flat[i],nb_classes)
    else:
        y_onehot=None

    return X, y, y_onehot

def layers_conv3d(x, n_kernels, kernel_size, stride=1, scope=None):
    if type(kernel_size) is tuple or type(kernel_size) is list:
        kernel_size = kernel_size[0]
    # create variables
    if scope is None:
        scope = 'conv%i'%(1+(len([v for v in tf.global_variables() if 'conv' in v.name])/2))
    with tf.variable_scope(scope):
        W = tf.get_variable('weights', shape=(kernel_size,kernel_size,kernel_size,
            x.get_shape().as_list()[-1],n_kernels), initializer=layers.xavier_initializer())
        b = tf.get_variable('bias', shape=(n_kernels,),initializer=tf.zeros_initializer)
    # make operation
    out = tf.nn.conv3d(x, W, strides=[1,stride,stride,stride,1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return tf.nn.relu(out)


def layers_conv3d_transpose(x, n_kernels, kernel_size, stride=1, scope=None):
    if type(kernel_size) is tuple or type(kernel_size) is list:
        kernel_size = kernel_size[0]
    # create variables
    if scope is None:
        scope = 'conv%i'%(len([v for v in tf.global_variables() if 'conv' in v.name])/2)
    with tf.variable_scope(scope):
        W = tf.get_variable('weights', shape=(kernel_size,kernel_size,kernel_size,
            n_kernels,x.get_shape().as_list()[-1]), initializer=layers.xavier_initializer())
        b = tf.get_variable('bias', shape=(n_kernels,), initializer=tf.zeros_initializer)
    # calculate output shape
    output_shape = [tf.shape(x)[0], tf.shape(x)[1]*stride, tf.shape(x)[2]*stride, tf.shape(x)[2]*stride,tf.shape(W)[-2]]
    output_shape = tf.stack(list(output_shape))
    # make operation
    out = tf.nn.conv3d_transpose(x, W, strides=[1,stride,stride,stride,1], padding='SAME',
        output_shape=output_shape)
    out = tf.nn.bias_add(out,b)
    return tf.nn.relu(out)

class tfConvDeconv2D(object):
    
    def __init__(self,
                conv_layers,
                batch_size=25,
                nb_epoch=100,
                nb_classes=None,
                model_only=True):

        self.conv_layers    = conv_layers
        self.BATCH_SIZE     = batch_size
        self.NB_EPOCH       = nb_epoch
        self.nb_classes     = nb_classes
        self.model_only     = model_only
        self.LEARN_RATE     = 1e-3


    def fit(self, x, y):
        tf.reset_default_graph()
        x_orig,y_orig,y_onehot = process_inputs_3D(x,y,self.nb_classes)

        in_shape        = [None]+list(x_orig.shape[1:])
        orig_out_shape  = list(y_orig.shape[1:])
        soft_out_shape  = [None]+list(y_onehot.shape[1:])

        ### CONSTRUCTION PHASE ###
        X = tf.placeholder(tf.float32, shape=in_shape, name='X')
        y = tf.placeholder(tf.int32, shape=soft_out_shape, name='y')


        # CONV LAYERS #
        with tf.variable_scope('conv_layers'):
                for idx, c in enumerate(self.conv_layers):
                    if idx == 0:
                        # connect to input tensor
                        conv = layers_conv3d(X, c[0], (c[1],c[1]), stride=1)
                    else:
                        # connect to previous conv layer
                        conv = layers_conv3d(conv, c[0], (c[1],c[1]), stride=1)

        # DECONV LAYERS #
        with tf.variable_scope('deconv_layers'):
                for idx, c in enumerate(self.conv_layers[::-1]):
                    if idx < len(self.conv_layers)-1:
                        # not last layer
                        conv = layers_conv3d_transpose(conv, c[0], (c[1],c[1]), stride=1)
                    else:
                        # last layer
                        conv = layers_conv3d_transpose(conv, orig_out_shape[-1]*self.nb_classes, 
                            (c[1],c[1]), stride=1)

        # SOFTMAX RESHAPE LAYER
        with tf.variable_scope('softmax_layer'):
            soft_shape = [tf.shape(conv)[0], np.prod(orig_out_shape),self.nb_classes]
            softmax_reshape = tf.reshape(conv,soft_shape)

        ## LOSS ##
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(softmax_reshape, y))

        ## OPTIMIZER ##
        with tf.variable_scope('train'):
            optimizer   = tf.train.AdamOptimizer(learning_rate=self.LEARN_RATE)
            train_op    = optimizer.minimize(loss)

        ## EVALUATORS ##
        with tf.name_scope('eval'):
            prob_map    = tf.nn.softmax(softmax_reshape)
            soft_flat   = tf.reshape(softmax_reshape,[-1,self.nb_classes]) # (logits, classes)
            y_flat      = tf.reshape(tf.argmax(y,2),[-1]) # (classes,)
            correct     = tf.nn.in_top_k(soft_flat, y_flat, 1)
            accuracy    = tf.reduce_mean(tf.cast(correct, tf.float32))

        ### EXECUTION PHASE ###
        init = tf.global_variables_initializer()
        print('Running Graph')
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.NB_EPOCH):
                for b_idx in range(int(x_orig.shape[0]/self.BATCH_SIZE)):
                    print('batch')
                    xbatch = x_orig[b_idx*self.BATCH_SIZE:(b_idx+1)*self.BATCH_SIZE]
                    ybatch = y_onehot[b_idx*self.BATCH_SIZE:(b_idx+1)*self.BATCH_SIZE]
                    # run train op
                    sess.run(train_op, feed_dict={X:xbatch,y:ybatch})
                # get test statistics
                test_acc    = sess.run(accuracy, feed_dict={X:xbatch,y:ybatch})
                test_loss   = sess.run(loss, feed_dict={X:xbatch,y:ybatch})
                print('Epoch : %i , Test Loss : %.04f, Test Acc: %.04f' % (epoch, test_loss, test_acc))


if __name__=='__main__':
    x_orig = np.random.randn(50,20,20,20)
    y_orig = x_orig.copy()
    y_orig[y_orig>0] = 1
    #y_orig[(y_orig<0.5)&(y_orig>-0.5)] = 2
    y_orig[y_orig<0]=0
    y_orig[[1,2,3,4]] =2
    #y_orig[[11,22,31,41]] =3
    
    ## model & experiment hyperparameters
    NB_EPOCH    = 300 # number of epochs
    BATCH_SIZE  = 10 # batch size
    CONV_LAYERS = [(3,3),(3,3)]
    NB_CLASSES  = 3
    ## instantiate the network
    network = tfConvDeconv2D(conv_layers=CONV_LAYERS,
        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
        nb_classes=NB_CLASSES,model_only=True)

    network.fit(x_orig,y_orig)




