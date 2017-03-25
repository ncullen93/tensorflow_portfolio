

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

def process_inputs_2D(X,y, nb_classes=None):
    # expand y dims if necessary
    
    if X is not None:
        if X.ndim == 3:
            X = np.expand_dims(X,axis=-1)

        if np.argmin(X.shape[1:])==0:
            X = X.transpose(0,2,3,1)

    if y is not None:
        if y.ndim == 3:
            y = np.expand_dims(y,axis=-1)

        if np.argmin(y.shape[1:])==0:
            y = y.transpose(0,2,3,1)
        
        if nb_classes is None:
            nb_classes = len(np.unique(y))
        # convert y to one-hot representation for softmax classification
        y_onehot    = np.empty((y.shape[0],np.prod(y.shape[1:]),nb_classes))
        y_flat      = y.reshape(y.shape[0],-1)
        for i in range(y.shape[0]):
            y_onehot[i] = to_categorical(y_flat[i],nb_classes)
    else:
        y_onehot=None

    return X, y, y_onehot


class tfConvDeconv2D(object):
    
    def __init__(self,
                conv_layers,
                batch_size=25,
                nb_epoch=100,
                nb_classes=None,
                smoothness=0.2,
                sparsity=0.2,
                learn_rate=1e-3,
                save_path=None,
                restore_path=None):
        """
        Initialize a Convolution-Deconvolution Network


        Arguments 
        ---------
        conv_layers : list of tuples
            model architecture [(n_kernels, kernel_size),...]

        batch_size  : integer
            number of samples to use per train iteration

        nb_epoch    : integer
            number of training epochs

        nb_classes  : integer
            number of possible classes in the target segmentation

        smoothness  : float
            L2 smoothness penalty on all weights

        sparsity    : float
            L1 sparsity penalty on all weights

        learn_rate  : float
            learning rate for the gradient descent routine

        save_path   : string
            file path to save model checkpoints. should end in .ckpt

        restore_path: string
            file path to restore a previously saved model. should end in .ckpt
        """

        self.CONV_LAYERS    = conv_layers
        self.BATCH_SIZE     = batch_size
        self.NB_EPOCH       = nb_epoch
        self.NB_CLASSES     = nb_classes
        self.LEARN_RATE     = learn_rate
        self.SAVE_PATH      = save_path
        self.RESTORE_PATH   = restore_path
        # regularizers
        self.L1_PENALTY     = sparsity
        self.L2_PENALTY     = smoothness


    def fit(self, x, y):
        """
        Fit a ConvDeconv2D model on data

        Arguments
        ---------
        x : np.ndarray
            array with 3 dimensions (nb_samples, height, width) or 
            array with 4 dimensions (nb_samples, height, width, channels)

        y : np.ndarray
            array with 3 dimensions (nb_samples, height, width) or 
            array with 4 dimensions (nb_samples, height, width, channels)
        """ 
        tf.reset_default_graph()
        x_orig,y_orig,y_onehot = process_inputs_2D(x,y,self.NB_CLASSES)

        in_shape        = [None]+list(x_orig.shape[1:])
        orig_out_shape  = list(y_orig.shape[1:])
        soft_out_shape  = [None]+list(y_onehot.shape[1:])

        ### CONSTRUCTION PHASE ###
        X = tf.placeholder(tf.float32, shape=in_shape, name='X')
        y = tf.placeholder(tf.int32, shape=soft_out_shape, name='y')


        # CONV LAYERS #
        with tf.variable_scope('conv_layers'):
            with framework.arg_scope([layers.conv2d],
                                    weights_initializer=layers.xavier_initializer(),
                                    weights_regularizer=layers.l1_l2_regularizer(\
                                        scale_l1=self.L1_PENALTY,scale_l2=self.L2_PENALTY),
                                    activation_fn=tf.nn.relu,
                                    padding='SAME'):
                for idx, c in enumerate(self.CONV_LAYERS):
                    if idx == 0:
                        # connect to input tensor
                        conv = layers.conv2d(X, c[0], (c[1],c[1]), stride=1)
                    else:
                        # connect to previous conv layer
                        conv = layers.conv2d(conv, c[0], (c[1],c[1]), stride=1)

        # DECONV LAYERS #
        with tf.variable_scope('deconv_layers'):
            with framework.arg_scope([layers.conv2d_transpose],
                                    weights_initializer=layers.xavier_initializer(),
                                    activation_fn=tf.nn.relu,
                                    weights_regularizer=layers.l1_l2_regularizer(\
                                        scale_l1=self.L1_PENALTY,scale_l2=self.L2_PENALTY),
                                    padding='SAME'):
                for idx, c in enumerate(self.CONV_LAYERS[::-1]):
                    if idx < len(self.CONV_LAYERS)-1:
                        # not last layer
                        conv = layers.conv2d_transpose(conv, c[0], (c[1],c[1]), stride=1)
                    else:
                        # last layer
                        conv = layers.conv2d_transpose(conv, orig_out_shape[-1]*self.NB_CLASSES, 
                            (c[1],c[1]), stride=1)

        # SOFTMAX RESHAPE LAYER #
        with tf.variable_scope('softmax_layer'):
            soft_shape = [tf.shape(conv)[0], np.prod(orig_out_shape),self.NB_CLASSES]
            softmax_reshape = tf.reshape(conv,soft_shape)

        # LOSS #
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(softmax_reshape, y))

        # OPTIMIZER #
        with tf.variable_scope('train'):
            optimizer   = tf.train.AdamOptimizer(learning_rate=self.LEARN_RATE)
            train_op    = optimizer.minimize(loss)

        # EVALUATORS #

        with tf.name_scope('eval'):
            prob_map    = tf.nn.softmax(softmax_reshape)
            soft_flat   = tf.reshape(softmax_reshape,[-1,self.NB_CLASSES]) # (logits, classes)
            y_flat      = tf.reshape(tf.argmax(y,2),[-1]) # (classes,)
            correct     = tf.nn.in_top_k(soft_flat, y_flat, 1)
            accuracy    = tf.reduce_mean(tf.cast(correct, tf.float32))

        ### EXECUTION PHASE ###

        # PRE-EXECUTION VARIABLES #
        if self.SAVE_PATH is not None:
            saver = tf.train.Saver()
        best_test_loss = 1e9
        init = tf.global_variables_initializer()

        # TRAINING ROUTINE #
        with tf.Session() as sess:
            
            if self.RESTORE_PATH is not None:
                print('Restoring Model')
                saver.restore(sess, self.RESTORE_PATH)
            else:
                print('Initializing Model')
                sess.run(init)

            for epoch in range(self.NB_EPOCH):
                for b_idx in range(int(x_orig.shape[0]/self.BATCH_SIZE)):
                    xbatch = x_orig[b_idx*self.BATCH_SIZE:(b_idx+1)*self.BATCH_SIZE]
                    ybatch = y_onehot[b_idx*self.BATCH_SIZE:(b_idx+1)*self.BATCH_SIZE]
                    # run train op
                    sess.run(train_op, feed_dict={X:xbatch,y:ybatch})
                # get test statistics
                test_acc    = sess.run(accuracy, feed_dict={X:xbatch,y:ybatch})
                test_loss   = sess.run(loss, feed_dict={X:xbatch,y:ybatch})
                print('Epoch : %i , Test Loss : %.04f, Test Acc: %.04f' % (epoch, test_loss, test_acc))

                # save model after each epoch if test loss is the best
                if self.SAVE_PATH is not None and test_loss < best_test_loss:
                    best_test_loss = test_loss
                    saver.save(sess, self.SAVE_PATH, write_meta_graph=False)


if __name__=='__main__':
    x_orig = np.random.randn(50,20,20)
    y_orig = x_orig.copy()
    y_orig[y_orig>0] = 1
    #y_orig[(y_orig<0.5)&(y_orig>-0.5)] = 2
    y_orig[y_orig<0]=0
    
    ## model & experiment hyperparameters
    NB_EPOCH            = 100 # number of epochs
    BATCH_SIZE          = 30 # batch size
    CONV_LAYERS         = [(3,3),(3,3)]
    NB_CLASSES          = 2
    SAVE_PATH       = '/users/nick/desktop/ckpt/model.ckpt'
    #SAVE_PATH      = None
    #RESTORE_PATH   = '/users/nick/desktop/ckpt/model.ckpt'
    RESTORE_PATH = None
    ## instantiate the network
    network = tfConvDeconv2D(conv_layers=CONV_LAYERS,
        batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
        nb_classes=NB_CLASSES, save_path=SAVE_PATH, 
        restore_path=RESTORE_PATH)

    network.fit(x_orig,y_orig)
    
    #network.load_from_ckpt('/users/nick/desktop')



