"""
feedforward neural network
"""

import numpy as np

import tensorflow as tf
tf.reset_default_graph()


X = tf.placeholder(tf.float32, shape=(None, 28*28), name='X')
y = tf.placeholder(tf.int64, shape=(None,10), name='y')

## USING A CUSTOM FULLY CONNECTED LAYER
"""
def my_fully_connected(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)

        W = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X,w) + b
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope('dnn'):
    hidden1 = my_fully_connected(X, 300, 'hidden1', activation='relu')
    hidden2 = my_fully_connected(hidden1, 100, 'hidden2', activation='relu')
    logits = my_fully_connected(hidden2, 10, 'outputs')
"""

## USING TF.CONTRIB.LAYERS FULLY CONNECTED LAYER
from tensorflow.contrib.layers import fully_connected

##########################
### CONTRSUCTION PHASE ###
##########################
with tf.name_scope('dnn'):
    hidden1 = fully_connected(X, 300, scope='hidden1')
    hidden2 = fully_connected(hidden1, 100, scope='hidden2')
    logits  = fully_connected(hidden2, 10, scope='outputs', activation_fn=None)

with tf.name_scope('loss'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
    loss = tf.reduce_mean(xentropy, name='loss')

lr = 1e-3
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, tf.argmax(y,axis=1), 1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))



##########################
### EVALUATION PHASE ###
##########################
from keras.datasets.mnist import load_data
(xtrain,ytrain),(xtest,ytest) = load_data()

from sklearn.preprocessing import MinMaxScaler
xtrain = MinMaxScaler((0,1)).fit_transform(xtrain.reshape(-1,28*28).astype('float32'))
xtest = MinMaxScaler((0,1)).fit_transform(xtest.reshape(-1,28*28).astype('float32'))

from keras.utils.np_utils import to_categorical
ytrain  = to_categorical(ytrain,10)
ytest   = to_categorical(ytest, 10)


nb_epoch = 400
batch_size = 50

# intiializer
init = tf.global_variables_initializer()
# saver
saver = tf.train.Saver()

train = False
if train:
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(nb_epoch):
            for batch_idx in range(int(np.ceil(xtrain.shape[0])/batch_size)):
                x_batch = xtrain[batch_idx*batch_size:(batch_idx+1)*batch_size]
                y_batch = ytrain[batch_idx*batch_size:(batch_idx+1)*batch_size]

                sess.run(training_op, feed_dict={X:x_batch,y:y_batch})

            acc_train = sess.run(accuracy, feed_dict = {X:x_batch,y:y_batch})
            acc_test = sess.run(accuracy, feed_dict={X:xtest,y:ytest})

            print 'Epoch: %i, Train Acc: %.02f , Test Acc: %.02f' % (epoch,acc_train,acc_test)

            save_path = saver.save(sess, '/users/nick/desktop/projects/tf_examples/ckpts/mnist_ff.ckpt')



### RESTORING A MODEL AND USING IT TO EVALUATE NEW DATA
with tf.Session() as sess:
    saver.restore(sess,'/users/nick/desktop/projects/tf_examples/ckpts/mnist_ff.ckpt' )
    test_acc = sess.run(accuracy, feed_dict={X:xtest,y:ytest})
    print 'Test Accuracy: ' , test_acc








