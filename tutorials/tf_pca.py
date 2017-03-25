"""
Autoencoders w/ tensorflow
"""
from __future__ import division

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


## PCA = AE w/ Linear Activation and MSE loss##

class tfPCA(object):

    def __init__(self, n_components=3, lr=1e-2, nb_epoch=100, batch_size=None):
        self.n_hidden = n_components
        self.lr = lr
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

    def fit(self, X):
        n_hidden = self.n_hidden
        nb_epoch  =self.nb_epoch

        X_data = X.reshape(X.shape[0],-1)

        n_inputs = X_data.shape[1]
        n_outputs = n_inputs

        if self.batch_size is None:
            self.batch_size = n_inputs
        batch_size = self.batch_size

        X = tf.placeholder(tf.float32, shape=(None, n_inputs))
        hidden = fully_connected(X, n_hidden, activation_fn=None, scope='hidden')
        outputs = fully_connected(hidden, n_outputs, activation_fn=None, scope='outputs')

        loss = tf.reduce_mean(tf.square(outputs-X))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        print 'Training model..'
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(nb_epoch):
                for b_idx in range(int(X_data.shape[0]/batch_size)):
                    x_batch = X_data[b_idx*batch_size:(b_idx+1)*batch_size]
                    # run training op
                    sess.run(training_op, feed_dict={X:x_batch})
                # check loss after each epoch
                epoch_loss = sess.run(loss, feed_dict={X:X_data})
                print 'Epoch: ' , epoch, ' Loss: ', epoch_loss

            # get weights from fully_connected layer(s)
            with tf.variable_scope('hidden',reuse=True):
                h_w = tf.get_variable('weights')
                x= sess.run(h_w)
        return x

if __name__=='__main__':
    tf.reset_default_graph()
    from keras.datasets.mnist import load_data
    (xtrain,xtest),(ytrain,ytest) = load_data()
    from sklearn.preprocessing import StandardScaler
    xtrain = StandardScaler().fit_transform(xtrain.reshape(xtrain.shape[0],-1))
    tfpca = tfPCA(n_components=10, nb_epoch=40, lr=5e-4)
    x= tfpca.fit(xtrain)










