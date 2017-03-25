"""
Code for reusing common tensorflow functionality
"""


def relu(X, n_out=1):
	with tf.name_scope('relu'):
		w_shape = (int(X.get_shape()[1]),n_out)
		w = tf.Variable(tf.random_normal(w_shape), name='weights')
		b = tf.Variable(0.0, name='bias')
		z = tf.add(tf.matmul(X,w),b, name='z')
		return tf.nn.relu(z, name='relu')

n_features = 3
X = tf.placeholder(tf.float32, shape=(None,n_features), name='X')

relus = [relu(X) for i in range(5)]
relu_addition = tf.add_n(relus, name='output')