
import tensorflow as tf
import numpy as np

import os
from datetime import datetime


class tfLinearRegression(object):

	def __init__(self):
		pass

	def fit(self, X, y):
		X_data = X
		y_data = y 
		n_samples 	= X_data.shape[0]
		n_features 	= X_data.shape[1]
		n_targets 	= y_data.shape[1]

		X = tf.placeholder(tf.float32, shape=[n_samples,n_features])
		y = tf.placeholder(tf.float32, shape=[n_samples,n_targets])

		XT = tf.transpose(X)
		
		theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

		mse = tf.reduce_mean(tf.square(tf.matmul(X, theta) - y))

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			final_theta= sess.run(theta,
				feed_dict={X: X_data, y: y_data})

			final_mse = sess.run(mse, feed_dict={X:X_data,y:y_data})
		print 'MSE: ' , final_mse
		self.params_ = final_theta


class tfSGDLinearRegression(object):

	def __init__(self, lr=1e-4, nb_epoch=100, 
		batch_size=None, opt='sgd', save_ckpt=False,
		restore=None, tensorboard=False):
		self.lr = lr
		self.nb_epoch = nb_epoch
		self.batch_size = batch_size

		assert opt in {'manual', 'sgd', 'momentum'}, 'Opt not valid'
		self.opt = opt

		self.save_ckpt = save_ckpt
		self.ckpt_dir = '/users/nick/desktop/projects/tf_examples/ckpts/'
		self.restore  = restore
		if self.restore is not None:
			self.save_ckpt = True

		self.tensorboard = tensorboard
		if self.tensorboard == True:
			now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
			root_logdir = '/users/nick/desktop/projects/tf_examples/logs'
			self.logdir = "{}/run-{}/".format(root_logdir, now)


	def fit(self, X, y):
		X_data = X
		y_data = y 
		n_samples 	= X_data.shape[0]
		n_features 	= X_data.shape[1]
		n_targets 	= y_data.shape[1]
		
		if self.batch_size is None:
			self.batch_size = n_samples
		
		n_batches = int(np.ceil(n_samples / batch_size))
		print 'Num Batches: ', n_batches

		with tf.name_scope('inputs') as scope:
			X = tf.placeholder(tf.float32, shape=[None,n_features], name='X')
			y = tf.placeholder(tf.float32, shape=[None,n_targets], name='y')

		
		XT = tf.transpose(X)
		theta = tf.Variable(tf.random_uniform([n_features,n_targets],-1.0,1.0),
			name='theta')
		y_pred 	= tf.matmul(X, theta, name='y_pred')

		with tf.name_scope('loss') as scope:
			error 	= y_pred - y
			cost 	= tf.reduce_mean(tf.square(error), name='cost')

		## using autodiff and manual training ops
		if self.opt == 'manual':
			gradients = tf.gradients(cost, [theta])[0]
			training_op = tf.assign(theta, theta - self.lr * gradients)
		elif self.opt == 'sgd':
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
			training_op = optimizer.minimize(cost)
		elif self.opt == 'momentum':
			optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
				momentum=0.9)
			training_op = optimizer.minimize(cost)

		init = tf.global_variables_initializer()
		if self.save_ckpt:
			saver = tf.train.Saver()
		if self.tensorboard:
			mse_summary = tf.summary.scalar('Cost', cost)
			summary_writer = tf.summary.FileWriter(self.logdir,tf.get_default_graph())

		with tf.Session() as sess:
			if self.restore is not None:
				try:
					saver.restore(sess, self.restore)
					print 'Loaded Checkpoint!'
				except:
					print 'Couldnt load checkpoint.. Starting from scratch'
					sess.run(init)
			else:
				sess.run(init)

			for epoch in range(self.nb_epoch):
				for batch_idx in range(n_batches):
					# create mini batch
					X_batch = X_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
					y_batch = y_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
					
					# run training op
					sess.run(training_op,
						feed_dict={X:X_batch,y:y_batch})

					if batch_idx % 100 == 0:
						if self.tensorboard:
							summary_str = sess.run(mse_summary,
								feed_dict={X:X_batch,y:y_batch})
							step = epoch * n_batches + batch_idx
							summary_writer.add_summary(summary_str, step)

				if epoch % 5 == 0 or epoch==(self.nb_epoch-1):
					mse_temp = sess.run(cost, feed_dict={X:X_data,y:y_data})
					print 'Epoch: ' , epoch , ' MSE: ' , mse_temp
					if self.save_ckpt:
						save_path = saver.save(sess, 
							os.path.join(self.ckpt_dir,'sgd_lm_intermediate.ckpt'))
						print 'Saved model'
					
			# dont have to pass in a feed_dict because theta has no dependencies on X or y
			final_theta = sess.run(theta)
			# save final model
			if self.save_ckpt:
				save_path = saver.save(sess,os.path.join(self.ckpt_dir,'sgd_lm_final.ckpt'))

			summary_writer.close()

		self.params_ = final_theta


if __name__ == '__main__':
	import numpy as np
	from sklearn.datasets import fetch_california_housing
	tf.reset_default_graph()

	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

	X = housing_data_plus_bias
	y = housing.target.reshape(-1, 1)

	### EXACT LINEAR REGRESSION ###
	print 'Running Exact Linear Regression:' 
	lm = tfLinearRegression()
	lm.fit(X,y)
	print 'Fit Params:\n', lm.params_ 

	### GRADIENT DESCENT LINEAR REGRESSION ###
	print 'Running Gradient Descent Linear Regression:'
	tf.reset_default_graph()
	lm = tfSGDLinearRegression(opt='momentum', lr=1e-5, nb_epoch=20,
		batch_size=64, tensorboard=True)
		#save_ckpt=True, 
		#restore='/users/nick/desktop/projects/tf_examples/ckpts/sgd_lm_final.ckpt')
	from sklearn.preprocessing import StandardScaler
	X = StandardScaler().fit_transform(X)
	y = StandardScaler().fit_transform(y)
	lm.fit(X,y)
	print 'Fit Params:\n', lm.params_

	## run tensorboard
	"""
	python /Users/nick/.local/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py 
	--logdir /users/nick/desktop/projects/tf_examples/logs/

	Navigate to http://130.91.249.177:6006
	"""





