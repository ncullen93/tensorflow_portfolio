
# AUTOMATIC USAGE IN AN OPTIMIZER
# note global_step = epoch .. so change this if you are loading a half-trained model
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 1/10
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps, decay_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
training_op = optimizer.minimize(loss, global_step=global_step)


# passing global_step as a placeholder

initial_learning_rate = 0.1
decay_steps = 200
decay_rate = 1e-2
global_step = tf.placeholder(tf.int32,shape=(),name='global_step')
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps, decay_rate)
init = tf.global_variables_initializer()
vals = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(decay_steps):
        vals.append(sess.run(learning_rate, feed_dict={global_step:epoch}))

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(vals)
plt.ylim([0,initial_learning_rate])
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')