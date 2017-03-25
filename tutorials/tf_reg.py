# regularizers

# MANUAL IMPLEMENTATION
base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
reg_losses = tf.reduce_sum(tf.abs(weights1)) + tf.reduce_sum(tf.abs(weights2))
loss = tf.add(base_loss, scale * reg_losses, name="loss")

# FUNCTION IMPLEMENTATION
base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
l1_fn = t1.contrib.layers.l1_regularizer(scale=0.01)
reg_losses = l1_fn(weights1) + l1_fn(weights_2)
loss = tf.add(base_loss, reg_losses, name="loss")

# add_n implementation
base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
l1_fn = t1.contrib.layers.l1_regularizer(scale=0.01)
reg_loss_list = [l1_fn(w) for w in weights_dict]
total_reg_loss = tf.add_n(reg_loss_list)
loss = tf.add(base_loss, total_reg_loss, name="loss")

# WITH ARG SCOPE 
with arg_scope(
		[fully_connected],
		weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):
	hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
	hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
	logits = fully_connected(hidden2, n_outputs, activation_fn=None,scope="out")

# then add to loss as such:
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add(base_loss, reg_losses, name="loss")


## DROPOUT ##
from tensorflow.contrib.layers import dropout

keep_prob = 0.5
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)


### GET WEIGHTS FROM fully_connected LAYER ###
## NOTE: this is obviously not needed if you implement the layer from scratch

from tensorflow.contrib.layers import batch_norm, fully_connected
from tensorflow.contrib.framework import arg_scope

X = tf.placeholder(tf.float32, shape=(1000,100), name='x')
with arg_scope(
		[fully_connected],
		weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):
	hidden1 = fully_connected(X, 50, scope="hidden1")
	hidden2 = fully_connected(hidden1, 30, scope="hidden2")
	logits 	= fully_connected(hidden2, 10, activation_fn=None, scope="out")

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
with tf.variable_scope("hidden1", reuse=True):
	weights1 = tf.get_variable("weights")
print sess.run(weights1)
sess.close()

# IF YOU DONT KNOW THE VARIABLES NAME - CHECK IT AS SUCH:
for variable in tf.all_variables():
    print variable.name

# OR use this, but drop the index after the colon (e.g. use weights1 not weights1:0)
for variable in tf.trainable_variables():
	print variable.name

# OR:
for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
	print variable.name


### MAX NORM (CUSTOM) REGULARIZER ON WEIGHTS ###
def max_norm_regularizer(threshold, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # there is no regularization loss term
    return max_norm

max_norm_reg = max_norm_regularizer(threshold=1.0)
hidden1 = fully_connected(X, n_hidden1, scope="hidden1",
                          weights_regularizer=max_norm_reg)

## tf.assign has to be run after each batch/epoch, so you need this:
clip_all_weights = tf.get_collection("max_norm")
with tf.Session() as sess:
    [...]
    for epoch in range(n_epochs):
        [...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)








