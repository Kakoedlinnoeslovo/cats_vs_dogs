import tensorflow as tf
from Preporator import DataPreporator
import numpy as np



class Classifier:
	def __init__(self, path):
		self.preporator = DataPreporator(path)


	def run_network(self, features,  mode, labels):
		input_tensor = tf.reshape(features["x"], [-1, 128, 128, 3])
		conv1 = tf.layers.conv2d(inputs = input_tensor,
								 filters = 64,
								 kernel_size=[5, 5],
								 kernel_initializer=tf.contrib.layers.xavier_initializer(),
								 padding='same',
								 activation=tf.nn.relu,
								 name='conv1')

		pool1 = tf.layers.max_pooling2d(inputs=conv1,
							   pool_size=[2, 2],
							   strides=2,
							   name = 'pool1')

		conv2 = tf.layers.conv2d(inputs = pool1,
								 filters=128,
								 kernel_size=[5, 5],
								 kernel_initializer=tf.contrib.layers.xavier_initializer(),
								 padding='same',
								 activation=tf.nn.relu,
								 name = 'conv2')

		pool2 = tf.layers.max_pooling2d(inputs=conv2,
							   pool_size=[2, 2],
							   strides=2,
							   name  = 'pool2')


		pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 128])

		dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=True)

		# Logits Layer
		logits = tf.layers.dense(inputs=dropout, units=1)

		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
		}

		loss = tf.losses.sigmoid_cross_entropy(multi_class_labels =labels, logits=logits)
		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


		if mode == tf.estimator.ModeKeys.TRAIN:
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

		# Add evaluation metrics (for EVAL mode)
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=labels, predictions=predictions["classes"])}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main():
	batch_size = 10
	image_size = 128

	my_network = Classifier("train/")
	x_train = my_network.preporator.run(batch_size, image_size)
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
		model_fn=my_network.run_network, model_dir="/tmp/mnist_convnet_model")
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "sigmoid_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50
	)


	x_train = x_train.__next__()
	x_train, labels = np.array(x_train[0], dtype=np.float32), np.array(x_train[1]).reshape(-1,1)
	print(x_train.shape)
	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x= {"x": x_train},
		y= labels,
		batch_size=len(labels),
		num_epochs=None,
		shuffle=True)
	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=20000,
		hooks=[logging_hook])


if __name__ == "__main__":
	main()