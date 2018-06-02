import tensorflow as tf
from Preporator import DataPreporator
import numpy as np
from tqdm import tqdm


class Classifier:
	def __init__(self, path):
		self.preporator = DataPreporator(path)


	def run_network(self, features,  mode, labels):
		input_tensor = tf.reshape(features["x"], [-1, 128, 128, 3])
		conv1 = tf.layers.conv2d(inputs = input_tensor,
								 filters = 32,
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
								 filters=64,
								 kernel_size=[5, 5],
								 kernel_initializer=tf.contrib.layers.xavier_initializer(),
								 padding='same',
								 activation=tf.nn.relu,
								 name = 'conv2')

		pool2 = tf.layers.max_pooling2d(inputs=conv2,
							   pool_size=[2, 2],
							   strides=2,
							   name  = 'pool2')


		pool2_flat = tf.reshape(pool2, [-1, 32 * 32 * 64])

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
			optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
	batch_size = 100
	image_size = 128
	my_network = Classifier("train/")
	x_train = my_network.preporator.run(batch_size, image_size)
	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(
		model_fn=my_network.run_network)
	# Set up logging for predictions
	tensors_to_log = {"probabilities": "sigmoid_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50
	)

	for i in tqdm(range(100)):
		train = x_train.__next__()
		train, labels = np.array(train[0], dtype=np.float32), np.array(train[1]).reshape(-1,1)
		#print("Train shape: {}".format(train.shape))
		#print(labels)
		# Train the model
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x= {"x": train},
			y= labels,
			batch_size=len(labels),
			num_epochs=None,
			shuffle=True)
		mnist_classifier.train(
			input_fn=train_input_fn,
			steps=10,
			hooks=[logging_hook])

		eval = x_train.__next__()
		eval_data, eval_labels = np.array(eval[0], dtype=np.float32), np.array(eval[1]).reshape(-1, 1)
		#print("Eval shape: {}".format(eval_data.shape))

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": eval_data},
			y=eval_labels,
			num_epochs=1,
			shuffle=False)
		eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
		print("Eval results: {}".format(eval_results))
		#print("Ground true labels: {}".format(labels))


if __name__ == "__main__":
	main()