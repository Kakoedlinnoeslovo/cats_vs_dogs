#import tensorflow as tf
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from random import shuffle

class DataPreporator:
	def __init__(self, path):
		self.path = path

	def run(self, batch_size, img_size):
		#tf.reset_default_graph()
		# x_temp = tf.placeholder(tf.float32, (128, 128, 3))
		# tf_img = tf.image.resize_images(x_temp, (img_size, img_size),
		# 								 tf.image.ResizeMethod.NEAREST_NEIGHBOR)

		X_train, labels = list(), list()
		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
		shuffle(files)
		for i, file in enumerate(files):
			im = cv2.imread(self.path + "/" + file)

			temp_label = file.split(".")[0]
			if temp_label == 'cat':
				labels.append(0)
			else:
				labels.append(1)

			im = cv2.resize(im, (img_size,img_size), interpolation=cv2.INTER_NEAREST)

			#resized_img = sess.run(tf_img, feed_dict={tf_img: im})
			X_train.append(im)
			if len(X_train) == batch_size:
				yield X_train, labels
				X_train, labels = list(), list()