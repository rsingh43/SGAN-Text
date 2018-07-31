from __future__ import division

import sys
import argparse

import time
from datetime import timedelta

from tabulate import tabulate

import numpy as np

from sklearn.model_selection import KFold

from keras.datasets import reuters
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, UpSampling1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

def vectorize_sequences(sequences, dimension):
	# Create an all-zero matrix of shape (len(sequences), dimension)
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1. # Sets specific indices of results[i] to 1s
	return results

def build_discriminator(txt_shape, num_classes):
	model = Sequential()

	model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=txt_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))

	model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Flatten())

	#model.summary()

	txt = Input(shape=txt_shape)

	features = model(txt)
	valid = Dense(1, activation="sigmoid")(features)
	label = Dense(num_classes+1, activation="softmax")(features)

	return Model(txt, [valid, label])

def build_generator(latent_dim, num_words):
	model = Sequential()

	model.add(Dense(128 * 50,  activation="relu", input_dim=latent_dim))
	model.add(Reshape((50, 128)))
	model.add(BatchNormalization(momentum=0.8))
	model.add(UpSampling1D())

	model.add(Conv1D(1, kernel_size=3, padding="same"))
	model.add(Activation("tanh"))

	#model.summary()

	noise = Input(shape=(latent_dim,))
	txt = model(noise)

	return Model(noise, txt)

def train(discriminator, generator, combined, x_train, y_train, num_classes, num_words, epochs=200, batch_size=32, save_interval=50):
	half_batch = int(batch_size / 2)
	
	# Adversarial ground truths
	valid = np.ones((batch_size, 1))
	fake = np.zeros((batch_size, 1))
	
	#generator.trainable = True
	
	for epoch in range(epochs):
		# ---------------------
		#  Train Discriminator
		# ---------------------

		# Select a random half of txt
		idx = np.random.randint(0, x_train.shape[0], batch_size)
		txts = x_train[idx]

		# Sample noise and generate a batch of new txt
		noise = np.random.normal(0, 1, (batch_size, num_words))
		gen_txts = generator.predict(noise)
	   
		# One-hot encoding of labels
		labels = to_categorical(y_train[idx], num_classes=num_classes+1)
		fake_labels = to_categorical(np.full((batch_size, 1), num_classes), num_classes=num_classes+1)
		
		# Train the discriminator
		discriminator.trainable = True
		
		d_loss_real = discriminator.train_on_batch(txts, [valid, labels])
		d_loss_fake = discriminator.train_on_batch(gen_txts, [fake, fake_labels])
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# ---------------------
		#  Train Generator
		# ---------------------

		# Train the generator (wants discriminator to mistake txt as real)
		discriminator.trainable = False
		
		noise = np.random.normal(0, 1, (batch_size, num_words))
		validity = np.ones((batch_size, 1))
		
		g_loss = combined.train_on_batch(noise, validity)
		#g_loss = combined.train_on_batch(noise, np.ones((half_batch, 1)))

		# Plot the progress
		#print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))

def main(epochs, num_words):
	txt_rows = num_words
	latent_dim = 100
	
	optimizer = Adam(0.0002, 0.5)
	txt_cols = 1
	txt_shape = (txt_rows, txt_cols)
	num_classes = 46

	#load data
	(x_data, y_data), __ = reuters.load_data(num_words=num_words, test_split=0)

	x_data = vectorize_sequences(x_data, num_words)  # vectorized training data
	x_data = np.expand_dims(x_data, axis=3)

	y_data = y_data.reshape(-1, 1)

	ten_fold_cv = KFold(n_splits=10)

	print "epochs:", epochs

	test_filename = "{0}-words.{1:05d}-epochs.testing.csv".format(num_words, epochs)
	train_filename = "{0}-words.{1:05d}-epochs.training.csv".format(num_words, epochs)

	with open(test_filename, "w") as test_fp, open(train_filename, "w") as train_fp:
		test_acc = []
		train_acc = []
		for fold, (train_idx, test_idx) in enumerate(ten_fold_cv.split(x_data)):
			print "fold {0}:".format(fold),
			
			# Build and compile the discriminator
			discriminator = build_discriminator(txt_shape, num_classes)
			discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
					loss_weights=[0.5, 0.5],
					optimizer=optimizer,
					metrics=['accuracy'])

			# Build the generator
			generator = build_generator(latent_dim, num_words)
			#generator.compile(loss='binary_crossentropy', optimizer=optimizer)

			# The generator takes noise as input and generates text
			noise = Input(shape=(num_words,))
			txt = generator(noise)

			# For the combined model we will only train the generator
			discriminator.trainable = False

			# The valid takes generated images as input and determines validity
			valid, _ = discriminator(txt)

			# The combined model  (stacked generator and discriminator)
			# Trains generator to fool discriminator
			combined = Model(noise , valid)
			combined.compile(loss=['binary_crossentropy'],
					optimizer=optimizer)
		
			start = time.clock()	
			train(discriminator, generator, combined, x_data[train_idx], y_data[train_idx], num_classes, num_words, epochs=epochs)
			elapsed = time.clock() - start

			print timedelta(seconds=elapsed),

			binary_prediction, class_predictions = discriminator.predict(x_data[test_idx])
			correct = np.sum( np.argmax(class_predictions, axis=1) == (y_data[test_idx].reshape(len(test_idx))) )
			test_acc.append( 100.0 * correct / len(test_idx) )
			print test_acc[-1],
			
			binary_prediction, class_predictions = discriminator.predict(x_data[train_idx])
			correct = np.sum( np.argmax(class_predictions, axis=1) == (y_data[train_idx].reshape(len(train_idx))) )
			train_acc.append( 100.0 * correct / len(train_idx) )
			print train_acc[-1]

		
		test_results = [epochs, np.amin(test_acc), np.amax(test_acc), np.mean(test_acc), np.median(test_acc), np.std(test_acc)]
		print >> test_fp, ",".join(map(str,test_results))
		
		train_results = [epochs, np.amin(train_acc), np.amax(train_acc), np.mean(train_acc), np.median(train_acc), np.std(train_acc)]
		print >> train_fp, ",".join(map(str,train_results))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="10-Fold Cross Validation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("-n", "--num-words", type=int, default=100,  metavar="int", help="number of words to use")
	parser.add_argument("-e", "--epochs",    type=int, default=1000, metavar="int", help="number of training epochs")

	arguments = vars(parser.parse_args())

	main(**arguments)

