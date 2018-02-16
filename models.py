"""
@author: Yi Cui
"""

import tensorflow as tf
import utils
import keras.backend as K
import numpy as np


class SurvivalModel:

	def __init__(self, model_builder):
		'''
		Args
			model_builder: a function which returns user defined Keras model
		'''
		self.model_builder = model_builder



	def fit(self, datasets_train, datasets_val, loss_func='hinge', epochs=500, lr=0.001, mode='decentralize', batch_size = 64):
		'''
		Train a deep survival model
		Args
			datasets_train:     training datasets, a list of (X, time, event) tuples
			datasets_val:       validation datasets, a list of (X, time, event) tuples
			loss_func:          loss function to approximate concordance index, {'hinge', 'log', 'cox'}
			epochs:             number of epochs to train
			lr:                 learning rate
			mode:               if mode=='merge', merge datasets before training
								if mode='decentralize', treat each dataset as a mini-batch
			batch_size:         only effective for 'merge' mode
		'''

		self.datasets_train = datasets_train
		self.datasets_val = datasets_val
		self.loss_func = loss_func
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size

		## build a tensorflow graph to define loss function
		self.__build_graph()
		
		## train the model
		if mode=='merge':
		  self.__train_merge()
		elif mode=='decentralize':
		  self.__train_decentralize()



	def __build_graph(self):
		'''
		Build a tensorflow graph. Call this within self.fit()
		'''

		input_shape = self.datasets_train[0][0].shape[1:]

		with tf.name_scope('input'):
			X = tf.placeholder(dtype=tf.float32, shape=(None, )+input_shape, name='X')
			time = tf.placeholder(dtype=tf.float32, shape=(None, ), name='time')
			event = tf.placeholder(dtype=tf.int16, shape=(None, ), name='event')

		with tf.name_scope('model'):
			model = self.model_builder(input_shape)

		with tf.name_scope('output'):
			score = tf.identity(model(X), name='score')

		with tf.name_scope('metric'):
			ci = self.__concordance_index(score, time, event)
			if self.loss_func=='hinge':
				loss = self.__hinge_loss(score, time, event)
			elif self.loss_func=='log':
				loss = self.__log_loss(score, time, event)
			elif self.loss_func=='cox':
				loss = self.__cox_loss(score, time, event)

		with tf.name_scope('train'):
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			train_op = optimizer.minimize(loss, name='train_op')

		## save the tensors and ops so that we can use them later
		self.__X = X
		self.__time = time
		self.__event = event
		self.__score = score
		self.__ci = ci
		self.__loss = loss
		self.__train_op = train_op



	def __train_decentralize(self):
		'''
		Decentralized training mode. Each dataset is regarded as a mini-batch
		'''

		## start training
		self.__sess = tf.Session()
		self.__sess.run(tf.global_variables_initializer())
		for epoch in range(self.epochs):
			for X_batch, time_batch, event_batch in self.datasets_train:
				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 1})
			if epoch%100==0:
				print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
				self.__print_loss_ci()
				


	def __train_merge(self):
		'''
		Merge training datasets into a single dataset. Sample mini-batches from the merged dataset for training
		'''

		## Merge training datasets
		X_train, time_train, event_train = utils.combine_datasets(self.datasets_train)

		## To fetch mini-batches
		next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)

		## start training
		self.__sess = tf.Session()
		self.__sess.run(tf.global_variables_initializer())
		for epoch in range(self.epochs):
			for _ in range(num_batches):
				X_batch, time_batch, event_batch = next_batch()
				self.__sess.run(self.__train_op, feed_dict={self.__X: X_batch, self.__time: time_batch, self.__event: event_batch, K.learning_phase(): 1})
			if epoch%100==0:
				print('-'*20 + 'Epoch: {0}'.format(epoch) + '-'*20)
				self.__print_loss_ci()



	def predict(self, X_test):
		'''
		Args
			X: design matrix of shape (num_samples, ) + input_shape
		'''

		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'
		
		return self.__sess.run(self.__score, feed_dict = {self.__X: X_test, K.learning_phase():0})



	def evaluate(self, X_test, time_test, event_test):
		'''
		Evaluate the loss and c-index of the model for the given test data
		'''
		assert X_test.shape[1:]==self.datasets_train[0][0].shape[1:], 'Shapes of testing and training data must equal'

		return self.__sess.run([self.__loss, self.__ci], feed_dict = {self.__X: X_test, self.__time: time_test, self.__event: event_test, K.learning_phase(): 0})



	def __concordance_index(self, score, time, event):
		'''
		Args
			score: 		predicted score, tf tensor of shape (None, )
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## count how many score[i]<score[j]
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		ci = tf.reduce_mean(tf.cast(s1<s2, tf.float32), name='c_index')

		return ci



	def __hinge_loss(self, score, time, event):
		'''
		Args
			score:	 	predicted score, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur hinge loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.maximum(1+s1-s2, 0.0), name='loss')

		return loss



	def __log_loss(self, score, time, event):
		'''
		Args
			score: 	predicted survival time, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		'''

		## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
		ix = tf.where(tf.logical_and(tf.expand_dims(time, axis=-1)<time, tf.expand_dims(tf.cast(event, tf.bool), axis=-1)), name='ix')

		## if score[i]>score[j], incur log loss
		s1 = tf.gather(score, ix[:,0])
		s2 = tf.gather(score, ix[:,1])
		loss = tf.reduce_mean(tf.log(1+tf.exp(s1-s2)), name='loss')

		return loss



	def __cox_loss(self, score, time, event):
		'''
		Args
			score: 		predicted survival time, tf tensor of shape (None, 1)
			time:		true survival time, tf tensor of shape (None, )
			event:		event, tf tensor of shape (None, )
		Return
			loss:		partial likelihood of cox regression
		'''

		## cox regression computes the risk score, we want the opposite
		score = -score

		## find index i satisfying event[i]==1
		ix = tf.where(tf.cast(event, tf.bool)) # shape of ix is [None, 1]
		
		## sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
		sel_mat = tf.cast(tf.gather(time, ix)<=time, tf.float32)

		## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
		p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(sel_mat * tf.transpose(tf.exp(score)), axis=-1))
		loss = -tf.reduce_mean(p_lik)

		return loss


	def __print_loss_ci(self):
		'''
		Helper function to print the losses and c-indices on training & validation datasets
		'''
		## losses and c-indices on traning
		loss_train = np.zeros(len(self.datasets_train))
		ci_train = np.zeros(len(self.datasets_train))
		for i, (X_batch, time_batch, event_batch) in enumerate(self.datasets_train):
			loss_train[i], ci_train[i] = self.evaluate(X_batch, time_batch, event_batch)

		## losses and c-indices on validation
		loss_val = np.zeros(len(self.datasets_val))
		ci_val = np.zeros(len(self.datasets_val))
		for i, (X_batch, time_batch, event_batch) in enumerate(self.datasets_val):
			loss_val[i], ci_val[i] = self.evaluate(X_batch, time_batch, event_batch)

		## print them
		print('loss_train={0}'.format(np.round(loss_train, 2)))
		print('loss_val={0}'.format(np.round(loss_val, 2)))
		print('ci_train={0}'.format(np.round(ci_train, 2)))
		print('ci_val={0}'.format(np.round(ci_val, 2)))
		print()