"""
@author: Yi Cui
"""

import tensorflow as tf
import utils
import keras.backend as K


class SurvivalModel:

	def __init__(self, model_builder):
		'''
		Args
			model_builder: a function which returns user defined Keras model
		'''
		self.model_builder = model_builder

	def fit(self, datasets_train, datasets_val, loss_func='hinge', epochs=500, lr=0.001, mode='decentral', batch_size = 64):
		'''
		Train a deep survival model
		Args
			datasets_train:     training datasets, a list of (X, time, event) tuples
			datasets_val:       validation datasets, a list of (X, time, event) tuples
			loss_func:          loss function to approximate concordance index, {'hinge', 'logloss', 'cox'}
			epochs:             number of epochs to train
			lr:                 learning rate
			mode:               if mode=='merge', merge datasets before training
								if mode='decentral', treat each dataset as a mini-batch
			batch_size:         only effective for 'merge' mode
		'''

		self.datasets_train = datasets_train
		self.datasets_val = datasets_val
		self.loss_func = loss_func
		self.epochs = epochs
		self.lr = lr
		self.batch_size = batch_size

		self._build_graph()
		
		if mode=='merge':
		  self._train_merge()
		elif mode=='decentral':
		  self._train_decentral()


	def predict(self, X_test):
		'''
		Args
			X: a n_samples*n_features design matrix
		'''

		assert X_test.shape[1]==self.datasets_train[0][0].shape[1], '#features of testing and training data must equal'

		if hasattr(self, 'sess'):
			## fetch tensors and ops from graph
			g = tf.get_default_graph()
			X = g.get_tensor_by_name('input/X:0')
			y_pred = g.get_tensor_by_name('output/y_pred:0')
			y_pred = self.sess.run(y_pred, feed_dict = {X: X_test, K.learning_phase():0})
		else:
			print('Model has not been trained yet. Run fit() first')
		
		return y_pred


	def _build_graph(self):
		'''
		Build a tensorflow graph; only call this within self.train()
		'''

		feature_dim = self.datasets_train[0][0].shape[1]

		tf.reset_default_graph()

		with tf.name_scope('input'):
			X = tf.placeholder(dtype=tf.float32, shape=[None, feature_dim], name='X')
			idx1 = tf.placeholder(dtype=tf.int32, shape=[None, ], name='idx1')
			idx2 = tf.placeholder(dtype=tf.int32, shape=[None, ], name='idx2')

		with tf.name_scope('phase'):
			phase = tf.identity(K.learning_phase(), name='phase')

		with tf.name_scope('model'):
			model = self.model_builder(feature_dim)

		with tf.name_scope('output'):
			y_pred = tf.identity(model(X), name='y_pred')
			y1 = tf.gather(y_pred, idx1)
			y2 = tf.gather(y_pred, idx2)

		with tf.name_scope('metrics'):
			if self.loss_func=='hinge':
				loss = tf.reduce_mean(tf.maximum(1+y1-y2, 0.0), name='loss')
			elif self.loss_func=='logloss':
				loss = tf.reduce_mean(tf.log(1+tf.exp(y1-y2)), name='loss')
			ci = tf.reduce_mean(tf.cast(y1<y2, tf.float32), name='c_index')

		with tf.name_scope('train'):
			optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
			train_op = optimizer.minimize(loss, name='train_op')

		with tf.name_scope('init'):
			init = tf.global_variables_initializer()


	def _train_decentral(self):
		'''
		Train survival in a decentralized way. Each dataset is regarded as a mini-batch
		'''

		## precompute index pairs
		index_pairs_train = utils.get_index_pairs(self.datasets_train) 
		index_pairs_val = utils.get_index_pairs(self.datasets_val)

		## fetch tensors and ops from graph
		g = tf.get_default_graph()
		X = g.get_tensor_by_name('input/X:0')
		idx1 = g.get_tensor_by_name('input/idx1:0')
		idx2 = g.get_tensor_by_name('input/idx2:0')
		loss = g.get_tensor_by_name('metrics/loss:0')
		ci = g.get_tensor_by_name('metrics/c_index:0')
		train_op = g.get_operation_by_name('train/train_op')
		init = g.get_operation_by_name('init/init')

		## start training
		sess = tf.Session()

		sess.run(init)

		for epoch in range(self.epochs):
			for i, (X_batch, _, _) in enumerate(self.datasets_train):
				idx1_batch, idx2_batch = index_pairs_train[i]
				sess.run(train_op, feed_dict={X: X_batch, idx1: idx1_batch, idx2: idx2_batch, K.learning_phase(): 1})
			if epoch%100==0:
				loss_train, ci_train = self._total_loss_ci(self.datasets_train, index_pairs_train, sess)
				loss_val, ci_val = self._total_loss_ci(self.datasets_val, index_pairs_val, sess)
				print('Epoch {0}: loss_train={1:5.4f}, ci_train={2:5.4f}, loss_val={3:5.4f}, ci_val={4:5.4f}'.format(epoch, loss_train, ci_train, loss_val, ci_val))
		self.sess = sess



	def _train_merge(self):
		'''
		Merge training datasets into a single dataset. Sample mini-batches from the merged dataset for training
		'''

		## precompute index pairs
		index_pairs_train = utils.get_index_pairs(self.datasets_train) 
		index_pairs_val = utils.get_index_pairs(self.datasets_val)

		## Merge training datasets
		X_train, time_train, event_train = utils.combine_datasets(self.datasets_train)

		## To fetch mini-batches
		next_batch, num_batches = utils.batch_factory(X_train, time_train, event_train, self.batch_size)

		## fetch tensors and ops from graph
		g = tf.get_default_graph()
		X = g.get_tensor_by_name('input/X:0')
		idx1 = g.get_tensor_by_name('input/idx1:0')
		idx2 = g.get_tensor_by_name('input/idx2:0')
		loss = g.get_tensor_by_name('metrics/loss:0')
		ci = g.get_tensor_by_name('metrics/c_index:0')
		train_op = g.get_operation_by_name('train/train_op')
		init = g.get_operation_by_name('init/init')

		## start training
		sess = tf.Session()

		sess.run(init)

		for epoch in range(self.epochs):
			for _ in range(num_batches):
				X_batch, time_batch, event_batch = next_batch()
				idx1_batch, idx2_batch = utils.get_index_pairs([(X_batch, time_batch, event_batch)])[0]
				sess.run(train_op, feed_dict={X: X_batch, idx1: idx1_batch, idx2: idx2_batch, K.learning_phase(): 1})
			if epoch%100==0:
				loss_train, ci_train = self._total_loss_ci(self.datasets_train, index_pairs_train, sess)
				loss_val, ci_val = self._total_loss_ci(self.datasets_val, index_pairs_val, sess)
				print('Epoch {0}: loss_train={1:5.4f}, ci_train={2:5.4f}, loss_val={3:5.4f}, ci_val={4:5.4f}'.format(epoch, loss_train, ci_train, loss_val, ci_val))
		self.sess = sess


	def _total_loss_ci(self, datasets, index_pairs, sess):
		'''
		A function to make model eval eaiser; only call this in self._train_decentral() or self._train_merge()
		'''

		## fetch tensors from graph
		g = tf.get_default_graph()
		X = g.get_tensor_by_name('input/X:0')
		idx1 = g.get_tensor_by_name('input/idx1:0')
		idx2 = g.get_tensor_by_name('input/idx2:0')
		loss = g.get_tensor_by_name('metrics/loss:0')
		ci = g.get_tensor_by_name('metrics/c_index:0')

		loss_total = 0
		ci_total = 0
		n_total = 0

		for i, (X_batch, _, _) in enumerate(datasets):
			idx1_batch, idx2_batch = index_pairs[i]
			loss_batch, ci_batch = sess.run([loss, ci], feed_dict={X:X_batch, \
																   idx1:idx1_batch, \
																   idx2:idx2_batch, \
																   K.learning_phase():0})
			loss_total += len(idx1_batch)*loss_batch
			ci_total += len(idx1_batch)*ci_batch
			n_total += len(idx1_batch)
		loss_total /= n_total
		ci_total /= n_total

		return loss_total, ci_total