"""
@author: Yi Cui
"""

import numpy as np
import random


def generate_toy_datasets(num_datasets, n_min, n_max, dim, lam=10, prob=0.5):
    '''
    Generate toy datasets for testing
    Arg
        num_datasets: (int), number of datasets to generate
        n_min: (int), minimum number of samples in each dataset
        n_max: (int), maximum number of samples in each dataset
        dim: (int), feature dimension
        lam: (float), mean of exponential distribution to sample survival time
        prob: (float), probability of events
    Return:
        datasets: a list of (X, time, event) tuples
    '''
    datasets = []
    for _ in range(num_datasets):
        n = random.randint(n_min, n_max)
        X = np.random.randn(n, dim)
        time = np.random.exponential(lam, n)
        event = np.random.binomial(1, prob, n)
        datasets.append((X, time, event)) 
    return datasets


def train_test_split(datasets, test_size):
    '''
    Split datasets by stratified sampling
    Each dataset in datasets are equally split according to test_size
    Arg
        datasets: a list of (X, time, event) tuples
        test_size: (float) proportion of datasets assigned for test data
    Return
        datasets_train: a list of (X_train, time_train, event_train) tuples
        datasets_test: a list of (X_test, time_test, event_test) tuples
    '''
    datasets_train = [] 
    datasets_test = []
    for X, time, event in datasets:
        n = X.shape[0]
        idx = np.random.permutation(n)
        idx_train = idx[int(n*test_size):]
        idx_test = idx[:int(n*test_size)]
        datasets_train.append((X[idx_train], time[idx_train], event[idx_train]))
        datasets_test.append((X[idx_test], time[idx_test], event[idx_test]))
    return datasets_train, datasets_test


def combine_datasets(datasets):
    '''
    Combine all the datasets into a single dataset
    Arg
        datasets: datasets: a list of (X, time, event) tuples
    Return
        X: combined design matrix
        time: combined survival time
        event: combined event
    '''
    X, time, event = zip(*datasets)
    X = np.concatenate(X, axis=0)
    time = np.concatenate(time, axis=0)
    event = np.concatenate(event, axis=0)
    return X, time, event


def get_index_pairs(datasets):
    '''
    For each dataset in datasets, get index pairs (idx1,idx2) satisfying time[idx1]<time[idx2] and event[idx1]=1
    Arg
        datasets: a list of (X, time, event) tuples
    Return
        index_pairs: a list of (idx1, idx2) tuples, where idx1 and idx2 are index vectors of the same length
    '''
    index_pairs = []
    for _, time, event in datasets:
        index_pairs.append(np.nonzero(np.logical_and(np.expand_dims(time,-1)<time, np.expand_dims(event,-1))))
    return index_pairs


def batch_factory(X, time, event, batch_size):
    n = X.shape[0]
    num_batches = n//batch_size
    idx = np.random.permutation(n)
    X, time, event = X[idx], time[idx], event[idx] # randomly shuffle data
    start = 0
    def next_batch():
        nonlocal start
        X_batch = X[start:start+batch_size]
        time_batch = time[start:start+batch_size]
        event_batch = event[start:start+batch_size]
        start = (start+batch_size)%n
        return X_batch, time_batch, event_batch
    return next_batch, num_batches


if __name__=='__main__':

	n_datasets = 10
	n_min, n_max = 20, 30
	n_features = 40
	datasets = generate_toy_datasets(n_datasets, n_min, n_max, n_features)
	index_pairs = get_index_pairs(datasets)
	for i, (_, time, event) in enumerate(datasets):
	    idx1, idx2 = index_pairs[i]
	    assert np.all(time[idx1]<time[idx2])
	    assert np.all(event[idx1])