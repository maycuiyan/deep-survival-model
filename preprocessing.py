"""
@author: Yi Cui
"""

import numpy as np

class FeatureTransformer:
    
    def __init__(self, transform_func):
        self.transform_func = transform_func
    
    def fit_transform(self, datasets):
        '''
        Transform the design matrix X of each dataset in datasets according to transform_func
        Arg
            datasets: a list of (X, time, event) tuples
        Return
            datasets_new: a list of (X_new, time, event) tuples
        '''
        datasets_new = []
        for X, time, event in datasets:
            X_new = self.transform_func(X)
            datasets_new.append([X_new, time, event])
        return datasets_new


def rank_transform(X):
    '''
    Transform each row of X into normalized rank
    Args
        X: numpy array of shape (n, p)
    Return
        Y: numpy array of shape (n, p)
    '''
    n, p = X.shape
    Y = np.copy(X)
    Y[np.expand_dims(np.arange(n),-1), np.argsort(X, axis=1)] = np.arange(1, p+1)/(p+1)
    return Y


def zscore_transform(X):
    '''
    Transform each column (feature) of X to have zero mean and unit standard deviation
    Args
        X: numpy array of shape (n, p)
    Return
        Y: numpy array of shape (n, p)
    '''
    Y = (X-np.mean(X, axis=0, keepdims=True))/np.std(X, axis=0, keepdims=True)
    return Y


if __name__=='__main__':

	X = np.random.randn(5, 10)
	Y = rank_transform(X)
	assert np.all(np.argsort(X)==np.argsort(Y))

	X = np.random.rand(5, 10)
	Y = zscore_transform(X)
	print(np.mean(Y, axis=0)) # should be all zeros
	print(np.std(Y, axis=0)) # should be all ones