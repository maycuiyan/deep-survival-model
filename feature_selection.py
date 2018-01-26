import numpy as np

class SelectKBestMeta:
    
    def __init__(self, score_func, K):
        '''
        Arg
            score_func: any function which taks X, time, event as input and returns (score, weight) as output
            K: (int) number of best features to select
        '''
        self.K = K
        self.score_func = score_func
        
    def fit(self, datasets):
        '''
        Produce the indices of the best K features
        Arg
            datasets: a list of (X, time, event) tuples
        '''
        score_total = 0
        weight_total = 0
        for X, time, event in datasets:
            score, weight = self.score_func(X, time, event)
            score_total += score*weight
            weight_total += weight
        score_total /= weight_total
        self.indices = np.argsort(np.abs(score_total-0.5))[-self.K:]
        
    def transform(self, datasets):
        '''
        Transform dataset by keeping only the best K features
        '''
        datasets_new = []
        for X, time, event in datasets:
            datasets_new.append((X[:,self.indices], time, event))
        return datasets_new



def concordance_index(X, time, event):
    '''
    Calculate the concordance index of each feature of the input design matrix
    Args
        X: array of shape (n, p)
        time: a vector of survival time, shape (n, )
        event: a vector of events, shape (n, ) 
    Return
        CI: a vector of concordance indices, shape (p, )
        num_pairs: (int) number of valid pairs (i,j) satisfying event[i]=1 and time[i]<time[j]
    '''
    n = X.shape[0]
    CI = 0
    num_pairs = 0
    for i in range(n):
        if event[i]==1:
            for j in range(n):
                if time[i]<time[j]:
                    num_pairs += 1
                    CI += X[i,:]<X[j,:]
    return CI/num_pairs, num_pairs

if __name__=='__main__':

	from utils import generate_toy_datasets
	n_datasets = 10
	n_min, n_max = 20, 30 # size of each dataset is between 20 and 30
	n_features = 40
	datasets = generate_toy_datasets(n_datasets, n_min, n_max, n_features)
	k_best = 3
	feature_selector = SelectKBestMeta(concordance_index, k_best)
	feature_selector.fit(datasets)
	datasets_new = feature_selector.transform(datasets)