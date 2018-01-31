# Deep Survival Model

This repo contains the tensorflow implementation of building a deep survival model. It is a deep learning extension of the framework proposed in the following paper:
> Yi Cui, Bailiang Li, and Ruijiang Li. "Decentralized Learning Framework of Meta-Survival Analysis for Developing Robust Prognostic Signatures." JCO Clinical Cancer Informatics 1 (2017): 1-13.

A step-by-step guide to use the codes is demonstrated in [template.ipynb](https://github.com/maycuiyan/deep-survival-model/blob/master/template.ipynb).

We provide functions and classes to streamline traning a survival model from multiple (possibly very heterogneous) datasets. In particular, 

* [utils.py](https://github.com/maycuiyan/deep-survival-model/blob/master/utils.py) provides various utility functions to maniputate multiple survival datasets. 

* [preprocessing.py](https://github.com/maycuiyan/deep-survival-model/blob/master/preprocessing.py) provides **zscore** and **percentile-rank** transformation methods that are commonly used to standardize gene expression profiles. 

* [feature_selection.py](https://github.com/maycuiyan/deep-survival-model/blob/master/feature_selection.py) provides a generic class SelectKBestMeta applying **meta-analysis** for feature selection with multiple datasets. A concordance index-based score function to be used with the SelectKBestMeta class is also provided. 

* [models.py](https://github.com/maycuiyan/deep-survival-model/blob/master/models.py) provides the core SurvivalModel class, which has the fit() and predict() methods to facilitate training and deploying of survival models. This class also renders the flexibility of user specified deep learning models by leveraging the high-level Keras callable models (see [template.ipynb](https://github.com/maycuiyan/deep-survival-model/blob/master/template.ipynb) for details). 
