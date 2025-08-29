"""
Interdependence Score (IDS) computation.
Based on the work by Adityanarayanan Radhakrishnan (MIT License)
Original: https://github.com/aradha/interdependence_scores
"""
import numpy as np 
import math
import sys
from tqdm import tqdm

SEED = 1717
np.random.seed(SEED)

EPSILON = sys.float_info.epsilon

def transform(y, num_terms=6, bandwidth_term=1/2):
    B = bandwidth_term 
    exp = np.exp(-B * y**2)
    terms = []
    for i in range(num_terms):
        terms.append(exp * (y)**i / math.sqrt(math.factorial(i) *1.))
    y_ = np.concatenate(terms, axis=-1)
    return y_

def center(X):
    return X - np.mean(X, axis=0, keepdims=True)


def compute_p_val(C, X, Y=None, num_terms=6, p_norm='max', n_tests=100, bandwidth_term=1/2):

    gt = C
    count = 0

    n, dx = X.shape
    for i in tqdm(range(n_tests)):

        # Used to shuffle data
        random_noise = np.random.normal(size=(n, dx))
        permutations = np.argsort(random_noise, axis=0)
        X_permuted = X[permutations, np.arange(dx)[None, :]]

        if Y is not None:
            n, dy = Y.shape
            random_noise = np.random.normal(size=(n, dy))
            permutations = np.argsort(random_noise, axis=0)
            Y_permuted = Y[permutations, np.arange(dy)[None, :]]
            null = compute_IDS_numpy(X_permuted, Y=Y_permuted, num_terms=num_terms, 
                                     p_norm=p_norm, bandwidth_term=bandwidth_term)
        else:
            null = compute_IDS_numpy(X_permuted, Y=Y, num_terms=num_terms, 
                                     p_norm=p_norm, bandwidth_term=bandwidth_term)


        count += np.where(null > gt, 1, 0)

    p_vals = count / n_tests
    return p_vals


def compute_IDS_numpy(X, Y=None, num_terms=6, p_norm='max', 
                      p_val=False, num_tests=100, bandwidth_term=1/2):
    n, dx = X.shape
    X_t = transform(X, num_terms=num_terms, bandwidth_term=bandwidth_term)
    X_t = center(X_t)

    if Y is not None:
        _, dy = Y.shape
        Y_t = transform(Y, num_terms=num_terms, bandwidth_term=bandwidth_term)
        Y_t = center(Y_t)        
        cov = X_t.T @ Y_t
        X_std = np.sqrt(np.sum(X_t**2, axis=0))
        Y_std = np.sqrt(np.sum(Y_t**2, axis=0))        
        correlations = cov / (X_std.reshape(-1, 1) + EPSILON)
        C = correlations / (Y_std.reshape(1, -1) + EPSILON)
        C = C.reshape(num_terms, dx, num_terms, dy)
    else: 
        C = np.corrcoef(X_t.T)
        C = C.reshape(num_terms, dx, num_terms, dx)

    C = np.nan_to_num(C, nan=0, posinf=0, neginf=0)
    C = np.abs(C)

    if p_norm == 'max':
        C = np.amax(C, axis=(0, 2))
    elif p_norm == 2:
        C = C**2
        C = np.mean(C, axis=0)
        C = np.mean(C, axis=1)
        C = np.sqrt(C)    
    elif p_norm == 1:
        C = np.mean(C, axis=0)
        C = np.mean(C, axis=1)

    if p_val:
        p_vals = compute_p_val(C, X, Y=Y, num_terms=num_terms, p_norm=p_norm, 
                               n_tests=num_tests, bandwidth_term=bandwidth_term)
        return C, p_vals
    else: 
        return C
