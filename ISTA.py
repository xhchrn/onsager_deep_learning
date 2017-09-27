#-*- coding: utf-8 -*-
"""
ISTA.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of ISTA algorithm.
"""

import numpy as np

def ISTA(A, X, Y, lam=0.01, T=10000):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :T: int, maximum iteration steps
    :lam: float, threshold lambda in ISTA, should be tuned
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: np array, averaged nmse respect to all samples

    """
    m, n = A.shape
    _, l = Y.shape
    Xhat = np.zeros_like(X) # initial estimation of X, column vectors
    nmse = np.concatenate(([np.ones(l)], np.zeros((T, L))), axis=0);
        # nmse(t, k): nmse between Xhat(:, j) and X(:, j) at time t
    eta = np.vectorize(lambda v: (1 if v>=0 else -1) * max(0, abs(v) - lam)) # soft-thresholding
    for t in range(T):
        Z = Y - A.dot(Xhat)
        R = Xhat + A.T.dot(z)
        Xhat = eta(R)
        nmse[t+1,:] = np.sum(np.square(Xhat-X), 1) / np.sum(np.square(X), 1)

    return Xhat, np.mean(nmse, 2)

