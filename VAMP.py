#-*- coding: utf-8 -*-
"""
VAMP.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of VAMP algorithm.
"""

import numpy as np
from scipy import linalg

def VAMP(A, X, Y, alf=1.1402, Tm=100):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :Tm: int, maximum iteration steps
    :alf: float, hyperparameter in AMP, should be tuned
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: np array, averaged nmse respect to all samples

    """
    m, n = A.shape
    _, l = Y.shape
    Xhat = np.zeros_like(X).astype(np.float64) # initial estimation of X, column vectors
    nmse = np.concatenate(([np.ones(l)], np.zeros((Tm, l))), axis=0);
        # nmse(t, k): nmse between Xhat(:, j) and X(:, j) at time t
    sum_square_X = np.sum(np.square(X.astype(np.float64)), 0)
    sum_square_X = sum_square_X + ( sum_square_X == 0 )

    eta = np.vectorize(lambda v, tau: (1. if v>=0. else -1.) * max(0., abs(v) - tau))
    Z = np.zeros_like(Y).astype(np.float64)
        # soft-thresholding
    for t in range(Tm):
        b = np.sum(np.abs(Xhat) != 0, axis=0) / m
        Z = Y - A.dot(Xhat) + b*Z
        lam = alf / np.sqrt(m) * linalg.norm(Z, ord=2, axis=0)
        R = Xhat + A.T.dot(Z)
        Xhat = eta(R, lam)
        # print("max value in Xhat in %d step is %f" % (t, np.max(Xhat)))
        nmse[t+1,:] = np.sum(np.square(Xhat-X), 0) / sum_square_X

    return Xhat, np.mean(nmse, 1)

