#-*- coding: utf-8 -*-
"""
ISTA.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of ISTA algorithm.
"""

import numpy as np

def ISTA(A, X, Y, lam=0.01, Tm=10000):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :Tm: int, maximum iteration steps
    :lam: float, threshold lambda in ISTA, should be tuned
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

    scale = .999 / np.linalg.norm(A.T.dot(A), ord=2)
    B   = scale * np.transpose(A)
    tau = lam * scale
    for t in range(Tm):
        Z = Y - A.dot(Xhat)
        R = Xhat + np.dot(B, Z)
        Xhat = np.sign(R) * np.maximum( np.abs(R) - tau, 0 )
        nmse[t+1,:] = np.sum(np.square(Xhat-X), 0) / sum_square_X

    return Xhat, np.mean(nmse, 1)

