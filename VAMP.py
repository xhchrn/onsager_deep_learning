#-*- coding: utf-8 -*-
"""
VAMP.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of VAMP algorithm.
"""

import numpy as np
from scipy import linalg

def VAMP(A, X, Y, gamma1=0.5, gammaw=1.0, Tm=100):
    """

    :A: np array, measurement matrix
    :X: np array, ground truth sparse coding as column vectors
    :Y: np array, measurements as column vectors
    :Tm: int, maximum iteration steps
    :gamma1: float, initial value of gamma1
    :gammaw: float, assumed noise precision of X0
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: np array, averaged nmse respect to all samples

    """
    m, n = A.shape
    _, l = Y.shape

    U, s, Vh = linalg.svd(A, full_matrices=False)
    S = linalg.diagsvd(s, len(s), len(s))
    Ytil = gammaw * np.dot(S.T, np.dot(U.T, Y))

    nmse = np.concatenate(([np.ones(l)], np.zeros((tm, l))), axis=0);
        # nmse(t, k): nmse between xhat(:, j) and x(:, j) at time t
    sum_square_x = np.sum(np.square(x.astype(np.float64)), 0)
    sum_square_x = sum_square_x + ( sum_square_x == 0 )

    R1 = Y # initialized R1
    g1 = np.vectorize(lambda v, tau: (1. if v>=0. else -1.) * max(0., abs(v) - tau))
        # soft-thresholding
    for t in range(tm):
        # Denoising step
        Xhat1   = g1(R1, gamma1)
        alf1    = np.sum(np.abs(xhat1) != 0, axis=0) / m
        eta1    = gamma1 / alf1
        gamma2  = eta1 - gamma1
        R2      = (eta1 * Xhat1 - gamma1 * R1) / gamma2

        # LMMSE estimation
        D       = np.diag(1. / (gammaw * np.square(s) + gamma2))
        Xhat2   = np.dot(np.dot(Vh.T, D), Ytil + gamma2 * np.dot(Vh, R2))
        alf2    = np.trace(D) * gamma2 / N
        eta2    = gamma2 / alf2
        gamma1  = eta2 - gamma2
        R1      = ( eta2 * Xhat2 - gamma2 * R2 ) / gamma1

        # print("max value in Xhat in %d step is %f" % (t, np.max(Xhat)))
        nmse[t+1,:] = np.sum(np.square(Xhat1-X), 0) / sum_square_X

    return Xhat1, np.mean(nmse, 1)

