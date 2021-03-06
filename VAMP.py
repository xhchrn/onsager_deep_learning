#-*- coding: utf-8 -*-
"""
VAMP.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of VAMP algorithm.
"""

import numpy as np
from scipy import linalg

def VAMP(A, X, Y, gamma1=0.001, gammaw=1.0, Tm=100):
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

    nmse = np.concatenate(([np.ones(l)], np.zeros((Tm, l))), axis=0);
        # nmse(t, k): nmse between xhat(:, j) and x(:, j) at time t
    sum_square_X = np.sum(np.square(X.astype(np.float64)), 0)
    sum_square_X = sum_square_X + ( sum_square_X == 0 )

    gamma1 = gamma1 * np.ones(l).astype(np.float64)
    R1 = np.dot(A.T, Y) # initialized R1
    Xhat2 = np.zeros_like(X).astype(np.float64)
    alf2  = np.zeros(l)
    for t in range(Tm):
        # Denoising step
        Xhat1   = np.sign(R1) * np.maximum( np.abs(R1) - gamma1, 0 )
        alf1    = np.sum(np.abs(Xhat1) != 0, axis=0) / n
        eta1    = gamma1 / alf1
        gamma2  = eta1 - gamma1
        R2      = (eta1 * Xhat1 - gamma1 * R1) / gamma2

        # LMMSE estimation
        for i in range(l):
            D = np.diag( 1./(gammaw * np.square(s) + gamma2[i]) )
            Xhat2[:, i] = np.dot(np.dot(Vh.T, D), Ytil[:,i] + gamma2[i] * np.dot(Vh, R2[:, i]))
            alf2[i]     = np.trace(D) * gamma2[i] / n
        eta2    = gamma2 / alf2
        gamma1  = eta2 - gamma2
        R1      = ( eta2 * Xhat2 - gamma2 * R2 ) / gamma1

        # print("max value in Xhat in %d step is %f" % (t, np.max(Xhat)))
        nmse[t+1,:] = np.sum(np.square(Xhat1-X), 0) / sum_square_X

    return Xhat1, np.mean(nmse, 1)

