#-*- coding: utf-8 -*-
"""
VAMP.py
author: chernxh@tamu.edu
date  : 09/26/2017

Python implementation of VAMP algorithm.
"""

import numpy as np
from scipy import linalg

def VAMP1(A, X, Y, alf=1.1402, Sigt=0.001, Sigw=1.0, Tm=100):
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
    S = np.diag(s)
    Yt = np.dot( S, np.dot( U.T, Y ))

    nmse = np.concatenate(([np.ones(l)], np.zeros((Tm, l))), axis=0);
        # nmse(t, k): nmse between xhat(:, j) and x(:, j) at time t
    sum_square_X = np.sum(np.square(X.astype(np.float64)), 0)
    sum_square_X = sum_square_X + ( sum_square_X == 0 )

    Sigt = Sigt * np.ones(l).astype(np.float64)
    Rt = np.dot(A.T, Y) # initialized R1
    Xht = np.zeros_like(X).astype(np.float64)
    R   = np.zeros_like(Y).astype(np.float64)
    Nut  = np.zeros(l)
    for t in range(Tm):
        # LMMSE estimation
        for i in range(l):
            coef = Sigw**2 / Sigt[i]**2
            D = np.diag( 1./( np.square(s) + coef ) )
            Xht[:, i] = np.dot(np.dot(Vh.T, D), Yt[:,i] + coef * np.dot(Vh, Rt[:, i]))
            Nut[i] = np.trace(D) * coef / n
        R   = ( Xht - Nut*Rt ) / ( 1 - Nut )
        Sig = np.sqrt( np.square(Sigt) * Nut / ( 1 - Nut ) )

        # Denoising step
        Xh   = np.sign(R) * np.maximum( np.abs(R) - alf * Sig, 0 )
        Nu   = np.sum(np.abs(Xh) != 0, axis=0) / ( n+1 )
        Rt   = ( Xh - Nu * R ) / ( 1 - Nu )
        Sigt = np.sqrt( np.square(Sig) * Nu / ( 1 - Nu ) )

        # print("max value in Xhat in %d step is %f" % (t, np.max(Xhat)))
        nmse[t+1,:] = np.sum(np.square(Xh-X), 0) / sum_square_X

    return Xh, np.mean(nmse, 1)

