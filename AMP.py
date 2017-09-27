# -*- coding: utf-8 -*-
"""
AMP.py
author: xhchrn
        chernxh@tamu.edu
date  : 09/27/2017

Python implementation of AMP algorithm
"""

import numpy as np

def AMP(A, X, Y, alf=1.1402, T=100):
    """TODO: Docstring for AMP.

    :A  : np array, measurements matrix
    :X  : np array, ground truth sparse coding as column vectors
    :Y  : np array, measurements as column vectors
    :T  : integer , maximum iteration steps
    :alf: float   , tuning hyperparameter in AMP algorithm
    :returns:
        :Xhat : np array, approximated sparse coding
        :anmse: avaraged nmse

    """
    m, n = A.shape


