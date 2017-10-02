"""
exp.py
usage: python exp.py kappa exp_id

Experiment with ISTA, AMP, VAMP, LAMP-l1 and compare their resistance to A with large condition
number kappa.

author: chernxh@tamu.edu
date  : 09/20/2017
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUITE!!!

import numpy as np
import tensorflow as tf
from matplotlib import pyplot

from tools import networks, train, problems

# import iterative solvers
from ISTA  import ISTA
from FISTA import FISTA
from AMP   import AMP

if __name__ == "__main__":
    kappa = float(sys.argv[1])
    exp_id = int(sys.argv[2])

    prob = 'probs/problem_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)
    prob = problems.read_prob_from_npz(prob)

    LISTA = 'models/LISTA_bg_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)
    LAMP  = 'models/LAMP_bg_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)



