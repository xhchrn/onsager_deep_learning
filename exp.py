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
import matplotlib.pyplot as plt

from tools import networks, train, problems

# import iterative solvers
from ISTA  import ISTA
from FISTA import FISTA
from AMP   import AMP

if __name__ == "__main__":
    kappa = float(sys.argv[1])
    exp_id = int(sys.argv[2])

    prob_name = 'probs/problem_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)

    model_LISTA = 'models/LISTA_bg_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)
    model_LAMP  = 'models/LAMP_bg_k{0:04.1f}_{1}.npz'.format(kappa, exp_id)

    g_LISTA = tf.Graph()
    g_LAMP  = tf.Graph()

    with g_LISTA.as_default() as g:
        prob = problems.read_prob_from_npz(prob_name)
        layers_LISTA  = networks.build_LISTA(prob, T=6, initial_lambda=.1, untied=False)
        _, nmse_LISTA = train.test(g, model_LISTA, layers_LISTA, prob)

    with g_LAMP.as_default() as g:
        prob = problems.read_prob_from_npz(prob_name)
        layers_LAMP  = networks.build_LAMP(prob, T=6, shrink='bg', untied=False)
        _, nmse_LAMP = train.test(g, model_LAMP, layers_LAMP, prob)

    _, nmse_ISTA  = ISTA(prob.A, prob.xval, prob.yval)
    _, nmse_FISTA = FISTA(prob.A, prob.xval, prob.yval)
    _, nmse_AMP   = AMP(prob.A, prob.xval, prob.yval)

    # draw nmse plots
    t_ISTA  = np.arange(10000+1)
    t_FISTA = np.arange(1000+1)
    t_AMP   = np.arange(100+1)
    t_LISTA = np.arange(6+1)
    t_LAMP  = np.arange(6+1)

    plt.semilogx(
            t_ISTA,  10*np.log10(nmse_ISTA),
            t_FISTA, 10*np.log10(nmse_FISTA),
            t_AMP,   10*np.log10(nmse_AMP),
            t_LISTA, 10*np.log10(nmse_LISTA),
            t_LAMP,  10*np.log10(nmse_LAMP)
            )
#     plt.legend(
#             'ISTA',
#             'FISTA',
#             'AMP',
#             'LISTA',
#             'LAMP'
#             )

    if os.path.exists("plots") == False:
        os.mkdir("plots")

    plt.savefig('plots/nmse_k{0:04.1f}_{1}.jpg'.format(kappa, exp_id))

