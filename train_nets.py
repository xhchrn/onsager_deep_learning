#-*- coding: utf-8 -*-

"""
train_nets.py
author: chernxh@tamu.edu
data  : 09/20/2017

This file is written with repect to https://github.com/mborgerding/onsager_deep_learning/LAMP.py
to
- generate problems and save to './probs/problem_k{kappa}_{id}.npz' file
- to train LISTA, LAMP, LVAMP networks with different
    - conditional numbers (kappa)
    - shrinkage functions
and to save models in './trained/{LISTA, LAMP, LVAMP}_{shink}_k{kappa}.npz'
"""

from __future__ import division
from __future__ import print_function

import argparse
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!!!

import numpy as np
import tensorflow as tf

# import problems, networks and training modules
from tools import problems, networks, train


parser = argparse.ArgumentParser()

parser.add_argument('--M', type=int, default=25, help='dimension of y')
parser.add_argument('--N', type=int, default=50, help='dimension of x')
parser.add_argument('--L', type=int, default=500, help='number of training samples')
parser.add_argument('--kappa', type=float, default=0.0, help='conditional number of matrix A')
parser.add_argument('--pnz', type=float, default=0.1, help='pnz')
parser.add_argument('--SNR', type=int, default=40, help='SNR')
parser.add_argument('--id', type=int, default=0, help='the number of this experiment')
parser.add_argument('--dest-models', type=str, default='trained', help='destination folder of trained models')
parser.add_argument('--dest-probs', type=str, default='probs', help='destination folder for generated problems')


if __name__ == "__main__":
    parsed, unparsed = parser.parse_known_args()

    if not os.path.exists(parsed.dest-models):
        os.mkdir(parsed.dest-models)
    if not os.path.exists(parsed.dest-probs):
        os.mkdir(parsed.dest-probs)

    prob = problems.bernoulli_gaussian_trial(
            M=parsed.M,
            N=parsed.N,
            L=parsed.L,
            pnz=parsed.pnz,
            kappa=parsed.kappa,
            SNR=parsed.SNR)
    # save problems
    base = parsed.dest-probs + "problem_k{0:04.1f}_{1}".format(parsed.kappa, parsed.id)
    problems.save_problem(base, prob)



