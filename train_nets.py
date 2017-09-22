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
parser.add_argument('--T', type=int, default=6, help='number of layers in the network')
parser.add_argument('--kappa', type=float, default=0.0, help='conditional number of matrix A')
parser.add_argument('--pnz', type=float, default=0.1, help='pnz')
parser.add_argument('--SNR', type=int, default=40, help='SNR')
parser.add_argument('--id', type=int, default=0, help='the number of this experiment')
parser.add_argument('--dest_models', type=str, default='models', help='destination folder of trained models')
parser.add_argument('--dest_probs', type=str, default='probs', help='destination folder for generated problems')
parser.add_argument('--model', type=str, default='LISTA', help='choose your learned model')
parser.add_argument('--shrink', type=str, default='bg', help='choose your shrinkage function')

if __name__ == "__main__":
    parsed, unparsed = parser.parse_known_args()

    if not os.path.exists(parsed.dest_models):
        os.mkdir(parsed.dest_models)
    if not os.path.exists(parsed.dest_probs):
        os.mkdir(parsed.dest_probs)

    prob = problems.bernoulli_gaussian_trial(
            M=parsed.M,
            N=parsed.N,
            L=parsed.L,
            pnz=parsed.pnz,
            kappa=parsed.kappa,
            SNR=parsed.SNR)
    # save problems
    base = parsed.dest_probs + "/problem_k{0:04.1f}_{1}".format(parsed.kappa, parsed.id)
    problems.save_problem(base, prob)

    if parsed.model == 'LVAMP':
        layers = networks.build_LVAMP(prob, T=parsed.T, shrink=parsed.shrink)
    elif parsed.model == 'LAMP':
        layers = networks.build_LAMP(prob, T=parsed.T, shrink=parsed.shrink, untied=False)
    elif parsed.model == 'LISTA':
        layers = networks.build_LISTA(prob, T=parsed.T, untied=False)
    else:
        raise ValueError('Wrong model designated')

    # plan the training
    training_stages = train.setup_training(layers, prob, trinit=1e-3, refinements=(.5,.1,.01))

    # get saved model file name as 'models/LAMP_bg_k05.1_2.npz'
    model_name = parsed.dest_models + '/' + parsed.model + '_' + parsed.shrink + \
                    '_k{0:04.1f}_{1}.npz'.format(parsed.kappa, parsed.id)
    # do the learning (takes a while)
    sess = train.do_training(training_stages, prob, model_name)

