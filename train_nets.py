#-*- coding: utf-8 -*-

"""
train_nets.py
author: chernxh@tamu.edu
data  : 09/20/2017

This file is written with repect to https://github.com/mborgerding/onsager_deep_learning/LAMP.py
to read problems from './probs/' and to train LISTA, LAMP, LVAMP networks with different
    - conditional numbers (kappa)
    - shrinkage functions
and to save models in './trained/{LISTA, LAMP, LVAMP}_{shink}_k{kappa}.npz'
"""

from __future__ import division
from __future__ import print_function

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!!!

import numpy as np
import tensorflow as tf



