"""
exp.py

author: chernxh@tamu.edu
date  : 09/20/2017

Experiment with AMP, VAMP, LAMP-l1 and replace AMP's shrinkage function with functions mentioned in
the paper other than soft-thresholding.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUITE!!!

import numpy as np
import tensorflow as tf
from matplotlib import pyplot



