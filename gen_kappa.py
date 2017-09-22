#!/usr/bin/python
from tools import problems

import numpy as np
import numpy.linalg as la
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

# np.random.seed(1) # numpy is good about making repeatable output
# tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
from tools import problems,networks,train

from scipy.io import savemat

if __name__ == "__main__":
    if not os.path.exists("probs"):
        os.mkdir("probs")
    start = 4.0
    end   = 15.0
    step  = 0.5
    for kappa in np.arange(start, end+step, step):
        for i in range(5):
            base = "probs/" + "problem_k{0:04.1f}_{1}".format(kappa, i)
            problems.save_problem(base,problems.bernoulli_gaussian_trial(M=50,N=100,L=500,pnz=.1,kappa=kappa,SNR=40))
# save_problem('problem_k5',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=5,SNR=40))
# save_problem('problem_k15',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=15,SNR=40))
# save_problem('problem_k100',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=100,SNR=40))
# save_problem('problem_k1000',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=1000,SNR=40))
# save_problem('problem_rap1',problems.random_access_problem(1))
# save_problem('problem_rap2',problems.random_access_problem(2))

# save_problem('problem_k100',problems.bernoulli_gaussian_trial(M=250,N=500,L=1000,pnz=.1,kappa=100,SNR=40))
