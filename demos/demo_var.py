# Import our classes
from pynats.data import Data
from pynats.container import CalculatorFrame
from pynats.calculator import Calculator
import pynats.plot as natplt

import matplotlib.pyplot as plt
import numpy as np

import os
import random

# TODO: change to generate_var_data

# a) Setup time-series configuration
M = 5
T = 250
R = 1

armat = np.zeros((M,M))
armat[0,4] = .2
armat[0,1] = .5
armat[1,2] = .4
armat[2,3] = .6
armat[3,4] = .23
armat[1,0] = .2
armat[2,1] = .8



print('Autoregressive matrix:', armat)

random.seed(a=None, version=2)

# c) Load the data
dataset = Data(Data.generate_var_data(n_observations=T,
                                        n_realisations=1,
                                        coefficient_matrices=np.reshape(armat,(1,M,M))))

configfile = '/home/oliver/Workspace/code/research/pynats/pynats/reducedconfig.yaml'
calc = Calculator(dataset=dataset,configfile=configfile)

# Compute all adjacency matrices
calc.compute()

# Prune special values
calc.prune()

# Check speed of computations, etc.
natplt.diagnostics(calc)

# Compare to ground-truth AR params
# natplt.truth(calc,armat)

# Plot results
natplt.clustermap(calc,which_measure='all',plot_data=True)

natplt.clusterall(calc,approach='reduction')

plt.show()