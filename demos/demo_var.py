# Import our classes
from pynats.data import Data
from pynats.btsa import btsa

import matplotlib.pyplot as plt
import numpy as np

import dill
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
armat = armat.reshape((1,M,M))

print('Autoregressive matrix:', armat)

random.seed(a=None, version=2)

# c) Load the data
data = Data()
data.generate_var_data(n_samples=T,
                        n_replications=R,
                        coefficient_matrices=armat)

calc = btsa()

# Load the VAR dataset
calc.load(data)

# Compute all adjacency matrices
calc.compute()

# Prune special values
calc.prune()

savefile = os.path.dirname(__file__) + '/pynats_var.pkl'
print('Saving object to dill database: "{}"'.format(savefile))

with open(savefile, 'wb') as f:
    dill.dump(calc, f)

# Check speed of computations, etc.
calc.diagnostics()

# Compare to ground-truth AR params
calc.truth(armat[0])

# Plot results
calc.clustermap('all')
calc.heatmaps(6)
calc.flatten(normalize=True)
plt.show()