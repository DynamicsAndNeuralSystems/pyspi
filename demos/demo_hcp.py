# Import our classes
import os
from pynats.data import Data

from pynats.container import CalculatorFrame
from pynats.calculator import Calculator

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import dill 

# a) Setup time-series configuration
rsdat = loadmat( os.path.dirname(__file__) + '/data/hcp/hcp_rsfMRI.mat')
netdat = loadmat( os.path.dirname(__file__) + '/data/hcp/networks.mat')

S = 5
T = 250

tsm0 = rsdat['dat'][:S,:T,0]
tsm1 = rsdat['dat'][:S,:T,1]
nets = np.squeeze(netdat['id1plus'][:S])

print('Loaded HCP data as a {} {}'.format(tsm0.shape,type(tsm0)))
print('Loaded network info as a {} {}'.format(nets.shape,type(nets)))

# c) Load the data
calc0 = Calculator(dataset=Data(tsm0, dim_order='ps'),name='HCP0')
calc1 = Calculator(dataset=Data(tsm1, dim_order='ps'),name='HCP0')

cf = CalculatorFrame()
cf.add_calculator(calc0)
cf.add_calculator(calc1)

# Compute all adjacency matrices
cf.compute()

# Prune special values
cf.prune()

savefile = os.path.dirname(__file__) + '/pynats_hcp.pkl'
print('Saving object to dill database: "{}"'.format(savefile))

with open(savefile, 'wb') as f:
    dill.dump(cf, f)

print('Done.')

cf.clustermap(which_measure='all',sa_plot=True)

# calc.clustermap(4,categories=nets,linewidth=0.001)

plt.show()