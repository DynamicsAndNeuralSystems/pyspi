# Import our classes
import os
from pynats.data import Data
from pynats.ptsa import ptsa

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import dill 

# a) Setup time-series configuration
rsdat = loadmat( os.path.dirname(__file__) + '/data/hcp/hcp_rsfMRI.mat')
netdat = loadmat( os.path.dirname(__file__) + '/data/hcp/networks.mat')

S = 50

tsm = rsdat['dat'][:S,:,1]
nets = np.squeeze(netdat['id1plus'][:S])

print('Loaded HCP data as a {} {}'.format(tsm.shape,type(tsm)))
print('Loaded network info as a {} {}'.format(nets.shape,type(nets)))

# c) Load the data
data = Data(tsm, dim_order='ps')

calc = ptsa()

calc.load(data)

calc.compute()

calc.prune()

savefile = os.path.dirname(__file__) + '/pynats_hcp.pkl'
print('Saving object to dill database: "{}"'.format(savefile))

with open(savefile, 'wb') as f:
    dill.dump(calc, f)

print('Done.')

calc.heatmaps(6)
calc.flatten()
calc.clustermap('all',linewidth=0.001)
calc.clustermap(4,categories=nets,linewidth=0.001)
plt.show()