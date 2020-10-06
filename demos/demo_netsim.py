# Import our classes
import os
from pynats.data import Data
from pynats.btsa import btsa

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import dill

# a) Setup time-series configuration
sims = loadmat( os.path.dirname(__file__) + '/data/netsim/sim5.mat')
print('Loaded netsim simulation as a {} {}'.format(np.shape(sims['ts']),type(sims['ts'])))

# b) Load the data
data = Data(sims['ts'][:1000], dim_order='sp')

calc = btsa()

# c) Load the MVTS and run the calcs
calc.load(data)
calc.compute()

# d) Prune output (make sure all calcs were valid)
calc.prune()

savefile = os.path.dirname(__file__) + '/pynats_netsim.pkl'
print('Saving object to dill database: "{}"'.format(savefile))

with open(savefile, 'wb') as f:
    dill.dump(calc, f)

print('Done.')

# e) Pretty visualisations
calc.heatmaps(6)
calc.flatten()
calc.clustermap('all',linewidth=0.005)

plt.show()