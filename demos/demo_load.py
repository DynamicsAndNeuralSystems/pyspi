import os

import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

input_file = open('demos/pynats_hcp.pkl', 'rb')
# input_file = open('demos/pynats_netsim.pkl', 'rb')

calc = dill.load(input_file)

netdat = loadmat( os.path.dirname(__file__) + '/data/hcp/networks.mat')
nets = np.squeeze(netdat['id1plus'][:calc.data.n_processes])

calc.heatmaps(6)
calc.clustermap('all',linewidth=0.005)
calc.clustermap(4,categories=nets,linewidth=0.005) # Ledoit Wolf Corr
# calc.clustermap(6,categories=nets,linewidth=0.005) # Partial Corr
# calc.clustermap(14,categories=nets,linewidth=0.005) # MI Kraskov
plt.show()