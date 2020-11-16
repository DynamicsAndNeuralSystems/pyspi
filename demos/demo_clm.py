# Import our classes
from pynats.data import Data
from pynats.calculator import Calculator

import dill
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# a) Setup time-series configuration
M = 5
T = 250
R = 1

clm_adj = np.zeros((M,M))
clm_adj[0,1] = 1.0
clm_adj = clm_adj.reshape((1,M,M))

print('CLM:', clm_adj)

# c) Load the data
data = Data()
data.generate_logistic_maps_data(n_samples=T,
                                    n_replications=R,
                                    coefficient_matrices=clm_adj)

calc = btsa()

calc.load(data)

calc.compute()

calc.prune()

savefile = os.path.dirname(__file__) + '/pynats_clm.pkl'
print('Saving object to dill database: "{}"'.format(savefile))

with open(savefile, 'wb') as f:
    dill.dump(calc, f)

calc.diagnostics()

# calc.truth(clm_adj[0])

# calc.heatmaps(6)
calc.flatten()
calc.clustermap('all',sa_plot=True)
plt.show()