# Import our classes
from pynats.data import Data
from pynats.ptsa import ptsa

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# a) Setup time-series configuration
T = 5000
R = 1
M = 5

clm_adj = np.triu(np.random.rand(M,M)).reshape((1,M,M))
clm_adj[np.nonzero(clm_adj < 0.5)] = 0

print('CLM:', clm_adj)

# c) Load the data
data = Data()
data.generate_logistic_maps_data(n_samples=T,
                                    n_replications=R,
                                    coefficient_matrices=clm_adj)

calc = ptsa()

calc.load(data)

calc.compute()

calc.prune()

calc.heatmaps(6)
calc.flatten()
calc.clustermap('all')
plt.show()