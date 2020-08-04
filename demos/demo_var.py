# Import our classes
from pynats.data import Data
from pynats.ptsa import ptsa

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# TODO: change to generate_var_data

# a) Setup time-series configuration
M = 20
T = 5000
R = 1

clm_adj = np.triu(np.random.rand(M,M)).reshape((1,M,M))
clm_adj[np.nonzero(clm_adj < 0.5)] = 0

print('CLM:', clm_adj)

# c) Load the data
data = Data()
data.generate_var_data(n_samples=T,
                        n_replications=R,
                        coefficient_matrices=clm_adj)

calc = ptsa()

calc.load(data)

calc.compute()

calc.prune()

calc.diagnostics()

calc.clustermap('all')
calc.heatmaps(6)
calc.flatten(normalize=True)
plt.show()