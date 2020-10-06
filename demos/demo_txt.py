# Import our classes
import os
from pynats.data import Data
from pynats.btsa import btsa

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# a) Setup time-series configuration
sol = np.loadtxt( os.path.dirname(__file__) + '/data/ecosystem.csv.gz',delimiter=',')
sol = sol[:,:500]

# c) Load the data
data = Data(sol, dim_order='ps')

calc = btsa()

calc.load(data)

calc.compute()

calc.prune()

calc.diagnostics()

calc.heatmaps(6)
calc.flatten()
calc.clustermap('all')
plt.show()