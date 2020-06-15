# Import our classes
from nats.data import Data
from nats.pnats import pnats

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# a) Setup time-series configuration
m = 25
T = 250
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag

# b) Generate test data
y = np.zeros((m,T))
for i in range(m):
    y[i] = arma.arma_generate_sample(ar, ma, T)

# c) Load the data
data = Data(y, dim_order='ps')

calc = pnats()

calc.load(data)

calc.compute()

calc.heatmap(3)
calc.flatten()
plt.show()