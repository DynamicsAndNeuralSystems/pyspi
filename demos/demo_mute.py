# Import our classes
from nats.data import Data
from nats.pnats import pnats

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.arima_process as arma

# a) Setup time-series configuration
T = 1000
R = 1

# c) Load the data
data = Data()
data.generate_mute_data(n_samples=T, n_replications=R)

calc = pnats()

calc.load(data)

calc.compute()

calc.heatmap(3)
calc.flatten()
plt.show()