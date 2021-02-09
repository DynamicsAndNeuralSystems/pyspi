# Load some of the packages
import numpy as np

from pynats.calculator import Calculator
from pynats.data import Data
import pynats.plot as natplt

import matplotlib.pyplot as plt

# Generate three random sequences (not time series yet)
M, T = 3, 100

procs = np.random.normal(size=(M,T))

def make_ar(x):
    for i in range(1,len(x)):
        x[i] += 0.5 * x[i-1]

make_ar(procs[0])
for i in range(1,M):
    make_ar(procs[i]) # Add autoregression
    procs[i][1:] += 0.25 * procs[i-1][:-1] # Add dependence on previous process

# Load the Data class

# Create an unnamed dataset with 3 processes and 1000 observations (1 replication)
#   - dim_order specifies processes is the first dimension and samples/observations are the second
#   - normalise z-scores the data
data = Data(procs, dim_order='ps', normalise=True)


natplt.plot_spacetime(data)

# Let's create an unnamed calculator
calc = Calculator(dataset=data)

calc.compute()

natplt.clustermap(calc)

plt.show()