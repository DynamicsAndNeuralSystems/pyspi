import os
from pynats.calculator import Calculator
from pynats.data import Data
import pynats.plot as natplt
import matplotlib.pyplot as plt

datafile = os.path.dirname(os.path.abspath(__file__)) + '/sinusoid.npy'

calc = Calculator(dataset=Data(data=datafile, dim_order='ps',name='sinusoid'))

calc.compute()
calc.prune()

natplt.clustermap(calc)

plt.show()