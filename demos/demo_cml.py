# Import our classes
from pynats.data import Data
from pynats.container import CalculatorFrame
from pynats.calculator import Calculator
import pynats.plot as natplt

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

from pynats.lib.pyCML import cml, maps, couplings, evolution

import dill

cmls = [None] * 7
Ts = [100,1000,12*1500,1000,25*2000,45*5000,1000]
td = [1,1,12,1,2000,5000,1,1]

# Frozen random patterns
cmls[0] = cml.CML(dim=100, coupling=couplings.TwoNeighbor(strength=0.2, map_obj=maps.KanekoLogistic(alpha=1.45)))

# Pattern selection with suppression of chaos
cmls[1] = cml.CML(dim=100, coupling=couplings.TwoNeighbor(strength=0.4, map_obj=maps.KanekoLogistic(alpha=1.71)))

# Spatiotemporal intermittency
cmls[2] = cml.CML(dim=200, coupling=couplings.TwoNeighbor(strength=0.00115, map_obj=maps.KanekoLogistic(alpha=1.7522)))
cmls[3] = cml.CML(dim=200, coupling=couplings.TwoNeighbor(strength=0.3, map_obj=maps.KanekoLogistic(alpha=1.75)))

# Traveling wave
cmls[4] = cml.CML(dim=50, coupling=couplings.TwoNeighbor(strength=0.6, map_obj=maps.KanekoLogistic(alpha=1.47)))

# Chaotic traveling wave
cmls[5] = cml.CML(dim=50, coupling=couplings.TwoNeighbor(strength=0.5, map_obj=maps.KanekoLogistic(alpha=1.69)))

# Fully developed spatiotemporal chaos
cmls[6] = cml.CML(dim=100, coupling=couplings.TwoNeighbor(strength=0.3, map_obj=maps.KanekoLogistic(alpha=2.00)))

names = ['Frozen random patterns',
            'Pattern selection',
            'Spatiotemporal intermittency I',
            'Spatiotemporal intermittency II',
            'Traveling wave',
            'Chaotic traveling wave',
            'Spatiotemporal chaos']

datasets = []
for i in range(len(cmls)):
    ev = evolution.Evolution(cmls[i])
    dat = ev.time_evolution(iterations=Ts[i])
    dat = dat[::td[i],:]
    datasets.append(Data(dat,dim_order='sp',name=names[i],normalise=False))

cf = CalculatorFrame(datasets=datasets,names=names)

cf.plot_data()

# Compute all adjacency matrices
cf.compute()

# Prune special values
cf.prune()

# Plot some results
cf.clustermap(which_measure='all',sa_plot=True,cmap='PiYG')
cf.clusterall(approach='mean',cmap='PiYG')

natplt.statespace(cf)

plt.show()