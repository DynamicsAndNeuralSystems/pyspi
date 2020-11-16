import os
from pynats.data import Data
from pynats.container import CalculatorFrame
import pynats.plot as natplt

import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pynats.lib.ScientificColourMaps6 as SCM6

savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_cml.pkl'
try:
    with open(savefile, 'rb') as f:
        dill.load(f)
except FileNotFoundError as err:

    dataset_base = '/home/oliver/Workspace/code/research/mvts-database/'

    use_oscillators = False

    if use_oscillators:
        npy_files = [dataset_base + 'oscillators/kuramoto_w1.npy',
                        dataset_base + 'oscillators/kuramoto_w6.npy',
                        dataset_base + 'oscillators/kuramoto_w12.npy']

        names = ['Kuramoto (K=1)', 'Kuramoto (K=6)', 'Kuramoto (K=12)']
        dim_order = 'ps'
    else:
        npy_files = [dataset_base + 'coupled_map_lattice/frozen_random_patterns.npy',
                        dataset_base + 'coupled_map_lattice/pattern_selection.npy',
                        dataset_base + 'coupled_map_lattice/spatiotemporal_intermittency_i.npy',
                        dataset_base + 'coupled_map_lattice/spatiotemporal_intermittency_ii.npy',
                        dataset_base + 'coupled_map_lattice/travelling_wave.npy',
                        dataset_base + 'coupled_map_lattice/chaotic_travelling_wave.npy',
                        dataset_base + 'coupled_map_lattice/spatiotemporal_chaos.npy']

        names = ['Frozen random patterns',
                'Pattern selection',
                'Spatiotemporal intermittency I',
                'Spatiotemporal intermittency II',
                'Traveling wave',
                'Chaotic traveling wave',
                'Spatiotemporal chaos']

        dim_order = 'sp'

    Tmax = 250
    Mmax = 10
    datasets = []
    for i, _file in enumerate(npy_files):
        npdat = np.load(_file)
        if use_oscillators:
            npdat = npdat[:Mmax,:Tmax]
        else:
            npdat = npdat[:Tmax,:Mmax]
        datasets.append(Data(npdat,dim_order=dim_order,name=names[i],normalise=False))

    cf = CalculatorFrame(datasets=datasets,names=names)

    cf.plot_data(cluster=False)

    cf.compute()

    cf.prune()

    print('Saving object to dill database: "{}"'.format(savefile))
    with open(savefile, 'wb') as f:
        dill.dump(cf, f)

cf.clustermap(which_measure='all',plot_data=True,cmap='PiYG')
cf.clusterall(approach='mean',cmap='PiYG')

natplt.statespace(cf)

plt.show()
