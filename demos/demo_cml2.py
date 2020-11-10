import os
from pynats.data import Data
from pynats.container import CalculatorFrame
import pynats.plot as natplt

import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler, PowerTransformer

reduced_measure_set = False

if reduced_measure_set:
    savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_cml_reduced.pkl'
    configfile = '/home/oliver/Workspace/code/research/pynats/pynats/reducedconfig.yaml'
    apx = ''
else:
    savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_cml_full.pkl'
    configfile = '/home/oliver/Workspace/code/research/pynats/pynats/config.yaml'
    apx = '_full'

dataset_base = '/home/oliver/Workspace/code/research/mvts-database/'

try:
    with open(savefile, 'rb') as f:
        cf = dill.load(f)
except FileNotFoundError as err:

    npy_files = [dataset_base + 'coupled_map_lattice/frozen_random_patterns.npy',
                    dataset_base + 'coupled_map_lattice/pattern_selection.npy',
                    dataset_base + 'coupled_map_lattice/spatiotemporal_intermittency_i.npy',
                    dataset_base + 'coupled_map_lattice/spatiotemporal_intermittency_ii.npy',
                    dataset_base + 'coupled_map_lattice/travelling_wave.npy',
                    dataset_base + 'coupled_map_lattice/chaotic_travelling_wave.npy',
                    dataset_base + 'coupled_map_lattice/spatiotemporal_chaos.npy']

    dim_order = 'sp'

    datasets = []
    names = []
    Tmax = 10000
    MMax = 10000
    for i, _file in enumerate(npy_files):
        npdat = np.load(_file)
        npdat = npdat[:Tmax,:MMax]
        names.append(os.path.basename(_file)[:-4])
        datasets.append(Data(npdat,dim_order=dim_order,name=names[i],normalise=False))

        # natplt.plot_spacetime(datasets[-1],cluster=False)

    cf = CalculatorFrame(datasets=datasets,names=names,configfile=configfile)

    cf.compute()

    cf.prune()

    print('Saving object to dill database: "{}"'.format(savefile))
    with open(savefile, 'wb') as f:
        dill.dump(cf, f)

cf.prune(meas_nans=0)

dataset_base = dataset_base + 'plots/cml/'

for cname in cf.calculators.index:
    calc = cf.calculators.loc[cname][0]

    _, fig = natplt.clustermap(calc,which_measure='all',plot_data=True)
    fig.savefig(dataset_base + cname + '_clustermap' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_clustermap' + apx + '.png', bbox_inches="tight")

    _, fig = natplt.flatten(calc,transformer=PowerTransformer())
    fig.savefig(dataset_base + cname + '_flat' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_flat' + apx + '.png', bbox_inches="tight")

    _, fig = natplt.measurespace(cf,averaged=True,flatten_kwargs={'transformer': PowerTransformer()})
    fig.savefig(dataset_base + cname + '_measurespace' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_measurespace' + apx + '.png', bbox_inches="tight")

_, fig = natplt.clusterall(cf,approach='mean')
fig.savefig(dataset_base + 'cml_clustermap' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'cml_clustermap' + apx + '.png', bbox_inches="tight")

_, fig = natplt.measurespace(cf,averaged=False,flatten_kwargs={'transformer': PowerTransformer()})
fig.savefig(dataset_base + 'cml_measurespace' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'cml_measurespace' + apx + '.png', bbox_inches="tight")

_, fig = natplt.measurespace(cf,averaged=True,flatten_kwargs={'transformer': PowerTransformer()})
fig.savefig(dataset_base + 'cml_measurespace-avg' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'cml_measurespace-avg' + apx + '.png', bbox_inches="tight")