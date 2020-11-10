import os
import glob
from pynats.data import Data
from pynats.container import CalculatorFrame
import pynats.plot as natplt

import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import RobustScaler, PowerTransformer
import seaborn as sns

reduced_measure_set = False

if reduced_measure_set:
    # savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_sinwave_reduced.pkl'
    savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_sinwave_reduced_M10.pkl'
    configfile = '/home/oliver/Workspace/code/research/pynats/pynats/reducedconfig.yaml'
    apx = '_M10'
else:
    # savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_sinwave_full.pkl'
    savefile = '/home/oliver/Workspace/code/research/pynats/demos/pynats_sinwave_full_M10.pkl'
    configfile = '/home/oliver/Workspace/code/research/pynats/pynats/config.yaml'
    apx = '_full_M10'

dataset_base = '/home/oliver/Workspace/code/research/mvts-database/'

try:
    with open(savefile, 'rb') as f:
        cf = dill.load(f)
except FileNotFoundError as err:

    match_str = 'linear/sinwave_M10_*.npy'
    npy_files = glob.glob(dataset_base + match_str)

    dim_order = 'ps'

    datasets = []
    names = []
    Tmax = 200
    nOffsets = 5
    for i, _file in enumerate(npy_files):
        npdat = np.load(_file)
        npdat = npdat[:,i:Tmax-i]
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

dataset_base = dataset_base + 'plots/sinwave/'

flatten_kwargs = {'transformer': PowerTransformer(), 'cmap': sns.color_palette("coolwarm", as_cmap=True)}
if reduced_measure_set:
    clustermap_kwargs = {'cmap': sns.color_palette("coolwarm", as_cmap=True), 'linewidth': 0.1, 'annot': True}
else:
    clustermap_kwargs = {'cmap': sns.color_palette("coolwarm", as_cmap=True)}

for cname in cf.calculators.index:
    calc = cf.calculators.loc[cname][0]

    _, fig = natplt.clustermap(calc, which_measure='all', plot_data=True, data_cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True), **clustermap_kwargs)
    fig.savefig(dataset_base + cname + '_clustermap' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_clustermap' + apx + '.png', bbox_inches="tight")

    _, fig = natplt.flatten(calc, **flatten_kwargs)
    fig.savefig(dataset_base + cname + '_flat' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_flat' + apx + '.png', bbox_inches="tight")

    _, fig = natplt.measurespace(cf,averaged=True, flatten_kwargs=flatten_kwargs)
    fig.savefig(dataset_base + cname + '_measurespace' + apx + '.pdf', bbox_inches="tight")
    fig.savefig(dataset_base + cname + '_measurespace' + apx + '.png', bbox_inches="tight")

_, fig = natplt.clusterall(cf,approach='mean', clustermap_kwargs=clustermap_kwargs)
fig.savefig(dataset_base + 'sinwave_clustermap' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'sinwave_clustermap' + apx + '.png', bbox_inches="tight")

_, fig = natplt.measurespace(cf, averaged=False, flatten_kwargs=flatten_kwargs, )
fig.savefig(dataset_base + 'sinwave_measurespace' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'sinwave_measurespace' + apx + '.png', bbox_inches="tight")

_, fig = natplt.measurespace(cf, averaged=True, flatten_kwargs=flatten_kwargs)
fig.savefig(dataset_base + 'sinwave_measurespace-avg' + apx + '.pdf', bbox_inches="tight")
fig.savefig(dataset_base + 'sinwave_measurespace-avg' + apx + '.png', bbox_inches="tight")