# Import our classes
from pynats.data import Data
from pynats.container import CalculatorFrame
from pynats.calculator import Calculator
import pynats.plot as natplt

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from pandas_datareader import data

import dill
import os
import random

savefile = os.path.dirname(__file__) + '/pynats_multiple.pkl'

try:
    with open(savefile,'rb') as f:
        cf = dill.load(f)
except FileNotFoundError as err:

    M = 5
    T = 200

    noise = np.random.normal(0,1,size=(M, T))

    noise_dat = Data(noise, dim_order='ps', name='White noise')

    M = 5
    T = 100
    R = 1

    armat = np.zeros((M,M))
    armat[0,4] = .2
    armat[0,1] = .5
    armat[1,2] = .4
    armat[2,3] = .6
    armat[3,4] = .23
    armat[1,0] = .2
    armat[2,1] = .8
    armat = armat.reshape((1,M,M))

    print('Autoregressive matrix:', armat)

    random.seed(a=None, version=2)

    """
        VAR data
    """
    var_dat0 = Data(Data.generate_var_data(n_observations=T,
                                            n_realisations=1,
                                            coefficient_matrices=armat), name='VAR0')

    var_dat1 = Data(Data.generate_var_data(n_observations=T,
                                            n_realisations=1,
                                            coefficient_matrices=np.moveaxis(armat,1,2)), name='VAR1')

    """
        CLM data
    """
    M = 3
    T = 150
    R = 1

    clm_adj = np.zeros((M,M))
    clm_adj[0,1] = 1.0
    clm_adj = clm_adj.reshape((1,M,M))
    clm_dat0 = Data(Data.generate_logistic_maps_data(n_observations=T,
                                                    n_realisations=1,
                                                    coefficient_matrices=clm_adj), name='CLM')


    clm_dat1 = Data(Data.generate_logistic_maps_data(n_observations=T,
                                                    n_realisations=1,
                                                    coefficient_matrices=np.moveaxis(clm_adj,1,2)), name='CLM')

    """
        MuTe data
    """
    T = 250
    R = 1
    mute_dat = Data(Data.generate_mute_data(n_observations=T, n_realisations=R), name='MuTe')

    """
        HCP data
    """
    rsdat = loadmat( os.path.dirname(__file__) + '/data/hcp/hcp_rsfMRI.mat')
    taskdat = loadmat( os.path.dirname(__file__) + '/data/hcp/hcp_tfMRI.mat')

    S = 10
    T = 200

    rsdat0 = rsdat['dat'][:S,:T,0]
    rsdat1 = rsdat['dat'][:S,:T,1]
    rs_dat0 = Data(rsdat0, dim_order='ps', name='rsfMRI s0')
    rs_dat1 = Data(rsdat1, dim_order='ps', name='rsfMRI s1')

    taskdat0 = taskdat['dat'][:S,:T,0]
    taskdat1 = taskdat['dat'][:S,:T,1]
    task_dat0 = Data(taskdat0, dim_order='ps', name='tfMRI s0')
    task_dat1 = Data(taskdat1, dim_order='ps', name='tfMRI s1')

    """
        NetSim data
    """
    T = 400
    sims = loadmat( os.path.dirname(__file__) + '/data/netsim/sim5.mat')

    netsim_dat = Data(sims['ts'][:T], dim_order='sp', name='NetSim 5M')

    faang_tickers = ['F','AMZN','AAPL','NFLX','GOOG']
    forex_tickers = ['DEXJPUS','DEXUSEU','DEXCHUS','DEXUSUK','DEXCAUS','DEXUSAL','DEXSZUS']

    start_date = '2015-01-01'
    end_date = '2019-12-31'

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    faang_panel_data = data.DataReader(faang_tickers, 'yahoo', start_date, end_date).fillna(method='ffill')
    forex_panel_data = data.DataReader(forex_tickers, 'fred', start_date, end_date).fillna(method='ffill')

    (fang_S,fang_P) = faang_panel_data['Close'].shape
    (forex_S,forex_P) = forex_panel_data.shape

    N = 200

    print('FAANG data has {} time series with {} observations each.'.format(fang_P,fang_S))
    faang_dat = np.zeros((N,fang_P))
    for i in range(fang_P):
        faang_ts = faang_panel_data['Close'][faang_tickers[i]].to_numpy()
        faang_dat[:,i] = faang_ts[-N:]

    print('Forex data has {} time series with {} observations each.'.format(forex_P,forex_S))
    forex_dat = np.zeros((N,forex_P))
    for i in range(forex_P):
        forex_ts = forex_panel_data[forex_tickers[i]].to_numpy()
        forex_dat[:,i] = forex_ts[-N:]
        
    fang_dat = Data(faang_dat, dim_order='sp', name='FAANG')
    forex_dat = Data(forex_dat, dim_order='sp', name='Forex')

    datasets = [noise_dat,var_dat0,var_dat1,clm_dat0,clm_dat1,mute_dat,rs_dat0,rs_dat1,task_dat0,task_dat1,netsim_dat,fang_dat,forex_dat]

    names = [ 'White noise',
                'VAR0', 'VAR1',
                'CLM0', 'CLM1',
                'MuTe',
                'rsfMRI s0', 'rsfMRI s1',
                'tfMRI s0', 'tfMRI s1',
                'NetSim',
                'FAANG', 'Forex' ]

    labels = [ ['noise','gaussian','artificial'],
                ['linear','autoregressive','artificial'],
                ['linear','autoregressive','artificial'],
                ['nonlinear','chaotic','artificial'],
                ['nonlinear','chaotic','artificial'],
                ['nonlinear','fmri','artificial'],
                ['nonlinear','fmri','rest','real'],
                ['nonlinear','fmri','rest','real'],
                ['nonlinear','fmri','task','real'],
                ['nonlinear','fmri','task','real'],
                ['nonlinear', 'fmri','artificial'],
                ['nonlinear','financial','real'],
                ['nonlinear','financial','real'] ]

    cf = CalculatorFrame(datasets=datasets,names=names,labels=labels)

    # Compute all adjacency matrices
    cf.compute()

    # Prune special values
    cf.prune()

    print('Saving object to dill database: "{}"'.format(savefile))
    with open(savefile, 'wb') as f:
        dill.dump(cf, f)

# Check speed of computations, etc.
natplt.diagnostics(cf.calculators.loc['White noise'][0])

# Compare to ground-truth AR params
# calc0.truth(armat[0])

# Plot results
cf.clustermap(which_measure='all',sa_plot=True,cmap='PiYG')
# calc0.heatmaps(6)
cf.clusterall(approach='mean',cmap='PiYG')

natplt.statespace(cf)

plt.show()