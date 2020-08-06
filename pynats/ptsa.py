# Science/maths/computing tools
import numpy as np
import pandas as pd
from scipy import stats
import yaml
import importlib
import math
import time
import multiprocessing

import warnings

# Plotting tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from tqdm import tqdm
from tqdm import trange

# From this package
from .data import Data
from . import pynats_utils as utils

class ptsa():
    """Pairwise time series analysis
    """
    
    # Initializer / Instance Attributes
    def __init__(self,configfile="./pynats/config.yaml"):
        self.data = Data()
        self.adjacency = None

        self._load_yaml(configfile)
        self._nmeasures = len(self._measures)
        self._nclasses = len(self._classes)
        self._proctimes = np.empty(self._nmeasures)

        self._cmap = 'vlag'
        print("Number of pairwise measures: {}".format(self._nmeasures))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,value):
        if not isinstance(value,Data):
            raise TypeError('Data type must be pypynats.data.Data')
        self._data = value

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self,value):
        if (value is not None) and (value is not isinstance(value,np.ndarray) ):
            raise TypeError('Adjacency type must be None or numpy.ndarray. Type: {}'.format(type(value)))
        self._data = value

    def _load_yaml(self,document):
        print("Loading configuration file: {}".format(document))
        self._classes = []
        self._measures = []

        self._class_names = []
        self._measure_names = []
        with open(document) as f:
            yf = yaml.load(f,Loader=yaml.FullLoader)

            # Instantiate the calc classes 
            for module_name in yf:
                print("*** Importing module {}".format(module_name))
                classes = yf[module_name]
                module = importlib.import_module(module_name,__package__)

                for class_name in classes:
                    paramlist = classes[class_name]

                    self._classes.append(getattr(module, class_name))
                    self._class_names.append(class_name)
                    if paramlist is not None:
                        for params in paramlist:
                            print("[{}] Adding measure {}.{}(x,y,{})...".format(len(self._measures),module_name,class_name,params))
                            self._measures.append(self._classes[-1](**params))
                            self._measure_names.append(self._measures[-1].name)
                            print('Succesfully initialised as {}'.format(self._measures[-1].name))
                    else:
                        print("[{}] Adding measure {}.{}(x,y)...".format(len(self._measures),module_name,class_name))
                        self._measures.append(self._classes[-1]())
                        self._measure_names.append(self._measures[-1].name)
                        print('Succesfully initialised as {}'.format(self._measures[-1].name))

    def load(self,data):
        self.data = data
        self._adjacency = np.empty((self._data.n_processes,self._data.n_processes,
                                    self._nmeasures))
        self._adjacency[:] = np.NaN

    def compute(self):
        """ Compute the dependency measures for all target processes
        """
        dat = np.squeeze(self._data.data)

        pbar = tqdm(range(self._nclasses))
        for i in pbar:
            pbar.set_description('Pre-processing [{}]'.format(self._class_names[i]))
            self._classes[i].preprocess(dat)

        pbar = tqdm(range(self._nmeasures))
        for i in pbar:
            pbar.set_description('Processing [{}]'.format(self._measure_names[i]))
            start_time = time.time()
            self._adjacency[:,:,i] = self._measures[i].getadj(dat)
            self._proctimes[i] = time.time() - start_time
        pbar.close()

    def prune(self,meas_nans=0.1,proc_nans=0.9):
        """Prune the bad processes/measures
        """
        print('Pruning:\n\t- Processes with more than {}% bad values'
                ', and\n\t- Measures with more than {}% bad values'
                ''.format(100*meas_nans,100*proc_nans))

        # First, iterate through the time-series and remove any that have NaN's > ts_nans
        M = self._nmeasures * (2*(self._data.n_processes-1))
        threshold = M * proc_nans
        rm_list = []
        for proc in range(self._data.n_processes):

            other_procs = [i for i in range(self._data.n_processes) if i != proc]

            flat_adj = self._adjacency[other_procs,proc,:].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,self._adjacency[proc,other_procs,:].reshape((M//2,1))))

            nzs = np.count_nonzero(np.isnan(flat_adj))
            if nzs > threshold:
                print('Removing process {} with {} ({:.1f}%) special characters.'
                        ''.format(proc, nzs, 100*nzs/M ) )
                rm_list.append(proc)

        # Remove from the data object
        self._data.remove_process(rm_list)

        # Remove from the adjacency matrix (should probs move this to an attribute that cannot be set)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=0)
        self._adjacency = np.delete(self._adjacency,rm_list,axis=1)

        # Then, iterate through the measures and remove any that have NaN's > meas_nans
        M = self._data.n_processes ** 2 - self._data.n_processes
        threshold = M * meas_nans
        il = np.tril_indices(self._data.n_processes,-1)

        rm_list = []
        for meas in range(self._nmeasures):

            flat_adj = self._adjacency[il[1],il[0],meas].reshape((M//2,1))
            flat_adj = np.concatenate((flat_adj,
                                        self._adjacency[il[0],il[1],meas].reshape((M//2,1))))

            nzs = np.count_nonzero(np.isnan(flat_adj))
            if nzs > threshold:
                rm_list.append(meas)
                print('Removing measure "[{}] {}" with {} ({:.1f}%) '
                        'NaNs (max is {} [{}%])'.format(meas, self._measure_names[meas],
                                                        nzs,100*nzs/M, threshold, 100*meas_nans))

        # Remove from the adjacency and process times matrix
        self._adjacency = np.delete(self._adjacency,rm_list,axis=2)
        self._proctimes = np.delete(self._proctimes,rm_list,axis=0)

        # Remove from the measure lists (move to a method and protect measure)
        for meas in sorted(rm_list,reverse=True):
            del self._measures[meas]
            del self._measure_names[meas]

        self._nmeasures = len(self._measures)
        print('Number of pairwise measures after pruning: {}'.format(self._nmeasures))

    def threshold(self,pvalue):
        """Threshold the pairwise matrices using the p-values
            (should a separate function return those p-values?)
        """
        pass

    def diagnostics(self):
        """ TODO: print out all diagnostics, e.g., compute time, failures, etc.
        """
        sid = np.argsort(self._proctimes)
        print('Processing times for all {} measures:'.format(len(sid)))
        for i in sid:
            print('[{}] {}: {} s'.format(i,self._measure_names[i],self._proctimes[i]))

    # TODO: only use the top nmeasures features
    def heatmaps(self,ncols=5,nmeasures=None,split=False):
        if nmeasures is None:
            nmeasures = self._nmeasures
        nrows = int(np.ceil(nmeasures / ncols))

        fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,squeeze=False)

        lw = 0.0
        if split is True:
            lw = 0.01

        for i in range(nmeasures):
            ccol = i % ncols
            crow = int(i / ncols)
            myax = axs[crow,ccol]

            # Ensures colorbar and plot are same height:
            divider = make_axes_locatable(myax)
            cax = divider.append_axes("right", size="5%", pad=0.1)    

            mask = np.zeros_like(self._adjacency[:,:,i])
            
            sns.heatmap(self._adjacency[:,:,i],
                        ax=myax, cbar_ax=cax,
                        square=True, linewidth=lw,
                        cmap=self._cmap, mask=mask, center=0.00)
            myax.set_title('[' + str(i) + '] ' + utils.strshort(self._measure_names[i],20))

        # Make remaining subplots empty:
        for ax in axs[-1,(nmeasures % ncols):]:
            ax.axis('off')

    def clustermap(self,which_measure,categories=None,**kwargs):

        if isinstance(which_measure,int):

            adj = self._adjacency[:,:,which_measure]
            np.fill_diagonal(adj, 0)
            adj = np.nan_to_num(adj)

            cat_colors = None
            if categories is not None:
                cats = pd.Series(categories)
                # Create a categorical palette to identify the networks
                category_pal = sns.color_palette('pastel',len(categories))
                category_lut = dict(zip(cats.unique(), category_pal))
                cat_colors = cats.map(category_lut).tolist()

            g = sns.clustermap(adj, cmap=self._cmap,
                                center=0.0, figsize=(7, 7),
                                col_colors=cat_colors, row_colors=cat_colors,**kwargs )

            ax = g.ax_heatmap
        elif which_measure == 'all':

            adjs = np.resize(self._adjacency,(self._adjacency.shape[0]**2,self._adjacency.shape[2]))
            df = pd.DataFrame(adjs,columns=utils.strshort(self._measure_names,25))
            
            df.fillna(0,inplace=True)
            corrs = df.corr(method='spearman')
            corrs.fillna(0,inplace=True)
            g = sns.clustermap(corrs, cmap=self._cmap,
                                center=0.0, figsize=(7, 7),
                                **kwargs, xticklabels=1, yticklabels=1 )
            ax = g.ax_heatmap

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # TODO: only use the top nmeasures features
    def flatten(self,nmeasures=None,split=False,normalize=True,cluster=True,row_cluster=False):
        if nmeasures is None:
            nmeasures = self._nmeasures 

        il = np.tril_indices(self._data.n_processes,-1)

        ytickl = [str(il[0][i]) + '->' + str(il[1][i]) for i in range(len(il[0]))]
        yticku = [str(il[1][i]) + '->' + str(il[0][i]) for i in range(len(il[0]))]

        yticks = [0 for i in range(2*len(il[0]))]
        yticks[slice(0,-1,2)] = yticku
        yticks[slice(1,len(yticks),2)] = ytickl

        yticks = yticks
        xticks = utils.strshort(self._measure_names,25)

        pwarr = np.empty((2*len(il[0]),nmeasures))
        for f in range(nmeasures):
            pwarr[slice(0,-1,2),f] = self._adjacency[il[1],il[0],f]
            pwarr[slice(1,len(yticks),2),f] = self._adjacency[il[0],il[1],f]

        if normalize is True:
            pwarr = stats.zscore(pwarr,axis=0,nan_policy='omit')

            # For the positive measures make sure we scale these properl
            for meas in [i for i in range(nmeasures) if self._measures[i].ispositive() is True]:
                pwarr[:,meas] = ( pwarr[:,meas] - np.nanmin(pwarr[:,meas]) ) / 2

        df = pd.DataFrame(pwarr, index=yticks, columns=xticks)

        df.columns.name = 'Pairwise measure'
        df.index.name = 'Processes'
        if cluster is True:
            df.fillna(0,inplace=True)
            try:
                g = sns.clustermap(df,
                                    cmap=self._cmap,
                                    center=0.0, row_cluster=row_cluster,
                                    vmin=-3.0, vmax=3.0,
                                    figsize=(7, 7), dendrogram_ratio=(.2, .2),
                                    xticklabels=1 )
                ax = g.ax_heatmap
            except FloatingPointError:
                warnings.warn('Disimilarity value returned NaN. Have you run .prune()?',RuntimeWarning)
                return
        else:
            fig, ax = plt.subplots(1,1)
            ax = sns.heatmap(df, ax=ax, cmap=self._cmap,
                                center=0.0,
                                xticklabels=1 )

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')