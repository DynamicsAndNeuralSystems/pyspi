# Science/maths tools
import numpy as np
from scipy import stats as stats
from scipy import signal as sig
from collections import Counter

# Plotting tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# From this package
from .data import Data
from .jidt import jidt

class pnats():
    """Pairwise network analysis of time series
    """
    
    # Initializer / Instance Attributes
    def __init__(self):
        self.data = None
        self.features = None
        self._feature_objs = [
            self._pearsonr,
            self._spearmanr,
            self._kendalltau,
            self._coherence,
            self._te,
            self._mi
        ]
        self.feature_strs = [
            "Pearson's r",
            "Spearman's r",
            "Kendall's tau",
            "Coherence",
            "Transfer entropy (kernel)",
            "Mutual information (kernel)"
        ]
        self.vlims = [
            (-1,1),
            (-1,1),
            (-1,1),
            (0,1),
            (0,None),
            (0,None)
        ]

        self._jidt_calc = jidt()
        self._nfeatures = len(self._feature_objs)
        print("Number of features: {}".format(self._nfeatures))

    def load(self,data):
        self.data = data
        self.features = np.empty((self.data.n_processes,self.data.n_processes,
            self._nfeatures))
        self.features[:] = np.NaN

    def compute(self):
        """ Compute the dependency measures for all target processes
        """
        for i in range(self.data.n_processes):
            self.compute_univariate(i)
        
    def compute_univariate(self, target, sources='all'):
        """ Compute the dependency measures for a single target process
        """
        target_proc = self.data.data[target]
        if target_proc.ndim > 1:
            target_proc = target_proc.flatten()

        print("Computing {} features for target {}".format(self.features.shape[0],target))
        for source in [x for x in range(self.data.n_processes) if x != target]:
                source_proc = self.data.data[source]
                if source_proc.ndim > 1:
                    source_proc = source_proc.flatten()
                self.features[source,target,:] = self._compute_features(source_proc,target_proc)

    # TODO: only use the top nfeatures features
    def heatmap(self,ncols=3,nfeatures=None):
        if nfeatures is None:
            nfeatures = self._nfeatures 
        nrows = int(np.ceil(nfeatures / ncols))

        fig, axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,squeeze=False)
            
        for f in range(nfeatures):
            # Where am I in the grid?
            ccol = f % ncols
            crow = int(f / ncols)
            myax = axs[crow,ccol]

            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Ensures colorbar and plot are same height:
            divider = make_axes_locatable(myax)
            cax = divider.append_axes("right", size="5%", pad=0.1)    

            mask = np.zeros_like(self.features[:,:,f])

            vmin = self.vlims[f][0]
            vmax = self.vlims[f][1]
            if (vmin != None) & (vmax != None):
                sns.heatmap(self.features[:,:,f],
                            ax=myax, cbar_ax=cax,
                            square=True, linewidths=.01,
                            mask=mask, cmap=cmap,
                            vmin=vmin, vmax=vmax)
            else:
                sns.heatmap(self.features[:,:,f],
                        ax=myax, cbar_ax=cax,
                        square=True, linewidths=.01,
                        mask=mask)
            myax.set_title(self.feature_strs[f])

        # Make remaining subplots empty:
        for ax in axs[-1,(nfeatures % ncols):]:
            ax.axis('off')

        plt.tight_layout()

    # TODO: only use the top nfeatures features
    def flatten(self,nfeatures=None,split=False):
        if nfeatures is None:
            nfeatures = self._nfeatures 

        iu = np.triu_indices(self.data.n_processes,1)
        il = np.tril_indices(self.data.n_processes,-1)

        feature_arr_up = np.empty((len(iu[0]),nfeatures))
        feature_arr_down = np.empty((len(iu[0]),nfeatures))
        for f in range(nfeatures):
            arr = self.features[:,:,f]
            feature_arr_up[:,f] = arr[iu]
            feature_arr_down[:,f] = arr[il]

        feature_up_norm = stats.zscore(feature_arr_up, axis=None)
        feature_down_norm = stats.zscore(feature_arr_down, axis=None)
        fig, ax = plt.subplots(2,1)
        sns.heatmap(feature_up_norm, ax=ax[0], vmin=-3, vmax=3)
        ax[0].set_title("Upper triangle")
        sns.heatmap(feature_down_norm, ax=ax[1], vmin=-3, vmax=3)
        ax[1].set_title("Lower triangle")
        plt.tight_layout()

        if split is True:
            splits = np.cumsum(np.unique(iu[0],return_counts=True)[1])
            hl = ax[1].hlines(splits, *ax[1].get_xlim(),colors='w')
            hl.set_linewidth(0.8)
            hl = ax[0].hlines(splits, *ax[0].get_xlim(),colors='w')
            hl.set_linewidth(0.8)

    def _compute_features(self,x,y):
        featurevec = np.zeros((len(self._feature_objs),))
        for f in range(len(self._feature_objs)):
            featurevec[f] = self._feature_objs[f](x,y)
        return featurevec

    def _pearsonr(self,x,y):
        return stats.pearsonr(x,y)[0]
    
    def _spearmanr(self,x,y):
        return stats.spearmanr(x,y).correlation
    
    def _kendalltau(self,x,y):
        return stats.kendalltau(x,y).correlation

    def _coherence(self,x,y,hz=(0.009,0.08)):
        f, Cxy = sig.coherence(x,y)
        return np.mean([Cxy[i] for i in range(len(f)) if f[i] >= hz[0] and f[i] <= hz[1] ])

    def _te(self,x,y):
        return self._jidt_calc.te(x,y)

    def _mi(self,x,y):
        return self._jidt_calc.mi(x,y)