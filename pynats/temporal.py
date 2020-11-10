from statsmodels.tsa.stattools import coint as ci
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pyEDM as edm
import pandas as pd
from math import isnan
from hyppo.time_series import MGCX, DcorrX
import warnings
from pynats.base import directed, undirected, parse, positive, real
from collections import namedtuple

class coint(directed,real):
    
    humanname = "Cointegration"
    name = "coint"
    cache = namedtuple('cache','max_eig_stat trace_stat tstat pvalue')

    def __init__(self,method='johansen',statistic='pvalue'):
        self._method = method
        self._statistic = statistic
        self.name = self.name + '_' + method + '_' + statistic

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse
    def bivariate(self,data,i=None,j=None,verbose=False):

        z = data.to_numpy(squeeze=True)
        M = data.n_processes
        nullmat = np.empty((M,M))
        nullmat[:] = np.NaN

        if not hasattr(data,'coint'):
            data.coint = coint.cache(nullmat,nullmat,nullmat,nullmat)

        if self._method == 'johansen':
            if isnan(data.coint.max_eig_stat[i,j]):
                z_ij_T = np.transpose(z[[i,j],:])
                stats = coint_johansen(z_ij_T,det_order=1,k_ar_diff=10)
                data.coint.max_eig_stat[[i,j],[j,i]] = stats.max_eig_stat
                data.coint.trace_stat[[i,j],[j,i]] = stats.trace_stat
        if self._method == 'aeg':
            stats = ci(z[i,:],z[j,:])
            data.coint.tstat[i,j] = stats[0]
            data.coint.pvalue[i,j] = stats[1]

        if self._statistic == 'tstat':
            return -data.coint.tstat[i,j], data
        elif self._statistic == 'pvalue':
            return 1-data.coint.pvalue[i,j], data
        elif self._statistic == 'max_eig_stat':
            return data.coint.max_eig_stat[i,j], data
        elif self._statistic == 'trace_stat':
            return data.coint.trace_stat[i,j], data

class ccm(directed,real):

    humanname = "Convergent cross-maping"
    name = "ccm"
    cache = namedtuple('cache','embedding score')

    def __init__(self,statistic='mean'):
        self._statistic = statistic
        self.name = self.name + '_' + statistic

    @parse
    def bivariate(self,data,i=None,j=None):
        if not hasattr(data,'ccm'):
            z = data.to_numpy(squeeze=True)

            M = data.n_processes
            N = data.n_observations
            df = pd.DataFrame(range(0,N),columns=['index'])
            embedding = np.zeros((M,1))

            names = []

            # First pass: infer optimal embedding
            for j in range(M):
                names.append('var' + str(j))
                df[names[j]] = z[j,:]
                pred = str(10) + ' ' + str(N-10)
                embed_df = edm.EmbedDimension(dataFrame=df,lib=pred,
                                                pred=pred,columns=str(j),showPlot=False)
                embedding[j] = embed_df.iloc[embed_df.idxmax().rho,0]
            
            # Get some reasonable library lengths
            nlibs = 5
            E = int(max(embedding))
            upperE = int(np.floor((N-E-1)/10)*10)
            lowerE = int(np.ceil(2*E/10)*10)
            inc = int((upperE-lowerE) / nlibs)
            lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)

            # Second pass: compute CCM
            score = np.zeros((M,M,nlibs+1))
            for j in range(M):
                for i in range(j+1,M):
                    E = int(max(embedding[i],embedding[j]))
                    ccm_df = edm.CCM(dataFrame=df,E=E,columns=names[i],target=names[j],
                                        libSizes=lib_sizes,sample=100)
                    sc1 = ccm_df.iloc[:,1]
                    sc2 = ccm_df.iloc[:,2]
                    score[i,j,:] = np.array(sc1)
                    score[j,i,:] = np.array(sc2)

            data.ccm = ccm.cache(embedding=embedding, score=score)

        if self._statistic == 'mean':
            stat = np.nanmean(data.ccm.score[i,j])
        elif self._statistic == 'max':
            stat = np.nanmax(data.ccm.score[i,j])
        elif self._statistic == 'diff':
            stat = np.nanmax(data.ccm.score[i,j] - data.ccm.score[j,i])

        return stat, data

class dcorrx(undirected,positive):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "dcorrx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i,:]
        y = z[j,:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = DcorrX(max_lag=self._max_lag).test(x, y, reps=0 )
        return stat, data

class mgcx(undirected,positive):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "mgcx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    @parse
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i,:]
        y = z[j,:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGCX(max_lag=self._max_lag).test(x, y, reps=0)
        return stat, data