from statsmodels.tsa.stattools import coint as ci
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from . import base
import numpy as np
import pyEDM as edm
import pandas as pd
from math import isnan
from hyppo.time_series import MGCX, DcorrX
import warnings

class coint(base.directed):
    
    humanname = "Cointegration"
    name = "coint"
    _methods = []

    def __init__(self,method='johansen',statistic='pvalue'):
        self._method = method
        self._statistic = statistic
        self.name = self.name + '_' + method + '_' + statistic
        coint._methods.append(method)

    @staticmethod
    def preprocess(z):
        M = z.shape[0]
        z = np.transpose(z)

        if 'johansen' in coint._methods:
            coint._max_eig_stat = np.empty((M,M))
            coint._max_eig_stat[:] = np.NaN
            coint._trace_stat = np.empty((M,M))
            coint._trace_stat[:] = np.NaN
        if 'aeg' in coint._methods:
            coint._tstat = np.empty((M,M))
            coint._tstat[:] = np.NaN
            coint._pvalue = np.empty((M,M))
            coint._pvalue[:] = np.NaN
        for j in range(M):
            for i in [ii for ii in range(M) if ii != j]:
                if 'johansen' in coint._methods:
                    if isnan(coint._max_eig_stat[i,j]):
                        stats = coint_johansen(z[:,[i,j]],det_order=1,k_ar_diff=10)
                        coint._max_eig_stat[[i,j],[j,i]] = stats.max_eig_stat
                        coint._trace_stat[[i,j],[j,i]] = stats.trace_stat
                if 'aeg' in coint._methods:
                    stats = ci(z[:,i],z[:,j])
                    coint._tstat[i,j] = stats[0]
                    coint._pvalue[i,j] = stats[1]

    # Return the negative t-statistic (proxy for how co-integrated they are)
    def bivariate(self,x,y,i,j):
        if self._statistic == 'tstat':
            return -coint._tstat[i,j]
        elif self._statistic == 'pvalue':
            return 1-coint._pvalue[i,j]
        elif self._statistic == 'max_eig_stat':
            return coint._max_eig_stat[i,j]
        elif self._statistic == 'trace_stat':
            return coint._trace_stat[i,j]


class ccm(base.directed):

    humanname = "Convergent cross-maping"
    name = "ccm"

    def __init__(self,statistic='mean'):
        self._statistic = statistic
        self.name = self.name + '_' + statistic

    @staticmethod
    def preprocess(z):
        M = z.shape[0]
        N = z.shape[1]
        df = pd.DataFrame(range(0,N),columns=['index'])
        ccm._embedding = np.zeros((M,1))

        names = []

        # First pass: infer optimal embedding
        for j in range(M):
            names.append('var' + str(j))
            df[names[j]] = z[j,:]
            pred = str(10) + ' ' + str(N-10)
            embed_df = edm.EmbedDimension(dataFrame=df,lib=pred,
                                            pred=pred,columns=str(j),showPlot=False)
            ccm._embedding[j] = embed_df.iloc[embed_df.idxmax().rho,0]
        
        # Get some reasonable library lengths
        nlibs = 5
        E = int(max(ccm._embedding))
        upperE = int(np.floor((N-E-1)/10)*10)
        lowerE = int(np.ceil(E/10)*10)
        inc = int((upperE-lowerE) / nlibs)
        lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)

        # Second pass: compute CCM
        ccm._score = np.zeros((M,M,nlibs+1))
        for j in range(M):
            for i in range(j+1,M):
                E = int(max(ccm._embedding[i],ccm._embedding[j]))
                ccm_df = edm.CCM(dataFrame=df,E=E,columns=names[i],target=names[j],
                                    libSizes=lib_sizes,sample=100)
                sc1 = ccm_df.iloc[:,1]
                sc2 = ccm_df.iloc[:,2]
                ccm._score[i,j,:] = np.array(sc1)
                ccm._score[j,i,:] = np.array(sc2)


    def bivariate(self,x,y,i,j):
        if self._statistic == 'mean':
            return np.nanmean(ccm._score[i,j])
        elif self._statistic == 'max':
            return np.nanmax(ccm._score[i,j])
        elif self._statistic == 'diff':
            return np.nanmax(ccm._score[i,j] - ccm._score[j,i])

class dcorrx(base.undirected):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "dcorrx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    def bivariate(self,x,y,i,j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = DcorrX(max_lag=self._max_lag).test(x, y, reps=0 )
        return stat

class mgcx(base.undirected):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "mgcx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    def bivariate(self,x,y,i,j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGCX(max_lag=self._max_lag).test(x, y, reps=0)
        return stat