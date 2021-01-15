from statsmodels.tsa.stattools import coint as ci
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pyEDM as edm
import pandas as pd
from math import isnan
from hyppo.time_series import MGCX, DcorrX
import warnings
from pynats.base import directed, undirected, parse_bivariate, positive, real

class coint(undirected,real):
    
    humanname = "Cointegration"
    name = "coint"

    def __init__(self,method='johansen',statistic='pvalue'):
        self._method = method
        self._statistic = statistic
        self.name = self.name + '_' + method + '_' + statistic

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):

        z = data.to_numpy(squeeze=True)
        M = data.n_processes

        if not hasattr(data,'coint'):
            data.coint = {'max_eig_stat': np.full((M, M), np.NaN), 'trace_stat': np.full((M, M), np.NaN),
                            'tstat': np.full((M, M), np.NaN), 'pvalue': np.full((M, M), np.NaN)}

        if self._method == 'johansen':
            if isnan(data.coint['max_eig_stat'][i,j]):
                z_ij_T = np.transpose(z[[i,j]])
                stats = coint_johansen(z_ij_T,det_order=0,k_ar_diff=1)
                data.coint['max_eig_stat'][i,j] = stats.max_eig_stat[0]
                data.coint['trace_stat'][i,j] = stats.trace_stat[0]
        elif self._method == 'aeg':
            if isnan(data.coint['tstat'][i,j]):
                stats = ci(z[i],z[j])
                data.coint['tstat'][i,j] = stats[0]
                data.coint['pvalue'][i,j] = stats[1]
        else:
            raise TypeError(f'Unknown statistic: {self._method}')

        return data.coint[self._statistic][i,j]

class ccm(directed,real):

    humanname = "Convergent cross-maping"
    name = "ccm"

    def __init__(self,statistic='mean'):
        self._statistic = statistic
        self.name = self.name + '_' + statistic

    @staticmethod
    def _precompute(data):
        z = data.to_numpy(squeeze=True)

        M = data.n_processes
        N = data.n_observations
        df = pd.DataFrame(range(0,N),columns=['index'])
        embedding = np.zeros((M,1))

        names = []

        # First pass: infer optimal embedding
        for _i in range(M):
            names.append('var' + str(_i))
            df[names[_i]] = z[_i]
            pred = str(10) + ' ' + str(N-10)
            embed_df = edm.EmbedDimension(dataFrame=df,lib=pred,
                                            pred=pred,columns=str(_i),showPlot=False)
            embedding[_i] = embed_df.iloc[embed_df.idxmax().rho,0]
        
        # Get some reasonable library lengths
        nlibs = 5

        # Second pass: compute CCM
        score = np.zeros((M,M,nlibs+1))
        for _i in range(M):
            for _j in range(_i+1,M):
                E = int(max(embedding[[_i,_j]]))
                upperE = int(np.floor((N-E-1)/10)*10)
                lowerE = int(np.ceil(2*E/10)*10)
                inc = int((upperE-lowerE) / nlibs)
                lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)
                ccm_df = edm.CCM(dataFrame=df,E=E,columns=names[_i],target=names[_j],
                                    libSizes=lib_sizes,sample=100)
                sc1 = ccm_df.iloc[:,1]
                sc2 = ccm_df.iloc[:,2]
                score[_i,_j] = np.array(sc1)
                score[_j,_i] = np.array(sc2)

        data.ccm = {'embedding': embedding, 'score': score}

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        if not hasattr(data,'ccm'):
            ccm._precompute(data)

        if self._statistic == 'mean':
            stat = np.nanmean(data.ccm['score'][i,j])
        elif self._statistic == 'max':
            stat = np.nanmax(data.ccm['score'][i,j])
        elif self._statistic == 'diff':
            stat = np.nanmean(data.ccm['score'][i,j] - data.ccm['score'][j,i])
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}')

        return stat

class dcorrx(undirected,positive):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "dcorrx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = DcorrX(max_lag=self._max_lag).test(x, y, reps=0 )
        return stat

class mgcx(undirected,positive):
    """ Multi-graph correlation for time series
    """

    humanname = "Multi-scale graph correlation"
    name = "mgcx"

    def __init__(self,max_lag=1):
        self._max_lag = max_lag

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, _, _ = MGCX(max_lag=self._max_lag).test(x, y, reps=0)
        return stat