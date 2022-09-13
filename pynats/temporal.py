from statsmodels.tsa import stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pyEDM as edm
import pandas as pd
from math import isnan
from hyppo.time_series import MGCX, DcorrX
import warnings
from pynats.base import directed, undirected, parse_bivariate, parse_multivariate, unsigned, signed

import importlib
import scipy.spatial.distance as distance
import tslearn.metrics
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter

class coint(undirected,unsigned):
    
    humanname = "Cointegration"
    name = "coint"
    labels = ['unsigned','temporal','undirected','lagged','nonlinear']

    def __init__(self,method='johansen',statistic='trace_stat',
                    det_order=1,k_ar_diff=1,
                    autolag='aic',maxlag=10,trend='c'):
        self._method = method
        self._statistic = statistic
        if method == 'johansen':
            self.name += f'_{method}_{statistic}_order-{det_order}_ardiff-{k_ar_diff}'
            self._det_order = det_order
            self._k_ar_diff = k_ar_diff
        else:
            self._autolag = autolag
            self._maxlag = maxlag
            self._trend = trend
            self.name += f'_{method}_{statistic}_trend-{trend}_autolag-{autolag}_maxlag-{maxlag}'

    @property
    def key(self):
        key = (self._method,)
        if self._method == 'johansen':
            return key + (self._det_order,self._k_ar_diff)
        else:
            return key + (self._autolag,self._maxlag,self._trend)

    def _from_cache(self,data,i,j):
        idx = (i,j)
        try:
            ci = data.coint[self.key][idx]
        except (KeyError,AttributeError):
            z = data.to_numpy(squeeze=True)

            if self._method == 'aeg':
                stats = stattools.coint(z[i],z[j],autolag=self._autolag,maxlag=self._maxlag,trend=self._trend)
                ci = {'tstat': stats[0]}
            else:
                stats = coint_johansen(z[[i,j]].T,det_order=self._det_order,k_ar_diff=self._k_ar_diff)
                ci = {'max_eig_stat': stats.max_eig_stat[0], 'trace_stat': stats.trace_stat[0]}

            try:
                data.coint[self.key][idx] = ci
            except AttributeError:
                data.coint = {self.key: {idx: ci} }
            except KeyError:
                data.coint[self.key] = {idx: ci}
            data.coint[self.key][(j,i)] = ci

        return ci

    # Return the negative t-statistic (proxy for how co-integrated they are)
    @parse_bivariate
    def bivariate(self,data,i=None,j=None,verbose=False):
        ci = self._from_cache(data,i,j)
        return ci[self._statistic]

class ccm(directed,signed):

    humanname = "Convergent cross-maping"
    name = "ccm"
    labels = ['embedding','temporal','directed','lagged','causal','nonlinear','signed']

    def __init__(self,statistic='mean',embedding_dimension=None):
        self._statistic = statistic
        self._E = embedding_dimension

        self.name += f'_E-{embedding_dimension}_{statistic}'

    @property
    def key(self):
        return self._E

    def _from_cache(self,data):
        try:
            ccmf = data.ccm[self.key]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True)

            M = data.n_processes
            N = data.n_observations
            df = pd.DataFrame(np.concatenate([np.atleast_2d(np.arange(0,N)),z]).T,
                                columns=['index']+[f'proc{p}' for p in range(M)])

            # Get the embedding
            if self._E is None:
                embedding = np.zeros((M,1))

                # Infer optimal embedding from simplex projection
                for _i in range(M):
                    pred = str(10) + ' ' + str(N-10)
                    embed_df = edm.EmbedDimension(dataFrame=df,lib=pred,
                                                    pred=pred,columns=df.columns.values[_i+1],showPlot=False)
                    embedding[_i] = embed_df.max()['E']
            else:
                embedding = np.array([self._E]*M)

            # Compute CCM from the fixed or optimal embedding
            nlibs = 21
            ccmf = np.zeros((M,M,nlibs+1))
            for _i in range(M):
                for _j in range(_i+1,M):
                    try:
                        E = int(max(embedding[[_i,_j]]))
                    except NameError:
                        E = int(self._E)

                    # Get list of library sizes given nlibs and lower/upper bounds based on embedding dimension
                    upperE = int(np.floor((N-E-1)/10)*10)
                    lowerE = int(np.ceil(2*E/10)*10)
                    inc = int((upperE-lowerE) / nlibs)
                    lib_sizes = str(lowerE) + ' ' + str(upperE) + ' ' + str(inc)
                    srcname = df.columns.values[_i+1]
                    targname = df.columns.values[_j+1]
                    ccm_df = edm.CCM(dataFrame=df,E=E,
                                        columns=srcname,target=targname,
                                        libSizes=lib_sizes,sample=100)
                    ccmf[_i,_j] = ccm_df.iloc[:,1].values[:(nlibs+1)]
                    ccmf[_j,_i] = ccm_df.iloc[:,2].values[:(nlibs+1)]

            try:
                data.ccm[self.key] = ccmf
            except AttributeError:
                data.ccm = {self.key: ccmf}
        return ccmf

    @parse_multivariate
    def adjacency(self,data):
        ccmf = self._from_cache(data)

        if self._statistic == 'mean':
            return np.nanmean(ccmf,axis=2)
        elif self._statistic == 'max':
            return np.nanmax(ccmf,axis=2)
        elif self._statistic == 'diff':
            return np.nanmean(ccmf-np.transpose(ccmf,axes=[1,0,2]),axis=2)
        else:
            raise TypeError(f'Unknown statistic: {self._statistic}')

class dcorrx(directed,unsigned):
    """ Cross-distance correlation
    """

    humanname = "Cross-distance correlation"
    name = "dcorrx"
    labels = ['unsigned','independence','temporal','directed','lagged','nonlinear']

    def __init__(self,max_lag=1):
        self._max_lag = max_lag
        self.name += f'_maxlag-{max_lag}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _ = DcorrX(max_lag=self._max_lag).statistic(x,y)
        return stat

class mgcx(directed,unsigned):
    """ Cross-multiscale graph correlation
    """

    humanname = "Cross-multiscale graph correlation"
    name = "mgcx"
    labels = ['unsigned','independence','temporal','directed','lagged','nonlinear']

    def __init__(self,max_lag=1):
        self._max_lag = max_lag
        self.name += f'_maxlag-{max_lag}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy()
        x = z[i]
        y = z[j]
        stat, _, _ = MGCX(max_lag=self._max_lag).statistic(x,y)
        return stat

class time_warping(undirected, unsigned):

    labels = ['unsigned','distance','temporal','undirected','lagged','nonlinear']

    def __init__(self,global_constraint=None):
        gcstr = global_constraint
        if gcstr is not None:
            gcstr = gcstr.replace('_','-')
            self.name += f'_constraint-{gcstr}'
        self._global_constraint = global_constraint

    @property
    def simfn(self):
        try:
            return self._simfn
        except AttributeError:
            raise NotImplementedError(f'Add the similarity function for {self.name}')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return self._simfn(z[i],z[j],global_constraint=self._global_constraint)

class dynamic_time_warping(time_warping):

    humanname = 'Dynamic time warping'
    name = 'dtw'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.dtw

class canonical_time_warping(time_warping):

    humanname = 'Canonical time warping'
    name = 'ctw'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.ctw    

class longest_common_subsequence(time_warping):

    humanname = 'Longest common subsequence'
    name = 'lcss'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._simfn = tslearn.metrics.lcss

class soft_dynamic_time_warping(time_warping):

    humanname = 'Dynamic time warping'
    name = 'softdtw'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.soft_dtw(z[i],z[j])

class global_alignment_kernel(time_warping):

    humanname = 'Global alignment kernel'
    name = 'gak'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.gak(z[i],z[j])

class lb_keogh(unsigned,directed):
    humanname = 'LB Keogh'
    name = 'lbk'
    labels = ['unsigned','distance','temporal','undirected','lagged']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.lb_keogh(ts_query=z[j],ts_candidate=z[j])

class barycenter(directed,signed):

    humanname = 'Barycenter'
    name = 'bary'
    labels = ['signed','undirected','unpaired','temporal','nonlinear']

    def __init__(self,mode='euclidean',squared=False,statistic='mean'):
        if mode == 'euclidean':
            self._fn = euclidean_barycenter
        elif mode == 'dtw':
            self._fn = dtw_barycenter_averaging
        elif mode == 'sgddtw':
            self._fn = dtw_barycenter_averaging_subgradient
        elif mode == 'softdtw':
            self._fn = softdtw_barycenter
        else:
            raise NameError(f'Unknown barycenter mode: {mode}')
        self._mode = mode

        self._squared = squared
        self._preproc = lambda x : x
        if squared:
            self._preproc = lambda x : x**2
            self.name += f'-sq'
            
        if statistic == 'mean':
            self._statfn = lambda x : np.nanmean(self._preproc(x))
        elif statistic == 'max':
            self._statfn = lambda x : np.nanmax(self._preproc(x))
        else:
            raise NameError(f'Unknown statistic: {statistic}')

        self.name += f'_{mode}_{statistic}'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        try:
            bc = data.barycenter[self._mode][(i,j)]
        except (AttributeError,KeyError):
            z = data.to_numpy(squeeze=True)
            bc = self._fn(z[[i,j]])
            try:
                data.barycenter[self._mode][(i,j)] = bc
            except AttributeError:
                data.barycenter = {self._mode: {(i,j): bc}}
            except KeyError:
                data.barycenter[self._mode] = {(i,j): bc}
            data.barycenter[self._mode][(j,i)] = data.barycenter[self._mode][(i,j)]
        
        return self._statfn(bc)
