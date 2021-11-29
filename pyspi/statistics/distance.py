from hyppo.independence import MGC, Dcorr, HHG, Hsic
from hyppo.time_series import MGCX, DcorrX

import scipy.spatial.distance as distance
import tslearn.metrics
from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter

from pyspi.base import directed, undirected, parse_bivariate, unsigned, signed
import numpy as np

""" TODO: include optional kernels in each method
"""
class hsic(undirected,unsigned):
    """ Hilbert-Schmidt Independence Criterion (Hsic)
    """

    humanname = "Hilbert-Schmidt Independence Criterion"
    name = 'hsic'
    labels = ['unsigned','distance','unordered','nonlinear','undirected']

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = Hsic(bias=self._biased).statistic(x,y)
        return stat

class hhg(directed,unsigned):
    """ Heller-Heller-Gorfine independence criterion
    """

    humanname = "Heller-Heller-Gorfine Independence Criterion"
    name = 'hhg'
    labels = ['unsigned','distance','unordered','nonlinear','directed']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = HHG().statistic(x,y)
        return stat

class dcorr(undirected,unsigned):
    """ Distance correlation
    """

    humanname = "Distance correlation"
    name = 'dcorr'
    labels = ['unsigned','distance','unordered','nonlinear','undirected']

    def __init__(self,biased=False):
        self._biased = biased
        if biased:
            self.name += '_biased'
    
    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        """
        """
        x, y = data.to_numpy()[[i,j]]
        stat = Dcorr(bias=self._biased).statistic(x,y)
        return stat

class mgc(undirected,unsigned):
    """ Multiscale graph correlation
    """

    humanname = "Multiscale graph correlation"
    name = "mgc"
    labels = ['distance','unsigned','unordered','nonlinear','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        x, y = data.to_numpy()[[i,j]]
        stat = MGC().statistic(x,y)
        return stat


class dcorrx(directed,unsigned):
    """ Cross-distance correlation
    """

    humanname = "Cross-distance correlation"
    name = "dcorrx"
    labels = ['distance','unsigned','temporal','directed','nonlinear']

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
    labels = ['unsigned','distance','temporal','directed','nonlinear']

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

    labels = ['unsigned','distance','temporal','undirected','nonlinear']

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
    labels = ['nonlinear','unsigned','distance','temporal','undirected']

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):
        z = data.to_numpy(squeeze=True)
        return tslearn.metrics.lb_keogh(ts_query=z[j],ts_candidate=z[j])

class barycenter(directed,signed):

    humanname = 'Barycenter'
    name = 'bary'
    labels = ['distance','signed','undirected','temporal','nonlinear']

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