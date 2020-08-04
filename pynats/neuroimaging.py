from nilearn import connectome as fc
from sklearn import covariance as cov
import numpy as np
from . import pynats_utils as utils
from . import basedep as base
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu

"""
Toolkits generally used for functional or effective connectivity (i.e., neuro)
"""

class functional(base.symmeas):
    """ Functional connectivity
    
    This is equivalent to Pearson's r if the empirical covariance matrix is selected
    
    Information on other covariance estimators at: https://scikit-learn.org/stable/modules/covariance.html
    """

    humanname = "Functional connectivity"

    # Current just using the defaults. Let user input in future
    _ledoit_wolf = cov.LedoitWolf
    _empirical = cov.EmpiricalCovariance
    _shrunk = cov.ShrunkCovariance
    _oas = cov.OAS

    def __init__(self,
                    cov_estimator='ledoit_wolf',
                    kind='correlation'):
        self.name = 'fc' + '_' + kind + '_' + cov_estimator
        
        self._cov_estimator = eval('self._' + cov_estimator + '()')
        self._kind = kind

    def getadj(self,z):
        fc_measure = fc.ConnectivityMeasure(cov_estimator=self._cov_estimator,
                                                     kind=self._kind)

        z = np.transpose(z)
        fc_matrix = fc_measure.fit_transform([z])[0]
        np.fill_diagonal(fc_matrix,np.nan)
        return fc_matrix

    def ispositive(self):
        return False

class effective(base.symmeas):
    """ TODO: effective connectivity class
    Probably through dcm
    """

    def __init__(self):
        pass

class coherence(base.symmeas):
    humanname = "Coherence"

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15):
        self._TR = tr
        self._f_lb = f_lb
        self._f_ub = f_ub
        self.name = 'coh_t-{}_flb-{}_fub-{}'.format(tr,f_lb,f_ub)
        self.name = self.name.replace('.','')

    def ispositive(self):
        return True

    def getadj(self,z):
        pdata = tsu.percent_change(z)
        time_series = ts.TimeSeries(pdata, sampling_interval=self._TR)
        C1 = nta.CoherenceAnalyzer(time_series)
        freq_idx_C = np.where((C1.frequencies > self._f_lb) * (C1.frequencies < self._f_ub))[0]
        coh = np.nan_to_num(np.mean(C1.coherence[:, :, freq_idx_C], -1))
        np.fill_diagonal(coh,np.nan)
        return coh

class spectralgranger(base.asymmeas):

    humanname = "Spectral Granger causality"

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15,order=None,criterion=None):
        self._TR = tr
        self._f_lb = f_lb
        self._f_ub = f_ub
        self._order = order
        if order is None:
            if criterion is None:
                self.name = 'gc_flb-{}_fub-{}_t-{}'.format(f_lb,f_ub,tr)
            else:
                self.name = 'gc_c-{}_flb-{}_fub-{}_t-{}'.format(criterion,order,f_lb,f_ub,tr)
        else:
            self.name = 'gc_p-{}_flb-{}_fub-{}_t-{}'.format(order,f_lb,f_ub,tr)
        self.name = self.name.replace('.','')

    def ispositive(self):
        return True

    def getadj(self,z):
        m = z.shape[0]

        pdata = tsu.percent_change(z)
        time_series = ts.TimeSeries(pdata, sampling_interval=self._TR)
        G = nta.GrangerAnalyzer(time_series, order=self._order, max_order=30)
        freq_idx_G = np.where((G.frequencies > self._f_lb) * (G.frequencies < self._f_ub))[0]

        try:
            gc_triu = np.mean(G.causality_xy[:,:,freq_idx_G], -1)
            gc_tril = np.mean(G.causality_yx[:,:,freq_idx_G], -1)

            gc = np.empty((m,m))
            triu_id = np.triu_indices(m)
            tril_id = np.tril_indices(m)

            gc[triu_id] = gc_triu[triu_id]
            gc[triu_id[1],triu_id[0]] = gc_tril[triu_id]
        except ValueError as err:
            print('Spectral GC failed {0}'.format(err))
            gc = np.empty((m,m))
            gc[:] = np.NaN

        return gc