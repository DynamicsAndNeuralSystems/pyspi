import numpy as np
from . import pynats_utils as utils
from . import basedep as base
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu

"""
Toolkits generally used for functional or effective connectivity (i.e., neuro)
"""

class effective(base.undirected):
    """ TODO: effective connectivity class
    Probably through dcm
    """

    def __init__(self):
        pass

class coherence(base.undirected):
    humanname = "Coherence"

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15):
        self._TR = tr # Not yet implemented
        self._f_lb = f_lb
        self._f_ub = f_ub
        self.name = 'coh_t-{}_flb-{}_fub-{}'.format(tr,f_lb,f_ub)
        self.name = self.name.replace('.','')

    def ispositive(self):
        return True

    @staticmethod
    def preprocess(z):
        pdata = tsu.percent_change(z)
        time_series = ts.TimeSeries(pdata, sampling_interval=1)
        C1 = nta.CoherenceAnalyzer(time_series)
        coherence._coh = C1.coherence
        coherence._freq = C1.frequencies

    def getadj(self,z):
        freq_idx_C = np.where((self._freq > self._f_lb) * (self._freq < self._f_ub))[0]
        coh = np.nan_to_num(np.mean(self._coh[:, :, freq_idx_C], -1))
        np.fill_diagonal(coh,np.nan)
        coh = np.nan_to_num(np.mean(self._coh[:, :, freq_idx_C], -1))
        np.fill_diagonal(coh,np.nan)
        return coh

class granger(base.directed):

    humanname = "Spectral Granger causality"

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15,order=None,criterion=None):
        self._TR = tr # Not yet implemented
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
        time_series = ts.TimeSeries(pdata, sampling_interval=1)
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