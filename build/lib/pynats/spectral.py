import numpy as np
from pynats import utils
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
from spectral_connectivity import Multitaper, Connectivity
from pynats.base import directed, undirected, parse, positive, real

from collections import namedtuple
"""
Toolkits used for spectral analysis of time series
"""

class effective(undirected):
    """ TODO: effective connectivity class
    Probably through dcm
    """

    def __init__(self):
        pass

class connectivity():

    cache = namedtuple('cache','connectivity multitaper')

    def __init__(self,fs=1,f_lb=0.0,f_ub=0.2):
        self._fs = fs # Not yet implemented
        self._f_lb = f_lb
        self._f_ub = f_ub
        paramstr = f'_fs-{fs}_flb-{f_lb}_fub-{f_ub}'.replace('.','')
        self.name = self.name + paramstr

def spectral(adjacency):

    @parse
    def preprocess(self,data):
        if not hasattr(data,'spectral'):
            z = np.squeeze(np.moveaxis(data.to_numpy(),0,1))
            m = Multitaper(z,
                            sampling_frequency=self._fs,
                            time_halfbandwidth_product=3,
                            start_time=0)
            c = Connectivity(fourier_coefficients=m.fft(),
                                        frequencies=m.frequencies)
            data.spectral = connectivity.cache(connectivity=c,multitaper=m)

        return adjacency(self,data)

    return preprocess

class coherence(connectivity,undirected,positive):

    humanname = "Coherence"

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        gamma = np.nanmean(data.spectral.connectivity.coherence_magnitude()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(gamma,np.nan)

        return gamma, data

class icoherence(connectivity,undirected,positive):

    humanname = 'Imaginary Coherence'

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super(icoherence,self).__init__(**kwargs)
        

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        gamma = np.nanmean(data.spectral.connectivity.imaginary_coherence()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(gamma,np.nan)

        return gamma, data

class phase(connectivity,undirected,positive):
    humanname = 'Phase consistency'

    def __init__(self,**kwargs):
        self.name = 'phase'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.nanmean(data.spectral.connectivity.pairwise_phase_consistency()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class phase_lag(connectivity,undirected,real):
    humanname = 'Phase lag'

    def __init__(self,**kwargs):
        self.name = 'phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.mean(data.spectral.connectivity.phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class weighted_phase_lag(connectivity,undirected,real):
    humanname = 'Weighted phase lag'

    def __init__(self,**kwargs):
        self.name = 'w-phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.mean(data.spectral.connectivity.weighted_phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class debiased_squared_weighted_phase_lag(connectivity,undirected,positive):
    humanname = 'Debiased squared weighted phase lag'

    def __init__(self,**kwargs):
        self.name = 'dsqw-phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.mean(data.spectral.connectivity.debiased_squared_phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class partial_coherence(connectivity,undirected,positive):
    humanname = "Partial coherence"
    cache = namedtuple('cache', 'gamma freq')

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15):
        self._TR = tr # Not yet implemented
        self._f_lb = f_lb
        self._f_ub = f_ub
        self.name = 'pcoh_t-{}_flb-{}_fub-{}'.format(tr,f_lb,f_ub)
        self.name = self.name.replace('.','')

    @parse
    def adjacency(self,data):        
        # This should be changed to conditioning on all, rather than averaging all conditionals

        if not hasattr(data,'pcoh'):
            z = np.squeeze(data.to_numpy())
            pdata = tsu.percent_change(z)
            time_series = ts.TimeSeries(pdata, sampling_interval=1)
            C1 = nta.CoherenceAnalyzer(time_series)
            data.pcoh = partial_coherence.cache(np.nanmean(C1.coherence_partial,axis=2),C1.frequencies)

        freq_idx_C = np.where((data.pcoh.freq > self._f_lb) * (data.pcoh.freq < self._f_ub))[0]
        pcoh = np.nan_to_num(np.mean(data.pcoh.gamma[:, :, freq_idx_C], -1))
        np.fill_diagonal(pcoh,np.nan)
        return pcoh, data

class partial_directed_coherence(connectivity,directed,positive):
    humanname = 'Partial directed coherence'

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.mean(data.spectral.connectivity.partial_directed_coherence()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class directed_transfer_function(connectivity,directed,positive):
    humanname = 'Directed transfer function'

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        phi = np.mean(data.spectral.connectivity.directed_transfer_function()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi, data

class spectral_granger(connectivity,directed,positive):
    humanname = 'Spectral Granger causality'

    def __init__(self,order=None,**kwargs):
        self.name = 'sgc'
        super().__init__(**kwargs)
        self.name = self.name + f'_o-{order}'

    @spectral
    def adjacency(self,data):
        freq = data.spectral.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        F = np.mean(data.spectral.connectivity.pairwise_spectral_granger_prediction()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(F,np.nan)
        return F, data