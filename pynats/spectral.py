import numpy as np
from pynats import utils
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
import spectral_connectivity as sc
from pynats.base import directed, undirected, parse_multivariate, positive, real

import warnings
"""
Toolkits used for spectral analysis of time series
"""

def spectral(adjacency):
    @parse_multivariate
    def decorator(self,data):
        if not hasattr(data,'connectivity'):
            z = np.squeeze(np.moveaxis(data.to_numpy(),0,1))
            m = sc.Multitaper(z,
                            sampling_frequency=self._fs,
                            time_halfbandwidth_product=3,
                            start_time=0)
            data.connectivity = sc.Connectivity(fourier_coefficients=m.fft(),
                                                frequencies=m.frequencies)

        return adjacency(self,data)
    return decorator

class connectivity():

    def __init__(self,fs=1,f_lb=0.0,f_ub=0.2):
        self._fs = fs # Not yet implemented
        self._f_lb = f_lb
        self._f_ub = f_ub
        paramstr = f'_fs-{fs}_flb-{f_lb}_fub-{f_ub}'.replace('.','')
        self.name = self.name + paramstr

class coherence(connectivity,undirected,positive):

    humanname = "Coherence"

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gamma = np.nanmean(data.connectivity.coherence_magnitude()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(gamma,np.nan)

        return gamma

class icoherence(connectivity,undirected,positive):

    humanname = 'Imaginary Coherence'

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super(icoherence,self).__init__(**kwargs)
        

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gamma = np.nanmean(data.connectivity.imaginary_coherence()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(gamma,np.nan)

        return gamma

class phase(connectivity,undirected,positive):
    humanname = 'Phase consistency'

    def __init__(self,**kwargs):
        self.name = 'phase'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.pairwise_phase_consistency()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class phase_lag(connectivity,undirected,real):
    humanname = 'Phase lag'

    def __init__(self,**kwargs):
        self.name = 'phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class weighted_phase_lag(connectivity,undirected,real):
    humanname = 'Weighted phase lag'

    def __init__(self,**kwargs):
        self.name = 'w-phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.weighted_phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class debiased_squared_weighted_phase_lag(connectivity,undirected,positive):
    humanname = 'Debiased squared weighted phase lag'

    def __init__(self,**kwargs):
        self.name = 'dsqw-phase-lag'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.debiased_squared_phase_lag_index()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class partial_coherence(connectivity,undirected,positive):
    humanname = "Partial coherence"

    def __init__(self,tr=1,f_lb=0.02,f_ub=0.15):
        self._TR = tr # Not yet implemented
        self._f_lb = f_lb
        self._f_ub = f_ub
        self.name = 'pcoh_t-{}_flb-{}_fub-{}'.format(tr,f_lb,f_ub)
        self.name = self.name.replace('.','')

    @parse_multivariate
    def adjacency(self,data):        
        # This should be changed to conditioning on all, rather than averaging all conditionals

        if not hasattr(data,'pcoh'):
            z = np.squeeze(data.to_numpy())
            pdata = tsu.percent_change(z)
            time_series = ts.TimeSeries(pdata, sampling_interval=1)
            C1 = nta.CoherenceAnalyzer(time_series)
            data.pcoh = {'gamma': np.nanmean(C1.coherence_partial,axis=2), 'freq': C1.frequencies}

        freq_idx_C = np.where((data.pcoh['freq'] > self._f_lb) * (data.pcoh['freq'] < self._f_ub))[0]
        pcoh = np.nan_to_num(np.mean(data.pcoh['gamma'][:, :, freq_idx_C], -1))
        np.fill_diagonal(pcoh,np.nan)
        return pcoh

class partial_directed_coherence(connectivity,directed,positive):
    humanname = 'Partial directed coherence'

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.partial_directed_coherence()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class directed_transfer_function(connectivity,directed,positive):
    humanname = 'Directed transfer function'

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(data.connectivity.directed_transfer_function()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class spectral_granger(connectivity,directed,positive):
    humanname = 'Spectral Granger causality'

    def __init__(self,order=None,**kwargs):
        self.name = 'sgc'
        super().__init__(**kwargs)
        self.name = self.name + f'_o-{order}'

    @spectral
    def adjacency(self,data):
        freq = data.connectivity.frequencies
        freq_id = np.where((freq > self._f_lb) * (freq < self._f_ub))[0]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            F = np.nanmean(data.connectivity.pairwise_spectral_granger_prediction()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(F,np.nan)
        return F