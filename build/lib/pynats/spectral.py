import numpy as np
from functools import partial
from pynats import utils
import spectral_connectivity as sc # For directed spectral measures (excl. spectral GC) 
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, positive, real
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu

import warnings

"""
The measures here come from three different toolkits:

    - Simple undirected measurements generally come from MNE (coherence, imaginary coherence, phase slope index)
        - This is not true anymore. I originally used this b/c they had additional (short-time fourier and Mortlet)
            ways of computing the spectral measures, but the use of epochs, etc. seemed dissimilar to literature.
            Need to look into this
    - Some directed measurements come from Eden Kramer Lab's spectral_connectivity toolkit (partial directed coherence and directed transfer function)
    - Spectral Granger causality comes from nitime (since Kramer's version doesn't optimise AR order)

Hopefully we'll eventually just use the cross-spectral density and VAR models to compute these directly, however this may involve integration with the temporal toolkits so may not ever get done..
"""
# class mne_connectivity():

#     def __init__(self,mode='multitaper',fs=1,fmin=None,fmax=None):
#         if fmin is None:
#             if mode == 'mortlet':
#                 fmin = 1
#             else:
#                 fmin = 0.05
#         if fmax is None:
#             fmax = np.pi/2

#         self._fs = fs # Not yet implemented
#         self._fmin = fmin
#         self._fmax = fmax
#         self._mode = mode
#         paramstr = f'_{self._mode}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
#         self.name = self.name + paramstr

#     def _get_csd(self,data):
#         if not hasattr(data,'connectivity'):
#             data.connectivity = {}
        
#         if self._mode not in data.connectivity:
#             """
#             TODO: Allow for appending frequencies
#             """
#             z = np.moveaxis(np.atleast_3d(data.to_numpy()),-1,0)
#             if self._mode == 'multitaper':
#                 data.connectivity[self._mode] = mtf.csd_array_multitaper(z,sfreq=self._fs,fmin=self._fmin,fmax=self._fmax,verbose='WARNING')
#             if self._mode == 'fourier':
#                 data.connectivity[self._mode] = mtf.csd_array_fourier(z, sfreq=self._fs,fmin=self._fmin,fmax=self._fmax,verbose='WARNING')
#             if self._mode == 'mortlet':
#                 data.connectivity[self._mode] = mtf.csd_array_morlet(z,sfreq=self._fs,frequencies=np.linspace(self._fmin,self._fmax,20),verbose='WARNING')

#         return data.connectivity[self._mode]

#     def _compute_stat(psd,i,j):
#         raise NotImplementedError('This method must be overloaded')

#     @parse_bivariate
#     def bivariate(self,data,i=None,j=None):

#         csd = self._get_csd(data)
        
#         freq = csd.frequencies
#         freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]
        
#         stat = np.zeros((freq_id.shape[0]))
#         for f in freq_id:
#             psd = csd.get_data(freq[f])
#             stat[f] = self._compute_stat(psd,i,j)

#         return np.nanmean(stat)

# class coherence(mne_connectivity,undirected):

#     humanname = "Coherence"

#     def __init__(self,**kwargs):
#         self.name = 'coh'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         return np.absolute(psd[i,j]) / np.real(np.sqrt(psd[i,i] * psd[j,j] ))

# class icoherence(mne_connectivity,undirected):

#     humanname = 'Imaginary Coherence'

#     def __init__(self,**kwargs):
#         self.name = 'icoh'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         """ It's unclear why I have to put -np.real to match the MNE implementation?
#         """
#         return -np.real(np.imag(psd[i,j]) / np.sqrt(psd[i,i] * psd[j,j] ))

# class phase_slope_index(undirected):

#     humanname = 'Phase slope index'

#     def __init__(self,mode='multitaper',fs=1,fmin=None,fmax=None):
#         self.name = 'psi'
#         if fmin is None:
#             if mode == 'mortlet':
#                 fmin = 1
#             else:
#                 fmin = 0.05
#         if fmax is None:
#             fmax = np.pi/2

#         self._fs = fs # Not yet implemented
#         self._fmin = fmin
#         self._fmax = fmax
#         self._mode = mode
#         paramstr = f'_{self._mode}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
#         self.name = self.name + paramstr

#     @parse_bivariate
#     def bivariate(self,data,i=None,j=None):
#         z = np.moveaxis(np.atleast_3d(data.to_numpy()[[i,j]]),-1,0)
#         psi = mnec.phase_slope_index(z,mode=self._mode,sfreq=self._fs,fmin=self._fmin,fmax=self._fmax,verbose='WARNING')
#         return np.mean(psi[1][0])


# class phase_locking_value(connectivity,undirected):

#     humanname = 'Phase-locking value'

#     def __init__(self,**kwargs):
#         self.name = 'plv'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')
#         return np.absolute( psd[i,j] / np.absolute(psd[i,j]) )

# class corrected_imaginary_phase_locking_value(connectivity,undirected):

#     humanname = 'Corrected imaginary phase-locking value'

#     def __init__(self,**kwargs):
#         self.name = 'ciplv'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')
#         acc = psd[i,j] / np.abs(psd[i,j])
#         imag_plv = np.abs(np.imag(acc))
#         real_plv = np.real(acc)
#         real_plv = np.clip(real_plv, -1, 1)  # bounded from -1 to 1
#         if np.abs(real_plv) == 1:
#             real_plv = 0
#         corrected_imag_plv = imag_plv / np.sqrt(1 - real_plv ** 2)
#         return corrected_imag_plv

# class pairwise_phase_consistency(connectivity,undirected):

#     humanname = 'Pairwise phase consistency'

#     def __init__(self,**kwargs):
#         self.name = 'ppc'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')
#         denom = np.abs(psd[i,j])
#         if denom == 0.:
#             denom = 1.
#             acc = 0.  # handle division by zero
#         else:
#             acc = psd[i,j] / denom

#         return np.real(acc * np.conj(acc))

# class phase_lag_index(connectivity,undirected):

#     humanname = 'Phase-lag index'

#     def __init__(self,**kwargs):
#         self.name = 'pli'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')
#         return np.absolute(np.sign(np.imag(psd[i,j])))

# class unbiased_squared_phase_lag_index(connectivity,undirected):

#     humanname = 'Unbiased estimator of squared phase-lag index'

#     def __init__(self,**kwargs):
#         self.name = 'spli'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')
#         return phase_lag_index._compute_stat(self,psd,i,j) ** 2

# class weighted_phase_lag_index(connectivity,undirected):

#     humanname = 'Weighted phase-lag index'

#     def __init__(self,**kwargs):
#         self.name = 'wpli'
#         super().__init__(**kwargs)

#     def _compute_stat(self,psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')

# class debiased_weighted_phase_lag_index(connectivity,undirected):

#     humanname = 'Weighted phase-lag index'

#     def __init__(self,**kwargs):
#         self.name = 'wpli'
#         super().__init__(**kwargs)
#         self._compute_stat = debiased_weighted_phase_lag_index._dwpli

#     @staticmethod
#     def _dwpli(psd,i,j):
#         raise NotImplementedError('This will be implemented as a multivariate statistic (not bivariate).')

class kramer_connectivity(directed):

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2):
        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    def _get_measure(self,C):
        raise NotImplementedError

    @parse_multivariate
    def adjacency(self, data):
        if not hasattr(data,'connectivity'):
            data.connectivity = {}
            z = np.squeeze(np.moveaxis(data.to_numpy(),0,1))
            m = sc.Multitaper(z,
                            sampling_frequency=self._fs,
                            time_halfbandwidth_product=3,
                            start_time=0)
            data.connectivity = sc.Connectivity.from_multitaper(m)

        C = data.connectivity

        freq = C.frequencies
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]
        
        measure = self._get_measure(C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            res = measure()
            try:
                phi = np.nanmean(np.real(res[0,freq_id,:,:]), axis=0)
            except IndexError: # For phase-slope index
                phi = res[0]
            except TypeError: # For group delay
                phi = res[1][0]
        np.fill_diagonal(phi,np.nan)
        return phi

class coherency(kramer_connectivity,undirected):
    humanname = 'Coherency'

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherency

class coherence_phase(kramer_connectivity,undirected):
    humanname = 'Coherence phase'

    def __init__(self,**kwargs):
        self.name = 'phase'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherence_phase

class coherence_magnitude(kramer_connectivity,undirected):
    humanname = 'Coherence magnitude'

    def __init__(self,**kwargs):
        self.name = 'cohmag'
        super().__init__(**kwargs)

    def _get_measure(self,C):
        return C.coherence_magnitude

class icoherence(kramer_connectivity,undirected):
    humanname = 'Coherence'

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super().__init__(**kwargs)
        self._measure = 'imaginary_coherence'

    def _get_measure(self,C):
        return C.imaginary_coherence

class phase_locking_value(kramer_connectivity,undirected):
    humanname = 'Phase-locking value'

    def __init__(self,**kwargs):
        self.name = 'plv'
        super().__init__(**kwargs)
        self._measure = 'phase_locking_value'

    def _get_measure(self,C):
        return C.phase_locking_value

class phase_lag_index(kramer_connectivity,undirected):
    humanname = 'Phase-locking value'

    def __init__(self,**kwargs):
        self.name = 'pli'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.phase_lag_index

class weighted_phase_lag_index(kramer_connectivity,undirected):
    humanname = 'Weighted phase-lag index'

    def __init__(self,**kwargs):
        self.name = 'wpli'
        super().__init__(**kwargs)
        
    def _get_measure(self,C):
        return C.weighted_phase_lag_index

class debiased_squared_phase_lag_index(kramer_connectivity,undirected):
    humanname = 'Debiased squared phase-lag value'

    def __init__(self,**kwargs):
        self.name = 'dspli'
        super().__init__(**kwargs)
        
    def _get_measure(self,C):
        return C.debiased_squared_phase_lag_index

class debiased_squared_weighted_phase_lag_index(kramer_connectivity,undirected):
    humanname = 'Debiased squared weighted phase-lag value'

    def __init__(self,**kwargs):
        self.name = 'dswpli'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.debiased_squared_weighted_phase_lag_index

class pairwise_phase_consistency(kramer_connectivity,undirected):
    humanname = 'Pairwise phase consistency'

    def __init__(self,**kwargs):
        self.name = 'ppc'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.pairwise_phase_consistency

class directed_coherence(kramer_connectivity,directed):
    humanname = 'Directed coherence'

    def __init__(self,**kwargs):
        self.name = 'dcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.directed_coherence

class partial_directed_coherence(kramer_connectivity,directed):
    humanname = 'Partial directed coherence'

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.partial_directed_coherence

class generalized_partial_directed_coherence(kramer_connectivity,directed):
    humanname = 'Generalized partial directed coherence'

    def __init__(self,**kwargs):
        self.name = 'gpdcoh'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.generalized_partial_directed_coherence

class directed_transfer_function(kramer_connectivity,directed):
    humanname = 'Directed transfer function'

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.directed_transfer_function

class direct_directed_transfer_function(kramer_connectivity,directed):
    humanname = 'Direct directed transfer function'

    def __init__(self,**kwargs):
        self.name = 'ddtf'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return C.direct_directed_transfer_function

class phase_slope_index(kramer_connectivity,directed):
    humanname = 'Phase slope index'

    def __init__(self,**kwargs):
        self.name = 'psi'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return partial(C.phase_slope_index,
                        frequencies_of_interest=[self._fmin,self._fmax],
                        frequency_resolution=0.1)

class group_delay(kramer_connectivity,directed):
    humanname = 'Group delay'

    def __init__(self,**kwargs):
        self.name = 'gd'
        super().__init__(**kwargs)
    
    def _get_measure(self,C):
        return partial(C.group_delay,
                        frequencies_of_interest=[self._fmin,self._fmax],
                        frequency_resolution=0.1)

class partial_coherence(undirected):

    humanname = 'Partial coherence'
    name = 'pcoh'

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2):
        self._TR = 1/fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    @parse_multivariate
    def adjacency(self,data):        
        # This should be changed to conditioning on all, rather than averaging all conditionals

        if not hasattr(data,'pcoh'):
            z = np.squeeze(data.to_numpy())
            pdata = tsu.percent_change(z)
            time_series = ts.TimeSeries(pdata, sampling_interval=1)
            C1 = nta.CoherenceAnalyzer(time_series)
            data.pcoh = {'gamma': np.nanmean(C1.coherence_partial,axis=2), 'freq': C1.frequencies}

        freq_idx_C = np.where((data.pcoh['freq'] > self._fmin) * (data.pcoh['freq'] < self._fmax))[0]
        pcoh = np.nan_to_num(np.mean(data.pcoh['gamma'][:, :, freq_idx_C], -1))
        np.fill_diagonal(pcoh,np.nan)
        return pcoh

class spectral_granger(directed):
    
    humanname = 'Spectral Granger causality'
    name = 'sgc'

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2,order=None):
        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self._order = order
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}_order-{order}'.replace('.','-')
        self.name = self.name + paramstr

    @parse_multivariate
    def adjacency(self, data):
        z = data.to_numpy(squeeze=True)
        m = data.n_processes
 
        pdata = tsu.percent_change(z)
        time_series = ts.TimeSeries(pdata, sampling_interval=1)
        G = nta.GrangerAnalyzer(time_series, order=self._order, max_order=30)
        try:
            freq_idx_G = np.where((G.frequencies > self._fmin) * (G.frequencies < self._fmax))[0]
        
            gc_triu = np.mean(G.causality_xy[:,:,freq_idx_G], -1)
            gc_tril = np.mean(G.causality_yx[:,:,freq_idx_G], -1)

            gc = np.empty((m,m))
            triu_id = np.triu_indices(m)

            gc[triu_id] = gc_triu[triu_id]
            gc[triu_id[1],triu_id[0]] = gc_tril[triu_id]
        except (ValueError,TypeError) as err:
            print('Spectral GC failed {0}'.format(err))
            gc = np.empty((m,m))
            gc[:] = np.NaN

        return gc