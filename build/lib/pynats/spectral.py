import numpy as np
from pynats import utils
import spectral_connectivity as sc # For directed spectral measures (excl. spectral GC) 
import mne.time_frequency as mtf
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, positive, real
# import pygc.parametric
# import pygc.granger
# import pygc.pySpec
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu

import warnings

"""
The measures here come from three different toolkits:

    - Undirected measurements generally come from MNE (coherence, imaginary coherence, phase-locking value, etc.)
    - Some directed measurements come from Eden Kramer Lab's spectral_connectivity toolkit (partial directed coherence and directed transfer function)
    - Spectral Granger causality comes from nitime (since Kramer's version doesn't optimise AR order)

Hopefully we'll eventually just use the cross-spectral density and VAR models to compute these directly, however this may involve integration with the temporal toolkits so may not ever get done..
"""
class connectivity():

    def __init__(self,mode='multitaper',fs=1,fmin=None,fmax=None):

        if fmin is None:
            if mode == 'mortlet':
                fmin = 1
            else:
                fmin = 0.05
        if fmax is None:
            fmax = np.pi/2

        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self._mode = mode
        paramstr = f'_{self._mode}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    def _get_csd(self,data):
        if not hasattr(data,'connectivity'):
            data.connectivity = {}
        
        if self._mode not in data.connectivity:
            """
            TODO: Allow for appending frequencies
            """
            z = np.moveaxis(np.atleast_3d(data.to_numpy()),-1,0)
            if self._mode == 'multitaper':
                data.connectivity[self._mode] = mtf.csd_array_multitaper(z, sfreq=self._fs,fmin=self._fmin,fmax=self._fmax,verbose='WARNING')
            if self._mode == 'fourier':
                data.connectivity[self._mode] = mtf.csd_array_fourier(z, sfreq=self._fs,fmin=self._fmin,fmax=self._fmax,verbose='WARNING')
            if self._mode == 'mortlet':
                data.connectivity[self._mode] = mtf.csd_array_morlet(z,sfreq=self._fs,frequencies=np.linspace(self._fmin,self._fmax,20),verbose='WARNING')

        return data.connectivity[self._mode]

    def _compute_stat(psd,i,j):
        raise NotImplementedError('This method must be overloaded')

    @parse_bivariate
    def bivariate(self,data,i=None,j=None):

        csd = self._get_csd(data)
        
        freq = csd.frequencies
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]
        
        stat = np.zeros((freq_id.shape[0]))
        for f in freq_id:
            psd = csd.get_data(freq[f])
            stat[f] = self._compute_stat(psd,i,j)

        return np.nanmean(stat)

class coherence(connectivity,undirected,positive):

    humanname = "Coherence"

    def __init__(self,**kwargs):
        self.name = 'coh'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        return np.absolute(psd[i,j]) / np.real(np.sqrt(psd[i,i] * psd[j,j] ))

class icoherence(connectivity,undirected,positive):

    humanname = 'Imaginary Coherence'

    def __init__(self,**kwargs):
        self.name = 'icoh'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        """ It's unclear why I have to put -np.real to match the MNE implementation?
        """
        return -np.real(np.imag(psd[i,j]) / np.sqrt(psd[i,i] * psd[j,j] ))

class phase_locking_value(connectivity,undirected,positive):

    humanname = 'Phase-locking value'

    def __init__(self,**kwargs):
        self.name = 'plv'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        return np.absolute( psd[i,j] / np.absolute(psd[i,j]) )

class corrected_imaginary_phase_locking_value(connectivity,undirected,positive):

    humanname = 'Corrected imaginary phase-locking value'

    def __init__(self,**kwargs):
        self.name = 'ciplv'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        acc = psd[i,j] / np.abs(psd[i,j])
        imag_plv = np.abs(np.imag(acc))
        real_plv = np.real(acc)
        real_plv = np.clip(real_plv, -1, 1)  # bounded from -1 to 1
        if np.abs(real_plv) == 1:
            real_plv = 0
        corrected_imag_plv = imag_plv / np.sqrt(1 - real_plv ** 2)
        return corrected_imag_plv

class pairwise_phase_consistency(connectivity,undirected,positive):

    humanname = 'Pairwise phase consistency'

    def __init__(self,**kwargs):
        self.name = 'ppc'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        return phase_locking_value._compute_stat(self,psd,i,j) ** 2

class phase_lag_index(connectivity,undirected,positive):

    humanname = 'Phase-lag index'

    def __init__(self,**kwargs):
        self.name = 'pli'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        return np.absolute(np.sign(psd[i,j]))

class unbiased_squared_phase_lag_index(connectivity,undirected,positive):

    humanname = 'Unbiased estimator of squared phase-lag index'

    def __init__(self,**kwargs):
        self.name = 'spli'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        return phase_lag_index._compute_stat(self,psd,i,j) ** 2

class weighted_phase_lag_index(connectivity,undirected,positive):

    humanname = 'Weighted phase-lag index'

    def __init__(self,**kwargs):
        self.name = 'wpli'
        super().__init__(**kwargs)

    def _compute_stat(self,psd,i,j):
        raise NotImplementedError('This cannot be implemented since without multiple realisations')

class debiased_weighted_phase_lag_index(connectivity,undirected,positive):

    humanname = 'Weighted phase-lag index'

    def __init__(self,**kwargs):
        self.name = 'wpli'
        super().__init__(**kwargs)
        self._compute_stat = debiased_weighted_phase_lag_index._dwpli

    @staticmethod
    def _dwpli(psd,i,j):
        raise NotImplementedError('This cannot be implemented since without multiple realisations')

class conditional_connectivity(directed,positive):

    def __init__(self,fs=1,fmin=0.05,fmax=np.pi/2):
        self._fs = fs # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        paramstr = f'_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name = self.name + paramstr

    @parse_multivariate
    def adjacency(self, data):
        if not hasattr(data,'directed_connectivity'):
            data.directed_connectivity = {}
            z = np.squeeze(np.moveaxis(data.to_numpy(),0,1))
            m = sc.Multitaper(z,
                            sampling_frequency=self._fs,
                            time_halfbandwidth_product=3,
                            start_time=0)
            data.directed_connectivity = sc.Connectivity(fourier_coefficients=m.fft(),
                                                frequencies=m.frequencies)

        C = data.directed_connectivity

        freq = C.frequencies
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]
        
        measure = getattr(C,self._measure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            phi = np.nanmean(measure()[0,freq_id,:,:], axis=0)
        np.fill_diagonal(phi,np.nan)
        return phi

class partial_directed_coherence(conditional_connectivity,directed,positive):
    humanname = 'Partial directed coherence'

    def __init__(self,**kwargs):
        self.name = 'pdcoh'
        super().__init__(**kwargs)
        self._measure = 'partial_directed_coherence'

class directed_transfer_function(conditional_connectivity,directed,positive):
    humanname = 'Directed transfer function'

    def __init__(self,**kwargs):
        self.name = 'dtf'
        super().__init__(**kwargs)
        self._measure = 'directed_transfer_function'

class partial_coherence(undirected,positive):

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

class spectral_granger(directed,positive):
    
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