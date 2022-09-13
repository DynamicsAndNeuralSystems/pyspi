import mne.connectivity as mnec
from pynats.base import directed, parse_bivariate, undirected, parse_multivariate, unsigned
import numpy as np
import warnings
from functools import partial

class mne(unsigned):

    def __init__(self,fs=1,fmin=0,fmax=None,statistic='mean'):
        if fmax is None:
            fmax = fs/2

        self._fs = fs
        if fs != 1:
            warnings.warn('Multiple sampling frequencies not yet handled.')
        self._fmin = fmin
        self._fmax = fmax
        if statistic == 'mean':
            self._statfn = np.nanmean
        elif statistic == 'max':
            self._statfn = np.nanmax
        else:
            raise NameError(f'Unknown statistic {statistic}')
        
        self._statistic = statistic

        paramstr = f'_wavelet_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}'.replace('.','-')
        self.name += paramstr

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f'Include measure for {self.humanname}')

    def _get_cache(self,data):
        try:
            conn, freq = data.mne[self.measure]
        except (KeyError,AttributeError):
            z = np.moveaxis(data.to_numpy(),2,0)

            cwt_freqs = np.linspace(0.2, 0.5, 125)
            cwt_n_cycles = cwt_freqs / 7.
            conn, freq, _, _, _ = mnec.spectral_connectivity(
                    data=z, method=self.measure, mode='cwt_morlet',
                    sfreq=self._fs, mt_adaptive=True,
                    fmin=5/data.n_observations,fmax=self._fs/2,
                    cwt_freqs=cwt_freqs,
                    cwt_n_cycles=cwt_n_cycles, verbose='WARNING')
                
            try:
                data.mne[self.measure] = (conn,freq)
            except AttributeError:
                data.mne = {self.measure: (conn,freq)}

        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return conn, freq_id

    @parse_multivariate
    def adjacency(self, data):
        adj_freq, freq_id = self._get_cache(data)
        try:
            adj = self._statfn(adj_freq[...,freq_id,:], axis=(2,3))
        except np.AxisError:
            adj = self._statfn(adj_freq[...,freq_id], axis=2)
        ui = np.triu_indices(data.n_processes,1)
        adj[ui] = adj.T[ui]
        np.fill_diagonal(adj,np.nan)
        return adj

def modify_stats(statfn,modifier):
    def parsed_stats(stats,statfn,modifier,**kwargs):
        return statfn(modifier(stats),**kwargs)
    return partial(parsed_stats, statfn=statfn, modifier=modifier)

class coherence_magnitude(mne,undirected):
    humanname = 'Coherence magnitude (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'cohmag'
        self._measure = 'coh'
        super().__init__(**kwargs)

class coherence_phase(mne,undirected):
    humanname = 'Coherence phase (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'phase'
        self._measure = 'cohy'
        super().__init__(**kwargs)

        # Take the angle before computing the statistic
        self._statfn = modify_stats(self._statfn,np.angle)

class icoherence(mne,undirected):
    humanname = 'Imaginary coherency (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'icoh'
        self._measure = 'imcoh'
        super().__init__(**kwargs)

class phase_locking_value(mne,undirected):
    humanname = 'Phase locking value (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'plv'
        self._measure = 'plv'
        super().__init__(**kwargs)

class pairwise_phase_consistency(mne,undirected):
    humanname = 'Pairwise phase consistency (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'ppc'
        self._measure = 'ppc'
        super().__init__(**kwargs)

class phase_lag_index(mne,undirected):
    humanname = 'Phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'pli'
        self._measure = 'pli'
        super().__init__(**kwargs)

class debiased_squared_phase_lag_index(mne,undirected):
    humanname = 'Debiased squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'dspli'
        self._measure = 'pli2_unbiased'
        super().__init__(**kwargs)

class weighted_phase_lag_index(mne,undirected):
    humanname = 'Weighted squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'wspli'
        self._measure = 'wpli'
        super().__init__(**kwargs)

class debiased_weighted_squared_phase_lag_index(mne,undirected):
    humanname = 'Debiased weighted squared phase lag index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'dwspli'
        self._measure = 'wpli2_debiased'
        super().__init__(**kwargs)

class phase_slope_index(mne,directed):
    humanname = 'Phase slope index (wavelet)'
    labels = ['unsigned','wavelet','undirected']

    def __init__(self,**kwargs):
        self.name = 'psi'
        super().__init__(**kwargs)
        self.name += f'_{self._statistic}'

    def _get_cache(self,data):
        try:
            psi = data.mne_psi['psi']
            freq = data.mne_psi['freq']
        except AttributeError:
            z = np.moveaxis(data.to_numpy(),2,0)

            freqs = np.linspace(0.2, 0.5, 10)
            psi, freq, _, _, _ = mnec.phase_slope_index(
                    data=z,mode='cwt_morlet',sfreq=self._fs,
                    mt_adaptive=True, cwt_freqs=freqs,
                    verbose='WARNING')
            freq = freq[0]
            data.mne_psi = dict(psi=psi,freq=freq)

        # freq = conn.frequencies
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return psi, freq_id

    @parse_multivariate
    def adjacency(self, data):
        adj_freq, freq_id = self._get_cache(data)
        adj = self._statfn(np.real(adj_freq[...,freq_id]), axis=(2,3))

        ui = np.triu_indices(data.n_processes,1)
        adj[ui] = adj.T[ui]
        np.fill_diagonal(adj,np.nan)
        return adj