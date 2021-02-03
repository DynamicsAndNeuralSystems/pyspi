"""
Redundant methods now (all in spectral.py)
"""
import numpy as np
from pynats import utils
from pynats.base import directed, undirected, parse_bivariate, positive, real

from scipy import ndimage
import warnings
import pywt
"""
"""

def cwt(bivariate):
    @parse_bivariate
    def decorator(self,data,i=None,j=None):
        if not hasattr(data,'wavelet'):
            z = data.to_numpy()

            dt = 1/self._fs
            n_notes = self._wmax - self._wmin
            n_octaves = np.int( np.log2( 2*np.floor( data.n_observations / 2.0 ) ) )
            scales = 2**np.arange(1, n_octaves, 1.0 / n_notes)

            data.wavelet = {'scales': scales,
                            'frequencies': pywt.scale2frequency('cmor1.5-1.0', scales) / dt,
                            'coeff': [], 'freq': []}
            for _i in range(data.n_processes):
                coeff, freqs = pywt.cwt(z[_i],scales,'cmor1.5-1.0')

                data.wavelet['coeff'].append(np.squeeze(coeff))
                data.wavelet['freq'].append(freqs)

        return bivariate(self,data,i,j)
    return decorator

class wcoh(undirected,positive):

    humanname = "Wavelet coherence"

    def __init__(self,fs=1,w_min=1,w_max=30,t_lb=0,t_ub=np.inf,f_lb=0,f_ub=np.inf):
        self.name = 'wcoh'
        self._widths = np.arange(w_min,w_max)
        self._fs = fs
        self._f_lb = f_lb
        self._f_ub = f_ub
        self._t_lb = t_lb
        self._t_ub = t_ub
        self._wmin = w_min
        self._wmax = w_max
        paramstr = f'_fs-{fs}_w{w_min}-{w_max}_tlb-{t_lb}_tub-{t_ub}_flb-{f_lb}_fub-{f_ub}'.replace('.','')
        self.name = self.name + paramstr

    @cwt
    def bivariate(self,data,i=None,j=None):
        freq = data.wavelet['frequencies']
        scales = data.wavelet['scales']

        scale_matrix = np.ones([1, data.n_observations]) * scales[:, None]
        
        coeff_i = data.wavelet['coeff'][i]
        coeff_j = data.wavelet['coeff'][j]
        coeff_ij = coeff_i * np.conj(coeff_j)

        S_i = ndimage.gaussian_filter((np.abs(coeff_i)**2 / scale_matrix), sigma=2)
        S_j = ndimage.gaussian_filter((np.abs(coeff_j)**2 / scale_matrix), sigma=2)
        S_ij = ndimage.gaussian_filter((np.abs(coeff_ij / scale_matrix)), sigma=2)
        wcoh = np.mean(S_ij**2 / (S_i * S_j))

        return wcoh