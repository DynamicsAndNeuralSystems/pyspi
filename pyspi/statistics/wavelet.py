import mne.connectivity as mnec
from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import numpy as np
import warnings
from functools import partial


class mne(Unsigned):
    def __init__(self, fs=1, fmin=0, fmax=None, statistic="mean"):
        if fmax is None:
            fmax = fs / 2

        self._fs = fs
        if fs != 1:
            warnings.warn("Multiple sampling frequencies not yet handled.")
        self._fmin = fmin
        self._fmax = fmax
        if statistic == "mean":
            self._statfn = np.nanmean
        elif statistic == "max":
            self._statfn = np.nanmax
        else:
            raise NameError(f"Unknown statistic {statistic}")

        self._statistic = statistic

        paramstr = (
            f"_wavelet_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _get_cache(self, data):
        try:
            conn, freq = data.mne[self.measure]
        except (KeyError, AttributeError):
            z = np.moveaxis(data.to_numpy(), 2, 0)

            cwt_freqs = np.linspace(0.2, 0.5, 125)
            cwt_n_cycles = cwt_freqs / 7.0
            conn, freq, _, _, _ = mnec.spectral_connectivity(
                data=z,
                method=self.measure,
                mode="cwt_morlet",
                sfreq=self._fs,
                mt_adaptive=True,
                fmin=5 / data.n_observations,
                fmax=self._fs / 2,
                cwt_freqs=cwt_freqs,
                cwt_n_cycles=cwt_n_cycles,
                verbose="WARNING",
            )

            try:
                data.mne[self.measure] = (conn, freq)
            except AttributeError:
                data.mne = {self.measure: (conn, freq)}

        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return conn, freq_id

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq_id = self._get_cache(data)
        try:
            adj = self._statfn(adj_freq[..., freq_id, :], axis=(2, 3))
        except np.AxisError:
            adj = self._statfn(adj_freq[..., freq_id], axis=2)
        ui = np.triu_indices(data.n_processes, 1)
        adj[ui] = adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj


def modify_stats(statfn, modifier):
    def parsed_stats(stats, statfn, modifier, **kwargs):
        return statfn(modifier(stats), **kwargs)

    return partial(parsed_stats, statfn=statfn, modifier=modifier)


class CoherenceMagnitude(mne, Undirected):
    name = "Coherence magnitude (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        self._measure = "coh"
        super().__init__(**kwargs)


class CoherencePhase(mne, Undirected):
    name = "Coherence phase (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "phase"
        self._measure = "cohy"
        super().__init__(**kwargs)

        # Take the angle before computing the statistic
        self._statfn = modify_stats(self._statfn, np.angle)


class ImaginaryCoherence(mne, Undirected):
    name = "Imaginary coherency (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        self._measure = "imcoh"
        super().__init__(**kwargs)


class PhaseLockingValue(mne, Undirected):
    name = "Phase locking value (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        self._measure = "plv"
        super().__init__(**kwargs)


class PairwisePhaseConsistency(mne, Undirected):
    name = "Pairwise phase consistency (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        self._measure = "ppc"
        super().__init__(**kwargs)


class PhaseLagIndex(mne, Undirected):
    name = "Phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        self._measure = "pli"
        super().__init__(**kwargs)


class DebiasedSquaredWeightedPhaseLagIndex(mne, Undirected):
    name = "Debiased squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        self._measure = "pli2_unbiased"
        super().__init__(**kwargs)


class weighted_PhaseLagIndex(mne, Undirected):
    name = "Weighted squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wspli"
        self._measure = "wpli"
        super().__init__(**kwargs)


class debiased_weighted_squared_PhaseLagIndex(mne, Undirected):
    name = "Debiased weighted squared phase lag index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dwspli"
        self._measure = "wpli2_debiased"
        super().__init__(**kwargs)


class PhaseSlopeIndex(mne, Undirected):
    name = "Phase slope index (wavelet)"
    labels = ["unsigned", "wavelet", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self.identifier += f"_{self._statistic}"

    def _get_cache(self, data):
        try:
            psi = data.mne_psi["psi"]
            freq = data.mne_psi["freq"]
        except AttributeError:
            z = np.moveaxis(data.to_numpy(), 2, 0)

            freqs = np.linspace(0.2, 0.5, 10)
            psi, freq, _, _, _ = mnec.phase_slope_index(
                data=z,
                mode="cwt_morlet",
                sfreq=self._fs,
                mt_adaptive=True,
                cwt_freqs=freqs,
                verbose="WARNING",
            )
            freq = freq[0]
            data.mne_psi = dict(psi=psi, freq=freq)

        # freq = conn.frequencies
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        return psi, freq_id

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq_id = self._get_cache(data)
        adj = self._statfn(np.real(adj_freq[..., freq_id]), axis=(2, 3))

        ui = np.triu_indices(data.n_processes, 1)
        adj[ui] = adj.T[ui]
        np.fill_diagonal(adj, np.nan)
        return adj
