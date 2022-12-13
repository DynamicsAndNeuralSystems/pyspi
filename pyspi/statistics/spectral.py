import numpy as np
from copy import deepcopy

import spectral_connectivity as sc  # For directed spectral statistics (excl. spectral GC)
from pyspi.base import (
    Directed,
    Undirected,
    Unsigned,
    parse_bivariate,
    parse_multivariate,
)
import nitime.analysis as nta
import nitime.timeseries as ts
import warnings


class NonparametricSpectral(Unsigned):
    """Base class for the nonparametric spectral methods from the Eden-Kramer repo"""

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
        elif statistic not in ["delay", "slope", "rvalue"]:
            raise NameError(f"Unknown statistic: {statistic}")
        else:
            self._statfn = None
        self._statistic = statistic
        paramstr = (
            f"_multitaper_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}".replace(
                ".", "-"
            )
        )
        self.identifier += paramstr

    @property
    def key(self):
        if isinstance(self, GroupDelay) or isinstance(self, PhaseSlopeIndex):
            return (self.measure, self._fmin, self._fmax)
        else:
            return (self.measure,)

    @property
    def measure(self):
        try:
            return self._measure
        except AttributeError:
            raise AttributeError(f"Include measure for {self.identifier}")

    def _get_statistic(self, C):
        raise NotImplementedError


class NonparametricSpectralMultivariate(NonparametricSpectral):
    def _get_cache(self, data):
        try:
            res = data.spectral_mv[self.key]
            freq = data.spectral_mv["freq"]
        except (AttributeError, KeyError):
            z = np.transpose(data.to_numpy(squeeze=True))
            m = sc.Multitaper(z, sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn, self.measure)()
            except TypeError:
                res = self._get_statistic(conn)

            freq = conn.frequencies
            try:
                data.spectral_mv[self.key] = res
            except AttributeError:
                data.spectral_mv = {"freq": freq, self.measure: res}

        return res, freq

    @parse_multivariate
    def multivariate(self, data):
        adj_freq, freq = self._get_cache(data)
        freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

        try:
            adj = self._statfn(adj_freq[0, freq_id, :, :], axis=0)
        except IndexError:  # For phase slope index
            adj = adj_freq[0]
        except TypeError:  # For group delay
            stat_id = [
                i
                for i, s in enumerate(["delay", "slope", "rvalue"])
                if self._statistic == s
            ][0]
            adj = adj_freq[stat_id][0]
        np.fill_diagonal(adj, np.nan)
        return adj


class NonparametricSpectralBivariate(NonparametricSpectral):
    def _get_cache(self, data, i, j):
        key = (self.measure, i, j)
        try:
            res = data.spectral_bv[key]
            freq = data.spectral_bv["freq"]
        except (KeyError, AttributeError):
            z = np.transpose(data.to_numpy(squeeze=True)[[i, j]])
            m = sc.Multitaper(z, sampling_frequency=self._fs)
            conn = sc.Connectivity.from_multitaper(m)
            try:
                res = getattr(conn, self.measure)()
            except TypeError:
                res = self._get_statistic(conn)

            freq = conn.frequencies
            try:
                data.spectral_bv[key] = res
            except AttributeError:
                data.spectral_bv = {"freq": freq, key: res}

        return res, freq

    @parse_bivariate
    def bivariate(self, data, i=None, j=None):
        """TODO: cache this result"""
        bv_freq, freq = self._get_cache(data, i, j)
        freq_id = np.where((freq > self._fmin) * (freq < self._fmax))[0]

        return self._statfn(bv_freq[0, freq_id, 0, 1])


class CoherenceMagnitude(NonparametricSpectralMultivariate, Undirected):
    name = "Coherence magnitude"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "cohmag"
        super().__init__(**kwargs)
        self._measure = "coherence_magnitude"


class CoherencePhase(NonparametricSpectralMultivariate, Undirected):
    name = "Coherence phase"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "phase"
        super().__init__(**kwargs)
        self._measure = "coherence_phase"


class ImaginaryCoherence(NonparametricSpectralMultivariate, Undirected):
    name = "Imaginary coherence"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "icoh"
        super().__init__(**kwargs)
        self._measure = "imaginary_coherence"


class PhaseLockingValue(NonparametricSpectralMultivariate, Undirected):
    name = "Phase locking value"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "plv"
        super().__init__(**kwargs)
        self._measure = "phase_locking_value"

        myfn = deepcopy(self._statfn)
        self._statfn = lambda x, **kwargs: myfn(np.absolute(x), **kwargs)


class PhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "pli"
        super().__init__(**kwargs)
        self._measure = "phase_lag_index"


class WeightedPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Weighted phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "wpli"
        super().__init__(**kwargs)
        self._measure = "weighted_phase_lag_index"


class DebiasedSquaredPhaseLagIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Debiased squared phase lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dspli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_phase_lag_index"


class DebiasedSquaredWeightedPhaseLagIndex(
    NonparametricSpectralMultivariate, Undirected
):
    name = "Debiased squared weighted phase-lag index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "dswpli"
        super().__init__(**kwargs)
        self._measure = "debiased_squared_weighted_phase_lag_index"


class PairwisePhaseConsistency(NonparametricSpectralMultivariate, Undirected):
    name = "Pairwise phase consistency"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "ppc"
        super().__init__(**kwargs)
        self._measure = "pairwise_phase_consistency"


"""
    These next several seem to segfault for large vector autoregressive processes (something to do with np.linalg solver).
    Switched them to bivariate for now until the issue is resolved
"""


class DirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "dcoh"
        super().__init__(**kwargs)
        self._measure = "directed_coherence"


class PartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "pdcoh"
        super().__init__(**kwargs)
        self._measure = "partial_directed_coherence"


class GeneralizedPartialDirectedCoherence(NonparametricSpectralBivariate, Directed):
    name = "Generalized partial directed coherence"
    labels = ["unsigned", "spectral", "directed"]

    def __init__(self, **kwargs):
        self.identifier = "gpdcoh"
        super().__init__(**kwargs)
        self._measure = "generalized_partial_directed_coherence"


class DirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "dtf"
        super().__init__(**kwargs)
        self._measure = "directed_transfer_function"


class DirectDirectedTransferFunction(NonparametricSpectralBivariate, Directed):
    name = "Direct directed transfer function"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "ddtf"
        super().__init__(**kwargs)
        self._measure = "direct_directed_transfer_function"


class PhaseSlopeIndex(NonparametricSpectralMultivariate, Undirected):
    name = "Phase slope index"
    labels = ["unsigned", "spectral", "undirected"]

    def __init__(self, **kwargs):
        self.identifier = "psi"
        super().__init__(**kwargs)
        self._measure = "phase_slope_index"

    def _get_statistic(self, C):
        return C.phase_slope_index(
            frequencies_of_interest=[self._fmin, self._fmax],
            frequency_resolution=(self._fmax - self._fmin) / 50,
        )


class GroupDelay(NonparametricSpectralMultivariate, Directed):
    name = "Group delay"
    labels = ["unsigned", "spectral", "directed", "lagged"]

    def __init__(self, **kwargs):
        self.identifier = "gd"
        super().__init__(**kwargs)
        self._measure = "group_delay"

    def _get_statistic(self, C):
        return C.group_delay(
            frequencies_of_interest=[self._fmin, self._fmax],
            frequency_resolution=(self._fmax - self._fmin) / 50,
        )


class SpectralGrangerCausality(NonparametricSpectralMultivariate, Directed, Unsigned):
    name = "Spectral Granger causality"
    identifier = "sgc"
    labels = ["unsigned", "embedding", "spectral", "directed", "lagged"]

    def __init__(
        self,
        fs=1,
        fmin=1e-5,
        fmax=0.5,
        method="nonparametric",
        order=None,
        max_order=50,
        statistic="mean",
        ignore_nan=True,
        nan_threshold=0.5,
    ):
        self._fs = fs  # Not yet implemented
        self._fmin = fmin
        self._fmax = fmax
        self.ignore_nan = ignore_nan
        self.nan_threshold = nan_threshold

        if self._fmin <= 0.0:
            warnings.warn(f"Frequency minimum set to {self._fmin}; overriding to 1e-5.")
            self._fmin = 1e-5

        if statistic == "mean":
            if self.ignore_nan:
                self._statfn = np.nanmean
            else:
                self._statfn = np.mean
        elif statistic == "max":
            if self.ignore_nan:
                self._statfn = np.nanmax
            else:
                self._statfn = np.max
        else:
            raise NameError(f"Unknown statistic {statistic}")

        self._method = method
        if self._method == "nonparametric":
            self._measure = "pairwise_spectral_granger_prediction"
            paramstr = f"_nonparametric_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}".replace(
                ".", "-"
            )
        else:
            self._order = order
            self._max_order = max_order
            paramstr = f"_parametric_{statistic}_fs-{fs}_fmin-{fmin:.3g}_fmax-{fmax:.3g}_order-{order}".replace(
                ".", "-"
            )

        self.identifier = self.identifier + paramstr

    def _getkey(self):
        if self._method == "nonparametric":
            return (self._method, -1, -1)
        else:
            return (self._method, self._order, self._max_order)

    def _get_cache(self, data):
        key = self._getkey()

        try:
            F = data.spectral_gc[key]["F"]
            freq = data.spectral_gc[key]["freq"]
        except (AttributeError, KeyError):

            if self._method == "nonparametric":
                F, freq = super()._get_cache(data)
            else:
                z = data.to_numpy(squeeze=True)
                time_series = ts.TimeSeries(z, sampling_interval=1)
                GA = nta.GrangerAnalyzer(
                    time_series, order=self._order, max_order=self._max_order
                )

                triu_id = np.triu_indices(data.n_processes)

                F = np.full(GA.causality_xy.shape, np.nan)
                F[triu_id[0], triu_id[1], :] = GA.causality_xy[
                    triu_id[0], triu_id[1], :
                ]
                F[triu_id[1], triu_id[0], :] = GA.causality_yx[
                    triu_id[0], triu_id[1], :
                ]

                F = np.transpose(np.expand_dims(F, axis=3), axes=[3, 2, 1, 0])
                freq = GA.frequencies
            try:
                data.spectral_gc[key] = {"freq": freq, "F": F}
            except AttributeError:
                data.spectral_gc = {key: {"freq": freq, "F": F}}

        return F, freq

    @parse_multivariate
    def multivariate(self, data):
        try:
            cache, freq = self._get_cache(data)
            freq_id = np.where((freq >= self._fmin) * (freq <= self._fmax))[0]

            result = self._statfn(cache[0, freq_id, :, :], axis=0)

            nan_pct = np.isnan(cache[0, freq_id, :, :]).mean(axis=0)
            np.fill_diagonal(nan_pct, 0.0)

            isna = nan_pct > self.nan_threshold
            if isna.any():
                warnings.warn(
                    f"Spectral GC: the following processes have >{self.nan_threshold*100:.1f}% "
                    f"NaN values:\n{np.transpose(np.where(isna))}\nThese indices will be set to NaN. "
                    "Set ignore_nan to False or modify nan_threshold parameter if required."
                )
                result[isna] = np.nan

            return result
        except ValueError as err:
            warnings.warn(err)
            return np.full((data.n_processes, data.n_processes), np.nan)
